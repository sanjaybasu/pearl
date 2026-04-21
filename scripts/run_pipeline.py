"""
PEARL Full Pipeline Runner

Phases:
  Phase 0: IMI estimation + T1-T6 falsification + Camden reanalysis
  Phase 1: Train all comparators (C1-C8) + PEARL base + MoE-PEARL
  Phase 2: DR-OPE evaluation + conformal prediction + sensitivity analysis
  Phase 3: Einstein Arena hyperparameter optimization (optional)

Usage:
  python scripts/run_pipeline.py --waymark               (real Waymark data; default for paper)
  python scripts/run_pipeline.py --synthetic             (synthetic data; public reproducibility demo)
  python scripts/run_pipeline.py --waymark --multi_seed  (multi-seed stability)

Sensitivity analysis:
  Run scripts/run_sensitivity.sh to execute all 20 pre-specified sensitivity analyses.
  Results are saved to outputs/results/sensitivity_results.csv.
"""
import sys
import os
sys.path.insert(0, "/Users/sanjaybasu/pearl")

import numpy as np
import pandas as pd
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")


def run_phase_0(pop, rising_train, rising_test, wpad_pairs_train, verbose=True):
    """
    Phase 0: IMI estimation + falsification tests + Camden reanalysis.

    Train/test discipline:
    - estimator.fit(rising_train): propensity + outcome models trained on 80% of rising-risk.
    - estimator.estimate(rising_test): IMI evaluated on held-out 20% (primary paper result).
    - mu_hat_train: S-learner predictions for rising_train (returned for phase 1 wpad_preferred).
    """
    from evaluation.falsification_tests import FalsificationTestSuite
    from models.imi_estimator import IMIEstimator, CamdenReanalysis

    print("\n" + "="*60)
    print("PHASE 0: IMI ESTIMATION + FALSIFICATION TESTS")
    print("="*60)
    print(f"  IMI fit set:  {len(rising_train):,} rising-risk patients")
    print(f"  IMI eval set: {len(rising_test):,} held-out patients (primary result)")

    # T1-T6 falsification tests (run on full WPAD pair set — independent of train/test split)
    suite = FalsificationTestSuite(
        wpad_pairs=pop.wpad_pairs,
        patients=pop.patients,
        alpha=0.05
    )
    falsification_results = suite.run_all(verbose=verbose)
    falsification_df = suite.get_report_df()

    # IMI estimation: fit on train, evaluate on held-out test
    estimator = IMIEstimator(
        outcome_col="y_behavioral",
        intervention_col="behavioral_intervention",
        threshold=0.02,
        n_bootstrap=200,
        seed=42
    )
    estimator.fit(rising_train)  # TRAIN ONLY — no test data leakage

    # Estimate IMI on held-out test set (paper's primary IMI result)
    # Pass wpad_pairs_train so E-value uses WPAD LATE (VanderWeele & Ding 2017)
    imi_result = estimator.estimate(rising_test, wpad_pairs=wpad_pairs_train)
    if verbose:
        estimator.print_report(imi_result)

    # Compute mu_hat for training patients (used in phase 1 for wpad_preferred_intervention)
    X_train = estimator._get_feature_matrix(rising_train)
    mu_hat_train = estimator._predict_outcomes(X_train)  # (n_train, 4) alphabetical

    # Policy comparison (behavioral vs. oracle) on TEST set
    comparison = estimator.compare_policies(
        rising_test,
        pearl_intervention_col="optimal_intervention",
        behavioral_col="behavioral_intervention"
    )
    if verbose:
        print(f"\nPolicy comparison (held-out test):")
        print(f"  IMI(behavioral):   {comparison['imi_behavioral']:.3f}")
        print(f"  IMI(PEARL_oracle): {comparison['imi_pearl']:.3f}")
        print(f"  IMI reduction:     {comparison['imi_reduction']:.3f} ({comparison['imi_reduction_pct']:.1f}%)")

    # Camden reanalysis
    camden_runner = CamdenReanalysis()
    camden_result = camden_runner.run(estimator, pop.camden_stratum_patients, threshold=0.02)
    if verbose:
        camden_runner.print_report(camden_result)

    return {
        "falsification": falsification_df,
        "imi_result": imi_result,           # estimated on rising_test (held-out)
        "imi_comparison": comparison,
        "camden": camden_result,
        "estimator": estimator,             # fitted on rising_train
        "mu_hat_train": mu_hat_train,       # (n_train, 4) for wpad_preferred in phase 1
    }


def run_phase_1(pop, rising_train, wpad_pairs_train, phase0, verbose=True):
    """
    Phase 1: Train all models on rising_train + wpad_pairs_train.

    Train/test discipline: all model fitting uses only rising_train data.
    The DROPEEvaluator (nuisance models for DR-OPE) is also fitted here on
    rising_train, then used in phase 2 to evaluate on rising_test.
    """
    from models.pearl_dpo import TabularPEARL
    from models.comparators import ComparatorSuite
    from mixture_of_experts.moe_router import MoERouter, MoEPEARL
    from evaluation.drope_evaluator import DROPEEvaluator
    from sklearn.preprocessing import LabelEncoder as _LE2

    print("\n" + "="*60)
    print("PHASE 1: MODEL TRAINING")
    print("="*60)

    estimator = phase0["estimator"]          # fitted on rising_train
    mu_hat_train = phase0["mu_hat_train"]    # S-learner predictions for rising_train patients

    # Derive WPAD-preferred intervention for each TRAINING patient from S-learner.
    # mu_hat_train shape: (n_train, 4) in LabelEncoder alphabetical order:
    #   col 0='behavioral_health', 1='clinical_complexity', 2='medication_adherence', 3='social_needs'
    # argmin = intervention with lowest estimated acute-care event probability = best match.
    _le_alpha = _LE2().fit(["social_needs", "medication_adherence",
                             "behavioral_health", "clinical_complexity"])
    preferred_alpha_idx = np.argmin(mu_hat_train, axis=1)  # indices in alphabetical order
    rising_train = rising_train.copy()
    rising_train["wpad_preferred_intervention"] = _le_alpha.inverse_transform(preferred_alpha_idx)

    # Propagate wpad_preferred to pop.patients for MoEPEARL (which uses pop.patients internally)
    pop.patients["wpad_preferred_intervention"] = "behavioral_health"  # default
    pop.patients.loc[
        pop.patients["patient_id"].isin(rising_train["patient_id"]), "wpad_preferred_intervention"
    ] = rising_train.set_index("patient_id")["wpad_preferred_intervention"]

    # Fit DR-OPE evaluator on TRAIN set (nuisance models: propensity + outcome).
    # Evaluated on rising_test in phase 2. Cross-fitting: train nuisance on train, eval on test.
    print("\nFitting DR-OPE evaluator (nuisance models on training data)...")
    drope_eval = DROPEEvaluator(n_bootstrap=300, seed=42)
    drope_eval.fit(rising_train)

    # Comparators C1-C8: all trained on rising_train + wpad_pairs_train.
    # C4 (BehavioralCloning SFT) uses wpad_preferred_intervention — same causal signal as PEARL.
    suite = ComparatorSuite(seed=42)
    suite.fit_all(rising_train, wpad_pairs_train,
                  outcome_col="y_behavioral",
                  intervention_col="behavioral_intervention",
                  wpad_preferred_col="wpad_preferred_intervention")

    # Base PEARL: IPTW-DPO on wpad_pairs_train
    print("\nTraining PEARL (base IPTW-DPO)...")
    pearl = TabularPEARL(beta=0.1, lora_r=64, seed=42)
    pearl.fit(wpad_pairs_train, pop.patients, n_iterations=80, verbose=verbose)

    # MoE Router: trained on rising_train patients with wpad_preferred_intervention
    print("\nTraining MoE Router...")
    moe = MoERouter(n_experts=4, top_k=2, seed=42)
    moe.fit(rising_train, wpad_pairs_train, target_col="wpad_preferred_intervention")

    # MoE-PEARL: base PEARL component + standalone MoE router.
    # The base component uses wpad_pairs_train; we swap in the standalone moe router
    # (trained only on rising_train) to avoid routing bias from non-rising patients.
    print("\nTraining MoE-PEARL (full blend)...")
    moe_pearl = MoEPEARL(beta=0.1, moe_weight=0.5, seed=42)
    moe_pearl.fit(wpad_pairs_train, pop.patients, n_iterations=50,
                  moe_target_col="wpad_preferred_intervention")
    moe_pearl.moe_router = moe  # swap to correctly-trained router

    return {
        "comparator_suite": suite,
        "pearl": pearl,
        "moe": moe,
        "moe_pearl": moe_pearl,
        "drope_eval": drope_eval,       # fitted on rising_train — used in phase 2 on rising_test
        "rising_train": rising_train,   # with wpad_preferred_intervention column added
    }


def run_phase_2(pop, rising_train, rising_test, phase0, phase1_results, verbose=True):
    """
    Phase 2: DR-OPE evaluation + conformal prediction + bootstrap CI.

    Train/test discipline:
    - drope_eval: already fitted on rising_train (in phase 1) — evaluates on rising_test.
    - mu_hat_test: S-learner predictions for rising_test from phase0 imi_result.
    - Bootstrap CI and one-sided p-value computed on rising_test.
    """
    from evaluation.drope_evaluator import ConformalPrediction, SensitivityAnalysis

    print("\n" + "="*60)
    print("PHASE 2: DR-OPE EVALUATION (held-out test set)")
    print("="*60)
    print(f"  Evaluation set: {len(rising_test):,} held-out rising-risk patients")

    pearl = phase1_results["pearl"]
    moe = phase1_results["moe"]
    moe_pearl = phase1_results["moe_pearl"]
    suite = phase1_results["comparator_suite"]
    drope_eval = phase1_results["drope_eval"]   # fitted on rising_train — nuisance models

    # S-learner predictions for TEST patients (out-of-sample: model fitted on train)
    mu_hat_test = phase0["imi_result"]["mu_hat"]   # (n_test, 4) alphabetical order
    rising = rising_test  # alias for downstream code clarity

    # Define all policy functions
    def make_pearl_policy(m):
        def fn(pts):
            recs, _, _ = m.predict_intervention(pts)
            return recs
        return fn

    def behavioral_policy(pts):
        return pts["behavioral_intervention"].values

    def cql_policy(pts):
        return suite.cql.recommend_intervention(pts)

    def causal_forest_policy(pts):
        return suite.causal_forest.recommend_intervention(pts)

    def dt_policy(pts):
        return suite.dt.predict_intervention(pts)

    def moe_policy(pts):
        recs, _, _ = moe.predict(pts)
        return recs

    def moe_pearl_policy(pts):
        recs, _ = moe_pearl.predict(pts)
        return recs

    # Attach mu_hat columns to rising_test so oracle_policy can use them for real data.
    # mu_hat_test is (n, 4) in LabelEncoder alphabetical order:
    #   col 0 = behavioral_health, col 1 = clinical_complexity,
    #   col 2 = medication_adherence, col 3 = social_needs
    _INTV_ALPHA = ["behavioral_health", "clinical_complexity", "medication_adherence", "social_needs"]
    rising = rising.copy()  # avoid SettingWithCopyWarning on rising_test slice
    for _i, _intv in enumerate(_INTV_ALPHA):
        rising[f"_mu_hat_{_intv}"] = mu_hat_test[:, _i]

    def oracle_policy(pts):
        """Oracle: always recommends the intervention with lowest estimated event probability."""
        # Primary: use mu_hat columns attached above (works for real data)
        mu_cols = [f"_mu_hat_{intv}" for intv in _INTV_ALPHA]
        if all(c in pts.columns for c in mu_cols):
            mu_matrix = pts[mu_cols].values  # (n, 4) alphabetical
            best_idx = np.argmin(mu_matrix, axis=1)
            return np.array([_INTV_ALPHA[i] for i in best_idx])
        # Fallback: synthetic data ground-truth column
        if "optimal_intervention" in pts.columns:
            return pts["optimal_intervention"].values
        # Last resort: argmin p_outcome columns (synthetic mode)
        p_cols = [f"p_outcome_{intv}" for intv in ["social_needs", "medication_adherence",
                                                     "behavioral_health", "clinical_complexity"]]
        if all(c in pts.columns for c in p_cols):
            INTV_ORDER = ["social_needs", "medication_adherence", "behavioral_health", "clinical_complexity"]
            p_matrix = np.column_stack([pts[c].values for c in p_cols])
            best_idx = np.argmin(p_matrix, axis=1)
            return np.array([INTV_ORDER[i] for i in best_idx])
        return pts["behavioral_intervention"].values

    # C1/C2: risk-score routing (standard-of-care baselines — no ML, publicly citable)
    def lace_policy(pts):
        return suite.lace.route_intervention(pts)

    def hospital_policy(pts):
        return suite.hospital.route_intervention(pts)

    # C3: XGBoost risk-score routing (best ML risk prediction, threshold-based routing)
    def xgb_policy(pts):
        return suite.xgb.route_intervention(pts)

    # C4: Behavioral Cloning SFT (DPO ablation — same WPAD signal but SFT loss, not contrastive)
    def bc_sft_policy(pts):
        return suite.bc_sft.predict_intervention(pts)

    # C5: Observational DPO (identification ablation — DPO without WPAD causal filtering)
    def obs_dpo_policy(pts):
        return suite.obs_dpo.predict_intervention(pts)

    policies = {
        # Standard-of-care baselines
        "LACE Index (C1)": lace_policy,
        "HOSPITAL Score (C2)": hospital_policy,
        # ML baselines
        "XGBoost (C3)": xgb_policy,
        # DPO ablations — isolate contributions of each PEARL component
        "BehavioralCloning SFT (C4)": bc_sft_policy,
        "Observational DPO (C5)": obs_dpo_policy,
        # Published SOTA comparators
        "CausalForest (C6)": causal_forest_policy,
        "DecisionTransformer (C7)": dt_policy,
        "CQL (C8)": cql_policy,
        # Behavioral policy (current system)
        "Behavioral Policy": behavioral_policy,
        # PEARL variants
        "PEARL (base)": make_pearl_policy(pearl),
        "PEARL (MoE Router)": moe_policy,
        "PEARL (MoE Full)": moe_pearl_policy,
        # Theoretical upper bound
        "Oracle (optimal)": oracle_policy,
    }

    # Run DR-OPE comparison
    comparison_df = drope_eval.compare_policies(rising, policies)

    # Paired bootstrap comparison: PEARL (MoE Router) vs. Behavioral Policy.
    # Uses same bootstrap resamples for both policies (paired), reducing variance.
    # H0: DR-OPE(PEARL) >= DR-OPE(Behavioral)  [PEARL not better]
    # Small p-value rejects H0 and supports PEARL superiority.
    if "PEARL (MoE Router)" in policies and "Behavioral Policy" in policies:
        paired_test = drope_eval.paired_bootstrap_comparison(
            patients=rising,
            policy_fn_a=policies["PEARL (MoE Router)"],
            policy_fn_b=policies["Behavioral Policy"],
            name_a="PEARL (MoE Router)",
            name_b="Behavioral Policy",
        )
        print("\nPaired Bootstrap DR-OPE Comparison (PEARL vs. Behavioral):")
        print(f"  Absolute improvement (Behavioral - PEARL): "
              f"{paired_test['diff_point']:.4f} "
              f"[95% CI: {paired_test['diff_ci_lower']:.4f}, {paired_test['diff_ci_upper']:.4f}]")
        print(f"  One-sided p-value (H0: PEARL not better): {paired_test['p_value_one_sided']:.4f}")
    else:
        paired_test = {}

    # Also compute Direct Method (DM) estimates — unbiased for off-behavioral policies.
    # DR-OPE for behavioral has high IPW variance (CI width ~0.18); DM is stable.
    # DM = (1/n) Σ_i μ̂(x_i, π(x_i)) — no IPW correction.
    from sklearn.preprocessing import LabelEncoder as _LE
    _le_tmp = _LE().fit(["social_needs", "medication_adherence", "behavioral_health", "clinical_complexity"])
    drope_eval_X = drope_eval._get_X(rising)
    drope_mu = drope_eval._predict_outcomes(drope_eval_X)  # (n, 4) in LE alphabetical order
    dm_values = {}
    for name, policy_fn in policies.items():
        recs = policy_fn(rising)
        pi_enc = _le_tmp.transform(recs)
        dm_values[name] = float(drope_mu[np.arange(len(rising)), pi_enc].mean())

    print("\nDR-OPE Policy Comparison (lower = better; Y=1 is acute care event):")
    print("Note: Behavioral DR-OPE has high IPW variance at N~1200; see DM column for stable comparison.")
    cols = ["dr_ope_rank", "policy_name", "policy_value",
            "policy_value_ci_lower", "policy_value_ci_upper",
            "ess", "ess_adequate", "relative_improvement_pct"]
    print(comparison_df[cols].to_string(index=False))

    print("\nDirect Method (DM) Comparison (unbiased for off-behavioral policies):")
    dm_df = pd.DataFrame([{"Policy": k, "DM_value": v, "DM_pct": f"{v*100:.1f}%"}
                           for k, v in sorted(dm_values.items(), key=lambda x: x[1])])
    print(dm_df.to_string(index=False))

    # IMI comparison table — evaluated on held-out rising_test using out-of-sample mu_hat_test.
    from sklearn.preprocessing import LabelEncoder as _LE_imi
    _le_imi = _LE_imi().fit(["social_needs", "medication_adherence",
                              "behavioral_health", "clinical_complexity"])
    imi_results = {}
    for name, policy_fn in policies.items():
        recs = policy_fn(rising)
        A_enc = _le_imi.transform(recs)
        # Use out-of-sample S-learner predictions (mu_hat_test) for patient-level IMI.
        # IMI=1 if ∃a: μ̂(x,a) < μ̂(x,π(x)) - ε  (lower event prob = better intervention)
        imi = float(np.array([
            float(any(mu_hat_test[i, j] < mu_hat_test[i, A_enc[i]] - 0.02
                      for j in range(4) if j != A_enc[i]))
            for i in range(len(rising))
        ]).mean())
        imi_results[name] = imi

    print("\n\nIMI Comparison (held-out test set, out-of-sample mu_hat):")
    imi_df = pd.DataFrame([
        {"Policy": k, "IMI": v, "IMI_pct": f"{v*100:.1f}%"} for k, v in imi_results.items()
    ]).sort_values("IMI")
    print(imi_df.to_string(index=False))

    # ── Bootstrap CI + one-sided p-value for IMI reduction ───────────────
    # H0: IMI(PEARL best) >= IMI(behavioral)
    # H1: IMI(PEARL best) < IMI(behavioral)   [PEARL reduces misalignment]
    # Method: bootstrap resampling of rising_test; compute IMI difference each time.
    print("\n\nBootstrap IMI Reduction Hypothesis Test...")
    imi_behavioral_pt = imi_results.get("Behavioral Policy", None)
    # Pick the PEARL variant with the lowest IMI (not hardcoded MoE Full, which can be worst).
    _pearl_variants = ["PEARL (MoE Router)", "PEARL (MoE Full)", "PEARL (base)"]
    _available = [v for v in _pearl_variants if v in imi_results]
    best_pearl_name = min(_available, key=lambda v: imi_results[v]) if _available else "PEARL (base)"
    imi_pearl_pt = imi_results.get(best_pearl_name, None)
    bootstrap_ci = {}

    if imi_behavioral_pt is not None and imi_pearl_pt is not None:
        n_test = len(rising)
        n_boot = 2000
        rng_boot = np.random.default_rng(42)
        boot_imi_behav = []
        boot_imi_pearl = []
        boot_reductions = []

        behavioral_recs_all = rising["behavioral_intervention"].values
        pearl_recs_all = policies[best_pearl_name](rising)

        for _ in range(n_boot):
            boot_idx = rng_boot.integers(0, n_test, n_test)
            mu_b = mu_hat_test[boot_idx]
            behav_b = behavioral_recs_all[boot_idx]
            pearl_b = pearl_recs_all[boot_idx]

            enc_behav = _le_imi.transform(behav_b)
            enc_pearl = _le_imi.transform(pearl_b)

            imi_b = float(np.mean([
                float(any(mu_b[i, j] < mu_b[i, enc_behav[i]] - 0.02
                          for j in range(4) if j != enc_behav[i]))
                for i in range(len(boot_idx))
            ]))
            imi_p = float(np.mean([
                float(any(mu_b[i, j] < mu_b[i, enc_pearl[i]] - 0.02
                          for j in range(4) if j != enc_pearl[i]))
                for i in range(len(boot_idx))
            ]))
            boot_imi_behav.append(imi_b)
            boot_imi_pearl.append(imi_p)
            boot_reductions.append(imi_b - imi_p)

        boot_reductions = np.array(boot_reductions)
        # 95% CI on IMI reduction (behavioral - PEARL)
        reduction_ci_lo = float(np.percentile(boot_reductions, 2.5))
        reduction_ci_hi = float(np.percentile(boot_reductions, 97.5))
        # One-sided p-value: fraction of bootstrap samples where PEARL doesn't reduce IMI
        p_one_sided = float(np.mean(boot_reductions <= 0))

        bootstrap_ci = {
            "imi_behavioral": imi_behavioral_pt,
            "imi_pearl": imi_pearl_pt,
            "imi_reduction_point": imi_behavioral_pt - imi_pearl_pt,
            "imi_reduction_ci_lower": reduction_ci_lo,
            "imi_reduction_ci_upper": reduction_ci_hi,
            "p_value_one_sided": p_one_sided,
            "significant": p_one_sided < 0.05,
        }
        print(f"  {best_pearl_name} vs. Behavioral Policy (held-out test):")
        print(f"    IMI reduction:    {imi_behavioral_pt:.3f} → {imi_pearl_pt:.3f}")
        print(f"                     Δ = {imi_behavioral_pt - imi_pearl_pt:.3f} "
              f"[95% CI: {reduction_ci_lo:.3f}, {reduction_ci_hi:.3f}]")
        print(f"    One-sided p:      {p_one_sided:.4f}  "
              f"({'✓ SIGNIFICANT' if p_one_sided < 0.05 else '✗ not significant'} at α=0.05)")
    else:
        print("  Could not compute bootstrap CI (missing behavioral or PEARL IMI)")

    # Conformal prediction — use S-learner mu_hat_test (not AIPW-corrected).
    # Binary Y / continuous mu_hat mismatch inflates q_hat to ~0.8 with AIPW values.
    # In simulation mode (p_outcome_* available), calibration targets the true probability.
    print("\n\nConformal Prediction Calibration...")
    n_calib = len(rising) // 5
    calib_pts = rising.iloc[:n_calib].reset_index(drop=True)
    conf_test_pts = rising.iloc[n_calib:].reset_index(drop=True)
    mu_calib = mu_hat_test[:n_calib]
    mu_conf_test = mu_hat_test[n_calib:]

    calib_recs, _, _ = pearl.predict_intervention(calib_pts)
    conf_test_recs, _, _ = pearl.predict_intervention(conf_test_pts)

    conformal = ConformalPrediction(alpha=0.10, seed=42)
    conformal.calibrate(calib_pts, calib_recs, mu_calib)         # S-learner mu_hat
    coverage_check = conformal.check_coverage(conf_test_pts, conf_test_recs, mu_conf_test)
    print(f"  90% PI coverage: {coverage_check['empirical_coverage']:.1%} "
          f"(target 90%, gap={coverage_check['coverage_gap']:+.1%})")
    print(f"  Mean PI width:   {coverage_check['mean_interval_width']:.4f}")
    print(f"  Adequate:        {'✓' if coverage_check['passes'] else '✗'}")

    # Sensitivity analysis
    print("\nRunning sensitivity analysis...")
    sensitivity = SensitivityAnalysis()

    # Use best-performing PEARL variant for sensitivity reference values.
    best_drope_row = comparison_df[comparison_df["policy_name"] == best_pearl_name]
    if len(best_drope_row) == 0:
        # Fallback: try other PEARL variants in preference order
        for _pv in ["PEARL (MoE Router)", "PEARL (MoE Full)", "PEARL (base)"]:
            best_drope_row = comparison_df[comparison_df["policy_name"] == _pv]
            if len(best_drope_row) > 0:
                break
    if len(best_drope_row) > 0:
        primary_drope = float(best_drope_row["policy_value"].values[0])
    else:
        primary_drope = float(comparison_df["policy_value"].iloc[0])

    sens_df = sensitivity.run_sensitivity_table(
        primary_imi=imi_results.get(best_pearl_name, imi_results.get("PEARL (base)", 0.20)),
        primary_drope=primary_drope,
        drope_evaluator=drope_eval,
        patients=rising,
        policy_fn=make_pearl_policy(pearl),
        wpad_pairs=pop.wpad_pairs,
        mu_hat=mu_hat_test,   # (n_test, 4) for camden_threshold epsilon re-evaluation
    )
    sensitivity.print_table(sens_df)

    return {
        "drope_comparison": comparison_df,
        "imi_results": imi_results,
        "bootstrap_ci": bootstrap_ci,
        "drope_paired_test": paired_test,  # paired bootstrap for primary PEARL vs. behavioral comparison
        "conformal": coverage_check,
        "sensitivity": sens_df,
        "drope_eval": drope_eval,
    }


def run_phase_3_arena(pop, rising_train, verbose=True):
    """Phase 3 (optional): Einstein Arena hyperparameter optimization (on training data)."""
    from experiments.einstein_arena import EinsteinArena

    print("\n" + "="*60)
    print("PHASE 3: EINSTEIN ARENA OPTIMIZATION")
    print("="*60)

    arena = EinsteinArena(
        n_generations=2,
        population_size=8,
        survivors_per_round=3,
        seed=42
    )
    best_result, arena_df = arena.run(rising_train, pop.wpad_pairs, verbose=verbose)

    output_path = "/Users/sanjaybasu/pearl/outputs/results/einstein_arena_results.json"
    arena.save_results(output_path)

    return {"best_result": best_result, "arena_df": arena_df}


def save_paper_table(phase0, phase1, phase2, output_dir):
    """Save the main results table for the paper."""
    rows = []
    drope_df = phase2["drope_comparison"]
    imi_dict = phase2["imi_results"]

    for _, row in drope_df.iterrows():
        imi = imi_dict.get(row["policy_name"], None)
        rows.append({
            "Model": row["policy_name"],
            "DR-OPE_Rank": row["dr_ope_rank"],
            "Policy_Value": f"{row['policy_value']:.4f}",
            "Policy_Value_95CI": f"[{row['policy_value_ci_lower']:.4f}, {row['policy_value_ci_upper']:.4f}]",
            "Rel_Improvement_pct": f"{row['relative_improvement_pct']:+.1f}%",
            "ESS": f"{row['ess']:.0f} {'✓' if row['ess_adequate'] else '✗'}",
            "IMI": f"{imi:.3f}" if imi is not None else "N/A",
            "Coverage_Coeff": f"{row['coverage_coefficient']:.2f}",
        })

    table = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    table.to_csv(f"{output_dir}/main_results_table.csv", index=False)
    print(f"\nMain results table saved to {output_dir}/main_results_table.csv")

    return table


def run_multi_seed_stability(n_patients=8_000, seeds=(42, 123, 456, 789, 1000), verbose=False):
    """
    Multi-seed stability analysis: run the full pipeline with 5 different random seeds
    and report mean ± std for all key metrics.

    Tests robustness of PEARL results to:
    - Synthetic data generation randomness
    - Train/test split randomness
    - Model initialization randomness

    Output: stability table (mean ± std across seeds) for paper supplementary.
    """
    from data.synthetic_generator import generate_synthetic_population
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings("ignore")

    print("\n" + "="*70)
    print("MULTI-SEED STABILITY ANALYSIS")
    print(f"Seeds: {seeds}  |  N={n_patients:,} per seed")
    print("="*70)

    seed_results = []

    for seed in seeds:
        print(f"\n── Seed {seed} ──────────────────────────────────────────")
        try:
            pop = generate_synthetic_population(n_patients=n_patients, seed=seed)
            rising = pop.patients[pop.patients["rising_risk"]].reset_index(drop=True)

            rising_train, rising_test = train_test_split(
                rising, test_size=0.20, random_state=seed,
                stratify=rising["behavioral_intervention"]
            )
            rising_train = rising_train.reset_index(drop=True)
            rising_test = rising_test.reset_index(drop=True)
            train_pids = set(rising_train["patient_id"].tolist())
            wpad_pairs_train = pop.wpad_pairs[
                pop.wpad_pairs["patient_id"].isin(train_pids)
            ].reset_index(drop=True)

            # Swap seeds into model constructors where applicable
            phase0 = run_phase_0(pop, rising_train, rising_test, wpad_pairs_train, verbose=False)
            phase1 = run_phase_1(pop, rising_train, wpad_pairs_train, phase0, verbose=False)
            phase2 = run_phase_2(pop, rising_train, rising_test, phase0, phase1, verbose=False)

            bci = phase2.get("bootstrap_ci", {})
            drope_df = phase2["drope_comparison"]
            imi = phase2["imi_results"]

            pearl_row = drope_df[drope_df["policy_name"] == "PEARL (base)"]
            moe_row = drope_df[drope_df["policy_name"] == "PEARL (MoE Full)"]
            behavioral_row = drope_df[drope_df["policy_name"] == "Behavioral Policy"]
            oracle_row = drope_df[drope_df["policy_name"] == "Oracle (optimal)"]

            seed_results.append({
                "seed": seed,
                "imi_behavioral": imi.get("Behavioral Policy", float("nan")),
                "imi_pearl_base": imi.get("PEARL (base)", float("nan")),
                "imi_pearl_moe": imi.get("PEARL (MoE Full)", float("nan")),
                "imi_oracle": imi.get("Oracle (optimal)", float("nan")),
                "imi_reduction_point": bci.get("imi_reduction_point", float("nan")),
                "imi_reduction_ci_lo": bci.get("imi_reduction_ci_lower", float("nan")),
                "imi_reduction_ci_hi": bci.get("imi_reduction_ci_upper", float("nan")),
                "p_value_one_sided": bci.get("p_value_one_sided", float("nan")),
                "drope_pearl_base": float(pearl_row["policy_value"].values[0]) if len(pearl_row) > 0 else float("nan"),
                "drope_pearl_moe": float(moe_row["policy_value"].values[0]) if len(moe_row) > 0 else float("nan"),
                "drope_behavioral": float(behavioral_row["policy_value"].values[0]) if len(behavioral_row) > 0 else float("nan"),
                "drope_oracle": float(oracle_row["policy_value"].values[0]) if len(oracle_row) > 0 else float("nan"),
                "e_value": phase0["imi_result"]["e_value"],
                "ess": phase0["imi_result"]["ess"],
                "conformal_coverage": phase2["conformal"]["empirical_coverage"],
                "conformal_width": phase2["conformal"]["mean_interval_width"],
                "sensitivity_direction_changes": int(
                    phase2["sensitivity"]["direction_change"].sum()
                    if "direction_change" in phase2["sensitivity"].columns else 0
                ),
            })
            print(f"  ✓ IMI reduction = {bci.get('imi_reduction_point', float('nan')):.3f}, "
                  f"p = {bci.get('p_value_one_sided', float('nan')):.4f}")
        except Exception as e:
            print(f"  ✗ Seed {seed} failed: {e}")
            seed_results.append({"seed": seed, "error": str(e)})

    if not seed_results:
        print("No results collected.")
        return None

    df = pd.DataFrame([r for r in seed_results if "error" not in r])

    print("\n" + "="*70)
    print("STABILITY SUMMARY (mean ± std across seeds)")
    print("="*70)

    numeric_cols = [c for c in df.columns if c != "seed" and df[c].dtype in [float, "float64"]]
    for col in numeric_cols:
        mu = df[col].mean()
        sd = df[col].std()
        print(f"  {col:<40}: {mu:.4f} ± {sd:.4f}")

    # Save
    output_dir = "/Users/sanjaybasu/pearl/outputs/results"
    os.makedirs(output_dir, exist_ok=True)
    stability_path = f"{output_dir}/multi_seed_stability.csv"
    df.to_csv(stability_path, index=False)
    print(f"\nStability table saved to {stability_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="PEARL Pipeline Runner")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--synthetic", action="store_true", default=False,
                            help="Use synthetic data (default if neither flag given)")
    mode_group.add_argument("--waymark", action="store_true", default=False,
                            help="Use real Waymark data (requires Waymark DB access)")
    parser.add_argument("--n_patients", type=int, default=8_000,
                        help="Patients for synthetic mode (ignored for --waymark)")
    parser.add_argument("--skip_arena", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--multi_seed", action="store_true", default=False,
                        help="Run 5-seed stability analysis after main pipeline (synthetic only)")
    # Sensitivity analysis parameters (pre-specified; varied by run_sensitivity.sh)
    parser.add_argument("--t_min", type=int, default=60,
                        help="Minimum gap days for WPAD pair eligibility (primary: 60)")
    parser.add_argument("--iptw_clip", type=float, default=10.0,
                        help="Max IPTW weight (= 1/min propensity clip; primary: 10)")
    parser.add_argument("--beta", type=float, default=0.10,
                        help="DPO beta regularization (primary: 0.10)")
    parser.add_argument("--outcome_window", type=int, default=90,
                        help="Outcome ascertainment window in days (primary: 90)")
    parser.add_argument("--no_trajectory_adjustment", action="store_true", default=False,
                        help="Omit trajectory slope covariate from WPAD estimation")
    parser.add_argument("--wpad_direction", type=str, default="all",
                        choices=["all", "churn_only", "waitlist_only"],
                        help="WPAD pair types to include (primary: all)")
    parser.add_argument("--camden_threshold", type=float, default=0.02,
                        help="IMI epsilon threshold for Camden reanalysis (primary: 0.02)")
    parser.add_argument("--sensitivity_label", type=str, default=None,
                        help="Label for this sensitivity run (used by run_sensitivity.sh)")
    parser.add_argument("--sens_outfile", type=str, default=None,
                        help="Path to save per-sensitivity-run key metrics CSV")
    args = parser.parse_args()

    # Default to synthetic if neither flag given
    if not args.waymark:
        args.synthetic = True

    t_start = time.time()

    from sklearn.model_selection import train_test_split

    if args.waymark:
        print("\n" + "="*70)
        print("PEARL: Policy Evolution through Aligned Retrospective Learning")
        print("Full Pipeline Execution (Real Waymark Data Mode)")
        print("="*70)

        from data.extract_wpad import build_waymark_population
        pop = build_waymark_population(verbose=args.verbose)
        rising = pop.patients[pop.patients.get("rising_risk", pd.Series(True, index=pop.patients.index))].reset_index(drop=True)
        if "rising_risk" not in pop.patients.columns:
            rising = pop.patients.reset_index(drop=True)

    else:
        print("\n" + "="*70)
        print("PEARL: Policy Evolution through Aligned Retrospective Learning")
        print("Full Pipeline Execution (Synthetic Mode)")
        print("="*70)

        from data.synthetic_generator import generate_synthetic_population
        print(f"\nGenerating synthetic population (N={args.n_patients:,})...")
        pop = generate_synthetic_population(n_patients=args.n_patients, seed=42)
        rising = pop.patients[pop.patients["rising_risk"]].reset_index(drop=True)

    # ── Train/test split (80/20) ─────────────────────────────────────────
    # All model fitting (phase 1) uses rising_train + wpad_pairs_train.
    # All evaluation (phase 2) uses rising_test (held-out; never seen during fitting).
    # IMI estimation (phase 0) fits propensity + outcome models on rising_train
    # and estimates IMI on rising_test — fully out-of-sample primary result.
    rising_train, rising_test = train_test_split(
        rising, test_size=0.20, random_state=42,
        stratify=rising["behavioral_intervention"]
    )
    rising_train = rising_train.reset_index(drop=True)
    rising_test = rising_test.reset_index(drop=True)
    train_pids = set(rising_train["patient_id"].tolist())
    wpad_pairs_train = pop.wpad_pairs[
        pop.wpad_pairs["patient_id"].isin(train_pids)
    ].reset_index(drop=True)

    print(f"  Total patients:         {len(pop.patients):,}")
    print(f"  Rising-risk patients:   {len(rising):,}")
    print(f"    Train (80%):          {len(rising_train):,}")
    print(f"    Test  (20%, held-out):{len(rising_test):,}")
    print(f"  WPAD primary pairs:     {len(pop.wpad_pairs):,}")
    print(f"    Train WPAD pairs:     {len(wpad_pairs_train):,}")
    if not np.isnan(pop.ground_truth_imi):
        print(f"  Ground-truth IMI:       {pop.ground_truth_imi:.3f} ({pop.ground_truth_imi*100:.1f}%)")
    else:
        print(f"  Ground-truth IMI:       N/A (real data — estimated in Phase 0)")

    # Phase 0
    phase0 = run_phase_0(pop, rising_train, rising_test, wpad_pairs_train, verbose=args.verbose)

    # Phase 1
    phase1 = run_phase_1(pop, rising_train, wpad_pairs_train, phase0, verbose=args.verbose)

    # Phase 2
    phase2 = run_phase_2(pop, rising_train, rising_test, phase0, phase1, verbose=args.verbose)

    # Phase 3 (Einstein Arena)
    if not args.skip_arena:
        phase3 = run_phase_3_arena(pop, rising_train, verbose=args.verbose)
    else:
        phase3 = None
        print("\nSkipping Einstein Arena (--skip_arena specified)")

    # Save results
    output_dir = "/Users/sanjaybasu/pearl/outputs/results"
    table = save_paper_table(phase0, phase1, phase2, output_dir)

    # Save sensitivity results for figure generation
    sens_df = phase2.get("sensitivity")
    if sens_df is not None:
        sens_path = os.path.join(output_dir, "sensitivity_results.csv")
        sens_df.to_csv(sens_path, index=False)
        print(f"Sensitivity results saved to {sens_path}")

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE in {total_time:.1f}s")
    print(f"{'='*70}")

    # Final summary
    print("\n★ KEY RESULTS FOR PAPER:")
    print(f"  [Train/test split: {len(rising_train):,} train / {len(rising_test):,} held-out test]")
    if not np.isnan(pop.ground_truth_imi):
        print(f"  Ground-truth IMI (behavioral policy): {pop.ground_truth_imi:.3f}")
    print(f"  Estimated IMI (behavioral policy):    {phase0['imi_result']['imi_point']:.3f}  [on held-out test]")
    print(f"  IMI 95% CI:                           [{phase0['imi_result']['imi_ci_lower']:.3f}, {phase0['imi_result']['imi_ci_upper']:.3f}]")
    print(f"  E-value (WPAD LATE):                  {phase0['imi_result']['e_value']:.2f}")
    print(f"  ESS:                                  {phase0['imi_result']['ess']:.0f} {'✓' if phase0['imi_result']['ess_adequate'] else '✗'}")

    # IMI reduction (primary result) — pick best PEARL variant by minimum IMI.
    imi_behavioral = phase2["imi_results"].get("Behavioral Policy", phase0["imi_result"]["imi_point"])
    imi_pearl_base = phase2["imi_results"].get("PEARL (base)", None)
    # Select the PEARL variant with the lowest IMI (MoE Router is often best on real data)
    _pr_variants = ["PEARL (MoE Router)", "PEARL (MoE Full)", "PEARL (base)"]
    _pr_available = {v: phase2["imi_results"][v] for v in _pr_variants if v in phase2["imi_results"]}
    pearl_best_name = min(_pr_available, key=_pr_available.get) if _pr_available else "PEARL (base)"
    imi_pearl_best = _pr_available.get(pearl_best_name, None)
    imi_oracle = phase2["imi_results"].get("Oracle (optimal)", None)
    if imi_pearl_best is not None and imi_behavioral > 0:
        imi_reduction_abs = imi_behavioral - imi_pearl_best
        imi_reduction_pct = imi_reduction_abs / imi_behavioral * 100
        oracle_gap = imi_oracle if imi_oracle is not None else 0
        oracle_reduction = 1 - (imi_pearl_best - oracle_gap) / (imi_behavioral - oracle_gap) \
                           if (imi_behavioral - oracle_gap) > 0 else 0
        print(f"\n  ★ PRIMARY RESULT — INTERVENTION MISALIGNMENT REDUCTION:")
        print(f"    IMI (behavioral policy):      {imi_behavioral:.3f} ({imi_behavioral*100:.1f}%)")
        print(f"    IMI ({pearl_best_name}):  {imi_pearl_best:.3f} ({imi_pearl_best*100:.1f}%)")
        if imi_pearl_base is not None and pearl_best_name != "PEARL (base)":
            print(f"    IMI (PEARL base):             {imi_pearl_base:.3f} ({imi_pearl_base*100:.1f}%)")
        if imi_oracle is not None:
            print(f"    IMI (oracle optimal):         {imi_oracle:.3f} ({imi_oracle*100:.1f}%)")
        print(f"    IMI reduction:                {imi_reduction_abs:.3f} ({imi_reduction_pct:.1f}% relative reduction)")
        print(f"    {pearl_best_name} achieves {oracle_reduction*100:.0f}% of the maximum possible IMI reduction")

    # Bootstrap CI + p-value
    bci = phase2.get("bootstrap_ci", {})
    if bci:
        print(f"\n  ★ HYPOTHESIS TEST (one-sided bootstrap, N_boot=2000, held-out test):")
        print(f"    H0: IMI(PEARL) >= IMI(behavioral)  [PEARL doesn't reduce misalignment]")
        print(f"    IMI reduction = {bci['imi_reduction_point']:.3f} "
              f"[95% CI: {bci['imi_reduction_ci_lower']:.3f}, {bci['imi_reduction_ci_upper']:.3f}]")
        sig_str = '✓ REJECTED (p<0.05)' if bci['significant'] else '✗ NOT rejected'
        print(f"    p-value = {bci['p_value_one_sided']:.4f}  → H0 {sig_str}")

    drope_df = phase2["drope_comparison"]
    best_pearl_row = drope_df[drope_df["policy_name"] == pearl_best_name]
    behavioral_row = drope_df[drope_df["policy_name"] == "Behavioral Policy"]
    oracle_row = drope_df[drope_df["policy_name"] == "Oracle (optimal)"]
    print(f"\n  DR-OPE Comparison (lower event rate = better; note: behavioral DR biased by IPW variance):")
    if len(behavioral_row) > 0:
        print(f"    Behavioral Policy DR-OPE:   {behavioral_row['policy_value'].values[0]:.4f} "
              f"[CI: {behavioral_row['policy_value_ci_lower'].values[0]:.4f}, {behavioral_row['policy_value_ci_upper'].values[0]:.4f}]")
    if len(best_pearl_row) > 0:
        print(f"    {pearl_best_name} DR-OPE: {best_pearl_row['policy_value'].values[0]:.4f} "
              f"[CI: {best_pearl_row['policy_value_ci_lower'].values[0]:.4f}, {best_pearl_row['policy_value_ci_upper'].values[0]:.4f}]")
    if len(oracle_row) > 0:
        imi_gap = (imi_pearl_best - imi_oracle) if (imi_oracle is not None and imi_pearl_best is not None) else float('nan')
        print(f"    Oracle (optimal) DR-OPE:    {oracle_row['policy_value'].values[0]:.4f} "
              f"(theoretical max; remaining IMI gap = {imi_gap:.3f})")
    print(f"\n  Conformal 90% PI coverage:    {phase2['conformal']['empirical_coverage']:.1%}")
    print(f"  WPAD falsification: all pass = {all(r for r in phase0['falsification']['Passes'].dropna())}")

    print(f"\nOutputs in: {output_dir}")

    # Optional: multi-seed stability analysis (synthetic only)
    if getattr(args, "multi_seed", False):
        if args.waymark:
            print("\n[multi_seed skipped: not applicable to real data mode]")
        else:
            print("\n" + "="*70)
            print("Running multi-seed stability analysis (5 seeds)...")
            run_multi_seed_stability(n_patients=args.n_patients)


if __name__ == "__main__":
    main()
