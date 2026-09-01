"""
PEARL Revision — model-free policy evaluation and evaluator diagnostics.

Reviewer 1 asks whether PEARL's advantage survives when the model used to
optimise the policy is not also the model used to score it. revision_analyses.py
answers that with independently fitted outcome models. This script removes the
outcome model from the comparison altogether.

R7. Self-normalised inverse propensity scoring (SNIPS) policy value. Uses only
    observed outcomes and an estimated action propensity, so no outcome model
    enters the estimate. Reported with a paired bootstrap against behavioral
    routing and an effective sample size.
R8. Evaluator diagnostics. Discrimination and calibration of each candidate
    outcome model for the action actually received, the between-action spread
    each model implies, and the agreement between models on which action is
    preferred. Establishes whether the models disagree because one is better or
    because the between-action contrast is not estimable.

Usage:
  python scripts/revision_modelfree.py
"""
import sys
import os
import json
import time
from pathlib import Path

_PEARL_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_PEARL_ROOT), str(_PEARL_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats

warnings.filterwarnings("ignore")

from revision_analyses import (
    INTV_ALPHA, SEED, EPSILON, feature_matrix, imi_from_mu,
    light_phase_0, build_policy_functions, crossfit_mu, IndependentEvaluator,
    calibration_metrics, RESULTS_DIR, disable_unused_crossval,
)

N_BOOT = 2000


# ═════════════════════════════════════════════════════════════════════════════
# R7. Model-free policy value
# ═════════════════════════════════════════════════════════════════════════════

def fit_propensity(train: pd.DataFrame, le: LabelEncoder):
    X = feature_matrix(train)
    a = le.transform(train["behavioral_intervention"].values)
    return LogisticRegression(C=0.1, max_iter=1000).fit(X, a)


def snips_value(y: np.ndarray, a_obs: np.ndarray, a_pol: np.ndarray,
                prop: np.ndarray, clip: float = 0.01) -> tuple:
    """
    Self-normalised IPS estimate of the policy value.

      V(pi) = sum_i w_i y_i / sum_i w_i,   w_i = 1{A_i = pi(X_i)} / e(A_i | X_i)

    Returns the estimate, the effective sample size of the weights, and the
    number of patients whose observed action matches the policy.
    """
    match = (a_obs == a_pol)
    e = np.clip(prop[np.arange(len(a_obs)), a_obs], clip, 1.0)
    w = np.where(match, 1.0 / e, 0.0)
    tot = w.sum()
    if tot <= 0:
        return float("nan"), 0.0, 0
    v = float((w * y).sum() / tot)
    ess = float(tot ** 2 / max((w ** 2).sum(), 1e-12))
    return v, ess, int(match.sum())


def run_r7(policies, test, prop_test, le, verbose=True) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("R7. MODEL-FREE POLICY VALUE (SNIPS; observed outcomes only)")
    print("=" * 70)
    y = test["y_behavioral"].values.astype(float)
    a_obs = le.transform(test["behavioral_intervention"].values)
    n = len(test)
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]

    recs = {k: le.transform(np.asarray(fn(test))) for k, fn in policies.items()}
    v_behav_boot = None
    rows = []
    for name, a_pol in recs.items():
        v, ess, nmatch = snips_value(y, a_obs, a_pol, prop_test)
        vb = np.array([snips_value(y[i], a_obs[i], a_pol[i], prop_test[i])[0]
                       for i in boot_idx])
        if name == "Behavioral Policy":
            v_behav_boot = vb
        rows.append({"policy": name, "snips_value": v,
                     "ci_lower": float(np.nanpercentile(vb, 2.5)),
                     "ci_upper": float(np.nanpercentile(vb, 97.5)),
                     "ess": ess, "n_matched": nmatch,
                     "pct_matched": 100.0 * nmatch / n,
                     "_boot": vb})

    for r in rows:
        if v_behav_boot is not None:
            diff = r["_boot"] - v_behav_boot
            r["diff_vs_behavioral"] = float(np.nanmean(diff))
            r["diff_ci_lower"] = float(np.nanpercentile(diff, 2.5))
            r["diff_ci_upper"] = float(np.nanpercentile(diff, 97.5))
            r["p_one_sided_better"] = float(np.nanmean(diff >= 0))
        del r["_boot"]

    df = pd.DataFrame(rows).sort_values("snips_value").reset_index(drop=True)
    if verbose:
        cols = ["policy", "snips_value", "ci_lower", "ci_upper", "ess",
                "pct_matched", "diff_vs_behavioral", "diff_ci_lower",
                "diff_ci_upper", "p_one_sided_better"]
        print(df[cols].to_string(index=False))
        print("\n  Lower SNIPS value = fewer observed 90-day acute care events.")
        print("  diff_vs_behavioral < 0 favours the policy over behavioral routing.")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# R8. Evaluator diagnostics
# ═════════════════════════════════════════════════════════════════════════════

def run_r8(test, mu_map: dict, le, verbose=True) -> tuple:
    print("\n" + "=" * 70)
    print("R8. EVALUATOR DIAGNOSTICS")
    print("=" * 70)
    y = test["y_behavioral"].values.astype(float)
    a_obs = le.transform(test["behavioral_intervention"].values)
    n = len(test)

    rows = []
    for name, mu in mu_map.items():
        p_own = mu[np.arange(n), a_obs]
        cal = calibration_metrics(y, p_own)
        spread = mu.max(axis=1) - mu.min(axis=1)
        rows.append({
            "evaluator": name,
            "auroc_observed_action": cal["auroc"],
            "brier": cal["brier"], "ece": cal["ece"],
            "calibration_slope": cal["calibration_slope"],
            "mean_predicted": float(p_own.mean()),
            "observed_rate": float(y.mean()),
            "mean_between_action_spread": float(spread.mean()),
            "median_between_action_spread": float(np.median(spread)),
            "pct_patients_spread_gt_epsilon": float((spread > EPSILON).mean() * 100),
        })
    diag = pd.DataFrame(rows)

    # agreement between evaluators on the preferred action
    names = list(mu_map)
    agree_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ai = mu_map[names[i]].argmin(axis=1)
            aj = mu_map[names[j]].argmin(axis=1)
            # Spearman correlation of the per-patient risk difference vs own action
            di = mu_map[names[i]][np.arange(n), a_obs] - mu_map[names[i]].min(axis=1)
            dj = mu_map[names[j]][np.arange(n), a_obs] - mu_map[names[j]].min(axis=1)
            rho = stats.spearmanr(di, dj).statistic
            agree_rows.append({
                "evaluator_a": names[i], "evaluator_b": names[j],
                "pct_same_preferred_action": float((ai == aj).mean() * 100),
                "spearman_rho_available_gain": float(rho),
            })
    agree = pd.DataFrame(agree_rows)

    if verbose:
        print(diag.to_string(index=False))
        print("\n  Agreement between evaluators:")
        print(agree.to_string(index=False))
    return diag, agree


# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    disable_unused_crossval()
    from data.extract_wpad import build_waymark_population
    from run_pipeline import run_phase_1

    le = LabelEncoder().fit(INTV_ALPHA)
    pop = build_waymark_population(verbose=False)
    rising = pop.patients.reset_index(drop=True)
    train, test = train_test_split(rising, test_size=0.20, random_state=SEED,
                                   stratify=rising["behavioral_intervention"])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    pairs_train = pop.wpad_pairs[
        pop.wpad_pairs["patient_id"].isin(set(train["patient_id"]))
    ].reset_index(drop=True)

    print(f"N train {len(train):,}  N test {len(test):,}  pairs {len(pairs_train):,}")

    phase0 = light_phase_0(train, test)
    phase1 = run_phase_1(pop, train, pairs_train, phase0, verbose=False)
    policies, test_p = build_policy_functions(pop, phase0, phase1, test)
    mu_primary = phase0["imi_result"]["mu_hat"]

    print("\nCross-fitting independent evaluators...")
    mu_rf = crossfit_mu(test_p, train, "rf", 5)
    mu_lg = crossfit_mu(test_p, train, "logit", 5)

    prop_model = fit_propensity(train, le)
    prop_test = prop_model.predict_proba(feature_matrix(test_p))

    r7 = run_r7(policies, test_p, prop_test, le, verbose=True)
    mu_map = {"primary_s_learner": mu_primary,
              "crossfit_rf": mu_rf, "crossfit_logit": mu_lg}
    diag, agree = run_r8(test_p, mu_map, le, verbose=True)

    r7.to_csv(RESULTS_DIR / "revision_r7_snips.csv", index=False)
    diag.to_csv(RESULTS_DIR / "revision_r8_evaluator_diagnostics.csv", index=False)
    agree.to_csv(RESULTS_DIR / "revision_r8_evaluator_agreement.csv", index=False)
    np.savez_compressed(RESULTS_DIR / "revision_mu_matrices.npz",
                        primary=mu_primary, crossfit_rf=mu_rf, crossfit_logit=mu_lg,
                        a_obs=le.transform(test_p["behavioral_intervention"].values),
                        y=test_p["y_behavioral"].values)
    print(f"\nSaved to {RESULTS_DIR}. Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
