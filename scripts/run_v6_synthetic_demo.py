#!/usr/bin/env python3
"""
Synthetic-Data Reproduction Pipeline for PEARL v6 (Zero Patient Data).

Generates synthetic Medicaid care-management cohorts with known heterogeneous
treatment effects across four multidisciplinary care-team tracks and runs the
exact leak-free PEARL v6 estimation, component-separation ablation,
covariate-determinism audit, switched-patient nonlinear propensity diagnostics,
end-to-end Bayesian bootstrap uncertainty, and matched-capacity decision-curve
comparison on CPU (~15–25 seconds).

Usage:
  python scripts/run_v6_synthetic_demo.py
  python scripts/run_v6_synthetic_demo.py --n_patients 3000 --n_boot 30
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

import pearl_v6_revision as v6


def generate_synthetic_cohort(n_patients: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic 4-track care-management cohort with acute-recency confounding."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(38, 14, n_patients), 18, 80).round(1)
    female = rng.binomial(1, 0.62, n_patients)
    race_eth = rng.choice(
        ["White non-Hispanic", "Black non-Hispanic", "Hispanic", "Asian", "Other"],
        size=n_patients,
        p=[0.36, 0.42, 0.12, 0.05, 0.05],
    )
    state = rng.choice(["VA", "WA", "OH"], size=n_patients, p=[0.55, 0.35, 0.10])
    month_num = rng.integers(0, 26, size=n_patients)
    primary_language = rng.choice(["English", "Spanish", "Other"], size=n_patients, p=[0.88, 0.09, 0.03])
    adi_percentile = rng.uniform(1, 100, n_patients).round(1)
    adi_quintile = pd.qcut(adi_percentile, 5, labels=False) + 1

    has_mh = rng.binomial(1, 0.40, n_patients)
    has_htn = rng.binomial(1, 0.28, n_patients)
    has_copd = rng.binomial(1, 0.18, n_patients)
    has_dm = rng.binomial(1, 0.15, n_patients)
    has_chf = rng.binomial(1, 0.06, n_patients)
    has_ckd = rng.binomial(1, 0.07, n_patients)
    n_chronic = has_mh + has_htn + has_copd + has_dm + has_chf + has_ckd
    charlson = np.clip(n_chronic * 0.6 + rng.poisson(0.4, n_patients), 0, 10).astype(float)

    prior_ed = rng.poisson(0.9 + 0.5 * has_mh + 0.4 * has_copd, n_patients)
    prior_hosp = rng.poisson(0.2 + 0.3 * has_chf, n_patients)
    pre_acute_30d = rng.binomial(1, np.clip(0.18 + 0.20 * (prior_ed >= 2) + 0.15 * has_mh, 0.05, 0.85))
    fills_90d = rng.poisson(2.5 + 1.5 * has_dm + 1.5 * has_htn, n_patients)
    missed_fills = rng.binomial(1, 0.22, n_patients)

    # Calibrated rising-risk score
    risk_logit = -1.6 + 0.22 * charlson + 0.35 * np.log1p(prior_ed) + 0.55 * pre_acute_30d
    risk_score = 1.0 / (1.0 + np.exp(-risk_logit))

    # Realistic multidisciplinary assignment with genuine overlap (non-deterministic)
    tracks = ["behavioral_health", "care_coordination", "chw_social", "pharmacy"]
    logits = np.column_stack([
        -1.3 + 1.2 * has_mh + 0.7 * pre_acute_30d,
        0.4 + 0.9 * pre_acute_30d + 0.15 * charlson,
        0.3 - 0.5 * pre_acute_30d + 0.01 * adi_percentile,
        -0.4 - 1.1 * pre_acute_30d + 0.8 * has_dm + 0.6 * has_htn,
    ])
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    action_idx = np.array([rng.choice(4, p=p) for p in probs])
    disc_action = np.array([tracks[i] for i in action_idx])

    # State-dependent potential outcome surface (post-acute BH/pharmacy benefit; stable CHW/CC)
    base_logit = -1.8 + 0.18 * charlson + 0.30 * np.log1p(prior_ed) + 0.85 * pre_acute_30d + 0.35 * has_mh
    tau = np.zeros((n_patients, 4))
    tau[:, 0] = -0.55 * pre_acute_30d * has_mh + 0.15 * (1 - has_mh)   # behavioral_health
    tau[:, 1] = -0.15 * pre_acute_30d                                   # care_coordination
    tau[:, 2] = 0.25 * pre_acute_30d - 0.15 * (1 - pre_acute_30d)       # chw_social
    tau[:, 3] = -0.35 * (has_dm | has_htn) - 0.20 * missed_fills        # pharmacy

    y_prob = 1.0 / (1.0 + np.exp(-(base_logit + tau[np.arange(n_patients), action_idx])))
    y_90d = rng.binomial(1, y_prob)

    return pd.DataFrame({
        "age": age,
        "female": female,
        "race_eth": race_eth,
        "race_eth_full": race_eth,
        "state": state,
        "month_num": month_num,
        "primary_language": primary_language,
        "adi_percentile": adi_percentile,
        "adi_quintile": adi_quintile,
        "charlson_score": charlson,
        "prior_ed_visits_6mo": prior_ed,
        "prior_hosp_6mo": prior_hosp,
        "n_chronic": n_chronic,
        "pharmacy_fills_90d": fills_90d,
        "missed_pharmacy_fills": missed_fills,
        "has_diabetes": has_dm,
        "has_chf": has_chf,
        "has_copd": has_copd,
        "has_hypertension": has_htn,
        "has_ckd": has_ckd,
        "has_mh": has_mh,
        "risk_score": risk_score,
        "risingRiskScore": risk_score,
        "disc_action": disc_action,
        "y_enc_90d": y_90d,
    })


def main():
    parser = argparse.ArgumentParser(description="Run PEARL v6 leak-free pipeline on synthetic data.")
    parser.add_argument("--n_patients", type=int, default=2500, help="Synthetic cohort size (default: 2500)")
    parser.add_argument("--n_boot", type=int, default=20, help="End-to-end Bayesian bootstrap replicates (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    t0 = time.time()
    df = generate_synthetic_cohort(n_patients=args.n_patients, seed=args.seed)
    res, dca, _ = v6.analyze_cohort(
        df,
        action_col="disc_action",
        outcome_col="y_enc_90d",
        min_n=50,
        label=f"Synthetic 4-track care-management demo (N={args.n_patients:,}, B={args.n_boot})",
        include_site_time=True,
        run_subgroups=False,
        run_dca=True,
        run_prop_floor_sens=True,
        run_disjoint_split=True,
        b_boot=args.n_boot,
    )
    elapsed = time.time() - t0
    ab_primary = res["ablation"][1]
    print("\n" + "=" * 72)
    print(f"SYNTHETIC DEMO COMPLETED CLEANLY in {elapsed:.1f}s (ZERO REAL DATA)")
    print(f"  Covariate predictability (GBDT):   {res['determinism']['gb_accuracy_pct']:.1f}%")
    print(f"  Independent RF AUROC:              {res['diagnostics'][1]['auroc']:.3f}")
    print(f"  Nested-selected lambda*:           {res['lambda_selected']}")
    print(f"  Baseline -> PEARL misalignment:    {ab_primary['misalign_baseline_at_lam_pct']:.2f}% -> {ab_primary['misalign_policy_at_lam_pct']:.2f}%")
    print(f"  Model-predicted events / 1,000:    {ab_primary['model_predicted_events_averted_per_1000_dm']:+.2f}")
    print("=" * 72)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"Saved synthetic summary to {out_path}")


if __name__ == "__main__":
    main()
