#!/usr/bin/env python3
"""
PEARL v6 — Strictly Leak-Free Offline Policy Evaluation and Audit Pipeline.

Implements the four-safeguard offline routing and audit framework:
  1. Action-assignment validation & temporal ordering:
     - Validates the deterministic clinical-flag heuristic against documented
       care-plan goals (both [0, +90d] timed and all-time) and against
       completed-encounter clinician discipline, reporting overall agreement,
       Cohen's kappa, and per-action agreement.
     - Quantifies covariate determinism (multinomial logit and nonlinear
       HistGBM accuracy + propensity distributions) to separate genuine
       treatment overlap from propensity-model smoothing.
     - Enforces complete 90-day outcome observation inside the hospital-visit
       feed window [2023-06-01, 2025-10-30] (N = 18,936 complete-follow-up
       rising-risk cohort; full unwindowed N = 34,971 retained as sensitivity).
     - Evaluates two strictly validated, temporally ordered care-management
       cohorts anchored at actual documented action dates:
         (a) Completed-encounter clinician discipline (N = 5,929 rising-risk;
             4 operational tracks: care coordination, CHW/social navigation,
             clinical pharmacy, behavioral health).
         (b) Documented care-plan goals (N = 2,576 rising-risk; plus N = 7,818
             all-risk-tier sensitivity).
  2. Unfiltered within-patient onboarding analysis & falsification tests:
     - Complete 2x2 pre/post cross-tabulations BEFORE any outcome-based
       exclusions (reconciling the 742 y_on = 1 pairs).
     - Paired risk differences with 95% CIs and McNemar tests without
       outcome-based selection, plus a calendar-matched regression-to-the-mean
       benchmark among non-onboarded rising-risk patients.
     - Complete T7 staffing-capacity test (narrow, full pooled, state-stratified,
       and unfiltered complete-window specifications).
  3. Explicit separation of policy components & switched-patient support:
     - Separates independent evaluation, action-support restriction (admissible
       mask), switching margin (epsilon), and uncertainty penalization
       (lambda = 0, 0.5, 1.0, 2.0), plus propensity-floor sensitivity
       (0.01, 0.05, 0.10).
     - Reports switched-patient support diagnostics (multinomial logit and
       nonlinear HistGBM propensity distributions, ESS, and top transitions)
       specifically for patients whose recommended action changes.
  4. Zero-leakage estimation & end-to-end bootstrap uncertainty:
     - Fits all propensity models, selection outcome models (M = 20 bootstrap
       ensemble on train for resampling-based sigma_boot), nested-split lambda
       selection, and independent evaluators (RF and penalized logistic)
       strictly on the training set — zero test-set contamination.
     - Runs B = 200 end-to-end nonparametric bootstrap replicates resampling
       both train and test (refitting propensity, selection model, nested
       lambda tuning, policy, and independent evaluator in every replicate)
       for 95% CIs on baseline misalignment, PEARL misalignment, their paired
       difference, and model-predicted events averted per 1,000.
     - Also runs a 50/50 patient-disjoint train split (policy half A,
       evaluator half B) and doubly robust AIPW policy-value sensitivity.
  5. Decision-curve analysis vs existing risk score & primary-policy subgroups:
     - Compares the misalignment flag against existing risingRiskScore and
       outcome-model baseline risk at exact matched review capacity and across
       threshold probabilities pt in [0.01, 0.30], with bootstrap 95% CIs.
     - Rebuilds Appendix Table 4 strictly for the primary leak-free policy
       across all race/ethnicity, state, sex, age, and risk-quintile strata
       with bootstrap 95% CIs.

Outputs canonical JSON and CSV files to PEARL_OUTPUT_BASE (default: ./outputs/).
"""
import os
# Prevent BLAS/OpenMP thread oversubscription inside joblib workers on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import sys
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, cohen_kappa_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data.extract_wpad import (
    GOAL_MAP,
    _build_clinical_features,
    _build_demographics,
    _build_id_bridge,
    _compute_90d_outcome,
    _compute_prior_utilization,
    _load_raw,
    build_waymark_population,
)
from revision_analyses import (
    INTV_ALPHA,
    SEED,
    calibration_metrics,
    disable_unused_crossval,
    feature_matrix,
    load_eligibility_attributes,
)
from staffing_falsification_t7 import build_monthly_chw_headcount

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = Path(os.environ.get("PEARL_OUTPUT_BASE", str(_ROOT / "outputs")))
RAW_DIR = Path(os.environ.get("PEARL_DATA_PATH", str(_ROOT / "data" / "real_inputs")))
SVI_PATH = Path(os.environ.get("PEARL_SVI_PATH", str(_ROOT / "data" / "processed" / "svi_by_zip.parquet")))

EPSILON = 0.02
MIN_PROPENSITY = 0.01
LAMBDAS = [0.0, 0.5, 1.0, 2.0]
M_ENSEMBLE = 10          # IDENTICAL ensemble size for point estimate AND every bootstrap replicate
M_ENSEMBLE_MAIN = M_ENSEMBLE
M_ENSEMBLE_BOOT = M_ENSEMBLE
B_OUTER_BOOT = 200       # End-to-end nonparametric bootstrap replicates for 95% CIs
N_JOBS = max(1, min(10, (os.cpu_count() or 4) - 2))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_stratify(a: np.ndarray, y: np.ndarray, min_count: int = 10) -> np.ndarray:
    """Stratification vector that groups rare classes safely."""
    k = int(a.max()) + 1
    counts = np.bincount(a, minlength=k)
    strat = np.where(counts[a] >= min_count, a, -1)
    if (strat == -1).sum() < 5:
        strat = y.astype(int)
    if len(np.unique(strat)) < 2 or np.bincount(strat - strat.min()).min() < 2:
        return None
    return strat


def build_features(df: pd.DataFrame, include_site_time: bool = True) -> np.ndarray:
    """Baseline feature matrix (optionally including state indicators and calendar month)."""
    X = feature_matrix(df)
    if include_site_time:
        st = (
            pd.get_dummies(df["state"].fillna("Unlinked"), drop_first=False)
            .reindex(columns=["VA", "WA", "OH", "Unlinked"], fill_value=0)
            .values.astype(float)
        )
        m = df["month_num"].fillna(0.0).values.astype(float).reshape(-1, 1)
        X = np.hstack([X, st, m])
    return X


def pure_clinical_flag_rule(row) -> str:
    """The deterministic clinical-flag heuristic from extract_wpad.py (no goal lookup)."""
    if row.get("has_mh", 0):
        return "mental_health"
    elif row.get("has_chf", 0):
        return "heart_failure"
    elif row.get("has_copd", 0):
        return "pulmonary"
    elif row.get("has_hypertension", 0):
        return "hypertension"
    elif row.get("has_diabetes", 0):
        return "diabetes"
    elif row.get("has_substance", 0):
        return "substance_use"
    elif row.get("poor_adherence", 0):
        return "medication_adherence"
    else:
        return "care_access"


def map_discipline(x) -> str:
    if pd.isna(x):
        return None
    s = str(x)
    if "Pharmac" in s:
        return "pharmacy"
    if "Therap" in s:
        return "behavioral_health"
    if "Care Coordinator" in s:
        return "care_coordination"
    if "Community Health Worker" in s:
        return "chw_social_navigation"
    return None


def fit_propensity_logit(X_tr, a_tr, X_eval, k_act, w_ext=None):
    sc = StandardScaler().fit(X_tr)
    pm = LogisticRegression(C=0.1, max_iter=400).fit(
        sc.transform(X_tr), a_tr, sample_weight=w_ext
    )
    p_tr = np.zeros((len(X_tr), k_act))
    p_tr[:, pm.classes_] = pm.predict_proba(sc.transform(X_tr))
    p_ev = np.zeros((len(X_eval), k_act))
    p_ev[:, pm.classes_] = pm.predict_proba(sc.transform(X_eval))
    return p_tr, p_ev


def fit_propensity_gb(X_tr, a_tr, X_eval, k_act, seed=SEED):
    gb = HistGradientBoostingClassifier(
        max_depth=5, max_iter=100, learning_rate=0.08,
        early_stopping=False, random_state=seed,
    ).fit(X_tr, a_tr)
    p_ev = np.zeros((len(X_eval), k_act))
    p_ev[:, gb.classes_] = gb.predict_proba(X_eval)
    return p_ev


def fit_bootstrap_ensemble(
    X_tr, y_tr, a_tr, X_eval, k_act, learner="hgb", m_boot=M_ENSEMBLE,
    seed=SEED, rf_n_jobs=1, w_ext=None
):
    """
    Identical estimator used for BOTH point estimates (w_ext=None) and every
    outer Bayesian-bootstrap replicate (w_ext ~ Exponential(1), Rubin 1981):
      - Propensity: L2 multinomial logistic regression on standardized X_tr.
      - learner='hgb': M_ENSEMBLE 80%-without-replacement subsamples of
        IPW-weighted HistGBM (zero row-duplication leaf collapse; exact
        resampling-based mu_mean and sigma_boot with ddof=1).
      - learner='rf': IPW-weighted RandomForestRegressor (200 bagged trees,
        max_features=0.5), partitioned into M_ENSEMBLE disjoint 20-tree
        sub-forests for mu_mean and sub-forest SD (ddof=1).
      - learner='logit': M_ENSEMBLE 80%-subsampled IPW-weighted L2-penalized
        logistic regression with action x covariate interactions.
    """
    rng = np.random.default_rng(seed)
    n_tr, n_ev = len(X_tr), len(X_eval)
    if w_ext is None:
        w_ext = np.ones(n_tr, dtype=float)
    mu_all = np.zeros((m_boot, n_ev, k_act))

    X_ev_rep = np.repeat(X_eval, k_act, axis=0)
    oh_ev_rep = np.tile(np.eye(k_act), (n_ev, 1))
    X_ev_aug = np.hstack([X_ev_rep, oh_ev_rep])

    p_tr_base, p_ev_base = fit_propensity_logit(X_tr, a_tr, X_eval, k_act, w_ext=w_ext)
    p_recv_base = np.clip(p_tr_base[np.arange(n_tr), a_tr], 0.05, 1.0)
    w_base = (1.0 / p_recv_base) * w_ext
    w_base = w_base * n_tr / w_base.sum()

    oh_tr_full = np.eye(k_act)[a_tr]
    X_aug_full = np.hstack([X_tr, oh_tr_full])

    if learner == "rf":
        n_trees_per = 20
        model = RandomForestRegressor(
            n_estimators=n_trees_per * m_boot, max_depth=8, min_samples_leaf=20,
            max_features=0.5, random_state=seed, n_jobs=rf_n_jobs,
        )
        model.fit(X_aug_full, y_tr, sample_weight=w_base)
        for m in range(m_boot):
            sub_est = model.estimators_[m * n_trees_per : (m + 1) * n_trees_per]
            sub_pred = np.mean([t.predict(X_ev_aug) for t in sub_est], axis=0)
            mu_all[m] = np.clip(sub_pred, 0.0, 1.0).reshape(n_ev, k_act)
        ddof = 1 if m_boot > 1 else 0
        return mu_all.mean(axis=0), mu_all.std(axis=0, ddof=ddof), p_ev_base

    sub_n = max(10, int(0.8 * n_tr))
    for m in range(m_boot):
        idx = rng.choice(n_tr, size=sub_n, replace=False)
        if len(np.unique(a_tr[idx])) < 2:
            idx = np.arange(n_tr)
        wb = w_base[idx]
        wb = wb * len(idx) / wb.sum()

        if learner == "hgb":
            model = HistGradientBoostingRegressor(
                max_iter=150, max_depth=3, learning_rate=0.05,
                min_samples_leaf=20, early_stopping=False, random_state=seed + m,
            )
            model.fit(X_aug_full[idx], y_tr[idx], sample_weight=wb)
            preds = np.clip(model.predict(X_ev_aug), 0.0, 1.0)
            mu_all[m] = preds.reshape(n_ev, k_act)
        elif learner == "logit":
            Xb = X_tr[idx]
            sc = StandardScaler().fit(Xb)
            Xb_s = sc.transform(Xb)
            Xev_s = sc.transform(X_eval)
            oh_b = oh_tr_full[idx]
            inter_tr = np.einsum("ij,ik->ijk", oh_b, Xb_s).reshape(len(Xb_s), -1)
            X_int_tr = np.hstack([Xb_s, oh_b, inter_tr])
            model = LogisticRegression(C=0.05, max_iter=500, solver="lbfgs")
            model.fit(X_int_tr, (y_tr[idx] > 0.5).astype(int), sample_weight=wb)
            for j in range(k_act):
                oh_ev = np.zeros((n_ev, k_act))
                oh_ev[:, j] = 1.0
                inter_ev = np.einsum("ij,ik->ijk", oh_ev, Xev_s).reshape(n_ev, -1)
                mu_all[m, :, j] = model.predict_proba(np.hstack([Xev_s, oh_ev, inter_ev]))[:, 1]
        else:
            raise ValueError(f"Unknown learner: {learner}")

    ddof = 1 if m_boot > 1 else 0
    return mu_all.mean(axis=0), mu_all.std(axis=0, ddof=ddof), p_ev_base


def apply_policy(mu, sigma, adm, a_obs, lam, eps=EPSILON, use_margin=True, use_adm=True):
    """
    General routing rule isolating each safeguard component:
      - use_adm: restrict alternatives to adm mask
      - use_margin: abstain unless gain > eps (if False, switch whenever alt < own)
      - lam: uncertainty penalty multiplier (0 = no uncertainty penalty)
    """
    n = len(a_obs)
    if not use_adm and not use_margin and lam == 0.0:
        best = mu.argmin(axis=1)
        sw = best != a_obs
        return best, sw
    if use_adm and not use_margin and lam == 0.0:
        masked = np.where(adm, mu, np.inf)
        # Fallback to a_obs if no action admissible
        no_adm = np.isinf(masked).all(axis=1)
        best = masked.argmin(axis=1)
        best[no_adm] = a_obs[no_adm]
        sw = best != a_obs
        return best, sw

    ub_own = mu[np.arange(n), a_obs] - lam * sigma[np.arange(n), a_obs]
    lb_alt = mu + lam * sigma
    if use_adm:
        lb_alt = np.where(adm, lb_alt, np.inf)
    lb_alt[np.arange(n), a_obs] = np.inf
    best = lb_alt.argmin(axis=1)
    gain = ub_own - lb_alt[np.arange(n), best]
    thresh = eps if use_margin else 0.0
    sw = np.isfinite(gain) & (gain > thresh)
    out = a_obs.copy()
    out[sw] = best[sw]
    return out, sw


def compute_misalignment(mu, sigma, adm, a_pol, lam, eps=EPSILON):
    """Binary per-patient misalignment indicator under (mu, sigma, adm, lam, eps)."""
    n = len(a_pol)
    ub_own = mu[np.arange(n), a_pol] - lam * sigma[np.arange(n), a_pol]
    lb_alt = np.where(adm, mu + lam * sigma, np.inf)
    lb_alt[np.arange(n), a_pol] = np.inf
    min_alt = lb_alt.min(axis=1)
    return (np.isfinite(min_alt) & (min_alt < ub_own - eps)).astype(float)


def compute_dr_value(mu, prop, y, a_obs, a_pol, clip_lo=0.05):
    """Doubly robust (AIPW) policy value estimate for a deterministic policy a_pol."""
    n = len(y)
    p_rec = np.clip(prop[np.arange(n), a_obs], clip_lo, 1.0)
    match = (a_pol == a_obs).astype(float)
    psi = mu[np.arange(n), a_pol] + match * (y - mu[np.arange(n), a_obs]) / p_rec
    return float(psi.mean())


def switched_support_diagnostics(a_obs, a_pol, prop_logit, prop_gb, action_names):
    """Support diagnostics specifically for patients whose recommended action changes."""
    sw = a_pol != a_obs
    n_sw = int(sw.sum())
    if n_sw == 0:
        return {
            "n_switched": 0, "pct_switched": 0.0,
            "logit_prop_median": None, "logit_prop_q25": None, "logit_prop_q75": None,
            "logit_prop_p10": None, "logit_prop_min": None,
            "logit_pct_lt_005": None, "logit_pct_lt_010": None,
            "gb_prop_median": None, "gb_prop_q25": None, "gb_prop_q75": None,
            "gb_prop_min": None, "gb_pct_lt_005": None, "gb_pct_lt_010": None,
            "ess_logit": None, "ess_ratio_logit": None,
            "top_transitions": [],
        }
    p_l = prop_logit[sw, a_pol[sw]]
    p_g = prop_gb[sw, a_pol[sw]]
    w_l = 1.0 / np.clip(p_l, 0.01, 1.0)
    ess = float((w_l.sum() ** 2) / (w_l ** 2).sum())

    trans = pd.Series([f"{action_names[o]} -> {action_names[r]}" for o, r in zip(a_obs[sw], a_pol[sw])])
    top_t = [
        {"transition": k, "n": int(v), "pct_of_switched": round(100.0 * v / n_sw, 1)}
        for k, v in trans.value_counts().head(5).items()
    ]
    return {
        "n_switched": n_sw,
        "pct_switched": round(100.0 * n_sw / len(a_obs), 2),
        "logit_prop_median": round(float(np.median(p_l)), 4),
        "logit_prop_q25": round(float(np.percentile(p_l, 25)), 4),
        "logit_prop_q75": round(float(np.percentile(p_l, 75)), 4),
        "logit_prop_p10": round(float(np.percentile(p_l, 10)), 4),
        "logit_prop_min": round(float(np.min(p_l)), 4),
        "logit_pct_lt_005": round(float(100.0 * (p_l < 0.05).mean()), 1),
        "logit_pct_lt_010": round(float(100.0 * (p_l < 0.10).mean()), 1),
        "gb_prop_median": round(float(np.median(p_g)), 4),
        "gb_prop_q25": round(float(np.percentile(p_g, 25)), 4),
        "gb_prop_q75": round(float(np.percentile(p_g, 75)), 4),
        "gb_prop_min": round(float(np.min(p_g)), 4),
        "gb_pct_lt_005": round(float(100.0 * (p_g < 0.05).mean()), 1),
        "gb_pct_lt_010": round(float(100.0 * (p_g < 0.10).mean()), 1),
        "ess_logit": round(ess, 1),
        "ess_ratio_logit": round(ess / n_sw, 3),
        "top_transitions": top_t,
    }


def nested_select_lambda(X_tr, y_tr, a_tr, k_act, min_n, min_prop=MIN_PROPENSITY,
                         m_boot=M_ENSEMBLE, seed=SEED, rf_n_jobs=1, w_ext=None):
    """Select lambda by nested 70/30 split strictly inside train, scored by independent RF."""
    strat = safe_stratify(a_tr, y_tr, min_count=10)
    idx_all = np.arange(len(X_tr))
    in_tr, in_va = train_test_split(idx_all, test_size=0.30, random_state=seed + 11, stratify=strat)
    w_in_tr = w_ext[in_tr] if w_ext is not None else None
    w_in_va = w_ext[in_va] if w_ext is not None else np.ones(len(in_va), dtype=float)
    w_in_va = w_in_va / w_in_va.sum()

    mu_s, sg_s, pr_iv = fit_bootstrap_ensemble(
        X_tr[in_tr], y_tr[in_tr], a_tr[in_tr], X_tr[in_va], k_act,
        learner="hgb", m_boot=m_boot, seed=seed + 13, rf_n_jobs=1, w_ext=w_in_tr,
    )
    mu_e, _, _ = fit_bootstrap_ensemble(
        X_tr[in_tr], y_tr[in_tr], a_tr[in_tr], X_tr[in_va], k_act,
        learner="rf", m_boot=m_boot, seed=seed + 17, rf_n_jobs=rf_n_jobs, w_ext=w_in_tr,
    )
    counts_in = np.bincount(a_tr[in_tr], minlength=k_act)
    adm_iv = (pr_iv >= min_prop) & (counts_in >= min_n)[None, :]
    a_va, y_va = a_tr[in_va], y_tr[in_va]

    rows = []
    for lam in LAMBDAS:
        ap, sw = apply_policy(mu_s, sg_s, adm_iv, a_va, lam, eps=EPSILON)
        dm_val = float(np.sum(w_in_va * mu_e[np.arange(len(a_va)), ap]))
        dr_val = compute_dr_value(mu_e, pr_iv, y_va, a_va, ap)
        rows.append({
            "lambda": lam,
            "dm_value_independent_innerval": dm_val,
            "dr_value_independent_innerval": dr_val,
            "pct_switched": float(100.0 * np.sum(w_in_va * sw)),
        })
    best_row = min(rows, key=lambda r: r["dm_value_independent_innerval"])
    return float(best_row["lambda"]), rows


def _single_outer_bootstrap(
    b, X_tr, y_tr, a_tr, X_te, y_te, a_te, k_act, min_n, min_prop,
    subgroup_masks_te=None, rising_risk_te=None, flag_eval_lam=None
):
    """
    One complete end-to-end Bayesian-bootstrap replicate (Rubin 1981):
      - Draws continuous Exponential(1) weights w_tr on train (zero duplicate rows
        so tree min_samples_leaf remains exact) -> refits propensity, selection
        ensemble (HistGBM), nested-split lambda selection, and independent RF
        evaluator ensemble.
      - Draws continuous Exponential(1) weights w_te on test -> evaluates
        baseline, PEARL (nested lambda), fixed-lambda policies (0.0, 0.5, 1.0),
        support-only / unconstrained comparators, matched-capacity review,
        and subgroup metrics.
    """
    rng = np.random.default_rng(SEED * 1000 + b)
    n_tr, n_te = len(X_tr), len(X_te)
    w_tr = rng.exponential(1.0, size=n_tr)
    w_tr = w_tr * n_tr / w_tr.sum()
    w_te = rng.exponential(1.0, size=n_te)
    w_te_norm = w_te / w_te.sum()

    lam_b, _ = nested_select_lambda(
        X_tr, y_tr, a_tr, k_act, min_n, min_prop=min_prop,
        m_boot=M_ENSEMBLE, seed=SEED + 5000 + b * 17, rf_n_jobs=1, w_ext=w_tr,
    )
    mu_s, sg_s, pr_te = fit_bootstrap_ensemble(
        X_tr, y_tr, a_tr, X_te, k_act, learner="hgb",
        m_boot=M_ENSEMBLE, seed=SEED + 10000 + b * 19, rf_n_jobs=1, w_ext=w_tr,
    )
    mu_e, sg_e, _ = fit_bootstrap_ensemble(
        X_tr, y_tr, a_tr, X_te, k_act, learner="rf",
        m_boot=M_ENSEMBLE, seed=SEED + 20000 + b * 23, rf_n_jobs=1, w_ext=w_tr,
    )
    counts_b = np.bincount(a_tr, minlength=k_act)
    adm_b = (pr_te >= min_prop) & (counts_b >= min_n)[None, :]

    lam_score = lam_b if flag_eval_lam is None else flag_eval_lam
    a_pearl, _ = apply_policy(mu_s, sg_s, adm_b, a_te, lam_b, eps=EPSILON)
    a_l0, _ = apply_policy(mu_s, sg_s, adm_b, a_te, 0.0, eps=EPSILON)
    a_l05, _ = apply_policy(mu_s, sg_s, adm_b, a_te, 0.5, eps=EPSILON)
    a_l10, _ = apply_policy(mu_s, sg_s, adm_b, a_te, 1.0, eps=EPSILON)
    a_adm_only, _ = apply_policy(mu_s, sg_s, adm_b, a_te, 0.0, eps=0.0, use_margin=False, use_adm=True)
    a_uncon, _ = apply_policy(mu_s, sg_s, adm_b, a_te, 0.0, eps=0.0, use_margin=False, use_adm=False)

    v_base = mu_e[np.arange(n_te), a_te]
    p_rec = np.clip(pr_te[np.arange(n_te), a_te], 0.05, 1.0)
    psi_base = v_base + (y_te - v_base) / p_rec

    def metrics_for(a_pol, lam_m):
        m_cur = compute_misalignment(mu_e, sg_e, adm_b, a_te, lam_m, EPSILON)
        m_pol = compute_misalignment(mu_e, sg_e, adm_b, a_pol, lam_m, EPSILON)
        v_pol = mu_e[np.arange(n_te), a_pol]
        psi_pol = v_pol + (a_pol == a_te).astype(float) * (y_te - v_base) / p_rec
        mc_w = float(np.sum(w_te_norm * m_cur) * 100.0)
        mp_w = float(np.sum(w_te_norm * m_pol) * 100.0)
        return {
            "mis_cur": mc_w,
            "mis_pol": mp_w,
            "mis_diff": mc_w - mp_w,
            "averted_dm": float(np.sum(w_te_norm * (v_base - v_pol)) * 1000.0),
            "averted_dr": float(np.sum(w_te_norm * (psi_base - psi_pol)) * 1000.0),
            "pct_sw": float(np.sum(w_te_norm * (a_pol != a_te)) * 100.0),
        }

    out = {
        "lam_selected": lam_b,
        "pearl_nested": metrics_for(a_pearl, lam_score),
        "lam_0.0": metrics_for(a_l0, 0.0),
        "lam_0.5": metrics_for(a_l05, 0.5),
        "lam_1.0": metrics_for(a_l10, 1.0),
        "adm_only": metrics_for(a_adm_only, 0.0),
        "unconstrained": metrics_for(a_uncon, 0.0),
        "eps_0.01_cur": float(np.sum(w_te_norm * compute_misalignment(mu_e, sg_e, adm_b, a_te, lam_b, 0.01)) * 100.0),
        "eps_0.01_pol": float(np.sum(w_te_norm * compute_misalignment(
            mu_e, sg_e, adm_b, apply_policy(mu_s, sg_s, adm_b, a_te, lam_b, 0.01)[0], lam_b, 0.01
        )) * 100.0),
        "eps_0.05_cur": float(np.sum(w_te_norm * compute_misalignment(mu_e, sg_e, adm_b, a_te, lam_b, 0.05)) * 100.0),
        "eps_0.05_pol": float(np.sum(w_te_norm * compute_misalignment(
            mu_e, sg_e, adm_b, apply_policy(mu_s, sg_s, adm_b, a_te, lam_b, 0.05)[0], lam_b, 0.05
        )) * 100.0),
        "auroc_sel": float(roc_auc_score(y_te, mu_s[np.arange(n_te), a_te], sample_weight=w_te)) if len(np.unique(y_te)) == 2 else np.nan,
        "auroc_ev": float(roc_auc_score(y_te, v_base, sample_weight=w_te)) if len(np.unique(y_te)) == 2 else np.nan,
    }

    if rising_risk_te is not None:
        flag_b = compute_misalignment(mu_e, sg_e, adm_b, a_te, lam_score, EPSILON).astype(int)
        k_fl = int(flag_b.sum())
        if k_fl > 0:
            ord_rs = np.argsort(-rising_risk_te, kind="mergesort")
            fl_rs = np.zeros(n_te, dtype=int); fl_rs[ord_rs[:k_fl]] = 1
            ord_mu = np.argsort(-v_base, kind="mergesort")
            fl_mu = np.zeros(n_te, dtype=int); fl_mu[ord_mu[:k_fl]] = 1

            def w_ppv(fl):
                wf = w_te[fl == 1]
                return float(np.sum(wf * y_te[fl == 1]) / wf.sum() * 100.0) if wf.sum() > 0 else np.nan

            out["mc_ppv_flag"] = w_ppv(flag_b)
            out["mc_ppv_rs"] = w_ppv(fl_rs)
            out["mc_ppv_mu"] = w_ppv(fl_mu)
            pt = 0.08
            odds = pt / (1.0 - pt)
            def w_nb(fl):
                return float(np.sum(w_te_norm * (fl == 1) * y_te) - np.sum(w_te_norm * (fl == 1) * (1 - y_te)) * odds)
            out["nb08_flag_minus_rs"] = w_nb(flag_b) - w_nb(fl_rs)
            out["nb08_flag_minus_mu"] = w_nb(flag_b) - w_nb(fl_mu)
        else:
            out["mc_ppv_flag"] = np.nan
            out["mc_ppv_rs"] = np.nan
            out["mc_ppv_mu"] = np.nan
            out["nb08_flag_minus_rs"] = np.nan
            out["nb08_flag_minus_mu"] = np.nan

    if subgroup_masks_te is not None:
        m_cur_vec = compute_misalignment(mu_e, sg_e, adm_b, a_te, lam_score, EPSILON)
        m_pol_vec = compute_misalignment(mu_e, sg_e, adm_b, a_pearl, lam_score, EPSILON)
        v_pol_vec = mu_e[np.arange(n_te), a_pearl]
        sg_res = {}
        for key, orig_mask in subgroup_masks_te.items():
            if orig_mask.sum() >= 10:
                wg = w_te[orig_mask]
                wg = wg / wg.sum()
                sg_res[key] = (
                    float(np.sum(wg * m_cur_vec[orig_mask]) * 100.0),
                    float(np.sum(wg * m_pol_vec[orig_mask]) * 100.0),
                    float(np.sum(wg * (v_base[orig_mask] - v_pol_vec[orig_mask])) * 1000.0),
                )
        out["subgroups"] = sg_res

    return out


def ci_from_vals(vals, point_est=None):
    """
    Efron Bias-Corrected (BC) 95% confidence interval (Efron & Tibshirani 1993).
    Equals the standard percentile CI [2.5%, 97.5%] when unbiased (median == point_est),
    and removes first-order nonlinear min-over-actions finite-sample bias when present.
    """
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) < 10:
        return [None, None]
    if point_est is None or not np.isfinite(point_est):
        return [round(float(np.percentile(arr, 2.5)), 3), round(float(np.percentile(arr, 97.5)), 3)]
    # Proportion of bootstrap replicates below point_est (with 0.5 continuity correction)
    prop_lt = (np.sum(arr < point_est) + 0.5 * np.sum(arr == point_est)) / len(arr)
    prop_lt = float(np.clip(prop_lt, 1.0 / (2.0 * len(arr)), 1.0 - 1.0 / (2.0 * len(arr))))
    z0 = float(stats.norm.ppf(prop_lt))
    a1 = float(stats.norm.cdf(2.0 * z0 - 1.96)) * 100.0
    a2 = float(stats.norm.cdf(2.0 * z0 + 1.96)) * 100.0
    lo = float(np.percentile(arr, np.clip(a1, 0.5, 99.5)))
    hi = float(np.percentile(arr, np.clip(a2, 0.5, 99.5)))
    # Ensure exact numerical ordering and containment of point estimate when boundary hits
    lo = min(lo, float(point_est))
    hi = max(hi, float(point_est))
    return [round(lo, 3), round(hi, 3)]


def analyze_cohort(
    df: pd.DataFrame,
    action_col: str,
    outcome_col: str,
    min_n: int,
    label: str,
    include_site_time: bool = True,
    run_subgroups: bool = False,
    run_dca: bool = False,
    run_prop_floor_sens: bool = False,
    run_disjoint_split: bool = False,
    b_boot: int = B_OUTER_BOOT,
):
    """Run the full leak-free analysis + end-to-end bootstrap on one cohort."""
    t0 = time.time()
    print(f"\n{'='*72}\nCOHORT: {label} (N={len(df):,}, min_n={min_n})\n{'='*72}")

    actions_unique = sorted(df[action_col].unique())
    le = LabelEncoder().fit(actions_unique)
    k_act = len(actions_unique)
    a_all = le.transform(df[action_col].values)
    y_all = df[outcome_col].values.astype(float)

    strat_all = safe_stratify(a_all, y_all, min_count=10)
    train, test = train_test_split(df, test_size=0.20, random_state=SEED, stratify=strat_all)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    X_tr = build_features(train, include_site_time=include_site_time)
    X_te = build_features(test, include_site_time=include_site_time)
    y_tr = train[outcome_col].values.astype(float)
    y_te = test[outcome_col].values.astype(float)
    a_tr = le.transform(train[action_col].values)
    a_te = le.transform(test[action_col].values)
    counts_tr = np.bincount(a_tr, minlength=k_act)
    counts_all = np.bincount(a_all, minlength=k_act)
    adm_actions = [actions_unique[j] for j in range(k_act) if counts_tr[j] >= min_n]

    # 1. Covariate determinism / propensity smoothing audit (5-fold CV on full cohort)
    X_all = build_features(df, include_site_time=include_site_time)
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pr_logit_cv = np.zeros((len(df), k_act))
    pr_gb_cv = np.zeros((len(df), k_act))
    strat_cv = safe_stratify(a_all, y_all, min_count=5)
    splitter = skf5.split(X_all, strat_cv) if strat_cv is not None else skf5.split(X_all, np.zeros(len(df)))
    for itr, ite in splitter:
        _, p_l = fit_propensity_logit(X_all[itr], a_all[itr], X_all[ite], k_act)
        pr_logit_cv[ite] = p_l
        pr_gb_cv[ite] = fit_propensity_gb(X_all[itr], a_all[itr], X_all[ite], k_act, seed=SEED)
    p_obs_l = pr_logit_cv[np.arange(len(df)), a_all]
    p_obs_g = pr_gb_cv[np.arange(len(df)), a_all]
    determinism = {
        "n": int(len(df)),
        "k_actions": int(k_act),
        "logit_accuracy_pct": round(100.0 * float((pr_logit_cv.argmax(1) == a_all).mean()), 1),
        "logit_p_obs_median": round(float(np.median(p_obs_l)), 3),
        "logit_p_obs_q25": round(float(np.percentile(p_obs_l, 25)), 3),
        "logit_p_obs_q75": round(float(np.percentile(p_obs_l, 75)), 3),
        "logit_pct_p_gt_090": round(100.0 * float((p_obs_l > 0.90).mean()), 1),
        "gb_accuracy_pct": round(100.0 * float((pr_gb_cv.argmax(1) == a_all).mean()), 1),
        "gb_p_obs_median": round(float(np.median(p_obs_g)), 3),
        "gb_pct_p_gt_090": round(100.0 * float((p_obs_g > 0.90).mean()), 1),
        "logit_mean_alt_ge_001": round(float(((pr_logit_cv >= 0.01).sum(1) - 1).mean()), 2),
        "gb_mean_alt_ge_001": round(float(((pr_gb_cv >= 0.01).sum(1) - 1).mean()), 2),
        "gb_pct_zero_alt_ge_005": round(100.0 * float(((pr_gb_cv >= 0.05).sum(1) <= 1).mean()), 1),
    }
    print(f"  Determinism: logit acc={determinism['logit_accuracy_pct']}%, "
          f"GB acc={determinism['gb_accuracy_pct']}%, "
          f"GB med p={determinism['gb_p_obs_median']}, "
          f"zero alt>=0.05: {determinism['gb_pct_zero_alt_ge_005']}%")

    # 2. Nested-split lambda selection strictly on train
    lam_star, lam_tuning = nested_select_lambda(
        X_tr, y_tr, a_tr, k_act, min_n, min_prop=MIN_PROPENSITY,
        m_boot=M_ENSEMBLE_MAIN, seed=SEED, rf_n_jobs=N_JOBS,
    )
    print(f"  Nested-split selected lambda* = {lam_star}")
    for r in lam_tuning:
        print(f"    lam={r['lambda']:<3}: innerval_DM={r['dm_value_independent_innerval']:.5f}, "
              f"innerval_DR={r['dr_value_independent_innerval']:.5f}, switched={r['pct_switched']:.2f}%")

    # 3. Fit selection (HistGBM) and independent evaluators (RF + penalized logistic) strictly on train
    mu_sel, sg_sel, pr_te_logit = fit_bootstrap_ensemble(
        X_tr, y_tr, a_tr, X_te, k_act, learner="hgb",
        m_boot=M_ENSEMBLE_MAIN, seed=SEED + 101, rf_n_jobs=1,
    )
    mu_ev, sg_ev, _ = fit_bootstrap_ensemble(
        X_tr, y_tr, a_tr, X_te, k_act, learner="rf",
        m_boot=M_ENSEMBLE_MAIN, seed=SEED + 202, rf_n_jobs=N_JOBS,
    )
    mu_lg, sg_lg, _ = fit_bootstrap_ensemble(
        X_tr, y_tr, a_tr, X_te, k_act, learner="logit",
        m_boot=min(10, M_ENSEMBLE_MAIN), seed=SEED + 303, rf_n_jobs=1,
    )
    pr_te_gb = fit_propensity_gb(X_tr, a_tr, X_te, k_act, seed=SEED + 404)

    adm_te = (pr_te_logit >= MIN_PROPENSITY) & (counts_tr >= min_n)[None, :]
    n_te = len(test)

    # 4. Model diagnostics
    diag = []
    for nm, m_, s_ in [
        (f"Selection (HistGBM, {M_ENSEMBLE}-boot ensemble)", mu_sel, sg_sel),
        (f"Independent primary (Random Forest, 200-tree {M_ENSEMBLE}-subforest)", mu_ev, sg_ev),
        (f"Independent secondary (Penalized Logistic, {M_ENSEMBLE}-boot ensemble)", mu_lg, sg_lg),
    ]:
        p_own = m_[np.arange(n_te), a_te]
        c = calibration_metrics(y_te, p_own)
        sp = np.where(adm_te, m_, np.nan)
        spread_val = float(np.nanmean(np.nanmax(sp, axis=1) - np.nanmin(sp, axis=1)))
        diag.append({
            "model": nm,
            "auroc": round(c["auroc"], 3),
            "brier": round(c["brier"], 4),
            "ece": round(c["ece"], 4),
            "calibration_slope": round(c["calibration_slope"], 2),
            "calibration_intercept": round(c["calibration_intercept"], 2),
            "mean_spread_admissible": round(spread_val, 4),
            "mean_boot_sd": round(float(s_.mean()), 4),
        })
    pref_sel = np.where(adm_te, mu_sel, np.inf).argmin(axis=1)
    pref_ev = np.where(adm_te, mu_ev, np.inf).argmin(axis=1)
    pref_lg = np.where(adm_te, mu_lg, np.inf).argmin(axis=1)
    agree_rf = round(float((pref_sel == pref_ev).mean() * 100.0), 1)
    agree_lg = round(float((pref_sel == pref_lg).mean() * 100.0), 1)
    agree_ev_lg = round(float((pref_ev == pref_lg).mean() * 100.0), 1)
    print(f"  Diagnostics: sel AUROC={diag[0]['auroc']}, RF AUROC={diag[1]['auroc']}, "
          f"Logit AUROC={diag[2]['auroc']} | pref agree GBM-RF={agree_rf}%, GBM-Logit={agree_lg}%")

    # 5. Component-separated policy ablation (explicitly separating every component!)
    policy_specs = [
        ("Baseline routing (reference)", a_te, lam_star, True, True, True, "rf"),
        (f"PEARL primary (support + margin eps=0.02 + nested lambda={lam_star:g})",
         apply_policy(mu_sel, sg_sel, adm_te, a_te, lam_star, EPSILON)[0], lam_star, True, True, lam_star > 0, "rf"),
        ("Support + margin (eps=0.02) + uncertainty penalty (lambda=0.5)",
         apply_policy(mu_sel, sg_sel, adm_te, a_te, 0.5, EPSILON)[0], 0.5, True, True, True, "rf"),
        ("Support + margin (eps=0.02) + uncertainty penalty (lambda=1.0)",
         apply_policy(mu_sel, sg_sel, adm_te, a_te, 1.0, EPSILON)[0], 1.0, True, True, True, "rf"),
        ("Support + margin (eps=0.02) + uncertainty penalty (lambda=2.0)",
         apply_policy(mu_sel, sg_sel, adm_te, a_te, 2.0, EPSILON)[0], 2.0, True, True, True, "rf"),
        ("Support restriction only (no margin eps=0, lambda=0)",
         apply_policy(mu_sel, sg_sel, adm_te, a_te, 0.0, 0.0, use_margin=False, use_adm=True)[0], 0.0, True, False, False, "rf"),
        ("Unconstrained argmin, all actions (independent RF evaluator)",
         apply_policy(mu_sel, sg_sel, adm_te, a_te, 0.0, 0.0, use_margin=False, use_adm=False)[0], 0.0, False, False, False, "rf"),
        ("Unconstrained argmin, all actions (self-evaluated by selection model)",
         apply_policy(mu_sel, sg_sel, adm_te, a_te, 0.0, 0.0, use_margin=False, use_adm=False)[0], 0.0, False, False, False, "sel"),
    ]

    v_base_rf = float(mu_ev[np.arange(n_te), a_te].mean())
    v_base_lg = float(mu_lg[np.arange(n_te), a_te].mean())
    v_base_sel = float(mu_sel[np.arange(n_te), a_te].mean())
    dr_base_rf = compute_dr_value(mu_ev, pr_te_logit, y_te, a_te, a_te)

    ablation_rows = []
    support_diags = {}
    for pname, apol, lam_m, has_sup, has_mar, has_pen, eval_src in policy_specs:
        match_pct = round(float((apol == a_te).mean() * 100.0), 2)
        sw_pct = round(float((apol != a_te).mean() * 100.0), 2)
        if eval_src == "rf":
            # Evaluate misalignment at lam_star for primary comparison AND at own lam_m
            m_cur_0 = float(compute_misalignment(mu_ev, sg_ev, adm_te, a_te, lam_star, EPSILON).mean() * 100.0)
            m_pol_0 = float(compute_misalignment(mu_ev, sg_ev, adm_te, apol, lam_star, EPSILON).mean() * 100.0)
            m_cur_own = float(compute_misalignment(mu_ev, sg_ev, adm_te, a_te, lam_m, EPSILON).mean() * 100.0)
            m_pol_own = float(compute_misalignment(mu_ev, sg_ev, adm_te, apol, lam_m, EPSILON).mean() * 100.0)
            val_dm = float(mu_ev[np.arange(n_te), apol].mean())
            averted_dm = (v_base_rf - val_dm) * 1000.0
            val_dr = compute_dr_value(mu_ev, pr_te_logit, y_te, a_te, apol)
            averted_dr = (dr_base_rf - val_dr) * 1000.0
            # Secondary independent evaluator (penalized logistic)
            m_cur_lg = float(compute_misalignment(mu_lg, sg_lg, adm_te, a_te, lam_m, EPSILON).mean() * 100.0)
            m_pol_lg = float(compute_misalignment(mu_lg, sg_lg, adm_te, apol, lam_m, EPSILON).mean() * 100.0)
            averted_lg = (v_base_lg - float(mu_lg[np.arange(n_te), apol].mean())) * 1000.0
        else:
            all_adm = np.ones_like(adm_te, dtype=bool)
            m_cur_0 = float(compute_misalignment(mu_sel, sg_sel, all_adm, a_te, 0.0, EPSILON).mean() * 100.0)
            m_pol_0 = float(compute_misalignment(mu_sel, sg_sel, all_adm, apol, 0.0, EPSILON).mean() * 100.0)
            m_cur_own, m_pol_own = m_cur_0, m_pol_0
            val_dm = float(mu_sel[np.arange(n_te), apol].mean())
            averted_dm = (v_base_sel - val_dm) * 1000.0
            averted_dr = np.nan
            m_cur_lg, m_pol_lg, averted_lg = np.nan, np.nan, np.nan

        ablation_rows.append({
            "policy": pname,
            "support_mask": has_sup,
            "switching_margin_002": has_mar,
            "uncertainty_penalty_lambda": lam_m if has_pen else 0.0,
            "evaluator": "Independent RF" if eval_src == "rf" else "Selection HistGBM (self)",
            "pct_recommendation_received": match_pct,
            "pct_switched": sw_pct,
            "misalign_baseline_at_lam_pct": round(m_cur_own, 2),
            "misalign_policy_at_lam_pct": round(m_pol_own, 2),
            "misalign_diff_at_lam_pp": round(m_cur_own - m_pol_own, 2),
            "misalign_policy_at_lamstar_pct": round(m_pol_0, 2),
            "expected_event_prob": round(val_dm, 5),
            "model_predicted_events_averted_per_1000_dm": round(averted_dm, 2),
            "model_predicted_events_averted_per_1000_dr": round(averted_dr, 2) if np.isfinite(averted_dr) else None,
            "logit_eval_misalign_baseline_pct": round(m_cur_lg, 2) if np.isfinite(m_cur_lg) else None,
            "logit_eval_misalign_policy_pct": round(m_pol_lg, 2) if np.isfinite(m_pol_lg) else None,
            "logit_eval_events_averted_per_1000": round(averted_lg, 2) if np.isfinite(averted_lg) else None,
        })
        if (apol != a_te).any() and eval_src == "rf":
            support_diags[pname] = switched_support_diagnostics(a_te, apol, pr_te_logit, pr_te_gb, actions_unique)

    # 6. Propensity-floor sensitivity (0.01, 0.05, 0.10)
    prop_floor_rows = []
    if run_prop_floor_sens:
        for pf in (0.01, 0.05, 0.10):
            adm_pf = (pr_te_logit >= pf) & (counts_tr >= min_n)[None, :]
            ap_pf, sw_pf = apply_policy(mu_sel, sg_sel, adm_pf, a_te, lam_star, EPSILON)
            m_c = float(compute_misalignment(mu_ev, sg_ev, adm_pf, a_te, lam_star, EPSILON).mean() * 100.0)
            m_p = float(compute_misalignment(mu_ev, sg_ev, adm_pf, ap_pf, lam_star, EPSILON).mean() * 100.0)
            av = (v_base_rf - float(mu_ev[np.arange(n_te), ap_pf].mean())) * 1000.0
            sd = switched_support_diagnostics(a_te, ap_pf, pr_te_logit, pr_te_gb, actions_unique)
            prop_floor_rows.append({
                "propensity_floor": pf,
                "pct_switched": round(float(100.0 * sw_pf.mean()), 2),
                "misalign_baseline_pct": round(m_c, 2),
                "misalign_pearl_pct": round(m_p, 2),
                "events_averted_per_1000": round(av, 2),
                "switched_logit_prop_median": sd["logit_prop_median"],
                "switched_gb_prop_median": sd["gb_prop_median"],
                "switched_ess": sd["ess_logit"],
            })

    # 7. Epsilon sensitivity (0.01, 0.02, 0.05)
    eps_rows = []
    for ep in (0.01, 0.02, 0.05):
        ap_e, sw_e = apply_policy(mu_sel, sg_sel, adm_te, a_te, lam_star, ep)
        m_c = float(compute_misalignment(mu_ev, sg_ev, adm_te, a_te, lam_star, ep).mean() * 100.0)
        m_p = float(compute_misalignment(mu_ev, sg_ev, adm_te, ap_e, lam_star, ep).mean() * 100.0)
        av = (v_base_rf - float(mu_ev[np.arange(n_te), ap_e].mean())) * 1000.0
        eps_rows.append({
            "epsilon": ep,
            "misalign_baseline_pct": round(m_c, 2),
            "misalign_pearl_pct": round(m_p, 2),
            "misalign_diff_pp": round(m_c - m_p, 2),
            "pct_switched": round(float(100.0 * sw_e.mean()), 2),
            "events_averted_per_1000": round(av, 2),
        })

    # 8. Patient-disjoint 50/50 train split sensitivity (policy half A, evaluator half B)
    disjoint_res = None
    if run_disjoint_split:
        strat_tr = safe_stratify(a_tr, y_tr, min_count=10)
        idx_a, idx_b = train_test_split(np.arange(len(train)), test_size=0.50, random_state=SEED + 77, stratify=strat_tr)
        mu_sa, sg_sa, pr_a = fit_bootstrap_ensemble(
            X_tr[idx_a], y_tr[idx_a], a_tr[idx_a], X_te, k_act, learner="hgb",
            m_boot=M_ENSEMBLE_MAIN, seed=SEED + 701, rf_n_jobs=1,
        )
        mu_eb, sg_eb, _ = fit_bootstrap_ensemble(
            X_tr[idx_b], y_tr[idx_b], a_tr[idx_b], X_te, k_act, learner="rf",
            m_boot=M_ENSEMBLE_MAIN, seed=SEED + 702, rf_n_jobs=N_JOBS,
        )
        counts_a = np.bincount(a_tr[idx_a], minlength=k_act)
        adm_a = (pr_a >= MIN_PROPENSITY) & (counts_a >= min_n // 2)[None, :]
        ap_a, sw_a = apply_policy(mu_sa, sg_sa, adm_a, a_te, lam_star, EPSILON)
        disjoint_res = {
            "n_half_a": int(len(idx_a)),
            "n_half_b": int(len(idx_b)),
            "pct_switched": round(float(100.0 * sw_a.mean()), 2),
            "misalign_baseline_pct": round(float(compute_misalignment(mu_eb, sg_eb, adm_a, a_te, lam_star, EPSILON).mean() * 100.0), 2),
            "misalign_pearl_pct": round(float(compute_misalignment(mu_eb, sg_eb, adm_a, ap_a, lam_star, EPSILON).mean() * 100.0), 2),
            "events_averted_per_1000": round(float((mu_eb[np.arange(n_te), a_te].mean() - mu_eb[np.arange(n_te), ap_a].mean()) * 1000.0), 2),
            "preferred_action_agreement_pct": round(float((np.where(adm_a, mu_sa, np.inf).argmin(1) == np.where(adm_a, mu_eb, np.inf).argmin(1)).mean() * 100.0), 1),
        }

    # 9. Observed-outcome check & Decision-curve analysis vs existing risingRiskScore & baseline RF risk
    flag_te = compute_misalignment(mu_ev, sg_ev, adm_te, a_te, lam_star, EPSILON).astype(int)
    p_own_rf = mu_ev[np.arange(n_te), a_te]
    obs_check = {
        "flagged_n": int(flag_te.sum()),
        "flagged_pct": round(float(100.0 * flag_te.mean()), 2),
        "events_flagged": int(y_te[flag_te == 1].sum()) if flag_te.sum() else 0,
        "rate_flagged_pct": round(float(100.0 * y_te[flag_te == 1].mean()), 2) if flag_te.sum() else None,
        "events_not_flagged": int(y_te[flag_te == 0].sum()),
        "rate_not_flagged_pct": round(float(100.0 * y_te[flag_te == 0].mean()), 2),
    }

    dca_df = None
    matched_cap = None
    if run_dca and flag_te.sum() > 0:
        k_fl = int(flag_te.sum())
        rs_tr = train["risingRiskScore"].fillna(train["risingRiskScore"].median()).values.reshape(-1, 1)
        rs_te = test["risingRiskScore"].fillna(train["risingRiskScore"].median()).values
        # Calibrated probability from existing risingRiskScore (fitted on train only!)
        cal_rs = LogisticRegression(C=1e4).fit(rs_tr, (y_tr > 0.5).astype(int))
        p_rs_te = cal_rs.predict_proba(rs_te.reshape(-1, 1))[:, 1]

        # Exact matched-capacity binary flags (top k_fl patients, deterministic tie-breaking by rank)
        order_rs = np.argsort(-rs_te, kind="mergesort")
        fl_rs_mc = np.zeros(n_te, dtype=int); fl_rs_mc[order_rs[:k_fl]] = 1

        order_mu = np.argsort(-p_own_rf, kind="mergesort")
        fl_mu_mc = np.zeros(n_te, dtype=int); fl_mu_mc[order_mu[:k_fl]] = 1

        # Combined trigger: logistic model of (risingRiskScore + misalignment_flag) or union?
        # Fit train-side flag via 5-fold OOF RF? Simpler & leak-free: combine rank of rs_te + flag_te at matched capacity
        # (flag patients prioritized, filled/trimmed by rs_te) — exact same review capacity k_fl!
        # Better: check union vs intersection too, plus continuous calibrated p_rs_te and p_own_rf
        prevalence = float(y_te.mean())
        total_events = float(y_te.sum())

        def strategy_summary(fl_bin, sname):
            tp = int(((fl_bin == 1) & (y_te == 1)).sum())
            fp = int(((fl_bin == 1) & (y_te == 0)).sum())
            n_rev = int(fl_bin.sum())
            ppv = 100.0 * tp / n_rev if n_rev else 0.0
            sens = 100.0 * tp / total_events if total_events else 0.0
            nbs = {}
            for pt in (0.04, 0.06, 0.08, 0.10, 0.12, 0.16):
                odds = pt / (1.0 - pt)
                nbs[f"nb_pt_{int(pt*100):02d}"] = round((tp / n_te) - (fp / n_te) * odds, 5)
            return {
                "strategy": sname,
                "n_reviewed": n_rev,
                "pct_reviewed": round(100.0 * n_rev / n_te, 2),
                "events_captured": tp,
                "ppv_observed_event_rate_pct": round(ppv, 2),
                "sensitivity_pct": round(sens, 1),
                "auroc_score": None,
                **nbs,
            }

        matched_cap = [
            strategy_summary(flag_te, "Misalignment flag (eps=0.02)"),
            strategy_summary(fl_rs_mc, "Existing rising-risk score (matched capacity)"),
            strategy_summary(fl_mu_mc, "Outcome-model baseline risk mu(X, A_obs) (matched capacity)"),
        ]
        matched_cap[1]["auroc_score"] = round(float(roc_auc_score(y_te, rs_te)), 3)
        matched_cap[2]["auroc_score"] = round(float(roc_auc_score(y_te, p_own_rf)), 3)

        # Full decision curve pt = 0.01 .. 0.30
        dc_rows = []
        for pt in np.arange(0.01, 0.31, 0.01):
            pt_r = round(float(pt), 2)
            odds = pt_r / (1.0 - pt_r)
            def nb_bin(fl):
                return float(((fl == 1) & (y_te == 1)).sum() / n_te - ((fl == 1) & (y_te == 0)).sum() / n_te * odds)
            fl_rs_cont = (p_rs_te >= pt_r).astype(int)
            fl_mu_cont = (p_own_rf >= pt_r).astype(int)
            dc_rows.append({
                "threshold_probability": pt_r,
                "nb_misalignment_flag": nb_bin(flag_te),
                "nb_rising_risk_matched": nb_bin(fl_rs_mc),
                "nb_rf_risk_matched": nb_bin(fl_mu_mc),
                "nb_rising_risk_continuous": nb_bin(fl_rs_cont),
                "nb_rf_risk_continuous": nb_bin(fl_mu_cont),
                "nb_review_all": prevalence - (1.0 - prevalence) * odds,
                "nb_review_none": 0.0,
            })
        dca_df = pd.DataFrame(dc_rows)

    # 10. Prepare subgroup masks (if requested, for Appendix Table 4)
    subgroup_masks = None
    subgroup_meta = None
    if run_subgroups:
        subgroup_masks = {}
        subgroup_meta = []
        strata_defs = [
            ("Race and ethnicity", test["race_eth_full"].fillna("Unlinked").values),
            ("State", test["state"].fillna("Unlinked").values),
            ("Sex", np.where(test["female"].values == 1, "Female", "Male")),
            ("Age band", pd.cut(test["age"], [0, 30, 45, 60, 120],
                                labels=["18-29", "30-44", "45-59", "60+"]).astype(str).values),
            ("Predicted-risk quintile", ("Q" + test["adi_quintile"].astype(int).astype(str)).values),
        ]
        for sname, svals in strata_defs:
            for grp in sorted(pd.unique(pd.Series(svals).dropna())):
                mask = (svals == grp)
                key = f"{sname}||{grp}"
                subgroup_masks[key] = mask
                subgroup_meta.append((key, sname, str(grp), mask))

    # 11. End-to-end nonparametric bootstrap (B = b_boot replicates)
    print(f"  Running B={b_boot} end-to-end bootstrap replicates across {N_JOBS} cores...")
    tb0 = time.time()
    rs_arr = test["risingRiskScore"].fillna(train["risingRiskScore"].median()).values if (run_dca and "risingRiskScore" in test.columns) else None
    boot_outs = Parallel(n_jobs=N_JOBS)(
        delayed(_single_outer_bootstrap)(
            b, X_tr, y_tr, a_tr, X_te, y_te, a_te, k_act, min_n, MIN_PROPENSITY,
            subgroup_masks_te=subgroup_masks, rising_risk_te=rs_arr, flag_eval_lam=None,
        )
        for b in range(b_boot)
    )
    print(f"  Bootstrap completed in {time.time() - tb0:.1f}s")

    # Summarize bootstrap CIs
    lam_dist = pd.Series([o["lam_selected"] for o in boot_outs]).value_counts().sort_index().to_dict()
    lam_dist = {str(k): int(v) for k, v in lam_dist.items()}

    # Map each bootstrap key to its exact point-estimate row in ablation_rows
    pkey_row_idx = {
        "pearl_nested": 1,
        "lam_0.0": 1 if lam_star == 0.0 else 1,
        "lam_0.5": 2,
        "lam_1.0": 3,
        "adm_only": 5,
        "unconstrained": 6,
    }
    # Compute exact point estimate for lam=0.0 separately in case lam_star != 0
    ap_l0_pt, _ = apply_policy(mu_sel, sg_sel, adm_te, a_te, 0.0, EPSILON)
    pt_l0 = {
        "misalign_baseline_at_lam_pct": float(compute_misalignment(mu_ev, sg_ev, adm_te, a_te, 0.0, EPSILON).mean() * 100.0),
        "misalign_policy_at_lam_pct": float(compute_misalignment(mu_ev, sg_ev, adm_te, ap_l0_pt, 0.0, EPSILON).mean() * 100.0),
        "model_predicted_events_averted_per_1000_dm": float((v_base_rf - mu_ev[np.arange(n_te), ap_l0_pt].mean()) * 1000.0),
        "model_predicted_events_averted_per_1000_dr": float((dr_base_rf - compute_dr_value(mu_ev, pr_te_logit, y_te, a_te, ap_l0_pt)) * 1000.0),
        "pct_switched": float((ap_l0_pt != a_te).mean() * 100.0),
    }
    pt_l0["misalign_diff_at_lam_pp"] = pt_l0["misalign_baseline_at_lam_pct"] - pt_l0["misalign_policy_at_lam_pct"]

    boot_cis = {
        "b_replicates": b_boot,
        "method": "Rubin Bayesian bootstrap (Exponential(1) weights on train and test) with Efron bias-corrected (BC) 95% CIs",
        "lambda_selection_counts": lam_dist,
        "auroc_sel_ci95": ci_from_vals([o["auroc_sel"] for o in boot_outs], diag[0]["auroc"]),
        "auroc_ev_ci95": ci_from_vals([o["auroc_ev"] for o in boot_outs], diag[1]["auroc"]),
    }
    for pkey in ("pearl_nested", "lam_0.0", "lam_0.5", "lam_1.0", "adm_only", "unconstrained"):
        pt_r = pt_l0 if pkey == "lam_0.0" else ablation_rows[pkey_row_idx[pkey]]
        boot_cis[pkey] = {
            "mis_cur_ci95": ci_from_vals([o[pkey]["mis_cur"] for o in boot_outs], pt_r["misalign_baseline_at_lam_pct"]),
            "mis_pol_ci95": ci_from_vals([o[pkey]["mis_pol"] for o in boot_outs], pt_r["misalign_policy_at_lam_pct"]),
            "mis_diff_ci95": ci_from_vals([o[pkey]["mis_diff"] for o in boot_outs], pt_r["misalign_diff_at_lam_pp"]),
            "averted_dm_ci95": ci_from_vals([o[pkey]["averted_dm"] for o in boot_outs], pt_r["model_predicted_events_averted_per_1000_dm"]),
            "averted_dr_ci95": ci_from_vals([o[pkey]["averted_dr"] for o in boot_outs], pt_r["model_predicted_events_averted_per_1000_dr"]),
            "pct_sw_ci95": ci_from_vals([o[pkey]["pct_sw"] for o in boot_outs], pt_r["pct_switched"]),
        }
    boot_cis["eps_0.01_cur_ci95"] = ci_from_vals([o["eps_0.01_cur"] for o in boot_outs], eps_rows[0]["misalign_baseline_pct"])
    boot_cis["eps_0.01_pol_ci95"] = ci_from_vals([o["eps_0.01_pol"] for o in boot_outs], eps_rows[0]["misalign_pearl_pct"])
    boot_cis["eps_0.05_cur_ci95"] = ci_from_vals([o["eps_0.05_cur"] for o in boot_outs], eps_rows[2]["misalign_baseline_pct"])
    boot_cis["eps_0.05_pol_ci95"] = ci_from_vals([o["eps_0.05_pol"] for o in boot_outs], eps_rows[2]["misalign_pearl_pct"])

    if run_dca and rs_arr is not None and matched_cap is not None:
        nb08_f_r = matched_cap[0]["nb_pt_08"] - matched_cap[1]["nb_pt_08"]
        nb08_f_m = matched_cap[0]["nb_pt_08"] - matched_cap[2]["nb_pt_08"]
        boot_cis["matched_capacity"] = {
            "ppv_flag_ci95": ci_from_vals([o.get("mc_ppv_flag") for o in boot_outs], matched_cap[0]["ppv_observed_event_rate_pct"]),
            "ppv_rs_ci95": ci_from_vals([o.get("mc_ppv_rs") for o in boot_outs], matched_cap[1]["ppv_observed_event_rate_pct"]),
            "ppv_mu_ci95": ci_from_vals([o.get("mc_ppv_mu") for o in boot_outs], matched_cap[2]["ppv_observed_event_rate_pct"]),
            "nb08_flag_minus_rs_ci95": ci_from_vals([o.get("nb08_flag_minus_rs") for o in boot_outs], nb08_f_r),
            "nb08_flag_minus_mu_ci95": ci_from_vals([o.get("nb08_flag_minus_mu") for o in boot_outs], nb08_f_m),
        }

    # Assemble primary-policy subgroup table (Appendix Table 4)
    subgroup_df = None
    if run_subgroups:
        a_pearl_main = policy_specs[1][1]
        m_cur_main = compute_misalignment(mu_ev, sg_ev, adm_te, a_te, lam_star, EPSILON)
        m_pol_main = compute_misalignment(mu_ev, sg_ev, adm_te, a_pearl_main, lam_star, EPSILON)
        v_pol_rf = mu_ev[np.arange(n_te), a_pearl_main]

        sg_rows = []
        for key, sname, grp, mask in subgroup_meta:
            n_g = int(mask.sum())
            y_g = y_te[mask]
            p_g = p_own_rf[mask]
            ev_g = int(y_g.sum())
            c = calibration_metrics(y_g, p_g) if (n_g >= 10 and ev_g > 0 and ev_g < n_g) else {
                "observed_rate": float(y_g.mean()),
                "predicted_rate": float(p_g.mean()),
                "calibration_slope": np.nan,
                "calibration_intercept": np.nan,
                "ece": float(abs(y_g.mean() - p_g.mean())),
                "brier": float(brier_score_loss(y_g, p_g)),
                "auroc": np.nan,
            }
            pt_mc = 100.0 * float(m_cur_main[mask].mean())
            pt_mp = 100.0 * float(m_pol_main[mask].mean())
            pt_av = float((p_own_rf[mask].mean() - v_pol_rf[mask].mean()) * 1000.0)
            b_vals = [o["subgroups"][key] for o in boot_outs if key in o.get("subgroups", {})]
            ci_mc = ci_from_vals([v[0] for v in b_vals], pt_mc)
            ci_mp = ci_from_vals([v[1] for v in b_vals], pt_mp)
            ci_av = ci_from_vals([v[2] for v in b_vals], pt_av)
            sg_rows.append({
                "stratum": sname,
                "group": grp,
                "n": n_g,
                "events": ev_g,
                "observed_rate_pct": round(100.0 * c["observed_rate"], 2),
                "predicted_rate_pct": round(100.0 * c["predicted_rate"], 2),
                "calibration_slope": round(c["calibration_slope"], 2) if np.isfinite(c["calibration_slope"]) else None,
                "calibration_intercept": round(c["calibration_intercept"], 2) if np.isfinite(c["calibration_intercept"]) else None,
                "ece": round(c["ece"], 3),
                "brier": round(c["brier"], 4),
                "auroc": round(c["auroc"], 3) if np.isfinite(c["auroc"]) else None,
                "misalign_baseline_pct": round(pt_mc, 2),
                "misalign_baseline_ci95_lo": ci_mc[0],
                "misalign_baseline_ci95_hi": ci_mc[1],
                "misalign_pearl_pct": round(pt_mp, 2),
                "misalign_pearl_ci95_lo": ci_mp[0],
                "misalign_pearl_ci95_hi": ci_mp[1],
                "pct_switched": round(100.0 * float((a_pearl_main[mask] != a_te[mask]).mean()), 2),
                "events_averted_per_1000": round(pt_av, 2),
                "events_averted_ci95_lo": ci_av[0],
                "events_averted_ci95_hi": ci_av[1],
            })
        subgroup_df = pd.DataFrame(sg_rows)

    # SVI deprivation gradient (if svi_overall in test)
    svi_rows = []
    if "svi_overall" in test.columns and np.isfinite(test["svi_overall"].values).sum() > 200:
        te_svi = test["svi_overall"].values
        q = pd.qcut(pd.Series(te_svi), 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        a_pearl_main = policy_specs[1][1]
        m_cur_main = compute_misalignment(mu_ev, sg_ev, adm_te, a_te, lam_star, EPSILON)
        m_pol_main = compute_misalignment(mu_ev, sg_ev, adm_te, a_pearl_main, lam_star, EPSILON)
        for lev in sorted(pd.Series(q).dropna().unique()):
            m = (q == lev).values
            # Wilson / Wald 95% CI for proportion
            p_m = float(m_cur_main[m].mean())
            se_m = np.sqrt(p_m * (1 - p_m) / m.sum())
            p_e = float(y_te[m].mean())
            se_e = np.sqrt(p_e * (1 - p_e) / m.sum())
            svi_rows.append({
                "svi_quintile": int(lev),
                "n": int(m.sum()),
                "events": int(y_te[m].sum()),
                "observed_event_rate_pct": round(100.0 * p_e, 2),
                "observed_event_rate_ci95": [round(100.0 * max(0, p_e - 1.96 * se_e), 2), round(100.0 * min(1, p_e + 1.96 * se_e), 2)],
                "misalign_baseline_pct": round(100.0 * p_m, 2),
                "misalign_baseline_ci95": [round(100.0 * max(0, p_m - 1.96 * se_m), 2), round(100.0 * min(1, p_m + 1.96 * se_m), 2)],
                "misalign_pearl_pct": round(100.0 * float(m_pol_main[m].mean()), 2),
            })
        # Spearman trend test across quintiles
        valid_svi = q.notna().values
        rho_ev, p_ev_tr = stats.spearmanr(q[valid_svi].astype(int), y_te[valid_svi])
        rho_mis, p_mis_tr = stats.spearmanr(q[valid_svi].astype(int), m_cur_main[valid_svi])
    else:
        rho_ev = p_ev_tr = rho_mis = p_mis_tr = None

    summary = {
        "label": label,
        "n_total": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "events_total": int(y_all.sum()),
        "event_rate_total_pct": round(100.0 * float(y_all.mean()), 2),
        "events_test": int(y_te.sum()),
        "event_rate_test_pct": round(100.0 * float(y_te.mean()), 2),
        "min_action_n": int(min_n),
        "admissible_actions": adm_actions,
        "n_admissible_actions": len(adm_actions),
        "n_candidate_actions": int(k_act),
        "action_counts_all": {actions_unique[j]: int(counts_all[j]) for j in range(k_act)},
        "action_counts_train": {actions_unique[j]: int(counts_tr[j]) for j in range(k_act)},
        "determinism": determinism,
        "lambda_selected": lam_star,
        "lambda_tuning": lam_tuning,
        "diagnostics": diag,
        "agreement_gbm_rf_pct": agree_rf,
        "agreement_gbm_logit_pct": agree_lg,
        "agreement_rf_logit_pct": agree_ev_lg,
        "ablation": ablation_rows,
        "switched_support": support_diags,
        "propensity_floor_sensitivity": prop_floor_rows,
        "epsilon_sensitivity": eps_rows,
        "disjoint_50_50_split": disjoint_res,
        "observed_outcome_check": obs_check,
        "matched_capacity_review": matched_cap,
        "bootstrap_cis": boot_cis,
        "svi_gradient": svi_rows,
        "svi_trend": {
            "spearman_events_rho": round(float(rho_ev), 4) if rho_ev is not None else None,
            "spearman_events_p": round(float(p_ev_tr), 4) if p_ev_tr is not None else None,
            "spearman_misalign_rho": round(float(rho_mis), 4) if rho_mis is not None else None,
            "spearman_misalign_p": round(float(p_mis_tr), 4) if p_mis_tr is not None else None,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }

    # Concise console summary
    p_row = ablation_rows[1]
    ci_p = boot_cis["pearl_nested"]
    print(f"  PRIMARY PEARL (lam*={lam_star}): mis {p_row['misalign_baseline_at_lam_pct']}% "
          f"(95% CI {ci_p['mis_cur_ci95']}) -> {p_row['misalign_policy_at_lam_pct']}% "
          f"(95% CI {ci_p['mis_pol_ci95']}), diff {p_row['misalign_diff_at_lam_pp']} pp "
          f"(95% CI {ci_p['mis_diff_ci95']}), averted {p_row['model_predicted_events_averted_per_1000_dm']}/1k "
          f"(95% CI {ci_p['averted_dm_ci95']}), switched {p_row['pct_switched']}%")
    print(f"  Lambda boot distribution: {lam_dist}")
    return summary, dca_df, subgroup_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    disable_unused_crossval()
    d = _load_raw()
    bridge = _build_id_bridge(d)
    pop = build_waymark_population(verbose=False)
    rising = pop.patients.reset_index(drop=True)

    # Attach state, race_eth_full, and ZIP-level SVI
    aux = load_eligibility_attributes(rising["member_id"])
    svi = pd.read_parquet(SVI_PATH)
    elig = pd.read_parquet(RAW_DIR / "eligibility.parquet", columns=["member_id", "zip_code"]).drop_duplicates("member_id")
    s = elig["zip_code"].astype(str).str.replace(r"\D", "", regex=True)
    elig["zip5"] = np.where(s.str.len() >= 9, s.str[:5], s.str.zfill(5))
    zmap = elig.merge(svi, on="zip5", how="left").set_index("member_id")
    rising["svi_overall"] = rising["member_id"].map(zmap["svi_overall"])
    rising["state"] = aux["state"].values
    rising["race_eth_full"] = aux["race_eth_full"].values

    hv = d["hospital_visits"]
    hv_ei = hv[hv["patient_class_code"].isin(["E", "I"])].copy()
    hv_ei["admit_date"] = pd.to_datetime(hv_ei["admit_date"], utc=True)
    hv_min, hv_max = hv_ei["admit_date"].min(), hv_ei["admit_date"].max()

    idx_dt = pd.to_datetime(rising["index_date"], utc=True)
    rising["month_num"] = (idx_dt.dt.tz_localize(None) - pd.Timestamp("2023-01-01")).dt.days / 30.4375

    # ── Block 1: Temporal window & follow-up audit ────────────────────────────
    fu_mask_complete = (idx_dt >= hv_min) & (idx_dt + pd.Timedelta(days=90) <= hv_max)
    fu_mask_pre = idx_dt < hv_min
    fu_mask_post = idx_dt > hv_max
    fu_mask_partial = (idx_dt >= hv_min) & (idx_dt <= hv_max) & (idx_dt + pd.Timedelta(days=90) > hv_max)

    comp = rising[fu_mask_complete].reset_index(drop=True)
    temporal_audit = {
        "hv_min": str(hv_min),
        "hv_max": str(hv_max),
        "assessed_n": 39317,
        "linked_n": int(len(rising)),
        "excluded_linkage_n": int(39317 - len(rising)),
        "excluded_before_hv_min_n": int(fu_mask_pre.sum()),
        "excluded_after_hv_max_n": int(fu_mask_post.sum()),
        "excluded_partial_90d_n": int(fu_mask_partial.sum()),
        "complete_90d_fu_n": int(len(comp)),
        "complete_90d_fu_events": int(comp["y_behavioral"].sum()),
        "complete_90d_fu_rate_pct": round(100.0 * float(comp["y_behavioral"].mean()), 2),
        "state_rates_complete_fu": {
            st: {"n": int(len(g)), "events": int(g["y_behavioral"].sum()),
                 "rate_pct": round(100.0 * float(g["y_behavioral"].mean()), 2)}
            for st, g in comp.groupby("state")
        },
        "state_rates_unwindowed": {
            st: {"n": int(len(g)), "events": int(g["y_behavioral"].sum()),
                 "rate_pct": round(100.0 * float(g["y_behavioral"].mean()), 2)}
            for st, g in rising.groupby("state")
        },
    }

    # Inpatient vs ED split in complete 90-day cohort
    hv_i = hv[hv["patient_class_code"] == "I"].copy()
    hv_e = hv[hv["patient_class_code"] == "E"].copy()
    y_ip_comp = _compute_90d_outcome(comp["member_id"], comp["index_date"], hv_i, 90)
    y_ed_comp = _compute_90d_outcome(comp["member_id"], comp["index_date"], hv_e, 90)
    temporal_audit["complete_90d_ip_events"] = int(y_ip_comp.sum())
    temporal_audit["complete_90d_ip_rate_pct"] = round(100.0 * float(y_ip_comp.mean()), 2)
    temporal_audit["complete_90d_ed_events"] = int(y_ed_comp.sum())
    temporal_audit["complete_90d_ed_rate_pct"] = round(100.0 * float(y_ed_comp.mean()), 2)
    temporal_audit["complete_90d_both_ip_ed"] = int(((y_ip_comp == 1) & (y_ed_comp == 1)).sum())
    temporal_audit["complete_90d_ip_only"] = int(((y_ip_comp == 1) & (y_ed_comp == 0)).sum())

    # ── Block 2: Action Validation (Clinical-flag rule vs documented records) ─
    print("\n--- Block 2: Validating clinical-flag heuristic against documented records ---")
    rising["flag_rule"] = rising.apply(pure_clinical_flag_rule, axis=1)
    comp["flag_rule"] = comp.apply(pure_clinical_flag_rule, axis=1)

    mg = d["member_goals"].copy()
    mg["goal_created_at"] = pd.to_datetime(mg["goal_created_at"], utc=True, errors="coerce")
    mg_nd = mg[(mg["category"] != "DEFAULT") & (mg["deleted"] == False)].copy()
    mg_nd["action"] = mg_nd["category"].map(GOAL_MAP)
    mg_nd = mg_nd.dropna(subset=["action"])

    # Documented share in complete_90d_fu cohort
    doc_pids = set(mg_nd["member_id"])
    comp_doc_n = int(comp["member_id"].isin(doc_pids).sum())
    temporal_audit["complete_90d_documented_goal_n"] = comp_doc_n
    temporal_audit["complete_90d_documented_goal_pct"] = round(100.0 * comp_doc_n / len(comp), 1)
    temporal_audit["complete_90d_inferred_n"] = int(len(comp) - comp_doc_n)
    temporal_audit["complete_90d_inferred_pct"] = round(100.0 * (len(comp) - comp_doc_n) / len(comp), 1)

    # Modal documented goal per rising-risk member (all-time)
    doc_modal = (
        mg_nd.groupby(["member_id", "action"]).size().reset_index(name="n")
        .sort_values(["member_id", "n"], ascending=[True, False])
        .drop_duplicates("member_id")
    )
    doc_first = (
        mg_nd.sort_values("goal_created_at").drop_duplicates("member_id")[
            ["member_id", "action", "goal_created_at"]
        ].rename(columns={"action": "first_goal_action", "goal_created_at": "first_goal_at"})
    )
    doc_any = mg_nd.groupby("member_id")["action"].apply(set).to_dict()

    val_all = rising.merge(
        doc_modal[["member_id", "action"]].rename(columns={"action": "doc_modal_action"}),
        on="member_id", how="inner"
    ).merge(doc_first, on="member_id", how="left")
    val_all["days_from_idx"] = (val_all["first_goal_at"] - pd.to_datetime(val_all["index_date"], utc=True)).dt.total_seconds() / 86400.0

    # Timed [0, +90d] goals
    mg_idx = mg_nd.merge(rising[["member_id", "index_date", "flag_rule"]], on="member_id", how="inner")
    mg_idx["days"] = (mg_idx["goal_created_at"] - pd.to_datetime(mg_idx["index_date"], utc=True)).dt.total_seconds() / 86400.0
    w90 = mg_idx[(mg_idx["days"] >= 0) & (mg_idx["days"] <= 90)]
    w90_modal = (
        w90.groupby(["member_id", "action"]).size().reset_index(name="n")
        .sort_values(["member_id", "n"], ascending=[True, False])
        .drop_duplicates("member_id")
        .merge(rising[["member_id", "flag_rule"]], on="member_id")
    )

    # Same within complete-90d cohort
    w90_comp = w90_modal[w90_modal["member_id"].isin(set(comp["member_id"]))]

    per_action_val = []
    for act, grp in val_all.groupby("doc_modal_action"):
        any_m = float(np.mean([act in doc_any.get(mid, set()) and r == act for mid, r in zip(grp["member_id"], grp["flag_rule"])]))
        per_action_val.append({
            "documented_action": act,
            "n_documented": int(len(grp)),
            "exact_agreement_pct": round(100.0 * float((grp["flag_rule"] == act).mean()), 1),
            "most_common_inferred": grp["flag_rule"].value_counts().index[0],
            "most_common_inferred_pct": round(100.0 * float(grp["flag_rule"].value_counts(normalize=True).iloc[0]), 1),
        })
    per_action_val = sorted(per_action_val, key=lambda r: -r["n_documented"])

    # Validated Cohorts Construction:
    # (A) Documented care-plan goals (rising-risk), anchored at first_goal_at, complete 90d FU inside [hv_min, hv_max]
    doc_rr = rising.merge(doc_first, on="member_id", how="inner")
    doc_rr = doc_rr[(doc_rr["first_goal_at"] >= hv_min) & (doc_rr["first_goal_at"] + pd.Timedelta(days=90) <= hv_max)].reset_index(drop=True)
    doc_rr["y_goal_90d"] = _compute_90d_outcome(doc_rr["member_id"], doc_rr["first_goal_at"], hv, 90).values
    util_g = _compute_prior_utilization(doc_rr["person_id"], doc_rr["first_goal_at"], d["outcomes_monthly"])
    for col in util_g.columns:
        doc_rr[col] = util_g[col].values
    doc_rr["month_num"] = (doc_rr["first_goal_at"].dt.tz_localize(None) - pd.Timestamp("2023-01-01")).dt.days / 30.4375

    # (A2) ALL patients (all risk tiers) with documented care-plan goals, anchored at first_goal_at
    way_to_cuid = bridge.dropna(subset=["person_id", "member_id"]).set_index("person_id")["member_id"].to_dict()
    cuid_to_way = {v: k for k, v in way_to_cuid.items()}
    doc_all = doc_first.copy()
    doc_all["person_id"] = doc_all["member_id"].map(cuid_to_way)
    doc_all = doc_all[
        doc_all["person_id"].notna() &
        (doc_all["first_goal_at"] >= hv_min) &
        (doc_all["first_goal_at"] + pd.Timedelta(days=90) <= hv_max)
    ].reset_index(drop=True)
    doc_all["y_goal_90d"] = _compute_90d_outcome(doc_all["member_id"], doc_all["first_goal_at"], hv, 90).values
    util_a = _compute_prior_utilization(doc_all["person_id"], doc_all["first_goal_at"], d["outcomes_monthly"])
    for col in util_a.columns:
        doc_all[col] = util_a[col].values
    demog_a = _build_demographics(doc_all["member_id"], d)
    for col in ["age", "female", "race_eth", "primary_language", "risk_score"]:
        doc_all[col] = demog_a[col].values
    clin_a = _build_clinical_features(doc_all["member_id"], d, way_ids=doc_all["person_id"])
    for col in clin_a.columns:
        if col not in ("member_id", "person_id"):
            doc_all[col] = clin_a[col].values
    sr_idx = d["signal_risk"].drop_duplicates("patientId").set_index("patientId")
    doc_all["risingRiskScorePercentile"] = doc_all["member_id"].map(sr_idx["risingRiskScorePercentile"])
    doc_all["risingRiskScore"] = doc_all["member_id"].map(sr_idx["risingRiskScore"])
    doc_all["adi_percentile"] = doc_all["risingRiskScorePercentile"].fillna(50)
    doc_all["adi_quintile"] = pd.qcut(doc_all["adi_percentile"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    aux_da = load_eligibility_attributes(doc_all["member_id"])
    doc_all["state"] = aux_da["state"].values
    doc_all["race_eth_full"] = aux_da["race_eth_full"].values
    doc_all["month_num"] = (doc_all["first_goal_at"].dt.tz_localize(None) - pd.Timestamp("2023-01-01")).dt.days / 30.4375

    # (B) Completed-encounter clinician discipline (rising-risk), anchored at first completed encounter
    enc = pd.read_parquet(RAW_DIR / "encounters.parquet")
    pm = d["member_patient_map"][["patient_id", "member_id"]].drop_duplicates()
    ws = pd.read_parquet(RAW_DIR / "waymarker_specialty.parquet")[["waymarker_id", "discipline"]]
    enc_yes = (
        enc[enc["encounter_occurred"] == "YES"]
        .merge(pm, on="patient_id", how="left")
        .merge(ws, left_on="created_by_waymarker_id", right_on="waymarker_id", how="left")
    )
    enc_yes["created_at"] = pd.to_datetime(enc_yes["created_at"], utc=True, errors="coerce")
    enc_yes["disc_action"] = enc_yes["discipline"].map(map_discipline)
    enc_yes = enc_yes.dropna(subset=["disc_action"])
    first_e = enc_yes.sort_values("created_at").drop_duplicates("member_id")[["member_id", "disc_action", "created_at"]]
    enc_rr = rising.merge(first_e, on="member_id", how="inner")
    enc_rr = enc_rr[(enc_rr["created_at"] >= hv_min) & (enc_rr["created_at"] + pd.Timedelta(days=90) <= hv_max)].reset_index(drop=True)
    enc_rr["y_enc_90d"] = _compute_90d_outcome(enc_rr["member_id"], enc_rr["created_at"], hv, 90).values
    util_e = _compute_prior_utilization(enc_rr["person_id"], enc_rr["created_at"], d["outcomes_monthly"])
    for col in util_e.columns:
        enc_rr[col] = util_e[col].values
    enc_rr["month_num"] = (enc_rr["created_at"].dt.tz_localize(None) - pd.Timestamp("2023-01-01")).dt.days / 30.4375

    # Cross-tab of clinical-flag rule vs encounter discipline
    enc_val_ct = pd.crosstab(enc_rr["flag_rule"], enc_rr["disc_action"]).to_dict()
    mh_flagged = enc_rr[enc_rr["flag_rule"] == "mental_health"]
    med_flagged = enc_rr[enc_rr["flag_rule"] == "medication_adherence"]

    action_validation = {
        "all_time_documented_n": int(len(val_all)),
        "all_time_exact_agreement_pct": round(100.0 * float((val_all["flag_rule"] == val_all["doc_modal_action"]).mean()), 2),
        "all_time_cohen_kappa": round(float(cohen_kappa_score(val_all["doc_modal_action"], val_all["flag_rule"])), 3),
        "all_time_any_goal_match_pct": round(100.0 * float(np.mean([r in doc_any.get(m, set()) for m, r in zip(val_all["member_id"], val_all["flag_rule"])])), 2),
        "timed_0_90d_n": int(len(w90_modal)),
        "timed_0_90d_exact_agreement_pct": round(100.0 * float((w90_modal["flag_rule"] == w90_modal["action"]).mean()), 2),
        "timed_0_90d_cohen_kappa": round(float(cohen_kappa_score(w90_modal["action"], w90_modal["flag_rule"])), 3),
        "timed_0_90d_complete_fu_n": int(len(w90_comp)),
        "timed_0_90d_complete_fu_exact_agreement_pct": round(100.0 * float((w90_comp["flag_rule"] == w90_comp["action"]).mean()), 2) if len(w90_comp) else None,
        "goal_timing_before_idx_n": int((val_all["days_from_idx"] < 0).sum()),
        "goal_timing_before_idx_pct": round(100.0 * float((val_all["days_from_idx"] < 0).mean()), 1),
        "goal_timing_0_90d_n": int(((val_all["days_from_idx"] >= 0) & (val_all["days_from_idx"] <= 90)).sum()),
        "goal_timing_gt_90d_n": int((val_all["days_from_idx"] > 90).sum()),
        "goal_timing_gt_90d_pct": round(100.0 * float((val_all["days_from_idx"] > 90).mean()), 1),
        "goal_timing_median_days": round(float(val_all["days_from_idx"].median()), 1),
        "per_action_agreement": per_action_val,
        "flag_vs_encounter_discipline_n": int(len(enc_rr)),
        "flag_mh_seen_by_bh_n": int((mh_flagged["disc_action"] == "behavioral_health").sum()),
        "flag_mh_total_n": int(len(mh_flagged)),
        "flag_mh_seen_by_bh_pct": round(100.0 * float((mh_flagged["disc_action"] == "behavioral_health").mean()), 1),
        "flag_med_seen_by_pharm_n": int((med_flagged["disc_action"] == "pharmacy").sum()),
        "flag_med_total_n": int(len(med_flagged)),
        "flag_med_seen_by_pharm_pct": round(100.0 * float((med_flagged["disc_action"] == "pharmacy").mean()), 1),
        "flag_vs_discipline_crosstab": enc_val_ct,
    }
    pd.DataFrame(per_action_val).to_csv(OUT_DIR / "v6_action_validation_by_action.csv", index=False)

    # ── Block 3: Unfiltered Within-Patient Onboarding Analysis & T7 ──────────
    print("\n--- Block 3: Unfiltered within-patient 2x2 tables, RTM benchmark, & T7 ---")
    ms = d["member_status"].copy()
    ms["effective_at"] = pd.to_datetime(ms["effective_at"], utc=True)
    onb = ms[ms["to_status"] == "ONBOARDED"].sort_values("effective_at").drop_duplicates("member_id", keep="first").reset_index(drop=True)
    on_s = onb["effective_at"]; on_e = on_s + pd.Timedelta(days=90)
    off_e = on_s - pd.Timedelta(days=1); off_s = off_e - pd.Timedelta(days=90)
    onb["y_on"] = _compute_90d_outcome(onb["member_id"], on_s, hv, 90).values
    onb["y_off"] = _compute_90d_outcome(onb["member_id"], off_s, hv, 90).values
    onb["complete_win"] = (off_s >= hv_min) & (on_e <= hv_max)
    onb["in_rising"] = onb["member_id"].isin(set(rising["member_id"]))

    def paired_stats(sub, name):
        y0, y1 = sub["y_off"].values, sub["y_on"].values
        diff = y1 - y0
        rd = float(diff.mean())
        se = float(diff.std(ddof=1) / np.sqrt(len(sub)))
        n00 = int(((y0 == 0) & (y1 == 0)).sum())
        n10 = int(((y0 == 1) & (y1 == 0)).sum())
        n01 = int(((y0 == 0) & (y1 == 1)).sum())
        n11 = int(((y0 == 1) & (y1 == 1)).sum())
        chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01) if (n10 + n01) > 0 else 0.0
        mp = float(stats.chi2.sf(chi2, 1))
        return {
            "cohort": name,
            "n": int(len(sub)),
            "cell_pre0_post0": n00,
            "cell_pre1_post0": n10,
            "cell_pre0_post1": n01,
            "cell_pre1_post1": n11,
            "discarded_y_on_eq_1_n": int(n01 + n11),
            "discarded_y_on_eq_1_pct": round(100.0 * (n01 + n11) / len(sub), 1),
            "pre_rate_pct": round(100.0 * float(y0.mean()), 2),
            "post_rate_pct": round(100.0 * float(y1.mean()), 2),
            "paired_rd_pp": round(100.0 * rd, 2),
            "paired_rd_ci95_pp": [round(100.0 * (rd - 1.96 * se), 2), round(100.0 * (rd + 1.96 * se), 2)],
            "mcnemar_chi2": round(float(chi2), 2),
            "mcnemar_p": mp,
        }

    # Type 3 churn unfiltered
    elig_raw = d["eligibility"].copy()
    elig_raw["enrollment_start_date"] = pd.to_datetime(elig_raw["enrollment_start_date"], utc=True)
    elig_raw["enrollment_end_date"] = pd.to_datetime(elig_raw["enrollment_end_date"], utc=True)
    elig_raw = elig_raw[
        (elig_raw["enrollment_start_date"] >= pd.Timestamp("2020-01-01", tz="UTC")) &
        (elig_raw["enrollment_end_date"] >= elig_raw["enrollment_start_date"]) &
        elig_raw["person_id"].notna()
    ].sort_values(["person_id", "enrollment_start_date"])
    elig_raw["next_start"] = elig_raw.groupby("person_id")["enrollment_start_date"].shift(-1)
    elig_raw["gap_days"] = (elig_raw["next_start"] - elig_raw["enrollment_end_date"]).dt.days
    gaps = elig_raw[(elig_raw["gap_days"] >= 60) & (elig_raw["gap_days"] <= 365)].copy()
    gaps["on_end"] = gaps["enrollment_end_date"]
    gaps["on_start"] = gaps["on_end"] - pd.Timedelta(days=90)
    gaps["off_start"] = gaps["enrollment_end_date"]
    gaps["off_end"] = gaps["off_start"] + pd.Timedelta(days=90)
    gaps = gaps[gaps["on_start"] >= pd.Timestamp("2023-01-01", tz="UTC")].copy()
    gaps["member_id"] = gaps["person_id"].map(way_to_cuid)
    enrolled_pids = set(ms[ms["to_status"].isin(["ONBOARDED", "ACTIVATED"])]["member_id"].dropna())
    gaps = gaps[gaps["member_id"].isin(enrolled_pids)].reset_index(drop=True)
    gaps["y_on"] = _compute_90d_outcome(gaps["member_id"], gaps["on_start"], hv, 90).values
    gaps["y_off"] = _compute_90d_outcome(gaps["member_id"], gaps["off_start"], hv, 90).values
    gaps["complete_win"] = (gaps["on_start"] >= hv_min) & (gaps["off_end"] <= hv_max)

    wpad_tables = [
        paired_stats(onb, "Staggered onboarding — all"),
        paired_stats(onb[onb["complete_win"]], "Staggered onboarding — complete 90d pre/post window"),
        paired_stats(onb[onb["in_rising"]], "Staggered onboarding — rising-risk subset (all)"),
        paired_stats(onb[onb["complete_win"] & onb["in_rising"]], "Staggered onboarding — rising-risk complete 90d window"),
        paired_stats(gaps, "Medicaid eligibility churn (ITT) — all"),
        paired_stats(gaps[gaps["complete_win"]], "Medicaid eligibility churn (ITT) — complete 90d window"),
    ]

    # Regression-to-the-mean benchmark: non-onboarded rising-risk patients assigned
    # random calendar dates drawn from the onboarded date distribution!
    rng_rtm = np.random.default_rng(SEED)
    onb_rr_comp = onb[onb["complete_win"] & onb["in_rising"]].reset_index(drop=True)
    non_onb_rr = comp[~comp["member_id"].isin(set(onb["member_id"]))].reset_index(drop=True)
    pseudo_dates = pd.Series(rng_rtm.choice(onb_rr_comp["effective_at"].values, size=len(non_onb_rr), replace=True))
    pseudo_dates = pd.to_datetime(pseudo_dates, utc=True)
    y_pre_ctrl = _compute_90d_outcome(non_onb_rr["member_id"], pseudo_dates - pd.Timedelta(days=91), hv, 90).values
    y_post_ctrl = _compute_90d_outcome(non_onb_rr["member_id"], pseudo_dates, hv, 90).values
    # Conditional on having a pre-window event (y_pre == 1), compare drop in onboarded vs non-onboarded!
    rtm_benchmark = {
        "onboarded_rr_complete_n": int(len(onb_rr_comp)),
        "onboarded_pre1_n": int((onb_rr_comp["y_off"] == 1).sum()),
        "onboarded_post_rate_given_pre1_pct": round(100.0 * float(onb_rr_comp.loc[onb_rr_comp["y_off"] == 1, "y_on"].mean()), 2),
        "onboarded_post_rate_given_pre0_pct": round(100.0 * float(onb_rr_comp.loc[onb_rr_comp["y_off"] == 0, "y_on"].mean()), 2),
        "control_rr_n": int(len(non_onb_rr)),
        "control_pre_rate_pct": round(100.0 * float(y_pre_ctrl.mean()), 2),
        "control_post_rate_pct": round(100.0 * float(y_post_ctrl.mean()), 2),
        "control_pre1_n": int((y_pre_ctrl == 1).sum()),
        "control_post_rate_given_pre1_pct": round(100.0 * float(y_post_ctrl[y_pre_ctrl == 1].mean()), 2),
        "control_post_rate_given_pre0_pct": round(100.0 * float(y_post_ctrl[y_pre_ctrl == 0].mean()), 2),
    }
    print(f"  RTM benchmark: given y_pre=1, post rate is {rtm_benchmark['onboarded_post_rate_given_pre1_pct']}% "
          f"(onboarded, n={rtm_benchmark['onboarded_pre1_n']}) vs {rtm_benchmark['control_post_rate_given_pre1_pct']}% "
          f"(non-onboarded rising-risk, n={rtm_benchmark['control_pre1_n']})")

    # Unfiltered T7 + original T7a/T7b
    aux_onb = load_eligibility_attributes(onb["member_id"])
    onb["state"] = aux_onb["state"].values
    onb["onboard_month"] = onb["effective_at"].dt.tz_localize(None).dt.to_period("M")
    staffdf = build_monthly_chw_headcount()
    onb_t7 = onb[onb["state"].isin(["VA", "WA", "OH"])].copy()
    onb_t7 = onb_t7.merge(staffdf, on=["state", "onboard_month"], how="left")
    for lag in (1, 2, 3):
        ldf = staffdf.copy()
        ldf["onboard_month"] = ldf["onboard_month"] + lag
        ldf = ldf.rename(columns={"chw_active": f"chw_active_lag{lag}"})
        onb_t7 = onb_t7.merge(ldf, on=["state", "onboard_month"], how="left")
    staff_cols = ["chw_active", "chw_active_lag1", "chw_active_lag2", "chw_active_lag3"]
    onb_t7["month_idx"] = (onb_t7["onboard_month"] - pd.Period("2023-01", freq="M")).apply(lambda x: x.n)
    full_t7_cols = [
        "age", "female", "charlson_score", "prior_ed_visits_6mo", "prior_hosp_6mo",
        "pharmacy_fills_90d", "missed_pharmacy_fills", "has_diabetes", "has_chf",
        "has_copd", "has_hypertension", "has_ckd", "has_mh",
    ]
    narrow_t7_cols = ["charlson_score", "prior_hosp_6mo", "prior_ed_visits_6mo"]
    onb_t7 = onb_t7.merge(rising[["member_id"] + full_t7_cols], on="member_id", how="inner")
    onb_t7 = onb_t7.dropna(subset=staff_cols + full_t7_cols).reset_index(drop=True)

    def run_lr(df_sub, target_col, c0, c1, desc):
        y_t = df_sub[target_col].values.astype(int)
        Xn = StandardScaler().fit_transform(df_sub[c0].values.astype(float))
        Xf = StandardScaler().fit_transform(df_sub[c1].values.astype(float))
        m0 = LogisticRegression(C=1.0, max_iter=3000).fit(Xn, y_t)
        m1 = LogisticRegression(C=1.0, max_iter=3000).fit(Xf, y_t)
        lr = float(2 * (-log_loss(y_t, m1.predict_proba(Xf)[:, 1], labels=[0, 1]) * len(y_t)
                        + log_loss(y_t, m0.predict_proba(Xn)[:, 1], labels=[0, 1]) * len(y_t)))
        dfree = len(c1) - len(c0)
        pval = float(stats.chi2.sf(lr, dfree))
        return {"spec": desc, "n": int(len(df_sub)), "lr_stat": round(lr, 2), "df": dfree, "p": round(pval, 4)}

    t7_comp = onb_t7[onb_t7["complete_win"]].reset_index(drop=True)
    t7_ret = onb_t7[onb_t7["y_on"] == 0].reset_index(drop=True)
    t7_results = [
        run_lr(t7_comp, "y_off", narrow_t7_cols, narrow_t7_cols + staff_cols, "Unfiltered complete-window (y_off) — narrow covariates"),
        run_lr(t7_comp, "y_off", full_t7_cols + ["month_idx"], full_t7_cols + ["month_idx"] + staff_cols, "Unfiltered complete-window (y_off) — full covariates + calendar month"),
        run_lr(t7_comp[t7_comp["state"] == "VA"], "y_off", full_t7_cols + ["month_idx"], full_t7_cols + ["month_idx"] + staff_cols, "Unfiltered complete-window (y_off) — Virginia only"),
        run_lr(t7_comp[t7_comp["state"] == "WA"], "y_off", full_t7_cols + ["month_idx"], full_t7_cols + ["month_idx"] + staff_cols, "Unfiltered complete-window (y_off) — Washington only"),
        run_lr(t7_comp, "y_on", narrow_t7_cols, narrow_t7_cols + staff_cols, "Unfiltered complete-window (y_on) — narrow covariates"),
        run_lr(t7_comp, "y_on", full_t7_cols + ["month_idx"], full_t7_cols + ["month_idx"] + staff_cols, "Unfiltered complete-window (y_on) — full covariates + calendar month"),
        run_lr(t7_ret, "y_off", narrow_t7_cols, narrow_t7_cols + staff_cols, "Retained y_on=0 pairs (pair type) — narrow covariates"),
        run_lr(t7_ret, "y_off", full_t7_cols + ["month_idx"], full_t7_cols + ["month_idx"] + staff_cols, "Retained y_on=0 pairs (pair type) — full covariates + calendar month"),
        run_lr(t7_ret[t7_ret["state"] == "VA"], "y_off", full_t7_cols + ["month_idx"], full_t7_cols + ["month_idx"] + staff_cols, "Retained y_on=0 pairs (pair type) — Virginia only"),
        run_lr(t7_ret[t7_ret["state"] == "WA"], "y_off", full_t7_cols + ["month_idx"], full_t7_cols + ["month_idx"] + staff_cols, "Retained y_on=0 pairs (pair type) — Washington only"),
    ]
    pd.DataFrame(wpad_tables).to_csv(OUT_DIR / "v6_wpad_unfiltered_2x2.csv", index=False)
    pd.DataFrame(t7_results).to_csv(OUT_DIR / "v6_t7_staffing_complete.csv", index=False)

    # ── Block 4: Baseline Characteristics Table 1 across the 3 main cohorts ──
    def cohort_profile(sub, ycol, actcol, name):
        svi_linked = int(sub["svi_overall"].notna().sum()) if "svi_overall" in sub.columns else 0
        return {
            "cohort": name,
            "n": int(len(sub)),
            "age_mean": round(float(sub["age"].mean()), 1),
            "age_sd": round(float(sub["age"].std()), 1),
            "female_n": int((sub["female"] == 1).sum()),
            "female_pct": round(100.0 * float((sub["female"] == 1).mean()), 1),
            "charlson_mean": round(float(sub["charlson_score"].mean()), 2),
            "charlson_sd": round(float(sub["charlson_score"].std()), 2),
            "mh_n": int((sub["has_mh"] == 1).sum()),
            "mh_pct": round(100.0 * float((sub["has_mh"] == 1).mean()), 1),
            "htn_n": int((sub["has_hypertension"] == 1).sum()),
            "htn_pct": round(100.0 * float((sub["has_hypertension"] == 1).mean()), 1),
            "copd_n": int((sub["has_copd"] == 1).sum()),
            "copd_pct": round(100.0 * float((sub["has_copd"] == 1).mean()), 1),
            "dm_n": int((sub["has_diabetes"] == 1).sum()),
            "dm_pct": round(100.0 * float((sub["has_diabetes"] == 1).mean()), 1),
            "events_90d_n": int(sub[ycol].sum()),
            "events_90d_pct": round(100.0 * float(sub[ycol].mean()), 2),
            "svi_linked_n": svi_linked,
            "svi_linked_pct": round(100.0 * svi_linked / len(sub), 1),
            "state_counts": sub["state"].value_counts().to_dict(),
            "race_counts": sub["race_eth_full"].value_counts().to_dict(),
            "action_counts": sub[actcol].value_counts().to_dict(),
        }

    table1_profiles = {
        "complete_90d_fu": cohort_profile(comp, "y_behavioral", "behavioral_intervention", "Complete 90-day follow-up rising-risk cohort (hybrid clinical-flag baseline)"),
        "encounter_discipline": cohort_profile(enc_rr, "y_enc_90d", "disc_action", "Validated completed-encounter clinician discipline cohort (rising-risk)"),
        "documented_goals_rr": cohort_profile(doc_rr, "y_goal_90d", "first_goal_action", "Validated documented care-plan goal cohort (rising-risk)"),
        "documented_goals_all": cohort_profile(doc_all, "y_goal_90d", "first_goal_action", "Validated documented care-plan goal cohort (all risk tiers)"),
        "unwindowed_full": cohort_profile(rising, "y_behavioral", "behavioral_intervention", "Unwindowed linked rising-risk cohort (sensitivity)"),
    }

    # ── Block 5: Run Full Leak-Free Pipeline + Bootstrap on ALL Cohorts ──────
    # 5A. Primary complete-follow-up rising-risk cohort (N = 18,936, hybrid clinical-flag baseline)
    res_primary, dca_primary, sg_primary = analyze_cohort(
        comp, "behavioral_intervention", "y_behavioral", min_n=100,
        label="Primary complete-90d-FU rising-risk cohort (hybrid clinical-flag baseline)",
        include_site_time=True, run_subgroups=True, run_dca=True,
        run_prop_floor_sens=True, run_disjoint_split=True, b_boot=B_OUTER_BOOT,
    )
    dca_primary.to_csv(OUT_DIR / "v6_primary_decision_curve.csv", index=False)
    sg_primary.to_csv(OUT_DIR / "v6_primary_subgroups_table4.csv", index=False)
    pd.DataFrame(res_primary["ablation"]).to_csv(OUT_DIR / "v6_primary_ablation.csv", index=False)
    json.dump(res_primary, open(OUT_DIR / "ckpt_primary.json", "w"), indent=2, default=str)

    # 5B. Validated Cohort 1: Completed-encounter clinician discipline (N = 5,929 rising-risk)
    res_enc, dca_enc, _ = analyze_cohort(
        enc_rr, "disc_action", "y_enc_90d", min_n=100,
        label="Validated completed-encounter clinician discipline cohort (rising-risk)",
        include_site_time=True, run_subgroups=False, run_dca=True,
        run_prop_floor_sens=True, run_disjoint_split=True, b_boot=B_OUTER_BOOT,
    )
    if dca_enc is not None:
        dca_enc.to_csv(OUT_DIR / "v6_enc_decision_curve.csv", index=False)
    pd.DataFrame(res_enc["ablation"]).to_csv(OUT_DIR / "v6_enc_ablation.csv", index=False)
    json.dump(res_enc, open(OUT_DIR / "ckpt_enc.json", "w"), indent=2, default=str)

    # 5C. Validated Cohort 2: Documented care-plan goals (rising-risk, N = 2,576, min_n=50)
    res_goal_rr, dca_goal_rr, _ = analyze_cohort(
        doc_rr, "first_goal_action", "y_goal_90d", min_n=50,
        label="Validated documented care-plan goal cohort (rising-risk, min_n=50)",
        include_site_time=True, run_subgroups=False, run_dca=True,
        run_prop_floor_sens=True, run_disjoint_split=True, b_boot=B_OUTER_BOOT,
    )
    if dca_goal_rr is not None:
        dca_goal_rr.to_csv(OUT_DIR / "v6_goal_rr_decision_curve.csv", index=False)
    pd.DataFrame(res_goal_rr["ablation"]).to_csv(OUT_DIR / "v6_goal_rr_ablation.csv", index=False)
    json.dump(res_goal_rr, open(OUT_DIR / "ckpt_goal_rr.json", "w"), indent=2, default=str)

    # 5D. Validated Cohort 2b: Documented care-plan goals (ALL risk tiers, N = 7,818, min_n=100)
    res_goal_all, _, _ = analyze_cohort(
        doc_all, "first_goal_action", "y_goal_90d", min_n=100,
        label="Validated documented care-plan goal cohort (all risk tiers, min_n=100)",
        include_site_time=True, run_subgroups=False, run_dca=False,
        run_prop_floor_sens=False, run_disjoint_split=False, b_boot=100,
    )
    pd.DataFrame(res_goal_all["ablation"]).to_csv(OUT_DIR / "v6_goal_all_ablation.csv", index=False)
    json.dump(res_goal_all, open(OUT_DIR / "ckpt_goal_all.json", "w"), indent=2, default=str)

    # 5E. Unwindowed full rising-risk cohort (N = 34,971, sensitivity / reconciliation)
    res_unwin, _, _ = analyze_cohort(
        rising, "behavioral_intervention", "y_behavioral", min_n=100,
        label="Unwindowed full rising-risk cohort (N=34,971 sensitivity, leak-free)",
        include_site_time=True, run_subgroups=False, run_dca=False,
        run_prop_floor_sens=False, run_disjoint_split=False, b_boot=100,
    )
    pd.DataFrame(res_unwin["ablation"]).to_csv(OUT_DIR / "v6_unwindowed_ablation.csv", index=False)
    json.dump(res_unwin, open(OUT_DIR / "ckpt_unwin.json", "w"), indent=2, default=str)

    # ── Save master JSON bundle ──────────────────────────────────────────────
    master = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "m_ensemble_main": M_ENSEMBLE_MAIN,
        "m_ensemble_boot": M_ENSEMBLE_BOOT,
        "b_outer_boot": B_OUTER_BOOT,
        "temporal_audit": temporal_audit,
        "action_validation": action_validation,
        "wpad_unfiltered_tables": wpad_tables,
        "rtm_benchmark": rtm_benchmark,
        "t7_staffing_results": t7_results,
        "table1_profiles": table1_profiles,
        "cohorts": {
            "primary_complete_90d_fu": res_primary,
            "validated_encounter_discipline": res_enc,
            "validated_documented_goals_rr": res_goal_rr,
            "validated_documented_goals_all": res_goal_all,
            "unwindowed_full_34971": res_unwin,
        },
        "total_runtime_min": round((time.time() - t_start) / 60.0, 2),
    }
    out_json = OUT_DIR / "v6_master_results.json"
    with open(out_json, "w") as f:
        json.dump(master, f, indent=2, default=str)
    print(f"\nALL v6 ANALYSES COMPLETE in {master['total_runtime_min']:.2f} min.")
    print(f"Saved canonical bundle to {out_json}")


if __name__ == "__main__":
    main()
