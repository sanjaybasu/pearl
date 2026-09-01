"""
PEARL Revision Analyses (Scientific Reports, submission 92ef2ac1)

Adds the analyses requested by Reviewer 1 and Reviewer 2 that the primary
pipeline does not produce. Run AFTER scripts/run_pipeline.py --waymark.

Phases:
  R0  Data provenance audit (state, race/ethnicity, deprivation, goal documentation)
  R1  Split-sample independent-evaluator IMI (Reviewer 1, comment 2)
  R2  Cross-fitted evaluator IMI (Reviewer 1, comment 2)
  R3  Clinical significance of IMI (Reviewer 1, comment 3)
  R4  Group-stratified fairness metrics (Reviewer 1, comment 8)
  R5  Social-needs documentation sensitivity (Reviewer 1, comment 6)
  R6  State stratification and leave-one-state-out (Reviewer 1 comment 4; Reviewer 2)

Usage:
  python scripts/revision_analyses.py --waymark
"""
import sys
import os
import json
import time
import argparse
import warnings
from pathlib import Path

_PEARL_ROOT = Path(__file__).resolve().parents[1]
if str(_PEARL_ROOT) not in sys.path:
    sys.path.insert(0, str(_PEARL_ROOT))
if str(_PEARL_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PEARL_ROOT / "scripts"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats

warnings.filterwarnings("ignore")

_REPO_ROOT = _PEARL_ROOT.parents[1]
OUTPUT_BASE = Path(os.environ.get(
    "PEARL_OUTPUT_BASE", str(_REPO_ROOT / "notebooks" / "pearl" / "outputs")))
RESULTS_DIR = OUTPUT_BASE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = "/Users/sanjaybasu/waymark-local/data/real_inputs"

INTERVENTIONS = [
    "care_access", "clinical_other", "diabetes", "financial_benefits", "food_security",
    "heart_failure", "housing", "hypertension", "maternal", "medication_adherence",
    "mental_health", "pulmonary", "substance_use", "transport_utilities",
]
INTV_ALPHA = sorted(INTERVENTIONS)
SOCIAL_NEEDS = ["food_security", "housing", "transport_utilities", "financial_benefits"]

DEMOGRAPHIC_COVS = ["age", "female", "race_eth", "primary_language",
                    "adi_percentile", "adi_quintile"]
CLINICAL_COVS = ["charlson_score", "prior_ed_visits_6mo", "prior_hosp_6mo", "n_chronic",
                 "pharmacy_fills_90d", "missed_pharmacy_fills", "has_diabetes", "has_chf",
                 "has_copd", "has_hypertension", "has_ckd", "has_mh"]

EPSILON = 0.02
SEED = 42


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def feature_matrix(patients: pd.DataFrame) -> np.ndarray:
    """Numeric covariate matrix; identical column set to IMIEstimator._get_feature_matrix."""
    cols = [c for c in DEMOGRAPHIC_COVS if c in patients.columns]
    cols += [c for c in CLINICAL_COVS if c in patients.columns]
    X = patients[cols].copy()
    for col in ["race_eth", "primary_language"]:
        if col in X.columns:
            X[col] = pd.Categorical(X[col]).codes
    return X.fillna(0).astype(float).values


def imi_from_mu(mu: np.ndarray, actions: np.ndarray, le: LabelEncoder,
                epsilon: float = EPSILON) -> np.ndarray:
    """Per-patient misalignment indicator under an outcome matrix mu (n x 14)."""
    a_enc = le.transform(actions)
    own = mu[np.arange(len(mu)), a_enc]
    best_other = mu.copy()
    best_other[np.arange(len(mu)), a_enc] = np.inf
    return (best_other.min(axis=1) < own - epsilon).astype(float)


def gain_from_mu(mu: np.ndarray, actions: np.ndarray, le: LabelEncoder) -> np.ndarray:
    """Modeled absolute risk reduction available by switching to the best alternative."""
    a_enc = le.transform(actions)
    own = mu[np.arange(len(mu)), a_enc]
    best_other = mu.copy()
    best_other[np.arange(len(mu)), a_enc] = np.inf
    return np.clip(own - best_other.min(axis=1), 0.0, None)


class IndependentEvaluator:
    """
    Outcome model used ONLY to score policies, never to train them.

    Differs from the primary IMIEstimator on two axes at once, so that agreement
    is not attributable to a shared model:
      1. fitted on a patient-disjoint sample (or cross-fitting fold);
      2. a different learner family (random forest, or penalized logistic
         regression with arm x covariate interactions) rather than the primary
         gradient-boosted S-learner.
    """

    def __init__(self, learner: str = "rf", seed: int = SEED):
        self.learner = learner
        self.seed = seed
        self._le = LabelEncoder().fit(INTV_ALPHA)

    def fit(self, patients: pd.DataFrame,
            outcome_col: str = "y_behavioral",
            intervention_col: str = "behavioral_intervention") -> "IndependentEvaluator":
        X = feature_matrix(patients)
        A = patients[intervention_col].values
        Y = patients[outcome_col].values.astype(float)
        n, k = len(patients), len(INTV_ALPHA)
        a_enc = self._le.transform(A)

        # IPW weights from an independently fitted propensity model
        prop = LogisticRegression(C=0.1, max_iter=1000).fit(X, a_enc)
        p_recv = prop.predict_proba(X)[np.arange(n), a_enc]
        w = 1.0 / np.clip(p_recv, 0.05, 10.0)
        w = w * n / w.sum()
        self._prop = prop

        one_hot = np.zeros((n, k))
        one_hot[np.arange(n), a_enc] = 1.0

        if self.learner == "rf":
            X_aug = np.hstack([X, one_hot])
            self._model = RandomForestRegressor(
                n_estimators=400, max_depth=8, min_samples_leaf=20,
                max_features="sqrt", random_state=self.seed, n_jobs=-1,
            ).fit(X_aug, Y, sample_weight=w)
            self._interact = False
        else:  # penalized logistic S-learner with explicit arm x covariate interactions
            X_aug = self._interaction_design(X, one_hot)
            self._model = LogisticRegression(
                C=0.05, max_iter=2000, solver="lbfgs",
            ).fit(X_aug, (Y > 0.5).astype(int), sample_weight=w)
            self._interact = True
        return self

    @staticmethod
    def _interaction_design(X: np.ndarray, one_hot: np.ndarray) -> np.ndarray:
        # main effects + arm indicators + arm x covariate products
        inter = np.einsum("ij,ik->ijk", one_hot, X).reshape(len(X), -1)
        return np.hstack([X, one_hot, inter])

    def predict_mu(self, patients: pd.DataFrame) -> np.ndarray:
        X = feature_matrix(patients)
        n, k = len(X), len(INTV_ALPHA)
        mu = np.zeros((n, k))
        for j in range(k):
            one_hot = np.zeros((n, k))
            one_hot[:, j] = 1.0
            if self._interact:
                Xa = self._interaction_design(X, one_hot)
                mu[:, j] = self._model.predict_proba(Xa)[:, 1]
            else:
                Xa = np.hstack([X, one_hot])
                mu[:, j] = np.clip(self._model.predict(Xa), 0.0, 1.0)
        return mu


def disable_unused_crossval():
    """
    models/imi_estimator.py:107 and evaluation/drope_evaluator.py:105 each compute
    a 5-fold cross-validated propensity matrix, assign it to self._prop_proba_cv,
    and never read it again. On BLAS builds where the multinomial fit is slow this
    dead computation dominates runtime. Substituting a single in-sample fit leaves
    every reported quantity unchanged because no code path consumes the result.
    Call before constructing any estimator.
    """
    import models.imi_estimator as _imi
    import evaluation.drope_evaluator as _dro

    def _single_fit(estimator, X, y, cv=None, method="predict_proba", **kwargs):
        return estimator.fit(X, y).predict_proba(X)

    _imi.cross_val_predict = _single_fit
    _dro.cross_val_predict = _single_fit


def light_phase_0(rising_fit: pd.DataFrame, rising_test: pd.DataFrame) -> dict:
    """
    Phase 0 without the bootstrap CI, falsification suite, or Camden reanalysis.
    Produces exactly the two artefacts phase 1 consumes: the fitted primary
    estimator and its S-learner predictions for the fitting and test sets.
    """
    from models.imi_estimator import IMIEstimator
    est = IMIEstimator(outcome_col="y_behavioral",
                       intervention_col="behavioral_intervention",
                       threshold=EPSILON, n_bootstrap=0, seed=SEED)
    est.fit(rising_fit)
    mu_fit = est._predict_outcomes(est._get_feature_matrix(rising_fit))
    mu_test = est._predict_outcomes(est._get_feature_matrix(rising_test))
    return {"estimator": est, "mu_hat_train": mu_fit,
            "imi_result": {"mu_hat": mu_test}}


def build_policy_functions(pop, phase0, phase1, rising_test):
    """Reconstruct the thirteen policy functions from a trained pipeline state."""
    suite = phase1["comparator_suite"]
    pearl = phase1["pearl"]
    moe = phase1["moe"]
    moe_pearl = phase1["moe_pearl"]
    mu_hat_test = phase0["imi_result"]["mu_hat"]

    test = rising_test.copy()
    for i, intv in enumerate(INTV_ALPHA):
        test[f"_mu_hat_{intv}"] = mu_hat_test[:, i]

    def oracle_policy(pts):
        mu_cols = [f"_mu_hat_{i}" for i in INTV_ALPHA]
        m = pts[mu_cols].values
        return np.array([INTV_ALPHA[i] for i in m.argmin(axis=1)])

    policies = {
        "LACE Index (C1)": lambda p: suite.lace.route_intervention(p),
        "HOSPITAL Score (C2)": lambda p: suite.hospital.route_intervention(p),
        "XGBoost (C3)": lambda p: suite.xgb.route_intervention(p),
        "BehavioralCloning SFT (C4)": lambda p: suite.bc_sft.predict_intervention(p),
        "Observational DPO (C5)": lambda p: suite.obs_dpo.predict_intervention(p),
        "CausalForest (C6)": lambda p: suite.causal_forest.recommend_intervention(p),
        "DecisionTransformer (C7)": lambda p: suite.dt.predict_intervention(p),
        "CQL (C8)": lambda p: suite.cql.recommend_intervention(p),
        "Behavioral Policy": lambda p: p["behavioral_intervention"].values,
        "PEARL (base)": lambda p: pearl.predict_intervention(p)[0],
        "PEARL (MoE Router)": lambda p: moe.predict(p)[0],
        "PEARL (MoE Full)": lambda p: moe_pearl.predict(p)[0],
        "Oracle (mu-hat optimal)": oracle_policy,
    }
    return policies, test


# ═════════════════════════════════════════════════════════════════════════════
# R0. Data provenance audit
# ═════════════════════════════════════════════════════════════════════════════

# Managed-care plan code -> state. The eligibility extract covers only the two
# states with a claims feed; state for every member is recovered from the payer
# code carried on the monthly utilization file.
PAYER_STATE = {
    "ABHVA": "VA", "SHPVA": "VA",
    "UHCWA": "WA", "PROVWA": "WA", "CHPW": "WA",
    "UHCOH": "OH",
}

_PAYER_STATE_CACHE = None


def _payer_state_map() -> pd.DataFrame:
    """Modal payer per member -> state."""
    global _PAYER_STATE_CACHE
    if _PAYER_STATE_CACHE is None:
        om = pd.read_parquet(f"{RAW_DIR}/outcomes_monthly.parquet",
                             columns=["member_id", "payer"])
        om = om.dropna(subset=["member_id", "payer"])
        modal = (om.groupby(["member_id", "payer"]).size().reset_index(name="n")
                   .sort_values("n", ascending=False)
                   .drop_duplicates("member_id"))
        modal["state_payer"] = modal["payer"].map(PAYER_STATE).fillna("Unlinked")
        _PAYER_STATE_CACHE = modal[["member_id", "payer", "state_payer"]]
    return _PAYER_STATE_CACHE


def load_eligibility_attributes(member_ids: pd.Series) -> pd.DataFrame:
    """State (from payer), ZIP, and race/ethnicity from the eligibility file."""
    e = pd.read_parquet(f"{RAW_DIR}/eligibility.parquet")
    e = e.drop_duplicates("member_id")
    race_map = {
        "white": "White non-Hispanic",
        "black or african american": "Black non-Hispanic",
        "hispanic": "Hispanic",
        "asian": "Asian",
        "native hawaiian or other pacific islander": "NHPI",
        "american indian or alaska native": "AIAN",
        "other race": "Other",
        "unknown": "Unknown",
    }
    e["race_eth_full"] = e["race"].str.lower().str.strip().map(race_map)
    e["zip5"] = e["zip_code"].astype(str).str.zfill(9).str[:5]
    keep = ["member_id", "state", "county", "zip5", "race_eth_full", "dual_status_code"]
    out = pd.DataFrame({"member_id": member_ids.values}).merge(
        e[keep], on="member_id", how="left")
    out = out.merge(_payer_state_map(), on="member_id", how="left")
    out["state_eligibility"] = out["state"].fillna("Unlinked")
    out["state"] = out["state_payer"].fillna("Unlinked")
    out["race_eth_full"] = out["race_eth_full"].fillna("Unlinked")
    return out


def run_r0(rising: pd.DataFrame, verbose=True) -> dict:
    print("\n" + "=" * 70)
    print("R0. DATA PROVENANCE AUDIT")
    print("=" * 70)
    aux = load_eligibility_attributes(rising["member_id"])
    res = {}

    res["state_counts"] = aux["state"].value_counts(dropna=False).to_dict()
    res["state_counts_eligibility_file"] = aux["state_eligibility"].value_counts(
        dropna=False).to_dict()
    res["payer_counts"] = aux["payer"].value_counts(dropna=False).to_dict()
    res["race_eth_counts"] = aux["race_eth_full"].value_counts(dropna=False).to_dict()
    res["zip_linked_n"] = int(aux["zip5"].notna().sum())
    res["zip_linked_pct"] = float(aux["zip5"].notna().mean() * 100)
    res["dual_status_available"] = bool(aux["dual_status_code"].notna().any())
    res["race_linked_pct"] = float((aux["race_eth_full"] != "Unlinked").mean() * 100)
    res["race_known_pct"] = float(
        (~aux["race_eth_full"].isin(["Unlinked", "Unknown"])).mean() * 100)

    # Deprivation: confirm the pipeline's adi_percentile is the MIRA risk percentile
    if {"adi_percentile", "risingRiskScorePercentile"} <= set(rising.columns):
        agree = float(np.mean(
            np.isclose(rising["adi_percentile"].fillna(50),
                       rising["risingRiskScorePercentile"].fillna(50))))
        res["adi_equals_risk_percentile_frac"] = agree

    # Goal documentation density
    g = pd.read_parquet(f"{RAW_DIR}/member_goals.parquet")
    res["goals_total"] = int(len(g))
    res["goals_default_pct"] = float((g["category"] == "DEFAULT").mean() * 100)
    documented = set(g.loc[g["category"] != "DEFAULT", "member_id"].dropna().unique())
    rising_documented = rising["member_id"].isin(documented)
    res["cohort_with_documented_goal_n"] = int(rising_documented.sum())
    res["cohort_with_documented_goal_pct"] = float(rising_documented.mean() * 100)

    if verbose:
        print(f"  States represented:            {res['state_counts']}")
        print(f"  Race/ethnicity (eligibility):  {res['race_eth_counts']}")
        print(f"  Race/ethnicity linked:         {res['race_linked_pct']:.1f}% "
              f"(known category {res['race_known_pct']:.1f}%)")
        print(f"  ZIP linked:                    {res['zip_linked_pct']:.1f}%")
        print(f"  Dual status available:         {res['dual_status_available']}")
        print(f"  adi_percentile == risk pctile: "
              f"{res.get('adi_equals_risk_percentile_frac', float('nan')):.3f}")
        print(f"  Goals with category=DEFAULT:   {res['goals_default_pct']:.1f}%")
        print(f"  Cohort with a documented goal: "
              f"{res['cohort_with_documented_goal_n']:,} "
              f"({res['cohort_with_documented_goal_pct']:.1f}%)")
    return res, aux


# ═════════════════════════════════════════════════════════════════════════════
# R1 / R2. Independent and cross-fitted evaluation
# ═════════════════════════════════════════════════════════════════════════════

def crossfit_mu(test: pd.DataFrame, train_all: pd.DataFrame, learner="rf",
                n_folds=5, verbose=True) -> np.ndarray:
    """
    Cross-fitted evaluation matrix. For each test fold, the evaluator is fitted on
    the training set plus the remaining test folds and used to score only the
    held-out fold, so no patient is scored by a model fitted on that patient.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    n = len(test)
    mu_cf = np.zeros((n, len(INTV_ALPHA)))
    for fold, (other_idx, held_idx) in enumerate(kf.split(np.arange(n)), start=1):
        fit_df = pd.concat([train_all, test.iloc[other_idx]], ignore_index=True)
        ev = IndependentEvaluator(learner, seed=SEED + fold).fit(fit_df)
        mu_cf[held_idx] = ev.predict_mu(test.iloc[held_idx])
        if verbose:
            print(f"    {learner} fold {fold}/{n_folds}: fitted on {len(fit_df):,}, "
                  f"scored {len(held_idx):,}")
    return mu_cf


def score_policies(policies, test, mu_map: dict, le: LabelEncoder,
                   verbose=True) -> tuple:
    """Misalignment and direct-method policy value for every policy under every mu."""
    rows, recs_cache = [], {}
    n = len(test)
    for name, fn in policies.items():
        recs = np.asarray(fn(test))
        recs_cache[name] = recs
        row = {"policy": name}
        enc = le.transform(recs)
        for key, mu in mu_map.items():
            row[f"imi_{key}"] = float(imi_from_mu(mu, recs, le).mean())
            row[f"dm_{key}"] = float(mu[np.arange(n), enc].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_key = f"imi_{list(mu_map)[0]}"
    df = df.sort_values(sort_key).reset_index(drop=True)
    if verbose:
        print(df.to_string(index=False))
    return df, recs_cache


# ═════════════════════════════════════════════════════════════════════════════
# R3. Clinical significance of IMI
# ═════════════════════════════════════════════════════════════════════════════

def run_r3_clinical(test, mu_eval, recs_cache, le, verbose=True) -> dict:
    print("\n" + "=" * 70)
    print("R3. CLINICAL SIGNIFICANCE OF INTERVENTION MISALIGNMENT")
    print("=" * 70)
    out = {}
    n = len(test)
    behav = recs_cache["Behavioral Policy"]
    y = test["y_behavioral"].values.astype(float)

    # (a) observed-outcome validation: does modelled misalignment track observed events?
    ind_b = imi_from_mu(mu_eval, behav, le)
    rate_mis = float(y[ind_b == 1].mean())
    rate_al = float(y[ind_b == 0].mean())
    tab = np.array([[y[ind_b == 1].sum(), (1 - y)[ind_b == 1].sum()],
                    [y[ind_b == 0].sum(), (1 - y)[ind_b == 0].sum()]])
    chi2, p_obs = stats.chi2_contingency(tab)[:2]
    out["observed_event_rate_misaligned"] = rate_mis
    out["observed_event_rate_aligned"] = rate_al
    out["observed_rate_difference_pp"] = (rate_mis - rate_al) * 100
    out["observed_risk_ratio"] = rate_mis / max(rate_al, 1e-9)
    out["observed_p"] = float(p_obs)
    out["n_misaligned"] = int(ind_b.sum())

    # risk-adjusted association (logistic, adjusting for the clinical covariates)
    Xc = test[[c for c in CLINICAL_COVS if c in test.columns]].fillna(0).astype(float).values
    Xa = np.hstack([ind_b.reshape(-1, 1), Xc])
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xa, (y > 0.5).astype(int))
    out["adjusted_or_misaligned"] = float(np.exp(lr.coef_[0][0]))

    # (b) magnitude of the modelled gain among misaligned patients
    gains = gain_from_mu(mu_eval, behav, le)
    g_mis = gains[ind_b == 1]
    out["gain_median_pp"] = float(np.median(g_mis) * 100) if len(g_mis) else float("nan")
    out["gain_q1_pp"] = float(np.percentile(g_mis, 25) * 100) if len(g_mis) else float("nan")
    out["gain_q3_pp"] = float(np.percentile(g_mis, 75) * 100) if len(g_mis) else float("nan")

    # (c) events averted per 1,000 patients relative to behavioral routing
    dm_b = float(mu_eval[np.arange(n), le.transform(behav)].mean())
    averted = {}
    for name, recs in recs_cache.items():
        dm = float(mu_eval[np.arange(n), le.transform(recs)].mean())
        averted[name] = {
            "dm": dm,
            "events_averted_per_1000": (dm_b - dm) * 1000,
            "nnt": (1.0 / (dm_b - dm)) if (dm_b - dm) > 1e-9 else float("inf"),
        }
    out["events_averted"] = averted

    # (d) epsilon sweep: misalignment, share of actions changed, modelled events averted
    pearl = recs_cache["PEARL (MoE Router)"]
    sweep = []
    for eps in [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]:
        ind_eps_b = imi_from_mu(mu_eval, behav, le, epsilon=eps)
        ind_eps_p = imi_from_mu(mu_eval, pearl, le, epsilon=eps)
        # actions that would change if we only switched patients flagged at this epsilon
        changed = float((pearl != behav)[ind_eps_b == 1].mean()) if ind_eps_b.sum() else 0.0
        gain_flagged = float(gains[ind_eps_b == 1].sum() / n * 1000) if ind_eps_b.sum() else 0.0
        sweep.append({
            "epsilon": eps,
            "imi_behavioral": float(ind_eps_b.mean()),
            "imi_pearl": float(ind_eps_p.mean()),
            "share_flagged_whose_action_changes": changed,
            "modelled_events_averted_per_1000_if_all_flagged_switched": gain_flagged,
        })
    out["epsilon_sweep"] = sweep

    if verbose:
        print(f"  Observed 90-day event rate, misaligned:  {rate_mis*100:.2f}% "
              f"(n = {int(ind_b.sum()):,})")
        print(f"  Observed 90-day event rate, aligned:     {rate_al*100:.2f}%")
        print(f"  Difference:                              "
              f"{out['observed_rate_difference_pp']:+.2f} pp (p = {p_obs:.4g})")
        print(f"  Risk-adjusted odds ratio:                "
              f"{out['adjusted_or_misaligned']:.2f}")
        print(f"  Modelled gain among misaligned (median): "
              f"{out['gain_median_pp']:.2f} pp "
              f"(IQR {out['gain_q1_pp']:.2f}-{out['gain_q3_pp']:.2f})")
        print("\n  Events averted per 1,000 patients vs. behavioral routing:")
        for k, v in sorted(averted.items(), key=lambda x: -x[1]["events_averted_per_1000"]):
            print(f"    {k:32s} {v['events_averted_per_1000']:+7.2f}  "
                  f"(NNT {v['nnt']:.0f})" if np.isfinite(v["nnt"]) else
                  f"    {k:32s} {v['events_averted_per_1000']:+7.2f}")
        print("\n  Epsilon sweep:")
        print(pd.DataFrame(sweep).to_string(index=False))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# R4. Group-stratified fairness metrics
# ═════════════════════════════════════════════════════════════════════════════

def calibration_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Calibration intercept, slope, ECE, Brier, and AUROC."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    res = {"n": int(len(y)), "observed_rate": float(y.mean()),
           "predicted_rate": float(p.mean())}
    try:
        m = LogisticRegression(max_iter=2000, C=1e6).fit(logit_p.reshape(-1, 1),
                                                         (y > 0.5).astype(int))
        res["calibration_slope"] = float(m.coef_[0][0])
        m0 = LogisticRegression(max_iter=2000, C=1e6, fit_intercept=True)
        # intercept with slope fixed at 1 (offset model), approximated by mean logit shift
        res["calibration_intercept"] = float(
            np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
            - np.log(max(p.mean(), 1e-6) / max(1 - p.mean(), 1e-6)))
    except Exception:
        res["calibration_slope"] = float("nan")
        res["calibration_intercept"] = float("nan")
    try:
        res["auroc"] = float(roc_auc_score(y, p))
    except Exception:
        res["auroc"] = float("nan")
    res["brier"] = float(brier_score_loss(y, p))
    # expected calibration error over equal-count bins
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    ece = sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)) / len(y)
    res["ece"] = float(ece)
    return res


def run_r4_fairness(test, aux_test, mu_eval, recs_cache, le, verbose=True) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("R4. GROUP-STRATIFIED FAIRNESS METRICS")
    print("=" * 70)
    n = len(test)
    y = test["y_behavioral"].values.astype(float)
    behav = recs_cache["Behavioral Policy"]
    pearl = recs_cache["PEARL (MoE Router)"]
    p_own = mu_eval[np.arange(n), le.transform(behav)]
    ind_b = imi_from_mu(mu_eval, behav, le)
    ind_p = imi_from_mu(mu_eval, pearl, le)
    dm_b = mu_eval[np.arange(n), le.transform(behav)]
    dm_p = mu_eval[np.arange(n), le.transform(pearl)]

    strata = {}
    strata["Race/ethnicity"] = aux_test["race_eth_full"].values
    strata["State"] = aux_test["state"].values
    strata["Sex"] = np.where(test["female"].values == 1, "Female", "Male")
    strata["Age band"] = pd.cut(test["age"], [0, 30, 45, 60, 120],
                                labels=["18-29", "30-44", "45-59", "60+"]).astype(str).values
    strata["Rising-risk percentile quintile"] = (
        "Q" + test["adi_quintile"].astype(int).astype(str)).values

    rows = []
    for sname, svals in strata.items():
        for grp in pd.unique(pd.Series(svals).dropna()):
            mask = (svals == grp)
            if mask.sum() < 50:
                continue
            cal = calibration_metrics(y[mask], p_own[mask])
            rows.append({
                "stratum": sname, "group": str(grp), "n": int(mask.sum()),
                "observed_event_rate": cal["observed_rate"],
                "predicted_event_rate": cal["predicted_rate"],
                "calibration_slope": cal["calibration_slope"],
                "calibration_intercept": cal["calibration_intercept"],
                "ece": cal["ece"], "brier": cal["brier"], "auroc": cal["auroc"],
                "imi_behavioral": float(ind_b[mask].mean()),
                "imi_pearl": float(ind_p[mask].mean()),
                "events_averted_per_1000": float((dm_b[mask] - dm_p[mask]).mean() * 1000),
            })
    df = pd.DataFrame(rows)
    if verbose:
        print(df.to_string(index=False))
    return df


# ═════════════════════════════════════════════════════════════════════════════
# R5. Social-needs documentation sensitivity
# ═════════════════════════════════════════════════════════════════════════════

def run_r5_documentation(test, mu_eval, recs_cache, le, verbose=True) -> dict:
    """
    Reviewer 1, comment 6: the social-needs gap may reflect documentation rather
    than prioritization. Restrict to patients with an explicitly documented
    (non-DEFAULT) care-plan goal, where the assigned action is recorded rather
    than imputed from a clinical proxy, and re-estimate the gap.
    """
    print("\n" + "=" * 70)
    print("R5. SOCIAL-NEEDS DOCUMENTATION SENSITIVITY")
    print("=" * 70)
    g = pd.read_parquet(f"{RAW_DIR}/member_goals.parquet")
    documented = set(g.loc[g["category"] != "DEFAULT", "member_id"].dropna().unique())
    mask_doc = test["member_id"].isin(documented).values

    behav = recs_cache["Behavioral Policy"]
    oracle = recs_cache["Oracle (mu-hat optimal)"]
    out = {"n_test": int(len(test)),
           "n_documented": int(mask_doc.sum()),
           "pct_documented": float(mask_doc.mean() * 100)}

    def profile(m, label):
        b = pd.Series(behav[m]).value_counts(normalize=True)
        o = pd.Series(oracle[m]).value_counts(normalize=True)
        d = {}
        for intv in SOCIAL_NEEDS:
            bp, op = float(b.get(intv, 0.0)), float(o.get(intv, 0.0))
            d[intv] = {"behavioral_pct": bp * 100, "preferred_pct": op * 100,
                       "ratio": (op / bp) if bp > 0 else float("inf")}
        d["_social_total_behavioral_pct"] = float(
            sum(b.get(i, 0.0) for i in SOCIAL_NEEDS) * 100)
        d["_social_total_preferred_pct"] = float(
            sum(o.get(i, 0.0) for i in SOCIAL_NEEDS) * 100)
        d["_imi_behavioral"] = float(imi_from_mu(mu_eval[m], behav[m], le).mean())
        d["_n"] = int(m.sum())
        out[label] = d
        return d

    profile(np.ones(len(test), bool), "all_test_patients")
    profile(mask_doc, "documented_goal_only")
    profile(~mask_doc, "proxy_assigned_only")

    if verbose:
        for label in ["all_test_patients", "documented_goal_only", "proxy_assigned_only"]:
            d = out[label]
            print(f"\n  {label} (n = {d['_n']:,}):")
            print(f"    social-needs share, behavioral routing: "
                  f"{d['_social_total_behavioral_pct']:.1f}%")
            print(f"    social-needs share, model-preferred:    "
                  f"{d['_social_total_preferred_pct']:.1f}%")
            print(f"    misalignment under behavioral routing:  "
                  f"{d['_imi_behavioral']*100:.1f}%")
            for intv in SOCIAL_NEEDS:
                r = d[intv]["ratio"]
                print(f"      {intv:22s} behavioral {d[intv]['behavioral_pct']:5.2f}%  "
                      f"preferred {d[intv]['preferred_pct']:5.2f}%  "
                      f"ratio {r:.1f}" if np.isfinite(r) else
                      f"      {intv:22s} behavioral {d[intv]['behavioral_pct']:5.2f}%  "
                      f"preferred {d[intv]['preferred_pct']:5.2f}%  ratio inf")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# R6. State stratification
# ═════════════════════════════════════════════════════════════════════════════

def run_r6_state(test, aux_test, mu_eval, recs_cache, le, verbose=True) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("R6. STATE-STRATIFIED PERFORMANCE")
    print("=" * 70)
    n = len(test)
    behav = recs_cache["Behavioral Policy"]
    pearl = recs_cache["PEARL (MoE Router)"]
    bc = recs_cache["BehavioralCloning SFT (C4)"]
    dm = {k: mu_eval[np.arange(n), le.transform(v)] for k, v in
          [("behavioral", behav), ("pearl", pearl), ("bc", bc)]}
    rows = []
    for st in pd.unique(aux_test["state"]):
        m = (aux_test["state"].values == st)
        if m.sum() < 50:
            continue
        rows.append({
            "state": st, "n": int(m.sum()),
            "observed_event_rate": float(test["y_behavioral"].values[m].mean()),
            "imi_behavioral": float(imi_from_mu(mu_eval[m], behav[m], le).mean()),
            "imi_pearl": float(imi_from_mu(mu_eval[m], pearl[m], le).mean()),
            "imi_bc_sft": float(imi_from_mu(mu_eval[m], bc[m], le).mean()),
            "events_averted_per_1000_pearl": float(
                (dm["behavioral"][m] - dm["pearl"][m]).mean() * 1000),
        })
    df = pd.DataFrame(rows)
    if verbose:
        print(df.to_string(index=False))
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waymark", action="store_true", default=True)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    disable_unused_crossval()
    from data.extract_wpad import build_waymark_population
    from run_pipeline import run_phase_1

    print("\nBuilding Waymark population...")
    pop = build_waymark_population(verbose=True)
    rising = pop.patients.reset_index(drop=True)

    rising_train, rising_test = train_test_split(
        rising, test_size=0.20, random_state=SEED,
        stratify=rising["behavioral_intervention"])
    rising_train = rising_train.reset_index(drop=True)
    rising_test = rising_test.reset_index(drop=True)
    train_pids = set(rising_train["patient_id"].tolist())
    wpad_pairs_train = pop.wpad_pairs[
        pop.wpad_pairs["patient_id"].isin(train_pids)].reset_index(drop=True)

    le = LabelEncoder().fit(INTV_ALPHA)

    # ── R0 ────────────────────────────────────────────────────────────────
    r0, aux_all = run_r0(rising, verbose=True)

    # ── Arm 1: policies trained on the full training set (as published) ──
    # Holds policy training fixed and varies only the evaluation model, which
    # isolates the shared-model concern from any loss of training sample.
    print("\n" + "=" * 70)
    print("ARM 1. POLICIES TRAINED ON THE FULL TRAINING SET (as published)")
    print("=" * 70)
    phase0_full = light_phase_0(rising_train, rising_test)
    phase1_full = run_phase_1(pop, rising_train, wpad_pairs_train, phase0_full,
                              verbose=False)
    policies_full, test = build_policy_functions(pop, phase0_full, phase1_full,
                                                 rising_test)
    mu_primary = phase0_full["imi_result"]["mu_hat"]

    print("\n  Cross-fitting the independent evaluators...")
    mu_cf_rf = crossfit_mu(test, rising_train, "rf", args.n_folds)
    mu_cf_lg = crossfit_mu(test, rising_train, "logit", args.n_folds)

    print("\n  Arm 1 policy scores "
          "(primary = published S-learner; crossfit_rf / crossfit_logit = independent):")
    arm1_df, recs_full = score_policies(
        policies_full, test,
        {"crossfit_rf": mu_cf_rf, "crossfit_logit": mu_cf_lg, "primary": mu_primary},
        le, verbose=True)

    # ── Arm 2: full sample splitting (policy half A, evaluator half B) ───
    print("\n" + "=" * 70)
    print("ARM 2. FULL SAMPLE SPLITTING (policy half A, evaluator half B)")
    print("=" * 70)
    half_a, half_b = train_test_split(
        rising_train, test_size=0.50, random_state=SEED,
        stratify=rising_train["behavioral_intervention"])
    half_a = half_a.reset_index(drop=True)
    half_b = half_b.reset_index(drop=True)
    pids_a = set(half_a["patient_id"].tolist())
    wpad_pairs_a = pop.wpad_pairs[
        pop.wpad_pairs["patient_id"].isin(pids_a)].reset_index(drop=True)
    print(f"  Policy half A: {len(half_a):,} patients, "
          f"{len(wpad_pairs_a):,} preference pairs")
    print(f"  Evaluator half B: {len(half_b):,} patients (never used for training)")

    phase0_a = light_phase_0(half_a, rising_test)
    phase1_a = run_phase_1(pop, half_a, wpad_pairs_a, phase0_a, verbose=False)
    policies_a, test_a = build_policy_functions(pop, phase0_a, phase1_a, rising_test)
    ev_b = IndependentEvaluator("rf").fit(half_b)
    mu_b = ev_b.predict_mu(test_a)
    print("\n  Arm 2 policy scores (evaluator fitted on half B only):")
    arm2_df, _ = score_policies(policies_a, test_a, {"splitsample_rf": mu_b}, le,
                                verbose=True)

    # ── R3-R6 use arm 1 policies and the cross-fitted evaluator ──────────
    aux_test = load_eligibility_attributes(test["member_id"])
    r3 = run_r3_clinical(test, mu_cf_rf, recs_full, le, verbose=True)
    r4_df = run_r4_fairness(test, aux_test, mu_cf_rf, recs_full, le, verbose=True)
    r5 = run_r5_documentation(test, mu_cf_rf, recs_full, le, verbose=True)
    r6_df = run_r6_state(test, aux_test, mu_cf_rf, recs_full, le, verbose=True)

    # ── Save ─────────────────────────────────────────────────────────────
    arm1_df.to_csv(RESULTS_DIR / "revision_arm1_independent_evaluator.csv", index=False)
    arm2_df.to_csv(RESULTS_DIR / "revision_arm2_splitsample.csv", index=False)
    r4_df.to_csv(RESULTS_DIR / "revision_r4_fairness.csv", index=False)
    r6_df.to_csv(RESULTS_DIR / "revision_r6_state.csv", index=False)
    pd.DataFrame(r3["epsilon_sweep"]).to_csv(
        RESULTS_DIR / "revision_r3_epsilon_sweep.csv", index=False)
    with open(RESULTS_DIR / "revision_summary.json", "w") as f:
        json.dump({"r0_provenance": r0, "r3_clinical": r3, "r5_documentation": r5,
                   "n_test": int(len(test)), "n_train": int(len(rising_train)),
                   "n_half_a": int(len(half_a)), "n_half_b": int(len(half_b)),
                   "n_wpad_pairs_train": int(len(wpad_pairs_train)),
                   "n_wpad_pairs_half_a": int(len(wpad_pairs_a)),
                   "runtime_minutes": (time.time() - t0) / 60},
                  f, indent=2, default=str)
    print(f"\nSaved revision outputs to {RESULTS_DIR}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
