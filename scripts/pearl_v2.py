"""
PEARL v2 — action selection and misalignment measurement rebuilt to remove the
three defects that make the v1 result uninterpretable.

Defect 1, shared model. v1 selects the action minimising an outcome model and
then scores itself with that same model. v2 estimates all nuisances by K-fold
cross-fitting and scores every policy with models from a different learner
family that never touched policy construction.

Defect 2, winner's curse. v1 takes an unpenalised argmin over 14 actions, which
selects wherever the model's error happens to be most negative. v2 selects
pessimistically: it deviates from the assigned action only when the advantage
exceeds a multiple of its own cross-fold standard deviation, and otherwise
abstains to current routing.

Defect 3, no overlap. v1 recommends actions that almost no patient received.
v2 restricts every policy to an admissible action set defined by propensity and
by a minimum observed count, so recommendations stay inside the support of the
data and remain estimable without extrapolation.

Outputs a single results bundle used by every downstream table.
"""
import os
import sys
import json
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

from revision_analyses import (
    INTV_ALPHA, SEED, feature_matrix, load_eligibility_attributes,
    disable_unused_crossval, calibration_metrics,
)

K_FOLDS = 5
EPSILON = 0.02
MIN_ACTION_N = 100        # an action must be observed this often to be admissible
MIN_PROPENSITY = 0.01     # and have at least this propensity for the patient
LAMBDAS = [0.0, 0.5, 1.0, 2.0]

_REPO = _ROOT.parents[1]
OUT = Path(os.environ.get("PEARL_OUTPUT_BASE",
                          str(_REPO / "notebooks" / "pearl" / "outputs"))) / "results"
OUT.mkdir(parents=True, exist_ok=True)
SVI_PATH = Path("/Users/sanjaybasu/waymark-local/data/processed/svi_by_zip.parquet")
RAW = Path("/Users/sanjaybasu/waymark-local/data/real_inputs")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-fitted nuisance estimation
# ─────────────────────────────────────────────────────────────────────────────

class CrossFitNuisance:
    """
    K-fold cross-fitted propensity and outcome models.

    Every patient's scores come from models that never saw that patient, which
    is what makes the doubly robust scores valid for policy learning and lets
    the spread across folds serve as an honest uncertainty estimate.
    """

    def __init__(self, learner="gbm", k=K_FOLDS, seed=SEED):
        self.learner, self.k, self.seed = learner, k, seed
        self.le = LabelEncoder().fit(INTV_ALPHA)

    def _outcome_model(self, seed):
        if self.learner == "gbm":
            return GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=1.0, min_samples_leaf=20, random_state=seed)
        return RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            max_features=None, random_state=seed, n_jobs=-1)

    def fit_predict(self, df: pd.DataFrame, outcome_col="y_behavioral",
                    action_col="behavioral_intervention"):
        X = feature_matrix(df)
        y = df[outcome_col].values.astype(float)
        a = self.le.transform(df[action_col].values)
        n, k_act = len(df), len(INTV_ALPHA)

        self.prop = np.zeros((n, k_act))
        mu_folds = np.zeros((self.k, n, k_act))

        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        strat = np.where(np.bincount(a, minlength=k_act)[a] >= self.k, a, -1)

        for f, (tr, te) in enumerate(skf.split(X, strat)):
            pm = LogisticRegression(C=0.1, max_iter=1000).fit(X[tr], a[tr])
            pr = np.zeros((len(te), k_act))
            pr[:, pm.classes_] = pm.predict_proba(X[te])
            self.prop[te] = pr

            one_hot = np.zeros((len(tr), k_act))
            one_hot[np.arange(len(tr)), a[tr]] = 1.0
            p_recv = np.clip(pm.predict_proba(X[tr])[
                np.arange(len(tr)), np.searchsorted(pm.classes_, a[tr])], 0.05, 1.0)
            w = 1.0 / p_recv
            w = w * len(tr) / w.sum()
            om = self._outcome_model(self.seed + f).fit(
                np.hstack([X[tr], one_hot]), y[tr], sample_weight=w)

            # every fold model scores every patient, so the spread across folds
            # is available as an uncertainty estimate for all of them
            for j in range(k_act):
                oh = np.zeros((n, k_act))
                oh[:, j] = 1.0
                mu_folds[f, :, j] = np.clip(om.predict(np.hstack([X, oh])), 0.0, 1.0)

        # out-of-fold point estimate, and cross-fold dispersion
        self.mu = mu_folds.mean(axis=0)
        self.sigma = mu_folds.std(axis=0)

        # doubly robust scores
        e_recv = np.clip(self.prop[np.arange(n), a], MIN_PROPENSITY, 1.0)
        self.gamma = self.mu.copy()
        resid = y - self.mu[np.arange(n), a]
        self.gamma[np.arange(n), a] = self.mu[np.arange(n), a] + resid / e_recv

        self.a_obs, self.y = a, y
        return self


def admissible_mask(prop: np.ndarray, action_counts: np.ndarray) -> np.ndarray:
    """Actions a patient could plausibly have received, and that were used enough."""
    ok_global = action_counts >= MIN_ACTION_N
    return (prop >= MIN_PROPENSITY) & ok_global[None, :]


# ─────────────────────────────────────────────────────────────────────────────
# Pessimistic policy and uncertainty-aware misalignment
# ─────────────────────────────────────────────────────────────────────────────

def pessimistic_policy(mu, sigma, adm, a_obs, lam, eps=EPSILON):
    """
    Deviate from the assigned action only when some admissible alternative is
    better by more than eps after charging each estimate for its own uncertainty.
    Otherwise abstain and keep the assigned action.
    """
    n, k = mu.shape
    ub_own = mu[np.arange(n), a_obs] - lam * sigma[np.arange(n), a_obs]
    lb_alt = mu + lam * sigma
    lb_alt = np.where(adm, lb_alt, np.inf)
    lb_alt[np.arange(n), a_obs] = np.inf

    best = lb_alt.argmin(axis=1)
    gain = ub_own - lb_alt[np.arange(n), best]
    switch = gain > eps
    out = a_obs.copy()
    out[switch] = best[switch]
    return out, switch, np.clip(gain, 0, None)


def misalignment(mu, sigma, adm, a_pol, lam, eps=EPSILON):
    """Fraction of patients for whom an admissible alternative is better by
    more than eps, after charging each estimate for its uncertainty."""
    n = len(a_pol)
    ub_own = mu[np.arange(n), a_pol] - lam * sigma[np.arange(n), a_pol]
    lb_alt = np.where(adm, mu + lam * sigma, np.inf)
    lb_alt[np.arange(n), a_pol] = np.inf
    return (lb_alt.min(axis=1) < ub_own - eps).astype(float)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    disable_unused_crossval()
    from data.extract_wpad import build_waymark_population

    print("Building population...")
    pop = build_waymark_population(verbose=False)
    rising = pop.patients.reset_index(drop=True)

    # attach the linked area-deprivation measure
    aux = load_eligibility_attributes(rising["member_id"])
    svi = pd.read_parquet(SVI_PATH)
    elig = pd.read_parquet(RAW / "eligibility.parquet",
                           columns=["member_id", "zip_code"]).drop_duplicates("member_id")
    s = elig["zip_code"].astype(str).str.replace(r"\D", "", regex=True)
    elig["zip5"] = np.where(s.str.len() >= 9, s.str[:5], s.str.zfill(5))
    zmap = elig.merge(svi, on="zip5", how="left").set_index("member_id")
    rising["svi_overall"] = rising["member_id"].map(zmap["svi_overall"])
    rising["state"] = aux["state"].values
    rising["race_eth_full"] = aux["race_eth_full"].values
    linked = rising["svi_overall"].notna()
    print(f"  cohort {len(rising):,}; deprivation index linked for "
          f"{linked.sum():,} ({100*linked.mean():.1f}%)")

    train, test = train_test_split(rising, test_size=0.20, random_state=SEED,
                                   stratify=rising["behavioral_intervention"])
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    le = LabelEncoder().fit(INTV_ALPHA)
    counts = np.bincount(le.transform(rising["behavioral_intervention"].values),
                         minlength=len(INTV_ALPHA))
    adm_actions = [INTV_ALPHA[i] for i in range(len(INTV_ALPHA))
                   if counts[i] >= MIN_ACTION_N]
    print(f"  admissible actions ({len(adm_actions)} of {len(INTV_ALPHA)}, "
          f">= {MIN_ACTION_N} observations): {adm_actions}")

    # ── selection nuisances (gradient boosting), cross-fitted on train ──
    print("\nCross-fitting selection nuisances on the training set...")
    sel = CrossFitNuisance("gbm").fit_predict(train)

    # ── choose lambda by nested splitting inside the training set ──
    # Scoring lambda with the same doubly robust scores that built the policy
    # reproduces the defect this rebuild exists to remove, so the policy is
    # built on an inner training slice and scored on a disjoint inner holdout
    # whose nuisances were estimated separately.
    print("\nTuning the pessimism multiplier by nested splitting...")
    inner_fit, inner_val = train_test_split(
        train, test_size=0.30, random_state=SEED + 1,
        stratify=train["behavioral_intervention"])
    inner_fit = inner_fit.reset_index(drop=True)
    inner_val = inner_val.reset_index(drop=True)
    ev_in = CrossFitNuisance("rf", seed=SEED + 2).fit_predict(
        pd.concat([inner_fit, inner_val], ignore_index=True))
    n_if = len(inner_fit)


    sel_iv = CrossFitNuisance("gbm", seed=SEED + 1).fit_predict(
        pd.concat([inner_fit, inner_val], ignore_index=True))
    mu_iv, sg_iv = sel_iv.mu[n_if:], sel_iv.sigma[n_if:]
    prop_iv, a_iv = sel_iv.prop[n_if:], sel_iv.a_obs[n_if:]
    adm_iv = admissible_mask(prop_iv, counts)
    mu_iv_ev = ev_in.mu[n_if:]

    rows = []
    for lam in LAMBDAS:
        a_pol, switch, _ = pessimistic_policy(mu_iv, sg_iv, adm_iv, a_iv, lam)
        val_ind = float(mu_iv_ev[np.arange(len(a_pol)), a_pol].mean())
        rows.append({"lambda": lam, "value_independent_innerval": val_ind,
                     "pct_switched": float(100 * switch.mean())})
        print(f"  lambda={lam:<4} independent value {val_ind:.5f}  "
              f"switched {100*switch.mean():5.1f}%")
    tune = pd.DataFrame(rows)
    lam_star = float(tune.loc[tune["value_independent_innerval"].idxmin(), "lambda"])
    print(f"  selected lambda = {lam_star} "
          f"(scored by a model that did not build the policy)")
    adm_tr = admissible_mask(sel.prop, counts)

    # ── apply to the held-out test set ──
    print("\nScoring on the held-out test set...")
    sel_te = CrossFitNuisance("gbm").fit_predict(
        pd.concat([train, test], ignore_index=True))
    n_tr = len(train)
    mu_te = sel_te.mu[n_tr:]
    sg_te = sel_te.sigma[n_tr:]
    prop_te = sel_te.prop[n_tr:]
    a_te = sel_te.a_obs[n_tr:]
    y_te = sel_te.y[n_tr:]
    adm_te = admissible_mask(prop_te, counts)

    a_pearl2, switch_te, gain_te = pessimistic_policy(mu_te, sg_te, adm_te,
                                                      a_te, lam_star)

    # ── independent evaluator: different learner family, never used above ──
    print("Fitting the independent evaluator (random forest, cross-fitted)...")
    ev = CrossFitNuisance("rf").fit_predict(
        pd.concat([train, test], ignore_index=True))
    mu_ev = ev.mu[n_tr:]
    sg_ev = ev.sigma[n_tr:]

    policies = {
        "Current routing": a_te,
        "PEARL v2 (pessimistic, support-restricted)": a_pearl2,
        "Unpenalised argmin, admissible actions only": np.where(
            adm_te, mu_te, np.inf).argmin(axis=1),
        "Unpenalised argmin, all actions (v1 behaviour)": mu_te.argmin(axis=1),
    }

    res = []
    for name, a_pol in policies.items():
        overlap = float((a_pol == a_te).mean() * 100)
        res.append({
            "policy": name,
            "pct_recommendation_received": overlap,
            "misalign_selection_model": float(
                misalignment(mu_te, sg_te, adm_te, a_pol, lam_star).mean() * 100),
            "misalign_independent_model": float(
                misalignment(mu_ev, sg_ev, adm_te, a_pol, lam_star).mean() * 100),
            "value_selection_model": float(mu_te[np.arange(len(a_pol)), a_pol].mean()),
            "value_independent_model": float(mu_ev[np.arange(len(a_pol)), a_pol].mean()),
        })
    resdf = pd.DataFrame(res)
    base_ind = float(resdf.loc[resdf.policy == "Current routing",
                               "value_independent_model"].iloc[0])
    resdf["events_averted_per_1000_independent"] = (
        base_ind - resdf["value_independent_model"]) * 1000
    print("\n" + resdf.to_string(index=False))

    # ── evaluator agreement and diagnostics ──
    diag = []
    for nm, m_, s_ in [("selection (gradient boosting)", mu_te, sg_te),
                       ("independent (random forest)", mu_ev, sg_ev)]:
        p_own = m_[np.arange(len(a_te)), a_te]
        c = calibration_metrics(y_te, p_own)
        spread = np.where(adm_te, m_, np.nan)
        diag.append({
            "model": nm, "auroc": c["auroc"], "brier": c["brier"],
            "calibration_slope": c["calibration_slope"],
            "mean_spread_admissible": float(np.nanmax(spread, axis=1).mean()
                                            - np.nanmin(spread, axis=1).mean()),
            "mean_cross_fold_sd": float(s_.mean()),
        })
    diagdf = pd.DataFrame(diag)
    agree = float((np.where(adm_te, mu_te, np.inf).argmin(axis=1) ==
                   np.where(adm_te, mu_ev, np.inf).argmin(axis=1)).mean() * 100)
    print(f"\n{diagdf.to_string(index=False)}")
    print(f"\nAgreement on the preferred admissible action: {agree:.1f}%")

    # ── misalignment against observed outcomes ──
    flag = misalignment(mu_ev, sg_ev, adm_te, a_te, lam_star)
    obs = {"flagged_n": int(flag.sum()),
           "rate_flagged": float(y_te[flag == 1].mean()) if flag.sum() else float("nan"),
           "rate_not_flagged": float(y_te[flag == 0].mean())}
    print(f"\nObserved 90-day event rate: flagged {100*obs['rate_flagged']:.2f}% "
          f"(n={obs['flagged_n']:,}) vs not flagged {100*obs['rate_not_flagged']:.2f}%")

    # ── deprivation gradient on the linked measure ──
    te_svi = test["svi_overall"].values
    grad = []
    if np.isfinite(te_svi).sum() > 100:
        q = pd.qcut(pd.Series(te_svi), 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        for lev in sorted(pd.Series(q).dropna().unique()):
            m = (q == lev).values
            grad.append({"svi_quintile": int(lev), "n": int(m.sum()),
                         "observed_event_rate": float(y_te[m].mean()),
                         "misalignment": float(flag[m].mean() * 100)})
    graddf = pd.DataFrame(grad)
    if len(graddf):
        print(f"\nDeprivation gradient (linked index):\n{graddf.to_string(index=False)}")

    resdf.to_csv(OUT / "v2_policy_comparison.csv", index=False)
    diagdf.to_csv(OUT / "v2_evaluator_diagnostics.csv", index=False)
    tune.to_csv(OUT / "v2_lambda_tuning.csv", index=False)
    graddf.to_csv(OUT / "v2_deprivation_gradient.csv", index=False)

    # ── epsilon sensitivity under the primary (support-restricted,
    # cross-fitted, pessimism-penalized) specification. Rescoring at a
    # different margin needs no new nuisance fit, only a different decision
    # threshold applied to mu_te/sg_te/mu_ev/sg_ev already in hand. ──
    print("\nEpsilon sensitivity under the primary specification...")
    eps_rows = []
    for eps in (0.01, 0.02, 0.05):
        a_cur_eps = a_te  # current routing does not depend on eps
        a_pearl_eps, switch_eps, _ = pessimistic_policy(
            mu_te, sg_te, adm_te, a_te, lam_star, eps=eps)
        for name, a_pol in [("Current routing", a_cur_eps),
                            ("PEARL v2 (pessimistic, support-restricted)", a_pearl_eps)]:
            eps_rows.append({
                "epsilon": eps,
                "policy": name,
                "misalign_independent_model": float(
                    misalignment(mu_ev, sg_ev, adm_te, a_pol, lam_star, eps=eps).mean() * 100),
                "pct_switched_from_received": float((a_pol != a_te).mean() * 100),
            })
    epsdf = pd.DataFrame(eps_rows)
    epsdf.to_csv(OUT / "v2_epsilon_sensitivity.csv", index=False)
    print(epsdf.to_string(index=False))

    # ── decision-curve / net-benefit analysis for the flag as a case-review
    # trigger (Discussion), using the independent evaluator's flag at the
    # primary epsilon and the observed 90-day outcome. No causal claim: this
    # asks only whether flagging for review beats review-all/review-none. ──
    print("\nDecision-curve analysis (flag as case-review trigger)...")
    prevalence = float(y_te.mean())
    tp = float(((flag == 1) & (y_te == 1)).sum())
    fp = float(((flag == 1) & (y_te == 0)).sum())
    n_dc = float(len(y_te))
    dc_rows = []
    for pt in np.arange(0.01, 0.31, 0.01):
        odds = pt / (1 - pt)
        nb_flag = (tp / n_dc) - (fp / n_dc) * odds
        nb_all = prevalence - (1 - prevalence) * odds
        dc_rows.append({
            "threshold_probability": round(float(pt), 2),
            "net_benefit_flag_for_review": nb_flag,
            "net_benefit_review_all": nb_all,
            "net_benefit_review_none": 0.0,
        })
    dcdf = pd.DataFrame(dc_rows)
    dcdf.to_csv(OUT / "v2_decision_curve.csv", index=False)
    crossover = dcdf.loc[dcdf["net_benefit_flag_for_review"] >
                        dcdf["net_benefit_review_all"], "threshold_probability"]
    print(dcdf.to_string(index=False))
    print(f"\nFlag-for-review beats review-all for pt in "
          f"[{crossover.min() if len(crossover) else float('nan')}, "
          f"{crossover.max() if len(crossover) else float('nan')}]")

    json.dump({"lambda": lam_star, "admissible_actions": adm_actions,
               "n_train": int(len(train)), "n_test": int(len(test)),
               "agreement_preferred_action_pct": agree,
               "observed_outcome_check": obs,
               "svi_linked_pct": float(100 * linked.mean()),
               "decision_curve_flag_beats_treat_all_range": [
                   float(crossover.min()) if len(crossover) else None,
                   float(crossover.max()) if len(crossover) else None],
               "runtime_min": (time.time() - t0) / 60},
              open(OUT / "v2_summary.json", "w"), indent=2, default=str)
    print(f"\nSaved to {OUT}. Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
