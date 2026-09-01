"""
Does intervention misalignment carry information about 90-day acute care events
beyond what the covariates already provide?

The misalignment flag is derived from an outcome model fitted on the same
covariates as any risk score, so a comparison against the rising-risk score alone
would credit the flag for information it merely relays. The decisive comparison
is against a covariate-only risk model built from the identical feature set: any
incremental value must then come from the action assignment itself, which is the
only information the flag adds.

Reports, on the held-out test set:
  - AUROC for a covariate-only risk model, and for that model plus the flag or
    the continuous available gain, with a paired bootstrap difference
  - likelihood ratio tests for the added terms
  - net benefit across decision thresholds
  - event rates by flag status within strata of predicted risk, which shows
    whether the flag separates patients the risk model already ranks together
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings("ignore")

from revision_analyses import INTV_ALPHA, SEED, feature_matrix, disable_unused_crossval
from pearl_v2 import (CrossFitNuisance, admissible_mask, misalignment,
                      pessimistic_policy, K_FOLDS, EPSILON, MIN_ACTION_N)

_REPO = _ROOT.parents[1]
OUT = Path(os.environ.get("PEARL_OUTPUT_BASE",
                          str(_REPO / "notebooks" / "pearl" / "outputs"))) / "results"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 2000


def crossfit_covariate_risk(df: pd.DataFrame, seed=SEED) -> np.ndarray:
    """Cross-fitted covariate-only risk, using no action information."""
    X = feature_matrix(df)
    y = df["y_behavioral"].values.astype(float)
    p = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, (y > 0.5).astype(int)):
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, subsample=1.0,
                                      min_samples_leaf=20, random_state=seed)
        m.fit(X[tr], y[tr])
        p[te] = np.clip(m.predict(X[te]), 1e-6, 1 - 1e-6)
    return p


def lr_test(y, base, full):
    """Likelihood ratio test for nested logistic models."""
    def ll(Xd):
        m = LogisticRegression(max_iter=2000, C=1e6).fit(Xd, y)
        pr = np.clip(m.predict_proba(Xd)[:, 1], 1e-9, 1 - 1e-9)
        return float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))), m
    ll0, _ = ll(base)
    ll1, m1 = ll(full)
    stat = 2 * (ll1 - ll0)
    dfree = full.shape[1] - base.shape[1]
    return stat, dfree, float(stats.chi2.sf(max(stat, 0), dfree)), m1


def fitted_prob(y, Xd):
    m = LogisticRegression(max_iter=2000, C=1e6).fit(Xd, y)
    return m.predict_proba(Xd)[:, 1]


def net_benefit(y, p, thresholds):
    n = len(y)
    out = []
    for t in thresholds:
        flag = p >= t
        tp = np.sum(flag & (y == 1))
        fp = np.sum(flag & (y == 0))
        out.append(tp / n - (fp / n) * (t / (1 - t)))
    return np.array(out)


def main():
    t0 = time.time()
    disable_unused_crossval()
    from data.extract_wpad import build_waymark_population

    pop = build_waymark_population(verbose=False)
    rising = pop.patients.reset_index(drop=True)
    train, test = train_test_split(rising, test_size=0.20, random_state=SEED,
                                   stratify=rising["behavioral_intervention"])
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    full = pd.concat([train, test], ignore_index=True)
    n_tr = len(train)

    le = LabelEncoder().fit(INTV_ALPHA)
    counts = np.bincount(le.transform(rising["behavioral_intervention"].values),
                         minlength=len(INTV_ALPHA))

    print("Cross-fitting action-aware nuisances...")
    nz = CrossFitNuisance("gbm").fit_predict(full)
    mu, sg = nz.mu[n_tr:], nz.sigma[n_tr:]
    prop, a_obs = nz.prop[n_tr:], nz.a_obs[n_tr:]
    y = nz.y[n_tr:]
    adm = admissible_mask(prop, counts)

    flag = misalignment(mu, sg, adm, a_obs, lam=0.0, eps=EPSILON)
    # continuous version: how much better the best admissible alternative looks
    n = len(y)
    lb_alt = np.where(adm, mu, np.inf)
    lb_alt[np.arange(n), a_obs] = np.inf
    gain = np.clip(mu[np.arange(n), a_obs] - lb_alt.min(axis=1), 0, None)

    print("Cross-fitting covariate-only risk (no action information)...")
    p_cov_all = crossfit_covariate_risk(full)
    p_cov = np.clip(p_cov_all[n_tr:], 1e-6, 1 - 1e-6)
    logit_cov = np.log(p_cov / (1 - p_cov))

    yb = (y > 0.5).astype(int)
    base = logit_cov.reshape(-1, 1)
    with_flag = np.column_stack([logit_cov, flag])
    with_gain = np.column_stack([logit_cov, gain])
    with_both = np.column_stack([logit_cov, flag, gain])

    print(f"\nHeld-out test set N = {n:,}; events {int(yb.sum()):,} "
          f"({100*yb.mean():.2f}%); flagged {int(flag.sum()):,} "
          f"({100*flag.mean():.2f}%)")

    models = {
        "covariate risk only": base,
        "+ misalignment flag": with_flag,
        "+ available gain": with_gain,
        "+ flag and gain": with_both,
    }
    probs, rows = {}, []
    for name, Xd in models.items():
        pr = fitted_prob(yb, Xd)
        probs[name] = pr
        rows.append({"model": name, "auroc": float(roc_auc_score(yb, pr))})
    res = pd.DataFrame(rows)

    # paired bootstrap for the AUROC difference against the covariate-only model
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    ref = probs["covariate risk only"]
    for name in list(models)[1:]:
        d = []
        for b in idx:
            if yb[b].sum() == 0 or yb[b].sum() == len(b):
                continue
            d.append(roc_auc_score(yb[b], probs[name][b])
                     - roc_auc_score(yb[b], ref[b]))
        d = np.array(d)
        res.loc[res.model == name, "delta_auroc"] = float(d.mean())
        res.loc[res.model == name, "ci_lower"] = float(np.percentile(d, 2.5))
        res.loc[res.model == name, "ci_upper"] = float(np.percentile(d, 97.5))
        res.loc[res.model == name, "p_two_sided"] = float(
            2 * min((d <= 0).mean(), (d >= 0).mean()))

    for name, Xd in list(models.items())[1:]:
        stat, dfree, p, _ = lr_test(yb, base, Xd)
        res.loc[res.model == name, "lr_stat"] = stat
        res.loc[res.model == name, "lr_p"] = p

    print("\nDiscrimination and incremental value:")
    print(res.to_string(index=False))

    # does the flag separate patients the covariate model ranks together?
    q = pd.qcut(pd.Series(p_cov), 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    strat = []
    for lev in sorted(pd.Series(q).dropna().unique()):
        m_ = (q == lev).values
        f1, f0 = m_ & (flag == 1), m_ & (flag == 0)
        strat.append({
            "risk_quintile": int(lev), "n": int(m_.sum()),
            "n_flagged": int(f1.sum()),
            "event_rate_flagged": float(yb[f1].mean()) if f1.sum() else np.nan,
            "event_rate_not_flagged": float(yb[f0].mean()) if f0.sum() else np.nan,
        })
    stratdf = pd.DataFrame(strat)
    stratdf["difference_pp"] = 100 * (stratdf["event_rate_flagged"]
                                      - stratdf["event_rate_not_flagged"])
    print("\nEvent rate by flag status within strata of covariate-only risk:")
    print(stratdf.to_string(index=False))

    # decision curve
    th = np.linspace(0.01, 0.20, 20)
    nb = pd.DataFrame({"threshold": th})
    for name in ["covariate risk only", "+ misalignment flag"]:
        nb[name] = net_benefit(yb, probs[name], th)
    nb["treat_all"] = [yb.mean() - (1 - yb.mean()) * (t / (1 - t)) for t in th]
    nb["gain_from_flag"] = nb["+ misalignment flag"] - nb["covariate risk only"]
    print("\nNet benefit (per patient) across decision thresholds:")
    print(nb.round(5).to_string(index=False))

    res.to_csv(OUT / "incremental_value.csv", index=False)
    stratdf.to_csv(OUT / "incremental_value_by_risk_stratum.csv", index=False)
    nb.to_csv(OUT / "incremental_value_decision_curve.csv", index=False)
    json.dump({"n_test": int(n), "events": int(yb.sum()),
               "flagged": int(flag.sum()),
               "auroc_covariate_only": float(res.loc[0, "auroc"]),
               "runtime_min": (time.time() - t0) / 60},
              open(OUT / "incremental_value_summary.json", "w"), indent=2)
    print(f"\nSaved to {OUT}. Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
