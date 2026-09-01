"""
T7 — post hoc falsification check using care-team staffing capacity.

T1-T6 (falsification_tests.py) test whether patient health predicts the
timing or direction of a within-patient administrative discontinuity. This
script adds a check on a different confound axis raised in review: whether
staffing capacity, a proxy for organizational readiness, predicts pair type
(primary vs. weak-positive, i.e. whether the pre-enrollment window was
unfavorable) beyond what patient covariates explain.

Y_on is 0 for every retained pair by construction (Algorithm S1 discards
Y_on = 1 pairs), so it cannot serve as the outcome here. Pair type carries
the same information T3 targets (does an administrative/organizational
signal predict something that should only reflect patient selection) and
has real variation in this sample.
"""
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from revision_analyses import disable_unused_crossval, load_eligibility_attributes

STAFF_PATH = Path("/Users/sanjaybasu/waymark-local/data/real_inputs/"
                   "Master Care Delivery Staffing Document 2023-2025 - Sheet1.csv")
N_FALSIFICATION_TESTS = 7  # T1-T6 pre-specified, T7 post hoc; Bonferroni across all 7


def build_monthly_chw_headcount():
    staff = pd.read_csv(STAFF_PATH)
    staff = staff[staff["Role"].str.contains("Community Health Worker", case=False, na=False)]
    staff = staff[~staff["Role"].str.contains("Lead", case=False, na=False)]
    staff["start"] = pd.to_datetime(staff["Start Date"], format="%m/%Y", errors="coerce")
    staff["end"] = pd.to_datetime(staff["EndDate"], format="%m/%Y",
                                   errors="coerce").fillna(pd.Timestamp("2025-12-01"))
    months = pd.period_range("2023-01", "2025-09", freq="M")
    rows = []
    for st in ("VA", "WA", "OH"):
        sub = staff[staff["State"] == st]
        for m in months:
            mstart = m.to_timestamp()
            active = int(((sub["start"] <= mstart) & (sub["end"] >= mstart)).sum())
            rows.append({"state": st, "onboard_month": m, "chw_active": active})
    return pd.DataFrame(rows)


def lr_test(df, y, cols_null, cols_full, label):
    Xn = StandardScaler().fit_transform(df[cols_null].values.astype(float))
    Xf = StandardScaler().fit_transform(df[cols_full].values.astype(float))
    m0 = LogisticRegression(C=1.0, max_iter=3000).fit(Xn, y)
    m1 = LogisticRegression(C=1.0, max_iter=3000).fit(Xf, y)
    null_ll = -log_loss(y, m0.predict_proba(Xn)[:, 1], labels=[0, 1]) * len(y)
    full_ll = -log_loss(y, m1.predict_proba(Xf)[:, 1], labels=[0, 1]) * len(y)
    lr_stat = 2 * (full_ll - null_ll)
    dfree = len(cols_full) - len(cols_null)
    p = stats.chi2.sf(lr_stat, dfree)
    alpha_bonf = 0.05 / N_FALSIFICATION_TESTS
    print(f"[{label}] N={len(df)} LR={lr_stat:.3f} df={dfree} p={p:.4f} "
          f"-> {'PASS' if p > alpha_bonf else 'CONCERN'}")
    return p


def main():
    disable_unused_crossval()
    from data.extract_wpad import build_waymark_population, _build_wpad_type1, _load_raw, _build_id_bridge

    d = _load_raw()
    bridge = _build_id_bridge(d)
    pairs = _build_wpad_type1(d, bridge).reset_index(drop=True)

    aux = load_eligibility_attributes(pairs["patient_id"])
    pairs["state"] = aux["state"].values
    pairs["onboard_month"] = pd.to_datetime(pairs["on_start"]).dt.to_period("M")
    pairs = pairs[pairs["state"].isin(["VA", "WA", "OH"])].reset_index(drop=True)

    staffdf = build_monthly_chw_headcount()
    pairs = pairs.merge(staffdf, on=["state", "onboard_month"], how="left")
    for lag in (1, 2, 3):
        lagdf = staffdf.copy()
        lagdf["onboard_month"] = lagdf["onboard_month"] + lag
        lagdf = lagdf.rename(columns={"chw_active": f"chw_active_lag{lag}"})
        pairs = pairs.merge(lagdf, on=["state", "onboard_month"], how="left")
    staff_cols = ["chw_active", "chw_active_lag1", "chw_active_lag2", "chw_active_lag3"]
    pairs = pairs.dropna(subset=staff_cols).reset_index(drop=True)
    pairs["month_idx"] = (pairs["onboard_month"] - pd.Period("2023-01", freq="M")).apply(lambda x: x.n)
    pairs["month_idx2"] = pairs["month_idx"] ** 2

    pop = build_waymark_population(verbose=False)
    base_cols = ["charlson_score", "prior_hosp_6mo", "prior_ed_visits_6mo"]
    full_cols = ["age", "female", "charlson_score", "prior_ed_visits_6mo", "prior_hosp_6mo",
                 "pharmacy_fills_90d", "missed_pharmacy_fills", "has_diabetes", "has_chf",
                 "has_copd", "has_hypertension", "has_ckd", "has_mh"]
    covs = pop.patients[["member_id"] + full_cols].copy()
    pairs = pairs.merge(covs, left_on="patient_id", right_on="member_id", how="left")
    pairs = pairs.dropna(subset=full_cols).reset_index(drop=True)

    print(f"N = {len(pairs)} (of 1,707 staggered-enrollment pairs; "
          f"{sum(pairs.pair_type == 'primary')} primary, "
          f"{sum(pairs.pair_type == 'weak_positive')} weak-positive; "
          f"by state: {pairs['state'].value_counts().to_dict()})")
    y = (pairs["pair_type"] == "primary").astype(int).values

    print("\n--- T7a: narrow (T3-matched) covariate set ---")
    lr_test(pairs, y, base_cols, base_cols + staff_cols, "staffing only")
    lr_test(pairs, y, base_cols + ["month_idx"], base_cols + ["month_idx"] + staff_cols,
            "staffing + linear calendar time")
    lr_test(pairs, y, base_cols + ["month_idx", "month_idx2"],
            base_cols + ["month_idx", "month_idx2"] + staff_cols,
            "staffing + quadratic calendar time")
    lr_test(pairs, y, base_cols, base_cols + ["month_idx"], "calendar time alone (no staffing)")

    print("\n--- T7b: full covariate set (13 vars) + linear calendar time ---")
    lr_test(pairs, y, full_cols + ["month_idx"], full_cols + ["month_idx"] + staff_cols,
            "all states pooled")
    for st in ("VA", "WA"):
        sub = pairs[pairs.state == st].reset_index(drop=True)
        y_sub = (sub["pair_type"] == "primary").astype(int).values
        lr_test(sub, y_sub, full_cols + ["month_idx"], full_cols + ["month_idx"] + staff_cols,
                f"{st} only")
    sub = pairs[pairs.state.isin(["VA", "WA"])].reset_index(drop=True)
    y_sub = (sub["pair_type"] == "primary").astype(int).values
    lr_test(sub, y_sub, full_cols + ["month_idx"], full_cols + ["month_idx"] + staff_cols,
            "VA+WA pooled, excluding OH")


if __name__ == "__main__":
    main()
