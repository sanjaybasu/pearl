#!/usr/bin/env python3
"""
PEARL v6 Extended Analyses:
  1. Daily pre-t0 acute-care recency adjustment (7, 14, 30, 90 days + 90d IP) on
     100%-documented Validated Cohort A (encounter discipline, N=5,927, B=200 boot +
     full subgroup table + matched-capacity DCA + post-acute vs stable stratification)
     and Validated Cohort B (documented care-plan goals, N=2,515, B=200 boot).
  2. Strictly pre-t0 time-stamped covariates sensitivity (zero cross-sectional
     snapshot comorbidities or snapshot risk scores) across all three cohorts.
  3. Camden Coalition illustrative simulation extracting both arm-specific
     simulated acute-care rates (uniform protocol vs shared-model personalized)
     plus 90% prediction intervals.

Merges cleanly into v6_master_results.json in PEARL_OUTPUT_BASE.
"""
import os
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data.extract_wpad import (
    GOAL_MAP,
    _build_camden_stratum,
    _compute_90d_outcome,
    _compute_prior_utilization,
    _load_raw,
    build_waymark_population,
)
from models.imi_estimator import CamdenReanalysis, IMIEstimator
from revision_analyses import SEED, disable_unused_crossval, load_eligibility_attributes
import pearl_v6_revision as v6

OUT_DIR = Path(os.environ.get("PEARL_OUTPUT_BASE", str(_ROOT / "outputs")))
RAW_DIR = Path(os.environ.get("PEARL_DATA_PATH", str(_ROOT / "data" / "real_inputs")))
SVI_PATH = Path(os.environ.get("PEARL_SVI_PATH", str(_ROOT / "data" / "processed" / "svi_by_zip.parquet")))
RECENCY_COLS = ["pre_acute_7d", "pre_acute_14d", "pre_acute_30d", "pre_acute_90d", "pre_ip_90d"]


def compute_daily_pre_acute(df: pd.DataFrame, t0_col: str, hv_ei: pd.DataFrame) -> pd.DataFrame:
    """Compute exact daily acute-visit counts strictly BEFORE t0 over 7, 14, 30, and 90 days."""
    ref = pd.DataFrame({
        "patient_id": df["member_id"].values,
        "t0": pd.to_datetime(df[t0_col], utc=True).values,
        "_row": np.arange(len(df)),
    })
    ref["t0"] = pd.to_datetime(ref["t0"], utc=True)
    merged = ref.merge(hv_ei[["patient_id", "admit_date", "patient_class_code"]], on="patient_id", how="left")
    merged["days_before"] = (merged["t0"] - merged["admit_date"]).dt.total_seconds() / 86400.0
    pre = merged[(merged["days_before"] > 0) & (merged["days_before"] <= 180)]
    out = pd.DataFrame(index=np.arange(len(df)))
    for win in (7, 14, 30, 90):
        cnt = pre[pre["days_before"] <= win].groupby("_row").size().reindex(np.arange(len(df))).fillna(0)
        out[f"pre_acute_{win}d"] = cnt.values.astype(float)
        out[f"had_acute_{win}d"] = (cnt.values > 0).astype(int)
    pre_ip90 = pre[(pre["days_before"] <= 90) & (pre["patient_class_code"] == "I")].groupby("_row").size().reindex(np.arange(len(df))).fillna(0)
    out["pre_ip_90d"] = pre_ip90.values.astype(float)
    return out


def build_features_with_recency(df: pd.DataFrame, include_site_time: bool = True) -> np.ndarray:
    base = v6._orig_build_features(df, include_site_time=include_site_time)
    rec = df[RECENCY_COLS].fillna(0.0).values.astype(float)
    return np.hstack([base, rec])


def build_features_strict_timestamped(df: pd.DataFrame, include_site_time: bool = True) -> np.ndarray:
    """100% strictly pre-t0 time-stamped or time-invariant covariates only (zero snapshots)."""
    ts_cols = [
        "age", "female", "prior_ed_visits_12mo", "prior_ip_12mo",
        "total_paid_12mo", "pharmacy_paid_12mo", "prior_ed_visits_6mo",
        "prior_hosp_6mo", "month_num",
    ] + RECENCY_COLS
    X = df[ts_cols].fillna(0.0).values.astype(float)
    rc = pd.Categorical(df["race_eth"].fillna("unknown")).codes.reshape(-1, 1).astype(float)
    st = (
        pd.get_dummies(df["state"].fillna("Unlinked"), drop_first=False)
        .reindex(columns=["VA", "WA", "OH", "Unlinked"], fill_value=0)
        .values.astype(float)
    )
    return np.hstack([X, rc, st])


def run_strict_ts_summary(df: pd.DataFrame, action_col: str, outcome_col: str, min_n: int, label: str) -> dict:
    """Quick leak-free sensitivity check using 100% strictly pre-t0 time-stamped covariates only."""
    acts = sorted(df[action_col].unique())
    le = LabelEncoder().fit(acts)
    k = len(acts)
    a_all = le.transform(df[action_col].values)
    y_all = df[outcome_col].values.astype(float)
    tr, te = train_test_split(df, test_size=0.20, random_state=SEED, stratify=v6.safe_stratify(a_all, y_all))
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    X_tr = build_features_strict_timestamped(tr)
    X_te = build_features_strict_timestamped(te)
    y_tr, y_te = tr[outcome_col].values.astype(float), te[outcome_col].values.astype(float)
    a_tr, a_te = le.transform(tr[action_col].values), le.transform(te[action_col].values)
    counts_tr = np.bincount(a_tr, minlength=k)

    lam_star, _ = v6.nested_select_lambda(X_tr, y_tr, a_tr, k, min_n, rf_n_jobs=v6.N_JOBS)
    ms, ss, pr = v6.fit_bootstrap_ensemble(X_tr, y_tr, a_tr, X_te, k, "hgb", v6.M_ENSEMBLE, SEED + 101)
    me, se, _ = v6.fit_bootstrap_ensemble(X_tr, y_tr, a_tr, X_te, k, "rf", v6.M_ENSEMBLE, SEED + 202, rf_n_jobs=v6.N_JOBS)
    adm = (pr >= 0.01) & (counts_tr >= min_n)[None, :]
    n_te = len(te)

    out = {
        "label": label,
        "n_total": int(len(df)),
        "n_test": int(n_te),
        "auroc_sel": round(float(roc_auc_score(y_te, ms[np.arange(n_te), a_te])), 3),
        "auroc_ev": round(float(roc_auc_score(y_te, me[np.arange(n_te), a_te])), 3),
        "agreement_gbm_rf_pct": round(float((np.where(adm, ms, np.inf).argmin(1) == np.where(adm, me, np.inf).argmin(1)).mean() * 100.0), 1),
        "lambda_selected": lam_star,
        "by_lambda": {},
    }
    v_base = me[np.arange(n_te), a_te].mean()
    for lam in (0.0, 0.5, 1.0):
        ap, sw = v6.apply_policy(ms, ss, adm, a_te, lam, v6.EPSILON)
        mc = float(v6.compute_misalignment(me, se, adm, a_te, lam, v6.EPSILON).mean() * 100.0)
        mp = float(v6.compute_misalignment(me, se, adm, ap, lam, v6.EPSILON).mean() * 100.0)
        av = float((v_base - me[np.arange(n_te), ap].mean()) * 1000.0)
        out["by_lambda"][str(lam)] = {
            "pct_switched": round(float(100.0 * sw.mean()), 2),
            "misalign_baseline_pct": round(mc, 2),
            "misalign_pearl_pct": round(mp, 2),
            "events_averted_per_1000": round(av, 2),
        }
    print(f"  [Strict pre-t0 TS only] {label}: AUROC {out['auroc_sel']}/{out['auroc_ev']}, agree {out['agreement_gbm_rf_pct']}%, lam0={out['by_lambda']['0.0']}, lam0.5={out['by_lambda']['0.5']}")
    return out


def main():
    t0 = time.time()
    disable_unused_crossval()
    d = _load_raw()
    pop = build_waymark_population(verbose=False)
    rising = pop.patients.reset_index(drop=True)

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

    # Primary complete-90d rising-risk cohort + daily pre-index acute recency
    comp = rising[(idx_dt >= hv_min) & (idx_dt + pd.Timedelta(days=90) <= hv_max)].reset_index(drop=True)
    pre_comp = compute_daily_pre_acute(comp, "index_date", hv_ei)
    for col in pre_comp.columns:
        comp[col] = pre_comp[col].values

    # Validated Cohort A: Completed-encounter clinician discipline (N = 5,927) + daily pre-encounter recency
    enc = pd.read_parquet(RAW_DIR / "encounters.parquet")
    pm = d["member_patient_map"][["patient_id", "member_id"]].drop_duplicates()
    ws = pd.read_parquet(RAW_DIR / "waymarker_specialty.parquet")[["waymarker_id", "discipline"]]
    enc_yes = (
        enc[enc["encounter_occurred"] == "YES"]
        .merge(pm, on="patient_id", how="left")
        .merge(ws, left_on="created_by_waymarker_id", right_on="waymarker_id", how="left")
    )
    enc_yes["created_at"] = pd.to_datetime(enc_yes["created_at"], utc=True, errors="coerce")
    enc_yes["disc_action"] = enc_yes["discipline"].map(v6.map_discipline)
    enc_yes = enc_yes.dropna(subset=["disc_action"])
    first_e = enc_yes.sort_values("created_at").drop_duplicates("member_id")[["member_id", "disc_action", "created_at"]]
    enc_rr = rising.merge(first_e, on="member_id", how="inner")
    enc_rr = enc_rr[(enc_rr["created_at"] >= hv_min) & (enc_rr["created_at"] + pd.Timedelta(days=90) <= hv_max)].reset_index(drop=True)
    enc_rr["y_enc_90d"] = _compute_90d_outcome(enc_rr["member_id"], enc_rr["created_at"], hv, 90).values
    util_e = _compute_prior_utilization(enc_rr["person_id"], enc_rr["created_at"], d["outcomes_monthly"])
    for col in util_e.columns:
        enc_rr[col] = util_e[col].values
    enc_rr["month_num"] = (enc_rr["created_at"].dt.tz_localize(None) - pd.Timestamp("2023-01-01")).dt.days / 30.4375
    pre_enc = compute_daily_pre_acute(enc_rr, "created_at", hv_ei)
    for col in pre_enc.columns:
        enc_rr[col] = pre_enc[col].values

    # Validated Cohort B: Documented care-plan goals (N = 2,515) + daily pre-goal recency
    mg = d["member_goals"].copy()
    mg["goal_created_at"] = pd.to_datetime(mg["goal_created_at"], utc=True, errors="coerce")
    mg_nd = mg[(mg["category"] != "DEFAULT") & (mg["deleted"] == False)].copy()
    mg_nd["action"] = mg_nd["category"].map(GOAL_MAP)
    mg_nd = mg_nd.dropna(subset=["action"])
    doc_first = mg_nd.sort_values("goal_created_at").drop_duplicates("member_id")[
        ["member_id", "action", "goal_created_at"]
    ].rename(columns={"action": "first_goal_action", "goal_created_at": "first_goal_at"})
    doc_rr = rising.merge(doc_first, on="member_id", how="inner")
    doc_rr = doc_rr[(doc_rr["first_goal_at"] >= hv_min) & (doc_rr["first_goal_at"] + pd.Timedelta(days=90) <= hv_max)].reset_index(drop=True)
    doc_rr["y_goal_90d"] = _compute_90d_outcome(doc_rr["member_id"], doc_rr["first_goal_at"], hv, 90).values
    util_g = _compute_prior_utilization(doc_rr["person_id"], doc_rr["first_goal_at"], d["outcomes_monthly"])
    for col in util_g.columns:
        doc_rr[col] = util_g[col].values
    doc_rr["month_num"] = (doc_rr["first_goal_at"].dt.tz_localize(None) - pd.Timestamp("2023-01-01")).dt.days / 30.4375
    pre_doc = compute_daily_pre_acute(doc_rr, "first_goal_at", hv_ei)
    for col in pre_doc.columns:
        doc_rr[col] = pre_doc[col].values

    # ── 1. Acute-Recency Confounding & Clinical-State Stratification Tables ──
    print("\n--- 1. Acute-recency balance & post-acute vs stable stratification ---")
    recency_by_disc = []
    for act, grp in enc_rr.groupby("disc_action"):
        recency_by_disc.append({
            "discipline": act,
            "n": int(len(grp)),
            "events_90d": int(grp["y_enc_90d"].sum()),
            "post90_rate_pct": round(100.0 * float(grp["y_enc_90d"].mean()), 2),
            "had_acute_7d_pct": round(100.0 * float(grp["had_acute_7d"].mean()), 1),
            "had_acute_14d_pct": round(100.0 * float(grp["had_acute_14d"].mean()), 1),
            "had_acute_30d_pct": round(100.0 * float(grp["had_acute_30d"].mean()), 1),
            "had_acute_90d_pct": round(100.0 * float(grp["had_acute_90d"].mean()), 1),
            "mean_pre_acute_90d": round(float(grp["pre_acute_90d"].mean()), 2),
        })
    strat_30d = []
    for st30 in (0, 1):
        sub_s = enc_rr[enc_rr["had_acute_30d"] == st30]
        for act, grp in sub_s.groupby("disc_action"):
            p = float(grp["y_enc_90d"].mean())
            se = float(np.sqrt(p * (1 - p) / len(grp)))
            strat_30d.append({
                "clinical_state": "Post-acute (ED/inpatient within prior 30d)" if st30 == 1 else "Stable (no ED/inpatient in prior 30d)",
                "discipline": act,
                "n": int(len(grp)),
                "events_90d": int(grp["y_enc_90d"].sum()),
                "post90_rate_pct": round(100.0 * p, 2),
                "ci95_pct": [round(100.0 * max(0, p - 1.96 * se), 2), round(100.0 * min(1, p + 1.96 * se), 2)],
            })
    pd.DataFrame(recency_by_disc).to_csv(OUT_DIR / "v6_enc_recency_balance.csv", index=False)
    pd.DataFrame(strat_30d).to_csv(OUT_DIR / "v6_enc_stratified_30d.csv", index=False)

    # ── 2. Full B=200 Bootstrap WITH Daily Pre-Encounter Acute Recency ──────
    v6._orig_build_features = v6.build_features
    v6.build_features = build_features_with_recency

    res_enc_rec, dca_enc_rec, sg_enc_rec = v6.analyze_cohort(
        enc_rr, "disc_action", "y_enc_90d", min_n=100,
        label="Validated Cohort A: encounter discipline WITH daily pre-encounter acute recency",
        include_site_time=True, run_subgroups=True, run_dca=True,
        run_prop_floor_sens=True, run_disjoint_split=True, b_boot=v6.B_OUTER_BOOT,
    )
    dca_enc_rec.to_csv(OUT_DIR / "v6_enc_recency_decision_curve.csv", index=False)
    sg_enc_rec.to_csv(OUT_DIR / "v6_enc_recency_subgroups.csv", index=False)
    pd.DataFrame(res_enc_rec["ablation"]).to_csv(OUT_DIR / "v6_enc_recency_ablation.csv", index=False)

    res_goal_rec, dca_goal_rec, _ = v6.analyze_cohort(
        doc_rr, "first_goal_action", "y_goal_90d", min_n=50,
        label="Validated Cohort B: documented goals WITH daily pre-goal acute recency",
        include_site_time=True, run_subgroups=False, run_dca=True,
        run_prop_floor_sens=True, run_disjoint_split=True, b_boot=v6.B_OUTER_BOOT,
    )
    if dca_goal_rec is not None:
        dca_goal_rec.to_csv(OUT_DIR / "v6_goal_rr_recency_decision_curve.csv", index=False)
    pd.DataFrame(res_goal_rec["ablation"]).to_csv(OUT_DIR / "v6_goal_rr_recency_ablation.csv", index=False)

    # Restore original build_features
    v6.build_features = v6._orig_build_features

    # ── 3. Strictly Pre-t0 Time-Stamped Covariates Sensitivity (Zero Snapshots) ──
    print("\n--- 3. Strictly pre-t0 time-stamped covariates sensitivity ---")
    strict_ts = {
        "validated_encounter_discipline": run_strict_ts_summary(enc_rr, "disc_action", "y_enc_90d", 100, "Validated Cohort A (encounter discipline)"),
        "validated_documented_goals_rr": run_strict_ts_summary(doc_rr, "first_goal_action", "y_goal_90d", 50, "Validated Cohort B (documented goals)"),
        "primary_complete_90d_fu": run_strict_ts_summary(comp, "behavioral_intervention", "y_behavioral", 100, "Primary rising-risk cohort (hybrid)"),
    }

    # ── 4. Fresh Camden Coalition Analog Simulation (Both Arm Rates) ────────
    print("\n--- 4. Camden Coalition analog simulation (both arms) ---")
    tr_c, _ = train_test_split(rising, test_size=0.20, random_state=SEED, stratify=rising["behavioral_intervention"])
    camden_pts = _build_camden_stratum(rising)
    est = IMIEstimator().fit(tr_c)
    camden_res = CamdenReanalysis().run(est, camden_pts, threshold=0.02)
    # Extract exact per-arm simulated rates with 1,000 bootstrap 90% PIs
    mu_cam = est._predict_outcomes(est._get_feature_matrix(camden_pts))
    care_idx = list(est._le.classes_).index("care_access")
    r_camden_arm = mu_cam[:, care_idx]
    r_pearl_arm = mu_cam.min(axis=1)
    rng_c = np.random.default_rng(SEED)
    b_cam, b_prl, b_dif = [], [], []
    for _ in range(1000):
        idx_b = rng_c.integers(0, len(camden_pts), len(camden_pts))
        rc = float(r_camden_arm[idx_b].mean())
        rp = float(r_pearl_arm[idx_b].mean())
        b_cam.append(rc); b_prl.append(rp); b_dif.append(rc - rp)
    camden_full = {
        "n_camden_profile": int(len(camden_pts)),
        "imi_camden_uniform_pct": round(100.0 * float(camden_res["imi_camden_protocol"]), 1),
        "uniform_protocol_rate_pct": round(100.0 * float(r_camden_arm.mean()), 2),
        "uniform_protocol_pi90_pct": [round(100.0 * float(np.percentile(b_cam, 5)), 2), round(100.0 * float(np.percentile(b_cam, 95)), 2)],
        "shared_personalized_rate_pct": round(100.0 * float(r_pearl_arm.mean()), 2),
        "shared_personalized_pi90_pct": [round(100.0 * float(np.percentile(b_prl, 5)), 2), round(100.0 * float(np.percentile(b_prl, 95)), 2)],
        "diff_pp": round(100.0 * float(r_camden_arm.mean() - r_pearl_arm.mean()), 1),
        "diff_pi90_pp": [round(100.0 * float(np.percentile(b_dif, 5)), 1), round(100.0 * float(np.percentile(b_dif, 95)), 1)],
        "redistribution": camden_res["pearl_intervention_redistribution"],
    }
    print("  Camden rates:", {k: v for k, v in camden_full.items() if k != "redistribution"})

    # ── Merge into v6_master_results.json cleanly ───────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master = json.load(open(OUT_DIR / "v6_master_results.json"))
    master["enc_recency_balance"] = recency_by_disc
    master["enc_stratified_30d"] = strat_30d
    master["cohorts"]["validated_encounter_discipline_recency_adj"] = res_enc_rec
    master["cohorts"]["validated_documented_goals_rr_recency_adj"] = res_goal_rec
    master["strict_pre_t0_timestamped_sensitivity"] = strict_ts
    master["camden_illustrative"] = camden_full

    with open(OUT_DIR / "v6_master_results.json", "w") as f:
        json.dump(master, f, indent=2, default=str)
    print(f"\nExtended v6 analyses merged into {OUT_DIR / 'v6_master_results.json'} in {(time.time()-t0)/60:.2f} min.")


if __name__ == "__main__":
    main()
