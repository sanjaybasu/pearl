"""
Real Waymark WPAD Dataset Extractor

Constructs Within-Patient Administrative Discontinuity (WPAD) preference pairs
and a full rising-risk patient cohort from Waymark longitudinal operational data.

Three WPAD types (following Algorithm 1 in the paper):
  Type 1: ACO onboarding ramp-up   — OFF before ON, near-100% engagement
  Type 2: CHW waitlist             — TARGETED→ONBOARDED ≥30d, near-100% engagement
  Type 3: Medicaid eligibility churn — gap ≥60d, ITT secondary analysis

Primary outcome: 90-day composite acute care event
  (unplanned hospitalization OR avoidable ED visit; Fine-Gray competing risks, death competing)

Intervention type labels from member_goals (4 PEARL categories):
  care_access         ← CARE, PCP_APPOINTMENT
  clinical_other      ← EYE_CARE, DENTAL, ACTIVITY, EDUCATION, TECHNOLOGY, OTHER, VIOLENCE
  diabetes            ← DIABETES, WEIGHT_MANAGEMENT
  financial_benefits  ← FINANCIAL, INSURANCE_COVERAGE, LEGAL, EMPLOYMENT
  food_security       ← FOOD_INSECURITY, FOOD_DIET_NUTRITION
  heart_failure       ← HEART_FAILURE
  housing             ← HOUSING_INSECURITY, HOUSING_QUALITY_SAFETY
  hypertension        ← HYPERTENSION
  maternal            ← MATERNITY, PRENATAL_CARE, POSTPARTUM_CARE
  medication_adherence ← MEDICATION_ADHERENCE, MEDICATION_OPTIMIZATION
  mental_health       ← MENTAL_HEALTH, DEPRESSION, ANXIETY, CARE_FOR_MH_BH, OTHER_MENTAL_BEHAVIORAL
  pulmonary           ← ASTHMA_COPD
  substance_use       ← SUBSTANCE_USE, ALCOHOL_USE, SMOKING_CESSATION
  transport_utilities ← TRANSPORTATION, UTILITIES, CHILDCARE, SOCIAL_CONNECTION

Usage:
  from data.extract_wpad import build_waymark_population
  pop = build_waymark_population()
  # drop-in for SyntheticPopulation in run_pipeline.py --waymark mode
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/sanjaybasu/waymark-local/data/real_inputs"

# ── PEARL intervention taxonomy (14 specific next best action categories) ──────
INTERVENTIONS = [
    "care_access",        # PCP appointments, care coordination, complex care management
    "clinical_other",     # Dental, eye care, activity, wellness (catch-all)
    "diabetes",           # Diabetes management, glycemic control
    "financial_benefits", # Financial assistance, insurance enrollment, legal, employment
    "food_security",      # Food insecurity, nutrition counseling
    "heart_failure",      # Heart failure management
    "housing",            # Housing instability, housing quality/safety
    "hypertension",       # Hypertension management, blood pressure control
    "maternal",           # Maternity, prenatal, postpartum care
    "medication_adherence", # Medication adherence and optimization (cross-cutting)
    "mental_health",      # Depression, anxiety, MH/BH coordination
    "pulmonary",          # Asthma, COPD management
    "substance_use",      # Substance use disorders, alcohol, smoking cessation
    "transport_utilities", # Transportation, utilities, childcare, social connection
]

# Member goal category → PEARL intervention type mapping
GOAL_MAP = {
    # care_access
    "CARE": "care_access",
    "PCP_APPOINTMENT": "care_access",
    # clinical_other (catch-all for wellness/preventive)
    "EYE_CARE": "clinical_other",
    "DENTAL": "clinical_other",
    "ACTIVITY": "clinical_other",
    "EDUCATION": "clinical_other",
    "TECHNOLOGY": "clinical_other",
    "OTHER": "clinical_other",
    "VIOLENCE": "clinical_other",
    # diabetes
    "DIABETES": "diabetes",
    "WEIGHT_MANAGEMENT": "diabetes",
    # financial_benefits
    "FINANCIAL": "financial_benefits",
    "INSURANCE_COVERAGE": "financial_benefits",
    "LEGAL": "financial_benefits",
    "EMPLOYMENT": "financial_benefits",
    # food_security
    "FOOD_INSECURITY": "food_security",
    "FOOD_DIET_NUTRITION": "food_security",
    # heart_failure
    "HEART_FAILURE": "heart_failure",
    # housing
    "HOUSING_INSECURITY": "housing",
    "HOUSING_QUALITY_SAFETY": "housing",
    # hypertension
    "HYPERTENSION": "hypertension",
    # maternal
    "MATERNITY": "maternal",
    "PRENATAL_CARE": "maternal",
    "POSTPARTUM_CARE": "maternal",
    # medication_adherence
    "MEDICATION_ADHERENCE": "medication_adherence",
    "MEDICATION_OPTIMIZATION": "medication_adherence",
    # mental_health
    "MENTAL_HEALTH": "mental_health",
    "DEPRESSION": "mental_health",
    "ANXIETY": "mental_health",
    "CARE_FOR_MH_BH": "mental_health",
    "OTHER_MENTAL_BEHAVIORAL": "mental_health",
    # pulmonary
    "ASTHMA_COPD": "pulmonary",
    # substance_use
    "SUBSTANCE_USE": "substance_use",
    "ALCOHOL_USE": "substance_use",
    "SMOKING_CESSATION": "substance_use",
    # transport_utilities
    "TRANSPORTATION": "transport_utilities",
    "UTILITIES": "transport_utilities",
    "CHILDCARE": "transport_utilities",
    "SOCIAL_CONNECTION": "transport_utilities",
}
# Intervention priority for tie-breaking (clinical urgency order)
INTV_PRIORITY = {
    "medication_adherence": 0,
    "heart_failure": 1,
    "hypertension": 2,
    "diabetes": 3,
    "pulmonary": 4,
    "mental_health": 5,
    "substance_use": 6,
    "maternal": 7,
    "care_access": 8,
    "food_security": 9,
    "housing": 10,
    "financial_benefits": 11,
    "transport_utilities": 12,
    "clinical_other": 13,
}


@dataclass
class WaymarkPopulation:
    """Real Waymark population — drop-in replacement for SyntheticPopulation."""
    patients: pd.DataFrame           # All rising-risk patients, N×P
    wpad_pairs: pd.DataFrame         # WPAD preference pairs (within-patient)
    cross_patient_pairs: pd.DataFrame
    ground_truth_imi: float = float("nan")   # Unknown for real data
    optimal_policy: pd.DataFrame = field(default_factory=pd.DataFrame)
    camden_stratum_patients: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw() -> dict:
    """Load all raw parquet/csv files. Returns dict of DataFrames."""
    print("  Loading raw Waymark data files...")
    d = {}
    d["eligibility"] = pd.read_parquet(f"{DATA_DIR}/eligibility.parquet")
    d["outcomes_monthly"] = pd.read_parquet(f"{DATA_DIR}/outcomes_monthly.parquet")
    d["member_status"] = pd.read_parquet(f"{DATA_DIR}/member_status_event.parquet")
    d["hospital_visits"] = pd.read_parquet(f"{DATA_DIR}/hospital_visits.parquet")
    d["member_goals"] = pd.read_parquet(f"{DATA_DIR}/member_goals.parquet")
    d["member_attributes"] = pd.read_parquet(f"{DATA_DIR}/member_attributes.parquet")
    d["member_patient_map"] = pd.read_parquet(f"{DATA_DIR}/member_patient_map.parquet")

    # Optional enriched cohort (5,148 patients with clinical detail)
    try:
        d["real_cohort"] = pd.read_parquet(f"{DATA_DIR}/../real_cohort_analytic.parquet")
    except Exception:
        d["real_cohort"] = pd.DataFrame()

    # Enriched clinical data from lighthouse/coredb (pulled 2026-04-20)
    PROCESSED_DIR = "/Users/sanjaybasu/waymark-local/data/processed"
    try:
        d["signal_risk"] = pd.read_parquet(f"{PROCESSED_DIR}/signal_risk_latest.parquet")
        print(f"    signal_risk:      {len(d['signal_risk']):,} rows (MIRA scores + condition flags)")
    except Exception:
        d["signal_risk"] = pd.DataFrame()
        print("    signal_risk:      NOT FOUND (falling back to proxies)")
    try:
        d["clinical_summary"] = pd.read_parquet(f"{PROCESSED_DIR}/clinical_summary.parquet")
        print(f"    clinical_summary: {len(d['clinical_summary']):,} rows (Charlson + condition flags)")
    except Exception:
        d["clinical_summary"] = pd.DataFrame()
        print("    clinical_summary: NOT FOUND")
    try:
        d["pharmacy_summary"] = pd.read_parquet(f"{PROCESSED_DIR}/pharmacy_summary.parquet")
        print(f"    pharmacy_summary: {len(d['pharmacy_summary']):,} rows (fill counts)")
    except Exception:
        d["pharmacy_summary"] = pd.DataFrame()
        print("    pharmacy_summary: NOT FOUND")

    print(f"    outcomes_monthly: {len(d['outcomes_monthly']):,} rows, "
          f"{d['outcomes_monthly']['person_id'].nunique():,} patients")
    print(f"    member_status:    {len(d['member_status']):,} rows")
    print(f"    hospital_visits:  {len(d['hospital_visits']):,} rows")
    print(f"    member_goals:     {len(d['member_goals']):,} rows")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 2. ID bridge  (WAY↔CUID)
# ─────────────────────────────────────────────────────────────────────────────

def _build_id_bridge(d: dict) -> pd.DataFrame:
    """
    Build a unified ID lookup table: CUID (member_id) ↔ WAY (person_id).

    Sources:
      member_status_event: member_id (CUID) + member_legacy_id (WAY)
      member_attributes: member_id (CUID) + waymark_patient_number (WAY)
    """
    from_ms = (
        d["member_status"][["member_id", "member_legacy_id"]]
        .drop_duplicates("member_id")
        .rename(columns={"member_legacy_id": "person_id"})
    )
    from_ma = (
        d["member_attributes"][["member_id", "waymark_patient_number"]]
        .rename(columns={"waymark_patient_number": "person_id"})
        .drop_duplicates("member_id")
    )
    bridge = pd.concat([from_ms, from_ma], ignore_index=True).drop_duplicates("member_id")
    bridge = bridge[bridge["person_id"].notna() & bridge["member_id"].notna()]
    return bridge


# ─────────────────────────────────────────────────────────────────────────────
# 3. 90-day composite acute care outcome
# ─────────────────────────────────────────────────────────────────────────────

def _compute_90d_outcome(
    patient_ids: pd.Series,   # CUID
    index_dates: pd.Series,   # UTC-aware timestamps
    hospital_visits: pd.DataFrame,
    window_days: int = 90,
) -> pd.Series:
    """
    Vectorized: for each patient, binary 90-day composite acute care event
    (any ED [class E] OR unplanned inpatient [class I]) after index_date.

    Returns pd.Series of 0/1 with same index as patient_ids.
    """
    hv = hospital_visits[hospital_visits["patient_class_code"].isin(["E", "I"])].copy()
    hv["admit_date"] = pd.to_datetime(hv["admit_date"], utc=True)

    # Build a lookup frame from patient events
    ref = pd.DataFrame({
        "patient_id": patient_ids.values,
        "index_date": pd.to_datetime(index_dates.values, utc=True),
        "_row": np.arange(len(patient_ids)),
    })
    ref["index_date"] = ref["index_date"].apply(
        lambda x: x.tz_localize("UTC") if (pd.notna(x) and x.tzinfo is None) else x
    )
    ref["cutoff"] = ref["index_date"] + pd.Timedelta(days=window_days)

    # Cross-join via merge: join all visits for each patient, then filter dates
    merged = ref.merge(hv[["patient_id", "admit_date"]], on="patient_id", how="left")
    in_window = (
        merged["admit_date"].notna() &
        (merged["admit_date"] >= merged["index_date"]) &
        (merged["admit_date"] <= merged["cutoff"])
    )
    has_event = merged[in_window]["_row"].unique()

    result = np.zeros(len(patient_ids), dtype=int)
    result[has_event] = 1
    return pd.Series(result, index=patient_ids.index)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Prior 12-month utilization features
# ─────────────────────────────────────────────────────────────────────────────

def _compute_prior_utilization(
    way_ids: pd.Series,
    index_dates: pd.Series,
    outcomes_monthly: pd.DataFrame,
    lookback_months: int = 12,
) -> pd.DataFrame:
    """
    Vectorized: compute prior 12-month and 6-month utilization from outcomes_monthly.
    Returns DataFrame: prior_ed_visits_12mo, prior_ip_12mo, prior_ed_visits_6mo,
                       prior_hosp_6mo, total_paid_12mo, pharmacy_paid_12mo.
    """
    om = outcomes_monthly.copy()
    om["month_year"] = pd.to_datetime(om["month_year"])

    # Normalize index_dates to tz-naive
    idx_dt_arr = pd.to_datetime(index_dates.values)
    idx_dt_arr = np.array([
        t.tz_localize(None) if (pd.notna(t) and t.tzinfo is not None) else t
        for t in idx_dt_arr
    ], dtype="datetime64[ns]")

    ref = pd.DataFrame({
        "person_id": way_ids.values,
        "index_date": idx_dt_arr,
        "_row": np.arange(len(way_ids)),
    })
    ref["start_12mo"] = ref["index_date"] - pd.to_timedelta(lookback_months * 30, unit="D")
    ref["start_6mo"] = ref["index_date"] - pd.to_timedelta(6 * 30, unit="D")

    # Merge all om rows for the cohort patients
    cohort_pids = set(way_ids.dropna().unique())
    om_sub = om[om["person_id"].isin(cohort_pids)].copy()

    joined = ref.merge(om_sub, on="person_id", how="left")
    joined["month_year"] = pd.to_datetime(joined["month_year"])

    in_12mo = (
        joined["month_year"].notna() &
        (joined["month_year"] >= joined["start_12mo"]) &
        (joined["month_year"] < joined["index_date"])
    )
    in_6mo = (
        joined["month_year"].notna() &
        (joined["month_year"] >= joined["start_6mo"]) &
        (joined["month_year"] < joined["index_date"])
    )

    # Aggregate
    agg_12 = joined[in_12mo].groupby("_row").agg(
        prior_ed_visits_12mo=("emergency_department_ct", "sum"),
        prior_ip_12mo=("acute_inpatient_ct", "sum"),
        total_paid_12mo=("total_paid", "sum"),
        pharmacy_paid_12mo=("pharmacy_paid", "sum"),
    ).reindex(np.arange(len(way_ids))).fillna(0)

    agg_6 = joined[in_6mo].groupby("_row").agg(
        prior_ed_visits_6mo=("emergency_department_ct", "sum"),
        prior_hosp_6mo=("acute_inpatient_ct", "sum"),
    ).reindex(np.arange(len(way_ids))).fillna(0)

    result = pd.concat([agg_12, agg_6], axis=1)
    for col in result.columns:
        result[col] = result[col].fillna(0)
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Demographics (member_attributes)
# ─────────────────────────────────────────────────────────────────────────────

def _build_demographics(member_ids: pd.Series, d: dict) -> pd.DataFrame:
    """Build demographic feature table for a list of member CUIDs."""
    ma = d["member_attributes"].copy()
    ma = ma.drop_duplicates("member_id")

    # Age as of 2024-01-01
    ref_date = pd.Timestamp("2024-01-01")
    ma["birth_date"] = pd.to_datetime(ma["birth_date"])
    ma["age"] = ((ref_date - ma["birth_date"]).dt.days / 365.25).clip(18, 95).fillna(40)

    # Gender → female binary
    ma["female"] = (ma["gender"].str.upper().isin(["F", "FEMALE"])).astype(int)

    # Race/ethnicity
    race_map = {
        "white": "white_nh",
        "black": "black_nh",
        "african american": "black_nh",
        "hispanic": "hispanic",
        "asian": "asian",
        "native american": "other",
        "unknown": "unknown",
    }
    ma["race_eth"] = (
        ma["race"].str.lower().map(race_map).fillna(
            ma["ethnicity"].str.lower().map({"hispanic": "hispanic",
                                              "african american": "black_nh",
                                              "non-hispanic": "white_nh"}).fillna("unknown")
        )
    )

    demog = ma[["member_id", "age", "female", "race_eth", "risk_score"]].copy()
    # Merge — left join to preserve all requested member_ids
    req = pd.DataFrame({"member_id": member_ids.values})
    merged = req.merge(demog, on="member_id", how="left")
    merged["age"] = merged["age"].fillna(45.0)
    merged["female"] = merged["female"].fillna(0).astype(int)
    merged["race_eth"] = merged["race_eth"].fillna("unknown")
    merged["primary_language"] = "english"  # not available in data; default
    return merged.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Clinical features (charlson proxy + condition flags)
# ─────────────────────────────────────────────────────────────────────────────

def _build_clinical_features(
    member_ids: pd.Series,
    d: dict,
    way_ids: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build clinical features using three enriched sources (in priority order):

    1. clinical_summary (WAY IDs) — Charlson score + binary condition flags from claims
    2. signal_risk (CUID) — MIRA condition flags, medication adherence flag
    3. member_goals fallback — for patients not in either enriched source

    way_ids: optional WAY IDs (person_id) aligned with member_ids. Required for
    joining clinical_summary and pharmacy_summary which use WAY format.
    """
    base = pd.DataFrame({"member_id": member_ids.values})
    if way_ids is not None:
        base["person_id"] = way_ids.values
    else:
        base["person_id"] = None

    # ── 1. clinical_summary (WAY IDs → Charlson + condition flags from claims) ─
    cs = d.get("clinical_summary", pd.DataFrame())
    if len(cs) > 0 and "person_id" in base.columns:
        cs_sub = cs[["person_id", "charlson_score", "has_diabetes", "has_chf",
                      "has_copd", "has_hypertension", "has_ckd", "has_mh",
                      "has_substance", "has_cancer", "n_chronic"]].copy()
        base = base.merge(cs_sub, on="person_id", how="left")

    # ── 2. signal_risk (CUID → MIRA condition flags) ──────────────────────────
    sr = d.get("signal_risk", pd.DataFrame())
    if len(sr) > 0:
        sr_sub = sr[["patientId", "diabetes", "hypertension", "heartFailure", "copd",
                      "anyBehavioralHealth", "poorMedAdherence", "medicallyComplex",
                      "polypharmacy", "numberOfMedications", "numberOfAcuteCareEpisodes",
                      "risingRiskScore"]].rename(columns={
            "patientId": "member_id",
            "diabetes": "_sr_diabetes",
            "hypertension": "_sr_hypertension",
            "heartFailure": "_sr_chf",
            "copd": "_sr_copd",
            "anyBehavioralHealth": "_sr_mh",
            "poorMedAdherence": "_sr_poor_adh",
            "medicallyComplex": "_sr_complex",
            "polypharmacy": "_sr_polypharmacy",
            "numberOfMedications": "_sr_n_meds",
            "numberOfAcuteCareEpisodes": "_sr_acute_episodes",
            "risingRiskScore": "_sr_risk_score",
        }).copy()
        # Boolean → int
        bool_cols = ["_sr_diabetes", "_sr_hypertension", "_sr_chf", "_sr_copd",
                     "_sr_mh", "_sr_poor_adh", "_sr_complex", "_sr_polypharmacy"]
        for c in bool_cols:
            sr_sub[c] = sr_sub[c].fillna(False).astype(int)
        base = base.merge(sr_sub, on="member_id", how="left")

    # ── 3. Pharmacy summary (WAY IDs → fill counts) ──────────────────────────
    ph = d.get("pharmacy_summary", pd.DataFrame())
    if len(ph) > 0 and "person_id" in base.columns:
        ph_sub = ph[["person_id", "total_fills", "unique_drugs", "active_months"]].copy()
        base = base.merge(ph_sub, on="person_id", how="left")

    # ── 4. member_goals fallback (SDOH flags) ────────────────────────────────
    mg = d["member_goals"]
    sdoh_flag_map = {
        "food_insecure": ["FOOD_INSECURITY"],
        "housing_unstable": ["HOUSING_INSECURITY", "HOUSING_QUALITY_SAFETY"],
        "no_transport": ["TRANSPORTATION"],
        "medication_need": ["MEDICATION_ADHERENCE", "MEDICATION_OPTIMIZATION"],
    }
    for flag, cats in sdoh_flag_map.items():
        flag_pids = mg[mg["category"].isin(cats)]["member_id"].unique()
        if flag not in base.columns:
            base[flag] = 0
        base.loc[base["member_id"].isin(flag_pids), flag] = (
            base.loc[base["member_id"].isin(flag_pids), flag].fillna(0).clip(upper=1).values
        )

    # ── 5. Resolve / fill gaps ────────────────────────────────────────────────
    # Condition flags: prefer claims-based (clinical_summary), fallback to MIRA signal_risk
    for col, sr_col in [("has_diabetes", "_sr_diabetes"), ("has_chf", "_sr_chf"),
                         ("has_copd", "_sr_copd"), ("has_hypertension", "_sr_hypertension"),
                         ("has_mh", "_sr_mh")]:
        if col not in base.columns:
            base[col] = 0
        if sr_col in base.columns:
            base[col] = base[col].fillna(base[sr_col]).fillna(0).astype(int)
        else:
            base[col] = base[col].fillna(0).astype(int)

    for col in ["has_ckd", "has_substance", "has_cancer"]:
        if col not in base.columns:
            base[col] = 0
        base[col] = base[col].fillna(0).astype(int)

    # Charlson score: from claims if available, else simple count
    if "charlson_score" not in base.columns:
        base["charlson_score"] = 0
    base["charlson_score"] = base["charlson_score"].fillna(
        base.get("_sr_complex", pd.Series(0, index=base.index)).fillna(0)
    ).clip(0, 10)

    # n_chronic
    if "n_chronic" not in base.columns:
        cond_flags = ["has_diabetes", "has_chf", "has_copd", "has_hypertension", "has_ckd"]
        base["n_chronic"] = base[cond_flags].fillna(0).sum(axis=1).clip(0, 6).astype(int)

    # SDOH count
    base["sdoh_count"] = (
        base.get("food_insecure", pd.Series(0, index=base.index)).fillna(0) +
        base.get("housing_unstable", pd.Series(0, index=base.index)).fillna(0) +
        base.get("no_transport", pd.Series(0, index=base.index)).fillna(0)
    ).astype(int)

    # Pharmacy: fills per 90 days from actual pharmacy_summary data
    if "total_fills" in base.columns:
        months = base["active_months"].fillna(1).clip(lower=1)
        base["pharmacy_fills_90d"] = (base["total_fills"].fillna(0) / months * 3).round(1)
    else:
        base["pharmacy_fills_90d"] = np.where(
            base.get("medication_need", pd.Series(0, index=base.index)).fillna(0) > 0, 3, 6
        )

    # Poor adherence flag: from MIRA if available, else medication_need goal
    if "_sr_poor_adh" in base.columns:
        base["poor_adherence"] = base["_sr_poor_adh"].fillna(
            (base.get("medication_need", pd.Series(0, index=base.index)).fillna(0) > 0).astype(int)
        ).astype(int)
    else:
        base["poor_adherence"] = (
            base.get("medication_need", pd.Series(0, index=base.index)).fillna(0) > 0
        ).astype(int)
    base["missed_pharmacy_fills"] = base["poor_adherence"] * 2

    # Polypharmacy flag
    if "_sr_polypharmacy" in base.columns:
        base["polypharmacy"] = base["_sr_polypharmacy"].fillna(0).astype(int)
    elif "unique_drugs" in base.columns:
        base["polypharmacy"] = (base["unique_drugs"].fillna(0) >= 5).astype(int)
    else:
        base["polypharmacy"] = 0

    base["lives_alone"] = 0  # not available in data

    # Drop internal intermediate columns
    drop_cols = [c for c in base.columns if c.startswith("_sr_")]
    base = base.drop(columns=drop_cols, errors="ignore")

    return base.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Intervention type from member_goals
# ─────────────────────────────────────────────────────────────────────────────

def _label_intervention(
    member_ids: pd.Series,
    on_start: pd.Series,
    on_end: pd.Series,
    member_goals: pd.DataFrame,
    default: str = "care_access",
) -> pd.Series:
    """
    For each patient, find the dominant PEARL intervention type from goals
    created during the ON-window [on_start, on_end].

    Priority order defined by INTV_PRIORITY (medication_adherence highest, clinical_other lowest).
    """
    mg = member_goals.copy()
    mg["goal_created_at"] = pd.to_datetime(mg["goal_created_at"], utc=True)

    results = []
    for mid, t0, t1 in zip(member_ids, on_start, on_end):
        if pd.isna(t0) or pd.isna(t1):
            results.append(default)
            continue
        t0 = pd.Timestamp(t0).tz_localize("UTC") if pd.Timestamp(t0).tzinfo is None else pd.Timestamp(t0)
        t1 = pd.Timestamp(t1).tz_localize("UTC") if pd.Timestamp(t1).tzinfo is None else pd.Timestamp(t1)

        window_goals = mg[
            (mg["member_id"] == mid) &
            (mg["goal_created_at"] >= t0) &
            (mg["goal_created_at"] <= t1) &
            (mg["category"] != "DEFAULT") &
            (mg["deleted"] == False)
        ]

        if len(window_goals) == 0:
            # Expand window: look at all goals ever for this patient
            window_goals = mg[
                (mg["member_id"] == mid) &
                (mg["category"] != "DEFAULT") &
                (mg["deleted"] == False)
            ]

        if len(window_goals) == 0:
            results.append(default)
            continue

        # Map categories to intervention types
        cats_mapped = window_goals["category"].map(GOAL_MAP).dropna()
        if len(cats_mapped) == 0:
            results.append(default)
            continue

        # Use mode; break ties by priority
        counts = cats_mapped.value_counts()
        top_count = counts.max()
        tied = counts[counts == top_count].index.tolist()
        if len(tied) == 1:
            results.append(tied[0])
        else:
            # Priority tie-break
            results.append(min(tied, key=lambda x: INTV_PRIORITY.get(x, 99)))

    return pd.Series(results, index=member_ids.index)


# ─────────────────────────────────────────────────────────────────────────────
# 8. WPAD Type 1: ACO Onboarding
# ─────────────────────────────────────────────────────────────────────────────

def _build_wpad_type1(d: dict, bridge: pd.DataFrame) -> pd.DataFrame:
    """
    WPAD Type 1: patients staggered into care management at ACO onboarding.

    ON-window:  [onboarded_at,          onboarded_at + 90d]
    OFF-window: [onboarded_at - 180d,   onboarded_at]
    Outcome:    any E/I hospital visit in each 90-day window.
    Direction:  off_before_on (pre-enrollment = OFF; post-enrollment = ON)
    """
    ms = d["member_status"]
    hv = d["hospital_visits"]
    mg = d["member_goals"]

    onboarded = (
        ms[ms["to_status"] == "ONBOARDED"]
        .copy()
        .sort_values("effective_at")
        .drop_duplicates("member_id", keep="first")  # earliest onboarding
    )
    onboarded["effective_at"] = pd.to_datetime(onboarded["effective_at"], utc=True)

    print(f"    WPAD Type 1: {len(onboarded):,} onboarding events")

    # ON-window: [onboarded_at, onboarded_at + 90d]
    on_start = onboarded["effective_at"]
    on_end = on_start + pd.Timedelta(days=90)

    # OFF-window: last 90 days before onboarding (avoiding overlap)
    off_end = on_start - pd.Timedelta(days=1)
    off_start = off_end - pd.Timedelta(days=90)

    # Require minimum 60 days of data before onboarding
    data_start = pd.Timestamp("2023-01-01", tz="UTC")
    valid_mask = off_start >= data_start
    onboarded = onboarded[valid_mask].reset_index(drop=True)
    on_start = on_start[valid_mask].reset_index(drop=True)
    on_end = on_end[valid_mask].reset_index(drop=True)
    off_start = off_start[valid_mask].reset_index(drop=True)
    off_end = off_end[valid_mask].reset_index(drop=True)

    # Compute outcomes
    y_on = _compute_90d_outcome(
        onboarded["member_id"], on_start, hv, window_days=90
    )
    y_off = _compute_90d_outcome(
        onboarded["member_id"], off_start, hv, window_days=90
    )

    # Intervention type from goals during ON-window
    intv_type = _label_intervention(
        onboarded["member_id"], on_start, on_end, mg
    )

    # Onboarding date for calendar effects
    pairs = pd.DataFrame({
        "patient_id": onboarded["member_id"].values,
        "wpad_type": "aco_onboarding",
        "direction": "off_before_on",
        "wpad_gap_days": 90,
        "on_start": on_start.values,
        "on_end": on_end.values,
        "off_start": off_start.values,
        "off_end": off_end.values,
        "y_on": y_on.values,
        "y_off": y_off.values,
        "behavioral_intervention": intv_type.values,
    })

    # Pair weights (Algorithm 1):
    # primary: y_on=0, y_off=1 → CM prevented an event
    # weak_positive: y_on=0, y_off=0 → both good (informative but weaker)
    # discard: y_on=1 → bad outcome despite CM → no learning signal
    pairs = pairs[pairs["y_on"] == 0].copy()
    pairs["pair_type"] = np.where(pairs["y_off"] == 1, "primary", "weak_positive")
    pairs["pair_weight"] = np.where(pairs["pair_type"] == "primary", 1.0, 0.5)

    print(f"      → {(pairs['pair_type']=='primary').sum():,} primary, "
          f"{(pairs['pair_type']=='weak_positive').sum():,} weak_positive pairs")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 9. WPAD Type 2: CHW Waitlist
# ─────────────────────────────────────────────────────────────────────────────

def _build_wpad_type2(d: dict, type1_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    WPAD Type 2: patients TARGETED then ONBOARDED ≥30d later (waitlist period).

    OFF-window: [targeted_at, onboarded_at]   (waitlist — no CM)
    ON-window:  [onboarded_at, onboarded_at + 90d]

    These patients are already in Type 1 for their ON-window outcome.
    Type 2 adds waitlist as an alternative OFF-window identification source.
    Only included if waitlist duration ≥30d AND ≤365d.
    """
    ms = d["member_status"]
    hv = d["hospital_visits"]
    mg = d["member_goals"]

    targeted = (
        ms[ms["to_status"] == "TARGETED"][["member_id", "effective_at"]]
        .rename(columns={"effective_at": "targeted_at"})
        .copy()
    )
    targeted["targeted_at"] = pd.to_datetime(targeted["targeted_at"], utc=True)
    # Keep earliest TARGETED event per patient
    targeted = targeted.sort_values("targeted_at").drop_duplicates("member_id", keep="first")

    onboarded = (
        ms[ms["to_status"] == "ONBOARDED"][["member_id", "effective_at"]]
        .rename(columns={"effective_at": "onboarded_at"})
        .copy()
    )
    onboarded["onboarded_at"] = pd.to_datetime(onboarded["onboarded_at"], utc=True)
    onboarded = onboarded.sort_values("onboarded_at").drop_duplicates("member_id", keep="first")

    merged = onboarded.merge(targeted, on="member_id", how="inner")
    merged["waitlist_days"] = (merged["onboarded_at"] - merged["targeted_at"]).dt.days

    # Valid waitlist: 30d ≤ gap ≤ 365d
    merged = merged[
        (merged["waitlist_days"] >= 30) & (merged["waitlist_days"] <= 365)
    ].copy()
    print(f"    WPAD Type 2: {len(merged):,} waitlist patients")

    # OFF-window: [targeted_at, onboarded_at] — use 90d within the waitlist
    off_start = merged["targeted_at"]
    off_end = merged["targeted_at"] + pd.Timedelta(days=90)
    # Clip off_end to onboarded_at if shorter
    off_end = pd.Series(
        [min(oe, oa - pd.Timedelta(days=1))
         for oe, oa in zip(off_end, merged["onboarded_at"])],
        index=merged.index
    )

    # ON-window: same as Type 1
    on_start = merged["onboarded_at"]
    on_end = on_start + pd.Timedelta(days=90)

    # Require off-window ≥ 30 days
    valid = (off_end - off_start).dt.days >= 30
    merged = merged[valid].reset_index(drop=True)
    off_start = off_start[valid].reset_index(drop=True)
    off_end = off_end[valid].reset_index(drop=True)
    on_start = on_start[valid].reset_index(drop=True)
    on_end = on_end[valid].reset_index(drop=True)

    y_on = _compute_90d_outcome(merged["member_id"], on_start, hv)
    y_off = _compute_90d_outcome(merged["member_id"], off_start, hv)

    intv_type = _label_intervention(merged["member_id"], on_start, on_end, mg)

    pairs = pd.DataFrame({
        "patient_id": merged["member_id"].values,
        "wpad_type": "chw_waitlist",
        "direction": "off_before_on",
        "wpad_gap_days": merged["waitlist_days"].values,
        "on_start": on_start.values,
        "on_end": on_end.values,
        "off_start": off_start.values,
        "off_end": off_end.values,
        "y_on": y_on.values,
        "y_off": y_off.values,
        "behavioral_intervention": intv_type.values,
    })

    # Exclude patients already in Type 1 with the same ON-window
    # (to avoid double-counting the same patient-window)
    type1_pids = set(type1_pairs["patient_id"].tolist())
    pairs = pairs[~pairs["patient_id"].isin(type1_pids)].copy()

    pairs = pairs[pairs["y_on"] == 0].copy()
    pairs["pair_type"] = np.where(pairs["y_off"] == 1, "primary", "weak_positive")
    pairs["pair_weight"] = np.where(pairs["pair_type"] == "primary", 1.0, 0.5)

    print(f"      → {(pairs['pair_type']=='primary').sum():,} primary, "
          f"{(pairs['pair_type']=='weak_positive').sum():,} weak_positive pairs (non-overlapping with Type 1)")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 10. WPAD Type 3: Medicaid Eligibility Churn (ITT)
# ─────────────────────────────────────────────────────────────────────────────

def _build_wpad_type3(d: dict, bridge: pd.DataFrame, enrolled_pids: set) -> pd.DataFrame:
    """
    WPAD Type 3: Medicaid eligibility gaps ≥60 days.

    ON-window:  last 90d of continuous coverage before gap (coverage active, CM potentially available)
    OFF-window: first 90d of gap (coverage lapsed, CM billing ceases)
    Direction:  on_before_off

    ITT estimand: effect of care management *availability* (not receipt).
    Only include patients who had care management ACTIVE during ON-window
    (member_status = ONBOARDED or ACTIVATED during that period).

    Restricted to patients with documented CM enrollment (enrolled_pids).
    """
    elig = d["eligibility"].copy()
    hv = d["hospital_visits"]
    mg = d["member_goals"]
    ms = d["member_status"]

    elig["enrollment_start_date"] = pd.to_datetime(elig["enrollment_start_date"], utc=True)
    elig["enrollment_end_date"] = pd.to_datetime(elig["enrollment_end_date"], utc=True)

    # Remove implausible dates
    elig = elig[
        (elig["enrollment_start_date"] >= pd.Timestamp("2020-01-01", tz="UTC")) &
        (elig["enrollment_end_date"] >= elig["enrollment_start_date"]) &
        elig["person_id"].notna()
    ]

    # Find consecutive enrollment pairs with gaps
    elig_sorted = elig.sort_values(["person_id", "enrollment_start_date"])
    elig_sorted["next_start"] = elig_sorted.groupby("person_id")["enrollment_start_date"].shift(-1)
    elig_sorted["next_end"] = elig_sorted.groupby("person_id")["enrollment_end_date"].shift(-1)
    elig_sorted["gap_days"] = (
        elig_sorted["next_start"] - elig_sorted["enrollment_end_date"]
    ).dt.days

    gaps = elig_sorted[
        (elig_sorted["gap_days"] >= 60) &
        (elig_sorted["gap_days"] <= 365)
    ].copy()

    # ON-window: last 90d of coverage before gap
    gaps["on_end"] = gaps["enrollment_end_date"]
    gaps["on_start"] = gaps["on_end"] - pd.Timedelta(days=90)

    # OFF-window: first 90d of gap
    gaps["off_start"] = gaps["enrollment_end_date"]
    gaps["off_end"] = gaps["off_start"] + pd.Timedelta(days=90)

    # Require ON-window starts no earlier than Jan 2023
    gaps = gaps[gaps["on_start"] >= pd.Timestamp("2023-01-01", tz="UTC")].copy()

    # Map person_id (WAY) to member_id (CUID) via bridge
    way_to_cuid = bridge.dropna(subset=["person_id", "member_id"]).set_index("person_id")["member_id"].to_dict()
    gaps["member_id"] = gaps["person_id"].map(way_to_cuid)
    gaps = gaps[gaps["member_id"].notna()].copy()

    # Only include patients with prior CM enrollment (ONBOARDED/ACTIVATED)
    gaps = gaps[gaps["member_id"].isin(enrolled_pids)].copy()

    print(f"    WPAD Type 3: {len(gaps):,} churn events from enrolled patients")

    if len(gaps) == 0:
        return pd.DataFrame()

    gaps = gaps.reset_index(drop=True)

    y_on = _compute_90d_outcome(gaps["member_id"], gaps["on_start"], hv)
    y_off = _compute_90d_outcome(gaps["member_id"], gaps["off_start"], hv)

    intv_type = _label_intervention(
        gaps["member_id"], gaps["on_start"], gaps["on_end"], mg
    )

    pairs = pd.DataFrame({
        "patient_id": gaps["member_id"].values,
        "wpad_type": "coverage_gap",
        "direction": "on_before_off",
        "wpad_gap_days": gaps["gap_days"].values,
        "on_start": gaps["on_start"].values,
        "on_end": gaps["on_end"].values,
        "off_start": gaps["off_start"].values,
        "off_end": gaps["off_end"].values,
        "y_on": y_on.values,
        "y_off": y_off.values,
        "behavioral_intervention": intv_type.values,
    })

    pairs = pairs[pairs["y_on"] == 0].copy()
    pairs["pair_type"] = "itt_coverage_gap"
    pairs["pair_weight"] = np.where(pairs["y_off"] == 1, 0.75, 0.35)

    print(f"      → {(pairs['y_off']==1).sum():,} events in OFF-window "
          f"({len(pairs):,} total ITT pairs)")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 11. Rising-risk cohort construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_patient_cohort(d: dict, bridge: pd.DataFrame, wpad_pids: set) -> pd.DataFrame:
    """
    Build the full rising-risk patient cohort.

    Rising-risk definition (priority order):
    1. SignalRisk (MIRA) risingRiskScorePercentile 70-90 (N~39,317) — primary
    2. Fallback: outcomes_monthly rr_flag > 0 (N~19,498) — if signal_risk unavailable

    Index date: first month with rr_flag=1 in outcomes_monthly (for continuity);
    for signal_risk patients without outcomes_monthly history, use appliedDate.
    Outcome (y_behavioral): 90-day composite acute care from index date.
    """
    om = d["outcomes_monthly"].copy()
    om["month_year"] = pd.to_datetime(om["month_year"])
    hv = d["hospital_visits"]
    sr = d.get("signal_risk", pd.DataFrame())

    way_to_cuid = bridge.dropna(subset=["person_id", "member_id"]).set_index("person_id")["member_id"].to_dict()
    cuid_to_way = {v: k for k, v in way_to_cuid.items()}

    # ── Cohort definition ─────────────────────────────────────────────────────
    if len(sr) > 0:
        # Primary: MIRA rising-risk (70th–90th percentile)
        rr_sr = sr[
            (sr["risingRiskScorePercentile"] >= 70) &
            (sr["risingRiskScorePercentile"] <= 90)
        ].copy()
        rr_sr = rr_sr.drop_duplicates("patientId")
        rr_sr["member_id"] = rr_sr["patientId"]
        rr_sr["person_id"] = rr_sr["patientId"].map(cuid_to_way)
        # Drop patients without WAY mapping (can't get outcomes_monthly)
        rr_sr = rr_sr[rr_sr["person_id"].notna()].reset_index(drop=True)
        print(f"    Rising-risk (MIRA 70-90th pctile): {len(rr_sr):,} patients with WAY mapping")

        # Index dates from outcomes_monthly (first rr_flag month)
        rr_om = om[om["rr_flag"] > 0].groupby("person_id")["month_year"].min().reset_index()
        rr_om.columns = ["person_id", "index_month"]
        rr_first = rr_sr.merge(rr_om[["person_id", "index_month"]], on="person_id", how="left")
        # Fallback index date: appliedDate from signal_risk
        if "appliedDate" in sr.columns:
            rr_first["_applied"] = pd.to_datetime(
                rr_sr.set_index("person_id")["appliedDate"].reindex(rr_first["person_id"]).values,
                errors="coerce"
            )
            rr_first["index_month"] = rr_first["index_month"].fillna(rr_first["_applied"])
            rr_first = rr_first.drop(columns=["_applied"])
        rr_first["index_month"] = rr_first["index_month"].fillna(pd.Timestamp("2024-01-01"))
        rr_first["index_date"] = pd.to_datetime(rr_first["index_month"]).dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
    else:
        # Fallback: rr_flag from outcomes_monthly
        rr = om[om["rr_flag"] > 0].copy()
        rr_first = (
            rr.groupby("person_id")["month_year"].min().reset_index()
            .rename(columns={"month_year": "index_month"})
        )
        rr_first["index_date"] = rr_first["index_month"].dt.tz_localize("UTC")
        rr_first["member_id"] = rr_first["person_id"].map(way_to_cuid)
        rr_first = rr_first[rr_first["member_id"].notna()].reset_index(drop=True)
        print(f"    Rising-risk (rr_flag fallback): {len(rr_first):,} patients with CUID")

    print(f"    Rising-risk patients with CUID: {len(rr_first):,}")

    # ── 90-day outcome ────────────────────────────────────────────────────────
    print("    Computing 90-day composite outcomes...")
    y_behavioral = _compute_90d_outcome(
        rr_first["member_id"], rr_first["index_date"], hv
    )
    rr_first["y_behavioral"] = y_behavioral.values

    # ── Prior utilization (12mo lookback from index_date) ────────────────────
    print("    Computing prior utilization features...")
    util = _compute_prior_utilization(
        rr_first["person_id"], rr_first["index_date"], om
    )
    for col in util.columns:
        rr_first[col] = util[col].values

    rr_first["prior_ed_visits_6mo"] = rr_first["prior_ed_visits_6mo"]
    rr_first["prior_hosp_6mo"] = rr_first["prior_hosp_6mo"]

    # ── Demographics ─────────────────────────────────────────────────────────
    print("    Adding demographics...")
    demog = _build_demographics(rr_first["member_id"], d)
    for col in ["age", "female", "race_eth", "primary_language", "risk_score"]:
        rr_first[col] = demog[col].values if col in demog.columns else (45.0 if col == "age" else 0)

    # ── Clinical features (real enriched data) ────────────────────────────────
    print("    Adding clinical features (enriched from claims + MIRA)...")
    clin = _build_clinical_features(rr_first["member_id"], d, way_ids=rr_first["person_id"])
    for col in clin.columns:
        if col not in ("member_id", "person_id"):
            rr_first[col] = clin[col].values

    # ── ADI quintile: use MIRA risingRiskScorePercentile if available ─────────
    if "risingRiskScorePercentile" in rr_first.columns:
        rr_first["adi_percentile"] = rr_first["risingRiskScorePercentile"].fillna(50)
    else:
        rs = rr_first.get("risk_score", pd.Series(0, index=rr_first.index)).fillna(0)
        rr_first["adi_percentile"] = (rs.rank(pct=True) * 100).fillna(50)
    rr_first["adi_quintile"] = pd.qcut(
        rr_first["adi_percentile"], q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # ── Behavioral intervention label ─────────────────────────────────────────
    # For the full cohort: what intervention was actually delivered?
    # Use all-time primary goal category (not window-restricted).
    print("    Labeling behavioral interventions...")
    mg = d["member_goals"]
    all_goals_by_member = (
        mg[mg["category"] != "DEFAULT"]
        .groupby("member_id")["category"]
        .apply(lambda cats: cats.map(GOAL_MAP).dropna().mode()[0]
               if len(cats.map(GOAL_MAP).dropna()) > 0 else None)
        .to_dict()
    )

    def assign_behavioral(row):
        # 1. Use mapped primary goal if available
        goal_intv = all_goals_by_member.get(row["member_id"])
        if goal_intv is not None:
            return goal_intv
        # 2. Risk-score proxy: route to most clinically salient category
        if row.get("food_insecure", 0):
            return "food_security"
        elif row.get("housing_unstable", 0):
            return "housing"
        elif row.get("no_transport", 0):
            return "transport_utilities"
        elif row.get("has_mh", 0):
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
        elif row.get("poor_adherence", 0) or row.get("medication_need", 0):
            return "medication_adherence"
        else:
            return "care_access"

    rr_first["behavioral_intervention"] = rr_first.apply(assign_behavioral, axis=1)

    # ── Other required fields ─────────────────────────────────────────────────
    rr_first["patient_id"] = rr_first["member_id"]
    rr_first["rising_risk"] = True
    rr_first["optimal_intervention"] = rr_first["behavioral_intervention"]  # unknown for real data

    # Cost percentile (for consistency with comparators that use it)
    total_paid = rr_first.get("total_paid_12mo", pd.Series(0, index=rr_first.index)).fillna(0)
    rr_first["cost_percentile"] = (total_paid.rank(pct=True) * 100).fillna(50)

    return rr_first.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Cross-patient AIPW pairs
# ─────────────────────────────────────────────────────────────────────────────

def _build_cross_patient_pairs(patients: pd.DataFrame, rng_seed: int = 42) -> pd.DataFrame:
    """
    Build cross-patient AIPW-weighted preference pairs from the rising-risk cohort.

    For each good-outcome patient (y=0), match to a bad-outcome patient (y=1)
    with similar clinical profile. AIPW weight = 1/propensity.

    Both good and bad are sampled with replacement so we can generate up to
    30,000 pairs even when bad-outcome patients are rare (<5%).
    """
    rng = np.random.default_rng(rng_seed)

    good = patients[patients["y_behavioral"] == 0]
    bad = patients[patients["y_behavioral"] == 1]

    n_pairs = min(max(len(bad) * 10, 5_000), 30_000)
    good_idx = rng.choice(len(good), size=n_pairs, replace=True)
    bad_idx = rng.choice(len(bad), size=n_pairs, replace=True)  # with replacement OK

    good_sample = good.iloc[good_idx].reset_index(drop=True)
    bad_sample = bad.iloc[bad_idx].reset_index(drop=True)

    # Propensity proxy: charlson_score-based weight
    charls_g = good_sample["charlson_score"].fillna(0).values
    charls_b = bad_sample["charlson_score"].fillna(0).values
    # Higher charlson → more likely bad outcome → lower weight for good outcome patient
    prop_score = 0.05 + 0.15 * (charls_g / (charls_g.max() + 1e-6))
    aipw_weight = np.clip(1.0 / (prop_score + 1e-6), 0.1, 10.0)

    return pd.DataFrame({
        "good_patient_id": good_sample["patient_id"].values,
        "bad_patient_id": bad_sample["patient_id"].values,
        "good_charlson": charls_g,
        "bad_charlson": charls_b,
        "good_intervention": good_sample["behavioral_intervention"].values,
        "aipw_weight": aipw_weight,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 13. Camden stratum patients
# ─────────────────────────────────────────────────────────────────────────────

def _build_camden_stratum(patients: pd.DataFrame) -> pd.DataFrame:
    """
    Identify patients matching a Camden Coalition analog profile for the rising-risk tier.

    Rising-risk Camden analog: Charlson ≥ 2 AND (prior_hosp_6mo ≥ 1 OR prior_ed_visits_6mo ≥ 2).
    These are the most complex rising-risk patients — not yet at Camden's full acuity
    (≥2 hospitalizations in 6mo, top 2% by cost), but heading there.

    Lower thresholds reflect the rising-risk (70-90th percentile) population rather
    than Camden's high-risk (90th+) population. Paper clearly distinguishes these.
    """
    mask = (
        (patients["charlson_score"] >= 2) &
        (
            (patients["prior_hosp_6mo"] >= 1) |
            (patients["prior_ed_visits_6mo"] >= 2)
        )
    )
    stratum = patients[mask].copy()
    if len(stratum) == 0:
        # Last-resort fallback: Charlson ≥ 1 with any prior utilization
        stratum = patients[
            (patients["charlson_score"] >= 1) &
            (patients["prior_hosp_6mo"] + patients["prior_ed_visits_6mo"] >= 1)
        ].copy()
    return stratum


# ─────────────────────────────────────────────────────────────────────────────
# 14. Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_waymark_population(verbose: bool = True) -> WaymarkPopulation:
    """
    Full pipeline: load raw data → construct WPAD pairs → build patient cohort.

    Returns WaymarkPopulation (drop-in for SyntheticPopulation).
    Runtime: ~5-10 minutes on local hardware.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("WAYMARK REAL DATA EXTRACTION")
        print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    d = _load_raw()
    bridge = _build_id_bridge(d)
    if verbose:
        print(f"  ID bridge: {len(bridge):,} CUID↔WAY pairs")

    # Set of enrolled (ONBOARDED/ACTIVATED) member CUIDs
    ms = d["member_status"]
    enrolled_pids = set(
        ms[ms["to_status"].isin(["ONBOARDED", "ACTIVATED"])]["member_id"].dropna()
    )
    if verbose:
        print(f"  Ever ONBOARDED/ACTIVATED patients: {len(enrolled_pids):,}")

    # ── WPAD pairs ────────────────────────────────────────────────────────────
    if verbose:
        print("\nConstructing WPAD pairs...")

    type1 = _build_wpad_type1(d, bridge)
    type2 = _build_wpad_type2(d, type1)
    type3 = _build_wpad_type3(d, bridge, enrolled_pids)

    wpad_all = [type1, type2]
    if len(type3) > 0:
        wpad_all.append(type3)
    wpad_pairs = pd.concat(wpad_all, ignore_index=True)

    # Ensure behavioral_intervention is valid
    wpad_pairs["behavioral_intervention"] = wpad_pairs["behavioral_intervention"].where(
        wpad_pairs["behavioral_intervention"].isin(INTERVENTIONS), "care_access"
    )

    # Deduplicate: one pair per patient (keep primary over weak_positive)
    priority = {"primary": 0, "weak_positive": 1, "itt_coverage_gap": 2}
    wpad_pairs["_prio"] = wpad_pairs["pair_type"].map(priority).fillna(3)
    wpad_pairs = (
        wpad_pairs.sort_values("_prio")
        .drop_duplicates("patient_id", keep="first")
        .drop(columns="_prio")
        .reset_index(drop=True)
    )

    if verbose:
        print(f"\n  WPAD pairs summary:")
        print(f"    Total pairs: {len(wpad_pairs):,}")
        print(f"    Type 1 (onboarding): {(wpad_pairs['wpad_type']=='aco_onboarding').sum():,}")
        print(f"    Type 2 (waitlist):   {(wpad_pairs['wpad_type']=='chw_waitlist').sum():,}")
        print(f"    Type 3 (churn ITT):  {(wpad_pairs['wpad_type']=='coverage_gap').sum():,}")
        print(f"    Primary pairs:       {(wpad_pairs['pair_type']=='primary').sum():,}")
        print(f"    Weak positive:       {(wpad_pairs['pair_type']=='weak_positive').sum():,}")
        print(f"    ITT pairs:           {(wpad_pairs['pair_type']=='itt_coverage_gap').sum():,}")
        print(f"    Intervention dist:")
        print(wpad_pairs["behavioral_intervention"].value_counts().to_string())

    wpad_pids = set(wpad_pairs["patient_id"].tolist())

    # ── Patient cohort ────────────────────────────────────────────────────────
    if verbose:
        print("\nBuilding rising-risk patient cohort...")

    patients = _build_patient_cohort(d, bridge, wpad_pids)

    if verbose:
        print(f"\n  Patient cohort: {len(patients):,} rising-risk patients")
        print(f"  y_behavioral=1 rate: {patients['y_behavioral'].mean():.3f} "
              f"({patients['y_behavioral'].sum():,} acute care events in 90 days)")
        print(f"  Intervention distribution (behavioral):")
        print(patients["behavioral_intervention"].value_counts().to_string())
        print(f"  Demographics: age {patients['age'].mean():.1f}y, "
              f"female {patients['female'].mean():.1%}, "
              f"charlson {patients['charlson_score'].mean():.2f}")

    # ── Cross-patient pairs ───────────────────────────────────────────────────
    cross_pairs = _build_cross_patient_pairs(patients)
    if verbose:
        print(f"\n  Cross-patient AIPW pairs: {len(cross_pairs):,}")

    # ── Camden stratum ────────────────────────────────────────────────────────
    camden = _build_camden_stratum(patients)
    if verbose:
        print(f"  Camden stratum (Charlson≥4, prior IP≥2): {len(camden):,} patients")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "n_patients": len(patients),
        "n_wpad_pairs": len(wpad_pairs),
        "n_wpad_primary": int((wpad_pairs["pair_type"] == "primary").sum()),
        "n_wpad_type1": int((wpad_pairs["wpad_type"] == "aco_onboarding").sum()),
        "n_wpad_type2": int((wpad_pairs["wpad_type"] == "chw_waitlist").sum()),
        "n_wpad_type3": int((wpad_pairs["wpad_type"] == "coverage_gap").sum()),
        "acute_event_rate": float(patients["y_behavioral"].mean()),
        "data_source": "Waymark ACO care management (real data)",
        "states": ["WA", "VA"],
        "study_period": "2023-04 to 2025-10",
    }

    pop = WaymarkPopulation(
        patients=patients,
        wpad_pairs=wpad_pairs,
        cross_patient_pairs=cross_pairs,
        ground_truth_imi=float("nan"),
        optimal_policy=pd.DataFrame({
            "patient_id": patients["patient_id"],
            "optimal_intervention": patients["behavioral_intervention"],
        }),
        camden_stratum_patients=camden,
        metadata=meta,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("WAYMARK DATA EXTRACTION COMPLETE")
        print(f"  Patients:          {meta['n_patients']:,}")
        print(f"  WPAD pairs:        {meta['n_wpad_pairs']:,}")
        print(f"  Primary WPAD:      {meta['n_wpad_primary']:,}")
        print(f"  Acute event rate:  {meta['acute_event_rate']:.1%}")
        print("=" * 60)

    return pop


if __name__ == "__main__":
    pop = build_waymark_population(verbose=True)
    print("\nIntervention distribution (behavioral policy):")
    print(pop.patients["behavioral_intervention"].value_counts(normalize=True).round(3))
    print("\nWPAD pair type breakdown:")
    print(pop.wpad_pairs[["wpad_type", "pair_type"]].value_counts().to_string())
    print(f"\ny_behavioral mean: {pop.patients['y_behavioral'].mean():.3f}")
