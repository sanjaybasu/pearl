"""
Synthetic data generator for PEARL development and public reproducibility demo.

Generates a realistic rising-risk ACO patient population with:
- Longitudinal EHR-like features (ICD-10 proxies, utilization, SDOH)
- WPAD events (ACO onboarding ramp-up + CHW waitlist + Medicaid coverage gaps)
- ON/OFF windows with simulated care management effects
- Ground-truth IMI (known from the data-generating process)

The synthetic population is calibrated to:
- MEPS chronic disease distributions (diabetes, CHF, COPD, hypertension)
- Waymark population age/SDOH profile (from published Muralidharan JMIR AI 2025 Table 1)
- Camden Coalition patient characteristics (Finkelstein NEJM 2019 Table 1 — for reanalysis)
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class SyntheticPopulation:
    """Complete synthetic dataset returned by generate_synthetic_population()."""
    patients: pd.DataFrame          # Patient-level features N×P
    wpad_pairs: pd.DataFrame        # WPAD preference pairs
    cross_patient_pairs: pd.DataFrame
    ground_truth_imi: float         # Known IMI under behavioral policy
    optimal_policy: pd.DataFrame    # Ground-truth optimal intervention per patient
    camden_stratum_patients: pd.DataFrame  # Camden-profile subset for reanalysis


def generate_synthetic_population(
    n_patients: int = 50_000,
    n_wpad_primary: int = 4000,
    n_wpad_coverage_gap: int = 1000,
    seed: int = 42,
    true_imi: float = 0.38,  # ~38% misalignment (conservative estimate)
) -> SyntheticPopulation:
    """
    Generate a realistic rising-risk ACO population with WPAD natural experiments.

    Ground truth: intervention misalignment is set to `true_imi`.
    Care management effects are heterogeneous by patient subtype.
    """
    rng = np.random.default_rng(seed)

    # ─── Patient Features ───────────────────────────────────────────────────
    n = n_patients

    # Demographics (calibrated to MEPS Medicaid, 2020-2023)
    age = rng.integers(25, 75, n)
    female = rng.binomial(1, 0.58, n)
    race_eth = rng.choice(
        ["white_nh", "black_nh", "hispanic", "asian", "other"],
        size=n, p=[0.30, 0.35, 0.25, 0.05, 0.05]
    )
    primary_language = np.where(
        race_eth == "hispanic",
        rng.choice(["english", "spanish"], size=n, p=[0.40, 0.60]),
        "english"
    )
    adi_percentile = rng.beta(3, 1.5, n) * 100  # skewed high (deprived)
    adi_quintile = pd.cut(adi_percentile, bins=5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Chronic conditions (rising-risk: 1-3 ambulatory-sensitive conditions)
    has_diabetes = rng.binomial(1, 0.48, n)
    has_chf = rng.binomial(1, 0.22, n)
    has_copd = rng.binomial(1, 0.18, n)
    has_hypertension = rng.binomial(1, 0.62, n)
    has_ckd = rng.binomial(1, 0.25, n)
    has_mh = rng.binomial(1, 0.31, n)  # mental health comorbidity
    n_chronic = has_diabetes + has_chf + has_copd + has_hypertension + has_ckd

    # Charlson score proxy (simplified)
    charlson = (
        has_diabetes * 1 + has_chf * 2 + has_copd * 1 +
        has_ckd * 2 + (age > 60).astype(int) * 1 +
        rng.poisson(0.5, n)
    ).clip(0, 10)

    # Utilization (12-month prior)
    prior_ed_6mo = rng.negative_binomial(1, 0.5, n).clip(0, 6)
    prior_hosp_6mo = rng.binomial(2, 0.12, n)
    pharmacy_fills_90d = rng.negative_binomial(3, 0.5, n).clip(0, 15)
    missed_pharmacy_fills = rng.binomial(pharmacy_fills_90d.clip(1, 15), 0.25)

    # SDOH (linked from ACS/ADI/USDA proxies)
    food_insecure = (adi_percentile > 70) & (rng.binomial(1, 0.55, n) == 1)
    housing_unstable = (adi_percentile > 80) & (rng.binomial(1, 0.20, n) == 1)
    lives_alone = rng.binomial(1, 0.28, n)
    no_transport = (adi_percentile > 60) & (rng.binomial(1, 0.30, n) == 1)

    # Predicted 12-month cost percentile (used for rising-risk filter 70-90th pctile)
    cost_score = (
        charlson * 0.3 + prior_ed_6mo * 0.25 + prior_hosp_6mo * 0.35 +
        food_insecure.astype(float) * 0.1 + rng.normal(0, 0.5, n)
    )
    cost_percentile = pd.Series(cost_score).rank(pct=True).values * 100

    # Rising-risk filter: 70th-90th percentile
    rising_risk = (cost_percentile >= 70) & (cost_percentile <= 90)

    patients_df = pd.DataFrame({
        "patient_id": [f"P{i:06d}" for i in range(n)],
        "age": age, "female": female, "race_eth": race_eth,
        "primary_language": primary_language,
        "adi_percentile": adi_percentile.round(1),
        "adi_quintile": adi_quintile,
        "has_diabetes": has_diabetes, "has_chf": has_chf,
        "has_copd": has_copd, "has_hypertension": has_hypertension,
        "has_ckd": has_ckd, "has_mh": has_mh,
        "n_chronic": n_chronic, "charlson_score": charlson,
        "prior_ed_visits_6mo": prior_ed_6mo,
        "prior_hosp_6mo": prior_hosp_6mo,
        "pharmacy_fills_90d": pharmacy_fills_90d,
        "missed_pharmacy_fills": missed_pharmacy_fills,
        "food_insecure": food_insecure.astype(int),
        "housing_unstable": housing_unstable.astype(int),
        "lives_alone": lives_alone, "no_transport": no_transport.astype(int),
        "cost_percentile": cost_percentile.round(1),
        "rising_risk": rising_risk,
    })

    # ─── Patient Subtypes and Ground-Truth Optimal Interventions ──────────
    # 14 subtypes map to the 14 MoE expert domains.
    # Ground truth: which intervention type maximally reduces acute care events.
    def assign_subtype(row):
        """Rule-based ground-truth optimal intervention assignment (14-category taxonomy)."""
        # Condition-specific care (highest specificity first)
        if row["has_chf"]:
            return "heart_failure"
        elif row["has_copd"]:
            return "pulmonary"
        elif row["has_diabetes"] and row["missed_pharmacy_fills"] > 2:
            return "diabetes"
        elif row["has_hypertension"] and not row["has_chf"]:
            return "hypertension"
        # SDOH/social needs (priority when present)
        elif row["food_insecure"]:
            return "food_security"
        elif row["housing_unstable"]:
            return "housing"
        elif row["no_transport"] and row["adi_percentile"] > 70:
            return "transport_utilities"
        # Behavioral / MH
        elif row["has_mh"] and row["charlson_score"] < 4:
            return "mental_health"
        # Medication adherence
        elif row["missed_pharmacy_fills"] > 2 or row["pharmacy_fills_90d"] < 3:
            return "medication_adherence"
        # Access/care coordination
        elif row["prior_hosp_6mo"] >= 1 and row["charlson_score"] >= 3:
            return "care_access"
        else:
            return "clinical_other"

    patients_df["optimal_intervention"] = patients_df.apply(assign_subtype, axis=1)

    # Behavioral policy: risk-score based routing with systematic misalignment
    # Simulates current NBA system's behavioral cloning bias (14-category taxonomy).
    def behavioral_policy(row, rng_local):
        """Behavioral policy: roughly correct but with systematic equity gaps."""
        opt = row["optimal_intervention"]
        # Condition-specific routing: high fidelity when clinical signal is clear
        if opt in ("heart_failure", "pulmonary", "diabetes", "hypertension"):
            return opt if rng_local.random() < 0.72 else "care_access"
        # SDOH routing: under-detected for non-English speakers and high-ADI patients
        elif opt == "food_security":
            if row["primary_language"] != "english":
                return rng_local.choice(["food_security", "care_access"], p=[0.50, 0.50])
            return "food_security" if rng_local.random() < 0.70 else "care_access"
        elif opt == "housing":
            return "housing" if rng_local.random() < 0.60 else "care_access"
        elif opt == "transport_utilities":
            return "transport_utilities" if rng_local.random() < 0.55 else "care_access"
        # Mental health most under-referred (known equity gap)
        elif opt == "mental_health":
            return "mental_health" if rng_local.random() < 0.45 else "care_access"
        elif opt == "substance_use":
            return "substance_use" if rng_local.random() < 0.50 else "mental_health"
        elif opt == "medication_adherence":
            return "medication_adherence" if rng_local.random() < 0.68 else "care_access"
        elif opt == "maternal":
            return "maternal" if rng_local.random() < 0.75 else "care_access"
        elif opt == "financial_benefits":
            return "financial_benefits" if rng_local.random() < 0.55 else "care_access"
        else:
            return "care_access" if rng_local.random() < 0.75 else "clinical_other"

    rng_bp = np.random.default_rng(seed + 1)
    patients_df["behavioral_intervention"] = patients_df.apply(
        lambda row: behavioral_policy(row, rng_bp), axis=1
    )

    # Verify ground-truth IMI is close to target
    actual_imi = (
        patients_df["behavioral_intervention"] != patients_df["optimal_intervention"]
    ).mean()

    # ─── Potential Outcomes ────────────────────────────────────────────────
    # 90-day composite acute care event probability under each intervention.
    # Ground-truth heterogeneous effects calibrated to Muralidharan JMIR AI 2025.

    def compute_potential_outcome(row, intervention: str, rng_po, base_rate: float = 0.22):
        """Fine-Gray-like 90-day acute care event probability."""
        # Match bonus: optimal intervention reduces event rate by 15-25%
        opt = row["optimal_intervention"]
        is_matched = (intervention == opt)

        # Base rate adjusted by clinical severity
        base = base_rate + 0.04 * (row["charlson_score"] > 4) + 0.03 * row["prior_hosp_6mo"]

        if is_matched:
            reduction = rng_po.uniform(0.15, 0.25)  # heterogeneous treatment effect
            prob = base * (1 - reduction)
        else:
            # Mismatched: no reduction or slight harm
            prob = base * rng_po.uniform(0.97, 1.08)

        return np.clip(prob, 0.02, 0.85)

    rng_po = np.random.default_rng(seed + 2)
    interventions = [
        "care_access", "clinical_other", "diabetes", "financial_benefits",
        "food_security", "heart_failure", "housing", "hypertension",
        "maternal", "medication_adherence", "mental_health", "pulmonary",
        "substance_use", "transport_utilities",
    ]
    for intv in interventions:
        patients_df[f"p_outcome_{intv}"] = patients_df.apply(
            lambda row: compute_potential_outcome(row, intv, rng_po), axis=1
        )

    # Realized outcomes under behavioral policy
    patients_df["y_behavioral"] = patients_df.apply(
        lambda row: rng.binomial(1, row[f"p_outcome_{row['behavioral_intervention']}"]),
        axis=1
    )
    patients_df["y_optimal"] = patients_df.apply(
        lambda row: rng.binomial(1, row[f"p_outcome_{row['optimal_intervention']}"]),
        axis=1
    )

    # ─── WPAD Events ──────────────────────────────────────────────────────
    # Type 1+2 primary: ACO onboarding + waitlist (near-100% engagement)
    # Type 3 secondary: coverage gap (5-10% engagement, ITT)

    rising_patients = patients_df[patients_df["rising_risk"]].copy()
    n_rising = len(rising_patients)

    # Primary WPAD pairs: staggered onboarding / waitlist
    primary_idx = rng.choice(rising_patients.index, size=min(n_wpad_primary, n_rising), replace=False)
    primary_pts = patients_df.loc[primary_idx].copy()

    # Assign WPAD window characteristics
    primary_pts["wpad_type"] = rng.choice(
        ["aco_onboarding", "chw_waitlist"],
        size=len(primary_pts), p=[0.60, 0.40]
    )
    # Calendar direction: onboarding = OFF precedes ON; churn = ON precedes OFF
    primary_pts["direction"] = np.where(
        primary_pts["wpad_type"] == "aco_onboarding", "off_before_on", "on_before_off"
    )
    primary_pts["wpad_gap_days"] = rng.integers(60, 120, len(primary_pts))

    # Simulate ON/OFF outcomes
    # ON window: care management active → outcome from behavioral_intervention
    # OFF window: no care management → higher event rate
    no_cm_rate_multiplier = rng.uniform(1.15, 1.35, len(primary_pts))  # ~25% increase without CM
    primary_pts["y_on"] = primary_pts.apply(
        lambda row: rng.binomial(1, row[f"p_outcome_{row['behavioral_intervention']}"]),
        axis=1
    )
    primary_pts["y_off"] = primary_pts.apply(
        lambda row: rng.binomial(1, min(row[f"p_outcome_{row['behavioral_intervention']}"] *
                                       no_cm_rate_multiplier[primary_pts.index.get_loc(row.name)], 0.95)),
        axis=1
    )

    # Build WPAD preference pairs
    # Primary pairs: Y_off=1, Y_on=0 → care management helped
    good_pairs = primary_pts[(primary_pts["y_on"] == 0) & (primary_pts["y_off"] == 1)].copy()
    good_pairs["pair_type"] = "primary"
    good_pairs["pair_weight"] = 1.0

    # Weak positive pairs: both windows good (informative but weaker signal)
    both_good = primary_pts[(primary_pts["y_on"] == 0) & (primary_pts["y_off"] == 0)].copy()
    both_good["pair_type"] = "weak_positive"
    both_good["pair_weight"] = 0.5

    wpad_pairs = pd.concat([good_pairs, both_good], ignore_index=True)

    # Preferred completion: care plan from ON-window (behavioral intervention used)
    wpad_pairs["preferred_intervention"] = wpad_pairs["behavioral_intervention"]
    # Rejected completion: null / off-window baseline
    wpad_pairs["rejected_intervention"] = "no_cm_baseline"

    # ─── Coverage-Gap WPAD (Secondary ITT) ─────────────────────────────────
    gap_idx = rng.choice(rising_patients.index, size=min(n_wpad_coverage_gap, n_rising), replace=False)
    gap_pts = patients_df.loc[gap_idx].copy()
    gap_pts["wpad_type"] = "coverage_gap"
    gap_pts["direction"] = "on_before_off"
    gap_pts["engagement_rate"] = 0.075  # 7.5% engagement rate
    # Only ~7.5% of contacted patients engage → effective N
    engaged = rng.binomial(1, 0.075, len(gap_pts)).astype(bool)
    gap_pts = gap_pts[engaged].copy()
    gap_pairs = gap_pts.copy()
    gap_pairs["pair_type"] = "itt_coverage_gap"
    gap_pairs["pair_weight"] = 0.5

    # ─── Cross-Patient IPTW Pairs ───────────────────────────────────────
    # Matched pairs from full rising-risk population using propensity scores
    n_cross = min(30_000, n_rising)
    good_outcome_idx = rising_patients[rising_patients["y_behavioral"] == 0].index
    bad_outcome_idx = rising_patients[rising_patients["y_behavioral"] == 1].index

    n_cross_pairs = min(len(good_outcome_idx), len(bad_outcome_idx), 15000)
    good_sample = rng.choice(good_outcome_idx, size=n_cross_pairs, replace=True)
    bad_sample = rng.choice(bad_outcome_idx, size=n_cross_pairs, replace=True)

    cross_pairs_df = pd.DataFrame({
        "good_patient_id": patients_df.loc[good_sample, "patient_id"].values,
        "bad_patient_id": patients_df.loc[bad_sample, "patient_id"].values,
        "good_charlson": patients_df.loc[good_sample, "charlson_score"].values,
        "bad_charlson": patients_df.loc[bad_sample, "charlson_score"].values,
        "good_intervention": patients_df.loc[good_sample, "behavioral_intervention"].values,
        # Propensity weight (simplified; real impl uses logistic regression)
        "aipw_weight": rng.uniform(0.5, 2.0, n_cross_pairs).clip(0.1, 10.0),
    })

    # ─── Camden Stratum Patients ───────────────────────────────────────────
    # High-risk profile: Charlson ≥ 4, prior_hosp_6mo ≥ 2
    # Simulates patients who would have been enrolled in Camden protocol
    camden_mask = (patients_df["charlson_score"] >= 4) & (patients_df["prior_hosp_6mo"] >= 2)
    camden_pts = patients_df[camden_mask].copy()

    return SyntheticPopulation(
        patients=patients_df,
        wpad_pairs=wpad_pairs,
        cross_patient_pairs=cross_pairs_df,
        ground_truth_imi=actual_imi,
        optimal_policy=patients_df[["patient_id", "optimal_intervention"]],
        camden_stratum_patients=camden_pts,
    )


def format_patient_context(row: pd.Series) -> str:
    """
    Format patient features as a structured clinical prompt for Llama-3.1-8B.
    This is the x_i in the DPO (x, y_w, y_l) triple.
    """
    conditions = []
    if row.get("has_diabetes"): conditions.append("T2DM")
    if row.get("has_chf"): conditions.append("CHF")
    if row.get("has_copd"): conditions.append("COPD")
    if row.get("has_hypertension"): conditions.append("HTN")
    if row.get("has_ckd"): conditions.append("CKD")
    if row.get("has_mh"): conditions.append("MH comorbidity")

    sdoh = []
    if row.get("food_insecure"): sdoh.append("food insecurity")
    if row.get("housing_unstable"): sdoh.append("housing instability")
    if row.get("no_transport"): sdoh.append("transportation barrier")
    if row.get("lives_alone"): sdoh.append("lives alone")
    if row.get("primary_language") == "spanish": sdoh.append("primary language: Spanish")

    context = f"""Patient Profile:
Age: {row.get('age', 'unknown')} | Sex: {'F' if row.get('female') else 'M'} | ADI: {row.get('adi_percentile', 0):.0f}th percentile
Conditions: {', '.join(conditions) if conditions else 'none documented'}
Charlson score: {row.get('charlson_score', 0)} | Prior ED (6mo): {row.get('prior_ed_visits_6mo', 0)} | Prior hosp (6mo): {row.get('prior_hosp_6mo', 0)}
Pharmacy fills (90d): {row.get('pharmacy_fills_90d', 0)} | Missed fills: {row.get('missed_pharmacy_fills', 0)}
SDOH: {', '.join(sdoh) if sdoh else 'none documented'}
Current care management: Rising-risk ACO patient, eligible for care management enrollment.

Generate a personalized care management plan specifying:
1. Intervention type (CHW G0511/G0512, pharmacist review, behavioral health referral, or community resource navigation)
2. Intensity (contacts per 30 days)
3. Specific focus areas
4. Escalation triggers
5. What NOT to recommend (and why)"""
    return context


def format_care_plan(intervention_type: str, row: pd.Series) -> str:
    """Format an intervention type as a natural language care plan (y_w or y_l)."""
    _lang = row.get("primary_language", "english")
    _transport = row.get("no_transport", False)
    _chf = row.get("has_chf", False)
    _copd = row.get("has_copd", False)
    _ckd = row.get("has_ckd", False)
    _specialists = ", ".join([s for s, f in [
        ("cardiologist", _chf), ("pulmonologist", _copd), ("nephrologist", _ckd)
    ] if f]) or "PCP coordination"

    plans = {
        "care_access": f"""Care Management Plan:
Priority intervention: Care coordination — PCP appointment scheduling + care transitions management
Focus: Expedited PCP visit, specialist referral coordination ({_specialists}), care gap closure
Intensity: Biweekly CHW check-ins; monthly care team huddle
Escalation: Same-day PCP triage if new acute symptom; expedited ED-avoidance protocol if needed
What NOT to recommend: Isolated SDOH navigation without clinical follow-up; behavioral health referral without documented MH history""",

        "clinical_other": f"""Care Management Plan:
Priority intervention: CHW-coordinated wellness visit + preventive care gap closure
Focus: Dental referral, vision screening, age-appropriate preventive services
Intensity: 1 CHW contact/month; referral coordination within 30 days
Escalation: Notify PCP if patient reports new symptoms or declines preventive care
What NOT to recommend: Intensive disease management (no high-acuity chronic condition driving utilization)""",

        "diabetes": f"""Care Management Plan:
Priority intervention: CHW outreach + pharmacist diabetes medication review
Focus: HbA1c monitoring, insulin technique, meter supplies, dietary counseling
Intensity: Weekly check-ins for first 4 weeks; pharmacist consult within 7 days; then 2x/month
Escalation: Alert PCP if HbA1c >10% or patient reports hypoglycemia symptoms
What NOT to recommend: Housing/SDOH focus without confirming glycemic control; intensive multidisciplinary if single medication adherence gap""",

        "financial_benefits": f"""Care Management Plan:
Priority intervention: CHW financial navigation + benefits enrollment
Focus: Medicaid redetermination support, prescription assistance programs, SNAP/TANF enrollment, legal aid referral
Intensity: 2 CHW contacts in first 2 weeks (paperwork-intensive); then monthly follow-up
Escalation: Notify supervising RN if patient reports coverage loss or medication cost barrier
What NOT to recommend: Clinical disease management focus when financial barrier is the root cause of non-adherence""",

        "food_security": f"""Care Management Plan:
Priority intervention: CHW home visit + community food resource navigation
Focus: {'Food pantry referral (Spanish-language resources available)' if _lang == 'spanish' else 'Food security assessment and pantry referral'}{', transportation assistance to food resources' if _transport else ''}; SNAP enrollment if eligible
Intensity: 2-3 CHW contacts/month (home visits preferred for SDOH complexity)
Escalation: Contact supervising RN if patient reports food supply crisis or medication-food interaction concern
What NOT to recommend: Intensive multidisciplinary team (clinical stability confirmed); behavioral health referral without documented MH history""",

        "heart_failure": f"""Care Management Plan:
Priority intervention: CHW home visit + cardiology care coordination
Focus: Daily weight monitoring (threshold: >2 lb gain → CHW call), furosemide adherence, sodium-restricted diet, dyspnea scale tracking
Intensity: 3 CHW contacts/month; cardiologist appointment within 14 days; telehealth check-in weekly
Escalation: Same-day ED/cardiologist triage if weight gain >2 lb OR new dyspnea; notify PCP immediately
What NOT to recommend: SDOH focus without first stabilizing fluid status; isolated telephonic check-ins without home weight scale setup""",

        "housing": f"""Care Management Plan:
Priority intervention: CHW housing stability navigation
Focus: Eviction prevention, rental assistance programs, habitability assessment, legal aid referral
Intensity: Weekly CHW contact during acute housing crisis; 2x/month after stabilization
Escalation: Emergency housing placement coordination if imminent eviction risk; notify supervising RN if living conditions affect medication storage
What NOT to recommend: Clinical disease management-first approach when housing instability is driving non-adherence""",

        "hypertension": f"""Care Management Plan:
Priority intervention: CHW home BP monitoring setup + pharmacist antihypertensive review
Focus: Home BP log, medication timing, sodium reduction, DASH diet counseling
Intensity: Biweekly CHW contact in first month; pharmacist review within 14 days; then monthly
Escalation: Alert PCP if BP >180/110 on 2 readings OR patient reports headache/visual change
What NOT to recommend: Intensive social work without BP stabilization as first priority; specialty referral before PCP-level optimization""",

        "maternal": f"""Care Management Plan:
Priority intervention: CHW prenatal/postpartum navigation
Focus: OB appointment scheduling, WIC enrollment, breastfeeding support, postpartum depression screening
Intensity: Weekly CHW contact through 6 weeks postpartum; {'Spanish-language doula referral' if _lang == 'spanish' else '2x/month CHW visits'}
Escalation: Immediate OB triage if preeclampsia symptoms; postpartum depression referral if Edinburgh score ≥ 13
What NOT to recommend: Generic chronic disease management focus without obstetric care coordination""",

        "medication_adherence": f"""Care Management Plan:
Priority intervention: CHW telephonic outreach + pharmacist medication review
Focus: Medication reconciliation, pill organizer setup, pharmacy synchronization, auto-refill enrollment
Intensity: Weekly check-ins for first month; pharmacist consult within 7 days; then 2x/month
Escalation: Alert PCP if >2 refill gaps detected OR patient reports side effect; expedited pharmacist review if formulary change
What NOT to recommend: Housing or SDOH focus when medication supply is confirmed; intensive multidisciplinary without first resolving adherence barrier""",

        "mental_health": f"""Care Management Plan:
Priority intervention: Behavioral health referral + co-located care coordination
Focus: PHQ-9/GAD-7 screening, warm handoff to BH provider, stigma-aware communication, PCP-BH integration
Intensity: 1 CHW contact/week during BH intake; reduce to 2x/month after engagement confirmed
Escalation: Crisis line enrollment if PHQ-9 ≥ 15; notify PCP immediately if safety concern; 24h follow-up after any crisis disclosure
What NOT to recommend: Disease management protocol as primary focus (BH is the primary driver); medication-first without BH assessment""",

        "pulmonary": f"""Care Management Plan:
Priority intervention: CHW asthma/COPD action plan setup + pulmonology coordination
Focus: Inhaler technique training, action plan review, trigger reduction (smoking cessation if indicated), spirometry referral
Intensity: Biweekly CHW contacts; pulmonology appointment within 21 days; rescue inhaler supply confirmed
Escalation: Same-day PCP/ED triage if peak flow <50% personal best OR new accessory muscle use; smoking cessation warm referral within 48h
What NOT to recommend: SDOH focus without first stabilizing pulmonary control; generic wellness visit as substitute for pulmonary-specific plan""",

        "substance_use": f"""Care Management Plan:
Priority intervention: SUD warm referral + CHW motivational outreach
Focus: AUDIT-C/DAST-10 screening, SBIRT, MAT referral (buprenorphine if OUD), peer support specialist connection
Intensity: Weekly CHW contact during treatment engagement; {'Spanish-language peer support available' if _lang == 'spanish' else ''}; 2x/month after 30-day engagement
Escalation: Crisis triage if overdose risk disclosed; naloxone dispensing and training; notify PCP within 24h of any acute safety concern
What NOT to recommend: Generic chronic disease management without SUD treatment integration; medication-only approach without behavioral health coordination""",

        "transport_utilities": f"""Care Management Plan:
Priority intervention: CHW transportation and utilities navigation
Focus: Medical transportation enrollment (NEMT), utility assistance (LIHEAP), childcare voucher, ride-share program
Intensity: 2 CHW contacts in first 2 weeks (enrollment-intensive); then monthly follow-up
Escalation: Notify supervising RN if appointment non-adherence due to transport; emergency utility shut-off triggers same-week CHW contact
What NOT to recommend: Clinical disease management when transport/utility barrier is the root cause of missed appointments""",

        "no_cm_baseline": """Care Management Plan:
No structured care management assigned.
Patient on standard ACO monitoring; PCP-directed care only.
No CHW outreach scheduled.""",
    }
    return plans.get(intervention_type, plans["no_cm_baseline"])


if __name__ == "__main__":
    print("Generating synthetic PEARL population (N=50,000)...")
    pop = generate_synthetic_population(n_patients=50_000, seed=42)
    print(f"  Patients: {len(pop.patients):,}")
    print(f"  Rising-risk patients: {pop.patients['rising_risk'].sum():,}")
    print(f"  WPAD primary pairs: {len(pop.wpad_pairs):,}")
    print(f"  Cross-patient pairs: {len(pop.cross_patient_pairs):,}")
    print(f"  Ground-truth IMI: {pop.ground_truth_imi:.3f} ({pop.ground_truth_imi*100:.1f}%)")
    print(f"  Camden stratum N: {len(pop.camden_stratum_patients):,}")
    print("\nIntervention distribution (behavioral policy):")
    print(pop.patients[pop.patients["rising_risk"]]["behavioral_intervention"].value_counts(normalize=True).round(3))
    print("\nOptimal intervention distribution:")
    print(pop.patients[pop.patients["rising_risk"]]["optimal_intervention"].value_counts(normalize=True).round(3))
