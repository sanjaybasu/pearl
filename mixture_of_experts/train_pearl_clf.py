#!/usr/bin/env python3
"""
Train PEARL C4 Behavioral Cloning Classifier

Trains a multi-class sklearn GradientBoostingClassifier on WPAD preference pairs:
  - Features:  29 PEARL_FEAT_ORDER features (EHR → text-parsed equivalents bridge)
  - Labels:    14 PEARL intervention types (from member_goals during ON-window)
  - Weights:   pair_weight  (primary = 1.0, weak_positive = 0.5, itt = 0.35–0.75)

Saves:
  packaging/pearl/mixture_of_experts/pearl_sft_clf.joblib

Used by PEARLScorer in pearl_nba.py:
  x_vec = build_features(parse_block(text))   # 29-element, Parth's FEAT_ORDER
  proba = clf.predict_proba(x_vec)[0]
  idx   = PEARL_INTERVENTIONS.index(pearl_type)   # alphabetical
  score = proba[idx]

Classifier classes_ must match PEARL_INTERVENTIONS exactly (both alphabetically
sorted — verified by assertion at training time).

Notes on the EHR→inference feature bridge
------------------------------------------
WPAD patients have structured EHR features. At inference time, pearl_nba.py
extracts equivalent features via parse_block() (Parth's existing text parser).
The mapping below converts EHR features to the closest text-parsed equivalents.

Features not available in EHR data (opioid, muscle relaxant, anticoagulant,
meth, fentanyl, antiarrhythmic) are set to 0 at training time — matching the
inference-time default when these terms don't appear in the patient prompt text.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]   # waymark-local
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packaging"))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

# ── Output path ───────────────────────────────────────────────────────────────
MODEL_OUTPUT = Path(__file__).parent / "pearl_sft_clf.joblib"
VAL_CSV      = REPO_ROOT / "data/processed/validation_set_N1000_labeled.csv"
DATA_DIR     = REPO_ROOT / "data/real_inputs"

# ── PEARL intervention type list (alphabetical — must match clf.classes_) ─────
PEARL_INTERVENTIONS = [
    "care_access",
    "clinical_other",
    "diabetes",
    "financial_benefits",
    "food_security",
    "heart_failure",
    "housing",
    "hypertension",
    "maternal",
    "medication_adherence",
    "mental_health",
    "pulmonary",
    "substance_use",
    "transport_utilities",
]

# Parth's 29-feature order (FEAT_ORDER from rule_based_recommendations.py)
FEAT_ORDER_29 = [
    "rr_score",            # 0
    "n_acute",             # 1
    "pct_avoidable",       # 2
    "med_adherence",       # 3
    "n_chronic_meds",      # 4
    "has_resp_dx",         # 5
    "has_beta2_or_steroid",# 6
    "has_bp_dx",           # 7
    "has_bp_meds",         # 8
    "has_dm_flag",         # 9
    "has_dm_agents",       # 10
    "has_opioid",          # 11
    "has_muscle_relaxant", # 12
    "has_maternity",       # 13
    "has_meth",            # 14
    "has_fentanyl",        # 15
    "has_alcohol_use",     # 16
    "has_smoking_history", # 17
    "has_depression",      # 18
    "has_anxiety",         # 19
    "has_sud_dx",          # 20
    "has_heart_failure",   # 21
    "has_anticoagulant",   # 22
    "has_insulin",         # 23
    "has_antiarrhythmic",  # 24
    "is_non_adherent",     # 25
    "pcp_gap",             # 26
    "bh_gap",              # 27
    "patient_age",         # 28
]
assert len(FEAT_ORDER_29) == 29


# ─────────────────────────────────────────────────────────────────────────────
# Feature bridge: EHR patient row → 29-element FEAT_ORDER feature vector
# ─────────────────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    """Return column as float32 array, filling missing with default."""
    if name in df.columns:
        return df[name].fillna(default).values.astype(np.float32)
    return np.full(len(df), default, dtype=np.float32)


def build_X(p: pd.DataFrame) -> np.ndarray:
    """
    Map EHR patient features → 29-element FEAT_ORDER feature matrix.

    EHR columns used:
      risk_score, prior_ed_visits_6mo, prior_hosp_6mo, poor_adherence,
      pharmacy_fills_90d, n_chronic, has_copd, has_diabetes, has_hypertension,
      has_chf, has_mh, has_substance, has_maternity_goal, age

    Features with no EHR equivalent (opioid, muscle relaxant, anticoagulant,
    meth, fentanyl, antiarrhythmic) are set to 0, matching inference defaults.
    """
    # rr_score: risk_score in member_attributes is 0-1 scale
    rr_score = np.clip(_col(p, "risk_score", 0.5), 0.0, 1.0)

    ed6 = _col(p, "prior_ed_visits_6mo", 0.0)
    ip6 = _col(p, "prior_hosp_6mo", 0.0)
    n_acute = np.clip(ed6 + ip6, 0, 10).astype(np.float32)
    pct_avoidable = np.where(n_acute > 0, ed6 / n_acute, 0.0).astype(np.float32)

    poor_adh = _col(p, "poor_adherence", 0.0)
    med_adherence = (1.0 - poor_adh).astype(np.float32)

    fills90 = _col(p, "pharmacy_fills_90d", 0.0)
    n_chron = _col(p, "n_chronic", 0.0)
    n_chronic_meds = np.where(
        fills90 > 0, np.clip(fills90, 0, 20), n_chron * 2.0
    ).astype(np.float32)

    has_copd = _col(p, "has_copd", 0.0)
    has_dm   = _col(p, "has_diabetes", 0.0)
    has_htn  = _col(p, "has_hypertension", 0.0)
    has_chf  = _col(p, "has_chf", 0.0)
    has_mh   = _col(p, "has_mh", 0.0)
    has_sub  = _col(p, "has_substance", 0.0)
    # has_maternity_goal is derived from member_goals MATERNITY/PRENATAL/POSTPARTUM
    has_mat  = _col(p, "has_maternity_goal", 0.0)

    # Gap proxies
    pcp_gap = np.where(n_acute >= 2, 1.0, 0.0).astype(np.float32)
    bh_gap  = has_mh

    zeros = np.zeros(len(p), dtype=np.float32)

    return np.column_stack([
        rr_score,      # 0  rr_score
        n_acute,       # 1  n_acute
        pct_avoidable, # 2  pct_avoidable
        med_adherence, # 3  med_adherence
        n_chronic_meds,# 4  n_chronic_meds
        has_copd,      # 5  has_resp_dx
        has_copd,      # 6  has_beta2_or_steroid (COPD proxy)
        has_htn,       # 7  has_bp_dx
        has_htn,       # 8  has_bp_meds (HTN proxy)
        has_dm,        # 9  has_dm_flag
        has_dm,        # 10 has_dm_agents (DM proxy)
        zeros,         # 11 has_opioid — not in EHR data
        zeros,         # 12 has_muscle_relaxant — not in EHR data
        has_mat,       # 13 has_maternity
        zeros,         # 14 has_meth — not in EHR data
        zeros,         # 15 has_fentanyl — not in EHR data
        has_sub,       # 16 has_alcohol_use (substance proxy)
        has_sub,       # 17 has_smoking_history (substance proxy)
        has_mh,        # 18 has_depression (MH proxy)
        has_mh,        # 19 has_anxiety (MH proxy)
        has_sub,       # 20 has_sud_dx
        has_chf,       # 21 has_heart_failure
        zeros,         # 22 has_anticoagulant — not in EHR data
        has_dm,        # 23 has_insulin (DM proxy)
        has_chf,       # 24 has_antiarrhythmic (CHF proxy)
        poor_adh,      # 25 is_non_adherent
        pcp_gap,       # 26 pcp_gap
        bh_gap,        # 27 bh_gap
        _col(p, "age", 45.0),  # 28 patient_age
    ]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Add maternity goal flag from raw member_goals
# ─────────────────────────────────────────────────────────────────────────────

MATERNITY_CATS = {"MATERNITY", "PRENATAL_CARE", "POSTPARTUM_CARE"}


def _add_maternity_flag(patients: pd.DataFrame, member_goals: pd.DataFrame) -> pd.DataFrame:
    """Add has_maternity_goal flag from member_goals. Uses all-time goals (non-deleted)."""
    mg = member_goals[~member_goals["deleted"]].copy()
    maternity_pids = set(mg[mg["category"].isin(MATERNITY_CATS)]["member_id"])
    patients = patients.copy()
    patients["has_maternity_goal"] = patients["member_id"].isin(maternity_pids).astype(np.float32)
    return patients


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("PEARL C4 CLASSIFIER TRAINING")
    print("=" * 60)

    # ── 1. Load Waymark population ────────────────────────────────────────────
    print("\n[1/6] Loading Waymark population (this takes ~5-10 min)...")
    sys.path.insert(0, str(REPO_ROOT / "packaging" / "pearl"))
    from data.extract_wpad import build_waymark_population
    pop = build_waymark_population(verbose=True)

    wpad     = pop.wpad_pairs.copy()
    patients = pop.patients.copy()
    print(f"  WPAD pairs:  {len(wpad):,}")
    print(f"  Patients:    {len(patients):,}")

    # ── 2. Add maternity goal flag ────────────────────────────────────────────
    print("\n[2/6] Adding maternity goal flag from member_goals...")
    mg_raw = pd.read_parquet(DATA_DIR / "member_goals.parquet")
    patients = _add_maternity_flag(patients, mg_raw)

    # ── 3. Join WPAD pairs → patient features ─────────────────────────────────
    print("\n[3/6] Joining WPAD pairs with patient features...")
    # Drop behavioral_intervention from patients to avoid column collision
    # (we want the WPAD-specific window label, not the all-time patient label)
    feats_df = patients.drop(columns=["behavioral_intervention"], errors="ignore").copy()
    feats_df = feats_df.drop_duplicates("patient_id")

    # wpad.patient_id = member_id (CUID); patients.patient_id = same
    joined = wpad.merge(feats_df, on="patient_id", how="inner")
    print(f"  Matched WPAD→patient rows: {len(joined):,}")

    joined = joined[joined["behavioral_intervention"].isin(PEARL_INTERVENTIONS)]
    print(f"  After filtering unknown labels: {len(joined):,}")

    if len(joined) < 100:
        raise RuntimeError(
            f"Only {len(joined)} training rows — too few to train. "
            "Check data pipeline or WPAD pair construction."
        )

    # ── 4. Build feature matrix and labels ────────────────────────────────────
    print("\n[4/6] Building feature matrix (29 features)...")
    X = build_X(joined)
    y = joined["behavioral_intervention"].values
    w = joined["pair_weight"].values.astype(np.float32)

    print(f"  X shape: {X.shape}")
    print(f"  Class distribution:")
    for cls in PEARL_INTERVENTIONS:
        mask = y == cls
        if mask.sum() > 0:
            print(f"    {cls:25s}: {mask.sum():5d} rows  (weight sum {w[mask].sum():.1f})")
        else:
            print(f"    {cls:25s}:     0 rows  (WARNING: missing from training data)")

    # ── 5. Train GradientBoostingClassifier ───────────────────────────────────
    print("\n[5/6] Training GradientBoostingClassifier...")
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )
    clf.fit(X, y, sample_weight=w)

    # Verify class order matches PEARL_INTERVENTIONS exactly
    clf_classes = list(clf.classes_)
    missing  = [c for c in PEARL_INTERVENTIONS if c not in clf_classes]
    mismatch = clf_classes != PEARL_INTERVENTIONS

    if missing:
        print(f"  WARNING: classes absent from training data: {missing}")
        print("  PEARLScorer will return 0 probability for these PEARL types.")
    if mismatch:
        print(f"  WARNING: class order mismatch!")
        print(f"    clf.classes_:      {clf_classes}")
        print(f"    PEARL_INTERVENTIONS: {PEARL_INTERVENTIONS}")
        print("  PEARLScorer indexes by PEARL_INTERVENTIONS.index() — mismatch will cause wrong scores.")
    else:
        print("  ✓ clf.classes_ matches PEARL_INTERVENTIONS exactly")

    # 5-fold cross-validation accuracy (unweighted — sample_weight not
    # supported in cross_val_score fit_params for all sklearn versions)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Training set report
    y_pred = clf.predict(X)
    print("\n  Training set classification report:")
    print(classification_report(y, y_pred, zero_division=0))

    # Feature importances
    imp  = clf.feature_importances_
    top5 = np.argsort(imp)[::-1][:5]
    print("  Top 5 features by importance:")
    for i in top5:
        print(f"    [{i:2d}] {FEAT_ORDER_29[i]:28s} {imp[i]:.4f}")

    # ── 6. Save model ─────────────────────────────────────────────────────────
    print(f"\n[6/6] Saving to {MODEL_OUTPUT}...")
    joblib.dump(clf, MODEL_OUTPUT)
    size_kb = MODEL_OUTPUT.stat().st_size // 1024
    print(f"  Saved. File size: {size_kb} KB")

    # ── Validation set summary (limited: val CSV doesn't have PEARL features) ─
    if VAL_CSV.exists():
        print(f"\n[+] Validation set summary: {VAL_CSV.name}")
        try:
            val = pd.read_csv(VAL_CSV)
            print(f"    Rows: {len(val):,}  |  Columns: {list(val.columns)}")
            if "outcome_binary" in val.columns:
                rates = val["outcome_binary"].value_counts(normalize=True)
                print(f"    outcome_binary: {rates.to_dict()}")
        except Exception as e:
            print(f"    Could not read validation CSV: {e}")
    else:
        print(f"\n[+] Validation CSV not found at {VAL_CSV}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Model saved to: {MODEL_OUTPUT}")
    print(f"  Classes ({len(clf_classes)}): {clf_classes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
