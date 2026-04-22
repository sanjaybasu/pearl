"""
PEARL next best action engine — drop-in replacement for rule_based_recommendations.py

Usage in recommendation_service.py (one-line change):
    # Before:
    from signal_recommendations.rule_based_recommendations import generate_rule_based_recommendations
    # After:
    from signal_recommendations.pearl_nba import generate_rule_based_recommendations

What changes vs. Parth's rule-based engine:
  1. Scoring: EFFECT_TABLE_PP hand-coded ARRs → PEARL classifier (C4 behavioral
     cloning SFT trained on WPAD within-patient natural experiment pairs).
     Best-performing method in the PEARL ablation study: IMI 4.4% vs. 27.0%
     behavioral baseline.
  2. Abstention: if the PEARL classifier's top probability < τ = 0.082, the
     engine falls back to Parth's rule-based ARR scoring transparently.

What does NOT change:
  - Input format:   generate_rule_based_recommendations(patient_prompts_list)
  - Output format:  same DataFrame schema (external_identifier, source, goal_tag,
                    title, recommendation_text, url, recommendation_order)
  - Workflows:      Parth's original 12 workflows exactly — no additions
  - Text parsing:   parse_block() and build_features() reused verbatim
  - Eligibility:    _eligible() reused verbatim
  - Rationale:      rationale_per_workflow() reused verbatim
  - Baseline risk:  _baseline_risk_proxy() reused verbatim

Classifier file:
  pearl_sft_clf.joblib — sklearn multi-class classifier trained on WPAD pairs.
  Features: Parth's 29 FEAT_ORDER features.
  Labels:   14 PEARL intervention types (from member_goals during ON-window).
  If absent: engine runs on Parth's rule-based ARR fallback transparently.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import Parth's implementation unchanged
# ---------------------------------------------------------------------------
from signal_recommendations.rule_based_recommendations import (  # noqa: E402
    FEAT_ORDER,
    WORKFLOW_GOALTAG_MAPPING,
    WORKFLOW_URLS,
    WORKFLOWS,
    _baseline_risk_proxy,
    _effect_pp_from_table,
    _eligible,
    build_features,
    nba_expected_reduction_with_rationales_gated,
    parse_block,
    rationale_per_workflow,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABSTENTION_THRESHOLD = 0.082   # τ: if max PEARL prob < τ, fall back to rule-based
MODEL_PATH = Path(__file__).parent / "pearl_sft_clf.joblib"

# PEARL intervention types trained on WPAD pairs (alphabetical — must match
# the order sklearn assigns to clf.classes_).
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

# PEARL intervention type → Parth workflow name.
# Workflows with no PEARL type (Gun Safety) always score via rule-based fallback.
PEARL_TO_WORKFLOW: Dict[str, str] = {
    "pulmonary":            "Asthma/COPD Routine Monitoring",
    "hypertension":         "Hypertension Routine Monitoring",
    "diabetes":             "Diabetes Routine Monitoring",
    "heart_failure":        "Heart Failure Routine Monitoring",
    "substance_use":        "Substance Use",
    "maternal":             "Maternity Support",
    "mental_health":        "Referral to Therapy",
    "medication_adherence": "Referral to Pharmacy",
    "clinical_other":       "Physical Activity-related Actions",
}
WORKFLOW_TO_PEARL: Dict[str, str] = {v: k for k, v in PEARL_TO_WORKFLOW.items()}
# Workflows that share a PEARL type
WORKFLOW_TO_PEARL["Tobacco Cessation"]    = "substance_use"
WORKFLOW_TO_PEARL["Diet Related Actions"] = "clinical_other"
# Gun Safety → no PEARL type; always rule-based fallback


# ---------------------------------------------------------------------------
# PEARL Scorer — replaces EFFECT_TABLE_PP for Parth's 12 workflows
# ---------------------------------------------------------------------------

class PEARLScorer:
    """
    Wraps the C4 behavioral cloning classifier (best-performing PEARL variant,
    IMI 4.4%) trained on WPAD within-patient natural experiment pairs.

    Input:  29-element feature vector (Parth's FEAT_ORDER).
    Output: per-workflow probability score replacing EFFECT_TABLE_PP ARRs.

    Falls back silently to Parth's rule-based ARR when:
      - pearl_sft_clf.joblib is absent or fails to load
      - The workflow has no PEARL intervention type (Gun Safety)
      - max classifier probability < ABSTENTION_THRESHOLD
    """
    _instance: "PEARLScorer | None" = None

    def __new__(cls) -> "PEARLScorer":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._clf = None
            obj._loaded = False
            cls._instance = obj
        return cls._instance

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if MODEL_PATH.exists():
            try:
                import joblib
                self._clf = joblib.load(MODEL_PATH)
            except Exception as exc:
                print(f"[PEARLScorer] Could not load {MODEL_PATH}: {exc}; using fallback.")

    def score(
        self,
        workflow: str,
        x_vec: np.ndarray,
        p0: float,
    ) -> float:
        """
        Return a PEARL probability score for this workflow for this patient.

        If the classifier is unavailable or the workflow has no PEARL type,
        returns Parth's EFFECT_TABLE_PP value unchanged.
        """
        self._load()
        pearl_type = WORKFLOW_TO_PEARL.get(workflow)

        if self._clf is not None and pearl_type is not None:
            try:
                proba = self._clf.predict_proba(x_vec.reshape(1, -1))[0]
                if pearl_type in PEARL_INTERVENTIONS:
                    idx = PEARL_INTERVENTIONS.index(pearl_type)
                    if idx < len(proba):
                        return float(proba[idx])
            except Exception:
                pass

        # Fallback: Parth's rule-based ARR
        return _effect_pp_from_table(workflow, x_vec)

    def max_prob(self, x_vec: np.ndarray) -> float:
        """Return the classifier's highest probability across all classes."""
        self._load()
        if self._clf is not None:
            try:
                return float(self._clf.predict_proba(x_vec.reshape(1, -1))[0].max())
            except Exception:
                pass
        return 1.0  # fallback: never abstain


# ---------------------------------------------------------------------------
# Core recommendation function
# ---------------------------------------------------------------------------

def nba_pearl_gated(
    text: str,
    k: int = 3,
    arr_threshold_pp: float = 0.2,
    min_baseline_risk: float = 0.02,
    capacity: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    PEARL-based next best action selector using Parth's 12 workflows.

    Identical to Parth's nba_expected_reduction_with_rationales_gated() except
    the workflow scoring uses PEARL classifier probabilities instead of
    EFFECT_TABLE_PP hand-coded ARRs. Falls back to EFFECT_TABLE_PP when the
    PEARL classifier is absent or confidence is below ABSTENTION_THRESHOLD.
    """
    feats   = parse_block(text)
    x_vec   = build_features(feats)         # 29-element Parth feature vector
    why_map = rationale_per_workflow(feats)
    p0      = _baseline_risk_proxy(feats)

    if p0 < min_baseline_risk:
        return {
            "baseline_risk_proxy": round(p0, 3),
            "recommendations": [],
            "display_text": "No high-impact workflow surfaced (very low baseline risk).",
        }

    cap    = {w: (capacity.get(w, 10**9) if capacity else 10**9) for w in WORKFLOWS}
    scorer = PEARLScorer()
    use_pearl = scorer.max_prob(x_vec) >= ABSTENTION_THRESHOLD

    ranked = []
    for action in WORKFLOWS:
        ok, _ = _eligible(action, feats, x_vec, cap)
        if not ok:
            continue

        if use_pearl:
            score = scorer.score(action, x_vec, p0)
        else:
            score = _effect_pp_from_table(action, x_vec)

        arr = float(min(score, p0 * 100.0))
        if arr >= arr_threshold_pp:
            ranked.append((action, arr))

    ranked = sorted(ranked, key=lambda z: z[1], reverse=True)

    if (feats.get("patient_age") or 0) <= 2:
        ranked = []

    top = []
    for action, arr in ranked:
        if cap[action] <= 0:
            continue
        top.append((action, round(arr, 2)))
        cap[action] -= 1
        if len(top) == k:
            break

    recs = []
    for action, arr in top:
        whys = why_map.get(action, [])
        rationale = " ".join(whys) if whys else "High expected reduction for this profile."
        recs.append({
            "goal_tag":            action,
            "title":               action,
            "recommendation_text": f"Signal suggests exploring the {action} workflow.",
            "Rationale":           rationale,
            "url":                 WORKFLOW_URLS.get(action),
        })

    return {"baseline_risk_proxy": round(p0, 3), "recommendations": recs}


# ---------------------------------------------------------------------------
# Public interface — exact drop-in replacement
# ---------------------------------------------------------------------------

def generate_rule_based_recommendations(
    patient_prompts_list: List[Any],
) -> pd.DataFrame:
    """
    PEARL-based recommendations. Identical signature and output schema to Parth's
    generate_rule_based_recommendations().

    Input:  list of patient dicts, each with:
              patient["prompt"]                    — raw patient summary text
              patient["id"]["external_identifier"] — patient ID
              patient["id"]["source"]              — source system
    Output: DataFrame with columns:
              external_identifier, source, goal_tag, title,
              recommendation_text, url, recommendation_order
    """
    recommendations_result_list = []
    for patient in patient_prompts_list:
        patient_recs = nba_pearl_gated(patient["prompt"], k=3)
        recommendations_result_list.append({
            "external_identifier": patient["id"]["external_identifier"],
            "source":              patient["id"]["source"],
            "recommendations":     patient_recs["recommendations"],
        })

    df = pd.DataFrame(recommendations_result_list)
    df = df.explode("recommendations").reset_index(drop=True)
    df = df.dropna(subset=["recommendations"])

    if df.empty:
        return df

    df = df.assign(
        recommendation_order=lambda d: d.groupby("external_identifier").cumcount() + 1
    )

    for col in ("goal_tag", "title", "recommendation_text", "Rationale", "url"):
        df[col] = df["recommendations"].apply(lambda x: x[col])
    df = df.drop(columns=["recommendations"])

    df["goal_tag"] = df["goal_tag"].map(WORKFLOW_GOALTAG_MAPPING)

    df["recommendation_text"] = (
        df["recommendation_text"] + "\nRationale: " + df["Rationale"]
    )
    df = df.drop(columns=["Rationale"])

    return df
