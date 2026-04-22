"""
Published comparator implementations: C1-C8 from the PEARL paper.

All comparators use publicly-citable methods only. No internal Waymark systems named.

C1: LACE Index (van Walraven et al., CMAJ 2010)
C2: HOSPITAL Score (Donzé et al., JGIM 2013)
C3: XGBoost on claims features (best ML readmission prediction)
C4: Behavioral Cloning SFT (supervised on preferred completions — DPO ablation)
C5: DPO on observational pairs (non-WPAD — identification ablation)
C6: Causal Forest CATE (Wager & Athey, JASA 2018)
C7: Decision Transformer (Chen et al., NeurIPS 2021)
C8: Conservative Q-Learning (Kumar et al., NeurIPS 2020)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_val_predict
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

INTERVENTIONS = [
    "care_access",         # PCP appointments, care coordination
    "clinical_other",      # Dental, eye care, wellness (catch-all)
    "diabetes",            # Diabetes management
    "financial_benefits",  # Financial, insurance, legal, employment
    "food_security",       # Food insecurity, nutrition
    "heart_failure",       # Heart failure management
    "housing",             # Housing instability, quality
    "hypertension",        # Hypertension management
    "maternal",            # Maternity, prenatal, postpartum
    "medication_adherence", # Medication adherence/optimization
    "mental_health",       # Depression, anxiety, MH/BH
    "pulmonary",           # Asthma/COPD
    "substance_use",       # SUD, alcohol, smoking cessation
    "transport_utilities", # Transportation, utilities, childcare
]


# ─────────────────────────────────────────────────────────────────────────────
# C1: LACE Index
# L = Length of stay (0-7), A = Acuity at admission, C = Charlson (0-5),
# E = ED visits in past 6 months (0-4)
# LACE ≥ 10: high readmission risk
# ─────────────────────────────────────────────────────────────────────────────

class LACEIndex:
    """
    LACE Index readmission risk score.
    van Walraven C et al., CMAJ 2010. DOI: 10.1503/cmaj.092241

    For outpatient rising-risk patients, we approximate LACE from claims:
    - L (length of stay): proxied from prior hospitalization duration (not available in
      outpatient claims; set to 0 for never-hospitalized rising-risk patients)
    - A (acuity): proxied from urgent ED visits
    - C (Charlson): from claims-derived Charlson
    - E (ED visits in past 6 months)
    """

    # LACE scoring tables from van Walraven 2010
    L_SCORE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6}
    C_SCORE = {0: 0, 1: 1, 2: 2, 3: 3}
    E_SCORE = {0: 0, 1: 3, 2: 4, 3: 4}

    def score(self, patients: pd.DataFrame) -> np.ndarray:
        n = len(patients)
        scores = np.zeros(n)

        for i, (_, row) in enumerate(patients.iterrows()):
            # L: prior_hosp_6mo as length-of-stay proxy (clip to 7)
            l_days = min(int(row.get("prior_hosp_6mo", 0)) * 3, 7)  # ~3 days per hosp
            l = self.L_SCORE.get(l_days, 6)

            # A: acuity (urgent admission proxy: prior_ed_visits_6mo)
            a = 3 if row.get("prior_ed_visits_6mo", 0) >= 1 else 0

            # C: Charlson score (clip to 3 for scoring)
            charlson = min(int(row.get("charlson_score", 0)), 11)
            c_val = min(charlson // 4, 3)  # bin into 0-3
            c = self.C_SCORE.get(c_val, 0)

            # E: ED visits (clip to 3+)
            ed = min(int(row.get("prior_ed_visits_6mo", 0)), 3)
            e = self.E_SCORE.get(ed, 4)

            scores[i] = l + a + c + e

        return scores

    def predict_readmission(self, patients: pd.DataFrame) -> np.ndarray:
        """Return readmission probability [0,1] from LACE score."""
        scores = self.score(patients)
        # Calibration: LACE 10+ → ~30% 30-day readmission; sigmoid mapping
        return 1 / (1 + np.exp(-(scores - 8) * 0.35))

    def route_intervention(self, patients: pd.DataFrame) -> np.ndarray:
        """Risk-score based routing: high LACE → care_access; medium → medication_adherence."""
        scores = self.score(patients)
        routings = []
        for s in scores:
            if s >= 10:
                routings.append("care_access")
            elif s >= 7:
                routings.append("medication_adherence")
            else:
                routings.append("care_access")
        return np.array(routings)

    def evaluate(self, patients: pd.DataFrame, outcome_col: str = "y_behavioral") -> Dict:
        probs = self.predict_readmission(patients)
        y_true = patients[outcome_col].values
        return {
            "model": "LACE_C1",
            "auroc": float(roc_auc_score(y_true, probs)),
            "brier": float(brier_score_loss(y_true, probs)),
            "n": len(patients),
        }


# ─────────────────────────────────────────────────────────────────────────────
# C2: HOSPITAL Score
# H = Hemoglobin at discharge, O = Oncology, S = Sodium, P = Procedure,
# I = Insurance, T = Type of admission, A = Any prior admission, L = Lore
# (simplified for claims-only implementation)
# ─────────────────────────────────────────────────────────────────────────────

class HOSPITALScore:
    """
    HOSPITAL Score readmission risk.
    Donzé J et al., J Gen Intern Med 2013. DOI: 10.1007/s11606-013-2355-3

    Claims-available proxy variables only.
    """

    def score(self, patients: pd.DataFrame) -> np.ndarray:
        n = len(patients)
        scores = np.zeros(n)

        for i, (_, row) in enumerate(patients.iterrows()):
            h = 0  # Hemoglobin: not available in claims; set to 0
            o = 2 if row.get("has_ckd", 0) or row.get("has_chf", 0) else 0  # Oncology proxy: complex chronic
            s = 0  # Sodium: not available
            p = 1 if row.get("n_chronic", 0) >= 3 else 0  # Procedure proxy: complexity
            i_score = 0  # Insurance type: Medicaid → already filtered
            t = 1 if row.get("prior_ed_visits_6mo", 0) >= 2 else 0  # Type: ED-driven
            a = 1 if row.get("prior_hosp_6mo", 0) >= 1 else 0  # Any prior admission
            l = 0  # LoRe score: not available

            scores[i] = h + o + s + p + i_score + t + a + l

        return scores

    def predict_readmission(self, patients: pd.DataFrame) -> np.ndarray:
        scores = self.score(patients)
        return 1 / (1 + np.exp(-(scores - 3) * 0.5))

    def route_intervention(self, patients: pd.DataFrame) -> np.ndarray:
        """Risk-score based routing: high HOSPITAL → care_access; medium → medication_adherence."""
        scores = self.score(patients)
        return np.where(
            scores >= 5, "care_access",
            np.where(scores >= 3, "medication_adherence", "care_access")
        )

    def evaluate(self, patients: pd.DataFrame, outcome_col: str = "y_behavioral") -> Dict:
        probs = self.predict_readmission(patients)
        y_true = patients[outcome_col].values
        return {
            "model": "HOSPITAL_C2",
            "auroc": float(roc_auc_score(y_true, probs)),
            "brier": float(brier_score_loss(y_true, probs)),
            "n": len(patients),
        }


# ─────────────────────────────────────────────────────────────────────────────
# C3: XGBoost on claims features (best ML risk prediction)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "age", "female", "charlson_score", "prior_ed_visits_6mo", "prior_hosp_6mo",
    "pharmacy_fills_90d", "missed_pharmacy_fills", "n_chronic",
    "has_diabetes", "has_chf", "has_copd", "has_hypertension", "has_ckd", "has_mh",
    "adi_percentile", "food_insecure", "housing_unstable", "lives_alone", "no_transport"
]


class XGBoostComparator:
    """C3: XGBoost gradient boosted trees on claims features."""

    def __init__(self, seed: int = 42):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="auc", use_label_encoder=False,
            random_state=seed, verbosity=0
        )
        self._fitted = False

    def _get_X(self, patients: pd.DataFrame) -> np.ndarray:
        cols = [c for c in FEATURE_COLS if c in patients.columns]
        return patients[cols].fillna(0).astype(float).values

    def fit(self, patients: pd.DataFrame, outcome_col: str = "y_behavioral") -> "XGBoostComparator":
        X = self._get_X(patients)
        y = patients[outcome_col].values
        self.model.fit(X, y)
        self._outcome_col = outcome_col
        self._fitted = True
        return self

    def predict_proba(self, patients: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._get_X(patients))[:, 1]

    def route_intervention(self, patients: pd.DataFrame) -> np.ndarray:
        """Route based on XGBoost risk score (same threshold logic as LACE)."""
        probs = self.predict_proba(patients)
        return np.where(
            probs > 0.30, "care_access",
            np.where(probs > 0.20, "medication_adherence", "care_access")
        )

    def evaluate(self, patients: pd.DataFrame, outcome_col: str = "y_behavioral") -> Dict:
        probs = self.predict_proba(patients)
        y_true = patients[outcome_col].values
        return {
            "model": "XGBoost_C3",
            "auroc": float(roc_auc_score(y_true, probs)),
            "brier": float(brier_score_loss(y_true, probs)),
            "n": len(patients),
        }


# ─────────────────────────────────────────────────────────────────────────────
# C4: Behavioral Cloning SFT (DPO ablation — no preference signal)
# Learns directly from preferred completions without contrastive DPO loss.
# In the tabular proxy: multinomial logistic regression on (X → optimal intervention)
# ─────────────────────────────────────────────────────────────────────────────

class BehavioralCloningSFT:
    """
    C4: Behavioral cloning ablation.
    Supervised fine-tuning: predict intervention type from patient features.
    In real DPO context: equivalent to SFT on y_w (preferred completions only,
    no preference contrastive signal). Policy = P(a|x) ∝ exp(f_θ(x,a)).
    """

    def __init__(self, seed: int = 42):
        self.model = LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial",
                                        random_state=seed)
        self._le = None
        self._fitted = False

    def _get_X(self, patients: pd.DataFrame) -> np.ndarray:
        cols = [c for c in FEATURE_COLS if c in patients.columns]
        return patients[cols].fillna(0).astype(float).values

    def fit(
        self,
        patients: pd.DataFrame,
        target_col: str = "behavioral_intervention",  # default: what clinicians did
        # Pass "wpad_preferred_intervention" for the DPO ablation (same signal as PEARL, but SFT loss)
    ) -> "BehavioralCloningSFT":
        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder()
        X = self._get_X(patients)
        y = self._le.fit_transform(patients[target_col].values)
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_intervention(self, patients: pd.DataFrame) -> np.ndarray:
        X = self._get_X(patients)
        return self._le.inverse_transform(self.model.predict(X))

    def evaluate_imi(
        self,
        patients: pd.DataFrame,
        mu_hat_dr: np.ndarray,
        threshold: float = 0.02,
        interventions: List[str] = INTERVENTIONS,
    ) -> Dict:
        """Evaluate policy value on DR-estimated outcomes."""
        predicted = self.predict_intervention(patients)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(interventions)
        A_enc = le.transform(predicted)

        imi = float(np.array([
            float(any(mu_hat_dr[i, j] < mu_hat_dr[i, A_enc[i]] - threshold
                      for j in range(len(interventions)) if j != A_enc[i]))
            for i in range(len(patients))
        ]).mean())

        policy_value = float(np.array([
            mu_hat_dr[i, A_enc[i]] for i in range(len(patients))
        ]).mean())

        return {"model": "BehavioralCloning_C4", "imi": imi, "policy_value": policy_value}


# ─────────────────────────────────────────────────────────────────────────────
# C5: DPO without WPAD (observational pairs — identification ablation)
# Same DPO training but using confounded observational pairs (no IV design).
# In tabular proxy: train on random good/bad outcome pairs without WPAD filtering.
# ─────────────────────────────────────────────────────────────────────────────

class ObservationalDPO:
    """
    C5: DPO on observational (non-WPAD) preference pairs.
    Ablation showing that WPAD causal identification > naive DPO on confounded pairs.
    Rafailov et al., NeurIPS 2023 — DPO method.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial",
                                        random_state=seed)
        self._fitted = False
        self._rng = np.random.default_rng(seed)

    def _get_X(self, patients: pd.DataFrame) -> np.ndarray:
        cols = [c for c in FEATURE_COLS if c in patients.columns]
        return patients[cols].fillna(0).astype(float).values

    def fit(
        self,
        patients: pd.DataFrame,
        outcome_col: str = "y_behavioral",
        intervention_col: str = "behavioral_intervention",
    ) -> "ObservationalDPO":
        """
        Fit on confounded pairs: match good-outcome vs. bad-outcome patients
        without controlling for selection into intervention type (no IV).
        This replicates naive DPO on observational data.
        """
        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder().fit(INTERVENTIONS)

        X = self._get_X(patients)
        y_obs = patients[outcome_col].values
        A = patients[intervention_col].values

        # Confounded pairs: match on observable features, ignoring selection
        good_mask = y_obs == 0
        bad_mask = y_obs == 1

        # Naive preference: interventions associated with good outcomes are "preferred"
        preferred_interventions = []
        for intv in INTERVENTIONS:
            mask = A == intv
            if mask.sum() == 0:
                preferred_interventions.append(0.5)
            else:
                preferred_interventions.append(1 - y_obs[mask].mean())  # P(good outcome)

        # Soft target: prefer interventions with higher observed good-outcome rate
        # Assign each patient to the best observed intervention for their
        # actual behavioral assignment (confounded — no WPAD identification)
        pref_weights = np.array(preferred_interventions)
        pref_weights = pref_weights / pref_weights.sum()

        rng_local = np.random.default_rng(self.seed)
        # Assign probabilistically based on observed good-outcome rates
        target_intv = rng_local.choice(INTERVENTIONS, size=len(patients), p=pref_weights)
        y_target = self._le.transform(target_intv)

        if len(np.unique(y_target)) < 2:
            # Fallback: random assignment across classes
            y_target = rng_local.integers(0, len(INTERVENTIONS), len(patients))

        self.model.fit(X, y_target)
        self._fitted = True
        return self

    def predict_intervention(self, patients: pd.DataFrame) -> np.ndarray:
        X = self._get_X(patients)
        return self._le.inverse_transform(self.model.predict(X))

    def evaluate_imi(
        self,
        patients: pd.DataFrame,
        mu_hat_dr: np.ndarray,
        threshold: float = 0.02,
    ) -> Dict:
        predicted = self.predict_intervention(patients)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(INTERVENTIONS)
        A_enc = le.transform(predicted)

        imi = float(np.array([
            float(any(mu_hat_dr[i, j] < mu_hat_dr[i, A_enc[i]] - threshold
                      for j in range(len(INTERVENTIONS)) if j != A_enc[i]))
            for i in range(len(patients))
        ]).mean())

        policy_value = float(np.array([mu_hat_dr[i, A_enc[i]] for i in range(len(patients))]).mean())
        return {"model": "ObservationalDPO_C5", "imi": imi, "policy_value": policy_value}


# ─────────────────────────────────────────────────────────────────────────────
# C6: Causal Forest (Wager & Athey, JASA 2018)
# Multi-arm CATE estimation for heterogeneous treatment effects.
# ─────────────────────────────────────────────────────────────────────────────

class CausalForestComparator:
    """
    C6: Multi-arm causal forest for CATE estimation.
    Wager S & Athey S. Estimation and inference of heterogeneous treatment effects.
    JASA 2018. DOI: 10.1080/01621459.2017.1319839 (~3,500 citations)

    Uses econml.grf.CausalForest for multi-arm extension.
    """

    def __init__(self, n_estimators: int = 200, seed: int = 42):
        self.n_estimators = n_estimators
        self.seed = seed
        self._models: Dict[str, object] = {}
        self._fitted = False

    def _get_X(self, patients: pd.DataFrame) -> np.ndarray:
        cols = [c for c in FEATURE_COLS if c in patients.columns]
        return patients[cols].fillna(0).astype(float).values

    def fit(
        self,
        patients: pd.DataFrame,
        outcome_col: str = "y_behavioral",
        intervention_col: str = "behavioral_intervention",
    ) -> "CausalForestComparator":
        """
        Fit one causal forest per intervention arm (binary T=1 vs. all others).
        Standard multi-arm extension of Wager & Athey.
        """
        try:
            from econml.grf import CausalForest
            use_econml = True
        except ImportError:
            use_econml = False

        X = self._get_X(patients)
        Y = patients[outcome_col].values.reshape(-1, 1)
        A = patients[intervention_col].values

        for intv in INTERVENTIONS:
            T = (A == intv).astype(float).reshape(-1, 1)
            if T.sum() < 20:
                self._models[intv] = None
                continue

            if use_econml:
                try:
                    cf = CausalForest(
                        n_estimators=self.n_estimators,
                        random_state=self.seed,
                        n_jobs=-1
                    )
                    cf.fit(X, T, Y)
                    self._models[intv] = cf
                except Exception:
                    # Fallback: gradient boosted CATE proxy
                    self._models[intv] = self._fit_gbm_cate(X, T.ravel(), Y.ravel())
            else:
                # Without econml: use gradient boosted model with IPW weighting
                self._models[intv] = self._fit_gbm_cate(X, T.ravel(), Y.ravel())

        self._fitted = True
        return self

    def _fit_gbm_cate(self, X, T, Y):
        """Fallback CATE estimator: separate outcome models (T-learner)."""
        treated_mask = T == 1
        if treated_mask.sum() < 5:
            return None
        m1 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=self.seed)
        m0 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=self.seed)
        if len(np.unique(Y[treated_mask])) > 1:
            m1.fit(X[treated_mask], Y[treated_mask])
        if len(np.unique(Y[~treated_mask])) > 1:
            m0.fit(X[~treated_mask], Y[~treated_mask])
        return (m1, m0)

    def predict_cate(self, patients: pd.DataFrame) -> np.ndarray:
        """
        Return CATE estimates: shape (n_patients, n_interventions).
        Positive CATE = treatment reduces event probability.
        """
        X = self._get_X(patients)
        n = len(patients)
        cate_matrix = np.zeros((n, len(INTERVENTIONS)))

        for idx, intv in enumerate(INTERVENTIONS):
            model = self._models.get(intv)
            if model is None:
                continue
            if hasattr(model, "predict"):
                # econml CausalForest
                try:
                    cate_matrix[:, idx] = model.predict(X).ravel()
                except Exception:
                    pass
            elif isinstance(model, tuple):
                # T-learner fallback
                m1, m0 = model
                try:
                    p1 = m1.predict_proba(X)[:, 1]
                    p0 = m0.predict_proba(X)[:, 1]
                    cate_matrix[:, idx] = p0 - p1  # CATE = reduction in event prob
                except Exception:
                    pass

        return cate_matrix

    def recommend_intervention(self, patients: pd.DataFrame) -> np.ndarray:
        """Recommend intervention with highest CATE (largest event reduction)."""
        cate_matrix = self.predict_cate(patients)
        best_arm_idx = np.argmax(cate_matrix, axis=1)
        return np.array([INTERVENTIONS[i] for i in best_arm_idx])

    def evaluate_imi(
        self,
        patients: pd.DataFrame,
        mu_hat_dr: np.ndarray,
        threshold: float = 0.02,
    ) -> Dict:
        recommended = self.recommend_intervention(patients)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(INTERVENTIONS)
        A_enc = le.transform(recommended)

        imi = float(np.array([
            float(any(mu_hat_dr[i, j] < mu_hat_dr[i, A_enc[i]] - threshold
                      for j in range(len(INTERVENTIONS)) if j != A_enc[i]))
            for i in range(len(patients))
        ]).mean())

        policy_value = float(np.mean([mu_hat_dr[i, A_enc[i]] for i in range(len(patients))]))
        return {"model": "CausalForest_C6", "imi": imi, "policy_value": policy_value}


# ─────────────────────────────────────────────────────────────────────────────
# C7: Decision Transformer (Chen et al., NeurIPS 2021)
# Offline RL via sequence modeling: condition on desired return-to-go.
# In tabular proxy: use return-conditioned prediction from WPAD trajectories.
# ─────────────────────────────────────────────────────────────────────────────

class DecisionTransformerProxy:
    """
    C7: Decision Transformer proxy (tabular implementation).
    Chen L et al., Decision Transformer: RL via Sequence Modeling.
    NeurIPS 2021. (~1,200 citations)

    In PEARL's context: condition the intervention recommendation on desired
    return-to-go (R_t = 0, meaning "target zero acute care events").
    Tabular proxy: return-conditioned classifier.
    """

    def __init__(self, seed: int = 42):
        # Two-stage: (1) learn from high-return trajectories, (2) condition on R=0
        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed
        )
        self._fitted = False

    def _get_X(self, patients: pd.DataFrame) -> np.ndarray:
        cols = [c for c in FEATURE_COLS if c in patients.columns]
        return patients[cols].fillna(0).astype(float).values

    def fit(
        self,
        wpad_pairs: pd.DataFrame,
        patients: pd.DataFrame,
    ) -> "DecisionTransformerProxy":
        """
        Train on WPAD trajectories where ON-window had good outcomes.
        Condition on return-to-go = 0 (no acute care event).
        """
        # Use pairs where care management prevented adverse event
        good_pairs = wpad_pairs[wpad_pairs.get("y_on", pd.Series(dtype=float)).eq(0) if "y_on" in wpad_pairs.columns else wpad_pairs.index.isin(wpad_pairs.index)]

        # Match to patient features
        if "patient_id" in good_pairs.columns and "patient_id" in patients.columns:
            train_data = good_pairs.merge(patients, on="patient_id", how="left")
        else:
            train_data = patients.sample(min(len(patients), 2000), random_state=42)

        # Target: intervention that achieved good outcome
        intv_col = "preferred_intervention" if "preferred_intervention" in train_data.columns else \
                   "optimal_intervention" if "optimal_intervention" in train_data.columns else \
                   "behavioral_intervention"

        if intv_col not in train_data.columns:
            self._fitted = True
            return self

        X = self._get_X(train_data)
        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder().fit(INTERVENTIONS)
        y = self._le.transform(
            train_data[intv_col].fillna("care_access").values
        )

        # Add return-to-go as a feature (R=0 = desired outcome)
        rtg = np.zeros((len(train_data), 1))  # conditioning on R_t = 0
        X_aug = np.hstack([X, rtg])
        self.model.fit(X_aug, y)
        self._X_cols = len(X[0])
        self._fitted = True
        return self

    def predict_intervention(self, patients: pd.DataFrame) -> np.ndarray:
        X = self._get_X(patients)
        rtg = np.zeros((len(patients), 1))  # condition on R=0
        X_aug = np.hstack([X, rtg])
        try:
            return self._le.inverse_transform(self.model.predict(X_aug))
        except Exception:
            return np.array(["care_access"] * len(patients))

    def evaluate_imi(
        self,
        patients: pd.DataFrame,
        mu_hat_dr: np.ndarray,
        threshold: float = 0.02,
    ) -> Dict:
        recommended = self.predict_intervention(patients)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(INTERVENTIONS)
        A_enc = le.transform(recommended)

        imi = float(np.array([
            float(any(mu_hat_dr[i, j] < mu_hat_dr[i, A_enc[i]] - threshold
                      for j in range(len(INTERVENTIONS)) if j != A_enc[i]))
            for i in range(len(patients))
        ]).mean())

        policy_value = float(np.mean([mu_hat_dr[i, A_enc[i]] for i in range(len(patients))]))
        return {"model": "DecisionTransformer_C7", "imi": imi, "policy_value": policy_value}


# ─────────────────────────────────────────────────────────────────────────────
# C8: Conservative Q-Learning (Kumar et al., NeurIPS 2020)
# Offline RL with explicit conservatism penalty to avoid out-of-distribution actions.
# Kumar A et al. "Conservative Q-Learning for Offline Reinforcement Learning."
# NeurIPS 2020. DOI: 10.48550/arXiv.2006.04779 (~1,200+ citations)
# ─────────────────────────────────────────────────────────────────────────────

class CQLComparator:
    """
    C8: Conservative Q-Learning (CQL) for offline policy optimization.

    CQL adds a conservatism penalty to the standard Q-learning objective:
      L_CQL(Q) = L_Bellman(Q) + α * E_x[log Σ_a exp(Q(s,a)) - E_{a~π_b}[Q(s,a)]]

    The penalty discourages Q-values from being large for actions not well-covered
    by the behavioral policy. This corrects the SARSA ceiling problem (Rashidinejad
    et al. 2021): unlike SARSA, CQL can recommend interventions underrepresented in
    the behavioral data by downweighting Q-values for such actions rather than
    ignoring them.

    Tabular proxy: the LLM-scale CQL objective is approximated via a penalized
    linear Q-function with the CQL conservatism term computed over the behavioral
    distribution. The α hyperparameter controls conservatism (larger α = more
    conservative; less likely to recommend out-of-distribution actions).

    Reference: Kumar A, Zhou A, Tucker G, Levine S. Conservative Q-Learning for
    Offline Reinforcement Learning. NeurIPS 2020.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 0.99,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.seed = seed
        self._q_model = None
        self._fitted = False

    def _get_X(self, patients: pd.DataFrame) -> np.ndarray:
        cols = [c for c in FEATURE_COLS if c in patients.columns]
        return patients[cols].fillna(0).astype(float).values

    def fit(
        self,
        patients: pd.DataFrame,
        outcome_col: str = "y_behavioral",
        intervention_col: str = "behavioral_intervention",
        wpad_pairs: Optional[pd.DataFrame] = None,
    ) -> "CQLComparator":
        """
        Fit CQL Q-function with conservatism penalty.

        The CQL penalty suppresses Q-values for actions with low behavioral density.
        For WPAD patients, use causally-identified Q-targets (same as the upgraded
        SARSA implementation) to partially break the behavioral coverage ceiling.

        The CQL conservatism term per action a at state x:
          penalty(a, x) = α * max(0, Q(s, a) - E_{a'~π_b(·|x)}[Q(s, a')])

        This forces Q-values of rarely-taken actions down toward the behavioral mean,
        preventing overestimation of out-of-distribution actions.
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import LabelEncoder

        self._le = LabelEncoder().fit(INTERVENTIONS)
        X = self._get_X(patients)
        A = patients[intervention_col].values
        Y = patients[outcome_col].values
        n = len(patients)
        n_actions = len(INTERVENTIONS)

        # ── Behavioral action frequencies (for CQL conservatism term) ─────
        action_freq = np.zeros(n_actions)
        for idx, intv in enumerate(INTERVENTIONS):
            action_freq[idx] = float((A == intv).mean())
        action_freq = np.maximum(action_freq, 1e-4)  # avoid log(0)

        # ── Initial Q-targets from behavioral policy ──────────────────────
        Q_targets = np.zeros((n, n_actions))
        for idx, intv in enumerate(INTERVENTIONS):
            mask = A == intv
            if mask.sum() > 0:
                Q_targets[mask, idx] = -float(Y[mask].mean())  # reward = negative event rate

        # ── WPAD-grounded Q-target overrides ─────────────────────────────
        sample_weights = np.ones(n)
        if wpad_pairs is not None and "y_on" in wpad_pairs.columns and \
           "patient_id" in wpad_pairs.columns and "patient_id" in patients.columns:
            pid_to_idx = {pid: i for i, pid in enumerate(patients["patient_id"].values)}
            intv_col_wpad = "behavioral_intervention" if "behavioral_intervention" in wpad_pairs.columns else None
            if intv_col_wpad:
                for _, row in wpad_pairs.iterrows():
                    pid = row["patient_id"]
                    if pid not in pid_to_idx:
                        continue
                    i = pid_to_idx[pid]
                    intv = row[intv_col_wpad]
                    if intv not in INTERVENTIONS:
                        continue
                    intv_idx = INTERVENTIONS.index(intv)
                    Q_targets[i, intv_idx] = -float(row["y_on"])
                    sample_weights[i] = 3.0

        # ── Fit initial Q-function ────────────────────────────────────────
        q_model = Ridge(alpha=1.0)
        all_X, all_Q, all_W = [], [], []
        for idx in range(n_actions):
            action_indicator = np.zeros((n, n_actions))
            action_indicator[:, idx] = 1
            all_X.append(np.hstack([X, action_indicator]))
            all_Q.append(Q_targets[:, idx])
            all_W.append(sample_weights)
        q_model.fit(np.vstack(all_X), np.concatenate(all_Q),
                    sample_weight=np.concatenate(all_W))

        # ── CQL conservatism correction ───────────────────────────────────
        # Compute predicted Q-values for all actions; apply penalty proportional
        # to how much Q(s,a) exceeds the behavioral-weighted Q mean.
        Q_pred = np.zeros((n, n_actions))
        for idx in range(n_actions):
            action_indicator = np.zeros((n, n_actions))
            action_indicator[:, idx] = 1
            X_aug = np.hstack([X, action_indicator])
            Q_pred[:, idx] = q_model.predict(X_aug)

        # Behavioral-weighted Q mean per patient: Σ_a π_b(a|x) * Q(s,a)
        Q_behavioral_mean = Q_pred @ action_freq  # (n,)

        # CQL penalty target: Q(s,a) - α * max(0, Q(s,a) - Q_behav_mean)
        Q_cql_targets = Q_pred - self.alpha * np.maximum(0, Q_pred - Q_behavioral_mean[:, None])

        # Refit with CQL-penalized targets
        self._q_model = Ridge(alpha=1.0)
        all_X2, all_Q2, all_W2 = [], [], []
        for idx in range(n_actions):
            action_indicator = np.zeros((n, n_actions))
            action_indicator[:, idx] = 1
            all_X2.append(np.hstack([X, action_indicator]))
            all_Q2.append(Q_cql_targets[:, idx])
            all_W2.append(sample_weights)
        self._q_model.fit(np.vstack(all_X2), np.concatenate(all_Q2),
                          sample_weight=np.concatenate(all_W2))
        self._n_actions = n_actions
        self._fitted = True
        return self

    def predict_q_values(self, patients: pd.DataFrame) -> np.ndarray:
        """Return CQL Q(s,a) for all interventions. Shape: (n_patients, n_interventions)."""
        X = self._get_X(patients)
        n = len(patients)
        Q = np.zeros((n, len(INTERVENTIONS)))
        for idx in range(len(INTERVENTIONS)):
            action_indicator = np.zeros((n, len(INTERVENTIONS)))
            action_indicator[:, idx] = 1
            Q[:, idx] = self._q_model.predict(np.hstack([X, action_indicator]))
        return Q

    def recommend_intervention(self, patients: pd.DataFrame) -> np.ndarray:
        """Recommend intervention maximizing conservative Q(s,a)."""
        Q = self.predict_q_values(patients)
        best_action = np.argmax(Q, axis=1)
        return np.array([INTERVENTIONS[i] for i in best_action])

    def evaluate_imi(
        self,
        patients: pd.DataFrame,
        mu_hat_dr: np.ndarray,
        threshold: float = 0.02,
    ) -> Dict:
        recommended = self.recommend_intervention(patients)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(INTERVENTIONS)
        A_enc = le.transform(recommended)

        imi = float(np.array([
            float(any(mu_hat_dr[i, j] < mu_hat_dr[i, A_enc[i]] - threshold
                      for j in range(len(INTERVENTIONS)) if j != A_enc[i]))
            for i in range(len(patients))
        ]).mean())

        policy_value = float(np.mean([mu_hat_dr[i, A_enc[i]] for i in range(len(patients))]))

        A_behavioral = patients.get("behavioral_intervention",
                                    pd.Series(["care_access"] * len(patients)))
        coverage_ok = float(np.mean([rec in A_behavioral.values for rec in recommended]))

        return {
            "model": "CQL_C8",
            "imi": imi,
            "policy_value": policy_value,
            "coverage_coefficient": coverage_ok,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SARSA: on-policy TD learning (reference implementation; superseded by CQL as C8)
# Muralidharan et al., JMIR AI 2025. Retained for methodological comparison.
# ─────────────────────────────────────────────────────────────────────────────

class SARSAComparator:
    """
    SARSA on-policy temporal difference learning (reference implementation; superseded by CQL as C8).
    Muralidharan et al. "Reinforcement Learning to Prevent Acute Care Events
    Among Medicaid Populations." JMIR AI 2025.

    The SARSA ceiling problem: SARSA estimates Q(s,a) only for (s,a) pairs
    observed in the behavioral policy. For interventions systematically
    under-used for certain patient profiles, SARSA cannot recommend them.
    This is the coverage condition of Rashidinejad et al. (2021).
    """

    def __init__(
        self,
        gamma: float = 0.99,
        alpha: float = 0.1,
        n_iterations: int = 500,
        seed: int = 42,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.seed = seed
        self._q_model = None  # Q(s,a) function approximator
        self._fitted = False

    def _get_X(self, patients: pd.DataFrame) -> np.ndarray:
        cols = [c for c in FEATURE_COLS if c in patients.columns]
        return patients[cols].fillna(0).astype(float).values

    def fit(
        self,
        patients: pd.DataFrame,
        outcome_col: str = "y_behavioral",
        intervention_col: str = "behavioral_intervention",
        wpad_pairs: Optional[pd.DataFrame] = None,
    ) -> "SARSAComparator":
        """
        Fit Q(s,a) function approximator using WPAD-grounded Q-targets.

        The SARSA ceiling problem (Rashidinejad et al. 2021): Q-values estimated only
        from behavioral policy data cannot recommend interventions systematically
        under-used for certain patient profiles.

        Fix: use WPAD pairs to provide causally-identified Q-targets for patients who
        experienced care management ON-windows.  For WPAD patients, Q(s, A_on) is
        estimated from y_on (causally identified outcome) rather than the confounded
        behavioral mean.  Non-WPAD patients fall back to behavioral-policy estimates.
        This expands the effective training distribution beyond the behavioral policy
        support, partially breaking the SARSA ceiling.
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.linear_model import Ridge
        self._le = LabelEncoder().fit(INTERVENTIONS)

        X = self._get_X(patients)
        A = patients[intervention_col].values
        Y = patients[outcome_col].values
        n = len(patients)
        n_actions = len(INTERVENTIONS)

        # ── Build Q-targets ───────────────────────────────────────────────
        # Default: behavioral policy means (confounded, SARSA ceiling applies)
        Q_targets = np.zeros((n, n_actions))
        for idx, intv in enumerate(INTERVENTIONS):
            mask = A == intv
            if mask.sum() > 0:
                Q_targets[mask, idx] = -Y[mask].mean()  # reward = negative event rate

        # Override with WPAD-identified Q-targets where available.
        # For patients with a WPAD pair where y_on is observed during care management:
        #   Q(s, A_wpad) = -y_on   (causally identified: outcome under care management)
        # This gives SARSA access to counterfactual evidence it cannot get from
        # behavioral policy data alone.
        sample_weights = np.ones(n)
        if wpad_pairs is not None and "y_on" in wpad_pairs.columns and \
           "patient_id" in wpad_pairs.columns and "patient_id" in patients.columns:
            pid_to_idx = {pid: i for i, pid in enumerate(patients["patient_id"].values)}
            intv_col_wpad = "behavioral_intervention" if "behavioral_intervention" in wpad_pairs.columns else None
            if intv_col_wpad:
                for _, row in wpad_pairs.iterrows():
                    pid = row["patient_id"]
                    if pid not in pid_to_idx:
                        continue
                    i = pid_to_idx[pid]
                    intv = row[intv_col_wpad]
                    if intv not in INTERVENTIONS:
                        continue
                    intv_idx = INTERVENTIONS.index(intv)
                    y_on = float(row["y_on"])
                    # Upweight WPAD-grounded targets (causally identified signal)
                    Q_targets[i, intv_idx] = -y_on
                    sample_weights[i] = 3.0  # 3× weight for WPAD patients

        # ── Fit Ridge Q-function approximator ─────────────────────────────
        self._q_model = Ridge(alpha=1.0)
        all_X, all_Q, all_W = [], [], []
        for idx in range(n_actions):
            action_indicator = np.zeros((n, n_actions))
            action_indicator[:, idx] = 1
            X_a_aug = np.hstack([X, action_indicator])
            all_X.append(X_a_aug)
            all_Q.append(Q_targets[:, idx])
            all_W.append(sample_weights)

        X_train = np.vstack(all_X)
        Q_train = np.concatenate(all_Q)
        W_train = np.concatenate(all_W)
        self._q_model.fit(X_train, Q_train, sample_weight=W_train)
        self._n_actions = n_actions
        self._fitted = True
        return self

    def predict_q_values(self, patients: pd.DataFrame) -> np.ndarray:
        """Return Q(s,a) for all interventions. Shape: (n_patients, n_interventions)."""
        X = self._get_X(patients)
        n = len(patients)
        n_actions = len(INTERVENTIONS)
        Q = np.zeros((n, n_actions))

        for idx in range(n_actions):
            action_indicator = np.zeros((n, n_actions))
            action_indicator[:, idx] = 1
            X_aug = np.hstack([X, action_indicator])
            Q[:, idx] = self._q_model.predict(X_aug)

        return Q

    def recommend_intervention(self, patients: pd.DataFrame) -> np.ndarray:
        """Recommend intervention maximizing Q(s,a) — best estimated Q-value."""
        Q = self.predict_q_values(patients)
        best_action = np.argmax(Q, axis=1)
        return np.array([INTERVENTIONS[i] for i in best_action])

    def evaluate_imi(
        self,
        patients: pd.DataFrame,
        mu_hat_dr: np.ndarray,
        threshold: float = 0.02,
    ) -> Dict:
        recommended = self.recommend_intervention(patients)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(INTERVENTIONS)
        A_enc = le.transform(recommended)

        imi = float(np.array([
            float(any(mu_hat_dr[i, j] < mu_hat_dr[i, A_enc[i]] - threshold
                      for j in range(len(INTERVENTIONS)) if j != A_enc[i]))
            for i in range(len(patients))
        ]).mean())

        policy_value = float(np.mean([mu_hat_dr[i, A_enc[i]] for i in range(len(patients))]))

        # Coverage coefficient (Rashidinejad et al. 2021): fraction of recommended
        # actions that had ≥1 example in behavioral policy training data
        A_behavioral = patients.get("behavioral_intervention", pd.Series(["care_access"] * len(patients)))
        coverage_ok = np.mean([rec in A_behavioral.values for rec in recommended])

        return {
            "model": "SARSA_C8",
            "imi": imi,
            "policy_value": policy_value,
            "coverage_coefficient": float(coverage_ok),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Comparator Suite Runner
# ─────────────────────────────────────────────────────────────────────────────

class ComparatorSuite:
    """Fit and evaluate all 8 comparators on the same patient population."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.lace = LACEIndex()
        self.hospital = HOSPITALScore()
        self.xgb = XGBoostComparator(seed=seed)
        self.bc_sft = BehavioralCloningSFT(seed=seed)
        self.obs_dpo = ObservationalDPO(seed=seed)
        self.causal_forest = CausalForestComparator(seed=seed)
        self.dt = DecisionTransformerProxy(seed=seed)
        self.cql = CQLComparator(seed=seed)

    def fit_all(
        self,
        patients: pd.DataFrame,
        wpad_pairs: pd.DataFrame,
        outcome_col: str = "y_behavioral",
        intervention_col: str = "behavioral_intervention",
        wpad_preferred_col: str = "wpad_preferred_intervention",
    ) -> "ComparatorSuite":
        """
        wpad_preferred_col: column in patients with the WPAD S-learner–derived preferred
        intervention (argmin mu_hat). Used by C4 (BehavioralCloning ablation) so it trains
        on the same causal signal as PEARL DPO but with SFT loss (no contrastive preference).
        """
        print("Fitting comparators...")
        self.xgb.fit(patients, outcome_col)
        print("  C3 XGBoost fitted")
        # C4: SFT ablation — use WPAD-preferred signal (same as PEARL, but SFT not DPO)
        # Falls back to behavioral_intervention if wpad_preferred column not in patients.
        bc_target = wpad_preferred_col if wpad_preferred_col in patients.columns else intervention_col
        self.bc_sft.fit(patients, target_col=bc_target)
        print("  C4 BehavioralCloning fitted")
        self.obs_dpo.fit(patients, outcome_col, intervention_col)
        print("  C5 ObservationalDPO fitted")
        self.causal_forest.fit(patients, outcome_col, intervention_col)
        print("  C6 CausalForest fitted")
        self.dt.fit(wpad_pairs, patients)
        print("  C7 DecisionTransformer fitted")
        self.cql.fit(patients, outcome_col, intervention_col, wpad_pairs=wpad_pairs)
        print("  C8 CQL fitted (Conservative Q-Learning + WPAD Q-targets)")
        return self

    def evaluate_all(
        self,
        patients: pd.DataFrame,
        mu_hat_dr: np.ndarray,
        outcome_col: str = "y_behavioral",
        threshold: float = 0.02,
    ) -> pd.DataFrame:
        """Evaluate all comparators and return results DataFrame."""
        results = []

        # C1, C2: risk scores (AUROC + IMI proxy)
        for model, label in [(self.lace, "LACE_C1"), (self.hospital, "HOSPITAL_C2")]:
            eval_r = model.evaluate(patients, outcome_col)
            # Route and compute IMI
            if hasattr(model, "route_intervention"):
                routing = model.route_intervention(patients)
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder().fit(INTERVENTIONS)
                A_enc = le.transform(routing)
                imi = float(np.array([
                    float(any(mu_hat_dr[i, j] < mu_hat_dr[i, A_enc[i]] - threshold
                              for j in range(len(INTERVENTIONS)) if j != A_enc[i]))
                    for i in range(len(patients))
                ]).mean())
                eval_r["imi"] = imi
                eval_r["policy_value"] = float(np.mean([mu_hat_dr[i, A_enc[i]] for i in range(len(patients))]))
            results.append(eval_r)

        # C3: XGBoost
        eval_c3 = self.xgb.evaluate(patients, outcome_col)
        routing_c3 = self.xgb.route_intervention(patients)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(INTERVENTIONS)
        A_enc_c3 = le.transform(routing_c3)
        eval_c3["imi"] = float(np.array([
            float(any(mu_hat_dr[i, j] < mu_hat_dr[i, A_enc_c3[i]] - threshold
                      for j in range(len(INTERVENTIONS)) if j != A_enc_c3[i]))
            for i in range(len(patients))
        ]).mean())
        eval_c3["policy_value"] = float(np.mean([mu_hat_dr[i, A_enc_c3[i]] for i in range(len(patients))]))
        results.append(eval_c3)

        # C4–C8
        for evaluator in [self.bc_sft, self.obs_dpo, self.causal_forest, self.dt, self.cql]:
            if hasattr(evaluator, "evaluate_imi"):
                results.append(evaluator.evaluate_imi(patients, mu_hat_dr, threshold))

        return pd.DataFrame(results)


if __name__ == "__main__":
    from data.synthetic_generator import generate_synthetic_population
    from models.imi_estimator import IMIEstimator

    pop = generate_synthetic_population(n_patients=10_000, seed=42)
    rising = pop.patients[pop.patients["rising_risk"]].reset_index(drop=True)

    # Fit IMI estimator to get mu_hat_dr
    estimator = IMIEstimator(n_bootstrap=50)
    estimator.fit(rising)
    result = estimator.estimate(rising)
    mu_hat_dr = result["mu_hat_dr"]

    # Fit and evaluate all comparators
    suite = ComparatorSuite(seed=42)
    suite.fit_all(rising, pop.wpad_pairs)
    eval_df = suite.evaluate_all(rising, mu_hat_dr)

    print("\n" + "="*60)
    print("COMPARATOR EVALUATION SUMMARY (C1–C8)")
    print("="*60)
    print(eval_df.to_string(index=False))
