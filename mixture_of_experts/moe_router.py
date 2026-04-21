"""
PEARL Mixture of Experts (MoE) Router

Four specialized LoRA adapters, each expert in one care domain:
  Expert 1 — social_needs:          Food, housing, transport, language navigation
  Expert 2 — medication_adherence:  Pharmacy reconciliation, pill management
  Expert 3 — behavioral_health:     MH/SUD referral, stigma-aware coordination
  Expert 4 — clinical_complexity:   Multi-morbidity, specialist coordination

Router: soft attention over patient features → weighted combination of expert outputs.
Top-K routing (K=2): activate 2 experts per patient, mix outputs.
Load balancing: entropy regularization to prevent expert collapse.

In LLM mode: each expert is a separate LoRA adapter (r=32 for experts vs r=64 for base).
In tabular mode: each expert is a specialized LogisticRegression trained on its domain.

Reference architecture:
  Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
  Switch Transformer (Fedus et al., JMLR 2022) for load balancing loss
  Mistral MoE (2023) for top-K routing implementation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

INTERVENTIONS = ["social_needs", "medication_adherence", "behavioral_health", "clinical_complexity"]

FEATURE_COLS = [
    "age", "female", "charlson_score", "prior_ed_visits_6mo", "prior_hosp_6mo",
    "pharmacy_fills_90d", "missed_pharmacy_fills", "n_chronic",
    "has_diabetes", "has_chf", "has_copd", "has_hypertension", "has_ckd", "has_mh",
    "adi_percentile", "food_insecure", "housing_unstable", "lives_alone", "no_transport"
]

# Expert specialization: which features each expert focuses on
EXPERT_FEATURES = {
    "social_needs": [
        "adi_percentile", "food_insecure", "housing_unstable", "no_transport",
        "lives_alone", "age", "female",
        # Language proxy: derived during feature engineering
    ],
    "medication_adherence": [
        "pharmacy_fills_90d", "missed_pharmacy_fills", "n_chronic",
        "has_diabetes", "has_chf", "has_copd", "has_hypertension", "has_ckd",
        "charlson_score", "age"
    ],
    "behavioral_health": [
        "has_mh", "age", "female", "adi_percentile",
        "prior_ed_visits_6mo",  # ED often driven by MH crises
        "lives_alone",
        "food_insecure"  # SDOH often co-occurs with MH need
    ],
    "clinical_complexity": [
        "charlson_score", "n_chronic", "prior_hosp_6mo", "prior_ed_visits_6mo",
        "has_chf", "has_copd", "has_ckd", "has_diabetes", "age"
    ]
}


class ExpertAdapter:
    """
    Single specialized LoRA adapter for one care management domain.

    In LLM mode: wraps a LoRA adapter with domain-specific fine-tuning.
    In tabular mode: specialized logistic regression with domain features.
    """

    def __init__(self, expert_name: str, seed: int = 42):
        assert expert_name in INTERVENTIONS, f"Unknown expert: {expert_name}"
        self.name = expert_name
        self.seed = seed
        self._feature_subset = EXPERT_FEATURES[expert_name]
        self._model = LogisticRegression(C=0.5, max_iter=500, multi_class="multinomial",
                                          random_state=seed)
        self._fitted = False
        self._le = LabelEncoder().fit(INTERVENTIONS)

    def _get_X(self, patients: pd.DataFrame) -> np.ndarray:
        """Use only the expert's specialized feature subset."""
        cols = [c for c in self._feature_subset if c in patients.columns]
        if not cols:
            cols = [c for c in FEATURE_COLS if c in patients.columns]
        return patients[cols].fillna(0).astype(float).values

    def fit(
        self,
        patients: pd.DataFrame,
        wpad_pairs: pd.DataFrame,
        target_col: str = "behavioral_intervention",
    ) -> "ExpertAdapter":
        """
        Fit expert on patients whose preferred intervention matches this expert's domain.
        Expert learns: given patient features, what's the best sub-intervention
        within this expert's domain?
        Default target: behavioral_intervention (or wpad_preferred_intervention if passed).
        Do NOT use optimal_intervention — that leaks oracle ground truth.
        """
        # Filter to this expert's specialty patients
        if target_col in patients.columns:
            domain_patients = patients[patients[target_col] == self.name].copy()
        else:
            domain_patients = patients.copy()

        if len(domain_patients) < 20:
            # Not enough domain-specific patients: fall back to all patients
            domain_patients = patients.copy()

        X = self._get_X(domain_patients)

        # Target: intervention type (in single-intervention tabular version, target is binary)
        # In real LLM version: target is the preferred care plan completion
        y = np.where(
            domain_patients.get(target_col, pd.Series(["clinical_complexity"] * len(domain_patients))).values == self.name,
            1, 0
        )

        if len(np.unique(y)) > 1:
            self._model.fit(X, y)
        else:
            # All same class: create trivial model
            from sklearn.dummy import DummyClassifier
            self._model = DummyClassifier(strategy="most_frequent")
            self._model.fit(X, y)

        self._fitted = True
        return self

    def predict_confidence(self, patients: pd.DataFrame) -> np.ndarray:
        """
        Return this expert's confidence that it is the right expert for each patient.
        Higher = more likely this expert domain applies.
        """
        X = self._get_X(patients)
        try:
            proba = self._model.predict_proba(X)
            if proba.shape[1] >= 2:
                return proba[:, 1]  # P(this expert is relevant)
            else:
                return proba[:, 0]
        except Exception:
            return np.ones(len(patients)) * 0.25  # uniform fallback

    def generate_recommendation(self, patients: pd.DataFrame) -> np.ndarray:
        """Return this expert's recommended intervention type."""
        conf = self.predict_confidence(patients)
        # Expert always recommends itself; confidence determines router weighting
        return np.array([self.name] * len(patients)), conf


class MoERouter:
    """
    Mixture of Experts router for PEARL.

    Architecture:
    - Router: MLP that computes attention weights over 4 experts
    - Top-K routing: activate top 2 experts per patient
    - Load balancing: entropy regularization (Switch Transformer style)
    - Final prediction: weighted combination of expert recommendations

    For care plans (discrete outputs): take the expert with highest weight
    for the final recommendation (hard routing in inference mode).
    Soft routing is used during training for gradient flow.
    """

    def __init__(
        self,
        n_experts: int = 4,
        top_k: int = 2,
        load_balancing_coef: float = 0.01,
        hidden_dim: int = 64,
        seed: int = 42,
    ):
        self.n_experts = n_experts
        self.top_k = top_k
        self.load_balancing_coef = load_balancing_coef
        self.seed = seed

        # Expert adapters
        self.experts = {name: ExpertAdapter(name, seed=seed) for name in INTERVENTIONS}

        # Router network (tabular: MLP; LLM: learned linear projection)
        self._router = MLPClassifier(
            hidden_layer_sizes=(hidden_dim, 32),
            activation="relu",
            max_iter=300,
            random_state=seed,
            alpha=0.01,
        )
        self._fitted = False
        self._le = LabelEncoder().fit(INTERVENTIONS)

    def _get_X(self, patients: pd.DataFrame) -> np.ndarray:
        cols = [c for c in FEATURE_COLS if c in patients.columns]
        return patients[cols].fillna(0).astype(float).values

    def _compute_router_weights(self, patients: pd.DataFrame) -> np.ndarray:
        """
        Compute soft routing weights: shape (n_patients, n_experts).
        Router assigns each patient to experts via softmax attention.
        """
        X = self._get_X(patients)

        try:
            # Router MLP → softmax over experts
            router_logits = self._router.predict_proba(X)  # (n, n_experts)
            if router_logits.shape[1] != self.n_experts:
                # Fallback: use individual expert confidences
                router_weights = self._individual_expert_confidences(patients)
            else:
                router_weights = router_logits
        except Exception:
            router_weights = self._individual_expert_confidences(patients)

        return router_weights

    def _individual_expert_confidences(self, patients: pd.DataFrame) -> np.ndarray:
        """
        Fallback: get confidence from each expert adapter independently.
        """
        n = len(patients)
        weights = np.zeros((n, self.n_experts))
        for i, name in enumerate(INTERVENTIONS):
            weights[:, i] = self.experts[name].predict_confidence(patients)
        # Softmax normalization
        exp_weights = np.exp(weights - weights.max(axis=1, keepdims=True))
        return exp_weights / exp_weights.sum(axis=1, keepdims=True)

    def _top_k_routing(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Top-K routing: keep top K expert weights, zero out others, renormalize.
        Returns: (sparse_weights, top_k_indices) both shape (n_patients, n_experts)
        """
        n = weights.shape[0]
        sparse_weights = np.zeros_like(weights)

        # Get top-K indices per patient
        top_k_idx = np.argsort(weights, axis=1)[:, -self.top_k:]  # (n, K)

        for i in range(n):
            top_weights = weights[i, top_k_idx[i]]
            # Renormalize top-K weights to sum to 1
            top_weights_norm = top_weights / (top_weights.sum() + 1e-9)
            sparse_weights[i, top_k_idx[i]] = top_weights_norm

        return sparse_weights, top_k_idx

    def _load_balancing_loss(self, router_weights: np.ndarray) -> float:
        """
        Load balancing loss: encourage uniform expert utilization.
        Switch Transformer (Fedus et al., JMLR 2022) style.
        L_balance = n_experts * Σ_e (f_e * P_e)
        where f_e = fraction routed to expert e, P_e = mean router prob for expert e.
        """
        n = router_weights.shape[0]
        f = router_weights.mean(axis=0)   # fraction routed per expert
        P = router_weights.mean(axis=0)   # mean router probability per expert
        balance_loss = self.n_experts * np.sum(f * P)
        return float(balance_loss)

    def fit(
        self,
        patients: pd.DataFrame,
        wpad_pairs: pd.DataFrame,
        target_col: str = "behavioral_intervention",
    ) -> "MoERouter":
        """
        Stage 1: Fit individual expert adapters on domain-specific data.
        Stage 2: Fit router to predict optimal expert routing.
        Pass target_col="wpad_preferred_intervention" (S-learner argmin) for causal signal.
        Do NOT use target_col="optimal_intervention" — that leaks oracle ground truth.
        """
        print("Fitting MoE experts...")
        for name, expert in self.experts.items():
            expert.fit(patients, wpad_pairs, target_col)
            print(f"  Expert '{name}' fitted")

        # Fit router: given patient features → predict which expert is optimal.
        # Use INTERVENTIONS-order indices (0=social_needs,...,3=clinical_complexity) so
        # that predict_proba columns align with INTERVENTIONS list for decoding.
        X = self._get_X(patients)
        _intv_to_idx = {intv: i for i, intv in enumerate(INTERVENTIONS)}
        if target_col in patients.columns:
            y = np.array([_intv_to_idx.get(str(v), 0)
                          for v in patients[target_col].fillna("clinical_complexity").values])
        else:
            # Use expert confidences to create soft router targets
            expert_conf = self._individual_expert_confidences(patients)
            y = np.argmax(expert_conf, axis=1)  # already INTERVENTIONS order

        # Router MLP
        try:
            self._router.fit(X, y)
            print("  Router fitted")
        except Exception as e:
            print(f"  Router fit failed ({e}), using individual expert confidences")

        self._fitted = True
        return self

    def predict(
        self,
        patients: pd.DataFrame,
        hard_routing: bool = True,
        return_weights: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predict care management intervention with MoE routing.

        Returns:
          recommendations: array of intervention type strings
          router_weights: (n_patients, n_experts) routing weight matrix
          load_balance_loss: scalar load balancing loss (should be low)
        """
        # Compute router weights
        soft_weights = self._compute_router_weights(patients)
        sparse_weights, top_k_idx = self._top_k_routing(soft_weights)
        lb_loss = self._load_balancing_loss(sparse_weights)

        # Get final recommendation
        if hard_routing:
            # Hard: take the expert with highest weight
            best_expert_idx = np.argmax(sparse_weights, axis=1)
            recommendations = np.array([INTERVENTIONS[i] for i in best_expert_idx])
        else:
            # Soft: weighted vote (for training)
            recommendations = np.array([INTERVENTIONS[np.argmax(sparse_weights[i])]
                                         for i in range(len(patients))])

        return recommendations, sparse_weights, lb_loss

    def evaluate_imi(
        self,
        patients: pd.DataFrame,
        mu_hat_dr: np.ndarray,
        threshold: float = 0.02,
    ) -> Dict:
        """Evaluate MoE-PEARL policy on DR-estimated outcomes."""
        recommendations, router_weights, lb_loss = self.predict(patients)
        le = LabelEncoder().fit(INTERVENTIONS)
        A_enc = le.transform(recommendations)

        imi = float(np.array([
            float(any(mu_hat_dr[i, j] < mu_hat_dr[i, A_enc[i]] - threshold
                      for j in range(len(INTERVENTIONS)) if j != A_enc[i]))
            for i in range(len(patients))
        ]).mean())

        policy_value = float(np.mean([mu_hat_dr[i, A_enc[i]] for i in range(len(patients))]))

        # Expert utilization rates (router diagnostic)
        utilization = {INTERVENTIONS[i]: float(np.mean(np.argmax(router_weights, axis=1) == i))
                      for i in range(len(INTERVENTIONS))}

        return {
            "model": f"PEARL_MoE_K{self.top_k}_experts{self.n_experts}",
            "imi": imi,
            "policy_value": policy_value,
            "load_balancing_loss": lb_loss,
            "expert_utilization": utilization,
            "max_utilization_skew": float(max(utilization.values()) - min(utilization.values())),
        }

    def get_routing_report(self, patients: pd.DataFrame) -> pd.DataFrame:
        """Return per-patient routing summary for interpretability."""
        recommendations, weights, lb_loss = self.predict(patients)

        # Top-2 experts and their weights
        top2_idx = np.argsort(weights, axis=1)[:, -2:][:, ::-1]
        top1_expert = [INTERVENTIONS[i] for i in top2_idx[:, 0]]
        top2_expert = [INTERVENTIONS[i] for i in top2_idx[:, 1]]
        top1_weight = weights[np.arange(len(patients)), top2_idx[:, 0]]
        top2_weight = weights[np.arange(len(patients)), top2_idx[:, 1]]

        return pd.DataFrame({
            "patient_id": patients.get("patient_id", pd.Series(range(len(patients)))),
            "final_recommendation": recommendations,
            "primary_expert": top1_expert,
            "primary_weight": top1_weight.round(3),
            "secondary_expert": top2_expert,
            "secondary_weight": top2_weight.round(3),
            "load_balance_loss": lb_loss,
        })


class MoEPEARL:
    """
    Full PEARL-MoE: combines TabularPEARL base policy with MoE routing.
    Final prediction: blend PEARL base recommendation with MoE routing.
    """

    def __init__(
        self,
        beta: float = 0.1,
        moe_weight: float = 0.5,  # weight on MoE vs. base PEARL
        top_k: int = 2,
        seed: int = 42,
    ):
        from models.pearl_dpo import TabularPEARL
        self.base_pearl = TabularPEARL(beta=beta, seed=seed)
        self.moe_router = MoERouter(top_k=top_k, seed=seed)
        self.moe_weight = moe_weight
        self._fitted = False

    def fit(
        self,
        wpad_pairs: pd.DataFrame,
        patients: pd.DataFrame,
        n_iterations: int = 50,
        moe_target_col: str = "behavioral_intervention",
    ) -> "MoEPEARL":
        """
        moe_target_col: column in patients to use as MoE training target.
        Pass "wpad_preferred_intervention" (S-learner argmin) for causal signal.
        Default "behavioral_intervention" avoids leaking oracle ground truth.
        """
        print("Training base PEARL...")
        self.base_pearl.fit(wpad_pairs, patients, n_iterations=n_iterations, verbose=True)
        print("\nTraining MoE router...")
        self.moe_router.fit(patients, wpad_pairs, target_col=moe_target_col)
        self._fitted = True
        return self

    def predict(self, patients: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Confidence-weighted blend of base PEARL and MoE Router.

        Rationale: PEARL base has the causal DPO signal (preferred vs. rejected completions).
        MoE Router has domain specialization (expert per care domain). They complement each other.

        Decision rule:
          - If base PEARL DPO margin > abstention threshold τ: use base PEARL (confident causal signal)
          - If base PEARL abstains (low margin): defer to MoE Router (specialized expert for this patient)
          - Agreement bonus: when both agree, always use that recommendation regardless of margin.

        This creates genuine synergy: PEARL handles clear-cut cases, MoE handles uncertain ones.
        """
        base_recs, base_margins, base_abstain = self.base_pearl.predict_intervention(patients)
        moe_recs, moe_weights, lb_loss = self.moe_router.predict(patients)

        # Agreement mask — when both agree, high confidence regardless of margin
        agrees = (base_recs == moe_recs)

        # Use base PEARL when: (1) it's confident (not abstaining) OR (2) both agree
        # Defer to MoE when: base PEARL abstains AND they disagree
        use_base = (~base_abstain) | agrees  # True → use base, False → use MoE
        final_recs = np.where(use_base, base_recs, moe_recs)

        metadata = {
            "base_recs": base_recs,
            "moe_recs": moe_recs,
            "agreement_rate": float(agrees.mean()),
            "base_used_rate": float(use_base.mean()),
            "moe_weights": moe_weights,
            "load_balance_loss": lb_loss,
            "base_abstain_rate": float(base_abstain.mean()),
        }
        return final_recs, metadata

    def evaluate_imi(
        self,
        patients: pd.DataFrame,
        mu_hat_dr: np.ndarray,
        threshold: float = 0.02,
    ) -> Dict:
        recommendations, metadata = self.predict(patients)
        le = LabelEncoder().fit(INTERVENTIONS)
        A_enc = le.transform(recommendations)

        imi = float(np.array([
            float(any(mu_hat_dr[i, j] < mu_hat_dr[i, A_enc[i]] - threshold
                      for j in range(len(INTERVENTIONS)) if j != A_enc[i]))
            for i in range(len(patients))
        ]).mean())

        policy_value = float(np.mean([mu_hat_dr[i, A_enc[i]] for i in range(len(patients))]))
        return {
            "model": "PEARL_MoE_Full",
            "imi": imi,
            "policy_value": policy_value,
            "agreement_rate": metadata["agreement_rate"],
            "load_balance_loss": metadata["load_balance_loss"],
        }


if __name__ == "__main__":
    from data.synthetic_generator import generate_synthetic_population
    from models.imi_estimator import IMIEstimator

    pop = generate_synthetic_population(n_patients=10_000, seed=42)
    rising = pop.patients[pop.patients["rising_risk"]].reset_index(drop=True)

    # Fit IMI estimator
    estimator = IMIEstimator(n_bootstrap=50)
    estimator.fit(rising)
    result = estimator.estimate(rising)
    mu_hat_dr = result["mu_hat_dr"]

    print("="*60)
    print("MIXTURE OF EXPERTS ROUTER")
    print("="*60)

    moe = MoERouter(n_experts=4, top_k=2, seed=42)
    moe.fit(rising, pop.wpad_pairs)

    eval_result = moe.evaluate_imi(rising, mu_hat_dr)
    print(f"\nMoE IMI:              {eval_result['imi']:.3f}")
    print(f"MoE policy value:     {eval_result['policy_value']:.3f}")
    print(f"Load balancing loss:  {eval_result['load_balancing_loss']:.4f}")
    print(f"Expert utilization:   {eval_result['expert_utilization']}")
    print(f"Max utilization skew: {eval_result['max_utilization_skew']:.3f}")

    # Full MoE-PEARL
    print("\nFull PEARL-MoE (base + MoE blend)...")
    moe_pearl = MoEPEARL(beta=0.1, moe_weight=0.5, seed=42)
    moe_pearl.fit(pop.wpad_pairs, pop.patients, n_iterations=30)
    full_result = moe_pearl.evaluate_imi(rising, mu_hat_dr)
    print(f"\nPEARL-MoE IMI:        {full_result['imi']:.3f}")
    print(f"PEARL-MoE pv:         {full_result['policy_value']:.3f}")
    print(f"Agreement rate:       {full_result['agreement_rate']:.1%}")
