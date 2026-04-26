"""
Einstein Arena: Adversarial Hyperparameter Optimization for PEARL.

Default output location: notebooks/pearl/outputs/results/einstein_arena_results.json
(resolved relative to the package; override via PEARL_OUTPUT_BASE).

Implements a multi-round adversarial optimization loop:
  1. Generate a population of candidate configurations
  2. Evaluate each on DR-OPE + IMI + fairness metrics
  3. Adversarial critique: simulate expert reviewer attacks on each configuration
  4. Select survivor configurations and generate next-generation candidates
  5. Repeat until convergence

The Einstein Arena framework operationalizes the paper's adversarial review process
as a formal optimization procedure over the PEARL hyperparameter space.

Hyperparameters searched:
  - lora_r: LoRA rank ∈ {16, 32, 64}
  - beta: DPO regularization ∈ {0.05, 0.1, 0.2}
  - t_min: WPAD minimum gap ∈ {30, 60, 90} days
  - wpad_type: {all, onboarding_waitlist, coverage_gap}
  - top_k: MoE top-K routing ∈ {1, 2}
  - moe_weight: base PEARL vs. MoE blend ∈ {0.3, 0.5, 0.7}
  - group_equal_weight: fairness constraint ∈ {True, False}

Scoring function (multi-objective):
  score = α * DR_OPE_improvement + β * IMI_reduction + γ * equity_imi_reduction
           + δ * ESS_adequacy + ε * load_balance_quality

  Primary metric: DR-OPE policy value improvement over behavioral policy
  Constraint: ESS > 500 AND equity-IMI gap < 0.05

Adversarial critics (simulated reviewer attacks):
  C1 — "Causal inference reviewer": test WPAD identification validity (T1-T6)
  C2 — "Offline RL reviewer": test ESS adequacy and coverage coefficient
  C3 — "Fairness reviewer": test equity-IMI gap and demographic parity
  C4 — "Clinical reviewer": test abstention rate and care plan quality
  C5 — "Stats reviewer": test CI width and sensitivity analysis stability
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from itertools import product
import json
import time
import warnings

# Default output base: notebooks/pearl/outputs/ relative to repo root.
_PEARL_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PEARL_ROOT.parents[1]
DEFAULT_RESULTS_DIR = Path(os.environ.get(
    "PEARL_OUTPUT_BASE",
    str(_REPO_ROOT / "notebooks" / "pearl" / "outputs"),
)) / "results"
warnings.filterwarnings("ignore")


@dataclass
class PEARLConfig:
    """Single hyperparameter configuration to be evaluated."""
    lora_r: int = 64
    beta: float = 0.1
    t_min: int = 60
    wpad_type: str = "all"
    top_k: int = 2
    moe_weight: float = 0.5
    group_equal_weight: bool = True
    # Automatically filled during evaluation
    config_id: str = ""

    def __post_init__(self):
        if not self.config_id:
            self.config_id = (
                f"r{self.lora_r}_b{self.beta}_t{self.t_min}_"
                f"wpad{self.wpad_type[:3]}_k{self.top_k}_moe{self.moe_weight}"
            )

    def to_dict(self) -> Dict:
        return {
            "config_id": self.config_id,
            "lora_r": self.lora_r, "beta": self.beta, "t_min": self.t_min,
            "wpad_type": self.wpad_type, "top_k": self.top_k,
            "moe_weight": self.moe_weight, "group_equal_weight": self.group_equal_weight,
        }


@dataclass
class EvaluationResult:
    """Results from evaluating a single PEARL configuration."""
    config: PEARLConfig
    # Primary metrics
    drope_improvement: float = 0.0  # relative to behavioral policy (higher = better)
    imi_reduction: float = 0.0       # IMI_behavioral - IMI_pearl (higher = better)
    equity_imi_gap: float = 0.0      # max group IMI - min group IMI (lower = better)
    # Constraints
    ess: float = 0.0
    ess_adequate: bool = False
    coverage_coefficient: float = 0.0
    abstention_rate: float = 0.0
    load_balance_loss: float = 0.0
    # Composite score
    composite_score: float = 0.0
    # Adversarial critique results
    adversarial_scores: Dict[str, float] = field(default_factory=dict)
    adversarial_fatal: bool = False
    adversarial_critiques: List[str] = field(default_factory=list)
    # Timing
    eval_time_seconds: float = 0.0

    def to_dict(self) -> Dict:
        d = self.config.to_dict()
        d.update({
            "drope_improvement": self.drope_improvement,
            "imi_reduction": self.imi_reduction,
            "equity_imi_gap": self.equity_imi_gap,
            "ess": self.ess,
            "ess_adequate": self.ess_adequate,
            "coverage_coefficient": self.coverage_coefficient,
            "abstention_rate": self.abstention_rate,
            "load_balance_loss": self.load_balance_loss,
            "composite_score": self.composite_score,
            "adversarial_fatal": self.adversarial_fatal,
            "adversarial_critiques": " | ".join(self.adversarial_critiques),
            "eval_time_seconds": self.eval_time_seconds,
        })
        return d


class AdversarialCritic:
    """
    Simulates adversarial reviewer critiques on a PEARL configuration.
    Each critic attacks from a different reviewer perspective.
    Returns critique strings and a pass/fail flag.
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def critique_all(self, result: EvaluationResult) -> Tuple[bool, List[str]]:
        """Run all five critics. Returns (passes_all, list_of_critiques)."""
        critiques = []
        fatal = False

        checks = [
            self._c1_causal_inference,
            self._c2_offline_rl,
            self._c3_fairness,
            self._c4_clinical,
            self._c5_stats,
        ]
        for check in checks:
            is_fatal, critique = check(result)
            if critique:
                critiques.append(critique)
            if is_fatal:
                fatal = True

        return fatal, critiques

    def _c1_causal_inference(self, result: EvaluationResult) -> Tuple[bool, str]:
        """C1: Causal inference reviewer — WPAD identification validity."""
        config = result.config

        # T_min=30 is too short (< 2 month minimum for Medicaid redetermination cycles)
        if config.t_min < 60:
            return False, (
                "C1-CAUTION: T_min=30 days is below the typical 60-day Medicaid "
                "redetermination cycle — coverage gaps this short may reflect voluntary "
                "disenrollment, not administrative disruption. Exclusion restriction weaker."
            )

        # Coverage-gap-only WPAD has engagement rate concern
        if config.wpad_type == "coverage_gap":
            return False, (
                "C1-CAUTION: Coverage-gap WPAD only → 5-10% engagement rate → "
                "~750 ITT pairs (not LATE). Report as ITT secondary, not primary."
            )

        return False, ""

    def _c2_offline_rl(self, result: EvaluationResult) -> Tuple[bool, str]:
        """C2: Offline RL reviewer — ESS and coverage coefficient."""
        if not result.ess_adequate:
            return True, (
                f"C2-FATAL: ESS = {result.ess:.0f} < 500 minimum. "
                "DR-OPE estimate is unreliable. Restrict evaluation to "
                "high-coverage subpopulation OR increase WPAD pair count."
            )

        if result.coverage_coefficient < 0.70:
            return False, (
                f"C2-CAUTION: Coverage coefficient = {result.coverage_coefficient:.2f} < 0.70. "
                "PEARL is recommending interventions without behavioral policy support for "
                f"{(1 - result.coverage_coefficient)*100:.0f}% of patients. "
                "Conformal prediction intervals may be underestimating uncertainty."
            )

        return False, ""

    def _c3_fairness(self, result: EvaluationResult) -> Tuple[bool, str]:
        """C3: Fairness reviewer — equity-IMI gap and demographic parity."""
        if result.equity_imi_gap > 0.10:
            return False, (
                f"C3-CAUTION: Equity-IMI gap = {result.equity_imi_gap:.3f} > 0.10. "
                "PEARL is producing disparate IMI reduction across demographic groups. "
                "Check group-stratified DPO convergence and demographic upsampling balance."
            )

        if not result.config.group_equal_weight:
            return False, (
                "C3-NOTE: group_equal_weight=False. The DPO loss is dominated by the "
                "majority group (likely non-Hispanic White). Equity-IMI reduction may "
                "be hollow. Require equal group weighting for fairness claims."
            )

        return False, ""

    def _c4_clinical(self, result: EvaluationResult) -> Tuple[bool, str]:
        """C4: Clinical reviewer — abstention and care plan quality."""
        if result.abstention_rate > 0.40:
            return False, (
                f"C4-CAUTION: Abstention rate = {result.abstention_rate:.1%} > 40%. "
                "PEARL is deferring for nearly half of patients. This may indicate "
                "insufficient WPAD pair coverage for these profiles, or beta is too "
                "low (policy too uncertain relative to reference). Consider lowering τ."
            )

        if result.config.top_k == 1 and result.load_balance_loss > 0.5:
            return False, (
                f"C4-CAUTION: MoE with top_k=1 and high load balance loss "
                f"({result.load_balance_loss:.3f}). Expert collapse likely — "
                "all patients being routed to one expert. Check expert utilization."
            )

        return False, ""

    def _c5_stats(self, result: EvaluationResult) -> Tuple[bool, str]:
        """C5: Stats reviewer — CI width and sensitivity robustness."""
        # If DR-OPE improvement is positive but tiny, it may not be clinically meaningful
        if 0 < result.drope_improvement < 0.01:
            return False, (
                f"C5-CAUTION: DR-OPE improvement = {result.drope_improvement:.4f} < 1%. "
                "Statistically may be significant but clinically trivial. "
                "Report alongside NNT for clinical decision makers."
            )

        # High beta = strong KL penalty = policy stays close to behavioral → small improvement
        if result.config.beta >= 0.2 and result.drope_improvement < 0.02:
            return False, (
                "C5-NOTE: β=0.2 (strong regularization) combined with small DR-OPE "
                "improvement suggests the KL penalty is dominating the preference signal. "
                "Try β=0.05 or β=0.1 for more aggressively causal recommendations."
            )

        return False, ""


class EinsteinArena:
    """
    Adversarial hyperparameter optimization for PEARL.

    Each round:
    1. Evaluate all configurations in the current population
    2. Apply adversarial critics to each
    3. Rank by composite score (with penalties for critic failures)
    4. Select top survivors and generate next-generation mutations
    5. Report round winner and accumulated insights

    The "Einstein Arena" metaphor: configurations compete; critics eliminate weak ones;
    survivors breed. Converges to a configuration that satisfies all five reviewers.
    """

    # Scoring weights for composite score
    SCORE_WEIGHTS = {
        "drope_improvement": 0.40,   # primary: DR-OPE policy improvement
        "imi_reduction": 0.30,       # mechanism: IMI reduction
        "equity_imi_gap": -0.15,     # constraint: lower equity gap is better
        "abstention_rate": -0.05,    # constraint: lower abstention is better
        "load_balance_loss": -0.05,  # constraint: lower = better expert distribution
        "ess_bonus": 0.05,           # bonus for ESS adequacy
    }

    def __init__(
        self,
        n_generations: int = 3,
        population_size: int = 12,
        survivors_per_round: int = 4,
        seed: int = 42,
    ):
        self.n_generations = n_generations
        self.population_size = population_size
        self.survivors_per_round = survivors_per_round
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.critic = AdversarialCritic(seed=seed)
        self.all_results: List[EvaluationResult] = []
        self.best_config: Optional[PEARLConfig] = None
        self.best_result: Optional[EvaluationResult] = None

    def _generate_initial_population(self) -> List[PEARLConfig]:
        """Generate diverse initial configurations covering the full search space."""
        configs = []

        # Primary anchor: plan's specified configuration
        configs.append(PEARLConfig(
            lora_r=64, beta=0.1, t_min=60, wpad_type="all",
            top_k=2, moe_weight=0.5, group_equal_weight=True
        ))

        # Systematic grid over key parameters
        lora_rs = [16, 32, 64]
        betas = [0.05, 0.1, 0.2]
        t_mins = [30, 60, 90]
        top_ks = [1, 2]

        for r, b, t, k in product(lora_rs[:2], betas[:2], [60, 90], top_ks[:1]):
            if len(configs) >= self.population_size:
                break
            configs.append(PEARLConfig(
                lora_r=r, beta=b, t_min=t, wpad_type="all",
                top_k=k, moe_weight=0.5, group_equal_weight=True
            ))

        # Add fairness ablation
        configs.append(PEARLConfig(
            lora_r=64, beta=0.1, t_min=60, wpad_type="all",
            top_k=2, moe_weight=0.5, group_equal_weight=False  # unfair ablation
        ))

        # Coverage-gap only (tests ITT framing)
        configs.append(PEARLConfig(
            lora_r=32, beta=0.1, t_min=60, wpad_type="coverage_gap",
            top_k=2, moe_weight=0.5, group_equal_weight=True
        ))

        return configs[:self.population_size]

    def _mutate_config(self, parent: PEARLConfig) -> PEARLConfig:
        """Mutate a surviving configuration to generate an offspring."""
        mutations = {
            "lora_r": lambda: int(self._rng.choice([16, 32, 64])),
            "beta": lambda: float(self._rng.choice([0.05, 0.1, 0.2])),
            "t_min": lambda: int(self._rng.choice([30, 60, 90])),
            "top_k": lambda: int(self._rng.choice([1, 2])),
            "moe_weight": lambda: float(self._rng.choice([0.3, 0.5, 0.7])),
        }
        # Mutate one or two parameters
        n_mutations = int(self._rng.choice([1, 2]))
        params_to_mutate = self._rng.choice(list(mutations.keys()), size=n_mutations, replace=False)

        new_config = PEARLConfig(
            lora_r=parent.lora_r, beta=parent.beta, t_min=parent.t_min,
            wpad_type=parent.wpad_type, top_k=parent.top_k,
            moe_weight=parent.moe_weight, group_equal_weight=parent.group_equal_weight,
        )
        for param in params_to_mutate:
            setattr(new_config, param, mutations[param]())
        new_config.config_id = ""
        new_config.__post_init__()
        return new_config

    def _evaluate_config(
        self,
        config: PEARLConfig,
        patients: pd.DataFrame,
        wpad_pairs: pd.DataFrame,
        drope_evaluator,
    ) -> EvaluationResult:
        """Evaluate a single PEARL configuration on the dataset."""
        from models.pearl_dpo import TabularPEARL
        from models.imi_estimator import IMIEstimator
        from mixture_of_experts.moe_router import MoEPEARL
        from sklearn.preprocessing import LabelEncoder

        t0 = time.time()
        result = EvaluationResult(config=config)

        try:
            # Filter WPAD pairs by type
            if config.wpad_type == "onboarding_waitlist":
                pairs = wpad_pairs[wpad_pairs.get("wpad_type", pd.Series()).isin(
                    ["aco_onboarding", "chw_waitlist"]
                )] if "wpad_type" in wpad_pairs.columns else wpad_pairs
            elif config.wpad_type == "coverage_gap":
                pairs = wpad_pairs[wpad_pairs.get("wpad_type", pd.Series()).isin(
                    ["coverage_gap"]
                )] if "wpad_type" in wpad_pairs.columns else wpad_pairs
            else:
                pairs = wpad_pairs

            if len(pairs) < 50:
                pairs = wpad_pairs  # fallback

            # Train PEARL with this configuration
            pearl = TabularPEARL(beta=config.beta, lora_r=config.lora_r, seed=self.seed)
            pearl.fit(pairs, patients, n_iterations=30, verbose=False)

            # Fit MoE router
            from mixture_of_experts.moe_router import MoERouter
            moe = MoERouter(top_k=config.top_k, seed=self.seed)
            moe.fit(patients, pairs)

            # Get IMI estimator (fit once, reuse across configs)
            estimator = IMIEstimator(n_bootstrap=50, seed=self.seed)
            estimator.fit(patients)
            imi_result = estimator.estimate(patients)
            mu_hat_dr = imi_result["mu_hat_dr"]

            # Policy function: MoE-PEARL blend
            def policy_fn(pts):
                base_recs, _, _ = pearl.predict_intervention(pts)
                moe_recs, _, _ = moe.predict(pts)
                # Blend based on moe_weight
                return np.where(
                    self._rng.random(len(pts)) < config.moe_weight,
                    moe_recs, base_recs
                )

            # DR-OPE evaluation
            drope_result = drope_evaluator.evaluate_policy(patients, policy_fn, config.config_id)
            result.ess = drope_result["ess"]
            result.ess_adequate = drope_result["ess_adequate"]
            result.drope_improvement = drope_result["relative_improvement_pct"] / 100

            # IMI evaluation
            pearl_recs, dpo_margins, abstain_mask = pearl.predict_intervention(patients)
            from models.pearl_dpo import INTERVENTIONS as _PEARL_INTERVENTIONS
            le = LabelEncoder().fit(_PEARL_INTERVENTIONS)
            A_enc = le.transform(pearl_recs)
            imi_pearl = float(np.array([
                float(any(mu_hat_dr[i, j] < mu_hat_dr[i, A_enc[i]] - 0.02
                          for j in range(len(_PEARL_INTERVENTIONS)) if j != A_enc[i]))
                for i in range(len(patients))
            ]).mean())
            result.imi_reduction = imi_result["imi_point"] - imi_pearl

            # Equity-IMI gap
            group_imis = []
            for grp_col in ["race_eth", "adi_quintile"]:
                if grp_col in patients.columns:
                    for grp in patients[grp_col].unique():
                        mask = (patients[grp_col] == grp).values
                        if mask.sum() < 10:
                            continue
                        grp_imi = float(np.array([
                            float(any(mu_hat_dr[i, j] < mu_hat_dr[i, A_enc[i]] - 0.02
                                      for j in range(4) if j != A_enc[i]))
                            for i in range(len(patients)) if mask[i]
                        ]).mean())
                        group_imis.append(grp_imi)
            result.equity_imi_gap = float(max(group_imis) - min(group_imis)) if len(group_imis) > 1 else 0.0

            # MoE metrics
            moe_eval = moe.evaluate_imi(patients, mu_hat_dr)
            result.load_balance_loss = moe_eval["load_balance_loss"]

            # Abstention rate
            result.abstention_rate = float(abstain_mask.mean())
            result.coverage_coefficient = drope_result.get("coverage_coefficient", 0.8)

            # Composite score
            result.composite_score = (
                self.SCORE_WEIGHTS["drope_improvement"] * result.drope_improvement +
                self.SCORE_WEIGHTS["imi_reduction"] * result.imi_reduction +
                self.SCORE_WEIGHTS["equity_imi_gap"] * result.equity_imi_gap +
                self.SCORE_WEIGHTS["abstention_rate"] * result.abstention_rate +
                self.SCORE_WEIGHTS["load_balance_loss"] * result.load_balance_loss +
                self.SCORE_WEIGHTS["ess_bonus"] * float(result.ess_adequate)
            )

        except Exception as e:
            result.adversarial_critiques.append(f"EVAL-ERROR: {str(e)[:100]}")
            result.composite_score = -999.0

        # Apply adversarial critics
        fatal, critiques = self.critic.critique_all(result)
        result.adversarial_fatal = fatal
        result.adversarial_critiques.extend(critiques)
        if fatal:
            result.composite_score = max(result.composite_score - 0.5, -1.0)

        result.eval_time_seconds = time.time() - t0
        return result

    def run(
        self,
        patients: pd.DataFrame,
        wpad_pairs: pd.DataFrame,
        verbose: bool = True,
    ) -> Tuple[EvaluationResult, pd.DataFrame]:
        """
        Run the Einstein Arena optimization loop.
        Returns (best_result, full_results_df).
        """
        from evaluation.drope_evaluator import DROPEEvaluator

        print("\n" + "="*70)
        print("EINSTEIN ARENA: ADVERSARIAL HYPERPARAMETER OPTIMIZATION")
        print("="*70)
        print(f"Generations: {self.n_generations} | Population: {self.population_size} | Survivors: {self.survivors_per_round}")

        # Fit DR-OPE evaluator once
        drope_eval = DROPEEvaluator(n_bootstrap=50, seed=self.seed)
        drope_eval.fit(patients)

        current_population = self._generate_initial_population()
        generation_results = []

        for gen in range(self.n_generations):
            print(f"\n--- Generation {gen+1}/{self.n_generations} ({len(current_population)} configs) ---")
            gen_results = []

            for i, config in enumerate(current_population):
                print(f"  [{i+1:2d}/{len(current_population)}] Evaluating: {config.config_id}...")
                result = self._evaluate_config(config, patients, wpad_pairs, drope_eval)
                gen_results.append(result)
                self.all_results.append(result)

                if verbose:
                    status = "FATAL" if result.adversarial_fatal else "OK"
                    print(f"         score={result.composite_score:.4f} | "
                          f"dr_ope={result.drope_improvement:.4f} | "
                          f"imi_red={result.imi_reduction:.4f} | "
                          f"ess={result.ess:.0f} {'[OK]' if result.ess_adequate else '[FAIL]'} | "
                          f"critic={status}")

            generation_results.extend(gen_results)

            # Rank by composite score, exclude fatals
            non_fatal = [r for r in gen_results if not r.adversarial_fatal]
            if not non_fatal:
                print("  WARNING: All configurations failed adversarial critics. Keeping best available.")
                non_fatal = gen_results

            ranked = sorted(non_fatal, key=lambda r: r.composite_score, reverse=True)
            survivors = ranked[:self.survivors_per_round]

            print(f"\n  Round {gen+1} winner: {survivors[0].config.config_id}")
            print(f"    Score={survivors[0].composite_score:.4f} | "
                  f"DR-OPE={survivors[0].drope_improvement:.4f} | "
                  f"IMI_reduction={survivors[0].imi_reduction:.4f}")
            if survivors[0].adversarial_critiques:
                print(f"    Critics: {survivors[0].adversarial_critiques[0][:80]}")

            # Generate next generation: survivors + mutations
            if gen < self.n_generations - 1:
                next_population = [s.config for s in survivors]
                for survivor in survivors:
                    mutations_needed = (self.population_size - len(survivors)) // len(survivors)
                    for _ in range(mutations_needed):
                        if len(next_population) < self.population_size:
                            next_population.append(self._mutate_config(survivor.config))
                current_population = next_population[:self.population_size]

        # Final winner
        all_non_fatal = [r for r in self.all_results if not r.adversarial_fatal]
        if all_non_fatal:
            self.best_result = max(all_non_fatal, key=lambda r: r.composite_score)
        else:
            self.best_result = max(self.all_results, key=lambda r: r.composite_score)

        self.best_config = self.best_result.config

        # Results DataFrame
        results_df = pd.DataFrame([r.to_dict() for r in self.all_results])
        results_df = results_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        self._print_final_report(results_df)
        return self.best_result, results_df

    def _print_final_report(self, results_df: pd.DataFrame):
        print("\n" + "="*70)
        print("EINSTEIN ARENA FINAL REPORT")
        print("="*70)

        print(f"\n* BEST CONFIGURATION: {self.best_config.config_id}")
        print(f"  LoRA r={self.best_config.lora_r}, β={self.best_config.beta}, "
              f"T_min={self.best_config.t_min}d, WPAD={self.best_config.wpad_type}")
        print(f"  MoE K={self.best_config.top_k}, blend={self.best_config.moe_weight}, "
              f"equal_groups={self.best_config.group_equal_weight}")
        print(f"\n  Primary metrics:")
        print(f"    DR-OPE improvement: {self.best_result.drope_improvement:+.4f}")
        print(f"    IMI reduction:      {self.best_result.imi_reduction:.4f}")
        print(f"    Equity-IMI gap:     {self.best_result.equity_imi_gap:.4f}")
        print(f"    ESS:                {self.best_result.ess:.0f} "
              f"{'[OK] (>500)' if self.best_result.ess_adequate else '[FAIL] (<500)'}")
        print(f"    Abstention rate:    {self.best_result.abstention_rate:.1%}")
        print(f"    Composite score:    {self.best_result.composite_score:.4f}")

        if self.best_result.adversarial_critiques:
            print(f"\n  Remaining reviewer concerns:")
            for c in self.best_result.adversarial_critiques:
                print(f"    • {c[:100]}")

        print(f"\n  Top-5 configurations:")
        top5_cols = ["config_id", "composite_score", "drope_improvement",
                     "imi_reduction", "ess", "adversarial_fatal"]
        top5 = results_df[top5_cols].head(5)
        print(top5.to_string(index=False))
        print("="*70)

    def save_results(self, output_path: str):
        """Save all results to JSON for paper supplementary."""
        results = [r.to_dict() for r in self.all_results]
        with open(output_path, "w") as f:
            json.dump({
                "best_config": self.best_config.to_dict() if self.best_config else {},
                "best_metrics": {
                    "drope_improvement": self.best_result.drope_improvement,
                    "imi_reduction": self.best_result.imi_reduction,
                    "equity_imi_gap": self.best_result.equity_imi_gap,
                    "ess": self.best_result.ess,
                    "composite_score": self.best_result.composite_score,
                } if self.best_result else {},
                "all_results": results,
                "n_configs_evaluated": len(results),
                "n_fatal_eliminated": sum(1 for r in self.all_results if r.adversarial_fatal),
            }, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    from data.synthetic_generator import generate_synthetic_population

    print("Generating synthetic population (N=5,000 for Arena speed)...")
    pop = generate_synthetic_population(n_patients=5_000, seed=42)
    rising = pop.patients[pop.patients["rising_risk"]].reset_index(drop=True)

    arena = EinsteinArena(
        n_generations=2,
        population_size=6,
        survivors_per_round=3,
        seed=42,
    )

    best_result, results_df = arena.run(rising, pop.wpad_pairs, verbose=True)
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arena.save_results(str(DEFAULT_RESULTS_DIR / "einstein_arena_results.json"))

    print(f"\n* OPTIMAL CONFIG: {best_result.config.config_id}")
    print(f"  Use these hyperparameters for the main PEARL paper results.")
