"""
PEARL Configuration
All tunable parameters in one place; referenced by all modules.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WPADConfig:
    # Minimum gap duration (days) to qualify as a WPAD event
    t_min_days: int = 60
    # Outcome ascertainment window (days)
    outcome_window_days: int = 90
    # Secondary outcome window (days) — readmission, for comparability
    secondary_outcome_window_days: int = 30
    # Target within-patient pairs
    target_wpad_pairs: int = 5000
    # Feasibility gate: minimum before contingency switch
    feasibility_gate_pairs: int = 3000
    # Pairs per demographic group for equity upsampling
    min_pairs_per_group: int = 500
    # IPTW weight clip bounds for cross-patient pairs
    iptw_clip_low: float = 0.1
    iptw_clip_high: float = 10.0
    # Weight for weak positive pairs (both windows good)
    weak_positive_weight: float = 0.5
    # ESS minimum (absolute count) for DR-OPE reliability
    ess_minimum: int = 500
    # Covariates used for T1-T6 falsification (Charlson, prior ED, diagnoses)
    balance_covariates: List[str] = field(default_factory=lambda: [
        "charlson_score", "prior_ed_visits_6mo", "prior_hosp_6mo",
        "n_chronic_conditions", "age", "adi_percentile"
    ])


@dataclass
class TrainingConfig:
    # Backbone model
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    # LoRA ranks to search over
    lora_ranks: List[int] = field(default_factory=lambda: [16, 32, 64])
    # Primary rank (best per DR-OPE search)
    lora_r: int = 64
    lora_alpha: int = 128  # = 2 * lora_r
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    # DPO beta (KL regularization)
    beta: float = 0.1
    beta_search: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    # QLoRA quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # effective batch = 32
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_seq_length: int = 2048
    # Group-stratified DPO
    demographic_groups: List[str] = field(default_factory=lambda: [
        "race_eth", "primary_language", "adi_quintile", "disability_status"
    ])
    # Abstention: DPO log-ratio margin below which PEARL defers
    abstention_threshold: float = 0.3  # tuned on calibration set
    # Checkpointing
    output_dir: str = "/Users/sanjaybasu/pearl/outputs/checkpoints"
    eval_steps: int = 100
    save_steps: int = 500


@dataclass
class MoEConfig:
    """Mixture of Experts: 14 specialized LoRA adapters + learned router."""
    n_experts: int = 14
    expert_names: List[str] = field(default_factory=lambda: [
        "care_access",          # PCP appointments, care coordination
        "clinical_other",       # Dental, eye care, wellness (catch-all)
        "diabetes",             # Diabetes management
        "financial_benefits",   # Financial, insurance, legal, employment
        "food_security",        # Food insecurity, nutrition
        "heart_failure",        # Heart failure management
        "housing",              # Housing instability, quality
        "hypertension",         # Hypertension management
        "maternal",             # Maternity, prenatal, postpartum
        "medication_adherence", # Medication adherence/optimization
        "mental_health",        # Depression, anxiety, MH/BH
        "pulmonary",            # Asthma/COPD
        "substance_use",        # SUD, alcohol, smoking cessation
        "transport_utilities",  # Transportation, utilities, childcare
    ])
    router_hidden_dim: int = 256
    router_top_k: int = 2  # activate top-2 experts per patient
    router_load_balancing_coef: float = 0.01  # entropy regularization
    expert_lora_r: int = 32  # experts use smaller r than base


@dataclass
class EvaluationConfig:
    # DR-OPE: AIPW estimator
    n_bootstrap_drope: int = 1000
    # Conformal prediction
    conformal_alpha: float = 0.1  # 90% prediction intervals
    # Camden reanalysis
    camden_imi_threshold: float = 0.02  # 2pp minimum IMI improvement
    # RDD cross-check (bandwidth in risk-score units)
    rdd_bandwidth: float = 0.05
    # Sensitivity analysis anchored values
    sensitivity_t_min: List[int] = field(default_factory=lambda: [30, 60, 90])
    sensitivity_iptw_clip: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0])
    sensitivity_beta: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    sensitivity_outcome_window: List[int] = field(default_factory=lambda: [14, 30, 60])


@dataclass
class PEARLConfig:
    wpad: WPADConfig = field(default_factory=WPADConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Reproducibility
    seed: int = 42
    # Data paths
    waymark_data_path: Optional[str] = None  # set if running on Waymark infra
    mimic_note_path: Optional[str] = None
    synthetic_demo: bool = True  # use synthetic data if no real data available
    output_base: str = "/Users/sanjaybasu/pearl/outputs"


# Singleton default config
DEFAULT_CONFIG = PEARLConfig()
