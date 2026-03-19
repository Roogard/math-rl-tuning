"""
Configuration management.

Loads the default YAML config and allows overrides via:
  - A custom YAML file path
  - Keyword arguments at runtime
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Locate the default config shipped with the package
# ---------------------------------------------------------------------------
_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return it as a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (base is mutated)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# Typed dataclasses for each config section (IDE-friendly access)
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    sft_output_dir: str = "./outputs/sft"
    sft_merged_dir: str = "./outputs/sft_merged"
    grpo_output_dir: str = "./outputs/grpo"
    eval_output_dir: str = "./outputs/eval"
    drive_mount: str = "/content/drive"
    drive_save_dir: str = "/content/drive/MyDrive/math-rl-tuning"


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-Math-7B"
    max_seq_length: int = 2048


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"          # NF4 (NormalFloat4) is theoretically optimal
                                               # for weights that follow a normal distribution —
                                               # which most LLM weights do after pretraining.
    bnb_4bit_compute_dtype: str = "bfloat16"  # Upcast to bf16 for the matrix multiplications
                                               # so quantization error doesn't compound during forward pass.
    bnb_4bit_use_double_quant: bool = True     # Quantizes the quantization constants themselves,
                                               # saving ~0.4 bits/parameter with negligible accuracy loss.


@dataclass
class LoRASubConfig:
    r: int = 32           # Rank: controls how many parameters are trainable.
                          # r=32 is a sweet spot for 7B models — enough capacity
                          # without the memory cost of full fine-tuning.
    alpha: int = 64       # Scaling factor; effective learning rate = lr * (alpha / r).
                          # alpha = 2*r is a common default that keeps LoRA updates
                          # at roughly the same scale as standard fine-tuning.
    dropout: float = 0.0  # Dropout on LoRA layers; 0.0 is fine for small adapters.
    bias: str = "none"    # Don't train bias terms — they add little with LoRA.
    task_type: str = "CAUSAL_LM"
    target_modules: Any = "all-linear"  # Apply LoRA to all linear layers; str or list[str]


@dataclass
class LoRAConfig:
    sft: LoRASubConfig = field(default_factory=LoRASubConfig)
    grpo: LoRASubConfig = field(default_factory=lambda: LoRASubConfig(
        r=32, alpha=32, dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))


@dataclass
class DatasetConfig:
    name: str = "AI-MO/NuminaMath-CoT"
    sft_keep_sources: List[str] = field(default_factory=lambda: ["gsm8k", "math", "cn_k12", "orca_math", "synthetic_math"])
    test_drop_sources: List[str] = field(default_factory=list)
    train_per_source: int = 12000
    train_per_source_overrides: Dict[str, int] = field(default_factory=dict)
    val_per_source: int = 500
    random_state: int = 42
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."


#very weird thing happened here, everytime i increased batch size the training took longer.
# think it had to do something with dataloaders, ended up using batch size of 1 here and a cheaper gpu 
@dataclass
class SFTTrainingConfig:
    num_train_epochs: int = 1       # One epoch is usually enough for SFT on a large curated dataset —
                                    # more epochs risk overfitting on the training format.
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True  # Recomputes activations during backward pass instead of
                                         # storing them — trades compute for memory, enabling larger
                                         # effective batch sizes on single-GPU setups.
    fp16: bool = False
    bf16: bool = False              # Both left False so the trainer auto-detects based on GPU capability.
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    logging_steps: int = 25
    eval_strategy: str = "steps"
    neftune_noise_alpha: Optional[float] = None  # NEFTune: adds noise to embeddings during training
                                                  # to improve generalization. Disabled by default.
    eval_steps: int = 100
    max_length: int = 2048
    dataset_text_field: str = "messages"
    completion_only_loss: bool = True  # Only compute loss on the assistant's response tokens, NOT the
                                       # prompt (question + system prompt). Without this, the model
                                       # wastes capacity learning to predict the question itself.
    packing: bool = False              # Packs multiple short examples into one sequence up to max_length
                                       # to avoid wasted padding. Efficient but can mix examples.
    dataset_num_proc: int = 24
    report_to: str = "wandb"
    wandb_project: str = "AI-MONuminaMath"
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    early_stopping_patience: int = 3


@dataclass
class GRPOTrainingConfig:
    learning_rate: float = 5e-6  # Much lower than SFT — RL training is noisy,
                                 # a too-large LR causes policy collapse.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99     # Higher than default (0.999) for better gradient tracking
                                 # in the noisy RL setting (from DeepSeek-R1 recipe).
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 6     # How many completions to sample per prompt.
                                 # GRPO computes advantages relative to the group mean reward —
                                 # more generations = more stable advantage estimates,
                                 # but linearly increases VRAM usage.
    max_prompt_length: int = 512
    max_completion_length: int = 256
    num_train_epochs: int = 3
    report_to: str = "wandb"
    beta: float = 0.01           # KL penalty coefficient. Prevents the RL policy from
                                 # drifting too far from the SFT reference model —
                                 # without this, the model could learn to "game" rewards
                                 # by generating gibberish that happens to match answers.
    grpo_sample_size: int = 7400
    max_grad_norm: float = 0.1
    logging_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.35


@dataclass
class RewardsConfig:
    correct_bonus: float = 3.0
    incorrect_penalty: float = -1.0
    format_bonus: float = 0.5
    format_penalty: float = -0.5
    strict_format_bonus: float = 0.5


@dataclass
class EvaluationConfig:
    num_samples: int = 100
    max_new_tokens: int = 1024
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."
    batch_size: int = 4
    eval_keep_sources: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level config object
# ---------------------------------------------------------------------------

@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sft_training: SFTTrainingConfig = field(default_factory=SFTTrainingConfig)
    grpo_training: GRPOTrainingConfig = field(default_factory=GRPOTrainingConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _apply_dict_to_dataclass(dc, d: dict):
    """Recursively set attributes on a dataclass from a dict."""
    for key, value in d.items():
        if not hasattr(dc, key):
            continue
        attr = getattr(dc, key)
        if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
            _apply_dict_to_dataclass(attr, value)
        else:
            setattr(dc, key, value)


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Build a Config object.

    Priority (lowest → highest):
        1. Defaults baked into the dataclasses
        2. Values from ``configs/default.yaml``
        3. Values from *config_path* (if provided)
        4. Values from *overrides* dict (if provided)
    """
    # Start with defaults from YAML
    raw: Dict[str, Any] = {}
    if _DEFAULT_CONFIG.exists():
        raw = load_yaml(_DEFAULT_CONFIG)

    # Layer on custom YAML
    if config_path is not None:
        custom = load_yaml(config_path)
        raw = deep_merge(raw, custom)

    # Layer on programmatic overrides
    if overrides is not None:
        raw = deep_merge(raw, overrides)

    cfg = Config()
    _apply_dict_to_dataclass(cfg, raw)
    return cfg
