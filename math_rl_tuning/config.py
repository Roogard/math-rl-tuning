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
    name: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    max_seq_length: int = 2048


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRASubConfig:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Any = "all-linear"  # str or list[str]


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
    grpo_dataset_name: str = "openai/gsm8k"
    name: str = "AI-MO/NuminaMath-CoT"
    sft_keep_sources: List[str] = field(default_factory=lambda: ["gsm8k", "math", "cn_k12", "orca_math", "synthetic_math"])
    test_drop_sources: List[str] = field(default_factory=list)
    grpo_drop_sources: List[str] = field(default_factory=list)
    train_per_source: int = 12000
    val_per_source: int = 500
    random_state: int = 42
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."
    grpo_system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."


@dataclass
class SFTTrainingConfig:
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = False
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    logging_steps: int = 25
    eval_strategy: str = "no"
    neftune_noise_alpha: Optional[float] = None
    eval_steps: int = 100
    max_length: int = 2048
    dataset_text_field: str = "messages"
    completion_only_loss: bool = True
    packing: bool = False
    dataset_num_proc: int = 24
    report_to: str = "wandb"
    wandb_project: str = "AI-MONuminaMath"
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3


@dataclass
class GRPOTrainingConfig:
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 6
    max_prompt_length: int = 512
    max_completion_length: int = 256
    num_train_epochs: int = 3
    report_to: str = "wandb"
    use_vllm: bool = False
    beta: float = 0.01
    grpo_sample_size: int = 7400
    max_grad_norm: float = 0.1
    logging_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3
    use_unsloth: bool = False
    gpu_memory_utilization: float = 0.5


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
    greedy: bool = True
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."
    batch_size: int = 4
    eval_keep_sources: List[str] = field(default_factory=list)


@dataclass
class InferenceConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


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
    inference: InferenceConfig = field(default_factory=InferenceConfig)


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
