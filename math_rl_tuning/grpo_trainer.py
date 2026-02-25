"""
GRPO (Group Relative Policy Optimization) trainer.

Simplified pipeline matching the official Unsloth Mistral GRPO notebook:
  1. Merge the SFT adapter into the base model
  2. Load with Unsloth FastLanguageModel (or standard HF fallback)
  3. Train with TRL GRPOTrainer
  4. Save
"""

import os
import shutil
from typing import Optional, Tuple

from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

from math_rl_tuning.config import Config
from math_rl_tuning.model import (
    merge_adapter,
    load_model_and_tokenizer,
    load_unsloth_model,
    patch_vocab_size,
)
from math_rl_tuning.data import prepare_grpo_data
from math_rl_tuning.rewards import build_reward_functions
from math_rl_tuning.utils import patch_colab_fileno, is_colab, is_bf16_supported

import importlib.util
HAS_UNSLOTH = importlib.util.find_spec("unsloth") is not None


def build_grpo_config(cfg: Config):
    """Create a TRL GRPOConfig from the project configuration."""
    gc = cfg.grpo_training

    return GRPOConfig(
        output_dir=cfg.paths.grpo_output_dir,
        learning_rate=gc.learning_rate,
        adam_beta1=gc.adam_beta1,
        adam_beta2=gc.adam_beta2,
        weight_decay=gc.weight_decay,
        warmup_ratio=gc.warmup_ratio,
        lr_scheduler_type=gc.lr_scheduler_type,
        optim=gc.optim,
        logging_steps=gc.logging_steps,
        per_device_train_batch_size=gc.per_device_train_batch_size,
        gradient_accumulation_steps=gc.gradient_accumulation_steps,
        num_generations=gc.num_generations,
        max_prompt_length=gc.max_prompt_length,
        max_completion_length=gc.max_completion_length,
        num_train_epochs=gc.num_train_epochs,
        save_steps=500,
        max_grad_norm=gc.max_grad_norm,
        report_to=gc.report_to,
    )


def run_grpo_training(
    cfg: Config,
    sft_adapter_path: Optional[str] = None,
    grpo_dataset: Optional[Dataset] = None,
    save_to_drive: bool = False,
) -> Tuple:
    """Run the full GRPO pipeline."""
    sft_path = sft_adapter_path or cfg.paths.sft_output_dir
    patch_colab_fileno()

    # --- Phase 1: Merge SFT adapter ---
    print("=" * 60)
    print("PHASE 1: Merge SFT Adapter")
    print("=" * 60)
    merged_path = merge_adapter(cfg, adapter_path=sft_path)
    patch_vocab_size(merged_path)

    # --- Phase 2: Load for RL ---
    print("\n" + "=" * 60)
    print("PHASE 2: Load Model for RL")
    print("=" * 60)
    if HAS_UNSLOTH:
        print("Using Unsloth FastLanguageModel")
        model, tokenizer = load_unsloth_model(
            cfg, model_path=merged_path, for_training=True
        )
    else:
        print("Unsloth not available — using standard HuggingFace")
        model, tokenizer = load_model_and_tokenizer(
            cfg, stage="grpo", model_path=merged_path
        )

    # --- Phase 3: Prepare dataset ---
    print("\n" + "=" * 60)
    print("PHASE 3: Prepare GRPO Dataset")
    print("=" * 60)
    if grpo_dataset is None:
        grpo_dataset = prepare_grpo_data(cfg, tokenizer)

    # --- Phase 4: Train ---
    print("\n" + "=" * 60)
    print("PHASE 4: GRPO Training")
    print("=" * 60)
    reward_fns = build_reward_functions(cfg.rewards)
    training_args = build_grpo_config(cfg)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
    )

    model.print_trainable_parameters()
    print("Starting GRPO training...")
    trainer.train()

    # --- Phase 5: Save ---
    print("\n" + "=" * 60)
    print("PHASE 5: Saving RL Model")
    print("=" * 60)
    save_dir = cfg.paths.grpo_output_dir
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"RL model saved to: {save_dir}")

    if save_to_drive:
        _copy_to_drive(save_dir, cfg)

    return trainer, model, tokenizer


def _copy_to_drive(source_dir: str, cfg: Config):
    """Copy saved model to Google Drive (Colab only)."""
    if not is_colab():
        return
    from math_rl_tuning.utils import mount_google_drive
    mount_google_drive(cfg.paths.drive_mount)
    dest = os.path.join(cfg.paths.drive_save_dir, "grpo")
    if os.path.exists(dest):
        print(f"Drive destination already exists: {dest}. Skipping.")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(source_dir, dest)
    print(f"Copied to Drive: {dest}")
