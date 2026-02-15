"""
GRPO (Group Relative Policy Optimization) trainer.

Orchestrates RL-based training on top of a previously SFT-tuned model:
  1. Merge the SFT adapter into the base model
  2. Load the merged model with standard HF + attach fresh LoRA
  3. Format GRPO dataset (prompt + ground-truth answer)
  4. Configure and run TRL's GRPOTrainer with a custom reward function
  5. Save the RL adapter + tokenizer

Uses the same standard HuggingFace stack as SFT (transformers + peft +
trl + bitsandbytes).  No unsloth or vLLM required.
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
    patch_vocab_size,
)
from math_rl_tuning.data import prepare_grpo_data
from math_rl_tuning.rewards import build_reward_function
from math_rl_tuning.utils import clean_memory, patch_colab_fileno, is_colab, is_bf16_supported


# ---------------------------------------------------------------------------
# Build GRPOConfig from project config
# ---------------------------------------------------------------------------

def build_grpo_config(cfg: Config):
    """Create a TRL GRPOConfig from the project configuration."""
    gc = cfg.grpo_training
    use_bf16 = is_bf16_supported()

    return GRPOConfig(
        output_dir=cfg.paths.grpo_output_dir,
        learning_rate=gc.learning_rate,
        per_device_train_batch_size=gc.per_device_train_batch_size,
        gradient_accumulation_steps=gc.gradient_accumulation_steps,
        num_generations=gc.num_generations,
        max_prompt_length=gc.max_prompt_length,
        max_completion_length=gc.max_completion_length,
        num_train_epochs=gc.num_train_epochs,
        report_to=gc.report_to,
        fp16=not use_bf16,
        bf16=use_bf16,
        beta=gc.beta,
    )


# ---------------------------------------------------------------------------
# High-level GRPO training entry point
# ---------------------------------------------------------------------------

def run_grpo_training(
    cfg: Config,
    sft_adapter_path: Optional[str] = None,
    grpo_dataset: Optional[Dataset] = None,
    save_to_drive: bool = False,
) -> Tuple:
    """
    Run the full GRPO reinforcement learning pipeline:

    1. Merge SFT adapter into base model (if not already done)
    2. Load merged model with standard HF + attach fresh LoRA
    3. Prepare GRPO dataset (if not provided)
    4. Build reward function
    5. Run GRPOTrainer
    6. Save RL adapter + tokenizer
    7. Optionally copy to Google Drive

    Args:
        cfg: Project configuration.
        sft_adapter_path: Path to the saved SFT LoRA adapter.
                          Defaults to ``cfg.paths.sft_output_dir``.
        grpo_dataset: Pre-prepared GRPO dataset. If None, will be prepared.
        save_to_drive: If True and running in Colab, copy outputs to Drive.

    Returns:
        (trainer, model, tokenizer) — the trained objects.
    """
    sft_path = sft_adapter_path or cfg.paths.sft_output_dir

    # --- Colab compatibility patches ---
    patch_colab_fileno()

    # --- Phase 1: Merge SFT adapter ---
    print("=" * 60)
    print("PHASE 1: Merge SFT Adapter")
    print("=" * 60)
    merged_path = merge_adapter(cfg, adapter_path=sft_path)
    patch_vocab_size(merged_path)

    # --- Phase 2: Load for RL (standard HF, same stack as SFT) ---
    print("\n" + "=" * 60)
    print("PHASE 2: Load Model for RL")
    print("=" * 60)
    model, tokenizer = load_model_and_tokenizer(
        cfg, stage="grpo", model_path=merged_path
    )

    # --- Phase 3: Prepare dataset ---
    print("\n" + "=" * 60)
    print("PHASE 3: Prepare GRPO Dataset")
    print("=" * 60)
    if grpo_dataset is None:
        grpo_dataset = prepare_grpo_data(cfg, tokenizer)

    # --- Phase 4: Build reward & train ---
    print("\n" + "=" * 60)
    print("PHASE 4: GRPO Training")
    print("=" * 60)
    reward_fn = build_reward_function(cfg.rewards)
    training_args = build_grpo_config(cfg)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
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

    # --- Optionally copy to Drive ---
    if save_to_drive:
        _copy_to_drive(save_dir, cfg)

    return trainer, model, tokenizer


def _copy_to_drive(source_dir: str, cfg: Config):
    """Copy saved model to Google Drive (Colab only)."""
    if not is_colab():
        print("Not in Colab — skipping Drive copy.")
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
