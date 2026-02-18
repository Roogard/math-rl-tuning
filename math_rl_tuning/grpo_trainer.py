"""
GRPO (Group Relative Policy Optimization) trainer.

Orchestrates RL-based training on top of a previously SFT-tuned model:
  1. Merge the SFT adapter into the base model
  2. Load the merged model with Unsloth (fast) or standard HF (fallback)
  3. Attach a fresh LoRA adapter for RL updates
  4. Format GRPO dataset (prompt + ground-truth answer)
  5. Configure and run TRL's GRPOTrainer with a custom reward function
  6. Save the RL adapter + tokenizer
"""

import os
import shutil
from typing import Optional, Tuple

# ── CRITICAL: import TRL FIRST, before unsloth touches it ─────────────
# ``import unsloth`` auto-patches trl.GRPOTrainer with
# UnslothGRPOTrainer, which has a known tensor-shape bug on Mistral
# (github.com/unslothai/unsloth/issues/1958).
# We import GRPOTrainer/GRPOConfig from trl NOW — before any unsloth
# import — so our local binding keeps the vanilla (working) class.
# Unsloth is only imported lazily inside load_unsloth_model() later.
# ───────────────────────────────────────────────────────────────────────
from trl import GRPOTrainer, GRPOConfig

import importlib.util
HAS_UNSLOTH = importlib.util.find_spec("unsloth") is not None
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
        max_completion_length=gc.max_completion_length,
        num_train_epochs=gc.num_train_epochs,
        report_to=gc.report_to,
        fp16=not use_bf16,
        bf16=use_bf16,
        beta=gc.beta,
        warmup_ratio=gc.warmup_ratio,
        weight_decay=gc.weight_decay,
        max_grad_norm=gc.max_grad_norm,
        lr_scheduler_type=gc.lr_scheduler_type,
        logging_steps=gc.logging_steps,
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
    2. Load merged model (Unsloth if available, else standard HF)
    3. Attach fresh LoRA adapter for RL
    4. Prepare GRPO dataset (if not provided)
    5. Build reward function
    6. Run GRPOTrainer
    7. Save RL adapter + tokenizer
    8. Optionally copy to Google Drive

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

    # --- Phase 2: Load for RL ---
    print("\n" + "=" * 60)
    print("PHASE 2: Load Model for RL")
    print("=" * 60)
    if HAS_UNSLOTH:
        print("Using Unsloth FastLanguageModel (2x faster training)")
        model, tokenizer = load_unsloth_model(
            cfg, model_path=merged_path, for_training=True
        )
    else:
        print("Unsloth not available — using standard HuggingFace")
        model, tokenizer = load_model_and_tokenizer(
            cfg, stage="grpo", model_path=merged_path
        )

    # ── Tokenizer setup for GRPO batch generation ──────────────────────
    # GRPO generates multiple completions per prompt in a batch.
    # Left-padding is required so prompts are right-aligned and the model
    # can generate continuations.  Without this, it attends to padding
    # tokens → garbage logits → CUDA device-side assert.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
    print(f"  padding_side={tokenizer.padding_side}, "
          f"pad_token_id={tokenizer.pad_token_id}, "
          f"vocab_size={len(tokenizer)}")

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
