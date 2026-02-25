"""
Supervised Fine-Tuning (SFT) trainer.

Wraps HuggingFace TRL's SFTTrainer with project-specific configuration,
including tokenizer setup, training arguments, and save/load helpers.
"""

import os
from datetime import datetime
from typing import Optional, Tuple

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer

from math_rl_tuning.config import Config
from math_rl_tuning.model import load_model_and_tokenizer
from math_rl_tuning.data import prepare_sft_data
from math_rl_tuning.utils import clean_memory


# ---------------------------------------------------------------------------
# Tokenizer setup (Mistral-specific)
# ---------------------------------------------------------------------------

def setup_tokenizer(tokenizer: AutoTokenizer, max_length: int = 2048):
    """
    Configure the tokenizer with correct special tokens, padding, and
    max length for Mistral-Instruct fine-tuning.
    """
    tokenizer.model_max_length = max_length
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Using eos_token as pad_token (id={tokenizer.eos_token_id})")

    return tokenizer


# ---------------------------------------------------------------------------
# Build SFTConfig from project config
# ---------------------------------------------------------------------------

def build_sft_config(cfg: Config, output_dir: Optional[str] = None) -> SFTConfig:
    """Create a TRL SFTConfig from the project configuration."""
    from math_rl_tuning.utils import is_bf16_supported

    sc = cfg.sft_training
    out = output_dir or cfg.paths.sft_output_dir

    # Auto-detect precision if neither fp16 nor bf16 explicitly set
    use_bf16 = sc.bf16
    use_fp16 = sc.fp16
    if not use_bf16 and not use_fp16:
        use_bf16 = is_bf16_supported()
        use_fp16 = not use_bf16
        print(f"Auto-detected precision: bf16={use_bf16}, fp16={use_fp16}")

    # Build WandB run name
    run_name = None
    if sc.report_to == "wandb":
        run_name = f"{sc.wandb_project}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    return SFTConfig(
        output_dir=out,
        num_train_epochs=sc.num_train_epochs,
        per_device_train_batch_size=sc.per_device_train_batch_size,
        per_device_eval_batch_size=sc.per_device_eval_batch_size,
        gradient_accumulation_steps=sc.gradient_accumulation_steps,
        gradient_checkpointing=sc.gradient_checkpointing,
        bf16=use_bf16,
        fp16=use_fp16,
        learning_rate=sc.learning_rate,
        lr_scheduler_type=sc.lr_scheduler_type,
        warmup_steps=sc.warmup_steps,
        logging_steps=sc.logging_steps,
        eval_strategy=sc.eval_strategy,
        neftune_noise_alpha=sc.neftune_noise_alpha,
        max_length=sc.max_length,
        dataset_text_field=sc.dataset_text_field,
        completion_only_loss=sc.completion_only_loss,
        packing=sc.packing,
        dataset_num_proc=sc.dataset_num_proc,
        push_to_hub=False,
        report_to=sc.report_to,
        run_name=run_name,
        resume_from_checkpoint=True,
    )


# ---------------------------------------------------------------------------
# High-level training entry point
# ---------------------------------------------------------------------------

def run_sft_training(
    cfg: Config,
    train_ds: Optional[Dataset] = None,
    val_ds: Optional[Dataset] = None,
    save_to_drive: bool = False,
) -> Tuple:
    """
    Run the full SFT training pipeline:

    1. Load and prepare data (if not provided)
    2. Load model with QLoRA
    3. Setup tokenizer
    4. Configure and run SFTTrainer
    5. Save model + tokenizer
    6. Optionally copy to Google Drive

    Args:
        cfg: Project configuration.
        train_ds: Pre-prepared training dataset. If None, will be prepared.
        val_ds: Pre-prepared validation dataset. If None, will be prepared.
        save_to_drive: If True and running in Colab, copy outputs to Drive.

    Returns:
        (trainer, model, tokenizer) — the trained objects.
    """
    # 1. Data
    if train_ds is None or val_ds is None:
        print("=" * 60)
        print("PHASE 1: Data Preparation")
        print("=" * 60)
        train_ds, val_ds = prepare_sft_data(cfg)

    # 2. Model
    print("\n" + "=" * 60)
    print("PHASE 2: Model Loading")
    print("=" * 60)
    model, tokenizer = load_model_and_tokenizer(cfg, stage="sft")

    # 3. Tokenizer
    tokenizer = setup_tokenizer(tokenizer, max_length=cfg.sft_training.max_length)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # 4. Training
    print("\n" + "=" * 60)
    print("PHASE 3: Training")
    print("=" * 60)
    sft_config = build_sft_config(cfg)

    # Note: LoRA is already applied inside load_model_and_tokenizer(),
    # so we do NOT pass peft_config here (that would double-wrap).
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    # 5. Save
    print("\n" + "=" * 60)
    print("PHASE 4: Saving")
    print("=" * 60)
    save_dir = cfg.paths.sft_output_dir
    os.makedirs(save_dir, exist_ok=True)

    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to: {save_dir}")

    # 6. Optionally copy to Drive
    if save_to_drive:
        _copy_to_drive(save_dir, cfg)

    # Switch to inference mode so callers can generate immediately
    model.eval()
    model.config.use_cache = True

    return trainer, model, tokenizer


def _copy_to_drive(source_dir: str, cfg: Config):
    """Copy saved model to Google Drive (Colab only)."""
    import shutil
    from math_rl_tuning.utils import is_colab, mount_google_drive

    if not is_colab():
        print("Not in Colab — skipping Drive copy.")
        return

    mount_google_drive(cfg.paths.drive_mount)

    dest = os.path.join(cfg.paths.drive_save_dir, "sft")
    if os.path.exists(dest):
        print(f"Drive destination already exists: {dest}. Skipping.")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(source_dir, dest)
    print(f"Copied to Drive: {dest}")
