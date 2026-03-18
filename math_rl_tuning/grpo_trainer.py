"""
GRPO (Group Relative Policy Optimization) trainer.

Simplified pipeline matching the official Unsloth GRPO notebook:
  1. Merge the SFT adapter into the base model
  2. Load with Unsloth FastLanguageModel
  3. Train with TRL GRPOTrainer
  4. Save
"""

"""
Works as a backend for the GRPO notebook

The pain here is the imports between TRL and Unsloth, caused lots of pain and suffering and debugging.
The lesson at the end of the day here is figure out your imports and environment before you start coding, 
especially with RL where training can be very slow and you don't want to waste time on import bugs.
"""

import os
from collections import defaultdict
from typing import Optional, Tuple

from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
from datasets import Dataset
from unsloth import FastLanguageModel

from math_rl_tuning.config import Config
from math_rl_tuning.model import (
    merge_adapter,
    load_unsloth_model,
    patch_vocab_size,
)
from math_rl_tuning.data import prepare_grpo_data
from math_rl_tuning.rewards import build_reward_functions
from math_rl_tuning.utils import patch_colab_fileno, is_colab, is_bf16_supported, copy_to_drive


class RewardTracker:
    """
    Wraps a reward function to capture return values for logging.

    TRL calls reward functions internally and doesn't expose the raw scores.
    By wrapping each function, we intercept the return values so
    RewardLoggingCallback can log per-function means to WandB.
    """

    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__
        self.__qualname__ = getattr(func, "__qualname__", func.__name__)
        self.last_rewards = []

    def __call__(self, *args, **kwargs):
        rewards = self._func(*args, **kwargs)
        self.last_rewards = rewards
        return rewards


class RewardLoggingCallback(TrainerCallback):
    """Logs per-reward-function means to WandB and stdout, and tracks history for plotting."""

    def __init__(self, tracked_fns):
        self.tracked_fns = tracked_fns
        self.history = defaultdict(list)
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        parts = []
        for fn in self.tracked_fns:
            if fn.last_rewards:
                mean_val = sum(fn.last_rewards) / len(fn.last_rewards)
                logs[f"reward/{fn.__name__}"] = round(mean_val, 4)
                self.history[fn.__name__].append(mean_val)
                parts.append(f"{fn.__name__}: {mean_val:.4f}")
        if parts:
            self.steps.append(step)
            print(f"[step {step}] " + " | ".join(parts))

    def plot(self):
        """Display a matplotlib chart of all reward curves. Call after training."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, values in self.history.items():
            ax.plot(self.steps[: len(values)], values, label=name, marker=".", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Reward Functions Over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


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
        warmup_ratio=gc.warmup_ratio,
        weight_decay=gc.weight_decay,
        max_grad_norm=gc.max_grad_norm,
        lr_scheduler_type=gc.lr_scheduler_type,
        logging_steps=gc.logging_steps,
        save_strategy=gc.save_strategy,
        save_steps=gc.save_steps,
        save_total_limit=gc.save_total_limit,
        optim=gc.optim,
        adam_beta1=gc.adam_beta1,
        adam_beta2=gc.adam_beta2,
    )


def run_grpo_training(
    cfg: Config,
    sft_adapter_path: Optional[str] = None,
    grpo_dataset: Optional[Dataset] = None,
    save_to_drive: bool = False,
    checkpoint_path: Optional[str] = None,
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
    gc = cfg.grpo_training
    # Total sequence length = prompt + completion.
    # Unsloth uses a fixed-size KV cache, so it needs the combined length upfront.
    grpo_seq_len = gc.max_prompt_length + gc.max_completion_length
    model, tokenizer = load_unsloth_model(
        cfg, model_path=merged_path, for_training=True,
        max_seq_length=grpo_seq_len,
    )

    # Ensure warnings_issued exists (TRL's GRPOTrainer uses it as a dict)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

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

    # Wrap reward functions for per-function WandB logging
    tracked_fns = [RewardTracker(fn) for fn in reward_fns]
    reward_callback = RewardLoggingCallback(tracked_fns)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=tracked_fns,
        args=training_args,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
        callbacks=[reward_callback],
    )

    model.print_trainable_parameters()
    print("Starting GRPO training...")
    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
    trainer.train(resume_from_checkpoint=checkpoint_path)

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
        copy_to_drive(save_dir, "grpo", cfg)

    return trainer, model, tokenizer, reward_callback
