"""
Math RL Tuning — Fine-tune and RL-train LLMs for mathematical reasoning.

Two-stage training pipeline:
    1. SFT (Supervised Fine-Tuning) on NuminaMath-CoT with Qwen2.5-Math-7B
       — teaches the model to reason step-by-step and output answers in \boxed{}
    2. GRPO (Group Relative Policy Optimization) reinforcement learning
       — further improves accuracy via reward signals (correctness + format)
    3. Evaluation and inference utilities

Key modules:
    config      — typed configuration dataclasses and YAML loader
    data        — dataset loading, filtering, and prompt formatting
    model       — QLoRA model loading and LoRA adapter management
    sft_trainer — TRL SFTTrainer wrapper for stage 1
    grpo_trainer— TRL GRPOTrainer wrapper for stage 2
    rewards     — reward functions consumed by GRPOTrainer
    evaluation  — batched greedy generation and accuracy scoring
    inference   — streaming and one-shot generation for interactive use
    utils       — shared helpers (memory, env, answer extraction)
"""

__version__ = "0.1.0"
