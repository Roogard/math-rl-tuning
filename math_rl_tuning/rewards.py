r"""
Reward functions for GRPO reinforcement learning.

TRL passes completions as list of list of message dicts:
    completions = [[{"role": "assistant", "content": "..."}], ...]

Three reward functions:
  1. correctness    — semantic match of \boxed{} answer via math-verify
  2. boxed_format   — does the output contain \boxed{}?
  3. strict_format  — output ends cleanly with \boxed{}
"""

import re

from math_rl_tuning.config import RewardsConfig
from math_rl_tuning.utils import extract_boxed, math_verify_equal


def _get_content(completion) -> str:
    """Extract text content from a completion (message dict or string)."""
    if isinstance(completion, list):
        return completion[0]["content"]
    if isinstance(completion, dict):
        return completion["content"]
    return str(completion)


# ---------------------------------------------------------------------------
# Reward functions (use config values via closure)
# ---------------------------------------------------------------------------

def _make_correctness_reward(cfg: RewardsConfig):
    r"""Correctness reward matching the evaluation pipeline exactly."""
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [_get_content(c) for c in completions]
        results = []

        for r, a in zip(responses, answer):
            pred = extract_boxed(r)
            a_str = str(a).strip()

            # Method 1: math_verify_equal (same lib as eval)
            is_correct = math_verify_equal(a_str, str(r))

            # Method 2: fallback — direct comparison of extracted boxed values
            if not is_correct and pred is not None:
                is_correct = pred.strip() == a_str

            results.append(cfg.correct_bonus if is_correct else cfg.incorrect_penalty)

        return results
    correctness_reward_func.__name__ = "correctness_reward_func"
    return correctness_reward_func


def _make_boxed_format_reward(cfg: RewardsConfig):
    r"""Reward for using \boxed{} format; penalize missing it."""
    def boxed_format_reward_func(completions, **kwargs) -> list[float]:
        responses = [_get_content(c) for c in completions]
        return [
            cfg.format_bonus if "\\boxed{" in r else cfg.format_penalty
            for r in responses
        ]
    boxed_format_reward_func.__name__ = "boxed_format_reward_func"
    return boxed_format_reward_func


def _make_strict_format_reward(cfg: RewardsConfig):
    r"""Reward for clean format: reasoning followed by \boxed{} near the end."""
    pattern = r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\.?\s*$"
    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        responses = [_get_content(c) for c in completions]
        return [
            cfg.strict_format_bonus if re.search(pattern, r) else 0.0
            for r in responses
        ]
    strict_format_reward_func.__name__ = "strict_format_reward_func"
    return strict_format_reward_func


# ---------------------------------------------------------------------------
# Build reward function list for GRPOTrainer
# ---------------------------------------------------------------------------

def build_reward_functions(rewards_cfg: RewardsConfig = None):
    r"""
    Return list of reward functions for GRPO training.

    All functions expect model completions to use \boxed{} format
    (matching the system prompt used in both SFT and GRPO).

    Usage::

        reward_fns = build_reward_functions(cfg.rewards)
        trainer = GRPOTrainer(reward_funcs=reward_fns, ...)
    """
    cfg = rewards_cfg or RewardsConfig()
    return [
        _make_boxed_format_reward(cfg),
        _make_strict_format_reward(cfg),
        _make_correctness_reward(cfg),
    ]
