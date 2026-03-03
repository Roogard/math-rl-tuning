r"""
Reward functions for GRPO reinforcement learning.

TRL passes completions as list of list of message dicts:
    completions = [[{"role": "assistant", "content": "..."}], ...]

Five reward functions:
  1. correctness    — semantic match of \boxed{} answer via math-verify (2.0)
  2. int_check      — is the extracted answer an integer? (0.5)
  3. boxed_format   — does the output contain \boxed{}? (0.5)
  4. strict_format  — output ends cleanly with \boxed{} (0.5)
  5. boxed_count    — graduated scoring for \boxed{} usage (up to 0.5)
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
# Reward functions
# ---------------------------------------------------------------------------

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    r"""Semantic match of extracted \boxed{} answer vs ground truth using math-verify."""
    responses = [_get_content(c) for c in completions]
    extracted = [extract_boxed(r) for r in responses]
    return [2.0 if math_verify_equal(a, e) else 0.0 for e, a in zip(extracted, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    r"""Check if extracted \boxed{} answer is a pure integer."""
    responses = [_get_content(c) for c in completions]
    extracted = [extract_boxed(r) or "" for r in responses]
    return [0.5 if e.isdigit() else 0.0 for e in extracted]


def boxed_format_reward_func(completions, **kwargs) -> list[float]:
    r"""Reward for using \boxed{} format in the answer."""
    responses = [_get_content(c) for c in completions]
    return [0.5 if "\\boxed{" in r else 0.0 for r in responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    r"""Reward for clean format: reasoning followed by \boxed{} near the end."""
    pattern = r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\.?\s*$"
    responses = [_get_content(c) for c in completions]
    return [0.5 if re.search(pattern, r) else 0.0 for r in responses]


def _score_boxed(text: str) -> float:
    """Graduated scoring for boxed answer format."""
    score = 0.0
    if "\\boxed{" in text:
        score += 0.25
    boxed = extract_boxed(text)
    if boxed is not None:
        score += 0.125
        # Penalize trailing text after boxed (model should stop after the answer)
        last_boxed_end = text.rfind("}")
        trailing = text[last_boxed_end + 1:].strip() if last_boxed_end != -1 else ""
        if len(trailing) < 20:
            score += 0.125
    return score


def boxed_count_reward_func(completions, **kwargs) -> list[float]:
    r"""Graduated \boxed{} format scoring."""
    contents = [_get_content(c) for c in completions]
    return [_score_boxed(c) for c in contents]


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
    return [
        boxed_count_reward_func,
        boxed_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ]
