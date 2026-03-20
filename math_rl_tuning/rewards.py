r"""
Reward functions for GRPO reinforcement learning.

TRL passes completions as list of list of message dicts:
    completions = [[{"role": "assistant", "content": "..."}], ...]

Three separate reward functions (rather than one combined score) because
GRPO logs each one individually — this lets us diagnose whether the model
is failing on correctness vs. format during training.

Reward design:
  1. correctness    — semantic match of \boxed{} answer via math-verify (+3.0 / -1.0)
                      The asymmetry (bigger bonus than penalty) encourages exploration.
  2. boxed_format   — does the output contain \boxed{}? (+0.5 / -0.5)
                      Enforces the format the system prompt asked for.
  3. strict_format  — output ends cleanly with \boxed{} (+0.5 / 0.0)
                      Stricter version: rewards concise answers, no trailing text.
                      Zero penalty (not -0.5) so it's aspirational, not punishing.
"""


"""
I also used to give reward for partial credit, like if a model got the right answer
anywhere it'd get credit. But at the end of the day, this just muddied the reward signal and
ended up spamming answers at one point early on. 
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

            # Method 1: math_verify_equal uses symbolic math to compare
            # (e.g. "1/2" == "0.5" == "\frac{1}{2}"). Pass the full response
            # so math-verify can extract and normalize internally.
            is_correct = math_verify_equal(a_str, str(r))

            # Method 2: exact string fallback for when math-verify parsing fails
            # (e.g. MCQ answers like "A", "B", or non-numeric values).
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
    # Regex breakdown:
    #   \\boxed\{          — literal \boxed{
    #   [^{}]*             — content without nested braces
    #   (?:\{[^{}]*\}[^{}]*)* — allow one level of nested braces (e.g. \boxed{\frac{1}{2}})
    #   \}                 — closing brace
    #   \s*\.?\s*$         — optional trailing period/space, then end of string
    pattern = r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\$?\s*\.?\s*$"
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
        _make_correctness_reward(cfg),
    ]
