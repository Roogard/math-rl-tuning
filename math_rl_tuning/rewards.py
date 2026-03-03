"""
Reward functions for GRPO reinforcement learning.

Matches the official Unsloth GRPO notebook pattern.
TRL passes completions as list of list of message dicts:
    completions = [[{"role": "assistant", "content": "..."}], ...]

Five reward functions (matching official notebook):
  1. correctness  — exact match of extracted answer (2.0 if correct)
  2. int_check    — is the extracted answer an integer? (0.5)
  3. strict_format — perfect XML structure (0.5)
  4. soft_format   — relaxed XML structure (0.5)
  5. xmlcount      — graduated per-tag scoring (up to 0.5)
"""

import re
from typing import List

from math_rl_tuning.config import RewardsConfig


def _extract_xml_answer(text: str) -> str:
    """Extract content from <answer>...</answer> tags."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def _get_content(completion) -> str:
    """Extract text content from a completion (message dict or string)."""
    if isinstance(completion, list):
        return completion[0]["content"]
    if isinstance(completion, dict):
        return completion["content"]
    return str(completion)


# ---------------------------------------------------------------------------
# Reward functions (matching official Unsloth notebook)
# ---------------------------------------------------------------------------

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Semantic match of extracted answer vs ground truth using math-verify."""
    from math_verify import parse, verify

    responses = [_get_content(c) for c in completions]
    extracted = [_extract_xml_answer(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, answer):
        try:
            gold = parse(a, extraction_mode="first_match")
            pred = parse(r, extraction_mode="first_match")
            rewards.append(2.0 if verify(gold, pred) else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def int_reward_func(completions, **kwargs) -> list[float]:
    """Check if extracted answer is a pure integer."""
    responses = [_get_content(c) for c in completions]
    extracted = [_extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Exact XML format: <reasoning>\\n...\\n</reasoning>\\n<answer>\\n...\\n</answer>\\n"""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [_get_content(c) for c in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Relaxed XML format: <reasoning>...</reasoning> then <answer>...</answer>"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [_get_content(c) for c in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def _count_xml(text: str) -> float:
    """Graduated scoring for individual XML tags."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Graduated XML tag scoring."""
    contents = [_get_content(c) for c in completions]
    return [_count_xml(c) for c in contents]


# ---------------------------------------------------------------------------
# Build reward function list for GRPOTrainer
# ---------------------------------------------------------------------------

def build_reward_functions(rewards_cfg: RewardsConfig = None):
    """
    Return list of reward functions matching the official Unsloth notebook.

    Usage::

        reward_fns = build_reward_functions(cfg.rewards)
        trainer = GRPOTrainer(reward_funcs=reward_fns, ...)
    """
    return [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ]
