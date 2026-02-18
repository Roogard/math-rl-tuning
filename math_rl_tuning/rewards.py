"""
Reward functions for GRPO reinforcement learning.

Four separate reward functions that evaluate different aspects:
  1. Exact format compliance  — full XML structure match
  2. Approximate format       — per-tag presence/absence scoring
  3. Answer correctness       — exact match, approximate, or wrong (with penalty)
  4. Number extraction        — can the model produce a parseable number?

All functions use <reasoning>...</reasoning> and <answer>...</answer> XML tags.
"""

import re
from typing import List

from math_rl_tuning.config import RewardsConfig


# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for efficiency)
# ---------------------------------------------------------------------------

# Full format: optional whitespace, <reasoning>...</reasoning>, <answer>...</answer>
_FULL_FORMAT_RE = re.compile(
    r"^\s*<reasoning>.+?</reasoning>\s*<answer>(.+?)</answer>\s*$",
    flags=re.DOTALL,
)

# Extract number(s) from inside <answer>...</answer>
_ANSWER_NUMBER_RE = re.compile(
    r"<answer>.*?([\d,]+\.?\d*)\s*</answer>",
    flags=re.DOTALL,
)


def _normalize_number(s: str) -> str:
    """Strip commas and surrounding whitespace from a number string."""
    return s.replace(",", "").strip()


# ---------------------------------------------------------------------------
# Reward Function 1: Exact Format Compliance
# ---------------------------------------------------------------------------

def reward_format_exact(completions, rewards_cfg: RewardsConfig, **kwargs) -> List[float]:
    """
    High reward for perfect XML structure compliance.
    Ensures the model learns the complete structured output pattern.
    """
    scores = []
    for completion in completions:
        if _FULL_FORMAT_RE.search(completion) is not None:
            scores.append(rewards_cfg.format_exact_bonus)
        else:
            scores.append(0.0)
    return scores


# ---------------------------------------------------------------------------
# Reward Function 2: Approximate Format (per-tag scoring)
# ---------------------------------------------------------------------------

def reward_format_approximate(completions, rewards_cfg: RewardsConfig, **kwargs) -> List[float]:
    """
    Graduated scoring for individual XML tag presence.
    Encourages learning components even when the full pattern isn't perfect.
    """
    scores = []
    for completion in completions:
        score = 0.0
        for tag in ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]:
            if completion.count(tag) == 1:
                score += rewards_cfg.format_tag_bonus
            else:
                score += rewards_cfg.format_tag_penalty
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# Reward Function 3: Answer Correctness
# ---------------------------------------------------------------------------

def reward_correctness(completions, answer, rewards_cfg: RewardsConfig, **kwargs) -> List[float]:
    """
    Graduated scoring for mathematical accuracy:
      - Exact match      → correct_bonus
      - Within 10%       → approximate_bonus
      - Wrong / no answer → incorrect_penalty
    """
    scores = []
    for completion, truth in zip(completions, answer):
        truth_str = str(truth).strip()

        # Try to extract the answer from <answer>...</answer>
        fmt_match = _FULL_FORMAT_RE.search(completion)
        guess = fmt_match.group(1).strip() if fmt_match else None

        if guess is None:
            scores.append(0.0)
            continue

        # Exact string match
        if _normalize_number(guess) == _normalize_number(truth_str):
            scores.append(rewards_cfg.correct_bonus)
            continue

        # Numerical approximate match
        try:
            guess_val = float(_normalize_number(guess))
            truth_val = float(_normalize_number(truth_str))
            if truth_val != 0:
                ratio = guess_val / truth_val
                if 0.9 <= ratio <= 1.1:
                    scores.append(rewards_cfg.approximate_bonus)
                    continue
        except (ValueError, ZeroDivisionError):
            pass

        # Wrong answer — apply penalty
        scores.append(rewards_cfg.incorrect_penalty)

    return scores


# ---------------------------------------------------------------------------
# Reward Function 4: Number Extraction
# ---------------------------------------------------------------------------

def reward_number_extraction(completions, answer, rewards_cfg: RewardsConfig, **kwargs) -> List[float]:
    """
    Tests the model's ability to produce a parseable number inside <answer> tags.
    Complementary to exact format — focuses on numerical output capability.
    """
    scores = []
    for completion, truth in zip(completions, answer):
        truth_str = str(truth).strip()

        num_match = _ANSWER_NUMBER_RE.search(completion)
        if num_match is None:
            scores.append(0.0)
            continue

        try:
            guess_val = float(_normalize_number(num_match.group(1)))
            truth_val = float(_normalize_number(truth_str))
            if guess_val == truth_val:
                scores.append(rewards_cfg.number_extraction_bonus)
            else:
                scores.append(0.0)
        except (ValueError, TypeError):
            scores.append(0.0)

    return scores


# ---------------------------------------------------------------------------
# Build all reward functions for GRPOTrainer
# ---------------------------------------------------------------------------

def build_reward_functions(rewards_cfg: RewardsConfig):
    """
    Return a list of reward function closures compatible with TRL's
    GRPOTrainer ``reward_funcs`` parameter.

    Usage::

        reward_fns = build_reward_functions(cfg.rewards)
        trainer = GRPOTrainer(reward_funcs=reward_fns, ...)
    """

    def fn_format_exact(completions, **kwargs):
        return reward_format_exact(completions, rewards_cfg=rewards_cfg, **kwargs)

    def fn_format_approx(completions, **kwargs):
        return reward_format_approximate(completions, rewards_cfg=rewards_cfg, **kwargs)

    def fn_correctness(completions, answer, **kwargs):
        return reward_correctness(completions, answer=answer, rewards_cfg=rewards_cfg, **kwargs)

    def fn_number_extraction(completions, answer, **kwargs):
        return reward_number_extraction(completions, answer=answer, rewards_cfg=rewards_cfg, **kwargs)

    return [fn_format_exact, fn_format_approx, fn_correctness, fn_number_extraction]
