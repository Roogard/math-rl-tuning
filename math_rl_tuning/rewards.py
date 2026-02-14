"""
Reward functions for GRPO reinforcement learning.

The reward function scores model completions based on:
  - Length (encourages step-by-step reasoning)
  - Format (uses \\boxed{} syntax)
  - Correctness (exact string match or symbolic LaTeX equality)
  - Partial match (ground truth appears somewhere in the completion)
"""

import re
from typing import List

from math_rl_tuning.config import RewardsConfig
from math_rl_tuning.utils import latex_equal


def compute_rewards(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    rewards_cfg: RewardsConfig,
    **kwargs,
) -> List[float]:
    """
    Score each (completion, ground_truth) pair.

    Args:
        prompts: The input prompts (unused but required by TRL API).
        completions: Model-generated completions.
        answer: Ground-truth answers (one per completion).
        rewards_cfg: Reward hyperparameters.

    Returns:
        List of float scores, one per completion.
    """
    scores = []

    for completion, truth in zip(completions, answer):
        score = 0.0

        # Clean up ground truth
        truth_str = str(truth)
        if "####" in truth_str:
            truth_clean = truth_str.split("####")[-1].strip()
        else:
            truth_clean = truth_str.strip()

        # --- Length bonus: encourage reasoning ---
        if len(completion) > rewards_cfg.length_threshold:
            score += rewards_cfg.length_bonus

        # --- Format bonus: uses \boxed{} ---
        box_match = re.search(r"\\boxed\{(.*?)\}", completion)
        predicted_content = None
        if box_match:
            score += rewards_cfg.format_bonus
            predicted_content = box_match.group(1).strip()

        # --- Correctness check ---
        found_correct = False

        if predicted_content:
            # Exact string match
            if predicted_content == truth_clean:
                found_correct = True
            else:
                # Symbolic LaTeX equality
                if latex_equal(predicted_content, truth_clean):
                    found_correct = True

        # --- Partial match: truth appears anywhere in completion ---
        if not found_correct and truth_clean in completion:
            found_correct = True
            # Partial match gets less reward than boxed correct
            score += rewards_cfg.partial_match_bonus
        elif found_correct:
            score += rewards_cfg.correct_bonus

        scores.append(score)

    return scores


def build_reward_function(rewards_cfg: RewardsConfig):
    """
    Return a reward function closure compatible with TRL's GRPOTrainer
    ``reward_funcs`` parameter.

    Usage::

        reward_fn = build_reward_function(cfg.rewards)
        trainer = GRPOTrainer(reward_funcs=reward_fn, ...)
    """

    def reward_fn(prompts, completions, answer, **kwargs):
        return compute_rewards(
            prompts=prompts,
            completions=completions,
            answer=answer,
            rewards_cfg=rewards_cfg,
            **kwargs,
        )

    return reward_fn
