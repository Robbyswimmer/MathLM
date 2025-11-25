"""GRPO-compatible reward functions for MathLM."""

from __future__ import annotations

from typing import List

from .reward_model import (
    extract_final_number,
    answers_match,
    has_reasoning_text,
    is_repetitive,
    is_too_long,
)


def math_correctness_reward(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    **kwargs
) -> List[float]:
    """
    Primary reward function for math problem correctness.

    Args:
        prompts: List of math problem prompts
        completions: List of model-generated solutions
        answers: List of ground truth answers from dataset

    Returns:
        List of float rewards (one per completion)
    """
    rewards = []

    for prompt, completion, answer in zip(prompts, completions, answers):
        reward = 0.0

        # Extract final number from completion
        extracted = extract_final_number(completion)

        if extracted is not None:
            # Main reward: correctness (increased to dominate signal)
            if answers_match(extracted, answer):
                reward += 10.0  # Strong reward for correct answer
            else:
                reward -= 2.0  # Significant penalty for wrong answer
        else:
            # Penalty for no answer extracted
            reward -= 1.0

        # Bonus for showing reasoning (encourage step-by-step)
        if has_reasoning_text(completion):
            reward += 0.5

        # Penalties for problematic outputs
        if is_repetitive(completion, prompt, threshold=0.5):
            reward -= 3.0  # Strong penalty for repetition (tightened threshold)

        # Removed length penalty - GSM8K needs detailed solutions

        rewards.append(reward)

    return rewards


def format_quality_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    Secondary reward for output format quality.

    Minimal format bonuses to avoid reward gaming.
    """
    rewards = []

    for completion in completions:
        reward = 0.0

        # Reduced format bonuses - these were too easily gamed
        # Only reward substantive structure, not superficial markers
        word_count = len(completion.split())

        # Bonus for math operations (actual computation indicator)
        math_symbols = ["=", "+", "-", "*", "/", "×", "÷"]
        if any(symbol in completion for symbol in math_symbols):
            reward += 0.05  # Reduced from 0.1

        rewards.append(reward)

    return rewards


# Combined reward function (primary + format quality)
def combined_math_reward(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    **kwargs
) -> List[float]:
    """
    Combined reward: correctness + format quality.

    This is the main reward function to use with GRPOTrainer.
    """
    correctness = math_correctness_reward(prompts, completions, answers, **kwargs)
    format_quality = format_quality_reward(prompts, completions, **kwargs)

    # Weight correctness very heavily, minimal format bonus to avoid gaming
    return [c + 0.1 * f for c, f in zip(correctness, format_quality)]
