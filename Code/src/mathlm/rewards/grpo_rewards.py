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
            # Bonus for extracting any answer
            reward += 0.5

            # Main reward: correctness
            if answers_match(extracted, answer):
                reward += 4.0  # Correct answer
            else:
                reward -= 0.5  # Penalty for wrong answer
        else:
            # No answer extracted at all
            reward -= 0.5

        # Bonus for showing reasoning (encourage step-by-step)
        if has_reasoning_text(completion):
            reward += 0.5

        # Penalties for problematic outputs
        if is_repetitive(completion, prompt, threshold=0.6):
            reward -= 2.0  # Strong penalty for repetition

        if is_too_long(completion, prompt, ratio_threshold=1.5):
            reward -= 0.5  # Penalty for excessive length

        rewards.append(reward)

    return rewards


def format_quality_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    Secondary reward for output format quality.

    Encourages well-structured, readable solutions.
    """
    rewards = []

    for completion in completions:
        reward = 0.0

        # Small bonus for reasonable length (not too short, not too long)
        word_count = len(completion.split())
        if 20 <= word_count <= 200:
            reward += 0.1

        # Bonus for containing step markers
        step_markers = ["step", "first", "then", "next", "finally", "therefore"]
        if any(marker in completion.lower() for marker in step_markers):
            reward += 0.1

        # Bonus for containing math operations
        math_symbols = ["=", "+", "-", "*", "/", "×", "÷"]
        if any(symbol in completion for symbol in math_symbols):
            reward += 0.1

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

    # Weight correctness heavily, format quality as bonus
    return [c + 0.3 * f for c, f in zip(correctness, format_quality)]
