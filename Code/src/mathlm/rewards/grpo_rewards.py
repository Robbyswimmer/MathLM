"""GRPO-compatible reward functions for MathLM."""

from __future__ import annotations

from collections import Counter
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


def self_consistency_reward(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    num_generations: int = 8,
    **kwargs
) -> List[float]:
    """
    Self-consistency reward with majority voting.

    GRPO generates N completions per prompt. We use majority voting:
    - Completions that match majority AND are correct get bonus
    - Encourages confident, consistent reasoning

    Args:
        prompts: List of prompts (length = batch_size * num_generations)
        completions: List of completions (length = batch_size * num_generations)
        answers: List of ground truth answers (length = batch_size * num_generations)
        num_generations: Number of completions per prompt (default: 8)
    """
    base_rewards = math_correctness_reward(prompts, completions, answers, **kwargs)

    # Group completions by prompt (GRPO generates num_generations per prompt)
    batch_size = len(completions) // num_generations

    for batch_idx in range(batch_size):
        start_idx = batch_idx * num_generations
        end_idx = start_idx + num_generations

        # Get all completions for this prompt
        batch_completions = completions[start_idx:end_idx]
        batch_answers = [extract_final_number(c) for c in batch_completions]
        ground_truth = answers[start_idx]  # Same for all generations

        # Find majority vote (most common answer)
        valid_answers = [a for a in batch_answers if a is not None]
        if not valid_answers:
            continue  # No valid answers, skip consistency bonus

        answer_counts = Counter(valid_answers)
        majority_answer, majority_count = answer_counts.most_common(1)[0]

        # Only apply bonus if there's strong consensus (>= 50% agreement)
        if majority_count >= num_generations / 2:
            for i in range(start_idx, end_idx):
                extracted = batch_answers[i - start_idx]

                # Bonus for matching majority
                if extracted == majority_answer:
                    # Extra bonus if majority is also correct
                    if answers_match(majority_answer, ground_truth):
                        base_rewards[i] += 2.0  # Consensus + correct bonus
                    else:
                        base_rewards[i] += 0.5  # Consensus bonus (even if wrong)

    return base_rewards


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


# Self-consistency version (recommended)
def combined_math_reward_with_consistency(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    **kwargs
) -> List[float]:
    """
    Combined reward with self-consistency voting.

    Use this as the main reward function for better performance.
    """
    consistency_rewards = self_consistency_reward(prompts, completions, answers, **kwargs)
    format_quality = format_quality_reward(prompts, completions, **kwargs)

    # Weight correctness very heavily, minimal format bonus to avoid gaming
    return [c + 0.1 * f for c, f in zip(consistency_rewards, format_quality)]
