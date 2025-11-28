"""GRPO-compatible reward functions for MathLM."""

from __future__ import annotations

from collections import Counter
from typing import List
import re

from .reward_model import (
    extract_final_number,
    answers_match,
    has_reasoning_text,
    is_repetitive,
    is_too_long,
)
from .sandbox import PythonSandbox, extract_python_blocks


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


# ============================================================================
# PROCESS-BASED REWARDS WITH CODE VERIFICATION (RL Enhancement)
# ============================================================================

def count_reasoning_steps(completion: str) -> int:
    """
    Count explicit reasoning steps in the completion.

    Looks for step markers like:
    - "Step 1:", "Step 2:", etc.
    - Numbered lists: "1.", "2.", etc.
    - Sequential words: "First", "Then", "Next", "Finally"

    Returns number of detected reasoning steps (process supervision signal).
    """
    step_patterns = [
        r'step\s+\d+:',  # "Step 1:", "Step 2:"
        r'^\d+\.',       # "1.", "2." at start of line
        r'\b(first|second|third|then|next|finally|lastly)\b',
    ]

    completion_lower = completion.lower()
    total_steps = 0

    # Count step markers
    for pattern in step_patterns:
        matches = re.findall(pattern, completion_lower, re.MULTILINE)
        total_steps += len(matches)

    return min(total_steps, 10)  # Cap at 10 to avoid reward gaming


def has_arithmetic_operations(completion: str) -> bool:
    """Check if completion shows arithmetic work (intermediate calculations)."""
    # Look for explicit calculations like "5 * 3 = 15" or "total = 20 + 5"
    arithmetic_patterns = [
        r'\d+\s*[\+\-\*\/]\s*\d+\s*=',  # "5 + 3 ="
        r'=\s*\d+\s*[\+\-\*\/]\s*\d+',  # "= 5 + 3"
    ]

    for pattern in arithmetic_patterns:
        if re.search(pattern, completion):
            return True

    return False


def extract_code_output(sandbox_result) -> str | None:
    """Extract numerical result from sandbox execution output."""
    if not sandbox_result.success or not sandbox_result.stdout:
        return None

    # Get last line of output (typically the print statement result)
    output = sandbox_result.stdout.strip()
    if not output:
        return None

    lines = output.split('\n')
    last_line = lines[-1].strip()

    # Extract number from output
    number_match = re.search(r'[-+]?[0-9]*\.?[0-9]+', last_line)
    if number_match:
        return number_match.group(0)

    return None


def process_reward_with_code_verification(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    num_generations: int = 8,
    **kwargs
) -> List[float]:
    """
    Process-based reward function with code execution verification.

    RL Components:
    - Outcome reward: +10 for correct answer (terminal reward)
    - Process rewards: Dense rewards for intermediate reasoning steps
    - Code verification: Additional reward for executable code that verifies answer
    - Self-consistency: Bonus for matching majority vote

    Reward Structure:
        R_total = R_outcome + R_process + R_code + R_consistency

    where:
        R_outcome: Terminal reward for correct/incorrect answer
        R_process: Dense reward for showing reasoning steps
        R_code: Verification reward for executable code
        R_consistency: Self-consistency voting bonus

    Args:
        prompts: List of prompts (length = batch_size * num_generations)
        completions: List of completions (same length)
        answers: List of ground truth answers (same length)
        num_generations: Number of completions per prompt

    Returns:
        List of reward values (one per completion)
    """
    rewards = []
    sandbox = PythonSandbox(timeout=2.0)

    # First pass: Compute base rewards with process supervision
    for prompt, completion, answer in zip(prompts, completions, answers):
        reward = 0.0

        # ===== OUTCOME REWARD (Terminal) =====
        extracted = extract_final_number(completion)

        if extracted is not None:
            if answers_match(extracted, answer):
                reward += 10.0  # Strong terminal reward
            else:
                reward -= 2.0   # Penalty for wrong answer
        else:
            reward -= 1.0  # Penalty for no answer

        # ===== PROCESS REWARDS (Dense, intermediate supervision) =====

        # Reward for explicit reasoning steps (process supervision)
        num_steps = count_reasoning_steps(completion)
        if num_steps > 0:
            reward += 0.3 * num_steps  # +0.3 per reasoning step (max +3.0)

        # Reward for showing arithmetic work
        if has_arithmetic_operations(completion):
            reward += 0.5  # Bonus for showing calculations

        # Basic reasoning bonus
        if has_reasoning_text(completion):
            reward += 0.5

        # ===== CODE VERIFICATION REWARDS =====

        python_blocks = extract_python_blocks(completion)
        code_executed = False
        code_correct = False

        if python_blocks:
            for block in python_blocks:
                # Try to execute the code
                result = sandbox.run(block)

                if result.success:
                    code_executed = True
                    reward += 2.0  # Bonus for executable code

                    # Check if code output matches ground truth
                    code_output = extract_code_output(result)
                    if code_output is not None and answers_match(code_output, answer):
                        code_correct = True
                        reward += 3.0  # Strong bonus for code verification

                    break  # Only reward first successful execution

        # Small penalty for code that doesn't execute (encourages fixing errors)
        if python_blocks and not code_executed:
            reward -= 1.0

        # ===== PENALTIES =====

        # Repetition penalty
        if is_repetitive(completion, prompt, threshold=0.5):
            reward -= 3.0

        rewards.append(reward)

    # Second pass: Add self-consistency bonus (same as existing implementation)
    batch_size = len(completions) // num_generations

    for batch_idx in range(batch_size):
        start_idx = batch_idx * num_generations
        end_idx = start_idx + num_generations

        batch_completions = completions[start_idx:end_idx]
        batch_answers = [extract_final_number(c) for c in batch_completions]
        ground_truth = answers[start_idx]

        valid_answers = [a for a in batch_answers if a is not None]
        if not valid_answers:
            continue

        answer_counts = Counter(valid_answers)
        majority_answer, majority_count = answer_counts.most_common(1)[0]

        # Self-consistency bonus (requires >= 50% agreement)
        if majority_count >= num_generations / 2:
            for i in range(start_idx, end_idx):
                extracted = batch_answers[i - start_idx]

                if extracted == majority_answer:
                    if answers_match(majority_answer, ground_truth):
                        rewards[i] += 2.0  # Consensus + correct
                    else:
                        rewards[i] += 0.5  # Consensus bonus

    return rewards


# Combined version with minimal format quality
def combined_process_reward(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    **kwargs
) -> List[float]:
    """
    Combined process-based reward with code verification.

    Use this as the main reward function for RL training with:
    - Dense process rewards
    - Code execution verification
    - Self-consistency voting
    """
    process_rewards = process_reward_with_code_verification(
        prompts, completions, answers, **kwargs
    )
    format_quality = format_quality_reward(prompts, completions, **kwargs)

    return [p + 0.1 * f for p, f in zip(process_rewards, format_quality)]
