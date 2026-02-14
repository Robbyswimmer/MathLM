"""Tests for zero-shot and few-shot prompt templates."""

from mathlm.prompts.zero_shot.zero_shot_prompts import (
    EXPLICIT_ZERO_SHOT_PROMPT,
    ZERO_SHOT_PROMPT,
)

from mathlm.prompts.few_shot.few_shot_prompts import (
    CONCISE_ONE_SHOT_PROMPT,
    CONCISE_TWO_SHOT_PROMPT,
    CONCISE_THREE_SHOT_PROMPT,
    ONE_SHOT_PROMPT,
    TWO_SHOT_PROMPT,
    THREE_SHOT_PROMPT,
)

def test_zero_shot_prompt_templates():
    problem = "What is 2 + 2?"
    zero_shot = ZERO_SHOT_PROMPT.format(problem=problem)
    explicit_zero_shot = EXPLICIT_ZERO_SHOT_PROMPT.format(problem=problem)
    assert "Solve the following math problem step-by-step." in zero_shot
    assert "Now, solve the following problem:" in explicit_zero_shot
    assert problem in zero_shot
    assert problem in explicit_zero_shot

def test_one_shot_prompt_templates():
    problem = "What is 3 * 3?"
    one_shot = ONE_SHOT_PROMPT.format(problem=problem)
    concise_one_shot = CONCISE_ONE_SHOT_PROMPT.format(problem=problem)
    assert "Example:" in one_shot
    assert "Q: Jane has 3 apples and buys 5 more. How many apples does she have?" in concise_one_shot
    assert problem in one_shot
    assert problem in concise_one_shot

def test_two_shot_prompt_templates():
    problem = "What is 5 - 2?"
    two_shot = TWO_SHOT_PROMPT.format(problem=problem)
    concise_two_shot = CONCISE_TWO_SHOT_PROMPT.format(problem=problem)
    assert "Example 1:" in two_shot
    assert "Example 2:" in two_shot
    assert "Example 1:" in concise_two_shot
    assert "Example 2:" in concise_two_shot
    assert problem in two_shot
    assert problem in concise_two_shot

def test_three_shot_prompt_templates():
    problem = "What is 10 / 2?"
    three_shot = THREE_SHOT_PROMPT.format(problem=problem)
    concise_three_shot = CONCISE_THREE_SHOT_PROMPT.format(problem=problem)
    assert "Example 1:" in three_shot
    assert "Example 2:" in three_shot
    assert "Example 3:" in three_shot
    assert "Example 1:" in concise_three_shot
    assert "Example 2:" in concise_three_shot
    assert "Example 3:" in concise_three_shot
    assert problem in three_shot
    assert problem in concise_three_shot
