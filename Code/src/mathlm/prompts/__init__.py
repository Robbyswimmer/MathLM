"""Prompting utilities for MathLM training and evaluation."""

from .zero_shot import get_zero_shot_prompt, ZERO_SHOT_PROMPT
from .few_shot import get_few_shot_prompt, ONE_SHOT_PROMPT, TWO_SHOT_PROMPT, THREE_SHOT_PROMPT

__all__ = [
    "get_zero_shot_prompt",
    "get_few_shot_prompt",
    "ZERO_SHOT_PROMPT",
    "ONE_SHOT_PROMPT",
    "TWO_SHOT_PROMPT",
    "THREE_SHOT_PROMPT",
]
