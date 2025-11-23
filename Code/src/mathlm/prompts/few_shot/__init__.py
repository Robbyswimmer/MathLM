"""Few-shot prompting templates for MathLM."""

from .few_shot_prompts import (
    ONE_SHOT_PROMPT,
    TWO_SHOT_PROMPT,
    THREE_SHOT_PROMPT,
    CONCISE_ONE_SHOT_PROMPT,
    CONCISE_TWO_SHOT_PROMPT,
    CONCISE_THREE_SHOT_PROMPT,
    get_few_shot_prompt,
)

__all__ = [
    "ONE_SHOT_PROMPT",
    "TWO_SHOT_PROMPT",
    "THREE_SHOT_PROMPT",
    "CONCISE_ONE_SHOT_PROMPT",
    "CONCISE_TWO_SHOT_PROMPT",
    "CONCISE_THREE_SHOT_PROMPT",
    "get_few_shot_prompt",
]
