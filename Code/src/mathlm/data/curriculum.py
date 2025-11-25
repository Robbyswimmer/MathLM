"""Curriculum helpers for slicing GSM8k into difficulty tiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .gsm8k import GSM8KExample


@dataclass
class CurriculumConfig:
    split: str = "full"
    max_problems: int | None = None
    difficulty_range: tuple[int, int] | None = None  # (min_difficulty, max_difficulty)


def _sort_key(example: GSM8KExample, curriculum_split: str) -> float:
    # If difficulty is available, use it
    if example.difficulty is not None:
        try:
            return float(example.difficulty)
        except (ValueError, TypeError):
            pass

    # Fallback to heuristics
    if curriculum_split == "easy":
        return len(example.question)
    if curriculum_split == "medium":
        return len(example.question) + len(example.answer)
    if curriculum_split == "hard":
        return -len(example.question)
    return 0


def apply_curriculum(examples: Sequence[GSM8KExample], config: CurriculumConfig) -> List[GSM8KExample]:
    """Return a list of examples filtered according to the curriculum config."""

    split = config.split.lower()

    # Filter by difficulty range if specified
    filtered = list(examples)
    if config.difficulty_range is not None:
        min_diff, max_diff = config.difficulty_range
        filtered = [
            ex for ex in filtered
            if ex.difficulty is not None and min_diff <= int(ex.difficulty) <= max_diff
        ]

    # Sort by difficulty or curriculum split
    if split == "full":
        ordered = filtered
    else:
        ordered = sorted(filtered, key=lambda ex: _sort_key(ex, split))

    # Limit number of problems
    if config.max_problems is not None:
        ordered = ordered[: config.max_problems]

    return ordered
