"""Curriculum helpers for slicing GSM8k into difficulty tiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .gsm8k import GSM8KExample


@dataclass
class CurriculumConfig:
    split: str = "full"
    max_problems: int | None = None


def _sort_key(example: GSM8KExample, curriculum_split: str) -> float:
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
    if split == "full":
        ordered = list(examples)
    else:
        ordered = sorted(examples, key=lambda ex: _sort_key(ex, split))
    if config.max_problems is not None:
        ordered = ordered[: config.max_problems]
    return ordered
