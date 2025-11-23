"""Dataset helpers for PPO training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from mathlm.data import GSM8KExample, load_raw_split
from mathlm.prompts import get_zero_shot_prompt, get_few_shot_prompt


@dataclass
class PromptExample:
    question: GSM8KExample
    prompt: str


class PromptDataset:
    def __init__(
        self,
        path: Path,
        template: str | None = None,
        shots: int = 0,
        prompt_type: str = "default",
    ):
        self.path = path
        self.shots = shots
        self.prompt_type = prompt_type
        self.template = template
        self.examples: List[PromptExample] = []
        self._load()

    def _load(self) -> None:
        examples = load_raw_split(self.path)
        for ex in examples:
            if self.template:
                # Use custom template if provided
                prompt = self.template.format(question=ex.question, problem=ex.question)
            elif self.shots == 0:
                # Zero-shot prompting
                prompt = get_zero_shot_prompt(template=self.prompt_type, problem=ex.question)
            else:
                # Few-shot prompting (1-3 shots)
                prompt = get_few_shot_prompt(shot_count=self.shots, type=self.prompt_type, problem=ex.question)
            self.examples.append(PromptExample(question=ex, prompt=prompt))

    def __iter__(self) -> Iterator[PromptExample]:
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        infinite: bool = True,
    ) -> Iterator[List[PromptExample]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.examples:
            raise ValueError("PromptDataset is empty; cannot create batches")
        while True:
            indices = list(range(len(self.examples)))
            if shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                yield [self.examples[idx] for idx in batch_indices]
            if not infinite:
                break
