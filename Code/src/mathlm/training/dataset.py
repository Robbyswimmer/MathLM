"""Dataset helpers for PPO training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from mathlm.data import GSM8KExample, load_raw_split


@dataclass
class PromptExample:
    question: GSM8KExample
    prompt: str


class PromptDataset:
    def __init__(self, path: Path, template: str | None = None):
        self.path = path
        self.template = template or "Solve the following math problem:\n{question}\nAnswer:"
        self.examples: List[PromptExample] = []
        self._load()

    def _load(self) -> None:
        examples = load_raw_split(self.path)
        for ex in examples:
            prompt = self.template.format(question=ex.question)
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
