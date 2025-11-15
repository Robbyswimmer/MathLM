"""Dataset helpers for PPO training."""

from __future__ import annotations

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
