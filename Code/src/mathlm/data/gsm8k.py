"""Utilities for loading and caching GSM8k data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - datasets not always available during tests
    load_dataset = None  # type: ignore


@dataclass
class GSM8KExample:
    """Simple container for a GSM8k problem."""

    uid: str
    question: str
    answer: str
    difficulty: Optional[str] = None

    def to_json(self) -> dict:
        return {
            "id": self.uid,
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
        }


def _default_raw_path(data_dir: Path, split: str) -> Path:
    return data_dir / "raw" / f"gsm8k_{split}.jsonl"


def ensure_raw_split(split: str, data_dir: Path, source: str = "huggingface") -> Path:
    """Ensure a raw JSONL file for the given split exists and return its path."""

    split = split.lower()
    raw_path = _default_raw_path(data_dir, split)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        return raw_path

    if source != "huggingface":
        raise FileNotFoundError(
            f"Missing {raw_path}. Provide a local file or use source='huggingface'."
        )

    if load_dataset is None:
        raise RuntimeError(
            "datasets library not available. Install `datasets` or provide a local JSONL."
        )

    try:
        dataset = load_dataset("gsm8k", "main", split=split)
    except Exception as err:  # pragma: no cover - depends on network
        raise RuntimeError(
            "Failed to download GSM8k split via Hugging Face; ensure network access or "
            "place a JSONL file under data/raw/."
        ) from err

    with raw_path.open("w", encoding="utf-8") as fout:
        for entry in dataset:
            record = {
                "id": entry.get("id") or entry.get("question")[:16],
                "question": entry["question"],
                "answer": entry["answer"],
            }
            fout.write(json.dumps(record) + "\n")

    return raw_path


def load_raw_split(path: Path) -> List[GSM8KExample]:
    """Load GSM8k examples from a JSONL file."""

    examples: List[GSM8KExample] = []
    with path.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            uid = payload.get("id") or f"{path.stem}-{idx:04d}"
            question = payload["question"].strip()
            answer = payload["answer"].strip()
            difficulty = payload.get("difficulty")
            examples.append(GSM8KExample(uid=uid, question=question, answer=answer, difficulty=difficulty))
    return examples


def save_examples(examples: Iterable[GSM8KExample], path: Path) -> None:
    """Write processed examples to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for ex in examples:
            fout.write(json.dumps(ex.to_json()) + "\n")
