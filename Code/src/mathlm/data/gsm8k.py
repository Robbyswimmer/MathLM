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


def ensure_raw_split(split: str, data_dir: Path, source: str = "huggingface", raw_file: Path | None = None, parquet_file: Path | None = None) -> Path:
    """Ensure a raw JSONL file for the given split exists and return its path."""

    if raw_file is not None and Path(raw_file).exists():
        return Path(raw_file)

    split = split.lower()
    raw_path = _default_raw_path(data_dir, split)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If raw_path exists, check if it's valid (e.g. > 10 lines). If not, regenerate if parquet is provided.
    # For now, just trust existence unless user deletes it, OR if parquet_file is provided, we can force regen if needed.
    # But to be safe, let's just use it if it exists.
    if raw_path.exists():
        # Simple check: if it's tiny (< 1KB) and we have a parquet file, maybe regenerate?
        # Let's rely on the user deleting the bad file or us overwriting it if source='parquet'
        if source != "parquet":
            return raw_path

    if parquet_file is not None:
        parquet_path = Path(parquet_file)
        if parquet_path.exists():
            print(f"Converting Parquet {parquet_path} to JSONL {raw_path}...", flush=True)
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_path)
                with raw_path.open("w", encoding="utf-8") as fout:
                    for _, row in df.iterrows():
                        record = {
                            "id": str(row.get("id", "")) or row.get("question")[:16],
                            "question": row["question"],
                            "answer": row["answer"],
                        }
                        fout.write(json.dumps(record) + "\n")
                return raw_path
            except ImportError:
                # Fallback to datasets if pandas is missing
                if load_dataset is not None:
                    ds = load_dataset("parquet", data_files=str(parquet_path), split="train")
                    with raw_path.open("w", encoding="utf-8") as fout:
                        for entry in ds:
                            record = {
                                "id": entry.get("id") or entry.get("question")[:16],
                                "question": entry["question"],
                                "answer": entry["answer"],
                            }
                            fout.write(json.dumps(record) + "\n")
                    return raw_path
                else:
                    raise RuntimeError("Pandas or datasets library required to convert Parquet.")

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
