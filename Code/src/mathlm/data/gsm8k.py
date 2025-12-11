"""Utilities for loading and caching GSM8k data."""

from __future__ import annotations

import json
import re
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


def estimate_difficulty(example: GSM8KExample) -> int:
    """
    Estimate problem difficulty based on heuristics.

    Returns a difficulty score (1-5):
    - 1: Simple (1-2 reasoning steps)
    - 2: Easy (3-4 reasoning steps)
    - 3: Medium (5-6 reasoning steps)
    - 4: Hard (7-8 reasoning steps)
    - 5: Very Hard (9+ reasoning steps)

    Heuristics:
    - Count reasoning steps in solution (sentences/lines)
    - Count arithmetic operations in solution
    - Question length and complexity indicators
    """
    answer = example.answer
    question = example.question

    # Count sentences in answer (better proxy for steps than << >>)
    # GSM8K format: "#### 18" at end, split by periods and newlines
    answer_sentences = [s.strip() for s in re.split(r'[.\n]', answer) if s.strip() and '####' not in s]
    step_count = len(answer_sentences)

    # Also count << >> step markers if present
    step_markers = answer.count("<<")
    if step_markers > 0:
        step_count = max(step_count, step_markers)

    # Simple heuristic: map step count to difficulty
    if step_count <= 2:
        return 1
    elif step_count <= 4:
        return 2
    elif step_count <= 6:
        return 3
    elif step_count <= 8:
        return 4
    else:
        return 5


def annotate_difficulty(examples: List[GSM8KExample], verbose: bool = True) -> List[GSM8KExample]:
    """Add difficulty scores to examples."""
    difficulty_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for ex in examples:
        if ex.difficulty is None:
            score = estimate_difficulty(ex)
            ex.difficulty = str(score)
        difficulty_counts[int(ex.difficulty)] += 1

    if verbose:
        print(f"  Difficulty distribution:", flush=True)
        for level in [1, 2, 3, 4, 5]:
            count = difficulty_counts[level]
            pct = 100 * count / len(examples)
            print(f"    Level {level}: {count:4d} ({pct:5.1f}%)", flush=True)

    return examples
