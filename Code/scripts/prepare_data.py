"""Prepare and cache GSM8k data for MathLM experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml optional
    yaml = None  # type: ignore

from mathlm.data import (
    CurriculumConfig,
    apply_curriculum,
    ensure_raw_split,
    load_raw_split,
    save_examples,
)


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", ""}:
        return None
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        if value.startswith("0") and len(value) > 1 and value[1].isdigit():
            raise ValueError
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if (value.startswith("\"") and value.endswith("\"")) or (
                value.startswith("'") and value.endswith("'")
            ):
                return value[1:-1]
            return value


def _minimal_yaml_load(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_part, _, value_part = line.lstrip().partition(":")
        key = key_part.strip()
        value = value_part.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if not value:
            new_dict: Dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent, new_dict))
        else:
            current[key] = _parse_scalar(value)
            stack.append((indent, current))
    return root


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if yaml is not None:  # type: ignore
        data = yaml.safe_load(text) or {}
    else:
        data = _minimal_yaml_load(text)
    include = data.pop("include", None)
    if include:
        include_path = path.parent / include
        base = load_config(include_path)
        return _merge_dict(base, data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GSM8k data for MathLM")
    parser.add_argument("--config", type=Path, default=Path("configs/curriculum_phase1.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-name", type=str, default=None, help="Optional override for processed file name")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    split = data_cfg.get("split", "train")
    source = data_cfg.get("source", "huggingface")

    curriculum_cfg = config.get("curriculum", {})
    curriculum = CurriculumConfig(
        split=curriculum_cfg.get("split", "full"),
        max_problems=curriculum_cfg.get("max_problems"),
    )

    data_dir = args.data_dir
    raw_path = ensure_raw_split(split=split, data_dir=data_dir, source=source)
    examples = load_raw_split(raw_path)
    curated = apply_curriculum(examples, curriculum)

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"gsm8k_{split}_{curriculum.split}"
    output_path = processed_dir / f"{output_name}.jsonl"
    save_examples(curated, output_path)

    summary = {
        "config": str(args.config),
        "raw_path": str(raw_path),
        "processed_path": str(output_path),
        "split": split,
        "curriculum": curriculum.split,
        "max_problems": curriculum.max_problems,
        "num_examples": len(curated),
    }

    metadata_path = data_dir / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    metadata[output_name] = summary
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
