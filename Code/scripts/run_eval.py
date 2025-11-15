"""Evaluate trained MathLM checkpoints against GSM8k baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mathlm.utils.yaml_loader import load_config as load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MathLM checkpoint")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_zero_shot.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("experiments/logs/eval.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    report = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "metric": {
            "accuracy": 0.0,
            "notes": "Evaluation implementation pending.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
