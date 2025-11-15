"""Generate plots and tables for MathLM reports and presentations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots from experiment logs")
    parser.add_argument("--logs", type=Path, nargs="+", help="Paths to JSONL metric logs")
    parser.add_argument("--output", type=Path, default=Path("experiments/figures/training.png"))
    return parser.parse_args()


def load_metrics(path: Path):
    steps = []
    rewards = []
    with path.open() as fin:
        for line in fin:
            record = json.loads(line)
            steps.append(record.get("step", len(steps)))
            rewards.append(record.get("reward", 0))
    return steps, rewards


def main() -> None:
    args = parse_args()
    plt.figure(figsize=(8, 4))
    for log_path in args.logs:
        steps, rewards = load_metrics(log_path)
        plt.plot(steps, rewards, label=log_path.stem)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("MathLM Training Rewards")
    plt.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
