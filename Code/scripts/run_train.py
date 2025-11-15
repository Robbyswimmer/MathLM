"""Launch PPO training for MathLM using predefined configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mathlm.data import (
    CurriculumConfig,
    apply_curriculum,
    ensure_raw_split,
    load_raw_split,
    save_examples,
)
from mathlm.rewards import RewardCalculator, RewardWeights
from mathlm.training import JSONLLogger, MathLMPPORunner, PromptDataset
from mathlm.utils import ExperimentConfig, parse_config
from mathlm.utils.yaml_loader import load_config as load_yaml_config

try:
    from trl import PPOTrainer, PPOConfig  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
except Exception:  # pragma: no cover - TRL may be absent during scaffolding
    PPOTrainer = None  # type: ignore
    PPOConfig = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore


def load_experiment(config_path: Path) -> ExperimentConfig:
    raw_cfg = load_yaml_config(config_path)
    return parse_config(raw_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MathLM with PPO")
    parser.add_argument("--config", type=Path, default=Path("configs/curriculum_phase1.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def bootstrap_data(config: ExperimentConfig, data_dir: Path) -> Path:
    curriculum = CurriculumConfig(
        split=config.data.curriculum_split,
        max_problems=config.data.max_problems,
    )
    raw_path = ensure_raw_split(config.data.split, data_dir, source=config.data.source)
    examples = load_raw_split(raw_path)
    subset = apply_curriculum(examples, curriculum)
    processed_path = data_dir / "processed" / f"gsm8k_{config.data.split}_{curriculum.split}.jsonl"
    save_examples(subset, processed_path)
    return processed_path


def main() -> None:
    args = parse_args()
    config = load_experiment(args.config)
    processed_path = bootstrap_data(config, args.data_dir)

    reward_weights = RewardWeights(**config.reward_weights)
    reward_calc = RewardCalculator(reward_weights)

    run_id = args.run_id or Path(args.config).stem
    run_dir = args.output_dir / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.__dict__, default=lambda o: o.__dict__, indent=2))
    (run_dir / "dataset.txt").write_text(str(processed_path))

    dataset = PromptDataset(processed_path)
    metrics_path = args.output_dir / "logs" / run_id / "metrics.jsonl"
    logger = JSONLLogger(metrics_path)
    checkpoint_dir = args.output_dir / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    runner_kwargs = {
        "minibatch_size": config.training.batch_size,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_interval": config.training.checkpoint_interval,
    }

    if PPOTrainer is None:
        print("TRL not installed; running stub trainer for logging only.", flush=True)
        runner = MathLMPPORunner(dataset, reward_calc, logger, **runner_kwargs)
        runner.run(total_steps=config.training.total_steps)
        return

    tokenizer = AutoTokenizer.from_pretrained(config.training.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.training.model_name)
    ppo_config = PPOConfig(
        model_name=config.training.model_name,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        target_kl=config.training.kl_target,
    )
    trainer = PPOTrainer(ppo_config, model, tokenizer)
    runner = MathLMPPORunner(
        dataset,
        reward_calc,
        logger,
        trainer=trainer,
        tokenizer=tokenizer,
        **runner_kwargs,
    )
    runner.run(total_steps=config.training.total_steps)


if __name__ == "__main__":
    main()
