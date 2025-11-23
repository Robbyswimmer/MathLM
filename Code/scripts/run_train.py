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

    print("="*60, flush=True)
    print("MathLM PPO Training", flush=True)
    print("="*60, flush=True)
    print(f"Config: {args.config}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print("="*60, flush=True)

    print("\nLoading configuration...", flush=True)
    config = load_experiment(args.config)

    print("Bootstrapping data...", flush=True)
    processed_path = bootstrap_data(config, args.data_dir)
    print(f"✓ Data prepared: {processed_path}", flush=True)

    reward_weights = RewardWeights(**config.reward_weights)
    reward_calc = RewardCalculator(reward_weights)

    run_id = args.run_id or Path(args.config).stem
    run_dir = args.output_dir / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.__dict__, default=lambda o: o.__dict__, indent=2))
    (run_dir / "dataset.txt").write_text(str(processed_path))

    print(f"\nLoading dataset with prompts...", flush=True)
    dataset = PromptDataset(
        processed_path,
        shots=config.prompting.shots,
        prompt_type=config.prompting.template,
    )
    print(f"✓ Dataset loaded: {len(dataset)} examples", flush=True)
    print(f"  Prompting: {config.prompting.shots}-shot, template={config.prompting.template}", flush=True)

    metrics_path = args.output_dir / "logs" / run_id / "metrics.jsonl"
    logger = JSONLLogger(metrics_path)
    checkpoint_dir = args.output_dir / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = args.output_dir / "traces" / run_id
    traces_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRun ID: {run_id}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    print(f"Checkpoints: {checkpoint_dir}", flush=True)
    print(f"Example traces: {traces_dir}", flush=True)

    runner_kwargs = {
        "minibatch_size": config.training.batch_size,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_interval": config.training.checkpoint_interval,
        "log_examples_interval": 100,
        "traces_dir": traces_dir,
    }

    if PPOTrainer is None:
        print("\n⚠ TRL not installed; running stub trainer for logging only.", flush=True)
        runner = MathLMPPORunner(dataset, reward_calc, logger, **runner_kwargs)
        runner.run(total_steps=config.training.total_steps)
        return

    print(f"\nLoading model: {config.training.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.training.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.training.model_name)
    print("✓ Model loaded", flush=True)

    print("\nInitializing PPO trainer...", flush=True)
    print(f"  Batch size: {config.training.batch_size}", flush=True)
    print(f"  Learning rate: {config.training.learning_rate}", flush=True)
    print(f"  KL target: {config.training.kl_target}", flush=True)
    print(f"  Total steps: {config.training.total_steps}", flush=True)

    # PPOConfig with minimal parameters that should work across TRL versions
    try:
        ppo_config = PPOConfig(
            learning_rate=config.training.learning_rate,
        )
    except TypeError as e:
        # If even minimal config fails, try completely empty
        print(f"Warning: Minimal config failed with {e}, using defaults", flush=True)
        ppo_config = PPOConfig()

    # PPOTrainer initialization - try different signatures
    try:
        # Try newer API: PPOTrainer(config, model, ref_model, tokenizer)
        trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
        )
    except TypeError:
        try:
            # Try older API: PPOTrainer(config, model, tokenizer)
            trainer = PPOTrainer(ppo_config, model, tokenizer)
        except TypeError:
            # Last resort: just model and tokenizer
            print("Warning: Using fallback PPOTrainer initialization", flush=True)
            trainer = PPOTrainer(model=model, tokenizer=tokenizer)
    print("✓ PPO trainer initialized", flush=True)

    runner = MathLMPPORunner(
        dataset,
        reward_calc,
        logger,
        trainer=trainer,
        tokenizer=tokenizer,
        **runner_kwargs,
    )

    print("\n" + "="*60, flush=True)
    print("STARTING PPO TRAINING", flush=True)
    print("="*60, flush=True)

    runner.run(total_steps=config.training.total_steps)

    print("\n" + "="*60, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()
