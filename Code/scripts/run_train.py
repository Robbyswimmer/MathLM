"""Launch PPO training for MathLM using predefined configs."""

from __future__ import annotations

import argparse
import inspect
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
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig  # type: ignore
    from datasets import Dataset  # type: ignore
except Exception:  # pragma: no cover - TRL may be absent during scaffolding
    PPOTrainer = None  # type: ignore
    PPOConfig = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoModelForCausalLMWithValueHead = None  # type: ignore
    GenerationConfig = None  # type: ignore
    Dataset = None  # type: ignore


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


def _ensure_generation_config(model, tokenizer) -> None:
    """Attach a GenerationConfig so TRL's PPOTrainer can set stop/pad tokens."""
    if model is None or GenerationConfig is None:  # pragma: no cover - defensive
        return
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        gen_cfg = GenerationConfig.from_model_config(model.config)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        gen_cfg.pad_token_id = tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        gen_cfg.eos_token_id = tokenizer.eos_token_id
    model.generation_config = gen_cfg


def _ensure_base_prefix(module) -> None:
    """Some TRL versions expect base_model_prefix to exist."""
    if module is None:  # pragma: no cover - defensive
        return
    if not hasattr(module, "base_model_prefix"):
        setattr(module, "base_model_prefix", "model")


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

    print(f"\nLoading model with value head: {config.training.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.training.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.training.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.training.model_name)
    print("✓ Model and reference model loaded", flush=True)

    # Create a simple reward model (unused for PPO but kept for compatibility)
    reward_model = AutoModelForCausalLM.from_pretrained(config.training.model_name)
    print("✓ Reward model loaded", flush=True)

    # Value model handle (used by some PPOTrainer signatures)
    value_model = model

    # Ensure TRL can find base model prefix
    _ensure_base_prefix(model)
    _ensure_base_prefix(ref_model)
    _ensure_base_prefix(value_model)

    # Ensure generation configs exist so PPOTrainer can set stop/pad tokens
    _ensure_generation_config(model, tokenizer)
    _ensure_generation_config(ref_model, tokenizer)

    # Convert dataset to HuggingFace Dataset format
    print("\nPreparing HuggingFace Dataset...", flush=True)
    dataset_dict = {
        "query": [ex.prompt for ex in dataset.examples],
        "input_ids": [tokenizer.encode(ex.prompt, truncation=True, max_length=512) for ex in dataset.examples],
    }
    hf_dataset = Dataset.from_dict(dataset_dict)
    print(f"✓ Dataset prepared: {len(hf_dataset)} examples", flush=True)

    print("\nInitializing PPO configuration...", flush=True)
    ppo_config = PPOConfig()
    # Populate common knobs; setattr guards against version differences.
    for key, value in [
        ("learning_rate", config.training.learning_rate),
        ("batch_size", config.training.batch_size),
        ("mini_batch_size", max(1, config.training.batch_size // 2)),
        ("target_kl", config.training.kl_target),
        ("init_kl_coef", config.training.kl_target),
        ("kl_penalty", "kl"),
        ("model_name", config.training.model_name),
    ]:
        try:
            setattr(ppo_config, key, value)
        except Exception:
            pass

    print("\nInitializing PPO trainer...", flush=True)
    trainer_kwargs = {}
    trainer_sig = inspect.signature(PPOTrainer.__init__)
    param_names = set(trainer_sig.parameters.keys())

    # Core required args
    if "config" in param_names:
        trainer_kwargs["config"] = ppo_config
    elif "args" in param_names:
        trainer_kwargs["args"] = ppo_config

    trainer_kwargs["model"] = model
    if "ref_model" in param_names:
        trainer_kwargs["ref_model"] = ref_model
    if "tokenizer" in param_names:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in param_names:
        trainer_kwargs["processing_class"] = tokenizer
    if "dataset" in param_names:
        trainer_kwargs["dataset"] = hf_dataset
    elif "train_dataset" in param_names:
        trainer_kwargs["train_dataset"] = hf_dataset
    if "reward_model" in param_names:
        trainer_kwargs["reward_model"] = reward_model
    if "value_model" in param_names and value_model is not None:
        trainer_kwargs["value_model"] = value_model

    try:
        trainer = PPOTrainer(**trainer_kwargs)
    except TypeError as err:
        # Fallbacks for older TRL: drop optional entries until it works
        for key in ["tokenizer", "processing_class", "value_model"]:
            if key in trainer_kwargs:
                trainer_kwargs.pop(key)
        trainer = PPOTrainer(**trainer_kwargs)
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
