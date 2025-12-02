"""Configuration dataclasses and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainingConfig:
    model_name: str
    batch_size: int
    learning_rate: float
    kl_target: float
    entropy_coef: float
    total_steps: int
    checkpoint_interval: int
    # Additional GRPO parameters
    num_generations: int = 8
    temperature: float = 0.9
    beta: float = 0.04
    gradient_accumulation_steps: int = 4
    max_completion_length: int = 512
    max_prompt_length: int = 256
    bf16: bool = True
    # Reward function selection
    use_self_consistency: bool = True
    use_process_rewards: bool = False


@dataclass
class DataConfig:
    split: str
    source: str
    curriculum_split: str
    max_problems: int | None
    raw_file: str | None = None
    parquet_file: str | None = None
    difficulty_range: tuple[int, int] | None = None


@dataclass
class PromptingConfig:
    shots: int
    template: str


@dataclass
class ExperimentConfig:
    data: DataConfig
    training: TrainingConfig
    prompting: PromptingConfig
    reward_weights: Dict[str, float]


def parse_config(config: Dict[str, Any]) -> ExperimentConfig:
    data_cfg = config.get("data", {})
    curriculum = config.get("curriculum", {})
    training_cfg = config.get("training", {})
    prompting_cfg = config.get("prompting", {})
    reward_cfg = config.get("reward_weights", {})
    # Parse difficulty_range if present
    difficulty_range = curriculum.get("difficulty_range")
    if difficulty_range and isinstance(difficulty_range, list):
        difficulty_range = tuple(difficulty_range)

    data = DataConfig(
        split=data_cfg.get("split", "train"),
        source=data_cfg.get("source", "huggingface"),
        curriculum_split=curriculum.get("split", "full"),
        max_problems=curriculum.get("max_problems"),
        raw_file=data_cfg.get("raw_file"),
        parquet_file=data_cfg.get("parquet_file"),
        difficulty_range=difficulty_range,
    )
    training = TrainingConfig(
        model_name=training_cfg.get("model_name") or config.get("model_name", "models/gemma-2-2b-it"),
        batch_size=training_cfg.get("batch_size", 64),
        learning_rate=training_cfg.get("learning_rate", 1e-5),
        kl_target=training_cfg.get("kl_target", 0.08),
        entropy_coef=training_cfg.get("entropy_coef", 0.01),
        total_steps=training_cfg.get("total_steps", 250_000),
        checkpoint_interval=training_cfg.get("checkpoint_interval", 1_000),
        # GRPO parameters
        num_generations=training_cfg.get("num_generations", 8),
        temperature=training_cfg.get("temperature", 0.9),
        beta=training_cfg.get("beta", 0.04),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        max_completion_length=training_cfg.get("max_completion_length", 512),
        max_prompt_length=training_cfg.get("max_prompt_length", 256),
        bf16=training_cfg.get("bf16", True),
        # Reward function flags
        use_self_consistency=training_cfg.get("use_self_consistency", True),
        use_process_rewards=training_cfg.get("use_process_rewards", False),
    )
    prompting = PromptingConfig(
        shots=prompting_cfg.get("shots", 0),
        template=prompting_cfg.get("template", "default"),
    )
    return ExperimentConfig(data=data, training=training, prompting=prompting, reward_weights=reward_cfg)
