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


@dataclass
class DataConfig:
    split: str
    source: str
    curriculum_split: str
    max_problems: int | None
    raw_file: str | None = None
    parquet_file: str | None = None


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
    data = DataConfig(
        split=data_cfg.get("split", "train"),
        source=data_cfg.get("source", "huggingface"),
        curriculum_split=curriculum.get("split", "full"),
        max_problems=curriculum.get("max_problems"),
        raw_file=data_cfg.get("raw_file"),
        parquet_file=data_cfg.get("parquet_file"),
    )
    training = TrainingConfig(
        model_name=training_cfg.get("model_name") or config.get("model_name", "models/gemma-2-2b-it"),
        batch_size=training_cfg.get("batch_size", 64),
        learning_rate=training_cfg.get("learning_rate", 1e-5),
        kl_target=training_cfg.get("kl_target", 0.08),
        entropy_coef=training_cfg.get("entropy_coef", 0.01),
        total_steps=training_cfg.get("total_steps", 250_000),
        checkpoint_interval=training_cfg.get("checkpoint_interval", 1_000),
    )
    prompting = PromptingConfig(
        shots=prompting_cfg.get("shots", 0),
        template=prompting_cfg.get("template", "default"),
    )
    return ExperimentConfig(data=data, training=training, prompting=prompting, reward_weights=reward_cfg)
