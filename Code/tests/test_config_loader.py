"""Tests for config parsing utilities."""

from pathlib import Path

from mathlm.utils import parse_config
from mathlm.utils.yaml_loader import load_config


def test_load_config_supports_include(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("model_name: gemma\ntraining:\n  batch_size: 32\n")
    child = tmp_path / "child.yaml"
    child.write_text("include: base.yaml\ntraining:\n  total_steps: 10\n")
    cfg = load_config(child)
    assert cfg["training"]["batch_size"] == 32
    assert cfg["training"]["total_steps"] == 10


def test_parse_config_sets_defaults():
    cfg = {
        "model_name": "gemma",
        "data": {"split": "train", "source": "local"},
        "curriculum": {"split": "easy", "max_problems": 10},
        "training": {"batch_size": 8, "total_steps": 100},
        "reward_weights": {"syntax": 0.1},
    }
    parsed = parse_config(cfg)
    assert parsed.data.curriculum_split == "easy"
    assert parsed.training.batch_size == 8
