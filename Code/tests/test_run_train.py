"""Tests for training script helpers."""

from pathlib import Path

from mathlm.utils import parse_config
from scripts.run_train import bootstrap_data


def test_bootstrap_data_creates_processed_file(tmp_path: Path):
    raw = tmp_path / "raw" / "gsm8k_train.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text('{"id": "1", "question": "Q?", "answer": "1"}')
    cfg_dict = {
        "model_name": "gemma",
        "data": {"split": "train", "source": "local"},
        "curriculum": {"split": "full", "max_problems": 1},
        "training": {"batch_size": 1, "total_steps": 1},
        "reward_weights": {},
    }
    config = parse_config(cfg_dict)

    processed = bootstrap_data(config, tmp_path)
    assert processed.exists()
    assert processed.read_text().strip() != ""
