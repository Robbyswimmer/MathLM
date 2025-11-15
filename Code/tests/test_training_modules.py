"""Tests for training dataset/logger/stub."""

from pathlib import Path

from mathlm.training import JSONLLogger, MathLMPPORunner, PromptDataset
from mathlm.rewards import RewardCalculator, RewardWeights
from mathlm.data import GSM8KExample, save_examples


def make_dataset(tmp_path: Path) -> Path:
    raw = tmp_path / "data.jsonl"
    save_examples(
        [
            GSM8KExample(uid="1", question="1+1?", answer="2"),
            GSM8KExample(uid="2", question="2+2?", answer="4"),
        ],
        raw,
    )
    return raw


def test_prompt_dataset_formats_questions(tmp_path: Path):
    raw = make_dataset(tmp_path)
    dataset = PromptDataset(raw, template="Question: {question}\nAnswer:")
    prompts = [ex.prompt for ex in dataset]
    assert prompts[0].startswith("Question: 1+1?")


def test_jsonl_logger_appends_records(tmp_path: Path):
    log_path = tmp_path / "log.jsonl"
    logger = JSONLLogger(log_path)
    logger.log({"step": 0, "reward": 1})
    logger.log({"step": 1, "reward": 2})
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_runner_logs_rewards_without_trl(tmp_path: Path):
    raw = make_dataset(tmp_path)
    dataset = PromptDataset(raw)
    reward_calc = RewardCalculator(RewardWeights())
    log_path = tmp_path / "metrics.jsonl"
    runner = MathLMPPORunner(dataset, reward_calc, JSONLLogger(log_path))
    runner.run(total_steps=1)
    assert log_path.exists()
    assert len(log_path.read_text().strip().splitlines()) == 1
