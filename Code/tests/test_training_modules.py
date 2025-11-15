"""Tests for training dataset/logger/stub."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from mathlm.training import JSONLLogger, MathLMPPORunner, PromptDataset
from mathlm.rewards import RewardCalculator, RewardWeights
from mathlm.data import GSM8KExample, save_examples


def make_dataset(tmp_path: Path, count: int = 2) -> Path:
    raw = tmp_path / "data.jsonl"
    examples = []
    for idx in range(1, count + 1):
        question = f"{idx}+{idx}?"
        answer = str(idx + idx)
        examples.append(GSM8KExample(uid=str(idx), question=question, answer=answer))
    save_examples(examples, raw)
    return raw


def test_prompt_dataset_formats_questions(tmp_path: Path):
    raw = make_dataset(tmp_path)
    dataset = PromptDataset(raw, template="Question: {question}\nAnswer:")
    prompts = [ex.prompt for ex in dataset]
    assert prompts[0].startswith("Question: 1+1?")


def test_prompt_dataset_iter_batches_shuffle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    raw = make_dataset(tmp_path, count=3)
    dataset = PromptDataset(raw)

    def reverse(seq: List[int]) -> None:
        seq.reverse()

    monkeypatch.setattr("mathlm.training.dataset.random.shuffle", reverse)
    batches = list(dataset.iter_batches(batch_size=2, shuffle=True, infinite=False))
    order = [example.question.uid for batch in batches for example in batch]
    assert order == ["3", "2", "1"]


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
    records = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert "reward_total" in record
    assert "reward_syntax" in record


class FakeTensorRow:
    def __init__(self, data: List[int]):
        self.data = data

    def __iter__(self):  # pragma: no cover - not needed but mirrors tensor API
        return iter(self.data)


class FakeTensor:
    def __init__(self, data: List[List[int]]):
        self.data = data
        self.device = None

    def to(self, device: str) -> "FakeTensor":
        self.device = device
        return self

    @property
    def shape(self) -> tuple[int, int]:
        if not self.data:
            return (0, 0)
        return (len(self.data), len(self.data[0]))

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        for row in self.data:
            yield FakeTensorRow(row)

    def __getitem__(self, key: Any) -> "FakeTensor":
        if not isinstance(key, tuple):  # pragma: no cover - sanity fallback
            raise TypeError("FakeTensor only supports tuple slicing")
        row_idx, col_idx = key
        rows = self._select_rows(row_idx)
        new_rows = []
        for row in rows:
            col_range = self._select_cols(col_idx, len(row))
            new_rows.append([row[i] for i in col_range])
        return FakeTensor(new_rows)

    def _select_rows(self, row_idx: slice) -> List[List[int]]:
        if isinstance(row_idx, slice):
            return self.data[row_idx]
        raise TypeError("Unsupported row index")

    def _select_cols(self, col_idx: slice, width: int) -> List[int]:
        if isinstance(col_idx, slice):
            return list(range(*col_idx.indices(width)))
        raise TypeError("Unsupported column index")


class FakeTokenizer:
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.saved_paths: List[Path] = []

    def __call__(self, prompts, return_tensors="pt", padding=True):  # pragma: no cover - simple stub
        batch = [[idx, idx + 1, idx + 2] for idx, _ in enumerate(prompts)]
        attention_mask = [[1] * len(row) for row in batch]
        return {"input_ids": FakeTensor(batch), "attention_mask": FakeTensor(attention_mask)}

    def batch_decode(self, generations: FakeTensor, skip_special_tokens: bool = True) -> List[str]:
        return self.responses[: len(generations)]

    def save_pretrained(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}")
        self.saved_paths.append(path)


class FakeModel:
    def __init__(self):
        self.saved_paths: List[Path] = []

    def save_pretrained(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "pytorch_model.bin").write_text("stub")
        self.saved_paths.append(path)


class FakeTrainer:
    def __init__(self):
        self.accelerator = SimpleNamespace(device="cuda:0")
        self.model = FakeModel()
        self.step_calls: List[dict] = []

    def generate(self, input_ids: FakeTensor, attention_mask=None, max_new_tokens: int = 0) -> FakeTensor:
        generated = [row + [9, 9] for row in input_ids.data]
        return FakeTensor(generated)

    def step(self, queries, responses, rewards):
        self.step_calls.append({"queries": queries, "responses": responses, "rewards": rewards})
        return {"ppo/kl": 0.5, "policy/entropy": 0.25, "train/loss": 1.0}


class FakeTorchTensor:
    def __init__(self, values, device=None, dtype=None):
        self.values = values
        self.device = device
        self.dtype = dtype


class FakeTorchModule:
    float32 = "float32"

    def tensor(self, values, **kwargs):  # pragma: no cover - trivial wrapper
        return FakeTorchTensor(values, kwargs.get("device"), kwargs.get("dtype"))


def test_runner_trl_path_logs_breakdowns_and_stats(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    raw = make_dataset(tmp_path, count=2)
    dataset = PromptDataset(raw)
    reward_calc = RewardCalculator(RewardWeights())
    log_path = tmp_path / "metrics.jsonl"
    fake_trainer = FakeTrainer()
    fake_tokenizer = FakeTokenizer(["stub answer", "stub answer 2"])
    monkeypatch.setattr("mathlm.training.ppo_loop.torch", FakeTorchModule())
    checkpoint_dir = tmp_path / "ckpts"
    runner = MathLMPPORunner(
        dataset,
        reward_calc,
        JSONLLogger(log_path),
        trainer=fake_trainer,
        tokenizer=fake_tokenizer,
        minibatch_size=2,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=1,
    )
    runner.run(total_steps=2)

    assert fake_trainer.step_calls, "trainer.step should be invoked"
    reward_tensor = fake_trainer.step_calls[0]["rewards"]
    assert isinstance(reward_tensor, FakeTorchTensor)
    assert reward_tensor.device == fake_trainer.accelerator.device

    logs = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    assert len(logs) == 2
    assert "reward_total" in logs[0]
    assert "trainer/ppo/kl" in logs[0]
    assert "trainer/kl" in logs[0]
    ckpt_step1 = checkpoint_dir / "step_000001" / "model"
    ckpt_step2 = checkpoint_dir / "step_000002" / "tokenizer"
    assert ckpt_step1.exists()
    assert ckpt_step2.exists()
