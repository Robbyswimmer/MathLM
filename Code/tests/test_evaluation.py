"""Tests for MathLM evaluation helpers."""

from pathlib import Path
from typing import List

from mathlm.data import GSM8KExample, save_examples
from mathlm.training import PromptDataset
from mathlm.evaluation import MathLMEvaluator


def make_dataset(tmp_path: Path) -> PromptDataset:
    raw = tmp_path / "eval.jsonl"
    save_examples(
        [
            GSM8KExample(uid="1", question="1+1?", answer="2"),
            GSM8KExample(uid="2", question="2+3?", answer="5"),
            GSM8KExample(uid="3", question="3+3?", answer="6"),
        ],
        raw,
    )
    return PromptDataset(raw, template="Solve: {question}\nAnswer:")


class FakeGenerator:
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.index = 0
        self.calls: List[List[str]] = []

    def generate(self, prompts: List[str]) -> List[str]:
        self.calls.append(list(prompts))
        batch = self.responses[self.index : self.index + len(prompts)]
        self.index += len(prompts)
        return batch


def test_evaluator_reports_accuracy(tmp_path: Path):
    dataset = make_dataset(tmp_path)
    generator = FakeGenerator([
        "The answer is 2",
        "My guess: 4",
        "Finally, 6",
    ])
    evaluator = MathLMEvaluator(generator)
    stats, predictions = evaluator.evaluate(dataset, batch_size=2, shuffle=False)
    assert stats.num_examples == 3
    assert stats.num_correct == 2
    assert abs(stats.accuracy - (2 / 3)) < 1e-9
    assert predictions[0].correct is True
    assert predictions[1].correct is False
    assert predictions[2].predicted_answer == "6"


def test_evaluator_respects_max_examples(tmp_path: Path):
    dataset = make_dataset(tmp_path)
    generator = FakeGenerator(["Ans 2", "Ans 5", "Ans 6"])
    evaluator = MathLMEvaluator(generator)
    stats, predictions = evaluator.evaluate(dataset, batch_size=4, max_examples=2)
    assert stats.num_examples == 2
    assert len(predictions) == 2
    assert generator.index == 2
