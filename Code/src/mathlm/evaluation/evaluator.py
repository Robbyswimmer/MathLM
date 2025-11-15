"""Evaluation helpers for MathLM models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some environments
    torch = None

from mathlm.rewards import answers_match, extract_final_number
from mathlm.training.dataset import PromptDataset, PromptExample


@dataclass
class Prediction:
    """Represents a single evaluated sample."""

    uid: str
    prompt: str
    response: str
    predicted_answer: Optional[str]
    reference_answer: str
    correct: bool

    def to_json(self) -> dict:
        return {
            "uid": self.uid,
            "prompt": self.prompt,
            "response": self.response,
            "predicted_answer": self.predicted_answer,
            "reference_answer": self.reference_answer,
            "correct": self.correct,
        }


@dataclass
class EvaluationStats:
    num_examples: int
    num_correct: int

    @property
    def accuracy(self) -> float:
        if self.num_examples == 0:
            return 0.0
        return self.num_correct / self.num_examples

    def to_dict(self) -> dict:
        return {
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "accuracy": self.accuracy,
        }


class ResponseGenerator:
    """Interface for objects that can generate responses for prompts."""

    def generate(self, prompts: Sequence[str]) -> List[str]:  # pragma: no cover - interface only
        raise NotImplementedError


class HFModelGenerator(ResponseGenerator):
    """Hugging Face generation backend for evaluation."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_new_tokens: int = 128,
        device: Optional[str] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for evaluation.")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.model.to(self.device)
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompts: Sequence[str]) -> List[str]:
        if not prompts:
            return []
        encodings = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
        }
        if generation_kwargs["attention_mask"] is None:
            generation_kwargs.pop("attention_mask")
        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)
        new_tokens = outputs[:, input_ids.shape[-1] :]
        responses = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return responses


class MathLMEvaluator:
    """Runs evaluation over a dataset using a response generator."""

    def __init__(self, generator: ResponseGenerator):
        self.generator = generator

    def evaluate(
        self,
        dataset: PromptDataset,
        *,
        batch_size: int = 8,
        max_examples: Optional[int] = None,
        shuffle: bool = False,
    ) -> tuple[EvaluationStats, List[Prediction]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        total = 0
        correct = 0
        predictions: List[Prediction] = []
        for batch in dataset.iter_batches(batch_size, shuffle=shuffle, infinite=False):
            if max_examples is not None and total >= max_examples:
                break
            if max_examples is not None:
                remaining = max_examples - total
                if remaining <= 0:
                    break
                current_batch = batch[:remaining]
            else:
                current_batch = batch
            prompts = [example.prompt for example in current_batch]
            responses = self.generator.generate(prompts)
            for example, response in zip(current_batch, responses):
                prediction = self._evaluate_single(example, response)
                predictions.append(prediction)
                total += 1
                if prediction.correct:
                    correct += 1
            if max_examples is not None and total >= max_examples:
                break
        stats = EvaluationStats(num_examples=total, num_correct=correct)
        return stats, predictions

    def _evaluate_single(self, example: PromptExample, response: str) -> Prediction:
        answer = example.question.answer
        predicted_value = extract_final_number(response)
        is_correct = False
        if predicted_value is not None:
            is_correct = answers_match(predicted_value, answer)
        return Prediction(
            uid=example.question.uid,
            prompt=example.prompt,
            response=response,
            predicted_answer=predicted_value,
            reference_answer=answer,
            correct=is_correct,
        )
