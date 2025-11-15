"""PPO training runner abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from mathlm.rewards import RewardCalculator

from .dataset import PromptDataset, PromptExample
from .logger import JSONLLogger


@dataclass
class Rollout:
    prompt: str
    response_text: str
    reward: float


class MathLMPPORunner:
    """Runs PPO training with TRL if available, otherwise falls back to a stub."""

    def __init__(
        self,
        dataset: PromptDataset,
        reward_calc: RewardCalculator,
        logger: JSONLLogger,
        trainer: Optional[object] = None,
        tokenizer: Optional[object] = None,
        max_new_tokens: int = 128,
        minibatch_size: int = 4,
    ) -> None:
        self.dataset = dataset
        self.reward_calc = reward_calc
        self.logger = logger
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.minibatch_size = minibatch_size
        self.use_trl = self.trainer is not None and self.tokenizer is not None

    def run(self, total_steps: int = 1000) -> None:
        step = 0
        batch: List[PromptExample] = []
        for example in self.dataset:
            batch.append(example)
            if len(batch) >= self.minibatch_size:
                step = self._process_batch(batch, step, total_steps)
                if step >= total_steps:
                    break
                batch = []
        if batch and step < total_steps:
            self._process_batch(batch, step, total_steps)

    def _process_batch(self, batch: List[PromptExample], start_step: int, total_steps: int) -> int:
        if self.use_trl:
            rollouts = self._trl_batch(batch)
        else:
            rollouts = self._stub_batch(batch)
        step = start_step
        for rollout in rollouts:
            self.logger.log({"step": step, "reward": rollout.reward})
            step += 1
            if step >= total_steps:
                break
        return step

    def _stub_batch(self, batch: List[PromptExample]) -> List[Rollout]:
        rollouts: List[Rollout] = []
        for example in batch:
            response_text = f"Stub response for: {example.prompt[:40]}"
            breakdown = self.reward_calc.evaluate(example.question, response_text)
            rollouts.append(Rollout(prompt=example.prompt, response_text=response_text, reward=breakdown.total))
        return rollouts

    def _trl_batch(self, batch: List[PromptExample]) -> List[Rollout]:
        if torch is None:
            raise RuntimeError("PyTorch is required for PPO training with TRL.")
        assert self.trainer is not None and self.tokenizer is not None
        accelerator = getattr(self.trainer, "accelerator", None)
        device = getattr(accelerator, "device", "cpu")
        prompts = [example.prompt for example in batch]
        encodings = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        response_tensors = self.trainer.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
        )
        generations = response_tensors[:, input_ids.shape[-1] :]
        responses = self.tokenizer.batch_decode(generations, skip_special_tokens=True)
        rewards_tensor = []
        rollouts: List[Rollout] = []
        for example, response_text in zip(batch, responses):
            breakdown = self.reward_calc.evaluate(example.question, response_text)
            rollouts.append(Rollout(prompt=example.prompt, response_text=response_text, reward=breakdown.total))
            rewards_tensor.append(breakdown.total)
        queries = [ids for ids in input_ids]
        responses_tensor = [gen for gen in generations]
        self.trainer.step(queries, responses_tensor, rewards_tensor)
        return rollouts
