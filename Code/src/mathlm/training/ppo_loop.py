"""PPO training runner abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from mathlm.rewards import RewardBreakdown, RewardCalculator

from .dataset import PromptDataset, PromptExample
from .logger import JSONLLogger


@dataclass
class Rollout:
    prompt: str
    response_text: str
    breakdown: RewardBreakdown

    @property
    def reward(self) -> float:
        return self.breakdown.total


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
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 100,
    ) -> None:
        self.dataset = dataset
        self.reward_calc = reward_calc
        self.logger = logger
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.minibatch_size = minibatch_size
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.use_trl = self.trainer is not None and self.tokenizer is not None

    def run(self, total_steps: int = 1000) -> None:
        if total_steps <= 0:
            return
        step = 0
        batch_iter = self.dataset.iter_batches(self.minibatch_size, shuffle=True, infinite=True)
        while step < total_steps:
            batch = next(batch_iter)
            step = self._process_batch(batch, step, total_steps)

    def _process_batch(self, batch: List[PromptExample], start_step: int, total_steps: int) -> int:
        if self.use_trl:
            rollouts, trainer_stats = self._trl_batch(batch)
        else:
            rollouts, trainer_stats = self._stub_batch(batch)
        trainer_payload = self._format_trainer_stats(trainer_stats)
        step = start_step
        for rollout in rollouts:
            record = self._build_log_record(step, rollout, trainer_payload)
            self.logger.log(record)
            step += 1
            self._maybe_checkpoint(step)
            if step >= total_steps:
                break
        return step

    def _stub_batch(self, batch: List[PromptExample]) -> Tuple[List[Rollout], Optional[Dict[str, Any]]]:
        rollouts: List[Rollout] = []
        for example in batch:
            response_text = f"Stub response for: {example.prompt[:40]}"
            breakdown = self.reward_calc.evaluate(example.question, response_text)
            rollouts.append(Rollout(prompt=example.prompt, response_text=response_text, breakdown=breakdown))
        return rollouts, None

    def _trl_batch(self, batch: List[PromptExample]) -> Tuple[List[Rollout], Optional[Dict[str, Any]]]:
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
        reward_values: List[float] = []
        rollouts: List[Rollout] = []
        for example, response_text in zip(batch, responses):
            breakdown = self.reward_calc.evaluate(example.question, response_text)
            rollouts.append(Rollout(prompt=example.prompt, response_text=response_text, breakdown=breakdown))
            reward_values.append(breakdown.total)
        queries = [ids for ids in input_ids]
        responses_tensor = [gen for gen in generations]
        dtype = getattr(torch, "float32", None)
        tensor_kwargs: Dict[str, Any] = {"device": device}
        if dtype is not None:
            tensor_kwargs["dtype"] = dtype
        rewards_tensor = torch.tensor(reward_values, **tensor_kwargs)
        stats = self.trainer.step(queries, responses_tensor, rewards_tensor)
        return rollouts, stats

    def _build_log_record(self, step: int, rollout: Rollout, trainer_stats: Dict[str, Any]) -> Dict[str, Any]:
        record: Dict[str, Any] = {"step": step}
        breakdown = rollout.breakdown
        breakdown_payload = {
            "reward_total": breakdown.total,
            "reward_syntax": breakdown.syntax_reward,
            "reward_execution": breakdown.execution_reward,
            "reward_extraction": breakdown.extraction_reward,
            "reward_reasoning": breakdown.reasoning_reward,
            "reward_exact": breakdown.exact_reward,
            "reward_penalty": breakdown.penalty,
        }
        record.update(breakdown_payload)
        record.update(trainer_stats)
        return record

    def _format_trainer_stats(self, stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not stats or not isinstance(stats, dict):
            return {}
        payload: Dict[str, Any] = {f"trainer/{key}": value for key, value in stats.items()}

        def pick(keys: List[str]) -> Any:
            for key in keys:
                if key in stats:
                    return stats[key]
            return None

        kl_value = pick(["kl", "ppo/kl", "objective/kl", "policy/kl"])
        entropy_value = pick(["entropy", "policy/entropy"])
        loss_value = pick(["loss", "train/loss", "total_loss", "train/total_loss"])
        derived = {
            "trainer/kl": kl_value,
            "trainer/entropy": entropy_value,
            "trainer/loss": loss_value,
        }
        for key, value in derived.items():
            if value is not None and key not in payload:
                payload[key] = value
        return payload

    def _maybe_checkpoint(self, step: int) -> None:
        if (
            not self.use_trl
            or self.checkpoint_dir is None
            or self.checkpoint_interval <= 0
            or step <= 0
            or step % self.checkpoint_interval != 0
        ):
            return
        assert self.trainer is not None and self.tokenizer is not None
        checkpoint_root = self.checkpoint_dir / f"step_{step:06d}"
        model_dir = checkpoint_root / "model"
        tokenizer_dir = checkpoint_root / "tokenizer"
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(tokenizer_dir)
