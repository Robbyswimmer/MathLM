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
        log_examples_interval: int = 100,
        traces_dir: Path | None = None,
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
        self.log_examples_interval = log_examples_interval
        self.traces_dir = traces_dir
        self.use_trl = self.trainer is not None and self.tokenizer is not None

        # EMA tracking
        self.ema_reward = 0.0
        self.ema_alpha = 0.01  # Smoothing factor

        if self.traces_dir:
            self.traces_dir.mkdir(parents=True, exist_ok=True)

    def run(self, total_steps: int = 1000) -> None:
        if total_steps <= 0:
            return
        step = 0
        batch_iter = self.dataset.iter_batches(self.minibatch_size, shuffle=True, infinite=True)
        print(f"\nTraining for {total_steps} steps...", flush=True)
        while step < total_steps:
            batch = next(batch_iter)
            step = self._process_batch(batch, step, total_steps)

            # Progress logging every 10 steps
            if step % 10 == 0:
                progress = (step / total_steps) * 100
                print(f"Step {step}/{total_steps} ({progress:.1f}%) - EMA Reward: {self.ema_reward:.3f}", flush=True)

    def _process_batch(self, batch: List[PromptExample], start_step: int, total_steps: int) -> int:
        if self.use_trl:
            rollouts, trainer_stats = self._trl_batch(batch)
        else:
            rollouts, trainer_stats = self._stub_batch(batch)
        trainer_payload = self._format_trainer_stats(trainer_stats)
        step = start_step
        for rollout in rollouts:
            # Update EMA reward
            self.ema_reward = self.ema_alpha * rollout.reward + (1 - self.ema_alpha) * self.ema_reward

            record = self._build_log_record(step, rollout, trainer_payload)
            record["ema_reward"] = self.ema_reward  # Add EMA to logged metrics
            self.logger.log(record)

            # Log example outputs periodically
            self._maybe_log_example(step, rollout)

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
        """Use TRL PPOTrainer to generate responses and perform PPO update."""
        if torch is None:
            raise RuntimeError("PyTorch is required for training.")
        assert self.trainer is not None and self.tokenizer is not None

        # Get model from trainer - unwrap to get the pretrained model
        model = self.trainer.model
        # TRL wraps the model, so we need to access the underlying pretrained_model
        if hasattr(model, 'pretrained_model'):
            gen_model = model.pretrained_model
        elif hasattr(model, 'base_model'):
            gen_model = model.base_model
        else:
            gen_model = model
        device = next(model.parameters()).device

        # Prepare prompts
        prompts = [example.prompt for example in batch]
        encodings = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        query_tensors = encodings["input_ids"].to(device)

        # Generate responses using the underlying pretrained model
        with torch.no_grad():
            response_tensors = gen_model.generate(
                query_tensors,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode responses (full output including prompt)
        responses = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        # Extract just the generated part
        prompt_texts = self.tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
        response_texts = [resp[len(prompt):].strip() for resp, prompt in zip(responses, prompt_texts)]

        # Compute rewards using our custom reward calculator
        reward_values: List[float] = []
        rollouts: List[Rollout] = []
        for example, response_text in zip(batch, response_texts):
            breakdown = self.reward_calc.evaluate(example.question, response_text)
            rollouts.append(Rollout(prompt=example.prompt, response_text=response_text, breakdown=breakdown))
            reward_values.append(breakdown.total)

        # Convert rewards to tensors for PPO update
        rewards = [torch.tensor(r, device=device) for r in reward_values]

        # Perform PPO step using TRL's trainer
        stats = self.trainer.step(query_tensors.tolist(), response_tensors.tolist(), rewards)

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
        print(f"✓ Checkpoint saved at step {step}", flush=True)

    def _maybe_log_example(self, step: int, rollout: Rollout) -> None:
        """Log example outputs periodically to track training progress."""
        if (
            self.traces_dir is None
            or self.log_examples_interval <= 0
            or step <= 0
            or step % self.log_examples_interval != 0
        ):
            return

        import json

        example_file = self.traces_dir / f"step_{step:06d}.txt"
        breakdown = rollout.breakdown

        with example_file.open("w", encoding="utf-8") as f:
            f.write(f"=" * 80 + "\n")
            f.write(f"Training Step: {step}\n")
            f.write(f"EMA Reward: {self.ema_reward:.3f}\n")
            f.write(f"=" * 80 + "\n\n")

            f.write("PROMPT:\n")
            f.write("-" * 80 + "\n")
            f.write(rollout.prompt + "\n\n")

            f.write("RESPONSE:\n")
            f.write("-" * 80 + "\n")
            f.write(rollout.response_text + "\n\n")

            f.write("REWARD BREAKDOWN:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Syntax:     {breakdown.syntax_reward:+.2f}\n")
            f.write(f"  Execution:  {breakdown.execution_reward:+.2f}\n")
            f.write(f"  Extraction: {breakdown.extraction_reward:+.2f}\n")
            f.write(f"  Reasoning:  {breakdown.reasoning_reward:+.2f}\n")
            f.write(f"  Exact:      {breakdown.exact_reward:+.2f}\n")
            f.write(f"  Penalty:    {breakdown.penalty:+.2f}\n")
            f.write(f"  TOTAL:      {breakdown.total:+.2f}\n")

        print(f"📝 Example logged at step {step} (reward: {breakdown.total:.3f})", flush=True)
