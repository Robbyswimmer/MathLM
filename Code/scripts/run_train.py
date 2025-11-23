"""Launch PPO training for MathLM using predefined configs."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from mathlm.data import (
    CurriculumConfig,
    apply_curriculum,
    ensure_raw_split,
    load_raw_split,
    save_examples,
)
from mathlm.rewards import RewardCalculator, RewardWeights
from mathlm.training import JSONLLogger, MathLMPPORunner, PromptDataset
from mathlm.utils import ExperimentConfig, parse_config
from mathlm.utils.yaml_loader import load_config as load_yaml_config

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import Dataset
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Any

# --- Monkeypatch for TRL v0.25.1 Compatibility ---
# TRL's AutoModelForCausalLMWithValueHead might return a tuple even with return_dict=True
# in some configurations. We patch it to ensure it returns an object with .logits and .value.

@dataclass
class CausalLMOutputWithValue:
    logits: torch.Tensor
    value: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None

_original_forward = AutoModelForCausalLMWithValueHead.forward

def _patched_forward(self, *args, **kwargs):
    # Force return_dict=True
    kwargs["return_dict"] = True
    output = _original_forward(self, *args, **kwargs)
    
    if isinstance(output, tuple):
        # Heuristic: usually (logits, past_key_values, value) or (logits, value)
        # But if output_hidden_states=True, it might be different.
        # Standard HF tuple: (logits, past_key_values, hidden_states, attentions)
        # TRL wrapper adds value.
        # Let's try to be robust.
        logits = output[0]
        value = None
        past_key_values = None
        hidden_states = None
        attentions = None
        
        # TRL's forward usually returns (logits, past_key_values, value) if no hidden states
        # If hidden states are present, they are usually after past_key_values.
        # This is tricky without knowing exact TRL version behavior.
        # However, we know TRL v0.25.1 expects an object.
        
        # Let's try to map based on type/shape if possible, or just standard positions.
        # If len is 3: (logits, past_key_values, value)
        # If len is 4: (logits, past_key_values, hidden_states, value) ??
        
        # Actually, let's just look at the last element for value, as TRL usually appends it.
        value = output[-1]
        
        if len(output) >= 2:
            # Check if second element is past_key_values (tuple)
            if isinstance(output[1], tuple):
                past_key_values = output[1]
        
        # Check for hidden_states (tuple of tensors)
        for item in output:
            if isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], torch.Tensor):
                # This looks like hidden_states (or attentions)
                # Hidden states usually have same shape as input (B, S, H)
                # Attentions (B, H, S, S)
                if item[0].dim() == 3:
                    hidden_states = item
                elif item[0].dim() == 4:
                    attentions = item
                    
        return CausalLMOutputWithValue(
            logits=logits, 
            value=value, 
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions
        )
    
    return output

AutoModelForCausalLMWithValueHead.forward = _patched_forward

# Add missing 'score' method which TRL v0.25.1 expects on the value model
if not hasattr(AutoModelForCausalLMWithValueHead, "score"):
    def _score(self, hidden_states):
        return self.v_head(hidden_states)
    AutoModelForCausalLMWithValueHead.score = _score
# -------------------------------------------------


def load_experiment(config_path: Path) -> ExperimentConfig:
    raw_cfg = load_yaml_config(config_path)
    return parse_config(raw_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MathLM with PPO")
    parser.add_argument("--config", type=Path, default=Path("configs/curriculum_phase1.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def bootstrap_data(config: ExperimentConfig, data_dir: Path) -> Path:
    curriculum = CurriculumConfig(
        split=config.data.curriculum_split,
        max_problems=config.data.max_problems,
    )

    raw_file = getattr(config.data, "raw_file", None)
    parquet_file = getattr(config.data, "parquet_file", None)
    raw_path = ensure_raw_split(
        config.data.split, 
        data_dir, 
        source=config.data.source, 
        raw_file=raw_file,
        parquet_file=parquet_file
    )
    examples = load_raw_split(raw_path)
    subset = apply_curriculum(examples, curriculum)
    processed_path = data_dir / "processed" / f"gsm8k_{config.data.split}_{curriculum.split}.jsonl"
    save_examples(subset, processed_path)
    return processed_path





def main() -> None:
    args = parse_args()

    print("="*60, flush=True)
    print("MathLM PPO Training", flush=True)
    print("="*60, flush=True)
    print(f"Config: {args.config}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print("="*60, flush=True)

    print("\nLoading configuration...", flush=True)
    config = load_experiment(args.config)

    print("Bootstrapping data...", flush=True)
    processed_path = bootstrap_data(config, args.data_dir)
    print(f"✓ Data prepared: {processed_path}", flush=True)

    reward_weights = RewardWeights(**config.reward_weights)
    reward_calc = RewardCalculator(reward_weights)

    run_id = args.run_id or Path(args.config).stem
    run_dir = args.output_dir / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.__dict__, default=lambda o: o.__dict__, indent=2))
    (run_dir / "dataset.txt").write_text(str(processed_path))

    print(f"\nLoading dataset with prompts...", flush=True)
    dataset = PromptDataset(
        processed_path,
        shots=config.prompting.shots,
        prompt_type=config.prompting.template,
    )
    print(f"✓ Dataset loaded: {len(dataset)} examples", flush=True)
    print(f"  Prompting: {config.prompting.shots}-shot, template={config.prompting.template}", flush=True)

    metrics_path = args.output_dir / "logs" / run_id / "metrics.jsonl"
    logger = JSONLLogger(metrics_path)
    checkpoint_dir = args.output_dir / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = args.output_dir / "traces" / run_id
    traces_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRun ID: {run_id}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    print(f"Checkpoints: {checkpoint_dir}", flush=True)
    print(f"Example traces: {traces_dir}", flush=True)

    runner_kwargs = {
        "minibatch_size": config.training.batch_size,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_interval": config.training.checkpoint_interval,
        "log_examples_interval": 100,
        "traces_dir": traces_dir,
    }

    print(f"\nLoading model with value head: {config.training.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.training.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.training.model_name, return_dict=True)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.training.model_name, return_dict=True)

    print("✓ Model and reference model loaded", flush=True)

    # Ensure generation_config is accessible on the wrapper for TRL v0.25.1+
    # The wrapper (AutoModelForCausalLMWithValueHead) might not expose it directly
    if not hasattr(model, "generation_config"):
        if hasattr(model, "pretrained_model"):
            model.generation_config = model.pretrained_model.generation_config
        else:
            model.generation_config = GenerationConfig.from_model_config(model.config)
    
    if not hasattr(ref_model, "generation_config"):
        if hasattr(ref_model, "pretrained_model"):
            ref_model.generation_config = ref_model.pretrained_model.generation_config
        else:
            ref_model.generation_config = GenerationConfig.from_model_config(ref_model.config)
            
    # Also ensure pad_token_id is set in generation_config if available
    if tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        ref_model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Ensure models return dicts (required by TRL v0.25.1+)
    model.config.return_dict = True
    ref_model.config.return_dict = True
    if hasattr(model, "pretrained_model"):
        model.pretrained_model.config.return_dict = True
    if hasattr(ref_model, "pretrained_model"):
        ref_model.pretrained_model.config.return_dict = True

    # Fix for TRL v0.25.1: Ensure base_model_prefix is set
    # AutoModelForCausalLMWithValueHead wraps the transformer in 'pretrained_model'
    if not hasattr(model, "base_model_prefix"):
        model.base_model_prefix = "pretrained_model"
    if not hasattr(ref_model, "base_model_prefix"):
        ref_model.base_model_prefix = "pretrained_model"

    # Restore is_gradient_checkpointing attribute (removed during cleanup but needed)
    if not hasattr(model, 'is_gradient_checkpointing'):
        model.is_gradient_checkpointing = True
        
    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False # Required for gradient checkpointing
        
    # Limit generation length to avoid OOM
    if hasattr(model, "generation_config"):
        model.generation_config.max_new_tokens = 512
        model.generation_config.min_new_tokens = 1
    if hasattr(ref_model, "generation_config"):
        ref_model.generation_config.max_new_tokens = 512
        ref_model.generation_config.min_new_tokens = 1
        ref_model.is_gradient_checkpointing = False

    # Create a simple reward model (unused for PPO but kept for compatibility)
    # reward_model = AutoModelForCausalLM.from_pretrained(config.training.model_name)
    # print("✓ Reward model loaded", flush=True)

    # Value model: use the full model
    value_model = model

    # Convert dataset to HuggingFace Dataset format
    print("\nPreparing HuggingFace Dataset...", flush=True)
    # TRL expects 'input_ids' for the data collator to work correctly
    # We also need attention_mask for proper padding
    input_ids = []
    attention_masks = []
    for ex in dataset.examples:
        tokenized = tokenizer(ex.prompt, truncation=True, max_length=512)
        input_ids.append(tokenized["input_ids"])
        attention_masks.append(tokenized["attention_mask"])

    dataset_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }
    # Do NOT add 'query' column as it confuses the data collator (it tries to pad it and fails)
    
    hf_dataset = Dataset.from_dict(dataset_dict)
    print(f"✓ Dataset prepared: {len(hf_dataset)} examples", flush=True)

    print("\nInitializing PPO configuration...", flush=True)
    # Use config values directly, assuming bf16 is handled by config
    # Initialize with safe arguments first
    ppo_config = PPOConfig(
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        mini_batch_size=1,
    )
    
    ppo_config.target_kl = config.training.kl_target
    ppo_config.init_kl_coef = config.training.kl_target
    
    # Set other attributes explicitly to support varying TRL versions
    if hasattr(config.training, "bf16"):
        ppo_config.bf16 = config.training.bf16
        ppo_config.fp16 = not config.training.bf16
    
    ppo_config.kl_penalty = "kl"
    ppo_config.model_name = config.training.model_name
    
    # Disable wandb as requested
    ppo_config.report_to = "none"

    # Create reward model wrapper
    from mathlm.rewards.model_wrapper import MathRewardModel
    reward_model_wrapper = MathRewardModel(reward_calc, tokenizer)
    print("✓ Reward model wrapper initialized", flush=True)

    print("\nInitializing PPO trainer...", flush=True)
    trainer = PPOTrainer(
        args=ppo_config,
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        train_dataset=hf_dataset,
        reward_model=reward_model_wrapper,
        value_model=value_model,
    )
    print(f"✓ PPO trainer initialized", flush=True)

    # --- Fix for TRL v0.25.1: PolicyAndValueWrapper missing gradient_checkpointing methods ---
    # The wrapper created by PPOTrainer doesn't expose these methods, causing crash in unwrap_model_for_generation
    if not hasattr(trainer.model, "gradient_checkpointing_disable"):
        print("Patching trainer.model with gradient_checkpointing_disable", flush=True)
        def _disable_gc(self):
            # Try to propagate to policy/model
            if hasattr(self, "policy") and hasattr(self.policy, "gradient_checkpointing_disable"):
                self.policy.gradient_checkpointing_disable()
            if hasattr(self, "model") and hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
                
        import types
        trainer.model.gradient_checkpointing_disable = types.MethodType(_disable_gc, trainer.model)

    if not hasattr(trainer.model, "gradient_checkpointing_enable"):
        print("Patching trainer.model with gradient_checkpointing_enable", flush=True)
        def _enable_gc(self):
            if hasattr(self, "policy") and hasattr(self.policy, "gradient_checkpointing_enable"):
                self.policy.gradient_checkpointing_enable()
            if hasattr(self, "model") and hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

        import types
        trainer.model.gradient_checkpointing_enable = types.MethodType(_enable_gc, trainer.model)
    # ---------------------------------------------------------------------------------------

    print("\n" + "="*60, flush=True)
    print("STARTING PPO TRAINING", flush=True)
    print("="*60, flush=True)

    # Use trainer.train() instead of custom runner
    trainer.train()

    print("\n" + "="*60, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()
