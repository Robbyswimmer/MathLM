"""Launch PPO training for MathLM using predefined configs."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
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
import os

from mathlm.utils import ExperimentConfig, parse_config
from mathlm.utils.yaml_loader import load_config as load_yaml_config

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import trl.models.utils as trl_utils
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import Dataset
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import torch.nn as nn
import math
from contextlib import contextmanager
import trl.models.utils as trl_utils

# Disable safetensors globally to avoid shared memory errors with tied weights
import os
os.environ["SAFETENSORS_FAST_GPU"] = "0"

# Monkey-patch PreTrainedModel.save_pretrained to always use safe_serialization=False
from transformers import PreTrainedModel
_original_save_pretrained = PreTrainedModel.save_pretrained

def _patched_save_pretrained(self, *args, **kwargs):
    kwargs["safe_serialization"] = False  # Force PyTorch bin format
    return _original_save_pretrained(self, *args, **kwargs)

PreTrainedModel.save_pretrained = _patched_save_pretrained
print("✓ Patched save_pretrained to disable safetensors", flush=True)

# Monkeypatch PolicyAndValueWrapper to add missing gradient_checkpointing methods
# This must be done before PPOTrainer is instantiated
def _noop_gc(self):
    """No-op for gradient checkpointing toggle."""
    pass

# Try multiple import paths for PolicyAndValueWrapper
_patched_pvw = False
for module_path in [
    "trl.models.modeling_value_head",
    "trl.trainer.ppo_trainer",
    "trl.models",
]:
    try:
        import importlib
        mod = importlib.import_module(module_path)
        if hasattr(mod, "PolicyAndValueWrapper"):
            PolicyAndValueWrapper = mod.PolicyAndValueWrapper
            if not hasattr(PolicyAndValueWrapper, "gradient_checkpointing_disable"):
                PolicyAndValueWrapper.gradient_checkpointing_disable = _noop_gc
            if not hasattr(PolicyAndValueWrapper, "gradient_checkpointing_enable"):
                PolicyAndValueWrapper.gradient_checkpointing_enable = _noop_gc
            print(f"✓ Patched PolicyAndValueWrapper from {module_path}", flush=True)
            _patched_pvw = True
            break
    except (ImportError, AttributeError):
        continue

if not _patched_pvw:
    print("! Could not find PolicyAndValueWrapper to patch", flush=True)

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
    original_tuple: Optional[Tuple] = None

    def __iter__(self):
        if self.original_tuple is not None:
            return iter(self.original_tuple)
        # Fallback if created manually (unlikely in this patch)
        components = [self.logits]
        if self.past_key_values is not None:
            components.append(self.past_key_values)
        if self.value is not None:
            components.append(self.value)
        if self.hidden_states is not None:
            components.append(self.hidden_states)
        if self.attentions is not None:
            components.append(self.attentions)
        return iter(components)
        
    def __getitem__(self, idx):
        return list(self)[idx]

# --- Robust Fix for TRL v0.25.1 Compatibility ---
# --- Robust Fix for TRL v0.25.1 Compatibility ---
# Monkeypatch AutoModelForCausalLMWithValueHead.forward to ensure it returns an object with .logits
# and is also iterable (for TRL hooks).

@dataclass
class CausalLMOutputWithValue:
    logits: torch.Tensor
    value: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    original_tuple: Optional[Tuple] = None

    def __iter__(self):
        if self.original_tuple is not None:
            return iter(self.original_tuple)
        components = [self.logits]
        if self.past_key_values is not None:
            components.append(self.past_key_values)
        if self.value is not None:
            components.append(self.value)
        if self.hidden_states is not None:
            components.append(self.hidden_states)
        if self.attentions is not None:
            components.append(self.attentions)
        return iter(components)
        
    def __getitem__(self, idx):
        return list(self)[idx]

_original_forward = AutoModelForCausalLMWithValueHead.forward

def _patched_forward(self, *args, **kwargs):
    # Force return_dict=True
    kwargs["return_dict"] = True
    
    # Call original forward
    output = _original_forward(self, *args, **kwargs)
    
    # If it's already a dict/object with logits, return it
    if hasattr(output, "logits"):
        return output

    # If it's a tuple, convert it
    if isinstance(output, tuple):
        logits = output[0]
        value = output[-1]
        past_key_values = None
        hidden_states = None
        attentions = None
        
        if len(output) >= 2:
            if isinstance(output[1], tuple):
                past_key_values = output[1]
        
        for item in output:
            if isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], torch.Tensor):
                if item[0].dim() == 3:
                    hidden_states = item
                elif item[0].dim() == 4:
                    attentions = item
                    
        return CausalLMOutputWithValue(
            logits=logits, 
            value=value, 
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            original_tuple=output
        )
    return output

print("Applying monkeypatch to AutoModelForCausalLMWithValueHead.forward...", flush=True)
AutoModelForCausalLMWithValueHead.forward = _patched_forward

# Add missing 'score' method which TRL v0.25.1 expects on the value model
if not hasattr(AutoModelForCausalLMWithValueHead, "score"):
    def _score(self, hidden_states):
        return self.v_head(hidden_states)
    AutoModelForCausalLMWithValueHead.score = _score

def verify_model_output(model, name="Model"):
    print(f"Verifying output format for {name}...", flush=True)
    try:
        # Get device safely
        device = getattr(model, "device", None)
        if device is None and hasattr(model, "pretrained_model"):
            device = getattr(model.pretrained_model, "device", None)
        if device is None:
            device = next(model.parameters()).device

        # Create dummy input
        dummy_input = torch.tensor([[1, 2, 3]], device=device)
        dummy_mask = torch.tensor([[1, 1, 1]], device=device)

        # Run forward pass
        with torch.no_grad():
            output = model(dummy_input, attention_mask=dummy_mask, return_dict=True)

        print(f"  Output type: {type(output)}", flush=True)
        if hasattr(output, "logits"):
            print(f"  ✓ Output has .logits attribute", flush=True)
        else:
            print(f"  ✗ Output MISSING .logits attribute!", flush=True)

        if hasattr(output, "value"):
            print(f"  ✓ Output has .value attribute", flush=True)

        if isinstance(output, tuple):
             print(f"  ! Output is a tuple (patched wrapper should handle this)", flush=True)

        # Check iterability
        try:
            iter(output)
            print(f"  ✓ Output is iterable", flush=True)
        except TypeError:
            print(f"  ✗ Output is NOT iterable!", flush=True)

    except Exception as e:
        print(f"  ✗ Verification failed with error: {e}", flush=True)
        import traceback
        traceback.print_exc()
# -------------------------------------------------


def log_cuda_memory(label: str) -> None:
    """Log CUDA memory usage in MiB if available."""
    if not torch.cuda.is_available():  # pragma: no cover - CUDA may be absent locally
        return
    device = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"[CUDA] {label}: alloc={alloc:.1f} MiB reserved={reserved:.1f} MiB max_alloc={max_alloc:.1f} MiB", flush=True)


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
    print("MathLM PPO Training - VERSION: PATCH_VERIFY_V1", flush=True)
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
    log_cuda_memory("After tokenizer load")

    # Load model with value head for PPO
    # Use bf16 to save memory (fp32 uses too much memory during backward pass)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.training.model_name,
        return_dict=True,
        torch_dtype=torch.bfloat16,
    )

    # Load reference model
    # In multi-GPU, keep on CPU to save GPU memory (will move batches to GPU as needed)
    # In single-GPU, load on GPU
    world_size_early = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_early > 1:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.training.model_name,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        print("✓ Reference model loaded on CPU (multi-GPU memory saving)", flush=True)
    else:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.training.model_name,
            return_dict=True,
            torch_dtype=torch.bfloat16,
        )
        print("✓ Reference model loaded on GPU (single-GPU)", flush=True)
    log_cuda_memory("After model + ref_model load")

    # Explicitly patch instances to ensure our forward is used
    import types
    model.forward = types.MethodType(_patched_forward, model)
    ref_model.forward = types.MethodType(_patched_forward, ref_model)
    print("✓ Patched model instances to handle tuple outputs", flush=True)
    
    # Verify output format
    verify_model_output(model, "Policy Model")
    verify_model_output(ref_model, "Ref Model")
    log_cuda_memory("After verify_model_output")

    print("✓ Models loaded (policy and ref on GPU)", flush=True)

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
    log_cuda_memory("After generation_config setup")

    # Ensure models return dicts (required by TRL v0.25.1+)
    model.config.return_dict = True
    ref_model.config.return_dict = True
    if hasattr(model, "pretrained_model"):
        model.pretrained_model.config.return_dict = True
    if hasattr(ref_model, "pretrained_model"):
        ref_model.pretrained_model.config.return_dict = True
    log_cuda_memory("After return_dict enforcement")

    # Fix for TRL v0.25.1: Ensure base_model_prefix is set
    # AutoModelForCausalLMWithValueHead wraps the transformer in 'pretrained_model'
    if not hasattr(model, "base_model_prefix"):
        model.base_model_prefix = "pretrained_model"
    if not hasattr(ref_model, "base_model_prefix"):
        ref_model.base_model_prefix = "pretrained_model"
    log_cuda_memory("After base_model_prefix setup")

    # Get world_size early for configuration
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Restore is_gradient_checkpointing attribute (removed during cleanup but needed)
    if not hasattr(model, 'is_gradient_checkpointing'):
        model.is_gradient_checkpointing = False

    # Disable gradient checkpointing in multi-GPU (conflicts with DDP)
    # In single GPU, we can enable it to save memory
    if world_size == 1 and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False # Required for gradient checkpointing
        model.is_gradient_checkpointing = True
        print("✓ Gradient checkpointing enabled (single GPU)", flush=True)
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        print("✓ Gradient checkpointing disabled (multi-GPU for DDP compatibility)", flush=True)

    # Limit generation length to avoid OOM (aggressive limits for multi-GPU)
    gen_max = int(os.environ.get("MAX_NEW_TOKENS", 48 if world_size > 1 else 128))
    if hasattr(model, "generation_config"):
        model.generation_config.max_new_tokens = gen_max
        model.generation_config.min_new_tokens = 1
    if hasattr(ref_model, "generation_config"):
        ref_model.generation_config.max_new_tokens = gen_max
        ref_model.generation_config.min_new_tokens = 1
        ref_model.is_gradient_checkpointing = False
    print(f"✓ Generation max_new_tokens set to {gen_max}", flush=True)

    # Create a simple reward model (unused for PPO but kept for compatibility)
    # reward_model = AutoModelForCausalLM.from_pretrained(config.training.model_name)
    # print("✓ Reward model loaded", flush=True)

    # Value model: use the full model
    value_model = model

    # Convert dataset to HuggingFace Dataset format
    print("\nPreparing HuggingFace Dataset...", flush=True)
    # TRL expects 'input_ids' for the data collator to work correctly
    # We also need attention_mask for proper padding
    # Reduce to 128 tokens for multi-GPU to save memory
    prompt_max_len = int(os.environ.get("PROMPT_MAX_LEN", 128))
    input_ids = []
    attention_masks = []
    for ex in dataset.examples:
        tokenized = tokenizer(ex.prompt, truncation=True, max_length=prompt_max_len)
        input_ids.append(tokenized["input_ids"])
        attention_masks.append(tokenized["attention_mask"])

    dataset_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }
    # Do NOT add 'query' column as it confuses the data collator (it tries to pad it and fails)
    
    hf_dataset = Dataset.from_dict(dataset_dict)
    print(f"✓ Dataset prepared: {len(hf_dataset)} examples", flush=True)
    log_cuda_memory("After dataset prep")

    print("\nInitializing PPO configuration...", flush=True)
    # Use batch_size=1 per device to minimize memory usage
    target_batch = 1
    per_device_cap = int(os.environ.get("PPO_BATCH_PER_DEVICE", 1))
    effective_batch = max(1, min(target_batch, per_device_cap))

    # Initialize with safe arguments first
    ppo_config = PPOConfig(
        learning_rate=config.training.learning_rate,
        batch_size=effective_batch,
        mini_batch_size=1,
        gradient_accumulation_steps=getattr(config.training, "gradient_accumulation_steps", 4),
        fp16=False,
        bf16=True,
        save_safetensors=False,  # Disable safetensors to avoid shared memory error
    )

    ppo_config.target_kl = config.training.kl_target
    ppo_config.init_kl_coef = config.training.kl_target

    # Set other attributes explicitly to support varying TRL versions
    ppo_config.bf16 = True
    ppo_config.fp16 = False

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
        eval_dataset=hf_dataset,
        reward_model=reward_model_wrapper,
        value_model=value_model,
    )
    print(f"✓ PPO trainer initialized", flush=True)
    log_cuda_memory("After PPOTrainer init")

    # Patch gradient_checkpointing methods on trainer.model (PolicyAndValueWrapper in multi-GPU)
    import types

    def _noop_gc_disable(self):
        """No-op gradient checkpointing disable."""
        pass

    def _noop_gc_enable(self):
        """No-op gradient checkpointing enable."""
        pass

    # Patch the trainer.model instance directly (this is PolicyAndValueWrapper in Accelerate mode)
    if not hasattr(trainer.model, "gradient_checkpointing_disable"):
        trainer.model.gradient_checkpointing_disable = types.MethodType(_noop_gc_disable, trainer.model)
        print("✓ Patched trainer.model.gradient_checkpointing_disable", flush=True)

    if not hasattr(trainer.model, "gradient_checkpointing_enable"):
        trainer.model.gradient_checkpointing_enable = types.MethodType(_noop_gc_enable, trainer.model)
        print("✓ Patched trainer.model.gradient_checkpointing_enable", flush=True)

    # Also patch any nested models we can find
    def _ensure_gc_hooks(obj, name=""):
        if obj is None:
            return
        if not hasattr(obj, "gradient_checkpointing_disable"):
            try:
                obj.gradient_checkpointing_disable = types.MethodType(_noop_gc_disable, obj)
                if name:
                    print(f"✓ Patched {name}.gradient_checkpointing_disable", flush=True)
            except Exception:
                pass
        if not hasattr(obj, "gradient_checkpointing_enable"):
            try:
                obj.gradient_checkpointing_enable = types.MethodType(_noop_gc_enable, obj)
                if name:
                    print(f"✓ Patched {name}.gradient_checkpointing_enable", flush=True)
            except Exception:
                pass

    _ensure_gc_hooks(getattr(trainer.model, "policy", None), "trainer.model.policy")
    _ensure_gc_hooks(getattr(trainer.model, "pretrained_model", None), "trainer.model.pretrained_model")

    # Patch trainer models to always yield logits (handles tuple outputs)
    def _patch_module_outputs(module, label: str) -> None:
        if module is None:
            return
        if hasattr(module, "config"):
            module.config.return_dict = True
        original = module.forward
        def _fwd(self, *args, **kwargs):
            kwargs["return_dict"] = True
            out = original(*args, **kwargs)
            if hasattr(out, "logits"):
                return out
            if isinstance(out, tuple) and out:
                return CausalLMOutputWithValue(logits=out[0], original_tuple=out)
            return out
        module.forward = types.MethodType(_fwd, module)
        print(f"✓ Patched {label}.forward to normalize outputs", flush=True)

    try:
        _patch_module_outputs(getattr(trainer, "ref_model", None), "trainer.ref_model")
    except Exception as err:
        print(f"! Failed to patch trainer.ref_model.forward: {err}", flush=True)
    try:
        _patch_module_outputs(getattr(trainer, "model", None), "trainer.model")
    except Exception as err:
        print(f"! Failed to patch trainer.model.forward: {err}", flush=True)
    log_cuda_memory("After forward patching")

    print("\n" + "="*60, flush=True)
    print("STARTING PPO TRAINING", flush=True)
    print("="*60, flush=True)

    # Use trainer.train() instead of custom runner
    # Wrap ref_model to ensure dict outputs even if TRL bypasses our global patch
    class SafeRefModel(nn.Module):
        def __init__(self, base, target_device):
            super().__init__()
            self.base = base
            self.target_device = target_device
        def forward(self, *args, **kwargs):
            kwargs["return_dict"] = True
            out = self.base(*args, **kwargs)
            if hasattr(out, "logits"):
                try:
                    out.logits = out.logits.to(self.target_device)
                    if hasattr(out, "value") and isinstance(out.value, torch.Tensor):
                        out.value = out.value.to(self.target_device)
                except Exception:
                    pass
                return out
            if isinstance(out, tuple) and out:
                logits = out[0]
                try:
                    logits = logits.to(self.target_device)
                except Exception:
                    pass
                return CausalLMOutputWithValue(logits=logits, original_tuple=out)
            return out
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.base, name)

    try:
        policy_device = next(model.parameters()).device
        trainer.ref_model = SafeRefModel(trainer.ref_model, policy_device)
        print("✓ Wrapped trainer.ref_model with SafeRefModel", flush=True)
    except Exception as err:
        print(f"! Failed to wrap trainer.ref_model: {err}", flush=True)
    log_cuda_memory("Before trainer.train()")

    # Store the last batch data for logging
    trainer._last_batch_data = {'queries': None, 'responses': None, 'rewards': None}

    # Patch the log method to print examples
    original_log = trainer.log
    last_logged_episode = {'value': -100}

    def log_with_examples(logs):
        # Call original log
        result = original_log(logs)

        # Print example every 100 episodes
        if 'episode' in logs:
            episode = logs['episode']
            # Also log key metrics at every episode
            if episode % 32 == 0:  # Log every batch
                score = logs.get('objective/scores', 'N/A')
                reward = logs.get('objective/rlhf_reward', 'N/A')
                print(f"[Episode {episode}] Score: {score}, Reward: {reward}", flush=True)

            if episode % 100 == 0 and episode != last_logged_episode['value']:
                last_logged_episode['value'] = episode

                print("\n" + "="*60, flush=True)
                print(f"EXAMPLE OUTPUT AT EPISODE {episode}", flush=True)
                print("="*60, flush=True)

                # Try to print last batch data
                batch_data = trainer._last_batch_data
                if batch_data['queries'] is not None and batch_data['responses'] is not None:
                    try:
                        query = tokenizer.decode(batch_data['queries'][0], skip_special_tokens=True)
                        response = tokenizer.decode(batch_data['responses'][0], skip_special_tokens=True)
                        reward = batch_data['rewards'][0] if batch_data['rewards'] is not None else 'N/A'

                        print(f"Query:\n{query[:400]}", flush=True)
                        print(f"\nResponse:\n{response[:400]}", flush=True)
                        print(f"\nReward: {reward}", flush=True)
                    except Exception as e:
                        print(f"Could not decode: {e}", flush=True)
                else:
                    print("No batch data available yet", flush=True)

                print("="*60 + "\n", flush=True)

        return result

    # SIMPLE: Just capture and print the actual training batch data
    original_compute_rewards = trainer.compute_rewards

    def compute_rewards_with_logging(scores, logprobs, ref_logprobs, masks, values=None):
        # Call original
        rewards = original_compute_rewards(scores, logprobs, ref_logprobs, masks, values)

        # Get episode count
        current_episode = getattr(trainer, 'current_episode', 0)

        # Print every 100 episodes
        if current_episode % 100 == 0 and current_episode > 0:
            print("\n" + "="*60, flush=True)
            print(f"TRAINING EXAMPLE AT EPISODE {current_episode}", flush=True)
            print("="*60, flush=True)
            print(f"Score: {scores[0].item() if len(scores) > 0 else 'N/A'}", flush=True)
            print(f"Reward: {rewards[0].mean().item() if len(rewards) > 0 else 'N/A'}", flush=True)

            # Try to get the actual text from last batch
            if hasattr(trainer, '_last_batch_data'):
                bd = trainer._last_batch_data
                if bd['queries'] is not None and bd['responses'] is not None:
                    try:
                        q = tokenizer.decode(bd['queries'][0], skip_special_tokens=True)
                        r = tokenizer.decode(bd['responses'][0], skip_special_tokens=True)
                        print(f"\nQuery:\n{q}\n", flush=True)
                        print(f"Response:\n{r}", flush=True)
                    except:
                        pass
            print("="*60 + "\n", flush=True)

        return rewards

    trainer.compute_rewards = compute_rewards_with_logging

    # Also capture the batch when it's created
    if hasattr(trainer, 'generate'):
        original_generate = trainer.generate
        def generate_with_capture(query_tensor, **kwargs):
            result = original_generate(query_tensor, **kwargs)
            trainer._last_batch_data['queries'] = query_tensor
            trainer._last_batch_data['responses'] = result
            return result
        trainer.generate = generate_with_capture

    trainer.log = log_with_examples
    print("✓ Added example logging (every 100 episodes)", flush=True)

    trainer.train()

    print("\n" + "="*60, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("="*60, flush=True)

    # Save final model
    final_checkpoint = checkpoint_dir / "final"
    final_checkpoint.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final model to {final_checkpoint}...", flush=True)
    try:
        # Save the underlying pretrained model (not the TRL wrapper)
        # This ensures config.json is saved properly
        trainer.model.pretrained_model.save_pretrained(str(final_checkpoint))
        tokenizer.save_pretrained(str(final_checkpoint))
        print(f"✓ Final model saved successfully", flush=True)
    except Exception as e:
        print(f"! Error saving final model: {e}", flush=True)
        print("Attempting fallback save with PyTorch...", flush=True)
        try:
            torch.save(trainer.model.state_dict(), str(final_checkpoint / "pytorch_model.bin"))
            tokenizer.save_pretrained(str(final_checkpoint))
            print(f"✓ Model saved with PyTorch fallback", flush=True)
        except Exception as e2:
            print(f"! Fallback save also failed: {e2}", flush=True)


if __name__ == "__main__":
    main()
