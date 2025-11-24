"""Launch GRPO training for MathLM using predefined configs."""

from __future__ import annotations

import argparse
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
from mathlm.prompts.zero_shot import get_zero_shot_prompt
from mathlm.rewards.grpo_rewards import combined_math_reward
from mathlm.training import PromptDataset
from mathlm.utils import ExperimentConfig, parse_config
from mathlm.utils.yaml_loader import load_config as load_yaml_config

from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch

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


def log_cuda_memory(label: str = ""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_alloc = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[CUDA] {label}: alloc={alloc:.1f} MiB reserved={reserved:.1f} MiB max_alloc={max_alloc:.1f} MiB", flush=True)


def load_experiment(config_path: Path) -> ExperimentConfig:
    raw_cfg = load_yaml_config(config_path)
    return parse_config(raw_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MathLM with GRPO")
    parser.add_argument("--config", type=Path, default=Path("configs/grpo_base.yaml"))
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
    print("MathLM GRPO Training", flush=True)
    print("="*60, flush=True)
    print(f"Config: {args.config}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print("="*60, flush=True)

    print("\nLoading configuration...", flush=True)
    config = load_experiment(args.config)

    print("Bootstrapping data...", flush=True)
    processed_path = bootstrap_data(config, args.data_dir)
    print(f"✓ Data prepared: {processed_path}", flush=True)

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

    checkpoint_dir = args.output_dir / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRun ID: {run_id}", flush=True)
    print(f"Checkpoints: {checkpoint_dir}", flush=True)

    print(f"\nLoading model: {config.training.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.training.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Critical for generation: pad on the left so generation can append new tokens correctly
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.training.model_name,
        device_map="auto",
        dtype=torch.bfloat16 if getattr(config.training, "bf16", True) else torch.float32,
    )
    model.config.use_cache = False
    print(f"✓ Model loaded", flush=True)
    log_cuda_memory("After model loading")

    # Prepare dataset for GRPO (needs to be HF Dataset with answer column)
    print("\nPreparing HF dataset for GRPO...", flush=True)

    # Get raw examples to extract answers
    raw_examples = load_raw_split(processed_path)

    # Build prompts with chat template
    prompts = []
    answers = []
    for example in raw_examples:
        prompt_text = get_zero_shot_prompt(template=config.prompting.template, problem=example.question)

        # Apply chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt_text

        prompts.append(formatted_prompt)
        answers.append(example.answer)

    hf_dataset = Dataset.from_dict({
        "prompt": prompts,
        "answers": answers,  # Use plural to match reward function parameter
    })
    print(f"✓ HF dataset created: {len(hf_dataset)} examples", flush=True)

    # Initialize GRPO config
    print("\nInitializing GRPO configuration...", flush=True)

    grpo_config = GRPOConfig(
        output_dir=str(checkpoint_dir),
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=getattr(config.training, "gradient_accumulation_steps", 4),
        num_generations=getattr(config.training, "num_generations", 8),
        max_completion_length=getattr(config.training, "max_completion_length", 512),
        max_prompt_length=getattr(config.training, "max_prompt_length", 256),
        temperature=getattr(config.training, "temperature", 0.9),
        beta=getattr(config.training, "beta", 0.04),
        bf16=getattr(config.training, "bf16", True),
        fp16=False,
        logging_steps=10,
        save_steps=config.training.checkpoint_interval,
        report_to="none",  # Disable wandb
        num_train_epochs=1,  # We'll control via max_steps
        max_steps=config.training.total_steps,
    )

    print(f"✓ GRPO config initialized", flush=True)
    print(f"  Batch size: {grpo_config.per_device_train_batch_size}", flush=True)
    print(f"  Gradient accumulation: {grpo_config.gradient_accumulation_steps}", flush=True)
    print(f"  Num generations per prompt: {grpo_config.num_generations}", flush=True)
    print(f"  Learning rate: {grpo_config.learning_rate}", flush=True)
    print(f"  Beta (KL penalty): {grpo_config.beta}", flush=True)
    print(f"  Temperature: {grpo_config.temperature}", flush=True)

    # Initialize GRPO trainer
    print("\nInitializing GRPO trainer...", flush=True)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[combined_math_reward],  # Our custom reward function
        args=grpo_config,
        train_dataset=hf_dataset,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )
    print(f"✓ GRPO trainer initialized", flush=True)
    log_cuda_memory("After GRPOTrainer init")

    # Add periodic testing callback
    test_problem = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    test_prompt_raw = get_zero_shot_prompt(template=config.prompting.template, problem=test_problem)

    # Apply chat template to test prompt
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": test_prompt_raw}]
        test_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        test_prompt = test_prompt_raw

    last_logged = {'step': -500}

    original_log = trainer.log

    def log_with_test(logs):
        result = original_log(logs)

        if 'step' in logs:
            step = logs['step']

            # Test model every 500 steps
            if step - last_logged['step'] >= 500 and step > 0:
                last_logged['step'] = step

                print("\n" + "="*60, flush=True)
                print(f"MODEL TEST AT STEP {step}", flush=True)
                print("="*60, flush=True)

                try:
                    device = next(model.parameters()).device
                    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
                    model.eval()
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    model.train()

                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Extract just the model's response (after the prompt)
                    if "<start_of_turn>model" in response:
                        model_output = response.split("<start_of_turn>model")[-1].strip()
                    else:
                        model_output = response[len(test_prompt):].strip()

                    print(f"Problem: Janet's ducks problem...\n", flush=True)
                    print(f"Model says:\n{model_output[:500]}...\n", flush=True)
                    print(f"(Expected: 18)", flush=True)
                    print("="*60 + "\n", flush=True)
                except Exception as e:
                    print(f"! Test generation failed: {e}", flush=True)

        return result

    trainer.log = log_with_test

    print("\n" + "="*60, flush=True)
    print("STARTING GRPO TRAINING", flush=True)
    print("="*60, flush=True)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n! Training interrupted by user", flush=True)
    except Exception as e:
        print(f"\n! Training failed with error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*60, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("="*60, flush=True)

    # Save final model
    final_model_path = checkpoint_dir / "final"
    print(f"\nSaving final model to {final_model_path}...", flush=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("✓ Final model saved", flush=True)


if __name__ == "__main__":
    main()
