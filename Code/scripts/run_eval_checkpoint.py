"""Evaluate a trained checkpoint on GSM8k test set."""

from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path

from mathlm.data import ensure_raw_split, load_raw_split
from mathlm.prompts.zero_shot import get_zero_shot_prompt
from mathlm.rewards import RewardCalculator, RewardWeights
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint directory")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to eval (default: all)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file for results")
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("MathLM Checkpoint Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print("="*60)

    # Auto-convert PyTorch checkpoint to safetensors if needed
    checkpoint_path = Path(args.checkpoint)
    bin_file = checkpoint_path / "pytorch_model.bin"
    safetensors_file = checkpoint_path / "model.safetensors"

    if bin_file.exists() and not safetensors_file.exists():
        print("\n⚠ Converting PyTorch checkpoint to safetensors format...")
        state_dict = torch.load(bin_file, map_location="cpu", weights_only=False)

        # Handle weight tying by cloning shared tensors
        if "lm_head.weight" in state_dict and "model.embed_tokens.weight" in state_dict:
            if state_dict["lm_head.weight"].data_ptr() == state_dict["model.embed_tokens.weight"].data_ptr():
                state_dict["lm_head.weight"] = state_dict["lm_head.weight"].clone()

        save_file(state_dict, str(safetensors_file))
        bin_file.unlink()
        print("✓ Converted to safetensors")

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))

    # Load model and move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.checkpoint),
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    print(f"✓ Model loaded on {device}")

    # Load test data
    print(f"\nLoading {args.split} data...")
    raw_path = ensure_raw_split(args.split, args.data_dir, source="parquet",
                                 parquet_file=args.data_dir / "gsm8k_raw/main" / f"{args.split}-00000-of-00001.parquet")
    examples = load_raw_split(raw_path)

    if args.max_examples:
        examples = examples[:args.max_examples]

    print(f"✓ Loaded {len(examples)} examples")

    # Initialize reward calculator
    reward_weights = RewardWeights(
        syntax=0.1,
        execution=0.2,
        extraction=0.3,
        reasoning=0.5,
        exact=2.0,
        penalty=-0.5
    )
    reward_calc = RewardCalculator(reward_weights)

    # Run evaluation with batching for speed
    print("\nRunning evaluation...", flush=True)
    batch_size = 8
    results = []
    correct = 0

    for batch_start in range(0, len(examples), batch_size):
        print(f"  Batch {batch_start//batch_size + 1}: Processing examples {batch_start}-{min(batch_start + batch_size, len(examples))}", flush=True)

        batch_end = min(batch_start + batch_size, len(examples))
        batch_examples = examples[batch_start:batch_end]

        # Generate prompts for batch
        prompts = [get_zero_shot_prompt(template="default", problem=ex.question) for ex in batch_examples]

        # Batch generate
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)

        # Move inputs to same device as model
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print(f"    Generating...", flush=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        print(f"    Decoding...", flush=True)

        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Process batch results
        for i, (example, prompt, response) in enumerate(zip(batch_examples, prompts, responses)):
            response_text = response[len(prompt):].strip()

            # Evaluate with reward calculator
            breakdown = reward_calc.evaluate(example, response_text)

            # Check if correct
            is_correct = breakdown.exact_reward > 0
            if is_correct:
                correct += 1

            results.append({
                "question": example.question,
                "answer": example.answer,
                "response": response_text,
                "correct": is_correct,
                "reward": breakdown.total,
                "exact_reward": breakdown.exact_reward,
            })

        # Progress logging - log every 10 batches (80 examples) or at milestones
        total_processed = batch_end
        if total_processed % 80 == 0 or total_processed % 100 == 0 or total_processed == len(examples):
            accuracy_so_far = 100 * correct / total_processed
            print(f"  Progress: {total_processed}/{len(examples)} ({100*total_processed/len(examples):.1f}%) | Accuracy: {accuracy_so_far:.2f}%", flush=True)

    accuracy = 100 * correct / len(examples)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total examples: {len(examples)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*60)

    # Save results if output specified
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "total": len(examples),
            "correct": correct,
            "accuracy": accuracy,
            "examples": results,
        }
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
