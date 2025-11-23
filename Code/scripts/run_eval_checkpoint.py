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
    model = AutoModelForCausalLM.from_pretrained(str(args.checkpoint))
    model.eval()
    print("✓ Model loaded")

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

    # Run evaluation
    print("\nRunning evaluation...")
    results = []
    correct = 0

    for i, example in enumerate(examples):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(examples)} ({100*(i+1)/len(examples):.1f}%)")

        # Generate prompt
        prompt = get_zero_shot_prompt(template="default", problem=example.question)

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=128,
            do_sample=False,  # Use greedy decoding for evaluation
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the generated part
        response_text = response[len(prompt):].strip()

        # Evaluate with reward calculator
        breakdown = reward_calc.evaluate(example.question, response_text)

        # Check if correct (exact reward > 0)
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
