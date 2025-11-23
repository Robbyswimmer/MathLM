"""Evaluate trained MathLM checkpoints against GSM8k baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from mathlm.data import ensure_raw_split
from mathlm.training import PromptDataset
from mathlm.evaluation import HFModelGenerator, MathLMEvaluator
from mathlm.utils import ExperimentConfig, parse_config
from mathlm.utils.yaml_loader import load_config as load_yaml_config

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - transformers may be unavailable in some tests
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MathLM checkpoints and baselines")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_zero_shot.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to fine-tuned checkpoint root (optional)")
    parser.add_argument("--output", type=Path, default=Path("experiments/logs/eval.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset-path", type=Path, default=None, help="Optional explicit JSONL dataset path")
    parser.add_argument("--baseline-model", type=str, default=None, help="Baseline model identifier (defaults to config training model)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of evaluation samples")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--predictions-dir", type=Path, default=None, help="Directory to store prediction traces")
    return parser.parse_args()


def load_experiment(config_path: Path) -> ExperimentConfig:
    raw_cfg = load_yaml_config(config_path)
    return parse_config(raw_cfg)


def load_dataset_path(config: ExperimentConfig, data_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    return ensure_raw_split(config.data.split, data_dir, source=config.data.source)


def resolve_model_and_tokenizer(ref: str | Path):
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError("transformers is required for evaluation; please install it in your environment.")
    ref_path = Path(ref)
    if ref_path.exists():
        print(f"Loading model from local path: {ref_path}", flush=True)
        model_dir = ref_path / "model"
        tokenizer_dir = ref_path / "tokenizer"
        model_source = model_dir if model_dir.exists() else ref_path
        tokenizer_source = tokenizer_dir if tokenizer_dir.exists() else ref_path
        print(f"  Model source: {model_source}", flush=True)
        print(f"  Tokenizer source: {tokenizer_source}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_source)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        model_name = ref_path.name
        print(f"✓ Model loaded: {model_name}", flush=True)
    else:
        print(f"Downloading model from HuggingFace: {ref}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(str(ref))
        tokenizer = AutoTokenizer.from_pretrained(str(ref))
        model_name = str(ref)
        print(f"✓ Model downloaded: {model_name}", flush=True)
    return model_name, model, tokenizer


def evaluate_model(
    label: str,
    model_ref: str | Path,
    dataset: PromptDataset,
    *,
    batch_size: int,
    max_examples: int | None,
    max_new_tokens: int,
    device: str | None,
    predictions_dir: Path,
) -> Dict[str, Any]:
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluating {label} model", flush=True)
    print(f"{'='*60}", flush=True)

    model_name, model, tokenizer = resolve_model_and_tokenizer(model_ref)

    print(f"Moving model to device: {device or 'auto'}", flush=True)
    generator = HFModelGenerator(model, tokenizer, max_new_tokens=max_new_tokens, device=device)
    evaluator = MathLMEvaluator(generator)

    num_eval = max_examples or len(dataset)
    print(f"Starting evaluation on {num_eval} examples (batch_size={batch_size})", flush=True)

    stats, predictions = evaluator.evaluate(
        dataset,
        batch_size=batch_size,
        max_examples=max_examples,
        shuffle=False,
    )

    print(f"✓ Evaluation complete!", flush=True)
    print(f"  Accuracy: {stats.accuracy:.2%} ({stats.num_correct}/{stats.num_examples})", flush=True)

    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / f"{label}_predictions.jsonl"
    print(f"Writing predictions to {predictions_path}", flush=True)

    with predictions_path.open("w", encoding="utf-8") as fout:
        for pred in predictions:
            fout.write(json.dumps(pred.to_json()) + "\n")

    print(f"✓ Predictions saved", flush=True)

    return {
        "label": label,
        "model_name": model_name,
        "num_examples": stats.num_examples,
        "num_correct": stats.num_correct,
        "accuracy": stats.accuracy,
        "predictions_path": str(predictions_path),
    }


def main() -> None:
    args = parse_args()

    print("="*60, flush=True)
    print("MathLM Evaluation", flush=True)
    print("="*60, flush=True)
    print(f"Config: {args.config}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"Device: {args.device or 'auto'}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Max examples: {args.max_examples or 'all'}", flush=True)
    print("="*60, flush=True)

    print("\nLoading configuration...", flush=True)
    config = load_experiment(args.config)

    print("Loading dataset...", flush=True)
    dataset_path = load_dataset_path(config, args.data_dir, args.dataset_path)
    dataset = PromptDataset(
        dataset_path,
        shots=config.prompting.shots,
        prompt_type=config.prompting.template,
    )
    print(f"✓ Dataset loaded: {len(dataset)} examples from {dataset_path}", flush=True)
    print(f"  Prompting: {config.prompting.shots}-shot, template={config.prompting.template}", flush=True)

    predictions_dir = args.predictions_dir or args.output.parent / "predictions"
    max_examples = args.max_examples if args.max_examples and args.max_examples > 0 else None

    report: Dict[str, Any] = {
        "config": str(args.config),
        "dataset": str(dataset_path),
        "max_examples": max_examples,
        "results": [],
    }

    baseline_model = args.baseline_model or config.training.model_name
    print(f"\nBaseline model: {baseline_model}", flush=True)

    baseline_result = evaluate_model(
        "baseline",
        baseline_model,
        dataset,
        batch_size=args.batch_size,
        max_examples=max_examples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        predictions_dir=predictions_dir,
    )
    report["results"].append(baseline_result)

    if args.checkpoint:
        print(f"\nCheckpoint model: {args.checkpoint}", flush=True)
        checkpoint_result = evaluate_model(
            "checkpoint",
            args.checkpoint,
            dataset,
            batch_size=args.batch_size,
            max_examples=max_examples,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            predictions_dir=predictions_dir,
        )
        checkpoint_result["checkpoint"] = str(args.checkpoint)
        report["results"].append(checkpoint_result)

    print(f"\nWriting final report to {args.output}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print("\n" + "="*60, flush=True)
    print("EVALUATION COMPLETE", flush=True)
    print("="*60, flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
