"""Dense reward computation for MathLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from mathlm.data import GSM8KExample

from .sandbox import PythonSandbox, SandboxResult, extract_python_blocks


@dataclass
class RewardWeights:
    syntax: float = 0.0      # No code, no syntax reward
    execution: float = 0.0   # No code, no execution reward
    extraction: float = 0.5  # Increased weight for finding an answer
    reasoning: float = 0.5   # Reward showing reasoning steps
    exact: float = 2.0       # Reward exact match on final answer
    penalty: float = -0.5    # Penalize incorrect answers


@dataclass
class RewardBreakdown:
    syntax_reward: float = 0.0
    execution_reward: float = 0.0
    extraction_reward: float = 0.0
    reasoning_reward: float = 0.0
    exact_reward: float = 0.0
    penalty: float = 0.0
    repetition_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.syntax_reward
            + self.execution_reward
            + self.extraction_reward
            + self.reasoning_reward
            + self.exact_reward
            + self.penalty
            + self.repetition_penalty
        )


class RewardCalculator:
    def __init__(
        self,
        weights: RewardWeights | None = None,
        sandbox: PythonSandbox | None = None,
        repetition_penalty: float = -2.0,
        repetition_threshold: float = 0.4,
        length_penalty: float = -0.5,
        length_ratio_threshold: float = 1.5,
    ):
        self.weights = weights or RewardWeights()
        self.sandbox = sandbox or PythonSandbox()
        self.repetition_penalty_value = repetition_penalty
        self.repetition_threshold = repetition_threshold
        self.length_penalty_value = length_penalty
        self.length_ratio_threshold = length_ratio_threshold

    def evaluate(self, example: GSM8KExample | str, model_output: str) -> RewardBreakdown:
        breakdown = RewardBreakdown()
        python_blocks = extract_python_blocks(model_output)
        question_text = example.question if hasattr(example, "question") else str(example)
        answer_text = example.answer if hasattr(example, "answer") else None
        syntax_ok = False
        execution_ok = False
        if python_blocks:
            for block in python_blocks:
                try:
                    compile(block, "<model>", "exec")
                    syntax_ok = True
                except SyntaxError:
                    continue
                # Heuristic: If the last line is an expression (not assignment/print), wrap it in print()
                lines = block.strip().split('\n')
                if lines:
                    last_line = lines[-1].strip()
                    if not last_line.startswith("print(") and "=" not in last_line and not last_line.startswith("return") and not last_line.startswith("def ") and not last_line.startswith("#"):
                        lines[-1] = f"print({last_line})"
                        block = "\n".join(lines)
                
                result = self.sandbox.run(block)
                # DEBUG: Print sandbox result
                if not execution_ok: # Only print if we haven't succeeded yet
                     import sys
                     print(f"[SANDBOX DEBUG] Block:\n{block}\n[SANDBOX DEBUG] Success: {result.success}, Stdout: {result.stdout}, Stderr: {result.stderr}", file=sys.stderr, flush=True)

                if result.success:
                    execution_ok = True
                    break
        if syntax_ok:
            breakdown.syntax_reward = self.weights.syntax
        if execution_ok:
            breakdown.execution_reward = self.weights.execution

        extracted_value = extract_final_number(model_output)
        if extracted_value is not None:
            breakdown.extraction_reward = self.weights.extraction
        if has_reasoning_text(model_output):
            breakdown.reasoning_reward = self.weights.reasoning
        if answer_text is not None:
            if extracted_value is not None and answers_match(extracted_value, answer_text):
                breakdown.exact_reward = self.weights.exact
            else:
                breakdown.penalty = self.weights.penalty
        # Repetition penalty: if model repeats prompt content too much
        if is_repetitive(model_output, question_text, threshold=self.repetition_threshold):
            breakdown.repetition_penalty = self.repetition_penalty_value
        # Length penalty: if output is too long relative to prompt
        if is_too_long(model_output, question_text, ratio_threshold=self.length_ratio_threshold):
            breakdown.penalty += self.length_penalty_value
        return breakdown


def extract_final_number(text: str) -> Optional[str]:
    matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    if matches:
        return matches[-1]
    return None


def normalize_answer(ans: str) -> str:
    ans = ans.strip().lower()
    ans = ans.replace(",", "")
    ans = re.sub(r"[^0-9.-]", " ", ans)
    parts = [part for part in ans.split() if part]
    return parts[-1] if parts else ""


def answers_match(predicted: str, ground_truth: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def has_reasoning_text(output: str) -> bool:
    stripped = output.strip()
    lines = [line for line in stripped.splitlines() if line.strip()]
    text_lines = [line for line in lines if not line.strip().startswith("```")]
    joined = " ".join(text_lines)
    return len(joined.split()) >= 10


def is_repetitive(output: str, prompt: str, threshold: float = 0.6) -> bool:
    """Simple repetition detector based on prompt overlap."""
    prompt_tokens = prompt.lower().split()
    output_tokens = output.lower().split()
    if not prompt_tokens or not output_tokens:
        return False
    prompt_set = set(prompt_tokens)
    overlap = sum(1 for tok in output_tokens if tok in prompt_set)
    overlap_ratio = overlap / max(1, len(output_tokens))
    return overlap_ratio >= threshold


def is_too_long(output: str, prompt: str, ratio_threshold: float = 1.5) -> bool:
    """Penalize outputs that are much longer than the prompt."""
    prompt_len = max(1, len(prompt.split()))
    output_len = len(output.split())
    return output_len / prompt_len >= ratio_threshold
