# MathLM Codebase – GRPO for Small-Scale Mathematical Reasoning

This directory contains the implementation for **MathLM**, a project that studies whether reinforcement learning with dense, process-based rewards can reliably improve mathematical reasoning in a compact language model. We fine-tune **Gemma-2-2B-Instruct** on **GSM8K** using **Group-Relative Policy Optimization (GRPO)** with curriculum learning, self-consistency, and sandboxed Python code verification.

## Project Summary

- **Task**: GSM8K grade-school math word problems (exact-match accuracy).
- **Base model**: Gemma-2-2B-Instruct (general-purpose).
- **Methods**:
  - GRPO with KL-penalized updates and multiple generations per prompt.
  - Dense rewards for explicit reasoning steps, arithmetic work, and executable Python code that verifies the answer.
  - Self-consistency via majority vote over multiple completions per problem.
  - Curriculum training over problem difficulty scores.
- **Key result**: A supervised baseline reaches \~14% exact-match accuracy on GSM8K, whereas our best GRPO configuration with dense process rewards and code verification attains \~38% test accuracy, more than doubling performance. PPO baselines are notably unstable and can collapse toward \~2% accuracy, underscoring the importance of GRPO and reward design.

## Repository Layout

This `Code/` tree is self-contained for the project submission:

- `src/mathlm/`
  - `data/`: GSM8K loaders, curriculum filtering, and difficulty annotation.
  - `prompts/`: zero-shot and code-assisted prompt templates aligned with the reward model.
  - `rewards/`: sandboxed Python executor, dense reward calculator, and GRPO reward functions.
  - `training/`: PPO runner used for early baselines and logging utilities.
  - `evaluation/`: evaluation helpers and plotting support.
  - `utils/`: configuration, logging, and general-purpose helpers.
- `scripts/`
  - `prepare_data.py`: download/convert GSM8K and build processed splits.
  - `run_grpo_train.py`: launch GRPO training with configurable reward functions.
  - `run_eval.py`, `run_eval_checkpoint.py`: evaluate checkpoints on GSM8K.
  - `make_plots.py`: generate learning-curve figures used in the report.
- `configs/`
  - `grpo_base.yaml`: base GRPO configuration.
  - `grpo_curriculum*.yaml`: curriculum-only stages.
  - `grpo_process*.yaml`: dense process-reward and code-verification runs.
- `data/`
  - `raw/`, `processed/`: local GSM8K cache created by `prepare_data.py` (not tracked in version control).
- `docs/`
  - `report/`: IEEE-style report sources and figures referenced in the submission.
  - `roadmap.md`, `repo_structure.md`: project planning notes kept for reproducibility.
- `tests/`
  - Unit tests for the reward calculator, sandbox, and prompt templates.
- `Plan/`
  - Roadmap and repository-organization documents used during development.

## Running the Core Experiment

Typical end-to-end workflow (launched from this `Code/` directory):

1. **Create and activate the environment**

   ```bash
   conda env create -f environment.yml
   conda activate mathlm
   pip install -e .
   ```

2. **Prepare GSM8K data**

   ```bash
   python scripts/prepare_data.py --data-dir data
   ```

3. **Train with GRPO and dense process rewards**

   ```bash
   python scripts/run_grpo_train.py \
     --config configs/grpo_process_rewards.yaml \
     --data-dir data \
     --output-dir experiments
   ```

4. **Evaluate a checkpoint**

   ```bash
   python scripts/run_eval_checkpoint.py \
     --checkpoint-path experiments/checkpoints/<run_id>/checkpoint-XXXX
   ```

Full methodological details, ablations, and result figures are documented in the accompanying IEEE-style report under `docs/report`.
