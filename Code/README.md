# MathLM Codebase

This directory houses the implementation for the MathLM project, which fine-tunes Gemma-2-2B-Instruct on GSM8k using PPO with dense intermediate rewards.

## Layout
- `src/mathlm`: Python package containing data loaders, prompt templates, reward logic, PPO training utilities, evaluation helpers, and shared utilities.
- `scripts`: CLI entry points for data prep, training, evaluation, and plotting.
- `configs`: YAML files that describe experiment settings for curriculum phases, baselines, and demos.
- `experiments`: Logs, checkpoints, qualitative traces, and generated figures from each run.
- `data`: Local cache for GSM8k and derived splits (ignored in version control).
- `docs`: Project documentation replicated from the planning folder for self-contained submissions.
- `Dev Logs`: Running notes that capture daily progress.

Refer to `docs/roadmap.md` and `docs/repo_structure.md` for the detailed project plan and repository organization guidelines.
