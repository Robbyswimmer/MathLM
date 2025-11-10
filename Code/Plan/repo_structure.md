# MathLM Code & Repository Organization Plan

This document describes how we will lay out every component of the MathLM project so development, experimentation, and deliverables remain structured and reproducible.

## 1. High-Level Goals
- Keep all runnable code, configs, and artifacts under `Code/` while documents supporting the assignment (roadmap, rubric, proposal) remain under `Assignment/`.
- Separate source modules, scripts, configs, experiment outputs, and documentation so collaborators can locate assets quickly and submission packaging is trivial.
- Ensure sensitive/large assets (datasets, checkpoints) are tracked but not committed by default using `.gitignore` patterns.

## 2. Directory Tree (proposed)
```
Code/
├── README.md                  # Project overview, setup, quickstart
├── environment.yml            # Conda env (or requirements.txt as fallback)
├── pyproject.toml / setup.cfg # Optional package metadata & dependency pins
├── src/
│   └── mathlm/
│       ├── __init__.py
│       ├── data/              # GSM8k loaders, curriculum filters, caching
│       ├── prompts/           # Prompt templates, formatting utilities
│       ├── rewards/           # Verifier, sandboxing, dense reward aggregation
│       ├── training/          # PPO loops, schedulers, callbacks, CLI glue
│       ├── evaluation/        # Accuracy scripts, qualitative trace builders, plotting
│       └── utils/             # Config/dataclass helpers, logging, seed control, sandbox helpers
├── scripts/
│   ├── prepare_data.py        # Download/tokenize GSM8k, build splits, log metadata
│   ├── run_train.py           # Launch PPO via config overrides (phase1/phase2)
│   ├── run_eval.py            # Evaluate checkpoints vs GSM8k test & baselines
│   └── make_plots.py          # Generate report/presentation figures
├── configs/
│   ├── base.yaml              # Defaults (model, tokenizer, optimizer, reward weights)
│   ├── curriculum_phase1.yaml
│   ├── curriculum_phase2.yaml
│   ├── baseline_zero_shot.yaml
│   └── demo_presentation.yaml
├── experiments/
│   ├── logs/                  # Run-wise JSONL metrics + resolved configs
│   ├── checkpoints/           # Saved PPO checkpoints (policy, value, tokenizer)
│   ├── traces/                # Sample reasoning traces + verifier transcripts
│   └── figures/               # Learning curves, comparison plots, animations
├── data/                      # Local cache (ignored): raw + processed GSM8k, metadata
│   ├── raw/
│   ├── processed/
│   └── metadata.json
├── docs/
│   ├── roadmap.md             # Copy of Assignment/Code/Plan/roadmap.md
│   ├── repo_structure.md      # This document (kept in sync with Plan copy)
│   ├── decision_logs.md       # ADR-style notes on major choices
│   └── report/                # IEEE draft, figures, tables for submission
├── Dev Logs/                  # Daily progress notes (already present)
└── tests/
    ├── test_verifier.py
    ├── test_reward_model.py
    └── test_prompt_templates.py
```

## 3. Module Boundaries
- `mathlm.data`: deterministic data ingestion, filtering, curriculum splits, dataset version tracking.
- `mathlm.prompts`: central prompt schema so training/evaluation stay consistent; houses template registry for curriculum tiers.
- `mathlm.rewards`: sandbox executor (`sandbox.py`), static safety checks, structured reward assignments, and logging of intermediate signals.
- `mathlm.training`: wrappers around TRL `PPOTrainer`, curriculum scheduler, KL/entropy controllers, gradient clipping, checkpoint logic.
- `mathlm.evaluation`: accuracy calculators, baseline runners, qualitative trace exporters, plotting utilities.
- `mathlm.utils`: configuration dataclasses, logging setup, random seed utilities, general helpers shared across modules.

## 4. Scripts & Automation
- All runnable entry points live under `scripts/` to keep `src/` importable.
- Common CLI arguments: `--config` (YAML), `--run-id`, `--output-dir`. Config resolution handled via Hydra or manual YAML loader with overrides.
- `scripts/prepare_data.py` writes dataset metadata and seeds to `data/metadata.json` for reproducibility.
- `scripts/run_train.py` automatically copies the resolved config + git commit hash to `experiments/logs/<run_id>/`.
- `scripts/run_eval.py` produces `metrics.json` and `qualitative.jsonl` in `experiments/traces/<run_id>/`.

## 5. Configuration Management
- YAML configs live in `configs/`. Each training phase/baseline inherits from `base.yaml`.
- Config files capture: model path, tokenizer path, learning rate, KL penalty, entropy bonus, reward weights, curriculum schedule, dataset filters, logging cadence.
- Hydra-style overrides let us run: `python scripts/run_train.py config=curriculum_phase2 trainer.max_epochs=2`. If Hydra is not used, replicate behavior with manual merging and CLI flags.

## 6. Experiment Logging & Checkpoints
- Each run gets a unique `run_id` (timestamp or short slug). Directory structure: `experiments/logs/<run_id>/` for configs/metrics, `experiments/checkpoints/<run_id>/` for model weights, `experiments/traces/<run_id>/` for sample outputs.
- Track `git rev-parse HEAD`, environment hash, and dataset metadata for every run to ensure reproducibility.
- Optionally mirror metrics to Weights & Biases; treat `experiments/logs/` as authoritative local copy for the final report.

## 7. Data & Secrets
- Add `.gitignore` entries for `data/`, `experiments/checkpoints/`, and `.env`.
- Hugging Face or Google credentials stored via environment variables or `.env` read by `dotenv`. Never committed.

## 8. Testing & Quality Gates
- Use pytest under `tests/` for unit coverage of verifier logic, reward aggregation, prompt formatting, and config loading.
- Add pre-commit hooks (black/ruff/mypy optional) to enforce formatting + lightweight static checks.
- Continuous testing strategy: run `pytest tests/` before major training runs; integrate with CI if repo hosted on GitHub.

## 9. Documentation Workflow
- Keep authoritative planning docs in `Assignment/Code/Plan/`. Mirror them into `Code/docs/` via copy or symlink before submission so the repository is self-contained.
- `docs/decision_logs.md` records significant architectural decisions (e.g., reward weight changes, curriculum adjustments). Use short ADR template: Context → Decision → Consequences.
- Presentation assets (slides, scripts, demo GIFs) stored under `docs/presentation/` with exports mirrored to `experiments/figures/` for reuse.

## 10. Submission Packaging
- Prior to 12/10 submission, run `python scripts/export_bundle.py` (to be written) that copies:
  - `src/`, `scripts/`, `configs/`, `docs/`, minimal `experiments/checkpoints/<best_run>`
  - `README.md`, environment files, and instructions into a `submission/` folder.
- Include `docs/report/MathLM_Final_Report.pdf`, presentation video path, and checksums for checkpoints.

This structure keeps experimentation disciplined while satisfying course deliverables. As the project evolves, update both this document and the actual directory layout together so new contributors or graders can onboard quickly.
