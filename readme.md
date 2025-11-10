# MathLM – Reinforcement Learning for Small-Scale Mathematical Reasoning

This repository hosts all coursework and implementation assets for our CS258/EE227 final project, **MathLM**, where we explore whether reinforcement learning with dense intermediate rewards can significantly improve the mathematical reasoning ability of compact language models.

## Project Snapshot
- **Model**: Gemma-2-2B-Instruct
- **Domain**: GSM8k grade-school math word problems
- **RL Method**: PPO (via TRL) with curriculum learning and structured rewards for intermediate reasoning steps
- **Goal**: Increase exact-match accuracy from ~25% (zero-shot) to 45–55% while keeping compute within a single 48 GB GPU node

## Repository Layout
- `Assignment/` – Course-facing materials: proposal, rubric, project guidelines, and planning documents.
- `Code/` – Full MathLM codebase, configs, experiments, documentation copies, and dev logs.

See `Code/README.md` for the engineering layout and setup instructions, and `Assignment/Code/Plan/roadmap.md` for the project timeline and milestones.
