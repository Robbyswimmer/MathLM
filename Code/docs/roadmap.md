# MathLM Reinforcement Learning Roadmap

## 1. Project Overview (project_proposal.txt)
- **Goal**: Improve mathematical reasoning of Gemma-2-2B-Instruct on GSM8k via PPO fine-tuning with dense intermediate rewards while keeping compute within a single 48GB GPU node.
- **Motivation**: Demonstrate that carefully shaped rewards and curriculum learning can close the accuracy gap between small general-purpose LLMs (~25% GSM8k) and large math-specialized models without massive pretraining.
- **Target outcome**: Achieve 45–55% GSM8k accuracy (+20–30% absolute) and provide reproducible evidence of the training recipe, reward design, and stability controls.

## 2. Course Requirements (assignment_description.md & proposal_rubric.md)
- **Team size**: 1–3 students; project must emphasize a clear RL concept (policy optimization with dense rewards and curriculum in our case).
- **Proposal (due 11/06)**: Submitted via Canvas; lays out scope, RL principles, data, resources (complete).
- **Presentation video (due 12/05 @ noon)**: 5–7 minute recording played during the 12/06 exam slot; should cover motivation, method, results, demo/visuals, and lessons learned.
- **Final submission (due 12/10)**: Canvas upload containing
  - IEEE-format report (Introduction, Methodology, Results, Code Status & Resources, Conclusion/Future Work)
  - Codebase + instructions
  - Trained checkpoints, evaluation artifacts, and any visualizations/animations referenced
- **Expectations**: Originality encouraged; acceptable themes include curriculum learning, RLHF, etc.; staff can help scope if needed.

## 3. MathLM Scope & Methodology Details
### 3.1 Data & Prompting
- GSM8k training set (8.5k problems) with 1.3k test set for held-out evaluation.
- Structured prompting template requiring the model to emit reasoning text plus executable Python blocks for numeric verification.

### 3.2 Reward Shaping (dense feedback channel)
- +0.1 syntactically valid Python block
- +0.2 successful execution in sandbox
- +0.3 correct numeric extraction / intermediate quantity
- +0.5 partially correct reasoning or intermediate answer
- +2.0 exact final answer match
- −0.5 incorrect final result or detected hallucination/unsafe code path

### 3.3 PPO Training Stack
- TRL `PPOTrainer` with Gemma-2-2B base; Hugging Face + PyTorch runtime.
- KL-penalty to anchor policy to pretrained model; entropy bonus for exploration; gradient clipping for stability.
- Curriculum schedule: Weeks 1–2 train on easiest 1k GSM8k examples; Weeks 3–4 scale to full set with enhanced rewards; optional stretch: MATH dataset (Lv1–2) or Phi-3.5-mini if resources permit.
- Logging: capture rewards, KL divergence, accuracy, and verifier stats each batch; checkpoint best policy regularly.

### 3.4 Evaluation Protocol
- Baselines: zero-shot Gemma-2-2B (~25%), 5-shot prompting (~30–35%).
- Main metric: GSM8k exact-match accuracy; report learning curves (accuracy vs PPO steps); include intermediate-step correctness and KL stability plots.
- Qualitative analysis: highlight representative improvements/failures; inspect sandbox logs for reward hacking.

## 4. Deliverables Breakdown
1. **Code**: Clean repository with data preprocessing scripts, verifier, PPO training loop, evaluation pipeline, configuration files, and README instructions.
2. **Models**: Final PPO-tuned checkpoints (policy + value/critic weights) with hash/version metadata.
3. **Artifacts**: Figures (learning curves, accuracy comparisons), tables (hyperparameters, compute budget), verifier logs, optional demo clips of reasoning traces.
4. **Report**: IEEE template, covering motivation, method (reward shaping, curriculum, PPO configuration), experimental results, limitations, future work.
5. **Presentation Video**: Slides + narration summarizing the above, highlighting dense reward insights and curriculum impact.

## 5. Integrated Timeline & Milestones
| Date | Milestone |
| --- | --- |
| 11/06 | Proposal submitted (complete). |
| 11/10 | Finalize verifier design, sandbox safety checks, and GSM8k data pipeline. |
| 11/15 | Integrate PPOTrainer with Gemma checkpoints; run smoke tests on toy subset. |
| 11/22 | Curriculum Phase 1 complete (1k easy problems); analyze reward/accuracy trends, tune KL + entropy coefficients. |
| 11/29 | Curriculum Phase 2 (full GSM8k) underway; generate baseline comparisons; decide on stretch goals. |
| 12/03 | Freeze best model(s); run full evaluation on GSM8k test set; capture qualitative analyses. |
| 12/05 | Record and submit presentation video. |
| 12/08 | Draft IEEE report, insert plots/tables, document code status/resources. |
| 12/10 | Final Canvas submission: report, code, models, artifacts, documentation. |

## 6. Workstreams & Ownership
- **Verification & Rewards**: Build sandbox executor, static analysis to block unsafe code, reward calculator, intermediate logging dashboard.
- **Training Infrastructure**: Dataset loaders, tokenizer prep, TRL PPO config, curriculum scheduler, checkpointing logic.
- **Evaluation & Analytics**: Baseline runs, GSM8k accuracy scripts, intermediate-metric aggregation, figure generation.
- **Documentation & Deliverables**: Dev logs, README updates, presentation storyboard, IEEE report assembly.

## 7. Risks & Mitigations
- **Compute/time overruns**: 2B PPO is expensive → stage runs (toy → 1k subset → full), monitor GPU utilization, checkpoint frequently to allow resume.
- **Reward hacking**: Model may exploit formatting rewards → add semantic checks, penalize inconsistent reasoning, manually inspect samples weekly.
- **Training instability**: PPO collapse due to large updates → enforce KL target, adaptive learning rates, gradient clipping, early stopping if KL spikes.
- **Timeline compression**: Multiple deliverables near end of quarter → lock verifier + infra early (by 11/15) to devote late November to experimentation and early December to polishing.

## 8. Success Criteria
- ≥45% GSM8k test accuracy (absolute gain ≥20%) with documented reproducible recipe.
- Clear empirical evidence that dense rewards + curriculum outperform baselines (plots + ablation notes).
- Complete deliverable package meeting course guidelines: IEEE report, ready-to-run code, checkpoints, and engaging presentation showcasing insights and future directions.
