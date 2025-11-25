# Curriculum Training with Self-Consistency

This guide explains the new curriculum learning and self-consistency features.

## What's New?

### 1. **Self-Consistency Reward** (+5-8% expected improvement)
- Generates N=8 completions per prompt (already configured)
- Uses **majority voting** to reward consistent answers
- Bonuses:
  - +2.0 for matching majority AND being correct
  - +0.5 for matching majority (even if wrong)
- Encourages confident, reproducible reasoning

### 2. **Curriculum Learning** (+3-5% expected improvement)
- Trains on easy problems first, gradually increasing difficulty
- Automatic difficulty scoring based on:
  - Number of reasoning steps
  - Arithmetic operations count
  - Problem complexity
- 4 stages: Easy (1-2) → Medium (1-3) → Hard (1-4) → All (1-5)

### Combined Expected Improvement: **~45% accuracy** (from 33.4%)

---

## Quick Start

### Option 1: Full Curriculum (Recommended)

Run all 4 stages sequentially (~40k total steps):

```bash
sbatch scripts/slurm/train_curriculum.sh
```

This will:
1. Train on easy problems (difficulty 1-2) for 10k steps
2. Add medium problems (difficulty 1-3) for 10k steps
3. Add hard problems (difficulty 1-4) for 10k steps
4. Train on all problems (difficulty 1-5) for 10k steps

### Option 2: Individual Stages

Train a specific stage:

```bash
# Stage 1: Easy problems only
sbatch --export=CONFIG_PATH="configs/grpo_curriculum_stage1.yaml",RUN_ID="curriculum_stage1" scripts/slurm/train_grpo.sh

# Stage 2: Resume from stage 1 checkpoint
sbatch --export=CONFIG_PATH="configs/grpo_curriculum_stage2.yaml",RUN_ID="curriculum_stage1",RESUME_CHECKPOINT="checkpoint-10000" scripts/slurm/train_grpo.sh
```

### Option 3: Self-Consistency Only (No Curriculum)

Just enable self-consistency on existing training:

```bash
# Add to your config YAML:
training:
  use_self_consistency: true

# Then run normally
sbatch scripts/slurm/train_grpo.sh
```

---

## Configuration Details

### Curriculum Stages

Each stage config specifies a difficulty range:

| Stage | Config | Difficulty | ~Examples | Steps |
|-------|--------|-----------|-----------|-------|
| 1 | `grpo_curriculum_stage1.yaml` | 1-2 | ~2000 | 10k |
| 2 | `grpo_curriculum_stage2.yaml` | 1-3 | ~4000 | 10k |
| 3 | `grpo_curriculum_stage3.yaml` | 1-4 | ~6000 | 10k |
| 4 | `grpo_curriculum_stage4.yaml` | 1-5 | ~7500 | 10k |

### Difficulty Levels

Problems are automatically scored 1-5:

- **1 (Simple)**: 1-2 operations (e.g., "5 + 3 = ?")
- **2 (Easy)**: 3-4 operations (e.g., "5 + 3 - 2 = ?")
- **3 (Medium)**: 5-6 operations (e.g., multi-step word problems)
- **4 (Hard)**: 7-8 operations (e.g., complex multi-step)
- **5 (Very Hard)**: 9+ operations (e.g., advanced reasoning)

### Self-Consistency Parameters

In `grpo_rewards.py`:

```python
def self_consistency_reward(
    num_generations: int = 8,  # Number of completions per prompt
    consensus_threshold: float = 0.5,  # 50% agreement required
):
    # Bonuses:
    # +2.0 if majority answer is correct
    # +0.5 if answer matches majority (even if wrong)
```

---

## Monitoring Progress

### Check Training Logs

```bash
# Watch curriculum training
tail -f logs/train_curriculum_*.txt

# Check rewards
grep "rewards/combined_math_reward" logs/train_curriculum_*.txt
```

### Evaluate Checkpoints

After each stage, evaluate on test set:

```bash
python scripts/run_eval_checkpoint.py \
  --checkpoint experiments/checkpoints/curriculum_12345/checkpoint-10000 \
  --split test \
  --output results_stage1.json
```

### Expected Progression

| Checkpoint | Stage | Difficulty | Expected Accuracy |
|-----------|-------|-----------|------------------|
| 10k | Stage 1 | 1-2 only | 70-80% (on easy) |
| 20k | Stage 2 | 1-3 | 50-55% (on all) |
| 30k | Stage 3 | 1-4 | 42-47% (on all) |
| 40k | Stage 4 | 1-5 | 43-48% (on all) |

---

## Troubleshooting

### Issue: Difficulty scores not appearing

Check if examples are being annotated:

```python
from mathlm.data import load_raw_split, annotate_difficulty

examples = load_raw_split("data/raw/gsm8k_train.jsonl")
examples = annotate_difficulty(examples)
print(examples[0].difficulty)  # Should print "1", "2", etc.
```

### Issue: Self-consistency not improving

Check that `num_generations=8` in your config:

```yaml
training:
  num_generations: 8  # Required for self-consistency
```

### Issue: Resume checkpoint not found

Ensure RUN_ID matches across stages:

```bash
# List checkpoints
ls experiments/checkpoints/curriculum_12345/

# Use exact checkpoint name
--resume-from-checkpoint checkpoint-10000
```

---

## Comparing to Baseline

### Before (Baseline):
- Reward: Final answer correctness only
- Training: Random problem order
- Accuracy: 33.4% @ 10k steps → **25% @ 20k steps** (degraded!)

### After (Curriculum + Self-Consistency):
- Reward: Correctness + consistency bonus
- Training: Easy → Hard progression
- Expected: **43-48% @ 40k steps** (stable improvement)

### Why It Works Better:

1. **Self-consistency prevents reward gaming**
   - Model can't get lucky on single sample
   - Must produce consistent reasoning

2. **Curriculum builds skills incrementally**
   - Learns basic arithmetic before complex problems
   - Prevents early policy collapse

3. **Combined effect is multiplicative**
   - Better reward signal × Better training order = Bigger gains

---

## Next Steps

After curriculum training, consider:

1. **Code execution** (+10-15%): Let model write/run Python
2. **Process rewards** (+10-15%): Reward intermediate steps
3. **Larger model** (+20%): Scale to 7B parameters

See main README for advanced techniques.
