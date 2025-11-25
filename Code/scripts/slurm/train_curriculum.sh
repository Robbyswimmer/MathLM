#!/bin/bash -l

#SBATCH --job-name="MathLM-Curriculum"
#SBATCH --output=logs/train_curriculum_%j.txt
#SBATCH --error=logs/train_curriculum_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -eo pipefail

export PYTHONUNBUFFERED=1

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-mathlm}
echo "Activating conda environment '${CONDA_ENV}'"
set +u
conda activate "${CONDA_ENV}"
set -u

echo "Starting MathLM Curriculum Training at $(date)"

DATA_DIR=${DATA_DIR:-"$PWD/data"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/experiments"}
RUN_ID=${RUN_ID:-"curriculum_${SLURM_JOB_ID}"}
INITIAL_CHECKPOINT=${INITIAL_CHECKPOINT:-""}  # Optional: start from existing checkpoint

mkdir -p logs
mkdir -p "${OUTPUT_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}/checkpoints"

echo "Data dir: ${DATA_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run ID: ${RUN_ID}"
echo "Initial checkpoint: ${INITIAL_CHECKPOINT}"

# Stage 1: Easy problems (Difficulty 1-2)
echo ""
echo "=========================================="
echo "STAGE 1: Easy Problems (Difficulty 1-2)"
echo "=========================================="
if [[ -n "${INITIAL_CHECKPOINT}" ]]; then
  echo "Resuming from initial checkpoint: ${INITIAL_CHECKPOINT}"
  python scripts/run_grpo_train.py \
    --config configs/grpo_curriculum_stage1.yaml \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}" \
    --run-id "${RUN_ID}" \
    --resume-from-checkpoint "${INITIAL_CHECKPOINT}"
else
  python scripts/run_grpo_train.py \
    --config configs/grpo_curriculum_stage1.yaml \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}" \
    --run-id "${RUN_ID}"
fi

# Stage 2: Easy to Medium (Difficulty 1-3)
# Resume from final checkpoint of Stage 1
STAGE1_CHECKPOINT=$(ls -td "${OUTPUT_ROOT}/checkpoints/${RUN_ID}/checkpoint-"* 2>/dev/null | head -1 | xargs basename)
echo ""
echo "=========================================="
echo "STAGE 2: Easy to Medium (Difficulty 1-3)"
echo "Resuming from: ${STAGE1_CHECKPOINT}"
echo "=========================================="
python scripts/run_grpo_train.py \
  --config configs/grpo_curriculum_stage2.yaml \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --resume-from-checkpoint "${STAGE1_CHECKPOINT}"

# Stage 3: Easy to Hard (Difficulty 1-4)
STAGE2_CHECKPOINT=$(ls -td "${OUTPUT_ROOT}/checkpoints/${RUN_ID}/checkpoint-"* 2>/dev/null | head -1 | xargs basename)
echo ""
echo "=========================================="
echo "STAGE 3: Easy to Hard (Difficulty 1-4)"
echo "Resuming from: ${STAGE2_CHECKPOINT}"
echo "=========================================="
python scripts/run_grpo_train.py \
  --config configs/grpo_curriculum_stage3.yaml \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --resume-from-checkpoint "${STAGE2_CHECKPOINT}"

# Stage 4: All Problems (Difficulty 1-5)
STAGE3_CHECKPOINT=$(ls -td "${OUTPUT_ROOT}/checkpoints/${RUN_ID}/checkpoint-"* 2>/dev/null | head -1 | xargs basename)
echo ""
echo "=========================================="
echo "STAGE 4: All Problems (Difficulty 1-5)"
echo "Resuming from: ${STAGE3_CHECKPOINT}"
echo "=========================================="
python scripts/run_grpo_train.py \
  --config configs/grpo_curriculum_stage4.yaml \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --resume-from-checkpoint "${STAGE3_CHECKPOINT}"

echo ""
echo "Curriculum training complete at $(date)"
