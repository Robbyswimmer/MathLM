#!/bin/bash -l

#SBATCH --job-name="MathLM-GRPO-Train"
#SBATCH --output=logs/train_grpo_%j.txt
#SBATCH --error=logs/train_grpo_%j.err
#SBATCH --time=48:00:00
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

echo "Starting MathLM GRPO training at $(date)"

CONFIG_PATH=${CONFIG_PATH:-"configs/grpo_base.yaml"}
DATA_DIR=${DATA_DIR:-"$PWD/data"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/experiments"}
RUN_ID=${RUN_ID:-"grpo_${SLURM_JOB_ID}"}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-""}

mkdir -p logs
mkdir -p "${OUTPUT_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}/checkpoints"

echo "Config: ${CONFIG_PATH}"
echo "Data dir: ${DATA_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run ID: ${RUN_ID}"
echo "Resume checkpoint: ${RESUME_CHECKPOINT}"

# Single GPU training with GRPO
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  python scripts/run_grpo_train.py \
    --config "${CONFIG_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}" \
    --run-id "${RUN_ID}" \
    --resume-from-checkpoint "${RESUME_CHECKPOINT}"
else
  python scripts/run_grpo_train.py \
    --config "${CONFIG_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}" \
    --run-id "${RUN_ID}"
fi

echo "GRPO training complete at $(date)"
