#!/bin/bash -l

#SBATCH --job-name="MathLM-Eval-Baseline"
#SBATCH --output=logs/eval_baseline_%j.txt
#SBATCH --error=logs/eval_baseline_%j.err
#SBATCH --time=08:00:00
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

echo "Starting MathLM GSM8k baseline eval at $(date)"

CONFIG_PATH=${CONFIG_PATH:-"configs/baseline_zero_shot.yaml"}
DATA_DIR=${DATA_DIR:-"$PWD/data"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/experiments/logs"}
TRACE_ROOT=${TRACE_ROOT:-"$PWD/experiments/traces"}
RUN_ID=${RUN_ID:-"baseline_eval_${SLURM_JOB_ID}"}
BASELINE_MODEL=${BASELINE_MODEL:-"gemma-2-2b-instruct"}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_EXAMPLES=${MAX_EXAMPLES:-1000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
DEVICE=${DEVICE:-cuda}

RUN_LOG_DIR="${OUTPUT_ROOT}/${RUN_ID}"
PRED_DIR="${TRACE_ROOT}/${RUN_ID}"
OUTPUT_JSON="${RUN_LOG_DIR}/eval_baseline.json"

mkdir -p logs
mkdir -p "${RUN_LOG_DIR}"
mkdir -p "${PRED_DIR}"

echo "Config: ${CONFIG_PATH}"
echo "Dataset dir: ${DATA_DIR}"
echo "Logs -> ${RUN_LOG_DIR}"
echo "Predictions -> ${PRED_DIR}"

python scripts/run_eval.py \
  --config "${CONFIG_PATH}" \
  --output "${OUTPUT_JSON}" \
  --data-dir "${DATA_DIR}" \
  --baseline-model "${BASELINE_MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --max-examples "${MAX_EXAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --device "${DEVICE}" \
  --predictions-dir "${PRED_DIR}"

echo "Baseline evaluation complete at $(date)"
