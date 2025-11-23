#!/bin/bash -l

#SBATCH --job-name="MathLM-Eval"
#SBATCH --output=logs/eval_%j.txt
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
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

echo "Starting MathLM checkpoint evaluation at $(date)"

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"experiments/checkpoints/phase1_232680/final"}
DATA_DIR=${DATA_DIR:-"$PWD/data"}
OUTPUT_FILE=${OUTPUT_FILE:-"experiments/eval_results_$(date +%Y%m%d_%H%M%S).json"}

echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Data dir: ${DATA_DIR}"
echo "Output: ${OUTPUT_FILE}"

python scripts/run_eval_checkpoint.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --data-dir "${DATA_DIR}" \
  --split test \
  --output "${OUTPUT_FILE}"

echo "Evaluation complete at $(date)"
