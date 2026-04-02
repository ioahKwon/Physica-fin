#!/bin/bash
#===============================================================================
# SLURM Job Array: addb2skel — With_Arm trials only (13,206 trials)
#
# Uses with_arm_task_ids.txt to map SLURM array index → manifest task_id.
#
# Strategy: use BOTH general + scavenger accounts to maximize GPU usage.
#   general:   max 1,000 submitted, 520 running
#   scavenger: max 5,000 submitted, 5,000 running (preemptible)
#
# Usage:
#   # general account (reliable, up to 1000):
#   sbatch --account=general   --array=0-999%500   run_with_arm_batch.sh
#
#   # scavenger account (preemptible, up to 5000):
#   sbatch --account=scavenger --partition=scavenger --array=1000-5999%1000  run_with_arm_batch.sh
#   sbatch --account=scavenger --partition=scavenger --array=6000-10999%1000 run_with_arm_batch.sh
#   sbatch --account=scavenger --partition=scavenger --array=11000-13205%1000 run_with_arm_batch.sh
#
#   # Monitor from domogpu:
#   bash monitor_jobs.sh
#===============================================================================

#SBATCH --job-name=addb2skel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/research/domolab/kwonjoon/Code/Physica/addb2skel/scripts/hpcc/logs/addb2skel_%A_%a.out
#SBATCH --error=/mnt/research/domolab/kwonjoon/Code/Physica/addb2skel/scripts/hpcc/logs/addb2skel_%A_%a.err

# ---- Environment Setup ----
ml purge
ml Miniforge3
conda activate physpt

# ---- Paths ----
SCRIPT_DIR="/mnt/research/domolab/kwonjoon/Code/Physica/addb2skel/scripts/hpcc"
cd "${SCRIPT_DIR}"
export PYTHONPATH="/mnt/research/domolab/kwonjoon/Code/Physica:/mnt/research/domolab/kwonjoon/Code/SKEL"
MANIFEST="${SCRIPT_DIR}/with_arm_manifest.csv"
OUTPUT_DIR="/mnt/scratch/kwonjoon/output/addb2skel_fixK"
SKEL_MODEL_PATH="/mnt/research/domolab/kwonjoon/skel_models_v1.1"

mkdir -p logs

# ---- SLURM array index = manifest task_id (direct mapping) ----
TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}, Task ID: ${TASK_ID}"
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Time: $(date)"
echo "=========================================="

python "${SCRIPT_DIR}/run_single_trial.py" \
    --manifest "${MANIFEST}" \
    --task_id "${TASK_ID}" \
    --output_dir "${OUTPUT_DIR}" \
    --skel_model_path "${SKEL_MODEL_PATH}" \
    --device cuda \
    --target_fps 100.0

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
echo "Finished: $(date)"
exit ${EXIT_CODE}
