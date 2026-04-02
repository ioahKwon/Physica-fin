#!/bin/bash
#===============================================================================
# SLURM Job Array: addb2skel full dataset processing
#
# Usage:
#   # Generate manifest first (on login node):
#   python generate_trial_manifest.py --input_dir /mnt/scratch/kwonjoon/AddB/train --output trial_manifest.csv
#
#   # Submit in batches (SLURM array limit = 10,000):
#   sbatch --array=0-9999%200     run_hpcc_batch.sh   # batch 1
#   sbatch --array=10000-19999%200 run_hpcc_batch.sh   # batch 2
#   sbatch --array=20000-29999%200 run_hpcc_batch.sh   # batch 3
#   sbatch --array=30000-38221%200 run_hpcc_batch.sh   # batch 4
#
#   # Monitor:
#   squeue -u kwonjoon
#   sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
#===============================================================================

#SBATCH --job-name=addb2skel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=v100:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/research/domolab/kwonjoon/Code/Physica/addb2skel/scripts/hpcc/logs/addb2skel_%A_%a.out
#SBATCH --error=/mnt/research/domolab/kwonjoon/Code/Physica/addb2skel/scripts/hpcc/logs/addb2skel_%A_%a.err

# ---- Environment Setup ----
ml purge
ml Miniforge3
conda activate physpt

# ---- Paths (adjust for your HPCC layout) ----
SCRIPT_DIR="/mnt/research/domolab/kwonjoon/Code/Physica/addb2skel/scripts/hpcc"
cd "${SCRIPT_DIR}"
MANIFEST="${SCRIPT_DIR}/trial_manifest.csv"
OUTPUT_DIR="/mnt/scratch/kwonjoon/output/addb2skel"
SKEL_MODEL_PATH="/mnt/research/domolab/kwonjoon/skel_models_v1.1"

# Create log directory
mkdir -p logs

# ---- Run ----
echo "=========================================="
echo "SLURM Job: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Time: $(date)"
echo "=========================================="

python "${SCRIPT_DIR}/run_single_trial.py" \
    --manifest "${MANIFEST}" \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    --output_dir "${OUTPUT_DIR}" \
    --skel_model_path "${SKEL_MODEL_PATH}" \
    --device cuda \
    --target_fps 100.0

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
echo "Finished: $(date)"
exit ${EXIT_CODE}
