#!/bin/bash
#===============================================================================
# HPCC Environment Setup Script
#
# Run this ONCE on the HPCC login node after transferring code.
# Prerequisites: data already transferred to /mnt/scratch/kwonjoon/
#
# Usage:
#   bash setup_hpcc_env.sh
#===============================================================================

set -e

echo "=========================================="
echo "HPCC Environment Setup for addb2skel"
echo "=========================================="

# ---- Module setup ----
ml purge
ml Miniforge3

# ---- Create conda environment ----
echo ""
echo "[1/5] Creating conda environment 'physpt'..."
conda create -n physpt python=3.10 -y
conda activate physpt

# ---- Check CUDA version ----
echo ""
echo "[2/5] Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "CUDA Version: ${CUDA_VER}"
else
    echo "WARNING: nvidia-smi not available (run on GPU node for GPU support)"
fi

# ---- Install PyTorch ----
echo ""
echo "[3/5] Installing PyTorch + dependencies..."
# Try CUDA 11.8 first (widely supported on HPCC V100/A100)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
    pip install torch torchvision

# Core dependencies
pip install numpy scipy trimesh matplotlib opencv-python smplx tqdm

# nimblephysics (critical for .b3d loading)
pip install nimblephysics

# ---- Install SKEL ----
echo ""
echo "[4/5] Installing SKEL body model..."
SKEL_DIR="/mnt/scratch/kwonjoon/Code/SKEL"
if [ -d "${SKEL_DIR}" ]; then
    cd "${SKEL_DIR}"
    pip install -e .
    echo "  SKEL installed from ${SKEL_DIR}"
else
    echo "  WARNING: SKEL not found at ${SKEL_DIR}"
    echo "  Transfer it first: rsync -avz /egr/.../SKEL/ kwonjoon@hpcc.msu.edu:${SKEL_DIR}/"
fi

# ---- Verify setup ----
echo ""
echo "[5/5] Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import nimblephysics
print('nimblephysics: OK')

import numpy, scipy, trimesh
print(f'numpy: {numpy.__version__}, scipy: {scipy.__version__}')

try:
    from skel.skel_model import SKEL
    print('SKEL: OK')
except ImportError as e:
    print(f'SKEL: FAILED ({e})')
"

# ---- Check paths ----
echo ""
echo "Checking data paths..."
for p in \
    "/mnt/scratch/kwonjoon/AddB/train" \
    "/mnt/scratch/kwonjoon/skel_models_v1.1" \
    "/mnt/scratch/kwonjoon/Code/Physica/addb2skel"; do
    if [ -d "$p" ]; then
        echo "  OK: $p"
    else
        echo "  MISSING: $p"
    fi
done

echo ""
echo "=========================================="
echo "Setup complete! Next steps:"
echo "  1. Generate manifest:  python generate_trial_manifest.py --input_dir /mnt/scratch/kwonjoon/AddB/train --output trial_manifest.csv"
echo "  2. Test with 3 trials: sbatch --array=0-2 run_hpcc_batch.sh"
echo "  3. Full run:           see run_hpcc_batch.sh header for batch submission commands"
echo "=========================================="
