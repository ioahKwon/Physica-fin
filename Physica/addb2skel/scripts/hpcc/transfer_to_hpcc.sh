#!/bin/bash
#===============================================================================
# Data Transfer: domogpu → HPCC
#
# Run from domogpu. Transfers AddB dataset, SKEL models, and code to HPCC.
# Uses rsync with resume support (--partial).
#
# Usage:
#   # Dry run (check what would be transferred):
#   bash transfer_to_hpcc.sh --dry-run
#
#   # Actual transfer:
#   bash transfer_to_hpcc.sh
#
# If rsync fails (firewall), use OneDrive+rclone as fallback (see comments).
#===============================================================================

HPCC_USER="kwonjoon"
HPCC_HOST="hpcc.msu.edu"
# Data + code → domolab (permanent), output → scratch (temporary)
HPCC_DATA="/mnt/research/domolab/${HPCC_USER}"
HPCC_SCRATCH="/mnt/scratch/${HPCC_USER}"

RSYNC_OPTS="-avz --partial --progress --human-readable"

# Check for --dry-run flag
if [ "$1" == "--dry-run" ]; then
    RSYNC_OPTS="${RSYNC_OPTS} --dry-run"
    echo "*** DRY RUN MODE ***"
fi

echo "=========================================="
echo "Data Transfer: domogpu → HPCC"
echo "Destination: ${HPCC_USER}@${HPCC_HOST}:${HPCC_DEST}/"
echo "=========================================="

# ---- 1. AddB Dataset (391GB) ----
echo ""
echo "[1/4] Transferring AddB dataset (391GB)..."
echo "  This may take several hours."
rsync ${RSYNC_OPTS} \
    /egr/research-zijunlab/kwonjoon/02_Dataset/AddB/ \
    ${HPCC_USER}@${HPCC_HOST}:${HPCC_DATA}/AddB/

RC=$?
if [ $RC -eq 23 ]; then
    echo ""
    echo "WARNING: rsync code 23 — some files not transferred (likely corrupted 0-byte files). Continuing."
elif [ $RC -ne 0 ]; then
    echo ""
    echo "ERROR: rsync failed (code $RC). Possible causes: firewall, authentication, disk space."
    exit 1
fi

# ---- 2. SKEL Models (176MB) ----
echo ""
echo "[2/4] Transferring SKEL models (176MB)..."
rsync ${RSYNC_OPTS} \
    /egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1/ \
    ${HPCC_USER}@${HPCC_HOST}:${HPCC_DATA}/skel_models_v1.1/

# ---- 3. Code: Physica (addb2skel) ----
echo ""
echo "[3/4] Transferring Physica code..."
rsync ${RSYNC_OPTS} \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    /egr/research-zijunlab/kwonjoon/01_Code/Physica/ \
    ${HPCC_USER}@${HPCC_HOST}:${HPCC_DATA}/Code/Physica/

# ---- 4. Code: SKEL ----
echo ""
echo "[4/4] Transferring SKEL code..."
rsync ${RSYNC_OPTS} \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    /egr/research-zijunlab/kwonjoon/01_Code/SKEL/ \
    ${HPCC_USER}@${HPCC_HOST}:${HPCC_DATA}/Code/SKEL/

echo ""
echo "=========================================="
echo "Transfer complete!"
echo ""
echo "Next steps on HPCC:"
echo "  ssh hpcc"
echo "  cd ${HPCC_DATA}/Code/Physica/addb2skel/scripts/hpcc/"
echo "  bash setup_hpcc_env.sh"
echo "=========================================="
