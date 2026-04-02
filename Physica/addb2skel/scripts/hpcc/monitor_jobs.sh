#!/bin/bash
#===============================================================================
# Real-time monitoring for addb2skel HPCC batch jobs
#
# Usage (from domogpu):
#   bash monitor_jobs.sh              # refresh every 30s
#   bash monitor_jobs.sh 10           # refresh every 10s
#   bash monitor_jobs.sh 0            # run once
#===============================================================================

INTERVAL=${1:-30}

run_once() {
    # Metrics summary (runs on dev node for scratch access)
    ssh -q hpcc "ssh -q dev-amd24-h200 'source /etc/profile.d/modules.sh 2>/dev/null; \
        module purge 2>/dev/null; module load Miniforge3 2>/dev/null; \
        source activate physpt 2>/dev/null && \
        cd /mnt/research/domolab/kwonjoon/Code/Physica/addb2skel/scripts/hpcc && \
        python monitor_jobs.py --interval 0'" 2>/dev/null

    echo ""
    echo "  ┌──────────────────────────────────────────────────────────────────┐"
    echo "  │  SLURM Queue                                                    │"
    echo "  └──────────────────────────────────────────────────────────────────┘"

    # Job summary
    RUNNING=$(ssh -q hpcc "squeue -u kwonjoon -t RUNNING -h 2>/dev/null | wc -l")
    PENDING=$(ssh -q hpcc "squeue -u kwonjoon -t PENDING -h 2>/dev/null | wc -l")
    echo "    Running: ${RUNNING:-0}    Pending: ${PENDING:-0}"
    echo ""

    # Show a few running jobs
    ssh -q hpcc "squeue -u kwonjoon -o '%.10i %.9P %.8j %.2t %.10M %.6D %R' 2>/dev/null | head -10"
    TOTAL_JOBS=$(ssh -q hpcc "squeue -u kwonjoon -h 2>/dev/null | wc -l")
    if [ "${TOTAL_JOBS:-0}" -gt 9 ]; then
        echo "    ... and $((TOTAL_JOBS - 9)) more jobs"
    fi
}

if [ "$INTERVAL" = "0" ]; then
    run_once
else
    echo "Monitoring every ${INTERVAL}s... (Ctrl+C to stop)"
    sleep 1
    while true; do
        clear
        run_once
        echo ""
        echo "  Next refresh in ${INTERVAL}s... (Ctrl+C to stop)"
        sleep "$INTERVAL"
    done
fi
