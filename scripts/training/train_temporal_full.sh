#!/bin/bash
# Train temporal_full (residual) model from scratch with full STA dataset (5446 samples)

set -e

PROJECT_ROOT="/media/eamurciab/TOSHIBA_EXT1/PhD/SR"
LOG_DIR="${PROJECT_ROOT}/logs"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts/training/variants"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

cd "${PROJECT_ROOT}"

echo "========================================================================"
echo "TRAINING TEMPORAL FULL (RESIDUAL) - Full STA Dataset"
echo "========================================================================"
echo "Start time: $(date)"
echo "Data: Clean dataset (5446 samples - all STAs)"
echo "========================================================================"

START_TIME=$(date +%s)

if python "${SCRIPTS_DIR}/temporal/train_multires_temporal.py" 2>&1 | tee "${LOG_DIR}/training_temporal_full.log"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    echo "[temporal_full] COMPLETED in ${HOURS}h ${MINUTES}m"
    echo "[temporal_full] COMPLETED in ${HOURS}h ${MINUTES}m at $(date)" >> "${LOG_DIR}/temporal_training.log"
else
    echo "[temporal_full] FAILED - check ${LOG_DIR}/training_temporal_full.log"
    echo "[temporal_full] FAILED at $(date)" >> "${LOG_DIR}/temporal_training.log"
    exit 1
fi

echo ""
echo "========================================================================"
echo "TEMPORAL FULL TRAINING COMPLETED at $(date)"
echo "========================================================================"
