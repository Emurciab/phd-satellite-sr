#!/bin/bash
# Train frequency models from scratch with full STA dataset (5446 samples)

set -e

PROJECT_ROOT="/media/eamurciab/TOSHIBA_EXT1/PhD/SR"
LOG_DIR="${PROJECT_ROOT}/logs"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts/training/variants"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

cd "${PROJECT_ROOT}"

echo "========================================================================"
echo "TRAINING FREQUENCY MODELS - Full STA Dataset"
echo "========================================================================"
echo "Start time: $(date)"
echo "Data: Clean dataset (5446 samples - all STAs)"
echo ""
echo "Models to train:"
echo "  1. freq_residual"
echo "  2. freq_direct"
echo "========================================================================"

# Function to train a variant
train_variant() {
    local name=$1
    local script=$2
    
    echo ""
    echo "========================================================================"
    echo "[$name] Starting at $(date)"
    echo "========================================================================"
    
    START_TIME=$(date +%s)
    
    if python "$script" 2>&1 | tee "${LOG_DIR}/training_${name}.log"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        HOURS=$((DURATION / 3600))
        MINUTES=$(((DURATION % 3600) / 60))
        echo "[$name] COMPLETED in ${HOURS}h ${MINUTES}m"
        echo "[$name] COMPLETED in ${HOURS}h ${MINUTES}m at $(date)" >> "${LOG_DIR}/frequency_training.log"
    else
        echo "[$name] FAILED - check ${LOG_DIR}/training_${name}.log"
        echo "[$name] FAILED at $(date)" >> "${LOG_DIR}/frequency_training.log"
        return 1
    fi
}

# Train frequency models
train_variant "freq_residual" "${SCRIPTS_DIR}/frequency/train_multires_freq_residual.py"
train_variant "freq_direct" "${SCRIPTS_DIR}/frequency/train_multires_freq_direct.py"

echo ""
echo "========================================================================"
echo "FREQUENCY MODELS TRAINING COMPLETED at $(date)"
echo "========================================================================"
