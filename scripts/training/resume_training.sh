#!/bin/bash
# Resume training from where it stopped
# 7 completed from variants, 9 remaining (7 from variants + 2 main scripts)

set -e

PROJECT_ROOT="/media/eamurciab/TOSHIBA_EXT1/PhD/SR"
LOG_DIR="${PROJECT_ROOT}/logs"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts/training/variants"
MAIN_SCRIPTS="${PROJECT_ROOT}/scripts/training"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

cd "${PROJECT_ROOT}"

echo "========================================================================"
echo "RESUME TRAINING - 9 REMAINING MODELS"
echo "========================================================================"
echo "Start time: $(date)"
echo "Data: Clean dataset (5446 samples)"
echo ""
echo "Completed (9):"
echo "  - bestframe_residual, bestframe_direct"
echo "  - median_residual, median_direct"
echo "  - multioutput_residual, multioutput_direct"
echo "  - multioutput_schedper_residual, multioutput_schedper_direct"
echo "  - temporal_residual"
echo ""
echo "To train (6):"
echo "  1. temporal_direct"
echo "  2. gan_temporal"
echo "  3. gan_direct"
echo "  4. freq_residual"
echo "  5. freq_direct"
echo "  6. v3_full"
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
        echo "[$name] COMPLETED in ${HOURS}h ${MINUTES}m at $(date)" >> "${LOG_DIR}/resume_training.log"
    else
        echo "[$name] FAILED - check ${LOG_DIR}/training_${name}.log"
        echo "[$name] FAILED at $(date)" >> "${LOG_DIR}/resume_training.log"
        return 1
    fi
}

# ============== VARIANTS ==============

# 1. MultiOutput SchedPer Direct - COMPLETED (115 epochs, stopped by shutdown but converged)
# train_variant "multioutput_schedper_direct" "${SCRIPTS_DIR}/multioutput_schedper/train_multires_temporal_multiOutput_schedPer_direct.py"

# 2. Temporal Residual - COMPLETED (101 epochs, plateaued at val_loss=0.298, PSNR=28.9)
# train_variant "temporal_residual" "${SCRIPTS_DIR}/temporal/train_multires_temporal.py"

# 3. Temporal Direct
train_variant "temporal_direct" "${SCRIPTS_DIR}/temporal/train_multires_temporal_direct.py"

# 4. GAN Temporal
train_variant "gan_temporal" "${SCRIPTS_DIR}/GAN/train_multires_temporal_GAN.py"

# 5. GAN Direct
train_variant "gan_direct" "${SCRIPTS_DIR}/GAN/train_multires_GAN_direct.py"

# 6. Frequency Residual
train_variant "freq_residual" "${SCRIPTS_DIR}/frequency/train_multires_freq_residual.py"

# 7. Frequency Direct
train_variant "freq_direct" "${SCRIPTS_DIR}/frequency/train_multires_freq_direct.py"

# ============== MAIN SCRIPTS ==============

# 8. V3 Full
train_variant "v3_full" "${MAIN_SCRIPTS}/train_5band_v3.py"

# NOTE: train_5band_multires.py requires hr.npy which doesn't exist in clean dataset
# Skipping multires_full

echo ""
echo "========================================================================"
echo "ALL 6 MODELS TRAINING COMPLETED at $(date)"
echo "========================================================================"
