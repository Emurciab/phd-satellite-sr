#!/bin/bash
# Master training script for ALL 14 model variants
# Run with: bash scripts/training/train_all_variants.sh
# 
# Expected total time: ~24-48 hours depending on GPU
# Each model takes 2-4 hours with 5472 training samples

set -e  # Exit on error

# Configuration
EPOCHS=200
BATCH_SIZE=4
LOG_DIR="logs"
CHECKPOINT_BASE="checkpoints/all_stas"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_BASE"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

cd /media/eamurciab/TOSHIBA_EXT1/PhD/SR

echo "========================================================================"
echo "MASTER TRAINING SCRIPT - ALL 14 MODEL VARIANTS"
echo "========================================================================"
echo "Start time: $(date)"
echo "Data: 5472 training samples from ALL STAs"
echo "Epochs: $EPOCHS per model"
echo "Batch size: $BATCH_SIZE"
echo "========================================================================"

# Track results
RESULTS_FILE="$LOG_DIR/training_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Training Results" > "$RESULTS_FILE"
echo "================" >> "$RESULTS_FILE"
echo "Started: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

train_model() {
    local model_name=$1
    local script=$2
    local extra_args="${3:-}"
    
    echo ""
    echo "========================================================================"
    echo "[$model_name] Starting at $(date)"
    echo "========================================================================"
    
    local log_file="$LOG_DIR/training_${model_name}.log"
    local start_time=$(date +%s)
    
    if python "$script" --epochs $EPOCHS --batch_size $BATCH_SIZE $extra_args > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        
        echo "[$model_name] COMPLETED in ${hours}h ${minutes}m"
        echo "$model_name: SUCCESS (${hours}h ${minutes}m)" >> "$RESULTS_FILE"
        
        # Extract best PSNR from log
        local psnr=$(grep -o "Best.*PSNR.*dB" "$log_file" | tail -1 || echo "N/A")
        echo "  $psnr" >> "$RESULTS_FILE"
    else
        echo "[$model_name] FAILED - check $log_file"
        echo "$model_name: FAILED" >> "$RESULTS_FILE"
    fi
}

# ============================================================================
# Model 1: BestFrame (Residual)
# ============================================================================
train_model "bestframe_residual" \
    "scripts/training/variants/bestframe/train_multires_temporal_bestFrame.py"

# ============================================================================
# Model 2: BestFrame (Direct)
# ============================================================================
train_model "bestframe_direct" \
    "scripts/training/variants/bestframe/train_multires_temporal_bestFrame_direct.py"

# ============================================================================
# Model 3: Median (Residual)
# ============================================================================
train_model "median_residual" \
    "scripts/training/variants/median/train_multires_temporal_median.py"

# ============================================================================
# Model 4: Median (Direct)
# ============================================================================
train_model "median_direct" \
    "scripts/training/variants/median/train_multires_temporal_median_direct.py"

# ============================================================================
# Model 5: MultiOutput (Residual)
# ============================================================================
train_model "multioutput_residual" \
    "scripts/training/variants/multioutput/train_multires_temporal_multiOutput.py"

# ============================================================================
# Model 6: MultiOutput (Direct)
# ============================================================================
train_model "multioutput_direct" \
    "scripts/training/variants/multioutput/train_multires_temporal_multiOutput_direct.py"

# ============================================================================
# Model 7: MultiOutput SchedPer (Residual)
# ============================================================================
train_model "multioutput_schedper_residual" \
    "scripts/training/variants/multioutput_schedper/train_multires_temporal_multiOutput_schedPer.py"

# ============================================================================
# Model 8: MultiOutput SchedPer (Direct)
# ============================================================================
train_model "multioutput_schedper_direct" \
    "scripts/training/variants/multioutput_schedper/train_multires_temporal_multiOutput_schedPer_direct.py"

# ============================================================================
# Model 9: Temporal (Residual)
# ============================================================================
train_model "temporal_residual" \
    "scripts/training/variants/temporal/train_multires_temporal.py"

# ============================================================================
# Model 10: Temporal (Direct)
# ============================================================================
train_model "temporal_direct" \
    "scripts/training/variants/temporal/train_multires_temporal_direct.py"

# ============================================================================
# Model 11: GAN
# ============================================================================
train_model "gan" \
    "scripts/training/variants/GAN/train_multires_temporal_GAN.py"

# ============================================================================
# Model 12: V3 (Full) - Direct prediction baseline
# ============================================================================
train_model "v3_full" \
    "scripts/training/train_5band_v3.py" \
    "--data_dir data/sr_training/all_stas_combined/mmap_train --checkpoint_dir checkpoints/v3_full_allstas"

# ============================================================================
# Model 13: MultiRes (Full)
# ============================================================================
train_model "multires_full" \
    "scripts/training/train_5band_multires.py" \
    "--data_dir data/sr_training/all_stas_combined/mmap_train --checkpoint_dir checkpoints/multires_full_allstas"

# ============================================================================
# Model 14: Frequency variants (if scripts exist)
# ============================================================================
if [ -f "scripts/training/variants/frequency/train_multires_freq_residual.py" ]; then
    train_model "frequency_residual" \
        "scripts/training/variants/frequency/train_multires_freq_residual.py"
fi

if [ -f "scripts/training/variants/frequency/train_multires_freq_direct.py" ]; then
    train_model "frequency_direct" \
        "scripts/training/variants/frequency/train_multires_freq_direct.py"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================================================"
echo "ALL TRAINING COMPLETE"
echo "========================================================================"
echo "End time: $(date)"
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Detailed logs in: $LOG_DIR/"
echo "Checkpoints in: checkpoints/"
