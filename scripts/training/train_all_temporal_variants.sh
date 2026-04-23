#!/bin/bash
# =============================================================================
# Train All Temporal Variants Sequentially
# =============================================================================
# This script trains all 3 temporal approaches one after another
# since we only have 1 GPU and can't run them in parallel.
#
# Estimated time: ~3-4 hours per model x 3 = ~9-12 hours total
# =============================================================================

set -e  # Exit on any error

# Setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
cd /media/eamurciab/TOSHIBA_EXT1/PhD/SR

# Create logs directory
mkdir -p logs

echo "=============================================="
echo "Starting Sequential Training of Temporal Variants"
echo "Start time: $(date)"
echo "=============================================="

# =============================================================================
# 1. MEDIAN COMPOSITE
# =============================================================================
echo ""
echo "[1/3] Training Median Composite Model..."
echo "Start: $(date)"

PYTHONUNBUFFERED=1 python scripts/training/train_multires_temporal_median.py \
    --dataset full \
    --epochs 200 \
    2>&1 | tee logs/median_training_$(date +%Y%m%d_%H%M%S).log

echo "Median training complete: $(date)"

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
sleep 10

# =============================================================================
# 2. BEST FRAME SELECTION
# =============================================================================
echo ""
echo "[2/3] Training Best Frame Selection Model..."
echo "Start: $(date)"

PYTHONUNBUFFERED=1 python scripts/training/train_multires_temporal_bestFrame.py \
    --dataset full \
    --epochs 200 \
    --metric combined \
    2>&1 | tee logs/bestframe_training_$(date +%Y%m%d_%H%M%S).log

echo "Best Frame training complete: $(date)"

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
sleep 10

# =============================================================================
# 3. MULTI-OUTPUT TEMPORAL
# =============================================================================
echo ""
echo "[3/3] Training Multi-Output Temporal Model..."
echo "Start: $(date)"

PYTHONUNBUFFERED=1 python scripts/training/train_multires_temporal_multiOutput.py \
    --dataset full \
    --epochs 200 \
    --alpha_temporal 0.5 \
    2>&1 | tee logs/multioutput_training_$(date +%Y%m%d_%H%M%S).log

echo "Multi-Output training complete: $(date)"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "All Training Complete!"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Checkpoints saved in:"
echo "  1. checkpoints/multires_median_median_full/"
echo "  2. checkpoints/multires_bestframe_combined_full/"
echo "  3. checkpoints/multires_multioutput_t0.5_full/"
echo ""
echo "Next step: Run evaluation with evaluate_models.py"
