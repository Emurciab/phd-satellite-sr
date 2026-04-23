#!/usr/bin/env bash
# ===========================================================================
#  Train all MISSING ablation models sequentially.
#
#  Already trained (skipped):
#    - ablation_no_sam_loss_adaptive        (residual, best epoch 127)
#    - ablation_no_spectral_adaptive        (residual, no SE + no SAM, best epoch 118)
#
#  Missing (trained by this script):
#    1. No SAM Loss — Direct
#    2. No SE Module — Residual  (SAM loss KEPT)
#    3. No SE Module — Direct    (SAM loss KEPT)
#    4. No Temporal + No Fusion — Residual
#    5. No Temporal + No Fusion — Direct
#
#  Usage:
#    chmod +x run_missing_ablations.sh
#    ./run_missing_ablations.sh              # train all missing
#    ./run_missing_ablations.sh --resume     # resume interrupted runs
#
#  All models use:
#    - epochs=200, patience=20 (early stopping)
#    - batch_size=2 (fits RTX 4060 8GB)
#    - resize_method=adaptive
#    - data: all_stas_combined/mmap_train_clean
# ===========================================================================

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="/mnt/TOSHIBA_EXT1/PhD/SR"
PYTHON="/home/eamurciab/miniconda3/envs/tf_gpu/bin/python"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/training/variants/ablationstudy"
DATA_DIR="${PROJECT_ROOT}/data/sr_training/all_stas_combined/mmap_train_clean"
CKPT_ROOT="${PROJECT_ROOT}/checkpoints"

RESUME_FLAG=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume"
    echo ">>> Resume mode enabled — will continue from latest checkpoints."
fi

# ─── Common training arguments ───────────────────────────────────────────────
COMMON_ARGS=(--data_dir "${DATA_DIR}" --epochs 200 --batch_size 2 --resize_method adaptive)
if [[ -n "${RESUME_FLAG}" ]]; then
    COMMON_ARGS+=(--resume)
fi

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

train_model() {
    local script_name="$1"
    local checkpoint_dir="$2"
    local description="$3"
    local log_file="${LOG_DIR}/${script_name%.py}_${TIMESTAMP}.log"

    echo ""
    echo "======================================================================"
    echo "  ${description}"
    echo "  Script:     ${script_name}"
    echo "  Checkpoint: ${checkpoint_dir}"
    echo "  Log:        ${log_file}"
    echo "======================================================================"

    # Skip if already trained (best_model.pt exists AND we're NOT resuming)
    if [[ -f "${checkpoint_dir}/best_model.pt" && -z "${RESUME_FLAG}" ]]; then
        echo "  ⏭  SKIPPING — checkpoint already exists. Use --resume to continue."
        return 0
    fi

    local start_time=$(date +%s)

    ${PYTHON} "${SCRIPT_DIR}/${script_name}" \
        "${COMMON_ARGS[@]}" \
        --checkpoint_dir "${checkpoint_dir}" \
        2>&1 | tee "${log_file}"

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    local hours=$(( elapsed / 3600 ))
    local minutes=$(( (elapsed % 3600) / 60 ))

    if [[ ${exit_code} -eq 0 ]]; then
        echo "  ✅ DONE in ${hours}h ${minutes}m"
    else
        echo "  ❌ FAILED (exit code ${exit_code}) after ${hours}h ${minutes}m"
        echo "  Check log: ${log_file}"
        # Continue to next model instead of aborting
    fi

    echo ""
    return ${exit_code}
}

# ===========================================================================
#  Main
# ===========================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           ABLATION STUDY — Train Missing Models                    ║"
echo "║           $(date)                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Python:  ${PYTHON}"
echo "  Data:    ${DATA_DIR}"
echo "  Ckpts:   ${CKPT_ROOT}/ablation_*"
echo ""

# Verify data exists
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    exit 1
fi

TOTAL_START=$(date +%s)
FAILED=0

# ─── 1. No SAM Loss — Direct ─────────────────────────────────────────────────
# SKIPPED — already early-stopped (best ep 116, LR ~0 at ep 130)
echo "  ⏭  SKIPPING Ablation 1/5: No SAM Loss — DIRECT (already early-stopped)"

# ─── 2. No SE Module — Residual (SAM loss KEPT) ──────────────────────────────
# SKIPPED — already early-stopped (best ep 135, 140 epochs trained)
echo "  ⏭  SKIPPING Ablation 2/5: No SE Module — RESIDUAL (already early-stopped)"

# ─── 3. No SE Module — Direct (SAM loss KEPT) ────────────────────────────────
train_model \
    "train_ablation_no_se_module_direct.py" \
    "${CKPT_ROOT}/ablation_no_se_module_direct_adaptive" \
    "Ablation 3/5: No SE Module — DIRECT (SE→Identity, SAM kept at 0.137)" \
    || FAILED=$((FAILED + 1))

# ─── 4. No Temporal CrossAttn + No Cross-Res Fusion — Residual ───────────────
train_model \
    "train_ablation_no_temporal_no_fusion_residual.py" \
    "${CKPT_ROOT}/ablation_no_temporal_no_fusion_residual_adaptive" \
    "Ablation 4/5: No Temporal + No Fusion — RESIDUAL" \
    || FAILED=$((FAILED + 1))

# ─── 5. No Temporal CrossAttn + No Cross-Res Fusion — Direct ─────────────────
train_model \
    "train_ablation_no_temporal_no_fusion_direct.py" \
    "${CKPT_ROOT}/ablation_no_temporal_no_fusion_direct_adaptive" \
    "Ablation 5/5: No Temporal + No Fusion — DIRECT" \
    || FAILED=$((FAILED + 1))

# ─── Summary ─────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_HOURS=$(( TOTAL_ELAPSED / 3600 ))
TOTAL_MINUTES=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                     ABLATION TRAINING COMPLETE                     ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m                                              ║"
echo "║  Failed: ${FAILED}/5                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Print checkpoint summary
echo "Checkpoint status:"
for name in \
    ablation_no_sam_loss_adaptive \
    ablation_no_spectral_adaptive \
    ablation_no_sam_loss_direct_adaptive \
    ablation_no_se_module_residual_adaptive \
    ablation_no_se_module_direct_adaptive \
    ablation_no_temporal_no_fusion_residual_adaptive \
    ablation_no_temporal_no_fusion_direct_adaptive; do

    ckpt="${CKPT_ROOT}/${name}/best_model.pt"
    if [[ -f "${ckpt}" ]]; then
        echo "  ✅ ${name}"
    else
        echo "  ❌ ${name}  (missing)"
    fi
done

echo ""
echo "Logs saved in: ${LOG_DIR}/"

exit ${FAILED}
