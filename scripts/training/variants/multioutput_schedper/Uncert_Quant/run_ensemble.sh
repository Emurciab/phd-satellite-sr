#!/usr/bin/env bash
# =============================================================================
# run_ensemble.sh — Train all M heteroscedastic ensemble members
# =============================================================================
#
# Usage:
#   bash run_ensemble.sh                   # train 5 members (default)
#   bash run_ensemble.sh --members 3       # train 3 members
#   bash run_ensemble.sh --resume          # resume interrupted training
#   bash run_ensemble.sh --evaluate        # only evaluate (skip training)
#
# Each member is trained with a different random seed but the same
# architecture and hyper-parameters. The diversity from different
# initialisations is what gives us epistemic uncertainty.
# =============================================================================

set -euo pipefail

# Force unbuffered Python output so logs appear in real time
export PYTHONUNBUFFERED=1

# ---- Defaults ----
N_MEMBERS=5
RESIZE_METHOD="adaptive"
EPOCHS=200
BATCH_SIZE=2
ACCUM=2
NLL_WARMUP=5
DATASET="full"
DO_TRAIN=true
DO_EVAL=true

# Seeds for each member — well-separated for independence
SEEDS=(1000 2000 3000 4000 5000)

# ---- Parse flags ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --members)    N_MEMBERS="$2"; shift 2;;
        --resize)     RESIZE_METHOD="$2"; shift 2;;
        --epochs)     EPOCHS="$2"; shift 2;;
        --dataset)    DATASET="$2"; shift 2;;
        --evaluate)   DO_TRAIN=false; shift;;
        --train-only) DO_EVAL=false; shift;;
        --resume)     shift;;  # resume is default (checkpoints auto-resume)
        *)            echo "Unknown flag: $1"; exit 1;;
    esac
done

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_het_ensemble_direct_learnedUps.py"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_ensemble.py"

# Compute PROJECT_ROOT (5 levels up: Uncert_Quant -> multioutput_schedper -> variants -> training -> scripts -> SR)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
TEST_DIR="${PROJECT_ROOT}/data/sr_training/all_stas_combined/mmap_test"
CHECKPOINT_BASE="${PROJECT_ROOT}/checkpoints"

echo "=============================================================="
echo "  Heteroscedastic Deep Ensemble — Direct + LearnedUps"
echo "=============================================================="
echo "  Members      : ${N_MEMBERS}"
echo "  Seeds        : ${SEEDS[*]:0:${N_MEMBERS}}"
echo "  Resize       : ${RESIZE_METHOD}"
echo "  Epochs       : ${EPOCHS}"
echo "  Dataset      : ${DATASET}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Checkpoints  : ${CHECKPOINT_BASE}"
echo "=============================================================="

# ---- Training ----
if $DO_TRAIN; then
    echo ""
    echo ">>> TRAINING PHASE"
    for ((i=0; i<N_MEMBERS; i++)); do
        SEED="${SEEDS[$i]}"
        CKPT_DIR="${CHECKPOINT_BASE}/het_ensemble_direct_learnedUps_${RESIZE_METHOD}_seed${SEED}_member${i}_${DATASET}"

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Member ${i}/${N_MEMBERS}  (seed=${SEED})"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Skip already-completed members
        if [[ -f "${CKPT_DIR}/COMPLETED" ]]; then
            echo "  ⏭ Already completed — skipping. ($(cat "${CKPT_DIR}/COMPLETED" | tr '\n' ' '))"
            continue
        fi

        conda run --no-capture-output -n tf_gpu python "${TRAIN_SCRIPT}" \
            --seed "${SEED}" \
            --ensemble_id "${i}" \
            --dataset "${DATASET}" \
            --resize_method "${RESIZE_METHOD}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --accumulation_steps "${ACCUM}" \
            --nll_warmup "${NLL_WARMUP}"

        echo ""
        echo "  ✓ Member ${i} (seed=${SEED}) complete."
    done
    echo ""
    echo ">>> All ${N_MEMBERS} members trained."
fi

# ---- Evaluation ----
if $DO_EVAL; then
    echo ""
    echo ">>> EVALUATION PHASE"

    # Build pattern to find trained checkpoints
    PATTERN="${CHECKPOINT_BASE}/het_ensemble_direct_learnedUps_${RESIZE_METHOD}_*_${DATASET}"
    
    conda run --no-capture-output -n tf_gpu python "${EVAL_SCRIPT}" \
        --checkpoint_pattern "${PATTERN}" \
        --test_dir "${TEST_DIR}" \
        --output_dir "${PROJECT_ROOT}/results/het_ensemble" \
        --save_maps

    echo ""
    echo ">>> Evaluation complete. Results in: ${PROJECT_ROOT}/results/het_ensemble/"
fi

echo ""
echo "=============================================================="
echo "  ALL DONE"
echo "=============================================================="
