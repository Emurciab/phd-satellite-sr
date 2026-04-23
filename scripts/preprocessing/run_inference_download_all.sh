#!/usr/bin/env bash
# ==============================================================================
# Batch download of inference data for all STAs — 2025 and 2026
# ==============================================================================
# Usage:
#   bash scripts/preprocessing/run_inference_download_all.sh
#
# Runs download_inference_data.py for each STA individually so that a failure
# in one STA does not block the others.  Logs are saved per-STA per-year.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Configuration ----------------------------------------------------------
CONDA_ENV="tf_gpu"
YEARS=("2025" "2026")
STAS=("STA-2" "STA-1E" "STA-1W" "STA-3/4" "STA-5/6")

SEQUENCE_LENGTH=3
COVERAGE_THRESHOLD=65
TILE_WORKERS=20
S2_CLOUD_PCT_MAX=60
S2_AOI_CLOUD_FRAC_MAX=0.25
PLANET_Q_CLOUD_MAX=25

LOG_DIR="$PROJECT_ROOT/logs/inference_download"
# -----------------------------------------------------------------------------

mkdir -p "$LOG_DIR"

TOTAL_START=$(date +%s)
PASS=0
FAIL=0
SKIP=0

echo "=============================================================="
echo "  BATCH INFERENCE DATA DOWNLOAD"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
echo "  Years : ${YEARS[*]}"
echo "  STAs  : ${STAS[*]}"
echo "  Logs  : $LOG_DIR"
echo "=============================================================="

for YEAR in "${YEARS[@]}"; do
    OUTPUT_DIR="$PROJECT_ROOT/cache/inference_${YEAR}"
    CACHE_DIR="$PROJECT_ROOT/cache/inference_${YEAR}_cache"

    echo ""
    echo "============================  ${YEAR}  ============================"

    for STA in "${STAS[@]}"; do
        STA_SAFE="${STA//\//_}"
        LOG_FILE="$LOG_DIR/${STA_SAFE}_${YEAR}.log"

        echo ""
        echo "--- ${STA}  (${YEAR}) ---"
        echo "    Log: $LOG_FILE"

        START_TS=$(date +%s)

        if conda run -n "$CONDA_ENV" python scripts/preprocessing/download_inference_data.py \
            --year "$YEAR" \
            --sta "$STA" \
            --output-dir "$OUTPUT_DIR" \
            --cache-dir "$CACHE_DIR" \
            --sequence-length "$SEQUENCE_LENGTH" \
            --coverage-threshold "$COVERAGE_THRESHOLD" \
            --tile-workers "$TILE_WORKERS" \
            --s2-cloud-pct-max "$S2_CLOUD_PCT_MAX" \
            --s2-aoi-cloud-frac-max "$S2_AOI_CLOUD_FRAC_MAX" \
            --planet-q-cloud-max "$PLANET_Q_CLOUD_MAX" \
            2>&1 | tee "$LOG_FILE"; then

            END_TS=$(date +%s)
            ELAPSED=$(( END_TS - START_TS ))
            echo "    ✓ ${STA} ${YEAR} finished in ${ELAPSED}s"
            PASS=$((PASS + 1))
        else
            END_TS=$(date +%s)
            ELAPSED=$(( END_TS - START_TS ))
            echo "    ✗ ${STA} ${YEAR} FAILED after ${ELAPSED}s — see $LOG_FILE"
            FAIL=$((FAIL + 1))
        fi
    done
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SEC=$(( TOTAL_ELAPSED % 60 ))

echo ""
echo "=============================================================="
echo "  BATCH COMPLETE  —  ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "  Passed: $PASS   Failed: $FAIL   Skipped: $SKIP"
echo "=============================================================="
