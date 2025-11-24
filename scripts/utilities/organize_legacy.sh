#!/bin/bash

# Organization script for moving buggy datasets to legacy folder
# Date: 2024-11-23
# Reason: Bug fix - LR-HR date mismatch (only S2 dates stored, Planet dates ignored)
# All existing samples have incorrect temporal labels and must be regenerated

set -e

LEGACY_ROOT="/media/eamurciab/TOSHIBA_EXT1/PhD/SR/legacy"
TIMESTAMP="2024-11-23_buggy_data"

echo "=========================================================================="
echo "ORGANIZING WORKSPACE - MOVING BUGGY DATA TO LEGACY"
echo "=========================================================================="
echo ""
echo "⚠️  REASON: Critical bug in date labeling"
echo "   - Only Sentinel-2 dates were stored"
echo "   - Planet dates (±7 days different) were ignored"
echo "   - All 400 samples have incorrect temporal information"
echo "   - See: BUG_FIX_LR_HR_DATE_MISMATCH.md"
echo ""

# Create legacy structure
echo "📁 Creating legacy folder structure..."
mkdir -p "$LEGACY_ROOT/$TIMESTAMP/datasets"
mkdir -p "$LEGACY_ROOT/$TIMESTAMP/visualizations"
mkdir -p "$LEGACY_ROOT/$TIMESTAMP/logs"
mkdir -p "$LEGACY_ROOT/$TIMESTAMP/scripts"
mkdir -p "$LEGACY_ROOT/$TIMESTAMP/docs"

# Move the main 400-sample dataset (BUGGY - 1.4GB)
echo ""
echo "📦 Moving main dataset (sta2_grid_parallel - 400 samples, 1.4GB)..."
if [ -d "data/datasets/sta2_grid_parallel" ]; then
    mv data/datasets/sta2_grid_parallel "$LEGACY_ROOT/$TIMESTAMP/datasets/"
    echo "   ✅ Moved sta2_grid_parallel (400 samples with wrong dates)"
else
    echo "   ⚠️  sta2_grid_parallel not found (already moved?)"
fi

# Move visualizations (820MB - shows the buggy date labels)
echo ""
echo "🖼️  Moving visualizations..."
if [ -d "visualizations/sequences" ]; then
    mv visualizations/sequences "$LEGACY_ROOT/$TIMESTAMP/visualizations/"
    echo "   ✅ Moved visualizations/sequences"
fi

if [ -d "visualizations/sequences_400samples" ]; then
    mv visualizations/sequences_400samples "$LEGACY_ROOT/$TIMESTAMP/visualizations/"
    echo "   ✅ Moved visualizations/sequences_400samples"
fi

if [ -f "visualizations/sequences_grid_400samples_overview.png" ]; then
    mv visualizations/sequences_grid_400samples_overview.png "$LEGACY_ROOT/$TIMESTAMP/visualizations/"
    echo "   ✅ Moved 400-sample grid overview"
fi

if [ -f "visualizations/sequences_grid_12samples.png" ]; then
    mv visualizations/sequences_grid_12samples.png "$LEGACY_ROOT/$TIMESTAMP/visualizations/"
    echo "   ✅ Moved 12-sample grid"
fi

if [ -f "visualizations/sequences_grid_ALL48samples.png" ]; then
    mv visualizations/sequences_grid_ALL48samples.png "$LEGACY_ROOT/$TIMESTAMP/visualizations/"
    echo "   ✅ Moved 48-sample grid"
fi

# Move generation logs
echo ""
echo "📋 Moving generation logs..."
for log in dataset_generation*.log; do
    if [ -f "$log" ]; then
        mv "$log" "$LEGACY_ROOT/$TIMESTAMP/logs/"
        echo "   ✅ Moved $log"
    fi
done

# Move old test datasets (all have the same bug)
echo ""
echo "🧪 Moving old test datasets..."
OLD_DATASETS=(
    "sta2_ultrafast"
    "sta2_20samples_harmonized"
    "test_12workers"
    "sr_dataset_5_test"
    "sr_dataset_20samples"
)

for dataset in "${OLD_DATASETS[@]}"; do
    if [ -d "data/datasets/$dataset" ]; then
        mv "data/datasets/$dataset" "$LEGACY_ROOT/$TIMESTAMP/datasets/"
        echo "   ✅ Moved $dataset"
    fi
done

# Move redundant visualization folders
echo ""
echo "🗑️  Removing redundant visualization folders..."
if [ -d "data/datasets/sta2_20samples_harmonized/visualizations" ]; then
    rm -rf "data/datasets/sta2_20samples_harmonized/visualizations"
    echo "   ✅ Removed duplicate visualizations in sta2_20samples_harmonized"
fi

if [ -d "data/datasets/sta2_ultrafast/visualizations" ]; then
    rm -rf "data/datasets/sta2_ultrafast/visualizations"
    echo "   ✅ Removed duplicate visualizations in sta2_ultrafast"
fi

# Copy bug documentation
echo ""
echo "📄 Copying bug documentation..."
if [ -f "BUG_FIX_LR_HR_DATE_MISMATCH.md" ]; then
    cp BUG_FIX_LR_HR_DATE_MISMATCH.md "$LEGACY_ROOT/$TIMESTAMP/docs/"
    echo "   ✅ Copied bug fix documentation"
fi

# Copy visualization scripts (buggy versions)
echo ""
echo "📝 Copying visualization scripts (buggy versions)..."
cp scripts/preprocessing/visualize_sequences.py "$LEGACY_ROOT/$TIMESTAMP/scripts/visualize_sequences_buggy.py" 2>/dev/null || echo "   ⚠️  visualize_sequences.py not found"
cp scripts/preprocessing/visualize_sequences_grid.py "$LEGACY_ROOT/$TIMESTAMP/scripts/visualize_sequences_grid_buggy.py" 2>/dev/null || echo "   ⚠️  visualize_sequences_grid.py not found"
cp visualize_20samples.py "$LEGACY_ROOT/$TIMESTAMP/scripts/" 2>/dev/null || true

# Create README in legacy folder
echo ""
echo "📝 Creating legacy README..."
cat > "$LEGACY_ROOT/$TIMESTAMP/README.md" << 'EOFREADME'
# Legacy Data - Buggy Date Labels (2024-11-23)

## ⚠️ WARNING: DO NOT USE THIS DATA FOR TRAINING

All datasets in this folder have a **critical bug** in temporal labeling.

## The Bug

**Problem**: Only Sentinel-2 acquisition dates were stored. Planet dates (which differ by ±7 days due to matching tolerance) were completely ignored.

**Impact**: 
- All HR images are labeled with LR (Sentinel-2) dates
- Temporal alignment verification is impossible
- Training would use incorrect temporal information

**Example**:
```
Frame 1: 
  - Planet acquired: 2020-01-06
  - Sentinel-2 acquired: 2020-01-01
  - Stored label: "2020-01-01" (WRONG for Planet!)
```

## Contents

### Datasets (INVALID - must be regenerated)
- `sta2_grid_parallel/` - 400 samples, 1.4GB (main dataset with bug)
- `sta2_ultrafast/` - 20 samples test run
- `test_12workers/` - Worker configuration tests
- Other test datasets with same bug

### Visualizations
- Shows the buggy date labels
- Grid overviews and individual sequences
- All visualizations show only S2 dates for both LR and HR

### Logs
- Generation logs showing the bug in action
- Performance metrics (still valid)
- Coverage statistics (still valid)

### Documentation
- `BUG_FIX_LR_HR_DATE_MISMATCH.md` - Complete bug analysis and fix

## What Was Fixed

1. **TrainingSample dataclass**: Added `planet_dates` field
2. **Date storage**: Changed from single date to tuple `(s2_date, planet_date)`
3. **Sample creation**: Separated dates when creating samples
4. **Visualizations**: Updated to show both LR and HR dates correctly

## Next Steps

1. Regenerate 400 samples with fixed code (~14 hours)
2. Verify new samples have correct date pairing
3. Continue to 1440 samples for full dataset
4. Process multiple regions (2020-2025)

## Performance Metrics (Still Valid)

- 400 samples: 827.7 minutes (13.79 hours)
- Time per sample: 124.1 seconds (2.07 minutes)
- Throughput: 29.01 samples/hour
- Workers: 15 parallel
- Success rate: 100% (generation succeeded, but labels were wrong)

Generated: 2024-11-23
Reason: Bug fix - LR-HR date mismatch
EOFREADME

echo "   ✅ Created legacy README"

# Create summary file
echo ""
echo "📊 Creating organization summary..."
cat > "$LEGACY_ROOT/$TIMESTAMP/ORGANIZATION_SUMMARY.txt" << 'EOFSUMMARY'
========================================================================
LEGACY DATA ORGANIZATION SUMMARY
========================================================================

Date: 2024-11-23
Reason: Critical bug in temporal labeling - LR-HR date mismatch

BUG DESCRIPTION:
- Only Sentinel-2 dates were stored in temporal_dates field
- Planet dates (±7 days different) were completely ignored
- All 400 samples have incorrect temporal information
- HR images labeled with S2 acquisition dates instead of Planet dates

WHAT WAS MOVED:

Main Dataset (INVALID):
  ✓ sta2_grid_parallel/ (400 samples, 1.4GB) - PRIMARY DATASET WITH BUG

Visualizations (820MB):
  ✓ sequences/ - Individual sequence visualizations
  ✓ sequences_400samples/ - 400-sample visualizations
  ✓ sequences_grid_*.png - Grid overview images

Generation Logs:
  ✓ dataset_generation_48samples_15workers.log
  ✓ dataset_generation_200samples_15workers.log
  ✓ dataset_generation_400samples_15workers.log

Old Test Datasets (all with same bug):
  ✓ sta2_ultrafast/ (20 samples)
  ✓ sta2_20samples_harmonized/
  ✓ test_12workers/
  ✓ sr_dataset_5_test/
  ✓ sr_dataset_20samples/

Redundant Folders Removed:
  ✓ Duplicate visualization folders in dataset directories

Documentation:
  ✓ BUG_FIX_LR_HR_DATE_MISMATCH.md
  ✓ Legacy README with bug explanation

CLEANED WORKSPACE:
  ✓ data/datasets/ - Only kept non-buggy datasets
  ✓ visualizations/ - Empty, ready for new visualizations
  ✓ Root logs moved to legacy

STATUS: Ready for regeneration with fixed code

NEXT STEPS:
1. Regenerate 400 samples with correct date pairing (~14 hours)
2. Verify new samples show both S2 and Planet dates
3. Continue to 1440 samples total
4. Process multiple regions (25 regions, 2020-2025)

========================================================================
EOFSUMMARY

echo "   ✅ Created organization summary"

# Calculate sizes
echo ""
echo "📊 Calculating sizes..."
LEGACY_SIZE=$(du -sh "$LEGACY_ROOT/$TIMESTAMP" 2>/dev/null | cut -f1)

echo ""
echo "=========================================================================="
echo "✅ ORGANIZATION COMPLETE!"
echo "=========================================================================="
echo ""
echo "📁 Legacy data location:"
echo "   $LEGACY_ROOT/$TIMESTAMP/"
echo ""
echo "💾 Total legacy size: $LEGACY_SIZE"
echo ""
echo "📋 Summary files created:"
echo "   - README.md (bug explanation)"
echo "   - ORGANIZATION_SUMMARY.txt (what was moved)"
echo ""
echo "🧹 Workspace cleaned:"
echo "   ✓ Invalid datasets moved to legacy"
echo "   ✓ Buggy visualizations archived"
echo "   ✓ Generation logs archived"
echo "   ✓ Redundant folders removed"
echo ""
echo "✅ Ready for regeneration with fixed code!"
echo ""
echo "Next command:"
echo "   python scripts/preprocessing/generate_sta2_grid_parallel.py \\"
echo "       --max-samples 400 \\"
echo "       --coverage-threshold 90 \\"
echo "       --coverage-workers 15"
echo ""
echo "=========================================================================="

