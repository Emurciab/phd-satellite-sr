# Temporal Sequence Visualizations

## Overview

This directory contains visualizations of the grid-based parallel dataset generated for super-resolution training.

## Dataset Information

- **Region**: STA-2
- **Total Samples**: 48
- **Sequence Length**: 3 temporal frames
- **Temporal Dates**: 2020-01-01 → 2020-01-06 → 2020-04-15
- **Coverage Threshold**: 90%
- **Sampling Strategy**: Grid-based systematic tile sampling

## Visualizations

### 1. Grid Overviews

**`sequences_grid_12samples.png`** (18 MB)
- Shows first 12 sequences in a compact grid
- Each row = one sample (different tile location)
- Columns alternate: LR-T1, HR-T1, LR-T2, HR-T2, LR-T3, HR-T3
- Good for quick quality assessment

**`sequences_grid_ALL48samples.png`** (71 MB)
- Shows all 48 samples in a single view
- Same layout as 12-sample grid
- Comprehensive overview of entire dataset

### 2. Individual Sequences (`sequences/` directory)

**48 high-resolution visualizations** (~1.5 MB each)
- One file per sample: `sequence_0000.png` to `sequence_0047.png`
- 2 rows × 3 columns layout:
  - **Top row**: Sentinel-2 LR frames (10m resolution, ~79×79 px)
  - **Bottom row**: Planet HR frames (3m resolution, ~260×260 px)
- Each frame labeled with date and dimensions
- Value ranges displayed for quality checking

## Data Characteristics

### Temporal Information
- Frame 1 → Frame 2: 5 days apart ✓
- Frame 2 → Frame 3: 100 days apart ⚠️ (Large gap flagged)

### Spatial Resolution
- **LR (Sentinel-2)**: ~79×79 pixels at 10m/px
- **HR (Planet)**: ~260×261 pixels at 3m/px
- **Scale factor**: ~3.3× spatial resolution improvement

### Spectral Bands
- **4 bands**: Blue, Green, Red, NIR
- **Processing**: Coregistered + Histogram matched

## Usage

### View Grid Overviews
```bash
# Quick overview (12 samples)
xdg-open visualizations/sequences_grid_12samples.png

# Complete dataset (48 samples)
xdg-open visualizations/sequences_grid_ALL48samples.png
```

### View Individual Sequences
```bash
# View specific sample
xdg-open visualizations/sequences/sequence_0000.png

# View all in sequence
eog visualizations/sequences/*.png
```

### Regenerate Visualizations

**Individual sequences:**
```bash
python scripts/preprocessing/visualize_sequences.py \
    --dataset-path ./data/datasets/sta2_grid_parallel/dataset.pkl \
    --max-samples 48 \
    --save-dir ./visualizations/sequences \
    --no-show
```

**Grid overview:**
```bash
python scripts/preprocessing/visualize_sequences_grid.py \
    --dataset-path ./data/datasets/sta2_grid_parallel/dataset.pkl \
    --max-sequences 12 \
    --save-path ./visualizations/sequences_grid_12samples.png \
    --no-show
```

## Quality Observations

✅ **Excellent coverage**: All samples have 100% data in all 3 frames
✅ **No nodata**: Grid-based sampling with pre-checking eliminated gaps
✅ **Consistent dimensions**: LR ~79×79, HR ~260×260 across all samples
✅ **Coregistration successful**: AROSICS reliability 44-69% (acceptable)
⚠️ **Large temporal gap**: 100 days between frames 2-3 (affects temporal attention)

## Next Steps

- Use these visualizations to verify data quality before training
- Select representative samples for model architecture testing
- Consider generating additional sequences with better temporal coverage
- Use grid overview to identify spatial coverage patterns
