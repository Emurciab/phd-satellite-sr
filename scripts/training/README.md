# Training Scripts

Super-resolution model training for satellite imagery.

## Overview

This directory contains training scripts for SR models, including both **interpolated** (original) and **multi-resolution** (correct) approaches.

---

## ⚠️ Important: B8A Interpolation Issue

**Problem:** Sentinel-2's B8A band is natively 20m resolution. Interpolating it to 10m before training introduces:
- Artificial smoothing and artifacts
- Corrupted NIR spectral signals
- Model learns interpolation, not true super-resolution
- Inflated metrics on interpolated test data
- Poor generalization to native 20m data

**Solution:** Use `train_multi_resolution_sr.py` which trains separate models for 10m (RGB) and 20m (NIR) bands at their native resolutions.

**Documentation:** See [MULTI_RESOLUTION_SR_APPROACH.md](../../docs/analysis/MULTI_RESOLUTION_SR_APPROACH.md)

---

## Scripts

### 1. `train_multi_resolution_sr.py` ✅ **RECOMMENDED**

Train separate SR models for 10m and 20m bands (correct approach).

**Features:**
- Preserves native resolution characteristics
- No interpolation artifacts in training data
- Spectral integrity for vegetation indices
- Realistic validation metrics
- Better generalization

**Usage:**
```bash
# Train both 10m and 20m models
python scripts/training/train_multi_resolution_sr.py \
    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \
    --output-dir sr_models_multi_res \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-4

# Train only 10m model (for testing)
python scripts/training/train_multi_resolution_sr.py \
    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \
    --train-10m-only \
    --epochs 20
```

**Output:**
- `best_model_10m.pth` - Best 10m RGB model
- `best_model_20m.pth` - Best 20m NIR model
- `final_model_10m.pth` - Final 10m model
- `final_model_20m.pth` - Final 20m model
- Checkpoints every 10 epochs

**Architecture:**
- **10m Model**: B2,B3,B4 (10m) → B2,B3,B4 (3m) | Scale: 3.33x
- **20m Model**: B8A (20m) → B8A (6.67m) | Scale: 3x

---

### 2. `train_with_chunked_dataset.py` ⚠️ **OLD APPROACH**

Original training script using interpolated B8A (kept for comparison).

**Issues:**
- ❌ Uses interpolated 20m→10m B8A data
- ❌ Learns interpolation artifacts, not true SR
- ❌ Corrupts NIR for vegetation analysis
- ❌ Inflated metrics on interpolated test sets

**Usage:** (For comparison only)
```bash
python scripts/training/train_with_chunked_dataset.py \
    --dataset-dir data/datasets/sta2_grid_parallel \
    --output-dir sr_models_interpolated \
    --batch-size 4 \
    --epochs 50
```

---

### 3. `compare_interpolation_approaches.py` 📊 **DEMO**

Demonstrates why interpolating B8A is problematic.

**Usage:**
```bash
python scripts/training/compare_interpolation_approaches.py
```

**Output:**
- Console: Detailed comparison and examples
- `interpolation_artifact_demo.png` - Visual comparison
- Impact analysis on vegetation indices

---

## Quick Start

### After Dataset Generation

1. **Convert dataset** (automatic):
```bash
python scripts/training/train_multi_resolution_sr.py \
    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \
    --output-dir sr_models_multi_res
```

The script automatically:
- Converts 4-channel data to multi-resolution format
- Removes interpolation artifacts
- Trains both 10m and 20m models

2. **Run comparison demo** (optional):
```bash
python scripts/training/compare_interpolation_approaches.py
```

3. **Inference** (after training):
```python
import torch
from train_multi_resolution_sr import SatelliteSRModel

# Load models
model_10m = SatelliteSRModel(in_channels=3, out_channels=3, upscale_factor=3)
model_20m = SatelliteSRModel(in_channels=1, out_channels=1, upscale_factor=3)

checkpoint_10m = torch.load('sr_models_multi_res/best_model_10m.pth')
checkpoint_20m = torch.load('sr_models_multi_res/best_model_20m.pth')

model_10m.load_state_dict(checkpoint_10m['model_state_dict'])
model_20m.load_state_dict(checkpoint_20m['model_state_dict'])

# Run inference
sr_rgb = model_10m(lr_10m_sequence)  # (B, 3, H*3.33, W*3.33)
sr_nir = model_20m(lr_20m_sequence)  # (B, 1, H/2*3, W/2*3)
```

---

## Technical Details

### Multi-Resolution Approach

**10m Bands (B2, B3, B4):**
- Native resolution: 10m
- Target resolution: 3m (Planet)
- Scale factor: 3.33x
- Channels: 3 (Blue, Green, Red)

**20m Band (B8A):**
- Native resolution: 20m (no interpolation!)
- Target resolution: 6.67m (Planet 3m downsampled 2x)
- Scale factor: 3x
- Channels: 1 (Near-Infrared)

**Why 6.67m target for 20m?**
- Planet provides 3m NIR natively
- Downsampling to 6.67m (2x) maintains proper scale relationship
- Avoids over-ambitious 20m→3m (6.67x) which may be unstable
- Still provides significant enhancement: 20m→6.67m = **3x improvement**

### Loss Function

`SpectralFidelityLoss` combines:
- **Pixel Loss**: L1 + MSE for accurate pixel values
- **Spectral Loss**: Spectral angle for band correlations
- **Perceptual Loss**: Gradient loss for edge preservation

---

## Model Architecture

### `SatelliteSRModel`

- **Feature Extraction**: 3×3 conv + ReLU
- **Residual Blocks**: Configurable (default: 8 blocks)
- **Temporal Fusion**: Attention mechanism for multi-frame sequences
- **Upsampling**: PixelShuffle for learned upscaling

**Parameters:**
- `base_channels`: Base feature channels (default: 64)
- `num_blocks`: Number of residual blocks (default: 8)
- `upscale_factor`: Upscaling factor (auto-detected)
- `sequence_length`: Temporal sequence length (default: 3)

---

## Expected Results

### Compared to Interpolated Approach

✅ **Lower training loss** - Learns real patterns, not artifacts  
✅ **Better PSNR/SSIM** on native test data  
✅ **More accurate NDVI/NDRE** - Preserved spectral integrity  
✅ **Temporal consistency** - Better time series analysis  
✅ **Better generalization** - Works on native 20m data

### For STA Applications

✅ **Biomass estimation** - Accurate from NDVI  
✅ **Vegetation health** - Reliable NDRE, chlorophyll status  
✅ **Phenology tracking** - Temporal NIR consistency  
✅ **Change detection** - Invasive species, die-off events

---

## Workflow Integration

### Current Status (Nov 23, 2024)

1. ✅ Multi-resolution training implemented
2. ✅ Documentation created
3. ✅ Comparison demo ready
4. 🔜 Dataset regeneration (after bug fix)
5. 🔜 Train multi-resolution models
6. 🔜 Validate on native test data
7. 🔜 Scale to 1440 samples
8. 🔜 Deploy for STA analysis

### After Dataset Generation

```bash
# 1. Generate dataset (with bug fix)
python scripts/preprocessing/generate_sta2_grid_parallel.py \
    --max-samples 400 \
    --coverage-threshold 90 \
    --coverage-workers 15

# 2. Train multi-resolution models
python scripts/training/train_multi_resolution_sr.py \
    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \
    --epochs 50

# 3. Validate results
python scripts/training/validate_sr_models.py \
    --models-dir sr_models_multi_res \
    --test-data data/datasets/sta2_grid_parallel_test/
```

---

## References

### Remote Sensing SR

- **Yokoya et al. (2017)**: "Hyperspectral and Multispectral Data Fusion" - Preserve native spectral characteristics
- **Lanaras et al. (2018)**: "Super-resolution of Sentinel-2 images" - Separate processing for mixed resolutions
- **Gargiulo et al. (2019)**: "Fast Super-Resolution of Multispectral Images" - Interpolation degrades vegetation indices

### STA-Specific

- NIR accuracy critical for wetland monitoring
- NDVI/NDRE for biomass and health
- Temporal analysis for seasonal patterns
- Change detection for ecosystem monitoring

---

## Troubleshooting

### Issue: Out of memory during training

**Solution:**
```bash
# Reduce batch size
python scripts/training/train_multi_resolution_sr.py \
    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \
    --batch-size 2  # Instead of 4

# Or reduce model size
python scripts/training/train_multi_resolution_sr.py \
    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \
    --num-blocks 4 \
    --base-channels 32
```

### Issue: Training too slow

**Solution:**
```bash
# Use fewer blocks or channels
python scripts/training/train_multi_resolution_sr.py \
    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \
    --num-blocks 6 \
    --base-channels 48
```

### Issue: Need to resume training

**Solution:**
```python
# Load checkpoint and continue
checkpoint = torch.load('sr_models_multi_res/checkpoint_10m_epoch_30.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## See Also

- [Multi-Resolution SR Documentation](../../docs/analysis/MULTI_RESOLUTION_SR_APPROACH.md)
- [Dataset Generation](../preprocessing/generate_sta2_grid_parallel.py)
- [Bug Fix Documentation](../../docs/analysis/BUG_FIX_LR_HR_DATE_MISMATCH.md)
