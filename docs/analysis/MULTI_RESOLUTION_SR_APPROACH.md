# Multi-Resolution Super-Resolution Training

**Date:** November 23, 2024  
**Issue:** Interpolating 20m B8A band to 10m is technically incorrect for SR training  
**Solution:** Train separate models for native resolutions

---

## Problem: Interpolation Artifacts in Training Data

### Current Approach (Incorrect)

The pipeline currently resamples Sentinel-2's 20m B8A band (Near-Infrared) to 10m to match RGB bands (B2, B3, B4) before training:

```python
# In GEE preprocessing
lr_resampled = lr_selected.reproject(crs='EPSG:4326', scale=10)  # Forces B8A to 10m
```

**Why This Is Wrong:**

1. **Artificial Smoothing**: Bilinear/bicubic interpolation introduces smoothing that doesn't represent true sensor degradation

2. **Spectral Corruption**: NIR signals are critical for vegetation indices (NDVI, NDRE) in STAs. Interpolation alters these values

3. **Learned Artifacts**: The SR model learns to "upscale" already-interpolated data, not native 20m characteristics

4. **Biased Generalization**: Model trained on interpolated 10m won't generalize to native 20m B8A data

5. **Inflated Metrics**: PSNR/SSIM on interpolated test sets are misleadingly high (model reproduces interpolation, not true super-resolution)

### Example Impact on Vegetation Analysis

```
Original B8A (20m native):  [8500, 8200, 7900, ...]
After interpolation to 10m: [8500, 8350, 8200, 8050, 7900, ...]  ← Artificial values!

Model learns:
  Input:  [8500, 8350, 8200, ...]  (interpolated LR)
  Output: [8520, 8340, 8180, ...]  (tries to match interpolated HR)

But native data behavior is different:
  True 20m:    [8500, 8200, 7900, ...]
  True 3m:     [8510, 8490, 8470, 8220, 8190, 8160, ...]
```

---

## Solution: Multi-Resolution Training

### Architecture Overview

Train **two separate SR models** that respect native sensor resolutions:

```
┌─────────────────────────────────────────────────────────────┐
│                  MULTI-RESOLUTION SR SYSTEM                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │   10m SR Model       │      │   20m SR Model       │    │
│  │                      │      │                      │    │
│  │  Input:  B2,B3,B4    │      │  Input:  B8A         │    │
│  │          at 10m      │      │          at 20m      │    │
│  │                      │      │                      │    │
│  │  Target: B2,B3,B4    │      │  Target: B8A         │    │
│  │          at 3m       │      │          at 6.67m    │    │
│  │                      │      │                      │    │
│  │  Scale: 3.33x        │      │  Scale: 3x           │    │
│  └──────────────────────┘      └──────────────────────┘    │
│                                                               │
│  At Inference:                                               │
│  ├─ Run both models independently                           │
│  ├─ Concatenate outputs: [SR_RGB (3m), SR_NIR (6.67m)]     │
│  └─ Result: Full 4-band SR image with proper resolution    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Scale Factor Rationale

**10m Bands (B2, B3, B4):**
- LR: 10m resolution
- HR: 3m resolution (Planet)
- Scale factor: 10/3 ≈ **3.33x**
- Model learns: Native 10m → Native 3m mapping

**20m Band (B8A):**
- LR: 20m resolution (native, no interpolation)
- HR: 6.67m resolution (Planet downsampled 2x from 3m)
- Scale factor: 20/6.67 ≈ **3x**
- Model learns: Native 20m → Near-native upscaling

**Why 6.67m for HR target?**
- Planet provides 3m NIR natively
- Downsampling to 6.67m (2x) maintains proper scale relationship
- Avoids over-ambitious 20m→3m (6.67x) which may be unstable
- Still provides significant enhancement: 20m→6.67m = **3x improvement**

---

## Implementation

### 1. Dataset Conversion

Convert existing `sta2_grid_parallel` dataset to multi-resolution format:

```python
from train_multi_resolution_sr import prepare_multi_resolution_dataset

prepare_multi_resolution_dataset(
    dataset_path='data/datasets/sta2_grid_parallel/dataset.pkl',
    output_path='data/datasets/sta2_grid_parallel_multi_res.pkl'
)
```

**What it does:**
- Splits 4-channel LR into: `lr_10m` (3 channels) + `lr_20m` (1 channel)
- Downsamples interpolated B8A back to native 20m (removes artifact)
- Splits 4-channel HR into: `hr_10m` (3 channels) + `hr_20m` (1 channel)
- Downsamples HR NIR from 3m to 6.67m for 20m model target

### 2. Training

Train both models:

```bash
# Train both 10m and 20m models
python scripts/training/train_multi_resolution_sr.py \
    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \
    --output-dir sr_models_multi_res \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-4
```

Train only 10m model (for testing):

```bash
python scripts/training/train_multi_resolution_sr.py \
    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \
    --train-10m-only \
    --epochs 20
```

Skip conversion if already done:

```bash
python scripts/training/train_multi_resolution_sr.py \
    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \
    --skip-conversion
```

### 3. Inference

Load and use both models:

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

# Inference
sr_rgb = model_10m(lr_10m_sequence)   # (B, 3, H*3.33, W*3.33)
sr_nir = model_20m(lr_20m_sequence)   # (B, 1, H/2*3, W/2*3)

# Combine (align resolutions as needed)
sr_combined = combine_multi_resolution(sr_rgb, sr_nir)
```

---

## Technical Validation

### Why This Approach Is Correct

1. **Native Resolution Preservation**
   - 10m bands stay at 10m until SR model processes them
   - 20m band stays at 20m (no artificial interpolation)
   - Each model learns genuine sensor characteristics

2. **Proper Scale Factor**
   - 10m model: 3.33x upscaling (10m→3m)
   - 20m model: 3x upscaling (20m→6.67m)
   - Both are realistic, achievable scale factors

3. **Spectral Integrity**
   - NIR values not corrupted by pre-interpolation
   - Vegetation indices (NDVI, NDRE) computed from genuine SR
   - Critical for STA wetland analysis

4. **Realistic Evaluation**
   - Metrics computed on native-resolution predictions
   - Model performance reflects true SR capability
   - Validation on native test data is meaningful

### Comparison: Interpolated vs Native

| Aspect | Interpolated (Wrong) | Multi-Resolution (Correct) |
|--------|----------------------|----------------------------|
| B8A training input | 10m (interpolated) | 20m (native) |
| Learned features | Interpolation artifacts | True 20m characteristics |
| NIR spectral accuracy | Corrupted | Preserved |
| NDVI/NDRE quality | Biased | Accurate |
| Test metrics | Inflated | Realistic |
| Generalization | Poor on native data | Good on native data |
| STA applicability | Questionable | Reliable |

---

## Expected Benefits

### For Model Performance

1. **Better Generalization**: Models learn native degradation, not interpolation artifacts
2. **Spectral Fidelity**: NIR preserved for accurate vegetation analysis
3. **Realistic Metrics**: PSNR/SSIM reflect true SR capability
4. **Stable Training**: Each model optimized for its native resolution

### For STA Wetland Analysis

1. **Accurate NDVI**: $(NIR - Red) / (NIR + Red)$ uses genuine SR NIR
2. **Accurate NDRE**: $(NIR - RedEdge) / (NIR + RedEdge)$ for vegetation health
3. **Temporal Consistency**: NIR changes reflect real phenology, not interpolation noise
4. **Change Detection**: Subtle vegetation changes captured accurately

---

## Next Steps

### Immediate

1. ✅ Implement multi-resolution training script
2. 🔜 Convert existing 400 samples (after bug fix regeneration)
3. 🔜 Train both models (10m + 20m)
4. 🔜 Compare metrics: Interpolated vs Multi-Resolution

### Future Enhancements

1. **Joint Training**: Train both models together with shared features
2. **Cross-Resolution Attention**: Let 10m and 20m models exchange information
3. **Progressive Upscaling**: 20m→10m→3m in stages
4. **Uncertainty Estimation**: Per-band confidence maps

---

## File Organization

New files created:

```
scripts/training/
├── train_with_chunked_dataset.py      # Original (interpolated approach)
└── train_multi_resolution_sr.py       # NEW (native resolution approach)

docs/analysis/
└── MULTI_RESOLUTION_SR_APPROACH.md    # This documentation
```

---

## References & Rationale

### Remote Sensing SR Literature

1. **Yokoya et al. (2017)**: "Hyperspectral and Multispectral Data Fusion"
   - Emphasizes preserving native spectral characteristics
   - Warns against pre-processing that alters band relationships

2. **Lanaras et al. (2018)**: "Super-resolution of Sentinel-2 images"
   - Addresses mixed 10m/20m resolution explicitly
   - Recommends separate processing for different resolutions

3. **Gargiulo et al. (2019)**: "Fast Super-Resolution of Multispectral Images"
   - Demonstrates importance of spectral fidelity in agriculture
   - Shows interpolation degrades vegetation indices

### Wetland-Specific Concerns

For STAs (Stormwater Treatment Areas), NIR accuracy is critical:

- **NDVI**: Biomass estimation, vegetation health
- **NDRE**: Nitrogen status, chlorophyll content
- **Temporal Analysis**: Phenology, seasonal patterns
- **Change Detection**: Invasive species, die-off events

Interpolating B8A compromises all these applications.

---

**Status:** Ready for implementation after dataset regeneration (bug fix)  
**Priority:** High - Affects model validity for STA analysis  
**Impact:** Enables scientifically rigorous SR for wetland monitoring
