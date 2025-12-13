# Satellite Super-Resolution Training# Satellite Super-Resolution Training# Training Scripts



Training pipeline for 5-band Sentinel-2 super-resolution (10m/20m → 3m/6m).



## 🏆 Best ModelTraining pipeline for multi-resolution satellite imagery super-resolution (Sentinel-2 → Planet).Super-resolution model training for satellite imagery.



**MultiRes with Perceptual Loss** achieved the best performance:

- **Test PSNR: 36.73 dB** (+2.02 dB over bilinear baseline)

- **Test SSIM: 0.8935** (+4.4% over bilinear)## Directory Structure## Overview

- **RMSE: 0.0262** (-30% vs bilinear)



Train with: `python train_best_multires_sr.py --dataset full --epochs 200`

```This directory contains training scripts for SR models, including both **interpolated** (original) and **multi-resolution** (correct) approaches.

---

scripts/training/

## Directory Structure

├── train_satellite_sr.py      # Main unified training script---

```

scripts/training/├── models/

├── train_best_multires_sr.py   # 🏆 BEST MODEL - MultiRes + Perceptual Loss

├── training_config.py          # Shared configuration and hyperparameters│   ├── __init__.py## ⚠️ Important: B8A Interpolation Issue

│

├── models/                     # Model architectures│   └── satellite_sr_model.py  # Multi-resolution SR model

│   ├── satellite_sr_model.py   # Original SR model

│   ├── satellite_sr_model_v2.py # 🏆 MultiRes architecture (BEST)├── legacy/                    # Archive of previous implementations**Problem:** Sentinel-2's B8A band is natively 20m resolution. Interpolating it to 10m before training introduces:

│   └── satellite_sr_model_v3.py # V3 lightweight architecture

││   ├── ImprovedModel.py       # Advanced model components (reference)- Artificial smoothing and artifacts

├── losses/                     # Loss functions

│   ├── __init__.py│   ├── Model.py               # Earlier model version- Corrupted NIR spectral signals

│   └── perceptual_loss.py      # VGG-based perceptual loss

││   ├── train_*.py             # Various training scripts- Model learns interpolation, not true super-resolution

├── evaluation/                 # Model evaluation scripts

│   ├── test_models_comprehensive.py│   └── ...- Inflated metrics on interpolated test data

│   └── test_models_full_metrics.py  # PSNR, RMSE, SSIM, SCC, ERGAS, SID

│└── README.md                  # This file- Poor generalization to native 20m data

├── tuning/                     # Hyperparameter optimization

│   ├── hyperparameter_tuning.py     # Optuna-based tuning```

│   └── tune_multires_perceptual.py  # Tuning for best model

│**Solution:** Use `train_multi_resolution_sr.py` which trains separate models for 10m (RGB) and 20m (NIR) bands at their native resolutions.

├── experiments/                # Experimental training scripts

│   ├── train_5band_v2.py       # V2 model experiments## Quick Start

│   ├── train_5band_v3.py       # V3 model experiments

│   ├── train_v3_perceptual.py  # V3 + perceptual loss**Documentation:** See [MULTI_RESOLUTION_SR_APPROACH.md](../../docs/analysis/MULTI_RESOLUTION_SR_APPROACH.md)

│   ├── train_5band_multires.py # Early multires experiments

│   └── ...                     # Other experimental scripts### Basic Training (Current Data)

│

└── legacy/                     # Archived older implementations---

    ├── Model.py

    ├── ImprovedModel.pyWith the current dataset (interpolated B8A at 10m):

    └── train_*.py

```## Scripts



---```bash



## Model Performance Comparisoncd scripts/training### 1. `train_multi_resolution_sr.py` ✅ **RECOMMENDED**



| Model | PSNR (dB) | SSIM | RMSE | SCC | ERGAS | SID |

|-------|-----------|------|------|-----|-------|-----|

| **MultiRes (Full)** 🏆 | **36.73** | **0.8935** | **0.0262** | 0.1701 | **6.32** | 0.0440 |python train_satellite_sr.py \Train separate SR models for 10m and 20m bands (correct approach).

| MultiRes (Filtered) | 36.20 | 0.8755 | 0.0343 | **0.1715** | 7.83 | **0.0362** |

| V3 (Full) | 34.36 | 0.8786 | 0.0315 | 0.1367 | 7.73 | 0.0372 |    --data_dir /path/to/data/datasets \

| V3 (Filtered) | 33.99 | 0.8682 | 0.0368 | 0.1363 | 8.59 | 0.0372 |

| Bilinear Baseline | 34.71 | 0.8557 | 0.0379 | 0.1107 | 8.86 | 0.0375 |    --epochs 100 \**Features:**



---    --batch_size 8 \- Preserves native resolution characteristics



## Quick Start    --crop_size 64- No interpolation artifacts in training data



### Train Best Model```- Spectral integrity for vegetation indices

```bash

cd /media/eamurciab/TOSHIBA_EXT1/PhD/SR- Realistic validation metrics

python scripts/training/train_best_multires_sr.py --dataset full --epochs 200

```### Multi-Resolution Training (Native 20m B8A)- Better generalization



### Evaluate Models

```bash

python scripts/training/evaluation/test_models_full_metrics.pyAfter regenerating data with native 20m B8A:**Usage:**

```

```bash

### Hyperparameter Tuning

```bash```bash# Train both 10m and 20m models

python scripts/training/tuning/tune_multires_perceptual.py --n_trials 50

```python train_satellite_sr.py \python scripts/training/train_multi_resolution_sr.py \



---    --data_dir /path/to/data/datasets \    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \



## Model Architectures    --multi_resolution \    --output-dir sr_models_multi_res \



### MultiRes (V2) - Best Model    --epochs 100 \    --batch-size 4 \

- **Parameters:** 938,926

- **Architecture:** 64 channels, 4 encoder blocks    --batch_size 8 \    --epochs 50 \

- **Features:** Separate branches for 10m (3 bands) and 20m (2 bands)

- **Loss:** L1 + MSE + SAM + Gradient + Perceptual (VGG)    --crop_size 64    --lr 1e-4



### V3 - Lightweight```

- **Parameters:** 484,160

- **Architecture:** 48 channels, 3 encoder blocks# Train only 10m model (for testing)

- **Features:** Faster training, lower memory

- **Use case:** When resources are limited### Training on Specific Yearspython scripts/training/train_multi_resolution_sr.py \



---    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \



## Loss Function```bash    --train-10m-only \



Combined loss with multiple components:python train_satellite_sr.py \    --epochs 20

- **Pixel Loss:** L1 + 0.1*MSE for reconstruction accuracy

- **SAM Loss:** Spectral Angle Mapper for spectral fidelity    --data_dir /path/to/data/datasets \```

- **Gradient Loss:** Sobel-based edge preservation

- **Perceptual Loss:** VGG features for visual quality    --years "2020,2021,2022" \



Weights (tuned):    --epochs 100**Output:**

```python

alpha_pixel = 1.0```- `best_model_10m.pth` - Best 10m RGB model

alpha_spectral = 0.137

alpha_gradient = 0.01- `best_model_20m.pth` - Best 20m NIR model

alpha_perceptual = 0.1

```## Model Architecture- `final_model_10m.pth` - Final 10m model



---- `final_model_20m.pth` - Final 20m model



## Checkpoints### MultiResolutionSatelliteSR- Checkpoints every 10 epochs



Best models saved at:

```

checkpoints/For data with separate 10m and 20m bands:**Architecture:**

├── multires_perceptual_full/      # 🏆 Best model

│   ├── best_model.pt- **10m Model**: B2,B3,B4 (10m) → B2,B3,B4 (3m) | Scale: 3.33x

│   └── training_history.json

├── multires_perceptual_filtered/```- **20m Model**: B8A (20m) → B8A (6.67m) | Scale: 3x

├── 5band_v3_perceptual_full/

└── 5band_v3_perceptual_filtered/┌─────────────────────────────────────────────────────────────────────────┐

```

│                                                                         │---

---

│   LR 10m (B2,B3,B4)              LR 20m (B8A)                           │

## Data

│   (B, T, 3, H, W)                (B, T, 1, H/2, W/2)                    │### 2. `train_with_chunked_dataset.py` ⚠️ **OLD APPROACH**

Training data location:

```│        │                              │                                 │

data/sr_training/5band_v3/combined/

├── mmap/           # Full dataset (all samples)│        ▼                              ▼                                 │Original training script using interpolated B8A (kept for comparison).

├── mmap_filtered/  # Filtered dataset (correlation > 0.7)

└── mmap_test/      # Test set (111 samples)│   ┌─────────────┐               ┌─────────────┐                        │

```

│   │ Encoder 10m │               │ Encoder 20m │                        │**Issues:**

│   └─────────────┘               └─────────────┘                        │- ❌ Uses interpolated 20m→10m B8A data

│        │                              │                                 │- ❌ Learns interpolation artifacts, not true SR

│        ▼                              ▼                                 │- ❌ Corrupts NIR for vegetation analysis

│   ┌─────────────┐               ┌─────────────┐                        │- ❌ Inflated metrics on interpolated test sets

│   │  Temporal   │               │  Temporal   │                        │

│   │  Attention  │               │  Attention  │                        │**Usage:** (For comparison only)

│   └─────────────┘               └─────────────┘                        │```bash

│        │                              │                                 │python scripts/training/train_with_chunked_dataset.py \

│        └──────────┬──────────────────┘                                 │    --dataset-dir data/datasets/sta2_grid_parallel \

│                   ▼                                                     │    --output-dir sr_models_interpolated \

│            ┌─────────────┐                                              │    --batch-size 4 \

│            │   Feature   │ (Cross-resolution fusion)                   │    --epochs 50

│            │   Fusion    │                                              │```

│            └─────────────┘                                              │

│                   │                                                     │---

│        ┌──────────┴──────────┐                                         │

│        ▼                     ▼                                          │### 3. `compare_interpolation_approaches.py` 📊 **DEMO**

│   ┌─────────────┐      ┌─────────────┐                                 │

│   │ Upscaler 10m│      │ Upscaler 20m│                                 │Demonstrates why interpolating B8A is problematic.

│   │  (~3.33x)   │      │  (~3.33x)   │                                 │

│   └─────────────┘      └─────────────┘                                 │**Usage:**

│        │                      │                                         │```bash

│        ▼                      ▼                                         │python scripts/training/compare_interpolation_approaches.py

│   SR 10m bands           SR 20m band                                    │```

│   (B, 3, H', W')         (B, 1, H'/2, W'/2)                            │

│                                                                         │**Output:**

└─────────────────────────────────────────────────────────────────────────┘- Console: Detailed comparison and examples

```- `interpolation_artifact_demo.png` - Visual comparison

- Impact analysis on vegetation indices

### SingleResolutionSatelliteSR

---

For simpler use cases with all bands at same resolution:

## Quick Start

```

LR (B, T, 4, H, W) → Encoder → Temporal Attention → Upscaler → SR (B, 4, H', W')### After Dataset Generation

```

1. **Convert dataset** (automatic):

## Key Features```bash

python scripts/training/train_multi_resolution_sr.py \

### Model Features    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \

- **Dual-branch encoder**: Separate processing for 10m and 20m bands    --output-dir sr_models_multi_res

- **Temporal attention**: Learns to weight frames based on quality```

- **Cross-resolution fusion**: Enables information sharing between resolutions

- **Spectral consistency**: Preserves band relationships (important for vegetation indices)The script automatically:

- **Physics-constrained output**: Bounds reflectance values to valid range [0, 1]- Converts 4-channel data to multi-resolution format

- Removes interpolation artifacts

### Training Features- Trains both 10m and 20m models

- **Mixed precision (AMP)**: Memory-efficient training for GPUs with limited VRAM

- **Multi-year dataset**: Automatic discovery and loading of yearly datasets2. **Run comparison demo** (optional):

- **Temporal sequences**: Uses 3-frame sequences with temporal attention```bash

- **Data augmentation**: Flips, rotations, random crops (temporally consistent)python scripts/training/compare_interpolation_approaches.py

- **Learning rate scheduling**: Cosine annealing with warm restarts```

- **Checkpoint management**: Best model tracking and resumption support

3. **Inference** (after training):

## Loss Functions```python

import torch

The `MultiResolutionLoss` combines:from train_multi_resolution_sr import SatelliteSRModel



1. **L1 Loss** (weight=1.0): Pixel-wise reconstruction# Load models

2. **Spectral Angle Loss** (weight=0.1): Spectral fidelity preservationmodel_10m = SatelliteSRModel(in_channels=3, out_channels=3, upscale_factor=3)

3. **Edge Loss** (weight=0.1): Sharpness/detail preservationmodel_20m = SatelliteSRModel(in_channels=1, out_channels=1, upscale_factor=3)



## Hardware Requirementscheckpoint_10m = torch.load('sr_models_multi_res/best_model_10m.pth')

checkpoint_20m = torch.load('sr_models_multi_res/best_model_20m.pth')

### Minimum (RTX 4060 6GB)

- Batch size: 4-8model_10m.load_state_dict(checkpoint_10m['model_state_dict'])

- Crop size: 64x64model_20m.load_state_dict(checkpoint_20m['model_state_dict'])

- AMP enabled: Required

# Run inference

### Recommended (RTX 3080 10GB+)sr_rgb = model_10m(lr_10m_sequence)  # (B, 3, H*3.33, W*3.33)

- Batch size: 16-32sr_nir = model_20m(lr_20m_sequence)  # (B, 1, H/2*3, W/2*3)

- Crop size: 96-128```

- AMP enabled: Optional

---

## Dataset Format

## Technical Details

### Current Format (Interpolated B8A)

```### Multi-Resolution Approach

data/datasets/

├── STA2_2020/**10m Bands (B2, B3, B4):**

│   ├── sample_0000/- Native resolution: 10m

│   │   ├── lr_s2.npy       # (T, 4, H, W) - B2, B3, B4, B8A- Target resolution: 3m (Planet)

│   │   ├── hr_planet.npy   # (T, 4-6, H', W') - Planet HR- Scale factor: 3.33x

│   │   └── metadata.json- Channels: 3 (Blue, Green, Red)

│   └── ...

└── STA2_2021/**20m Band (B8A):**

    └── ...- Native resolution: 20m (no interpolation!)

```- Target resolution: 6.67m (Planet 3m downsampled 2x)

- Scale factor: 3x

### Native 20m Format (After Regeneration)- Channels: 1 (Near-Infrared)

```

data/datasets/**Why 6.67m target for 20m?**

├── STA2_2020/- Planet provides 3m NIR natively

│   ├── sample_0000/- Downsampling to 6.67m (2x) maintains proper scale relationship

│   │   ├── lr_10m.npy      # (T, 3, H, W) - B2, B3, B4 at 10m- Avoids over-ambitious 20m→3m (6.67x) which may be unstable

│   │   ├── lr_20m.npy      # (T, 1, H/2, W/2) - B8A at native 20m- Still provides significant enhancement: 20m→6.67m = **3x improvement**

│   │   ├── hr.npy          # (T, 4, H', W') - Planet HR

│   │   └── metadata.json### Loss Function

│   └── ...

└── ...`SpectralFidelityLoss` combines:

```- **Pixel Loss**: L1 + MSE for accurate pixel values

- **Spectral Loss**: Spectral angle for band correlations

## Training Monitoring- **Perceptual Loss**: Gradient loss for edge preservation



Checkpoints saved to `./runs/<timestamp>/`:---

- `checkpoint_epoch_XXXX.pt`: Periodic checkpoints

- `best_model.pt`: Best validation loss## Model Architecture

- `config.json`: Training configuration

- `tensorboard/`: TensorBoard logs (if available)### `SatelliteSRModel`



Launch TensorBoard:- **Feature Extraction**: 3×3 conv + ReLU

```bash- **Residual Blocks**: Configurable (default: 8 blocks)

tensorboard --logdir ./runs- **Temporal Fusion**: Attention mechanism for multi-frame sequences

```- **Upsampling**: PixelShuffle for learned upscaling



## Legacy Code**Parameters:**

- `base_channels`: Base feature channels (default: 64)

Previous implementations archived in `legacy/` for reference:- `num_blocks`: Number of residual blocks (default: 8)

- `upscale_factor`: Upscaling factor (auto-detected)

| File | Description | Status |- `sequence_length`: Temporal sequence length (default: 3)

|------|-------------|--------|

| `ImprovedModel.py` | Advanced attention mechanisms | Reference only |---

| `Model.py` | Earlier model architecture | Deprecated |

| `train_full_dataset.py` | TensorFlow training | Deprecated |## Expected Results

| `train_multi_resolution_sr.py` | Multi-res with interpolated B8A | Deprecated |

| `train_sr_fast.py` | Quick testing script | Deprecated |### Compared to Interpolated Approach

| `train_with_chunked_dataset.py` | PyTorch chunked training | Integrated |

✅ **Lower training loss** - Learns real patterns, not artifacts  

## Next Steps✅ **Better PSNR/SSIM** on native test data  

✅ **More accurate NDVI/NDRE** - Preserved spectral integrity  

1. **Test current data**: Run training with single-resolution model on existing dataset✅ **Temporal consistency** - Better time series analysis  

2. **Regenerate data**: Create native 20m B8A dataset (~6-8 hours with caching)✅ **Better generalization** - Works on native 20m data

3. **Multi-resolution training**: Train with proper dual-branch architecture

4. **Evaluation**: Implement PSNR/SSIM/SAM metrics on held-out test set### For STA Applications


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
