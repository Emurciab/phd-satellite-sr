# Model Architecture Comparison

## Overview

This document compares the two training approaches used in this project:

| Aspect | Legacy (`train_with_chunked_dataset.py`) | Current (`train_native_resolution.py`) |
|--------|------------------------------------------|----------------------------------------|
| Model | `ImprovedSatelliteSR` | `SingleResolutionSatelliteSR` |
| Input Resolution | 256×256 (upsampled from native) | 77×77 (native 10m) |
| Output Resolution | 256×256 | 256×256 |
| Scale Factor | 4× (PixelShuffle) | 3.33× (PixelShuffle + interpolation) |
| Temporal Handling | Simple concatenation + attention | Learned temporal attention weights |
| Loss Function | SpectralFidelityLoss (3 components) | SRLoss (2 components) |

---

## 1. Model Architectures

### Legacy: `ImprovedSatelliteSR`

```
Input: (B, T, 4, 256, 256) - LR sequence (upsampled)
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  Feature Extraction (per frame)   │
    │  Conv2d(4 → 64) + ReLU           │
    └──────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  Residual Blocks (8 blocks)       │
    │  [Conv3x3 + BN + ReLU] × 2 each  │
    │  + Skip connection                │
    └──────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  Temporal Fusion (if T > 1)       │
    │  Concat features → Conv1x1        │
    │  → Sigmoid attention on center    │
    └──────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  Upsampling (4× PixelShuffle)     │
    │  Conv2d(64 → 64×16) → Shuffle    │
    │  → Conv2d(64 → 4)                │
    └──────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  Optional: Interpolate to target  │
    │  (if HR size differs)             │
    └──────────────────────────────────┘
                    │
                    ▼
Output: (B, 4, H', W') - SR image
```

**Key Components:**
- **Feature Extraction**: Simple Conv2d + ReLU
- **Residual Blocks**: Standard ResNet-style with BatchNorm
- **Temporal Fusion**: Concatenate + sigmoid attention
- **Upsampling**: 4× PixelShuffle (fixed scale)
- **Output**: Raw values (no constraints)

**Parameters**: ~464K (with 8 blocks, 64 channels)

---

### Current: `SingleResolutionSatelliteSR`

```
Input: (B, T, 3, 77, 77) - LR sequence (native 10m)
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  EncoderBranch (per frame)        │
    │  ├─ Stem: Conv7x7 + GN + GELU    │
    │  ├─ ResidualBlocks × 4           │
    │  │   └─ DepthwiseSeparable Conv  │
    │  │   └─ CBAM (every 2nd block)   │
    │  └─ Output projection            │
    └──────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  TemporalAttention                │
    │  ├─ Per-frame encoder             │
    │  ├─ Global pooling → MLP         │
    │  ├─ Softmax attention weights    │
    │  └─ Weighted sum + fusion conv   │
    └──────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  SpectralConsistencyModule        │
    │  └─ Spectral attention (SE-like) │
    └──────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  ProgressiveUpscaler              │
    │  ├─ 2× PixelShuffle (×1 step)    │
    │  ├─ Output conv                   │
    │  └─ Interpolate to target_size   │
    └──────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  PhysicsConstrainedHead           │
    │  ├─ Conv layers                   │
    │  └─ Sigmoid × (max-min) + min    │
    │      (clamps to [0, 1])          │
    └──────────────────────────────────┘
                    │
                    ▼
Output: (B, 3, 256, 256) - SR image (constrained)
```

**Key Components:**
- **Encoder**: Depthwise separable convolutions (efficient), GroupNorm, CBAM attention
- **Temporal Attention**: Learned per-frame weights via MLP + Softmax
- **Spectral Consistency**: Squeeze-and-Excitation style attention
- **Upsampling**: Progressive PixelShuffle + bilinear interpolation to exact size
- **Output**: Physics-constrained via sigmoid (values in [0, 1])

**Parameters**: ~464K (with 4 encoder blocks, 64 channels)

---

## 2. Loss Functions

### Legacy: `SpectralFidelityLoss`

```python
total_loss = sza_weight × (α_pixel × pixel_loss + 
                           α_spectral × spectral_loss + 
                           α_perceptual × perceptual_loss)
```

| Component | Formula | Weight (default) |
|-----------|---------|------------------|
| **Pixel Loss** | L1 + 0.1×MSE | α_pixel = 1.0 |
| **Spectral Angle Loss** | 1 - mean(cos_sim(pred, target)) | α_spectral = 0.1 |
| **Perceptual/Gradient Loss** | L1(∇pred, ∇target) | α_perceptual = 0.01 |
| **SZA Weighting** | exp(-(SZA - 45°)² / (2×30²)) | Optional |

**Features:**
- Multi-component loss with tunable weights
- Solar Zenith Angle weighting (penalizes extreme sun angles)
- Spectral angle loss preserves band relationships
- Gradient loss for edge preservation

---

### Current: `SRLoss`

```python
total_loss = l1_weight × L1_loss + ssim_weight × gradient_loss
```

| Component | Formula | Weight (default) |
|-----------|---------|------------------|
| **L1 Loss** | mean(|pred - target|) | l1_weight = 1.0 |
| **Gradient Loss** | L1(∇x_pred, ∇x_target) + L1(∇y_pred, ∇y_target) | ssim_weight = 0.5 |

**Features:**
- Simpler 2-component loss
- Gradient loss approximates structural similarity (SSIM-like)
- No SZA weighting (all samples weighted equally)
- No explicit spectral loss

---

## 3. Key Differences

### 3.1 Input Resolution
| | Legacy | Current |
|--|--------|---------|
| LR Input | 256×256 (upsampled from ~77×77) | 77×77 (native) |
| HR Target | 256×256 | 256×256 |
| Information | Interpolated artifacts | True native resolution |

**Impact**: The legacy approach loses information during upsampling LR to match HR size. The current approach learns true super-resolution from native resolution.

### 3.2 Normalization
| | Legacy | Current |
|--|--------|---------|
| Method | In dataloader: `x / 10000` | Pre-normalized in dataset |
| Range | [0, 1] (clamped) | [0, 1] (pre-normalized) |

### 3.3 Temporal Handling
| | Legacy | Current |
|--|--------|---------|
| Method | Concat → Conv1x1 → Sigmoid | MLP → Softmax → Weighted sum |
| Weighting | Center frame emphasized | Learned weights per frame |
| Complexity | Simple | More sophisticated |

### 3.4 Architectural Choices
| Feature | Legacy | Current |
|---------|--------|---------|
| Convolution | Standard Conv2d | Depthwise Separable |
| Normalization | BatchNorm | GroupNorm |
| Activation | ReLU | GELU |
| Attention | None in encoder | CBAM in encoder |
| Output constraint | None | Sigmoid clamping |

### 3.5 Upsampling Strategy
| | Legacy | Current |
|--|--------|---------|
| Method | 4× PixelShuffle | 2× PixelShuffle + interpolation |
| Scale Handling | Fixed 4× | Flexible (any scale factor) |
| Interpolation | Only if needed | Always to target_size |

---

## 4. Training Configuration Comparison

| Parameter | Legacy | Current |
|-----------|--------|---------|
| Batch Size | 4 | 8 |
| Learning Rate | 1e-4 | 1e-4 |
| Optimizer | AdamW | AdamW |
| Scheduler | StepLR | CosineAnnealing |
| Mixed Precision | No | Yes (AMP) |
| Augmentation | No | Yes (flip, rotate) |

---

## 5. Recommendations

### When to use Legacy (`ImprovedSatelliteSR`):
- Quick prototyping
- When LR and HR are similar sizes
- When spectral fidelity is critical (has SAM loss)
- When sun angle weighting is important

### When to use Current (`SingleResolutionSatelliteSR`):
- Production training with native resolution data
- When efficiency matters (depthwise separable convs)
- When output must be in valid reflectance range [0, 1]
- When sophisticated temporal fusion is needed

---

## 6. Performance Notes

From training runs with 2,804 samples:

| Metric | Legacy (expected) | Current (observed) |
|--------|-------------------|-------------------|
| PSNR | ~25-27 dB | ~27-28 dB |
| SSIM | ~0.75-0.80 | ~0.84-0.85 |
| Training Time | ~2 min/epoch | ~1.5 min/epoch |

The current approach with native resolution data achieves better metrics because:
1. No information loss from LR upsampling
2. Model learns true 3.33× super-resolution
3. More sophisticated attention mechanisms
4. Physics-constrained output prevents invalid values
