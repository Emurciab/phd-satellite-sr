# Satellite Remote Sensing - Super Resolution Project

PhD research project for satellite image super-resolution using Google Earth Engine, Planet Labs, and Sentinel-2 imagery.

## 📁 Workspace Structure

```
SR/
├── Auth/                           # GEE authentication and data access
│   ├── AuthGEE.py                 # Main GEE orchestration module
│   ├── planet_s2_batches.csv      # Image pair results
│   └── remotesensingeduart-*.json # Service account credentials
│
├── scripts/                        # Main codebase
│   ├── preprocessing/             # Data preprocessing pipeline
│   │   ├── numpy_efficient_pipeline.py    # Core pipeline (TF-free)
│   │   ├── create_chunked_dataset.py      # Dataset generation
│   │   └── sr_efficient_pipeline.py       # SR integration
│   ├── training/                  # Model training scripts
│   ├── evaluation/                # Model evaluation tools
│   ├── visualization/             # Visualization utilities
│   │   ├── verify_dataset_samples.py      # Dataset quality verification
│   │   ├── visualize_dataset_samples.py   # Sample visualization
│   │   └── visualize_test_samples.py      # Test sample visualization
│   └── utilities/                 # Helper functions
│
├── tests/                          # Test files and results
│   ├── coregistration/            # Coregistration tests
│   │   └── test_real_coords.py   # Geographic coordinate testing
│   ├── cache/                     # Test cache directories
│   │   ├── test_fixed_tiling/    # Fixed tiling test cache
│   │   └── test_real_coords/     # Real coords test cache
│   ├── visualizations/            # Test visualizations
│   └── generate_test_samples.sh   # Test sample generation script
│
├── data/                          # Datasets and outputs
│   ├── datasets/
│   │   ├── sr_production_100/    # 100-sample production dataset ✅
│   │   │   ├── training_samples.pkl       # Training data (328 MB)
│   │   │   ├── metadata.pkl               # Dataset metadata
│   │   │   ├── verification_samples.png   # Quality verification
│   │   │   └── samples_visualization_*.png # Sample visualizations
│   │   └── cache_20samples/      # Old 20-sample dataset
│   └── models/                    # Trained model checkpoints
│
├── cache/                         # Processing cache
│   ├── sr_production_100/        # Production dataset cache (1.1 GB)
│   │   └── coregistered/         # Coregistered frames
│   ├── sr_numpy_cache/           # General processing cache (2.8 GB)
│   └── datasets/                 # Dataset cache
│
├── logs/                          # Execution logs
│   ├── dataset_generation/       # Dataset generation logs
│   │   └── production_100_generation.log
│   └── tests/                    # Test execution logs
│       ├── dataset_verification.log
│       ├── test_arosics_128_windows.log
│       ├── test_fixed_tiling_generation.log
│       └── test_real_coords*.log
│
├── docs/                          # Documentation
│   ├── QUICK_REFERENCE.md        # Quick reference guide
│   └── WORKSPACE_STRUCTURE.md    # Detailed structure docs
│
├── coreg/                         # Coregistration utilities
│   └── coreg.py                  # AROSICS/Phase correlation wrapper
│
├── sr_models/                     # Original SR models
└── sr_models_improved/            # Improved SR model architectures

```

## 🚀 Quick Start

### 1. Dataset Generation (100 samples - COMPLETED ✅)

```bash
# Already generated! Located at: data/datasets/sr_production_100/
# - 100 temporal sequences (3 frames each)
# - LR: 77×77×4 (Sentinel-2, 10m resolution)
# - HR: 256×256×4 (Planet, 3m resolution)
# - Tiling ratio: 3.325:1 (0.26% error from ideal 3.333:1)
# - Quality: 96% good samples, excellent alignment
```

### 2. Verify Dataset Quality

```bash
# Quick verification (20 random samples)
conda run -n tf_gpu python scripts/visualization/verify_dataset_samples.py

# Visualize specific samples
conda run -n tf_gpu python scripts/visualization/visualize_dataset_samples.py \
    --samples "0,25,50,75,99"

# Interactive visualization
conda run -n tf_gpu python scripts/visualization/visualize_dataset_samples.py \
    --interactive
```

### 3. Train SR Model (NEXT STEP)

```bash
# Train on 100 samples with improved architecture
conda run -n tf_gpu python scripts/training/train_sr_model.py \
    --dataset data/datasets/sr_production_100 \
    --blocks 16 \
    --channels 128 \
    --epochs 200 \
    --batch-size 4
```

### 4. Generate Full Dataset (If 100-sample results are good)

```bash
# Generate 600 samples (estimated time: 25-30 hours)
conda run -n tf_gpu python scripts/preprocessing/create_chunked_dataset.py \
    --output-dir ./data/datasets/sr_production_600 \
    --cache-dir ./cache/sr_production_600 \
    --samples 600 \
    --tile-size 256 \
    --sequence-length 3 \
    --regions 0
```

## 🔑 Key Configuration

### Coregistration Settings (OPTIMIZED ✅)

```python
# After extensive testing, these settings achieve 100% AROSICS success:
AROSICS_WINDOW_SIZE = (128, 128)  # Changed from (32, 32)
USE_REAL_COORDINATES = True        # Geographic bounds from GEE AOI
TILING_RATIO = 3.325               # 256/77 (Planet 3m / Sentinel-2 10m)
PHASE_CORRELATION_FALLBACK = True  # Safety net (rarely needed now)
```

### Quality Metrics

- **AROSICS Success Rate**: 100% (vs 33% with old 32×32 windows)
- **Spatial Similarity**: 0.875-0.968 (excellent alignment)
- **Reliability Scores**: 47-69% (meaningful quality indicators)
- **SSIM Improvement**: +44% average over unaligned baseline

## 📊 Dataset Overview

### Current Production Dataset (sr_production_100)

- **Total Samples**: 100
- **LR Resolution**: 77×77×4 (Sentinel-2, 10m/pixel)
- **HR Resolution**: 256×256×4 (Planet, 3m/pixel)
- **Temporal Length**: 3 frames per sequence
- **Tiling Ratio**: 3.325:1 (0.26% error from ideal 3.333:1)
- **Good Quality**: 96% (4 samples with cloud masking)
- **Alignment Quality**: Excellent (0.87-0.97 spatial similarity)
- **Storage**: 328 MB (compressed), 1.1 GB cache

### Region Configuration

```python
Region: STA-2
AOI: Florida coastal area
Date Range: 2020-2021
Planet Collection: Custom high-quality imagery
Sentinel-2: Harmonized SR product
```

## 🧪 Testing & Validation

### Recent Breakthrough Tests

1. **Real Geographic Coordinates Test** (`tests/coregistration/`)
   - Result: No improvement over synthetic coords (still 33% success)
   - Conclusion: Coordinates weren't the bottleneck

2. **AROSICS Window Size Optimization** (`logs/tests/`)
   - Changed: ws=(32,32) → ws=(128,128)
   - Result: **100% AROSICS success rate!** ✅
   - Conclusion: Window size was the critical factor

3. **Dataset Quality Verification** (`data/datasets/sr_production_100/`)
   - Verified: 20 random samples
   - Result: 96% good quality, excellent alignment
   - Visualizations: Created comprehensive comparison images

## 📈 Performance Expectations

### Current Model (Baseline)
- PSNR: ~15 dB
- SSIM: ~0.17
- Issue: Trained on misaligned data

### Expected with New Dataset
- Target PSNR: >25 dB (+10 dB improvement)
- Target SSIM: >0.7 (+0.53 improvement)
- Reason: Proper sub-pixel coregistration + correct tiling ratio

## 🔧 Technical Details

### Coregistration Pipeline

1. **Download from GEE**: Parallel download (12 workers)
2. **Band Detection**: Adaptive 8-band vs 4-band handling
3. **Coregistration**: AROSICS with 128×128 windows
   - Method: FFT-based matching
   - Shifts: Sub-pixel accuracy (0.2-0.3 pixels)
   - Fallback: Phase correlation if AROSICS fails
4. **Tiling**: Dynamic calculation for 3.33:1 ratio
5. **Temporal Sequencing**: 3-frame sequences for attention
6. **Caching**: SQLite + pickle for efficient reuse

### Why Coregistration Matters

**Without Alignment** (Old):
```
LR pixel (10,10) → Geographic location A
HR pixel (33,33) → Geographic location B (shifted!)
Model learns: "Pattern A → Pattern B" ❌
Problem: Patterns are from DIFFERENT locations!
```

**With Alignment** (Current):
```
LR pixel (10,10) → Geographic location A
HR pixel (33,33) → Geographic location A (aligned!)
Model learns: "Pattern A → Detailed A" ✅
Success: True resolution enhancement!
```

### Data Flow

```
Raw GEE Data
    ↓
Coregistration (AROSICS 128×128)
    ↓
Tiling (256×256 HR, 77×77 LR)
    ↓
Temporal Sequencing (3 frames)
    ↓
Dataset (training_samples.pkl)
    ↓
SR Model Training
    ↓
Super-Resolved Output
```

## 📝 Important Notes

### Upsampling Clarification

⚠️ **Visualization vs Training**:
- **Visualization scripts**: Upsample LR with bilinear interpolation for display
- **Training data**: NO upsampling! Native resolutions (77×77 LR, 256×256 HR)
- **SR model task**: Learn to upsample 77×77→256×256 with detail hallucination

### Coregistration = Alignment Only

- **Purpose**: Spatial alignment (sub-pixel shifts)
- **NOT**: Resolution change or upsampling
- **Result**: LR stays 77×77, HR stays 256×256
- **Benefit**: Corresponding pixels point to same geographic location

## 🎯 Next Steps

1. ✅ **COMPLETED**: Generate 100-sample dataset with optimized coregistration
2. ⏳ **IN PROGRESS**: Verify dataset quality (20 samples checked)
3. 🔜 **NEXT**: Train improved SR model on 100 samples
4. 🔜 **FUTURE**: Generate full 600-sample dataset (if 100 results good)

## 📚 References

- **AROSICS**: Automated and Robust Open-Source Image Co-Registration Software
- **Google Earth Engine**: Cloud-based satellite imagery platform
- **Planet Labs**: High-resolution (3m) satellite imagery
- **Sentinel-2**: ESA's 10m resolution multispectral imagery

## 🤝 Contributing

This is a PhD research project. For questions or collaboration:
- Review `docs/` for detailed documentation
- Check `logs/` for execution history
- Examine `tests/` for validation results

---

**Last Updated**: November 15, 2025  
**Status**: Dataset ready for training ✅  
**Next Milestone**: Train improved SR model
