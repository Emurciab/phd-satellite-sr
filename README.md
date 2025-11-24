# Satellite Remote Sensing - Super Resolution Pipeline

PhD research project for satellite image super resolution using multi-temporal PlanetScope and Sentinel-2 imagery.

## 🎯 Project Overview

This pipeline processes paired satellite images from PlanetScope (3m resolution) and Sentinel-2 (10m resolution) to create training data for deep learning-based super resolution models. The system includes:

- **Spatial coregistration** using AROSICS for sub-pixel alignment
- **Spectral harmonization** using histogram matching to correct sensor differences  
- **Temporal sequence processing** for attention-based architectures
- **Efficient chunked downloading** from Google Earth Engine
- **Automated quality assessment** and filtering

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Earth Engine account with service credentials
- Required packages: `earthengine-api`, `arosics`, `scikit-image`, `numpy`, `pandas`

### Generate Dataset

```bash
# Generate full STA-2 dataset (all periods 2020-2025)
python scripts/preprocessing/generate_sta2_full_dataset.py \
    --output-dir ./data/datasets/sta2_full \
    --max-samples 600 \
    --fresh  # Force fresh download
```

### Validate Results

```bash
# Quick harmonization test
python test_harmonization_standalone.py

# Visualize samples
python scripts/visualization/verify_sta2_dataset.py \
    --dataset-path ./data/datasets/sta2_full
```

## 📁 Project Structure

```
SR/
├── scripts/
│   ├── preprocessing/         # Main pipeline
│   │   ├── numpy_efficient_pipeline.py      # Core processing engine
│   │   ├── generate_sta2_full_dataset.py    # Dataset generator
│   │   ├── create_chunked_dataset.py        # Dataset wrapper
│   │   └── chunked_download_fast.py         # Optimized GEE downloader
│   ├── training/              # Model training
│   │   ├── train_with_chunked_dataset.py    # Main training script
│   │   ├── train_sr_fast.py                 # Fast training
│   │   └── train_sr_model.py                # Standard training
│   ├── validation/            # Dataset validation
│   │   ├── validate_chunked_dataset.py      # Dataset quality checks
│   │   ├── validate_real_data.py            # Real data validation
│   │   ├── validate_multiple_images.py      # Multi-image validation
│   │   └── analyze_fixed_dataset.py         # Dataset analysis
│   ├── evaluation/            # Model evaluation
│   │   ├── evaluate_sr_model.py             # Model metrics
│   │   └── test_sr_models.py                # Model comparison
│   └── visualization/         # Dataset inspection tools
│       ├── visualize_sta2_samples.py        # STA-2 visualization
│       ├── verify_sta2_dataset.py           # STA-2 verification
│       └── (3 more viz tools)
├── tests/                     # Unit and integration tests
├── docs/
│   ├── current/              # Active documentation
│   └── archive/              # Historical documentation
├── Auth/                      # GEE credentials & pairing logic
├── cache/                     # Runtime cache (coregistration, harmonization)
├── data/                      # Generated datasets
├── legacy/                    # Archived old implementations
└── coreg/                     # AROSICS coregistration module
```

## 🔧 Pipeline Architecture

### 1. Download from Google Earth Engine
- Chunked parallel download (9×9 grid, 12 workers)
- Automatic band detection (4-band or 8-band)
- Unmask operation (0 fill for masked pixels)
- Scale: 3m (Planet), 10m (Sentinel-2)

### 2. Spatial Coregistration (AROSICS)
- **Reference**: Sentinel-2 (LR) fixed
- **Aligned**: PlanetScope (HR) shifted to match
- **Window size**: 128×128 pixels (optimal for cross-sensor)
- **Success rate**: 100%
- **Cache**: `cache/*/coregistered/`

### 3. Spectral Harmonization (NEW!)
- **Method**: Histogram matching (CDF-based)
- **Per-band processing**: Independent spectral adjustment
- **Direction**: PlanetScope → Sentinel-2 distribution
- **Performance**: 99.9% reduction in spectral mismatch
- **Cache**: `cache/*/harmonized/`

### 4. Band Count Harmonization
- Ensure LR and HR have matching band counts
- Take minimum of available bands

### 5. Tiling
- **Strategy**: Non-overlapping grid
- **Ratio**: 3.33:1 (actual, matches sensor resolutions)
- **HR tiles**: 256×256 pixels
- **LR tiles**: 77×77 pixels (adaptive based on actual ratio)

### 6. Training Data Output
- Format: TrainingSample objects with temporal sequences
- Structure: `(T, H, W, C)` for both LR and HR
- Storage: Pickle format with SQLite index

## 📊 Dataset Status

**Current Progress**:
- ✅ STA-2 Period 1 (2020-2021): 1,440 samples WITHOUT harmonization
- 🔄 **Need regeneration** to include spectral harmonization

**Target**:
- Full STA-2 (2020-2025): ~6,000-7,000 samples
- Quality: 63.9% good, 36.1% cloudy (automatic filtering)

## 📖 Documentation

### Current/Active Documentation
- [`SPECTRAL_HARMONIZATION.md`](docs/current/SPECTRAL_HARMONIZATION.md) - Histogram matching implementation
- [`COREGISTRATION_COMPARISON.md`](docs/current/COREGISTRATION_COMPARISON.md) - AROSICS vs Phase correlation
- [`TILING_FIX_SUMMARY.md`](docs/current/TILING_FIX_SUMMARY.md) - Resolution ratio implementation
- [`WORKSPACE_STRUCTURE.md`](docs/current/WORKSPACE_STRUCTURE.md) - Project organization
- [`QUICK_REFERENCE.md`](docs/current/QUICK_REFERENCE.md) - Command reference

### Key Concepts
- [`DATE_MATCHING_EXPLAINED.md`](docs/current/DATE_MATCHING_EXPLAINED.md) - Temporal pairing
- [`TEMPORAL_SEQUENCES_EXPLAINED.md`](docs/current/TEMPORAL_SEQUENCES_EXPLAINED.md) - Sequence generation
- [`WHAT_IS_A_SAMPLE.md`](docs/current/WHAT_IS_A_SAMPLE.md) - Data structure

### Training
- [`TRAINING_WORKFLOW.md`](docs/current/TRAINING_WORKFLOW.md) - Model training guide

## 🧪 Testing

```bash
# Test spectral harmonization
python test_harmonization_standalone.py

# Test coregistration comparison
python tests/compare_arosics_vs_phase.py

# Test tiling implementation
python tests/verify_tiling_fix.py
```

## 🎯 Next Steps

1. **Regenerate STA-2 dataset** with spectral harmonization enabled
2. **Train SR model** on harmonized data (target: PSNR >25 dB, SSIM >0.7)
3. **Compare performance** with non-harmonized baseline
4. **Extend to additional regions** if needed

## 📝 Citations

- **AROSICS**: Scheffler et al. (2017) - "AROSICS: An Automated and Robust Open-Source Image Co-Registration Software for Multi-Sensor Satellite Data"
- **PlanetScope**: Planet Labs (2021) - "Planet Imagery Product Specifications"
- **Sentinel-2**: ESA Copernicus Programme

## 📧 Contact

PhD Research Project - Satellite Remote Sensing
For questions or collaboration, see project documentation in `docs/current/`.

---

**Last Updated**: November 2025  
**Status**: Active development - spectral harmonization implementation complete
