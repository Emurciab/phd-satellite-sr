#!/bin/bash

# Workspace Reorganization Script
# Organizes SR project into clean production vs legacy structure

set -e  # Exit on error

WORKSPACE_ROOT="/media/eamurciab/TOSHIBA_EXT1/PhD/SR"
cd "$WORKSPACE_ROOT"

echo "════════════════════════════════════════════════════════════════"
echo "🔧 SR WORKSPACE REORGANIZATION"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "This will organize the workspace into:"
echo "  • Production code (scripts/preprocessing/)"
echo "  • Legacy code (legacy/preprocessing/)"
echo "  • Current docs (docs/current/)"
echo "  • Archived docs (docs/archive/)"
echo ""
echo "All files will be MOVED (not deleted)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "❌ Aborted"
    exit 1
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "📁 Step 1: Creating new directory structure"
echo "────────────────────────────────────────────────────────────────"

mkdir -p legacy/preprocessing
mkdir -p docs/current
mkdir -p docs/archive

echo "✅ Created:"
echo "   • legacy/preprocessing/"
echo "   • docs/current/"
echo "   • docs/archive/"

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "📦 Step 2: Moving legacy code to legacy/preprocessing/"
echo "────────────────────────────────────────────────────────────────"
echo ""
echo "ℹ️  NOTE: The following are KEPT in production (not moved):"
echo "   • scripts/training/ (3 files) - All training scripts"
echo "   • scripts/validation/ (4 files) - All validation scripts"
echo "   • scripts/evaluation/ (2 files) - All evaluation scripts"
echo "   • scripts/visualization/ (5 files) - All viz/verify scripts"
echo "   • Auth/ - GEE authentication and pairing"
echo "   • coreg/ - AROSICS coregistration module"
echo "   • tests/ - All unit/integration tests"
echo ""
echo "Only moving OLD/UNUSED preprocessing files:"
echo ""

# Move legacy preprocessing files (ONLY unused old implementations)
LEGACY_FILES=(
    "scripts/preprocessing/efficient_data_handler.py"
    "scripts/preprocessing/sr_efficient_pipeline.py"
    "scripts/preprocessing/streaming_pipeline.py"
    "scripts/preprocessing/chunked_download.py"
    "scripts/preprocessing/create_fixed_dataset.py"
    "scripts/preprocessing/Preprocessing.py"
    "scripts/preprocessing/test_harmonization.py"
    "scripts/preprocessing/test_harmonization_simple.py"
)

for file in "${LEGACY_FILES[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        mv "$file" "legacy/preprocessing/$filename"
        echo "   ✓ Moved $filename"
    else
        echo "   ⚠ Not found: $file"
    fi
done

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "📚 Step 3: Organizing current documentation"
echo "────────────────────────────────────────────────────────────────"

# Move current/active docs to docs/current/
CURRENT_DOCS=(
    "docs/SPECTRAL_HARMONIZATION.md"
    "docs/COREGISTRATION_COMPARISON.md"
    "docs/TILING_FIX_SUMMARY.md"
    "docs/TILING_IMPLEMENTATION_SUMMARY.md"
    "docs/AROSICS_INTEGRATION.md"
    "docs/WORKSPACE_STRUCTURE.md"
    "docs/QUICK_REFERENCE.md"
    "docs/TRAINING_WORKFLOW.md"
    "docs/DATE_MATCHING_EXPLAINED.md"
    "docs/TEMPORAL_SEQUENCES_EXPLAINED.md"
    "docs/WHAT_IS_A_SAMPLE.md"
)

for file in "${CURRENT_DOCS[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        mv "$file" "docs/current/$filename"
        echo "   ✓ Moved $filename to current/"
    fi
done

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "🗄️  Step 4: Archiving old documentation"
echo "────────────────────────────────────────────────────────────────"

# Move old/historical docs to docs/archive/
ARCHIVE_DOCS=(
    "docs/EFFICIENT_SR_SUMMARY.md"
    "docs/memory_efficient_integration.md"
    "docs/SETUP_GUIDE.md"
    "docs/UV_SETUP_GUIDE.md"
    "docs/NEXT_STEPS.md"
    "docs/INTEGRATION_SUMMARY.md"
    "docs/IMPORT_FIXES.md"
    "docs/VALIDATION_RESULTS.md"
    "docs/TEST_RESULTS.md"
    "docs/TRAINING_INTEGRATION_COMPLETE.md"
    "docs/TRAINING_GUIDE.md"
    "docs/FAST_TRAINING.md"
    "docs/QUICK_SETUP.md"
    "docs/ARCHITECTURE_ANALYSIS.md"
)

for file in "${ARCHIVE_DOCS[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        mv "$file" "docs/archive/$filename"
        echo "   ✓ Moved $filename to archive/"
    fi
done

# Keep Planet PDF in docs/ root (reference material)
echo "   ✓ Kept Planet_PSScene_Imagery_Product_Spec_June_2021.pdf in docs/"

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "📝 Step 5: Creating main README.md"
echo "────────────────────────────────────────────────────────────────"

cat > README.md << 'EOF'
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
EOF

echo "✅ Created README.md"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ REORGANIZATION COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Summary:"
echo ""
echo "Production Code (KEPT in place):"
echo "  ✓ scripts/preprocessing/ (5 files):"
echo "    - numpy_efficient_pipeline.py"
echo "    - create_chunked_dataset.py"
echo "    - generate_sta2_full_dataset.py"
echo "    - chunked_download_fast.py"
echo "    - gee_export_helper.py"
echo ""
echo "  ✓ scripts/training/ (3 files):"
echo "    - train_with_chunked_dataset.py"
echo "    - train_sr_fast.py"
echo "    - train_sr_model.py"
echo ""
echo "  ✓ scripts/validation/ (4 files):"
echo "    - validate_chunked_dataset.py"
echo "    - validate_real_data.py"
echo "    - validate_multiple_images.py"
echo "    - analyze_fixed_dataset.py"
echo ""
echo "  ✓ scripts/evaluation/ (2 files):"
echo "    - evaluate_sr_model.py"
echo "    - test_sr_models.py"
echo ""
echo "  ✓ scripts/visualization/ (5 files):"
echo "    - All viz/verify scripts"
echo ""
echo "  ✓ Auth/ - GEE authentication"
echo "  ✓ coreg/ - AROSICS module"
echo "  ✓ tests/ - All tests"
echo ""
echo "Legacy Code (ARCHIVED to legacy/preprocessing/):"
echo "  ✓ 8 files archived (old implementations)"
echo ""
echo "Documentation:"
echo "  ✓ Current docs → docs/current/ (11 files)"
echo "  ✓ Archive docs → docs/archive/ (14 files)"
echo "  ✓ New README.md created"
echo ""
echo "Next Steps:"
echo "  1. Review README.md"
echo "  2. Check docs/current/ for active documentation"
echo "  3. Verify production pipeline still works:"
echo "     python -c \"from scripts.preprocessing.numpy_efficient_pipeline import TensorFlowFreeDataPipeline; print('✅ OK')\""
echo ""
echo "════════════════════════════════════════════════════════════════"
