# SR Workspace Structure

## Directory Organization

```
SR/
├── Auth/                           # GEE Authentication & Configuration
│   ├── AuthGEE.py                  # Main GEE processing functions
│   ├── remotesensingeduart-*.json  # Service account credentials
│   └── planet_s2_batches*.csv      # Paired image metadata
│
├── docs/                           # Documentation
│   ├── README.md                   # Main documentation
│   ├── TRAINING_WORKFLOW.md        # End-to-end training guide
│   ├── TRAINING_INTEGRATION_COMPLETE.md
│   ├── SETUP_GUIDE.md              # Environment setup
│   ├── EFFICIENT_SR_SUMMARY.md     # Pipeline summary
│   ├── NEXT_STEPS.md               # Action items
│   └── [other documentation]
│
├── scripts/                        # All Python scripts
│   ├── preprocessing/              # Data acquisition & preprocessing
│   │   ├── numpy_efficient_pipeline.py      # Main pipeline (chunked downloads)
│   │   ├── create_chunked_dataset.py        # Dataset generation
│   │   ├── chunked_download_fast.py         # Parallel chunked download
│   │   ├── sr_efficient_pipeline.py         # Alternative pipeline
│   │   ├── Preprocessing.py                 # Legacy preprocessing
│   │   └── [other preprocessing scripts]
│   │
│   ├── training/                   # Model training
│   │   ├── train_with_chunked_dataset.py    # Main training script
│   │   ├── train_sr_model.py                # Alternative trainer
│   │   ├── ImprovedModel.py                 # SR model architecture
│   │   └── Model.py                         # Legacy model
│   │
│   ├── validation/                 # Data & model validation
│   │   ├── validate_chunked_dataset.py      # Dataset quality checks
│   │   ├── validate_real_data.py            # Image validation
│   │   └── validate_multiple_images.py      # Batch validation
│   │
│   ├── testing/                    # Testing & demos
│   │   ├── test_integrated_pipeline.py      # Integration test
│   │   ├── demo_training_integration.py     # Demo with synthetic data
│   │   ├── quick_check.py                   # Quick data check
│   │   └── [other test scripts]
│   │
│   ├── utilities/                  # Helper scripts
│   │   ├── performance_analysis.py          # Performance metrics
│   │   ├── calculate_chunk_size.py          # Chunk size calculator
│   │   ├── monitor_integration_test.py      # Test monitor
│   │   └── [other utilities]
│   │
│   └── setup/                      # Setup & initialization
│       ├── setup_tf_gpu_env.sh              # Environment setup
│       └── monitor_test.sh                  # Test monitoring
│
├── cache/                          # All cache directories
│   ├── test_cache/                 # Integration test cache
│   ├── sr_numpy_cache/             # Pipeline cache
│   ├── debug_cache/                # Debug cache
│   └── [other cache dirs]
│
├── data/                           # Data & models
│   ├── datasets/                   # Generated datasets
│   │   ├── demo_dataset/           # Synthetic demo data
│   │   ├── quick_test_dataset/     # Quick test (2 samples)
│   │   ├── sr_dataset_100/         # Small production (100 samples)
│   │   └── sr_dataset_full/        # Full production dataset
│   │
│   └── models/                     # Trained models
│       ├── checkpoints/            # Training checkpoints
│       ├── sr_models_100/          # Models from 100 samples
│       └── [other model dirs]
│
├── logs_archive/                   # Historical logs
│   ├── integrated_test_output*.log
│   ├── validation_results.log
│   └── [other logs]
│
├── coreg/                          # Coregistration utilities
│   └── coreg.py
│
└── [root files]
    ├── .gitignore
    └── WORKSPACE_STRUCTURE.md      # This file
```

## Quick Navigation

### 🚀 **Getting Started**
1. Read: `docs/README.md`
2. Setup: `scripts/setup/setup_tf_gpu_env.sh`
3. Guide: `docs/TRAINING_WORKFLOW.md`

### 📊 **Create Dataset**
```bash
python scripts/preprocessing/create_chunked_dataset.py --samples 100 --regions 0
```

### 🏋️ **Train Model**
```bash
python scripts/training/train_with_chunked_dataset.py --dataset-dir ./data/datasets/sr_dataset_100
```

### ✅ **Validate Data**
```bash
python scripts/validation/validate_chunked_dataset.py --dataset-dir ./data/datasets/sr_dataset_100
```

### 🧪 **Run Tests**
```bash
# Quick integration test
python scripts/testing/test_integrated_pipeline.py

# Demo with synthetic data
python scripts/testing/demo_training_integration.py
```

## Key Files

### **Core Pipeline** (Most Important)
- `scripts/preprocessing/numpy_efficient_pipeline.py` - Main data pipeline with chunked downloads
- `scripts/preprocessing/create_chunked_dataset.py` - Dataset generation
- `scripts/training/train_with_chunked_dataset.py` - Model training
- `Auth/AuthGEE.py` - GEE preprocessing and authentication

### **Documentation** (Start Here)
- `docs/README.md` - Main project documentation
- `docs/TRAINING_WORKFLOW.md` - Complete workflow guide
- `docs/TRAINING_INTEGRATION_COMPLETE.md` - Integration status

### **Validation** (Quality Assurance)
- `scripts/validation/validate_chunked_dataset.py` - 5-test quality suite
- `scripts/validation/validate_real_data.py` - Image validation (8 tests)

## Workflow Examples

### **Quick Test (2 samples, ~5 minutes)**
```bash
cd /media/eamurciab/TOSHIBA_EXT1/PhD/SR

# Create test dataset
python scripts/preprocessing/create_chunked_dataset.py --quick-test

# Validate
python scripts/validation/validate_chunked_dataset.py --dataset-dir ./data/datasets/quick_test_dataset

# Train (requires PyTorch)
python scripts/training/train_with_chunked_dataset.py --quick-test
```

### **Small Production Dataset (100 samples, ~18 hours)**
```bash
# Generate dataset
python scripts/preprocessing/create_chunked_dataset.py \
    --output-dir ./data/datasets/sr_dataset_100 \
    --samples 100 \
    --regions 0

# Validate
python scripts/validation/validate_chunked_dataset.py \
    --dataset-dir ./data/datasets/sr_dataset_100

# Train
python scripts/training/train_with_chunked_dataset.py \
    --dataset-dir ./data/datasets/sr_dataset_100 \
    --output-dir ./data/models/sr_models_100 \
    --epochs 50
```

## Cache Management

### **Clear Old Caches**
```bash
# Clear test caches (safe to remove)
rm -rf cache/test_*

# Clear debug caches
rm -rf cache/debug_*

# Keep production caches (saves time)
# cache/sr_numpy_cache/
# cache/sr_training_cache/
```

### **Cache Locations**
- **Test caches**: `cache/test_*` - Temporary, can be deleted
- **Production cache**: `cache/sr_numpy_cache/` - Keep for faster regeneration
- **Training cache**: `cache/sr_training_cache/` - Keep if training repeatedly

## Import Paths

After reorganization, update imports:

### **From root scripts**
```python
# Old
from numpy_efficient_pipeline import TensorFlowFreeDataPipeline

# New
import sys
sys.path.insert(0, 'scripts/preprocessing')
from numpy_efficient_pipeline import TensorFlowFreeDataPipeline
```

### **From Auth**
```python
# No change needed
from Auth.AuthGEE import regions, find_pairs
```

## Maintenance

### **Cleanup**
```bash
# Remove old logs (keep recent)
find logs_archive/ -name "*.log" -mtime +30 -delete

# Remove temporary test caches
rm -rf cache/test_*

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
```

### **Backup Important Files**
- `Auth/remotesensingeduart-*.json` (credentials)
- `Auth/planet_s2_batches.csv` (paired metadata)
- `data/datasets/` (generated datasets)
- `data/models/` (trained models)

## Notes

- **Integration test running**: `scripts/testing/test_integrated_pipeline.py` is currently executing (PID 455003)
- **Cache preserved**: All cache directories moved to `cache/` intact
- **Documentation centralized**: All `.md` files in `docs/`
- **Scripts organized by function**: preprocessing → training → validation → testing

---
**Last Updated**: November 6, 2025
**Reorganization**: Completed automatically with REORGANIZATION.sh
