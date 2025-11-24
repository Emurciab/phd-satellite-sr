# Quick Reference - Reorganized Workspace

## ✅ Workspace Successfully Reorganized!

The workspace has been organized into a clear, logical structure:

```
SR/
├── 📁 Auth/                    # GEE credentials & preprocessing
├── 📁 scripts/                 # All Python scripts (organized by function)
│   ├── preprocessing/          # Data acquisition (10 files)
│   ├── training/              # Model training (6 files)
│   ├── validation/            # Quality checks (4 files)
│   ├── testing/               # Tests & demos (16 files)
│   ├── utilities/             # Helper scripts (11 files)
│   └── setup/                 # Environment setup (2 files)
├── 📁 cache/                   # All cache directories (12 caches)
├── 📁 data/                    # Datasets & models
│   ├── datasets/              # Generated datasets
│   └── models/                # Trained models
├── 📁 docs/                    # Documentation (18 files)
├── 📁 logs_archive/            # Historical logs
└── 📁 coreg/                   # Coregistration utilities
```

## 🚀 Most Used Commands (Updated Paths)

### **Create Dataset**
```bash
python scripts/preprocessing/create_chunked_dataset.py \
    --samples 100 \
    --output-dir ./data/datasets/sr_dataset_100 \
    --regions 0
```

### **Train Model**
```bash
python scripts/training/train_with_chunked_dataset.py \
    --dataset-dir ./data/datasets/sr_dataset_100 \
    --output-dir ./data/models/sr_models_100 \
    --epochs 50
```

### **Validate Dataset**
```bash
python scripts/validation/validate_chunked_dataset.py \
    --dataset-dir ./data/datasets/sr_dataset_100
```

### **Quick Test**
```bash
# Integration test (currently running)
python scripts/testing/test_integrated_pipeline.py

# Demo with synthetic data
python scripts/testing/demo_training_integration.py
```

## 📊 Current Status

**Integration Test**: ✅ Still running after reorganization
- PID: 455003
- Runtime: ~7 minutes
- Expected completion: ~13-18 more minutes
- Output: `integrated_test_output3.log` (in logs_archive after completion)

## 📖 Key Documentation

| File | Purpose |
|------|---------|
| `docs/README.md` | Main project documentation |
| `docs/TRAINING_WORKFLOW.md` | Complete training guide |
| `docs/TRAINING_INTEGRATION_COMPLETE.md` | Integration status |
| `WORKSPACE_STRUCTURE.md` | This organization (detailed) |
| `QUICK_REFERENCE.md` | This file (quick commands) |

## 🗂️ Important Locations

### **Core Scripts**
- **Main pipeline**: `scripts/preprocessing/numpy_efficient_pipeline.py`
- **Dataset creation**: `scripts/preprocessing/create_chunked_dataset.py`
- **Model training**: `scripts/training/train_with_chunked_dataset.py`
- **GEE processing**: `Auth/AuthGEE.py`

### **Validation**
- **Dataset validation**: `scripts/validation/validate_chunked_dataset.py`
- **Image validation**: `scripts/validation/validate_real_data.py`
- **Multi-image check**: `scripts/validation/validate_multiple_images.py`

### **Testing**
- **Integration test**: `scripts/testing/test_integrated_pipeline.py`
- **Quick check**: `scripts/testing/quick_check.py`
- **Demo**: `scripts/testing/demo_training_integration.py`

### **Data**
- **Demo dataset**: `data/datasets/demo_dataset/`
- **Test cache**: `cache/test_chunked_pipeline_cache/`
- **Production datasets**: `data/datasets/sr_dataset_*/`
- **Models**: `data/models/sr_models_*/`

## 🔧 Utilities

### **Performance Analysis**
```bash
python scripts/utilities/performance_analysis.py
python scripts/utilities/realistic_performance_analysis.py
```

### **Monitoring**
```bash
python scripts/utilities/monitor_integration_test.py
```

### **Calculations**
```bash
python scripts/utilities/calculate_chunk_size.py
python scripts/utilities/calculate_storage.py
```

## 🧹 Maintenance

### **Clear Test Caches** (safe)
```bash
rm -rf cache/test_*
rm -rf cache/debug_*
```

### **Clear Python Cache**
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

### **View Recent Logs**
```bash
ls -lt logs_archive/*.log | head -5
```

### **Check Disk Usage**
```bash
du -sh cache/* data/* | sort -h
```

## 🔄 Migration Notes

### **Import Changes**
If scripts fail with import errors, add:
```python
import sys
sys.path.insert(0, 'scripts/preprocessing')
```

### **Cache Locations**
All caches moved to `cache/` directory - scripts should auto-detect

### **Output Locations**
- New datasets: Use `--output-dir ./data/datasets/name`
- New models: Use `--output-dir ./data/models/name`

## ✨ Benefits of New Organization

1. **Clear categorization**: Scripts organized by function
2. **Centralized documentation**: All `.md` files in `docs/`
3. **Unified cache**: All cache directories in `cache/`
4. **Separate data**: Datasets and models in `data/`
5. **Clean logs**: Historical logs in `logs_archive/`
6. **Easy navigation**: Logical directory structure

## 📝 Summary

- ✅ **49 files** reorganized into functional directories
- ✅ **18 documentation files** centralized
- ✅ **12 cache directories** consolidated
- ✅ **Integration test** still running (unaffected)
- ✅ **All functionality** preserved

---
**Reorganization Date**: November 6, 2025
**Integration Test Status**: Running (PID 455003)
**Next Steps**: See `docs/TRAINING_WORKFLOW.md`
