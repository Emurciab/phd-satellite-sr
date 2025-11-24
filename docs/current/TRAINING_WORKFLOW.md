# Complete Training Integration Workflow

## Overview
This document describes the end-to-end workflow from chunked dataset creation to SR model training.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline Flow                        │
└─────────────────────────────────────────────────────────────┘

1. GEE Image Processing (Auth/AuthGEE.py)
   ├── Planet: planet_best_per_day_adaptive()
   │   ├── UDM confidence filtering (60%)
   │   ├── Quality mosaicing (qscore)
   │   └── 4-band or 8-band detection
   └── Sentinel-2: S2 quality filtering
       ├── Cloud percentage < 60%
       └── AOI cloud fraction < 0.1

2. Parallel Chunked Download (numpy_efficient_pipeline.py)
   ├── Split AOI into 9×9 grid (81 chunks)
   ├── Download with 6 parallel workers
   ├── Unmask(0) for masked pixels
   ├── Reproject to native scale (3m/10m)
   └── Stitch chunks in memory

3. Dataset Creation (create_chunked_dataset.py)
   ├── Process regions with TensorFlowFreeDataPipeline
   ├── Generate TrainingSample objects
   │   ├── lr_sequence: (T, H, W, C) - Sentinel-2
   │   └── hr_sequence: (T, H, W, C) - Planet
   ├── Save to pickle format
   │   ├── training_samples.pkl
   │   └── metadata.pkl
   └── Validate during generation

4. Training (train_with_chunked_dataset.py)
   ├── Load ChunkedSRDataset (PyTorch Dataset)
   ├── Normalize to [0, 1] range
   ├── Train ImprovedSatelliteSR model
   │   ├── Temporal attention (if sequence_length > 1)
   │   ├── Residual blocks
   │   └── PixelShuffle upsampling
   ├── SpectralFidelityLoss
   │   ├── Pixel-level (L1 + MSE)
   │   ├── Spectral angle (band correlations)
   │   └── Gradient (edge preservation)
   └── Save checkpoints & best model
```

## Quick Start (Test with 2 Samples)

### Step 1: Wait for Integration Test
```bash
# The integration test is currently running
# It will validate the pipeline with 2 samples
# Expected completion: ~17-20 more minutes

# Monitor status:
ps aux | grep test_integrated_pipeline
```

### Step 2: Once Integration Test Completes
```bash
# The test will create cache in test_cache/
# We can use this to quickly test training

# Create quick test dataset (uses existing cache)
python create_chunked_dataset.py --quick-test

# Expected output:
# - ./quick_test_dataset/training_samples.pkl
# - ./quick_test_dataset/metadata.pkl
# - Time: ~1-2 minutes (reads from cache)
```

### Step 3: Validate Dataset Quality
```bash
python validate_chunked_dataset.py --dataset-dir ./quick_test_dataset

# Should show:
# ✅ ALL TESTS PASSED
# ✅ No zeros/fallbacks
# ✅ Realistic reflectance values
# ✅ Spatial variation
```

### Step 4: Test Training (2 Epochs)
```bash
python train_with_chunked_dataset.py --quick-test

# Expected:
# - Loads 2 samples
# - Trains for 2 epochs
# - Saves model to ./quick_test_models/
# - Time: ~2-5 minutes (depending on GPU)
```

## Production Workflow

### Phase 1: Small Dataset (100 Samples)
**Purpose**: Validate full pipeline and training before large-scale run

```bash
# Generate 100 samples from first region
python create_chunked_dataset.py \
    --output-dir ./sr_dataset_100 \
    --samples 100 \
    --tile-size 256 \
    --sequence-length 3 \
    --regions 0

# Expected time: ~18 hours
# (100 samples × ~10.5 min/sample with caching)
```

**Validate**:
```bash
python validate_chunked_dataset.py --dataset-dir ./sr_dataset_100

# Should pass all 5 tests
```

**Train**:
```bash
python train_with_chunked_dataset.py \
    --dataset-dir ./sr_dataset_100 \
    --output-dir ./sr_models_100 \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-4

# Expected time: ~2-4 hours (depending on GPU)
# Monitor GPU: watch -n 1 nvidia-smi
```

### Phase 2: Medium Dataset (500 Samples)
**Purpose**: Train better model with more data

```bash
# Generate 500 samples across multiple regions
python create_chunked_dataset.py \
    --output-dir ./sr_dataset_500 \
    --samples 250 \
    --tile-size 256 \
    --sequence-length 3 \
    --regions 0 1

# Expected time: ~3.5 days
# (500 samples × ~10.5 min/sample)
```

### Phase 3: Full Dataset (2000+ Samples)
**Purpose**: Production model training

```bash
# Generate full dataset from all regions
python create_chunked_dataset.py \
    --output-dir ./sr_dataset_full \
    --samples 500 \
    --tile-size 256 \
    --sequence-length 3

# Expected time: ~2 weeks
# Consider running overnight/over weekends
```

## Dataset Format

### TrainingSample Object
```python
class TrainingSample:
    lr_sequence: np.ndarray  # Shape: (T, H, W, C)
    hr_sequence: np.ndarray  # Shape: (T, H, W, C)
    metadata: dict
```

### Metadata Structure
```python
{
    'regions': [
        {
            'name': 'STA-2',
            'sample_count': 50,
            'start_date': '2020-01-01',
            'end_date': '2021-01-01'
        }
    ],
    'total_samples': 50,
    'tile_size': 256,
    'sequence_length': 3,
    'lr_shape': (3, 256, 256, 4),  # (T, H, W, C)
    'hr_shape': (3, 768, 768, 4),   # (T, H, W, C) - 3x upscale
    'creation_date': '2024-01-15T10:30:00'
}
```

## Model Architecture

### ImprovedSatelliteSR
- **Input**: LR sequence (B, T, C, H, W)
- **Output**: SR image (B, C, H×3, W×3)
- **Components**:
  - Feature extraction per frame
  - 8 residual blocks with BatchNorm
  - Temporal attention (if T > 1)
  - PixelShuffle upsampling (3x)
- **Parameters**: ~2.5M (base_channels=64)

### Loss Function: SpectralFidelityLoss
- **Pixel loss** (α=1.0): L1 + 0.1×MSE
- **Spectral loss** (α=0.1): 1 - cosine_similarity
- **Perceptual loss** (α=0.01): Gradient preservation

## Training Configuration

### Recommended Settings

**Quick Test (2 samples)**:
```bash
--batch-size 2
--epochs 2
--lr 1e-4
```

**Small Dataset (100 samples)**:
```bash
--batch-size 4
--epochs 50
--lr 1e-4
```

**Medium Dataset (500 samples)**:
```bash
--batch-size 8
--epochs 100
--lr 1e-4
```

**Large Dataset (2000+ samples)**:
```bash
--batch-size 16
--epochs 200
--lr 1e-4
```

### GPU Memory Requirements

| Batch Size | Tile Size | Model Channels | Memory Usage |
|------------|-----------|----------------|--------------|
| 2          | 256       | 64             | ~4 GB        |
| 4          | 256       | 64             | ~6 GB        |
| 8          | 256       | 64             | ~10 GB       |
| 16         | 256       | 64             | ~18 GB       |
| 4          | 512       | 64             | ~14 GB       |
| 8          | 512       | 64             | ~28 GB       |

## Monitoring & Debugging

### Check Dataset Creation Progress
```bash
# Monitor process
ps aux | grep create_chunked_dataset

# Check cache directory
ls -lh ./sr_dataset_100/cache/

# Count cached samples
find ./sr_dataset_100/cache/ -name "*.pkl" | wc -l
```

### Monitor Training
```bash
# GPU utilization
watch -n 1 nvidia-smi

# Training logs (if redirected)
tail -f training.log

# Check saved models
ls -lh ./sr_models_100/
```

### Validate Model Output
```python
import torch
from train_with_chunked_dataset import ImprovedSatelliteSR

# Load model
checkpoint = torch.load('./sr_models_100/best_model.pth')
model = ImprovedSatelliteSR(in_channels=4, out_channels=4)
model.load_state_dict(checkpoint['model_state_dict'])

# Test inference
import numpy as np
lr_test = np.random.rand(1, 3, 4, 256, 256).astype(np.float32)
lr_tensor = torch.from_numpy(lr_test)

model.eval()
with torch.no_grad():
    sr = model(lr_tensor)
    
print(f"Input shape: {lr_tensor.shape}")
print(f"Output shape: {sr.shape}")
# Should be: (1, 4, 768, 768)
```

## Troubleshooting

### Issue: Out of Memory During Training
**Solutions**:
- Reduce `--batch-size` (try 2 or 1)
- Reduce `--tile-size` in dataset creation
- Reduce `base_channels` in model (64 → 32)
- Use gradient accumulation

### Issue: Dataset Creation Too Slow
**Solutions**:
- Increase parallel workers (currently 6)
- Use cached regions (process in batches)
- Reduce sequence_length (3 → 1)
- Process regions separately

### Issue: Model Not Converging
**Solutions**:
- Check data normalization (validate_chunked_dataset.py)
- Reduce learning rate (1e-4 → 1e-5)
- Increase epochs
- Check for NaN in gradients
- Verify loss components are balanced

### Issue: High Zero Percentage in Data
**Solutions**:
- Adjust UDM confidence threshold in AuthGEE.py
- Check cloud filtering thresholds
- Validate preprocessing in pipeline
- Run validate_real_data.py

## Performance Optimization

### Dataset Generation
- **Caching**: First run is slow, subsequent runs use cache
- **Parallelization**: 6 workers optimal for most systems
- **Chunking**: 9×9 grid balances speed vs. GEE limits
- **Region batching**: Process one region at a time

### Training
- **Mixed Precision**: Enable AMP for faster training
- **DataLoader workers**: Use 2-4 workers
- **Pin memory**: Enable for GPU training
- **Gradient checkpointing**: For larger models

## Next Steps After Integration Test

1. ✅ **Integration test completes** (~17-20 more minutes)
2. **Verify test results**:
   ```bash
   python -c "import os; print('✅ Test complete!' if os.path.exists('test_cache/lr_20210429.pkl') else '⏳ Still running...')"
   ```
3. **Run quick training test**:
   ```bash
   python create_chunked_dataset.py --quick-test
   python validate_chunked_dataset.py --dataset-dir ./quick_test_dataset
   python train_with_chunked_dataset.py --quick-test
   ```
4. **Generate small production dataset**:
   ```bash
   python create_chunked_dataset.py --samples 100 --regions 0
   ```
5. **Train production model**:
   ```bash
   python train_with_chunked_dataset.py --epochs 50
   ```

## File Structure
```
SR/
├── Auth/
│   └── AuthGEE.py                    # GEE preprocessing
├── numpy_efficient_pipeline.py       # Chunked download pipeline
├── create_chunked_dataset.py         # Dataset generation ⭐
├── train_with_chunked_dataset.py    # Model training ⭐
├── validate_chunked_dataset.py       # Quality validation ⭐
├── test_integrated_pipeline.py       # Integration test (RUNNING)
├── monitor_integration_test.py       # Monitor script
└── [output directories]
    ├── test_cache/                   # Integration test cache
    ├── quick_test_dataset/           # Quick test (2 samples)
    ├── sr_dataset_100/               # Small dataset (100 samples)
    ├── sr_dataset_500/               # Medium dataset (500 samples)
    ├── sr_dataset_full/              # Full dataset (2000+ samples)
    └── sr_models_[size]/             # Trained models
        ├── best_model.pth
        ├── final_model.pth
        └── checkpoint_epoch_*.pth
```

## Timeline Summary

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 0 | Integration test | ~25 min | ⏳ RUNNING |
| 1 | Quick test (2 samples) | ~5 min | 🔜 NEXT |
| 2 | Small dataset (100 samples) | ~18 hours | 📅 PLANNED |
| 3 | Train on 100 samples | ~2-4 hours | 📅 PLANNED |
| 4 | Medium dataset (500 samples) | ~3.5 days | 📅 PLANNED |
| 5 | Full dataset (2000+ samples) | ~2 weeks | 📅 PLANNED |

---
**Created**: Based on validated chunked download pipeline
**Status**: Ready for production use after integration test completes
