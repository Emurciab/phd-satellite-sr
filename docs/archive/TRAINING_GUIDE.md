# Training Guide for Satellite Super Resolution

## ✅ What You Have Now

### 1. **ImprovedModel.py** (Production-Ready Model)
- ✅ All architectural issues fixed
- ✅ Handles 4-band and 6-band inputs adaptively
- ✅ Temporal modeling for sequences
- ✅ Spectral fidelity guarantees
- ✅ Physics-based output constraints
- ✅ 6.25M parameters, memory efficient
- ✅ Tested and working with real GEE data

### 2. **numpy_efficient_pipeline.py** (Data Pipeline)
- ✅ `TensorFlowFreeDataPipeline` - Downloads and caches GEE data
- ✅ `NumpyTrainingDataGenerator` - **NOW FIXED** with `pytorch_format=True`
- ✅ Outputs correct format: `(B, T, C, H, W)` for PyTorch
- ✅ Smart caching with SQLite
- ✅ Temporal-consistent augmentations
- ✅ Automatic normalization

### 3. **train_sr_model.py** (NEW - Complete Training Script)
- ✅ Full training loop with `NumpyTrainingDataGenerator`
- ✅ TensorBoard logging
- ✅ Checkpoint saving (best + periodic)
- ✅ Learning rate scheduling
- ✅ Gradient clipping
- ✅ Loss component monitoring

## 🚀 How to Train

### Step 1: Activate Environment
```bash
conda activate tf_gpu
```

### Step 2: Start Training
```bash
cd /media/eamurciab/TOSHIBA_EXT1/PhD/SR
python train_sr_model.py
```

### Step 3: Monitor Progress
In another terminal:
```bash
conda activate tf_gpu
tensorboard --logdir=./logs
```
Then open: http://localhost:6006

## 📊 Expected Output

```
================================================================================
🚀 SATELLITE SUPER RESOLUTION TRAINING
================================================================================

📊 Configuration:
   Device: cuda
   Base channels: 64
   Scale factor: 4
   Temporal: True
   Sequence length: 3
   Batch size: 4
   Learning rate: 0.0001

🔄 Initializing data pipeline...
🏗️ Creating model...
   Total parameters: 6,248,565
   Trainable parameters: 6,248,565
   Model size: 23.84 MB (FP32)

✅ Setup complete. Starting training...
================================================================================

================================================================================
Epoch 1/50
================================================================================
🔄 Processing STA-1 (NumPy pipeline)...
   📅 Created 10 temporal sequences
   🗺️ Created 4 spatial tiles
   ✅ Processed 5 samples...
   ✅ Processed 10 samples...
   🎉 Completed STA-1: 12 samples in 45.3s

   Batch 1: loss=0.7854, time=2.34s
   Batch 2: loss=0.7201, time=2.12s
   ...

📊 Epoch 1 Summary:
   Total loss: 0.7105
   - Pixel: 0.3845
   - SAM: 0.2891
   - Correlation: 0.0098
   - LR consistency: 0.3312
   - Edge: 0.2501
   Time: 67.2s (12 batches)
   Learning rate: 0.000100
   💾 Saved best model (loss: 0.7105)
```

## 🔧 Customization

### Adjust Training Parameters
Edit `train_sr_model.py` at the bottom:

```python
trained_model = train_sr_model(
    # Model
    base_channels=64,        # Try 32, 64, 96, 128
    scale_factor=4,          # S2→Planet is 4x
    use_temporal=True,       # Use sequences
    
    # Data
    sequence_length=3,       # Try 3, 5, 7
    tile_size=256,          # Try 128, 256, 512
    batch_size=4,           # Depends on GPU memory
    
    # Training
    num_epochs=50,          # Try 50, 100, 200
    learning_rate=1e-4,     # Try 1e-4, 5e-5, 1e-5
)
```

### Adjust Loss Weights
In the `train_sr_model()` function:

```python
loss_weights = {
    'pixel': 1.0,           # L1 pixel loss
    'sam': 0.5,             # Spectral angle mapper
    'correlation': 0.1,     # Spectral correlation
    'lr_consistency': 0.3,  # Downsampled consistency
    'edge': 0.2             # Edge preservation
}
```

## 📁 Output Structure

After training:
```
SR/
├── checkpoints/
│   ├── best_model.pth              # Best model (lowest loss)
│   ├── checkpoint_epoch_10.pth     # Periodic checkpoints
│   ├── checkpoint_epoch_20.pth
│   └── ...
├── logs/
│   └── 20251030_143022/            # TensorBoard logs
│       ├── events.out.tfevents...
│       └── ...
└── sr_training_cache/              # Cached GEE data
    ├── numpy_cache.db
    └── ...
```

## 🎯 Why NumpyTrainingDataGenerator IS Necessary

**Before the fix:**
- Generator output: `(B, T, H, W, C)` ❌
- PyTorch expects: `(B, T, C, H, W)` 
- Manual transpose needed in training loop

**After the fix:**
- Generator output with `pytorch_format=True`: `(B, T, C, H, W)` ✅
- PyTorch expects: `(B, T, C, H, W)` ✅
- Direct usage, no manual transpose!

**Benefits:**
1. ✅ **Batching** - Efficient batch generation
2. ✅ **Caching** - Reuses downloaded GEE data
3. ✅ **Augmentation** - Temporal-consistent flips/rotations
4. ✅ **Normalization** - Automatic [0,1] scaling
5. ✅ **Memory efficient** - Streams data, doesn't load all at once
6. ✅ **Multi-region** - Processes all regions automatically

## 🔍 Monitoring Training

### TensorBoard Metrics
- `Loss/total` - Overall loss
- `Loss/pixel` - L1 reconstruction loss
- `Loss/sam` - Spectral angle mapper
- `Loss/correlation` - Spectral correlation
- `Loss/lr_consistency` - LR downsampling consistency
- `Loss/edge` - Edge preservation
- `Learning_rate` - Current LR

### Console Output
- Real-time batch loss
- Epoch summaries with all loss components
- Cache hit/miss statistics
- Processing time per region
- Automatic checkpoint saving

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch_size
batch_size=2  # or even 1

# Or reduce tile_size
tile_size=128  # smaller tiles
```

### Slow Training
```python
# Increase batch_size if GPU has memory
batch_size=8

# Reduce augmentation
augment=False

# Use fewer regions initially
# Edit regions list in Auth/AuthGEE.py
```

### Cache Growing Too Large
```python
# Reduce max cache
max_cache_gb=2.0

# Or clear cache
rm -rf ./sr_training_cache
```

## 📈 Next Steps After Training

1. **Evaluate** - Test on held-out regions
2. **Visualize** - Generate SR images and compare
3. **Metrics** - Calculate PSNR, SSIM, SAM
4. **Ablation** - Test without temporal/spectral components
5. **Fine-tune** - Adjust loss weights based on results

## 🎓 Key Takeaways

✅ **NumpyTrainingDataGenerator is ESSENTIAL** - It provides:
- Proper data format for PyTorch
- Efficient batch generation
- Caching and augmentation
- Multi-region processing

✅ **train_sr_model.py is PRODUCTION-READY** - It includes:
- Complete training loop
- Checkpoint management
- TensorBoard logging
- All best practices

✅ **Ready to train** - All components tested and working!
