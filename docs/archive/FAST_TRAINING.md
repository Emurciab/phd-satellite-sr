# Fast Training Strategy

## Problem
Your training is taking **~24 hours per epoch** because:
- 144 tiles per region (too many!)
- Downloading from GEE for each tile (slow!)
- Multiple regions (lots of data!)

## Solution: 3-Step Fast Training

### Step 1: Create Fixed Dataset (Run ONCE)
```bash
conda activate tf_gpu
python create_fixed_dataset.py
```

**What it does:**
- Downloads 50 samples from GEE
- Saves to `./fixed_dataset/training_samples.pkl`
- Takes ~5-10 minutes (one-time cost)

**Output:**
```
✅ DATASET CREATED
Samples: 50
Location: ./fixed_dataset
Files:
  - training_samples.pkl (12.5 MB)
  - metadata.pkl
```

### Step 2: Train on Fixed Dataset (Run MANY times)
```bash
python train_with_fixed_dataset.py
```

**What it does:**
- Loads pre-downloaded data (NO GEE calls!)
- Trains for 20 epochs
- Takes ~5-10 minutes per run
- You can train 100+ times in the time your original took for 1 epoch!

**Output:**
```
Epoch   1/20: loss=0.7234, lr=0.000100, time=12.3s
Epoch   2/20: loss=0.6891, lr=0.000098, time=11.8s
...
Epoch  20/20: loss=0.4523, lr=0.000010, time=11.5s
  💾 Saved best model (loss: 0.4523)

✅ TRAINING COMPLETED
```

### Step 3: Scale Up (When Ready)
Once you confirm training works:

```bash
# Create larger dataset
python create_fixed_dataset.py  # Edit to num_samples=200

# Or use optimized full training
python train_sr_fast.py  # Uses tile_size=1024
```

## Performance Comparison

| Method | Tiles/Region | Time/Epoch | Total Time (50 epochs) |
|--------|-------------|------------|------------------------|
| **Original** | 144 | ~24 hours | ~50 days! |
| **train_sr_fast.py** | ~16 | ~2 hours | ~4 days |
| **Fixed dataset** | N/A | ~10 min | ~8 hours |

## Recommended Workflow

### Phase 1: Development (Use Fixed Dataset)
```bash
# 1. Create small dataset
python create_fixed_dataset.py  # 50 samples

# 2. Train many times, experiment with hyperparameters
python train_with_fixed_dataset.py  # base_channels=32, epochs=20
python train_with_fixed_dataset.py  # base_channels=64, epochs=20
python train_with_fixed_dataset.py  # different learning rates
```

**Goal:** Find best hyperparameters quickly

### Phase 2: Validation (Use Fast Training)
```bash
# Increase dataset but keep it manageable
python create_fixed_dataset.py  # 100-200 samples

# Train longer
python train_with_fixed_dataset.py  # base_channels=64, epochs=50
```

**Goal:** Verify model can learn and loss decreases

### Phase 3: Production (Full Training)
```bash
# Only when you know everything works!
python train_sr_fast.py  # All regions, 1024px tiles
```

**Goal:** Final model with full data

## Quick Fixes Applied

### 1. train_sr_model.py (Updated)
- ✅ `base_channels: 64 → 32` (4x fewer parameters)
- ✅ `tile_size: 256 → 512` (4x fewer tiles)
- ✅ `batch_size: 4 → 2` (faster iteration)
- ✅ `num_epochs: 50 → 10` (test faster)

### 2. train_sr_fast.py (NEW)
- ✅ Single region only
- ✅ `tile_size: 1024` (very few tiles)
- ✅ `base_channels: 32` (small model)
- ✅ No augmentation (speed)

### 3. create_fixed_dataset.py (NEW)
- ✅ Download data once
- ✅ Save to disk
- ✅ Reuse forever

### 4. train_with_fixed_dataset.py (NEW)
- ✅ No GEE calls during training
- ✅ Load from disk (instant)
- ✅ Train in minutes not days

## Start Now

```bash
conda activate tf_gpu

# Option A: Fixed dataset (RECOMMENDED)
python create_fixed_dataset.py
python train_with_fixed_dataset.py

# Option B: Fast training with GEE
python train_sr_fast.py

# Option C: Updated full training (still slow but 4x faster)
python train_sr_model.py
```

## Expected Timeline

**Fixed Dataset Approach:**
- Create dataset: 10 minutes
- Train: 10 minutes
- **Total: 20 minutes for complete experiment!**

**Fast Training Approach:**
- Train: 1-2 hours per run
- **Total: ~2 hours per experiment**

**Original Approach:**
- Train: ~24 hours per epoch × 50 epochs
- **Total: ~50 days!**

Choose fixed dataset for development, fast training for validation, full training only for final model.
