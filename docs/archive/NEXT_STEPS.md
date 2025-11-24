# Next Steps - Super Resolution Dataset Pipeline

## Current Status (Nov 6, 2025)

### ✅ Completed
1. **Parallel Chunked Download Implementation**
   - Created `chunked_download_fast.py` with 9×9 grid (81 chunks)
   - 6 parallel workers for optimal speed
   - Works within GEE API limits (262k pixels per chunk)
   - Preserves full native resolution (3m Planet, 10m S2)

2. **Data Validation** 
   - Multi-image test: 3/3 images passed ✅
   - 99.8% valid pixels, realistic reflectance values
   - No fallback patterns or test data
   - Strong spatial variation confirmed

3. **Pipeline Integration**
   - Integrated into `numpy_efficient_pipeline.py`
   - Replaced `getDownloadURL()` with chunked approach
   - Removed requests/io dependencies

### ⏳ In Progress
- **Integration Test**: 2-sample end-to-end test running (~20-25 min total)
- **Monitor**: Automated monitor checking every 60s, will report when complete

---

## Next Steps After Test Completes

### Step 1: Verify Integration Test Results (5 min)

**If test succeeds:**
```bash
cd /media/eamurciab/TOSHIBA_EXT1/PhD/SR

# Check cached samples
ls -lh test_chunked_pipeline_cache/

# Should see 2+ .pkl files with real data
# Each file contains hr_sequence and lr_sequence numpy arrays
```

**Validation checklist:**
- ✅ At least 2 samples generated
- ✅ HR shape: (256, 256, 1, 4) or (256, 256, 1, 8) 
- ✅ LR shape: (64, 64, 1, 4)
- ✅ HR mean values: 100-3000 (visible), 200-5000 (NIR)
- ✅ No zeros or fallback patterns

**If test fails:**
- Check error in `monitor_output.log`
- Debug specific issue
- Re-run with more logging

---

### Step 2: Create Production Dataset Script (10 min)

Create `create_production_dataset.py`:

```python
"""
Production dataset generation with chunked downloads
Estimated time: ~50 hours for full dataset
"""
from numpy_efficient_pipeline import TensorFlowFreeDataPipeline
from Auth.AuthGEE import regions

# Initialize pipeline
pipeline = TensorFlowFreeDataPipeline(
    cache_dir="production_sr_dataset",
    tile_size=256,
    overlap=64
)

# Process all regions
for region in regions:
    print(f"\n{'='*70}")
    print(f"Processing region: {region['name']}")
    print(f"{'='*70}\n")
    
    # Generator yields samples as they're created
    sample_count = 0
    for sample in pipeline.process_region_numpy(
        region,
        sequence_length=3,  # 3-frame temporal sequences
        use_cache=True
    ):
        sample_count += 1
        if sample_count % 10 == 0:
            print(f"Generated {sample_count} samples...")
    
    print(f"\n✅ Region {region['name']} complete: {sample_count} samples")

print("\n🎉 Production dataset generation complete!")
```

**Key parameters:**
- `sequence_length=3`: 3-frame temporal sequences for attention mechanism
- `tile_size=256`: 256×256 pixel tiles (configurable)
- `overlap=64`: Tile overlap to avoid edge artifacts

---

### Step 3: Run Production Dataset Generation

**Option A: Full production run (~50 hours)**
```bash
cd /media/eamurciab/TOSHIBA_EXT1/PhD/SR

# Run with nohup so it continues if terminal closes
nohup conda run -n tf_gpu python create_production_dataset.py > production_dataset.log 2>&1 &

# Monitor progress
tail -f production_dataset.log

# Or use the provided monitor script
python3 monitor_production.py
```

**Option B: Staged approach (recommended for first time)**

Run one region at a time to verify before full batch:

```python
# Start with first region only
pipeline = TensorFlowFreeDataPipeline(cache_dir="production_sr_dataset")
samples = list(pipeline.process_region_numpy(regions[0], sequence_length=3))
print(f"Region 1 complete: {len(samples)} samples")

# Verify quality, then continue with remaining regions
```

**Performance estimates:**
- **Per image**: ~10.5 minutes (Planet + S2)
- **Per temporal sequence**: ~31.5 minutes (3 frames)
- **Per region**: ~8-12 hours (depends on tile count)
- **Full dataset**: ~40-60 hours

---

### Step 4: Dataset Quality Validation (30 min)

After production run completes, validate the dataset:

```bash
# Run comprehensive validation
python validate_production_dataset.py

# Check statistics
python analyze_dataset_statistics.py
```

**Create `validate_production_dataset.py`:**
```python
"""Validate production dataset quality"""
import os
import pickle
import numpy as np
from pathlib import Path

cache_dir = "production_sr_dataset"
cache_files = list(Path(cache_dir).glob("*.pkl"))

print(f"Found {len(cache_files)} cached samples\n")

# Sample validation
valid_count = 0
invalid_count = 0
stats = {'hr_means': [], 'lr_means': [], 'hr_shapes': [], 'lr_shapes': []}

for i, cache_file in enumerate(cache_files[:100]):  # Check first 100
    with open(cache_file, 'rb') as f:
        sample = pickle.load(f)
    
    hr = sample.hr_sequence
    lr = sample.lr_sequence
    
    # Basic checks
    is_valid = True
    
    # Check shapes
    if hr.shape[-1] not in [4, 8] or lr.shape[-1] != 4:
        print(f"❌ Invalid shapes: HR={hr.shape}, LR={lr.shape}")
        is_valid = False
    
    # Check for zeros
    if hr.sum() == 0 or lr.sum() == 0:
        print(f"❌ All zeros in {cache_file.name}")
        is_valid = False
    
    # Check realistic values
    hr_mean = hr.mean()
    if hr_mean < 50 or hr_mean > 5000:
        print(f"❌ Unrealistic HR mean: {hr_mean}")
        is_valid = False
    
    if is_valid:
        valid_count += 1
        stats['hr_means'].append(hr_mean)
        stats['lr_means'].append(lr.mean())
        stats['hr_shapes'].append(hr.shape)
        stats['lr_shapes'].append(lr.shape)
    else:
        invalid_count += 1

print(f"\n{'='*70}")
print("VALIDATION SUMMARY")
print(f"{'='*70}")
print(f"Valid samples: {valid_count}")
print(f"Invalid samples: {invalid_count}")
print(f"Success rate: {100*valid_count/(valid_count+invalid_count):.1f}%")

if valid_count > 0:
    print(f"\nHR mean values: {np.mean(stats['hr_means']):.1f} ± {np.std(stats['hr_means']):.1f}")
    print(f"LR mean values: {np.mean(stats['lr_means']):.1f} ± {np.std(stats['lr_means']):.1f}")
    print(f"Most common HR shape: {max(set(map(str, stats['hr_shapes'])), key=stats['hr_shapes'].count)}")
    print(f"Most common LR shape: {max(set(map(str, stats['lr_shapes'])), key=stats['lr_shapes'].count)}")
```

---

### Step 5: Setup Training Pipeline (1-2 hours)

Once dataset is validated:

1. **Create data loader** for your SR model
2. **Setup training configuration**
3. **Initialize model architecture** (EDSR, RCAN, or custom)
4. **Start training** with validated dataset

**Example data loader:**
```python
"""SR data loader using cached samples"""
from numpy_efficient_pipeline import SuperResolutionDataGenerator
from Auth.AuthGEE import regions

# Create data generator
train_generator = SuperResolutionDataGenerator(
    cache_dir="production_sr_dataset",
    batch_size=8,
    augment=True,
    normalize=True
)

# Use in training loop
for batch_hr, batch_lr in train_generator.generate_batches(
    regions, 
    epochs=100, 
    shuffle=True
):
    # Train your SR model
    loss = model.train_step(batch_lr, batch_hr)
```

---

## Performance Optimization Tips

### If download is too slow:
1. **Increase parallel workers**: Try `max_workers=8` or `10`
2. **Process regions in parallel**: Run multiple instances
3. **Use smaller tile size**: 128×128 instead of 256×256
4. **Reduce sequence length**: Use 1 or 2 frames instead of 3

### If running out of memory:
1. **Reduce tile size**: 128×128 uses 1/4 memory of 256×256
2. **Process fewer samples at once**: Use generator, don't load all
3. **Clear cache periodically**: Delete old samples after validation

### If hitting GEE quotas:
1. **Add delays between requests**: `time.sleep(1)` between chunks
2. **Use service account**: Ensures separate quota
3. **Process during off-peak hours**: Nights/weekends

---

## Monitoring Production Run

**Create `monitor_production.py`:**
```python
"""Monitor production dataset generation"""
import os
import time
from pathlib import Path

cache_dir = "production_sr_dataset"
log_file = "production_dataset.log"

last_count = 0
start_time = time.time()

while True:
    if os.path.exists(cache_dir):
        current_count = len(list(Path(cache_dir).glob("*.pkl")))
        
        if current_count > last_count:
            elapsed = time.time() - start_time
            rate = current_count / (elapsed / 3600)  # samples per hour
            
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"Samples: {current_count} | "
                  f"Rate: {rate:.1f}/hr | "
                  f"Elapsed: {elapsed/3600:.1f}h")
            
            last_count = current_count
    
    time.sleep(300)  # Check every 5 minutes
```

---

## Troubleshooting

### Issue: Downloads timing out
**Solution**: Increase timeout, check internet connection, verify GEE credentials

### Issue: Cache growing too large
**Solution**: Validate and move batches to external storage periodically

### Issue: Inconsistent data quality
**Solution**: Run validation script, check for specific date ranges with issues

### Issue: Process crashes midway
**Solution**: Pipeline caches progress - restart will resume from last sample

---

## Success Criteria

✅ **Integration test passes** (2 samples with real data)
✅ **Production run completes** (~50 hours)
✅ **Dataset validation** (>95% valid samples)
✅ **Consistent data quality** (realistic values, proper shapes)
✅ **Ready for SR training** (data loader works, batches generate)

---

## Expected Timeline

- **Today**: Integration test completes (~20 min)
- **Day 1-2**: Production dataset generation (~50 hours)
- **Day 3**: Dataset validation and quality checks
- **Day 4+**: Begin SR model training

**Total time to training-ready dataset: ~3-4 days** ⏱️
