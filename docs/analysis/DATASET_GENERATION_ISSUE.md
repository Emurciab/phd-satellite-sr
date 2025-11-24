# Dataset Generation Performance Issue - Analysis & Solution

## Problem Summary

The dataset generation was taking **too long** (hours instead of minutes) due to architectural inefficiencies in the processing pipeline.

## Root Cause Analysis

### Current Implementation Issues

**1. SERIAL Sequence Processing** ❌
```python
# Line 638 in numpy_efficient_pipeline.py
for seq_idx, sequence in enumerate(sequences):  # ← SERIAL, NOT PARALLEL!
    # Download 3 full images per sequence
    # Takes ~13 minutes per sequence
```

**2. Full Image Downloads Even For Small Tiles** ❌
- Downloads entire 3800×5000 px Planet images (~223s each)
- Downloads entire 1150×1530 px Sentinel-2 images (~23s each)  
- Then crops small 256×256 tiles from them
- **Wasteful**: Downloads ~95% more data than needed!

**3. No Parallelization at Sequence Level** ❌
- Parallel download WITHIN each image: ✓ (12 workers, 9×9 chunks)
- Parallel processing ACROSS sequences: ❌ (serial for loop)

### Time Breakdown Per Sequence

```
Download Planet frames:      3 × 223s = 669s
Download Sentinel-2 frames:  3 × 23s  = 69s
Coregistration:             3 × 11s  = 33s
Harmonization:              3 × 2.6s = 8s
                            _______________
TOTAL PER SEQUENCE:                  779s (13 minutes!)
```

**For 20 samples from ~3-4 sequences: 39-52 minutes MINIMUM**

### What Actually Happened

From the logs:
- Process started: Nov 16, 13:49
- Only 1 coregistered file created: `STA-2_2020-01-01_2020-01-06.npy` (295MB)
- No harmonized files created (directory doesn't exist)
- Process died/hung after processing just 1 frame

## Why The Current Approach Fails

### The Architecture
```
Current: Download Full Images → Crop Tiles
         ↓
         3800×5000 px download (18 million pixels)
         ↓
         Crop to 256×256 (65k pixels - only 0.3% used!)
```

### The Math
- Full Planet image: 3806 × 5069 × 4 bands = ~77 MB per frame
- Actual tile needed: 256 × 256 × 4 bands = ~0.26 MB per frame
- **Waste factor: 296×** (downloading 296 times more data than needed!)

## Solutions

### Option 1: Direct Tile Download ⚡ (FASTEST)
**Strategy**: Download ONLY the tile regions from GEE, skip full images entirely

```python
# Ultra-fast approach
tile_geom = ee.Geometry.Rectangle(tile_bounds)  # Small 256×256 region
hr_tile = planet_img.clip(tile_geom).sampleRectangle()  # Direct download
lr_tile = s2_img.clip(tile_geom).sampleRectangle()
```

**Benefits**:
- ✅ Downloads only needed pixels (~0.26 MB vs 77 MB)
- ✅ 296× less data transfer
- ✅ No tiling step needed
- ✅ Estimated time: 5-10 minutes for 20 samples

**Created**: `scripts/preprocessing/generate_sta2_ultrafast.py`

### Option 2: Parallel Sequence Processing 🚀 (BETTER SCALING)
**Strategy**: Process multiple sequences in parallel using ProcessPoolExecutor

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_sequence, seq) for seq in sequences]
    for future in as_completed(futures):
        samples.extend(future.result())
```

**Benefits**:
- ✅ 4× faster with 4 workers
- ✅ Better for large datasets (100+ samples)
- ✅ Scales with CPU cores

**Trade-off**: Still downloads full images, just in parallel

### Option 3: Hybrid Approach 🎯 (PRODUCTION-READY)
**Strategy**: Combine both optimizations

1. Download full images once per sequence (in parallel)
2. Extract multiple tiles from each cached full image
3. Reuse cached images across multiple tile requests

**Benefits**:
- ✅ Amortizes download cost across tiles
- ✅ Good cache reuse
- ✅ Best for generating 100s-1000s of samples

## Recommendation

### For Testing (20-50 samples)
**Use Option 1: Ultra-Fast Direct Tile Download**

```bash
python scripts/preprocessing/generate_sta2_ultrafast.py --max-samples 20
```

Expected time: **5-10 minutes**

### For Production (1000+ samples)
**Implement Option 3: Hybrid Parallel + Caching**

Expected time: **~30 hours for 1440 samples** (down from ~100+ hours with current approach)

## Key Takeaway

The bottleneck is NOT the parallel download implementation (that works fine with 12 workers).

The bottleneck is:
1. **Serial sequence processing** (for loop, not parallel)
2. **Downloading full images when only small tiles are needed**

Both are architectural issues in `numpy_efficient_pipeline.py` that require refactoring.

## Action Items

- [x] Created ultra-fast script for testing (`generate_sta2_ultrafast.py`)
- [ ] Test ultra-fast script with 5 samples first
- [ ] If successful, generate 20 samples for visualization
- [ ] For production: refactor main pipeline to add sequence-level parallelization
- [ ] Consider hybrid approach for full STA-2 dataset generation

## Files Created

1. `scripts/preprocessing/generate_sta2_ultrafast.py` - Direct tile download (fastest)
2. `scripts/preprocessing/generate_sta2_quick.py` - Quick generation with caching

Both avoid the serial bottleneck and unnecessary full-image downloads.
