# AROSICS Integration Complete ✅

## What Was Added

### 1. Installed AROSICS Package
- **Version**: 1.13.2 via conda-forge
- **Dependencies**: GDAL, cartopy, and related geospatial libraries
- **Installation**: `conda install -n tf_gpu -c conda-forge arosics -y`

### 2. Added `coregister_lr_to_hr()` Function
**Location**: `scripts/preprocessing/numpy_efficient_pipeline.py` (lines ~40-120)

**Purpose**: Sub-pixel coregistration of LR (Sentinel-2) to HR (Planet) images

**Features**:
- Uses AROSICS COREG for automatic tie-point detection
- Calculates sub-pixel shifts (x, y) using red band
- Applies shifts to all bands
- Includes intelligent caching

### 3. Integrated into Processing Pipeline
- **Location**: After image download (line ~360 in `process_region_numpy()`)
- **Automatic**: Runs for every frame pair
- **Cached**: Results stored in `./sr_numpy_cache/coregistered/`

## How It Works

### Processing Flow
```
1. Download Planet HR (scale=3m) from GEE
2. Download Sentinel-2 LR (scale=10m) from GEE
3. ✨ NEW: Apply AROSICS coregistration
   a. Use red band for tie-point matching
   b. Calculate sub-pixel shifts (x, y)  
   c. Apply shifts to all bands
   d. Cache result
4. Continue with tiling and sample creation
```

### Coregistration Parameters
- **Reference**: Planet HR (3m resolution, 3806×5069 pixels)
- **Target**: Sentinel-2 LR (10m resolution, 1148×1527 pixels)
- **Method**: COREG (global coregistration)
- **Matching Band**: Red band (index 2)
- **Max Shift**: ±50 pixels
- **Output**: Sub-pixel aligned LR image

### Caching System
- **Directory**: `./sr_numpy_cache/coregistered/`
- **Naming**: `{region}_{s2_date}_{planet_date}.npy`
- **Example**: `STA-2_2020-01-15_2020-01-15.npy`

**Benefits**:
✓ Avoid reprocessing same image pairs
✓ Speed up regeneration if tiling changes  
✓ Can be cleared if parameters change

## Performance Impact

### Processing Time Per Frame

**Without AROSICS (original)**:
- Planet download: ~220s
- S2 download: ~19s
- **Total**: ~239s per frame

**With AROSICS (first time)**:
- Planet download: ~220s
- S2 download: ~19s
- AROSICS coregistration: ~30-60s
- **Total**: ~270-300s per frame **(+12-25%)**

**With AROSICS (cached)**:
- Planet download: ~220s
- S2 download: ~19s
- Load from cache: ~0.5s
- **Total**: ~239s per frame **(no overhead)**

### Full Dataset Impact

For 936 samples (39 sequences × 3 frames = 117 unique frames):
- **First generation**: +1-2 hours for coregistration
- **Subsequent**: No overhead (all cached)

## Testing the Integration

### Recommended Workflow

1. **Clear old test data** (but keep coregistered cache):
   ```bash
   rm -rf ./data/datasets/sr_dataset_tiling_test
   rm -rf ./test_tiling_cache
   # Keep: ./sr_numpy_cache/coregistered/
   ```

2. **Regenerate 10 samples with AROSICS**:
   ```bash
   conda run -n tf_gpu python scripts/preprocessing/create_chunked_dataset.py \
     --output-dir ./data/datasets/sr_dataset_tiling_test \
     --cache-dir ./test_tiling_cache \
     --samples 10 \
     --tile-size 256 \
     --sequence-length 3 \
     --regions 0
   ```

3. **Re-run quality verification**:
   ```bash
   conda run -n tf_gpu python verify_tile_quality.py
   ```

4. **Check visualizations**:
   - Open `tile_quality_check/sample_2_tile_2_quality.png`
   - Verify HR and LR show **SAME geographic features**
   - Compare to previous version (should see improved alignment)

### Expected Console Output

You should see new coregistration output during generation:

```
      [FRAME 1/3] 2020-01-15... 4-band
      📥 Downloading Planet with chunked parallel download...
      ✅ Planet downloaded: (3806, 5069, 4)
      ⏱️  Planet download took 220.3s
      📥 Downloading S2 with chunked parallel download...
      ✅ S2 downloaded: (1148, 1527, 4)
      ⏱️  S2 download took 19.2s
      🔧 AROSICS coregistration... shift=(1.23, -0.87)px (45.2s)
      ✓
```

## Troubleshooting

### AROSICS Failures

The code includes **automatic fallback**:
- If coregistration fails → uses original LR (no crash)
- Warning printed: `⚠️ AROSICS failed: ...`
- Processing continues with GEE's alignment

**Common failure reasons**:
- Insufficient tie-points (clouds, water)
- No image overlap
- Very different radiometry

**Solution**: Failed samples use GEE coregistration (still works, just less accurate)

### When to Clear Cache

Clear `./sr_numpy_cache/coregistered/` when:
- Changing `max_shift` parameter
- Changing matching band
- Updating AROSICS version
- Suspecting bad coregistration

Command: `rm -rf ./sr_numpy_cache/coregistered/`

**Note**: Forces reprocessing but ensures quality

## Next Steps

1. **Test with 10 samples** (as described above)
2. **Verify alignment** improved in visualizations
3. **If successful**, generate full dataset:
   - 936 samples for 2020 (STA-2)
   - Will take ~36-38 hours (includes coregistration)
   - Cache will be populated for future use
4. **Train model** on properly coregistered data:
   - Expected: Better PSNR and SSIM
   - Model learns correct spatial relationships

## Technical Details

### Why Sub-Pixel Coregistration Matters

For super-resolution training, **sub-pixel accuracy is critical**:
- Model learns pixel-to-pixel mappings
- Even 0.5 pixel misalignment → blurred/incorrect features
- AROSICS provides < 0.1 pixel accuracy

### GEE vs AROSICS Coregistration

**GEE's reproject()**:
- Aligns to pixel grid for each scale
- scale=3m and scale=10m use different grids
- Can have sub-pixel offsets

**AROSICS COREG**:
- Detects tie-points between actual images
- Calculates precise sub-pixel shifts
- Warps target to match reference exactly

## Files Modified

- `scripts/preprocessing/numpy_efficient_pipeline.py`
  - Added `coregister_lr_to_hr()` function (lines ~40-120)
  - Integrated coregistration (line ~360)
  - Added caching logic

## Dependencies Added

Via conda:
- arosics 1.13.2
- gdal 3.8.3
- cartopy 0.25.0
- geopandas 1.1.1
- And various supporting libraries

---

**Integration Status**: ✅ Complete and ready for testing
**Next Action**: Run 10-sample test to verify alignment improvement
