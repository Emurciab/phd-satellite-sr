# Tiling Fix: Geographic Coverage Alignment

## Problem Identified

The LR and HR tiles were **not covering the same geographic area**, causing spatial misalignment in the training data.

### Root Cause

**Line 709 (OLD CODE):**
```python
lr_target = self.tile_size // 4  # Hardcoded 4× downsampling
```

This assumed a **4× resolution ratio**, but the actual ratio is:
- **Planet HR**: 3.0m pixel size  
- **Sentinel-2 LR**: 10.0m pixel size  
- **Actual ratio**: 10.0 / 3.0 = **3.33×** (not 4×!)

### Impact

With `tile_size=256`:
- **OLD**: LR tiles = 256 // 4 = **64×64 pixels** (4.00× ratio) ❌
- **NEW**: LR tiles = 256 / 3.33 = **~76×76 pixels** (3.33× ratio) ✅

**Geographic mismatch**: The 64×64 LR tile covered a **smaller** geographic area than the corresponding 256×256 HR tile, causing the shift you observed in visual comparisons.

## Solution Implemented

**Lines 706-722 (NEW CODE):**
```python
# Calculate actual resolution ratio from image dimensions
actual_ratio_h = hr_h / lr_h
actual_ratio_w = hr_w / lr_w
actual_ratio = (actual_ratio_h + actual_ratio_w) / 2  # Average ratio

# Use actual ratio for LR tile sizing
lr_target = int(self.tile_size / actual_ratio)  # NOT // 4
```

### Key Improvements

1. **Adaptive**: Calculates ratio from **actual downloaded dimensions**, not hardcoded values
2. **Accurate**: Respects the true 3.33× ratio between Planet (3.0m) and Sentinel-2 (10.0m)
3. **Geographic correspondence**: HR and LR tiles now cover the **exact same geographic area**

## Verification

Run the diagnostic again to confirm fix:
```bash
conda run -n tf_gpu python diagnose_spatial_coverage.py
```

**Expected results AFTER fix:**
- Tile ratio: **3.33× to 3.37×** (close to 10m/3m ratio) ✅
- Edge differences: **<0.05** (minimal mismatch) ✅  
- Visual alignment: **Geographic features match** ✅

## Next Steps

1. ✅ **DONE**: Fixed tiling logic to use actual resolution ratio
2. **TODO**: Regenerate dataset with corrected tiling
3. **TODO**: Verify alignment in new visualizations
4. **TODO**: Train SR model on properly aligned data

## Technical Details

### Resolution Specifications
- **Planet (PlanetScope)**: 3.0m ground sample distance (GSD)
- **Sentinel-2 (MSI)**: 10.0m GSD for RGB+NIR bands
- **Expected pixel ratio**: 10.0 / 3.0 = 3.333...

### Download Process
Both images downloaded at same AOI bounds but different scales:
```python
# Planet download (scale=3 for 3.0m pixels)
planet_arr = download_image_chunked_parallel(..., scale=3)

# Sentinel-2 download (scale=10 for 10.0m pixels)  
lr_arr = download_image_chunked_parallel(..., scale=10)
```

### Tiling Process (FIXED)
```python
# Example with actual dimensions
hr_h, hr_w = 768, 768  # Planet at 3.0m
lr_h, lr_w = 230, 230  # Sentinel-2 at 10.0m
actual_ratio = 768 / 230 = 3.34×  # Matches expected 3.33×

# HR tiles: 256×256 pixels
hr_tile_size = 256

# LR tiles: sized to match geographic coverage
lr_tile_size = int(256 / 3.34) = 76×76 pixels

# Result: Both tiles cover the same ground area!
```

## Why This Matters for SR Training

Super-resolution requires **pixel-perfect alignment** between LR and HR:
- Each HR pixel should have a corresponding position in the upscaled LR
- If tiles cover different areas, the model learns **wrong spatial relationships**
- Geographic mismatch introduces **spatial artifacts** in predictions

With this fix, the SR model can now learn the correct mapping from LR to HR for the **same geographic locations**.

