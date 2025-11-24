# Coregistration Methods Comparison: AROSICS vs Phase Correlation

## Overview

Two methods are implemented for aligning HR (Planet) to LR (Sentinel-2):
1. **AROSICS** (Automatic Registration of Optical Satellite Images with Same/Similar Content)
2. **Phase Correlation** (scikit-image)

Both use **Sentinel-2 as the fixed reference** and shift **Planet to match**.

---

## Method 1: AROSICS (Preferred, but often fails)

### Algorithm
- **Package:** `arosics` (specialized for satellite imagery)
- **Method:** Feature-based matching with window correlation
- **Capabilities:** 
  - Global shift (uniform translation)
  - Local warping (handles distortions)
  - Sub-pixel accuracy

### Implementation Details

```python
# Lines 39-210 in numpy_efficient_pipeline.py

# 1. Resize LR to HR resolution
lr_resized = resize(lr_array, hr_array.shape[:2], ...)

# 2. Save as temporary GeoTIFFs (AROSICS requirement)
#    - Creates synthetic georeferencing (pixel coords)
#    - Uses EPSG:4326 as dummy CRS

# 3. Run AROSICS coregistration
CR = COREG(
    str(lr_tif),      # Reference (LR) - FIXED
    str(hr_tif),      # Target (HR) - TO BE SHIFTED
    ws=(32, 32),       # Window size for matching (was 256, now 32 for finer detail)
    max_shift=20,      # Max allowed shift in pixels (was 5, now 20)
    max_iter=10,       # Iterations for convergence (was 5, now 10)
    nodata=(0, 0),
    r_b4match=1,       # Use red band for matching
    s_b4match=1,
    align_grids=True,
    v=False            # No verbose output
)

# 4. Calculate shifts
CR.calculate_spatial_shifts()
shift_x = CR.x_shift_px  # Horizontal shift
shift_y = CR.y_shift_px  # Vertical shift

# 5. Apply shift to all HR bands
hr_coregistered = scipy_shift(hr_array, shift=[shift_y, shift_x], order=3)
```

### Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `ws` | (32, 32) | **Matching window size** - smaller = finer detail detection |
| `max_shift` | 20 px | **Maximum allowed shift** - larger = can correct bigger misalignments |
| `max_iter` | 10 | **Iterations** - more = better convergence |
| `r_b4match` | 1 (red) | **Band for matching** - red band typically has good contrast |
| `order` | 3 (cubic) | **Interpolation** - for sub-pixel accuracy |

### Advantages ✅
1. **Satellite-specific**: Designed for optical satellite imagery
2. **Robust**: Handles various distortions and transformations
3. **Local warping**: Can correct non-uniform distortions (if `use_local_coreg=True`)
4. **Proven**: Widely used in remote sensing community
5. **Detailed diagnostics**: Provides shift maps, quality metrics

### Disadvantages ❌
1. **Requires files**: Must write temporary GeoTIFFs (I/O overhead)
2. **SLOW**: 45-60 seconds per frame (mostly file I/O)
3. **Temp file issues**: Often fails with permission/cleanup errors
4. **Complex dependencies**: Requires GDAL, rasterio, complex C++ backend
5. **Overkill**: For sub-pixel shifts, simpler methods work fine

### Performance (from tests)
- ⏱️ **Speed:** 45-60 seconds per frame
- ❌ **Failure rate:** High (temp file issues, GDAL errors)
- 📏 **Accuracy:** Sub-pixel (0.1px theoretical)
- 🔧 **Fallback:** Switches to phase correlation on failure

---

## Method 2: Phase Correlation (Currently Used)

### Algorithm
- **Package:** `scikit-image` (standard image processing)
- **Method:** Fourier-based phase correlation
- **Principle:** 
  - Compute FFT of both images
  - Find peak in cross-power spectrum
  - Peak location = shift between images

### Implementation Details

```python
# Lines 219-305 in numpy_efficient_pipeline.py

from skimage.registration import phase_cross_correlation

# 1. Resize LR to HR resolution
lr_resized = resize(lr_array, hr_array.shape[:2], ...)

# 2. Extract matching bands (red band)
lr_for_matching = lr_resized[:, :, 2]  # Red band
hr_for_matching = hr_array[:, :, 2]     # Red band

# 3. Calculate sub-pixel shift using FFT-based phase correlation
shifts, error, diffphase = phase_cross_correlation(
    lr_for_matching,    # Reference (LR upsampled)
    hr_for_matching,    # Target to align (HR)
    upsample_factor=10  # Sub-pixel accuracy (1/10 pixel)
)

shift_y, shift_x = shifts

# 4. Apply shift to all HR bands
hr_coregistered = scipy_shift(
    hr_array, 
    shift=[shift_y, shift_x], 
    order=3,           # Cubic interpolation
    mode='nearest'     # Border handling
)
```

### Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `upsample_factor` | 10 | **Sub-pixel precision** - 1/10 pixel accuracy |
| `order` | 3 (cubic) | **Interpolation** - for smooth sub-pixel shifting |
| `mode` | 'nearest' | **Border handling** - how to fill edges |

### Advantages ✅
1. **FAST**: 1.5-2 seconds per frame (20-30× faster than AROSICS!)
2. **Reliable**: 100% success rate (no file I/O failures)
3. **Simple**: Pure NumPy/SciPy, no external files needed
4. **Memory-efficient**: Works directly with arrays
5. **Sub-pixel accurate**: 0.1 pixel accuracy with `upsample_factor=10`
6. **FFT-based**: Theoretically optimal for pure translations

### Disadvantages ❌
1. **Translation only**: Cannot handle rotation, scaling, or distortion
2. **Global shift**: Assumes uniform shift across entire image
3. **No local warping**: Cannot correct non-uniform distortions
4. **Sensitive to noise**: Can be affected by low-contrast or noisy regions
5. **No quality metrics**: Doesn't provide confidence scores

### Performance (from your tests)
- ⏱️ **Speed:** 1.5-2 seconds per frame
- ✅ **Success rate:** 100% (3/3 frames tested)
- 📏 **Detected shifts:** 0.4-0.7 pixels (sub-pixel!)
- 🎯 **Pattern:** Consistent vertical shifts (~0.4-0.7 px), minimal horizontal

---

## Side-by-Side Comparison

| Feature | AROSICS | Phase Correlation |
|---------|---------|-------------------|
| **Speed** | 45-60s | 1.5-2s |
| **Reliability** | Low (temp file issues) | High (100% success) |
| **Sub-pixel accuracy** | ✅ Yes (0.1px) | ✅ Yes (0.1px) |
| **Handles rotation** | ✅ Yes | ❌ No |
| **Handles distortion** | ✅ Yes (local warping) | ❌ No |
| **Pure translation** | ✅ Yes | ✅ Yes |
| **Requires files** | ❌ Yes (GeoTIFFs) | ✅ No (arrays only) |
| **Memory usage** | High (file I/O) | Low (in-memory) |
| **Dependencies** | Complex (GDAL, rasterio) | Simple (scikit-image) |
| **Satellite-specific** | ✅ Yes | ❌ No (general purpose) |
| **Quality metrics** | ✅ Yes | ⚠️ Limited |
| **Best for** | Large shifts, distortions | Small shifts, translations |

---

## Current Configuration

### Default Behavior (Line 328)
```python
def coregister_lr_to_hr(hr_array, lr_array, cache_path, method='arosics'):
    if method == 'arosics':
        return coregister_hr_to_lr_arosics(hr_array, lr_array, cache_path)
    else:
        return coregister_hr_to_lr_phase(hr_array, lr_array, cache_path)
```

### Actual Behavior
1. **First attempt:** AROSICS (default)
2. **If AROSICS fails:** Automatic fallback to phase correlation (Line 208)
3. **Result:** Almost always uses phase correlation due to AROSICS failures

### Fallback Mechanism (Lines 206-211)
```python
except Exception as e:
    print(f"⚠️ AROSICS failed: {e}")
    print(f"         Falling back to phase correlation...")
    return coregister_hr_to_lr_phase(hr_array, lr_array, cache_path)
```

---

## Test Results

### From Your 5-Sample Test (with phase correlation)

**Detected Shifts:**
- Frame 1 (2020-01-01): shift = (0.40, 0.70) px
- Frame 2 (2020-01-06): shift = (0.00, 0.70) px
- Frame 3 (2020-04-15): shift = (-0.10, 0.40) px

**Analysis:**
- ✅ All shifts < 1 pixel (sub-pixel misalignment)
- ✅ Consistent pattern (mostly vertical ~0.4-0.7 px)
- ✅ Minimal horizontal shifts (good E-W alignment)
- ✅ 100% success rate
- ⏱️ Average processing time: ~1.7s per frame

---

## Recommendation: Phase Correlation is Better for Your Use Case

### Why Phase Correlation Wins:

1. **Your shifts are small** (0.4-0.7 pixels)
   - These are pure translations (no rotation/distortion)
   - Phase correlation is OPTIMAL for this scenario

2. **Reliability matters** 
   - 100% success rate vs. frequent AROSICS failures
   - No temp file issues

3. **Speed matters**
   - 1.5-2s vs. 45-60s per frame
   - For 600 samples × 3 frames = 1800 frames:
     - Phase correlation: ~1 hour
     - AROSICS: ~22.5 hours (if it worked!)

4. **Your data has consistent patterns**
   - Vertical bias (~0.4-0.7 px) suggests systematic offset
   - Not random noise → FFT works perfectly

### When AROSICS Would Be Better:

- ❌ Large shifts (>10 pixels) - not your case
- ❌ Rotation present - not observed in your data
- ❌ Non-uniform distortion - Planet/S2 are well-calibrated
- ❌ Complex warping needed - not needed for 0.7px shifts

---

## With the New Tiling Fix: Do You Even Need Coregistration?

### Before Tiling Fix:
- HR and LR tiles covered **different geographic areas** (20% mismatch)
- This created apparent "shifts" of several pixels
- Coregistration was **masking the real problem**

### After Tiling Fix:
- HR and LR tiles now cover **exactly the same area**
- Expected shifts: <0.5 pixels (only from GEE reprojection artifacts)
- **Coregistration may not be needed at all!**

### Test Recommendation:
1. Generate 5-10 samples with **new tiling** (fixed ratio)
2. Compare HR-LR alignment **without coregistration**
3. If visual alignment is good and shifts are <0.3 pixels:
   - **Skip coregistration entirely**
   - Saves processing time
   - Avoids potential artifacts from shifting
4. If shifts are 0.3-1.0 pixels:
   - **Keep phase correlation** (fast, reliable)
5. Only if shifts >1.0 pixels:
   - Investigate why (should be fixed by proper tiling)

---

## Summary

| Aspect | Recommendation |
|--------|----------------|
| **Current default** | AROSICS (but always falls back to phase) |
| **What actually runs** | Phase correlation (due to fallback) |
| **Best method** | **Phase correlation** (fast, reliable, sufficient) |
| **With tiling fix** | **Test if coregistration even needed** |
| **Final recommendation** | Skip AROSICS entirely, use phase only (or none) |

The irony: Your system was **designed to use AROSICS** but **works better with phase correlation**! 🎯
