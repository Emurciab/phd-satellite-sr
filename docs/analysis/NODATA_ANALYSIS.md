# Analysis: Why HR (Planet) Has Nodata While LR (Sentinel-2) Is Complete

## TL;DR - Root Cause

**Planet images are MOSAICS** with **limited spatial coverage**. Random tile sampling hits areas outside Planet's actual data coverage, while Sentinel-2 provides complete AOI coverage.

## Evidence from 5 Samples

```
Sample 1: HR 0.0% data,   LR 100.0% data  ⚠️ Complete miss
Sample 2: HR 20.7% data,  LR 100.0% data  ⚠️ Partial coverage
Sample 3: HR 74.8% data,  LR 100.0% data  ✓ Good coverage
Sample 4: HR 0.0% data,   LR 100.0% data  ⚠️ Complete miss
Sample 5: HR 100.0% data, LR 98.7% data   ✅ Perfect coverage
```

**Key observation**: LR (Sentinel-2) has 98-100% data coverage in ALL samples, while HR (Planet) varies from 0% to 100%.

## Why This Happens

### 1. Planet Collection Structure (CRITICAL)

From `Auth/AuthGEE.py`:

```python
# Line 151-161: planet_best_per_day function
def planet_best_per_day(col):
    dates = col.aggregate_array('date_key').distinct()
    def _pick(dk):
        day_col = col.filter(ee.Filter.eq('date_key', dk))
        first_img = day_col.first()
        
        # Create the mosaic ← KEY ISSUE
        mosaic = day_col.qualityMosaic('qscore')  
        
        return mosaic.copyProperties(first_img, ['date_key', 'id_pref', 'time_start'])
    
    return ee.ImageCollection(dates.map(_pick))
```

**What `qualityMosaic` does:**
- Combines multiple Planet scenes from the same day
- Selects best pixel based on `qscore` (quality score)
- **BUT**: Only creates pixels where Planet data exists
- **Does NOT fill gaps** - leaves nodata (masked) where no scenes overlap

### 2. Planet vs Sentinel-2 Coverage Differences

**Sentinel-2 (LR):**
- **Systematic coverage**: Full Earth coverage every 5 days
- **Single satellite swath**: Each image covers ~290 km × 290 km
- **Guaranteed overlap**: AOI (STA-2 region) fits within single swath
- **Result**: 100% coverage of AOI ✅

**Planet (HR):**
- **Opportunistic imaging**: Multiple small satellites (Doves)
- **Small footprint**: Each scene is ~24 km × 8 km (very narrow!)
- **Mosaicking required**: Need to combine 10+ scenes to cover AOI
- **Gaps possible**: Not all areas imaged on same day
- **Result**: Partial coverage with nodata gaps ⚠️

### 3. Random Tile Sampling Problem

From `generate_sta2_ultrafast.py` line 111-129:

```python
def generate_tile_bounds(aoi, num_tiles=10):
    # ...
    # Generate random tiles within bounds
    import random
    tiles = []
    for i in range(num_tiles):
        # Random position within AOI
        tile_min_lon = random.uniform(min_lon, max_lon - tile_size_deg)
        tile_min_lat = random.uniform(min_lat, max_lat - tile_size_deg)
        tiles.append([...])
    
    return tiles
```

**Problem**: 
- Samples random locations uniformly across AOI
- **Does NOT check** if Planet data exists at that location
- Can easily hit gaps in Planet mosaic

### 4. Download Process Does NOT Detect Coverage

From `generate_sta2_ultrafast.py` line 41-58:

```python
def download_tile_region(...):
    # Download HR tile
    hr_selected = planet_img.select(planet_bands).clip(tile_geom).unmask(0)  # ← unmask fills with 0
    hr_resampled = hr_selected.reproject(crs='EPSG:4326', scale=hr_scale)
    hr_rect = hr_resampled.sampleRectangle(region=tile_geom, defaultValue=0)  # ← defaultValue=0
```

**What happens:**
1. `clip(tile_geom)` clips Planet mosaic to tile bounds
2. **If tile is outside Planet coverage**: Result is fully masked (nodata)
3. `unmask(0)` converts masked pixels to zeros
4. `sampleRectangle(defaultValue=0)` fills any remaining nodata with zeros
5. **Result**: Arrays full of zeros are returned ⚠️

**No validation check** to detect if tile has valid data before proceeding!

## Why Not Padding? (Answer to Your Question)

### You asked: "Is it because samples were taken from zones near edges that were padded?"

**Answer: NO, it's NOT edge padding.**

Here's why:

1. **No padding in download code**: The ultra-fast script does NOT pad images
   - It downloads exact tile regions via `sampleRectangle`
   - No cropping or padding operations in the pipeline

2. **Different from full-image approach**: The OLD pipeline (that failed) would:
   - Download full 3800×5000 images
   - Crop tiles from them
   - Use `_pad_to_size()` for edge tiles (line 1413 in numpy_efficient_pipeline.py)
   
   But the **ultra-fast approach skips this entirely** - downloads tiles directly

3. **Verified by tile bounds**: Looking at the sample bounds:
   ```
   Sample 1: [-80.509, 26.427] → [-80.502, 26.434]  (interior of AOI)
   Sample 2: [-80.520, 26.358] → [-80.513, 26.365]  (interior)
   Sample 4: [-80.504, 26.337] → [-80.497, 26.344]  (interior)
   ```
   These are all **well within the AOI interior**, not edge tiles!

4. **Sentinel-2 has full coverage**: If it were edge padding:
   - Both HR and LR would be padded
   - But LR is 100% real data in all samples
   - Only HR has gaps

### You asked: "Why would it be padded from Planet only?"

**Answer: It's NOT padding - it's MISSING DATA in the Planet mosaic.**

The zeros in HR are **real nodata** from GEE, not padding added by our code:
- Planet mosaic has gaps (not all areas imaged)
- `unmask(0)` converts those gaps to zeros
- Sentinel-2 has no gaps (complete coverage)

## The Real Problem: Coverage Validation Missing

### What SHOULD happen (but doesn't):

```python
# BEFORE downloading tile
valid_pixels_hr = check_pixel_coverage(planet_img, tile_geom, threshold=0.9)
valid_pixels_lr = check_pixel_coverage(s2_img, tile_geom, threshold=0.9)

if not (valid_pixels_hr and valid_pixels_lr):
    print("      ⚠️ Insufficient coverage, skipping tile")
    continue  # Try next tile

# THEN download tile
hr_array, lr_array = download_tile_region(...)
```

### Why we don't have this:

Looking at the generation logs:
```
⚠️ AROSICS failed: The image to be shifted passed to AROSICS only contains nodata values.
```

This message appears AFTER download - we detect the problem too late!

## Why Sentinel-2 Always Has 100% Coverage

### From Sentinel-2 Processing in AuthGEE.py:

```python
# Lines 334-352: Sentinel-2 collection processing
s2_best = (
    s2_raw
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))  # Filter by cloud percentage
    .map(make_add_aoi_cloud_fraction(aoi))
    .filter(ee.Filter.lt('AOI_CLOUD_FRAC', 0.1))  # Filter by AOI-specific clouds
    .map(_add_s2_quality)
    .map(_best_per_day)
)
```

**Key differences:**
1. **Single swath coverage**: S2 image is ONE continuous swath
2. **No mosaicking**: Each date is a single image (or simple best_per_day from overlaps)
3. **Systematic acquisition**: Always covers entire AOI
4. **Cloud filtering only**: Filters by clouds, but data exists everywhere

**Planet by contrast:**
```python
# Lines 365-384: Planet collection with adaptive mosaicking
pl_best = planet_best_per_day_adaptive(pl_raw, aoi)
```
- Requires **mosaicking 10+ small scenes**
- Coverage depends on which scenes were acquired
- Gaps are normal and expected

## Solution: Implement Coverage Validation

### Option 1: Pre-check Before Sampling (RECOMMENDED)

```python
def get_valid_tile_locations(planet_col, s2_col, aoi, num_tiles=20):
    """
    Generate tiles ONLY in areas with valid Planet + S2 coverage.
    """
    # 1. Get Planet coverage mask for the date
    planet_img = planet_col.first()
    planet_mask = planet_img.mask().reduce(ee.Reducer.min())
    
    # 2. Sample random points within masked region
    valid_points = planet_mask.sample(
        region=aoi,
        scale=50,
        numPixels=num_tiles * 5,  # Oversample
        seed=42
    )
    
    # 3. Convert to tile bounds
    # ... (filter to num_tiles)
    
    return tile_bounds_list
```

### Option 2: Check After Download (CURRENT - WASTEFUL)

```python
# After download
if hr_array is None or lr_array is None:
    continue

# Check for nodata
hr_valid_percent = (np.count_nonzero(hr_array) / hr_array.size) * 100
if hr_valid_percent < 90:
    print(f"      ⚠️ HR only {hr_valid_percent:.1f}% valid, skipping")
    continue
```

**Problem**: Already wasted time downloading!

### Option 3: Use Coverage Metadata (BEST FOR PRODUCTION)

```python
# Use Planet's 'pixel_qa' or UDM bands to check coverage
def has_sufficient_coverage(img, geom, min_percent=90):
    # Get valid pixel mask
    valid_mask = img.select('B1').mask()
    
    # Count valid pixels in region
    stats = valid_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=10,
        maxPixels=1e6
    )
    
    valid_fraction = stats.get('B1').getInfo()
    return valid_fraction >= (min_percent / 100.0)
```

## Summary

| Aspect | Sentinel-2 (LR) | Planet (HR) |
|--------|----------------|-------------|
| **Coverage** | Systematic, 100% AOI | Opportunistic, partial |
| **Image type** | Single swath | Mosaic of 10+ scenes |
| **Gaps** | Never (except clouds) | Common |
| **Our sampling** | Always hits data | 40% miss rate (2/5 samples) |
| **Nodata cause** | Clouds only | Missing scenes |

**The nodata in HR is NOT from padding - it's from Planet's limited coverage in the mosaic.**

**Solution**: Implement coverage pre-checking before tile sampling, or switch to grid-based sampling starting from known valid regions.
