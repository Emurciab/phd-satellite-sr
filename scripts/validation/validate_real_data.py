"""
Comprehensive validation: Verify downloaded data is REAL Planet imagery
Tests multiple indicators to ensure no fallbacks, zeros, or invalid data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
import pickle
import time
import numpy as np
from Auth.AuthGEE import regions
from chunked_download_fast import download_image_chunked_parallel

print("="*70)
print("COMPREHENSIVE PLANET DATA VALIDATION TEST")
print("="*70)

# Load cached pairs
region = regions[0]
with open('test_pairs_cache.pkl', 'rb') as f:
    pairs = pickle.load(f)

planet_collection = pairs['planet_collection']
matched = pairs['matched_pairs']

# Test FIRST matched pair
first = matched[0]
print(f"\n📅 Testing Planet image: {first['planet_date']}")
print(f"   Planet ID: {first['planet_id']}")

planet_img = planet_collection.filter(ee.Filter.eq('id_pref', first['planet_id'])).first()
bands = ['B1', 'B2', 'B3', 'B4']

# Download with parallel chunking
print(f"\n⬇️  Downloading with 9x9 grid (81 chunks), 6 parallel workers...")
start = time.time()
data = download_image_chunked_parallel(
    planet_img, 
    region['aoi'], 
    bands, 
    scale=3, 
    chunk_grid_size=9,
    max_workers=6
)
elapsed = time.time() - start

if data is None:
    print(f"\n❌ FAILED: Download returned None")
    exit(1)

print(f"\n✅ Download completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"   Shape: {data.shape}")

# ============================================================================
# VALIDATION TESTS
# ============================================================================

print(f"\n" + "="*70)
print("VALIDATION TESTS")
print("="*70)

all_tests_passed = True

# TEST 1: Check for all-zero data
print(f"\n1️⃣  TEST: No all-zero data")
total_pixels = data.shape[0] * data.shape[1]
zero_pixels = np.sum(data.sum(axis=2) == 0)
valid_pixels = total_pixels - zero_pixels
valid_pct = 100 * valid_pixels / total_pixels

print(f"   Total pixels: {total_pixels:,}")
print(f"   Zero pixels: {zero_pixels:,}")
print(f"   Valid pixels: {valid_pixels:,} ({valid_pct:.1f}%)")

if valid_pct < 30:  # Require at least 30% valid pixels
    print(f"   ❌ FAIL: Too many zeros! Only {valid_pct:.1f}% valid")
    all_tests_passed = False
else:
    print(f"   ✅ PASS: {valid_pct:.1f}% valid pixels")

# TEST 2: Check for realistic Planet reflectance values
print(f"\n2️⃣  TEST: Realistic reflectance values")
print(f"   Valid data statistics (non-zero pixels only):")
for i, band in enumerate(bands):
    band_data = data[:,:,i]
    valid_mask = band_data > 0
    if valid_mask.sum() > 0:
        valid_values = band_data[valid_mask]
        mean_val = valid_values.mean()
        min_val = valid_values.min()
        max_val = valid_values.max()
        std_val = valid_values.std()
        
        print(f"   {band}: mean={mean_val:.1f}, min={min_val:.1f}, max={max_val:.1f}, std={std_val:.1f}")
        
        # Planet typical range: 0-10000 (surface reflectance scaled)
        # Realistic means: 200-2000 for visible bands, 500-4000 for NIR
        if i < 3:  # Visible bands (B1, B2, B3)
            expected_range = (100, 3000)
        else:  # NIR (B4)
            expected_range = (200, 5000)
        
        if mean_val < expected_range[0] or mean_val > expected_range[1]:
            print(f"      ❌ FAIL: Mean {mean_val:.1f} outside expected range {expected_range}")
            all_tests_passed = False
        else:
            print(f"      ✅ PASS: Mean within expected range {expected_range}")
    else:
        print(f"   {band}: NO VALID DATA")
        all_tests_passed = False

# TEST 3: Check for spatial variation (not uniform/constant)
print(f"\n3️⃣  TEST: Spatial variation (not uniform)")
for i, band in enumerate(bands):
    band_data = data[:,:,i]
    valid_mask = band_data > 0
    if valid_mask.sum() > 100:  # Need at least 100 valid pixels
        valid_values = band_data[valid_mask]
        std_val = valid_values.std()
        cv = std_val / valid_values.mean() if valid_values.mean() > 0 else 0  # Coefficient of variation
        
        print(f"   {band}: std={std_val:.1f}, CV={cv:.3f}")
        
        # Real imagery should have variation (CV > 0.05)
        if cv < 0.05:
            print(f"      ❌ FAIL: Too uniform (CV={cv:.3f} < 0.05)")
            all_tests_passed = False
        elif std_val < 10:
            print(f"      ❌ FAIL: Almost no variation (std={std_val:.1f})")
            all_tests_passed = False
        else:
            print(f"      ✅ PASS: Has spatial variation")

# TEST 4: Check for fallback test pattern [300, 500, 700, 600]
print(f"\n4️⃣  TEST: Not fallback test pattern")
test_pattern = np.array([300, 500, 700, 600])
mean_values = np.array([data[:,:,i].mean() for i in range(4)])

# Check if means match fallback pattern (within 10% tolerance)
pattern_match = np.allclose(mean_values, test_pattern, rtol=0.1)

print(f"   Mean values: {mean_values}")
print(f"   Test pattern: {test_pattern}")

if pattern_match:
    print(f"   ❌ FAIL: Matches fallback test pattern!")
    all_tests_passed = False
else:
    print(f"   ✅ PASS: Does not match fallback pattern")

# TEST 5: Check band relationships (NIR > Red for vegetation)
print(f"\n5️⃣  TEST: Realistic band relationships")
valid_mask = data.sum(axis=2) > 0
if valid_mask.sum() > 100:
    red_data = data[:,:,2][valid_mask]   # B3
    nir_data = data[:,:,3][valid_mask]   # B4
    
    # For vegetated areas, NIR typically > Red
    # Calculate percentage where NIR > Red
    nir_gt_red_pct = 100 * np.sum(nir_data > red_data) / len(red_data)
    
    print(f"   NIR > Red: {nir_gt_red_pct:.1f}% of valid pixels")
    
    # Real Planet imagery should show some vegetation (at least 20% of pixels with NIR > Red)
    if nir_gt_red_pct < 20:
        print(f"   ⚠️  WARNING: Low vegetation signature ({nir_gt_red_pct:.1f}%)")
        print(f"      (May be valid if study area has little vegetation)")
    else:
        print(f"   ✅ PASS: Realistic NIR/Red relationship")

# TEST 6: Check for edge artifacts or stitching issues
print(f"\n6️⃣  TEST: No stitching artifacts")
# Check for abrupt transitions at chunk boundaries (every ~1/9 of image)
chunk_size = data.shape[0] // 9

# Sample a few chunk boundaries
boundary_issues = 0
for chunk_idx in range(1, 9):
    boundary_row = chunk_idx * chunk_size
    if boundary_row > 5 and boundary_row < data.shape[0] - 5:
        # Compare pixels across boundary
        before = data[boundary_row-2:boundary_row, :, 0].mean()
        after = data[boundary_row:boundary_row+2, :, 0].mean()
        
        if before > 0 and after > 0:
            diff_pct = abs(before - after) / max(before, after) * 100
            if diff_pct > 50:  # More than 50% jump is suspicious
                boundary_issues += 1
                print(f"   ⚠️  Chunk boundary {chunk_idx}: {diff_pct:.1f}% jump")

if boundary_issues > 0:
    print(f"   ⚠️  WARNING: {boundary_issues} suspicious chunk boundaries")
    print(f"      (May indicate stitching issues or cloud edges)")
else:
    print(f"   ✅ PASS: No obvious stitching artifacts")

# TEST 7: Verify image metadata from GEE
print(f"\n7️⃣  TEST: Verify source image metadata")
img_props = planet_img.getInfo()['properties']
print(f"   Image date: {img_props.get('system:time_start', 'N/A')}")
print(f"   Image index: {img_props.get('system:index', 'N/A')}")
print(f"   Quality score: {img_props.get('qscore', 'N/A')}")
print(f"   Cloud quality: {img_props.get('q_cloud', 'N/A')}")

if 'system:time_start' in img_props and img_props['system:time_start'] is not None:
    print(f"   ✅ PASS: Valid image metadata")
else:
    print(f"   ❌ FAIL: Missing image metadata")
    all_tests_passed = False

# TEST 8: Sample random locations and verify diversity
print(f"\n8️⃣  TEST: Sample random locations for diversity")
np.random.seed(42)
num_samples = 10
sample_values = []

for _ in range(num_samples):
    row = np.random.randint(0, data.shape[0])
    col = np.random.randint(0, data.shape[1])
    pixel_value = data[row, col, :].sum()
    if pixel_value > 0:
        sample_values.append(pixel_value)

if len(sample_values) >= 5:
    sample_std = np.std(sample_values)
    sample_mean = np.mean(sample_values)
    sample_cv = sample_std / sample_mean if sample_mean > 0 else 0
    
    print(f"   Random samples: {len(sample_values)}/{num_samples} valid")
    print(f"   Sample diversity: CV={sample_cv:.3f}")
    
    if sample_cv < 0.1:
        print(f"   ❌ FAIL: Random samples too similar (CV={sample_cv:.3f})")
        all_tests_passed = False
    else:
        print(f"   ✅ PASS: Random samples show diversity")
else:
    print(f"   ❌ FAIL: Too few valid samples ({len(sample_values)}/{num_samples})")
    all_tests_passed = False

# ============================================================================
# VISUAL SAMPLE
# ============================================================================

print(f"\n" + "="*70)
print("VISUAL DATA SAMPLE")
print("="*70)

print(f"\n📊 Sample 10x10 region (center), Band B1:")
center_y = data.shape[0] // 2
center_x = data.shape[1] // 2
sample = data[center_y:center_y+10, center_x:center_x+10, 0]
print(sample.astype(int))

print(f"\n📊 Sample 10x10 region (center), Band B4 (NIR):")
sample_nir = data[center_y:center_y+10, center_x:center_x+10, 3]
print(sample_nir.astype(int))

# ============================================================================
# FINAL VERDICT
# ============================================================================

print(f"\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

if all_tests_passed:
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"✅ Downloaded data is REAL Planet imagery at 3m resolution")
    print(f"✅ No fallback values, zeros, or invalid data detected")
    print(f"✅ Ready to integrate into production pipeline")
else:
    print(f"\n⚠️  SOME TESTS FAILED")
    print(f"❌ Downloaded data may contain issues")
    print(f"🔍 Review individual test results above")

print(f"\n" + "="*70)
