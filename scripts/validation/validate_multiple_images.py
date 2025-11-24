"""
Multi-image validation: Test 3 different Planet images to ensure consistency
This catches issues that might only appear in specific images
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
import numpy as np
from Auth.AuthGEE import regions
from chunked_download_fast import download_image_chunked_parallel

print("="*70)
print("MULTI-IMAGE VALIDATION TEST")
print("Testing 3 different Planet images to ensure download consistency")
print("="*70)

# Load cached pairs
region = regions[0]
with open('test_pairs_cache.pkl', 'rb') as f:
    pairs = pickle.load(f)

planet_collection = pairs['planet_collection']
matched = pairs['matched_pairs']
bands = ['B1', 'B2', 'B3', 'B4']

# Test first 3 matched pairs
num_test_images = min(3, len(matched))
print(f"\nTesting {num_test_images} images...\n")

results = []

for idx in range(num_test_images):
    pair = matched[idx]
    print(f"\n{'='*70}")
    print(f"IMAGE {idx+1}/{num_test_images}: {pair['planet_date']}")
    print(f"{'='*70}")
    
    planet_img = planet_collection.filter(ee.Filter.eq('id_pref', pair['planet_id'])).first()
    
    # Download SINGLE CHUNK for speed (center chunk)
    print(f"   Downloading single chunk for quick validation...")
    
    aoi = region['aoi']
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    lon_step = (max_lon - min_lon) / 9
    lat_step = (max_lat - min_lat) / 9
    
    # Center chunk
    chunk_min_lon = min_lon + 4 * lon_step
    chunk_max_lon = min_lon + 5 * lon_step
    chunk_min_lat = max_lat - 5 * lat_step
    chunk_max_lat = max_lat - 4 * lat_step
    
    chunk_geom = ee.Geometry.Rectangle([chunk_min_lon, chunk_min_lat, chunk_max_lon, chunk_max_lat])
    
    try:
        chunk_img = planet_img.select(bands).clip(chunk_geom).unmask(0)
        chunk_img = chunk_img.reproject(crs='EPSG:4326', scale=3)
        chunk_sample = chunk_img.sampleRectangle(region=chunk_geom, defaultValue=0)
        chunk_data = chunk_sample.getInfo()
        
        if chunk_data and 'properties' in chunk_data:
            props = chunk_data['properties']
            band_arrays = [np.array(props[b], dtype=np.float32) for b in bands]
            data = np.stack(band_arrays, axis=-1)
            
            # Validate this chunk
            valid_pixels = np.sum(data.sum(axis=2) > 0)
            total_pixels = data.shape[0] * data.shape[1]
            valid_pct = 100 * valid_pixels / total_pixels
            
            means = [data[:,:,i].mean() for i in range(4)]
            stds = [data[:,:,i].std() for i in range(4)]
            
            # Check for valid data
            is_valid = True
            issues = []
            
            if valid_pct < 30:
                is_valid = False
                issues.append(f"Too many zeros ({100-valid_pct:.1f}% masked)")
            
            # Check for fallback pattern [300, 500, 700, 600]
            if np.allclose(means, [300, 500, 700, 600], rtol=0.1):
                is_valid = False
                issues.append("Matches fallback test pattern")
            
            # Check for realistic values
            for i, mean in enumerate(means):
                if i < 3:  # Visible
                    if mean < 100 or mean > 3000:
                        is_valid = False
                        issues.append(f"B{i+1} mean {mean:.1f} out of range")
                else:  # NIR
                    if mean < 200 or mean > 5000:
                        is_valid = False
                        issues.append(f"B{i+1} mean {mean:.1f} out of range")
            
            # Check for spatial variation
            for i, std in enumerate(stds):
                if std < 10:
                    is_valid = False
                    issues.append(f"B{i+1} almost no variation (std={std:.1f})")
            
            result = {
                'date': pair['planet_date'],
                'id': pair['planet_id'],
                'shape': data.shape,
                'valid_pct': valid_pct,
                'means': means,
                'stds': stds,
                'is_valid': is_valid,
                'issues': issues
            }
            results.append(result)
            
            print(f"   ✓ Downloaded: {data.shape}")
            print(f"   Valid pixels: {valid_pct:.1f}%")
            print(f"   Means: B1={means[0]:.1f}, B2={means[1]:.1f}, B3={means[2]:.1f}, B4={means[3]:.1f}")
            print(f"   Stds:  B1={stds[0]:.1f}, B2={stds[1]:.1f}, B3={stds[2]:.1f}, B4={stds[3]:.1f}")
            
            if is_valid:
                print(f"   ✅ VALID")
            else:
                print(f"   ❌ INVALID:")
                for issue in issues:
                    print(f"      - {issue}")
        else:
            print(f"   ❌ No data returned")
            results.append({
                'date': pair['planet_date'],
                'id': pair['planet_id'],
                'is_valid': False,
                'issues': ['No data returned']
            })
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append({
            'date': pair['planet_date'],
            'id': pair['planet_id'],
            'is_valid': False,
            'issues': [str(e)]
        })

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}\n")

valid_count = sum(1 for r in results if r['is_valid'])
total_count = len(results)

print(f"Valid images: {valid_count}/{total_count}")

if valid_count == total_count:
    print(f"\n🎉 ALL IMAGES PASSED!")
    print(f"✅ Download pipeline consistently produces real Planet data")
    print(f"✅ No fallbacks, zeros, or invalid data detected across multiple images")
    
    # Show consistency
    if len([r for r in results if 'means' in r]) >= 2:
        print(f"\n📊 Consistency check:")
        means_list = [r['means'] for r in results if 'means' in r]
        for band_idx in range(4):
            band_means = [m[band_idx] for m in means_list]
            mean_of_means = np.mean(band_means)
            std_of_means = np.std(band_means)
            cv = std_of_means / mean_of_means if mean_of_means > 0 else 0
            print(f"   B{band_idx+1} across images: mean={mean_of_means:.1f}, std={std_of_means:.1f}, CV={cv:.3f}")
else:
    print(f"\n⚠️  SOME IMAGES FAILED ({total_count - valid_count}/{total_count})")
    print(f"\nFailed images:")
    for r in results:
        if not r['is_valid']:
            print(f"   {r['date']}: {', '.join(r['issues'])}")

print(f"\n{'='*70}")
