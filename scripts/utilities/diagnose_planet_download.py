"""
Diagnose why Planet downloads are failing.
Test each step of the download process.
"""

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
import numpy as np
from Auth.AuthGEE import regions, PLANET_COLLECTION_ID, _add_pref_id
from Auth.AuthGEE import planet_best_per_day_adaptive

print("="*70)
print("PLANET DOWNLOAD DIAGNOSTIC")
print("="*70)

# Use first region (STA-2)
region = regions[0]
print(f"\n📍 Testing region: {region['name']}")
print(f"   Date range: {region['start']} to {region['end']}")
print(f"   AOI: {region['aoi'].bounds().getInfo()}")

# Step 1: Check Planet collection access
print("\n" + "="*70)
print("STEP 1: Check Planet Collection Access")
print("="*70)

try:
    pl_col = ee.ImageCollection(PLANET_COLLECTION_ID).filterBounds(region['aoi'])
    count = pl_col.size().getInfo()
    print(f"✅ Planet collection accessible")
    print(f"   Collection ID: {PLANET_COLLECTION_ID}")
    print(f"   Images in AOI: {count}")
    
    if count == 0:
        print("   ⚠️  WARNING: No images found in AOI!")
        print("   Trying without AOI filter...")
        pl_col_all = ee.ImageCollection(PLANET_COLLECTION_ID)
        total_count = pl_col_all.size().getInfo()
        print(f"   Total images in collection: {total_count}")
        
except Exception as e:
    print(f"❌ Failed to access Planet collection: {e}")
    exit(1)

# Step 2: Get first image and check properties
print("\n" + "="*70)
print("STEP 2: Check Image Properties")
print("="*70)

try:
    first_img = pl_col.first()
    props = first_img.propertyNames().getInfo()
    
    print(f"✅ First image properties ({len(props)} total):")
    important_props = ['system:id', 'system:index', 'system:time_start', 'id_pref']
    for prop in important_props:
        if prop in props:
            val = first_img.get(prop).getInfo()
            print(f"   - {prop}: {val}")
        else:
            print(f"   - {prop}: NOT FOUND")
    
    # Check bands
    bands = first_img.bandNames().getInfo()
    print(f"\n   Bands available: {bands}")
    
except Exception as e:
    print(f"❌ Failed to get image properties: {e}")
    exit(1)

# Step 3: Test planet_best_per_day_adaptive
print("\n" + "="*70)
print("STEP 3: Test planet_best_per_day_adaptive()")
print("="*70)

try:
    # Filter by date
    pl_col_filtered = pl_col.filterDate(region['start'], region['end'])
    print(f"   Images in date range: {pl_col_filtered.size().getInfo()}")
    
    # First, add id_pref since images don't have it
    pl_col_with_id = pl_col_filtered.map(_add_pref_id)
    print(f"   Added id_pref to images")
    
    # Apply planet_best_per_day_adaptive (only takes col and aoi)
    pl_best = planet_best_per_day_adaptive(pl_col_with_id, region['aoi'])
    
    pl_best_count = pl_best.size().getInfo()
    print(f"✅ planet_best_per_day_adaptive completed")
    print(f"   Output images: {pl_best_count}")
    
    if pl_best_count > 0:
        # Get first processed image
        first_processed = pl_best.first()
        proc_props = first_processed.propertyNames().getInfo()
        
        print(f"\n   Processed image properties:")
        key_props = ['id_pref', 'date_str', 'score', 'q_cloud', 'aoi_cloud_frac']
        for prop in key_props:
            if prop in proc_props:
                val = first_processed.get(prop).getInfo()
                print(f"   - {prop}: {val}")
        
        # Check bands
        proc_bands = first_processed.bandNames().getInfo()
        print(f"   - bands: {proc_bands}")
    else:
        print("   ⚠️  WARNING: No images after planet_best_per_day_adaptive!")
        
except Exception as e:
    print(f"❌ planet_best_per_day_adaptive failed: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Test actual download using reduceRegion
print("\n" + "="*70)
print("STEP 4: Test Data Download (reduceRegion)")
print("="*70)

try:
    if pl_best_count > 0:
        test_img = pl_best.first()
        test_bands = test_img.bandNames().getInfo()
        
        # Use first 4 bands (harmonized to 4-band)
        if len(test_bands) >= 4:
            bands_to_use = test_bands[:4]
        else:
            bands_to_use = test_bands
        
        print(f"   Testing bands: {bands_to_use}")
        
        # Get small region (1000m buffer around centroid)
        centroid = region['aoi'].centroid()
        test_region = centroid.buffer(1000)
        
        print(f"   Test region: 1km buffer around AOI centroid")
        
        # Try reduceRegion
        stats = test_img.select(bands_to_use).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=test_region,
            scale=10,
            maxPixels=1e8
        )
        
        stats_dict = stats.getInfo()
        
        print(f"✅ reduceRegion successful!")
        print(f"   Results:")
        for band, value in stats_dict.items():
            if value is not None:
                print(f"   - {band}: {value:.2f}")
            else:
                print(f"   - {band}: None")
        
        # Check if any values are None
        none_count = sum(1 for v in stats_dict.values() if v is None)
        if none_count > 0:
            print(f"   ⚠️  {none_count}/{len(stats_dict)} bands returned None!")
        
    else:
        print("   ⚠️  Skipping download test - no images available")
        
except Exception as e:
    print(f"❌ reduceRegion failed: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Test with specific tile bounds (like in actual pipeline)
print("\n" + "="*70)
print("STEP 5: Test Tile Download (256x256 px)")
print("="*70)

try:
    if pl_best_count > 0:
        # Get bounds of AOI
        aoi_bounds = region['aoi'].bounds().getInfo()['coordinates'][0]
        min_lon = min(pt[0] for pt in aoi_bounds)
        max_lon = max(pt[0] for pt in aoi_bounds)
        min_lat = min(pt[1] for pt in aoi_bounds)
        max_lat = max(pt[1] for pt in aoi_bounds)
        
        # Create a small tile in the center
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        tile_size_deg = 0.01  # ~1km at equator
        
        tile_bounds = [
            center_lon - tile_size_deg/2,
            center_lat - tile_size_deg/2,
            center_lon + tile_size_deg/2,
            center_lat + tile_size_deg/2
        ]
        
        tile_geom = ee.Geometry.Rectangle(tile_bounds)
        
        print(f"   Tile bounds: {tile_bounds}")
        
        # Try to get pixel data
        test_img = pl_best.first()
        bands_to_use = test_img.bandNames().getInfo()[:4]
        
        # Use reduceRegion to get mean values
        tile_stats = test_img.select(bands_to_use).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=tile_geom,
            scale=3,  # Planet scale
            maxPixels=1e9
        )
        
        tile_dict = tile_stats.getInfo()
        
        print(f"✅ Tile download successful!")
        print(f"   Mean values:")
        for band, value in tile_dict.items():
            if value is not None:
                print(f"   - {band}: {value:.2f}")
            else:
                print(f"   - {band}: None")
        
        # This is what would be used to create the tile array
        if all(v is not None for v in tile_dict.values()):
            mean_values = [tile_dict[band] for band in bands_to_use]
            print(f"\n   ✅ Would create tile with values: {mean_values}")
            
            # Check if these look like test fallback values
            test_fallback = [300, 500, 700, 600]
            if np.allclose(mean_values, test_fallback, atol=50):
                print(f"   ⚠️  WARNING: Values match test fallback pattern!")
            else:
                print(f"   ✅ Values look like real satellite data")
        else:
            print(f"   ❌ Some bands returned None - would trigger fallback!")
        
except Exception as e:
    print(f"❌ Tile download failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
print("\nCheck the results above to identify where the download fails.")
print("Common issues:")
print("  1. No images in AOI - check Planet collection coverage")
print("  2. planet_best_per_day_adaptive returns 0 images - check filters")
print("  3. reduceRegion returns None - check image availability/permissions")
print("  4. Values match test fallback - actual data never downloaded")
