"""
Test if the fixed download function can get real Planet data.
"""
"""
Minimal test to verify Planet tile download after the fix.
It requests mean values via reduceRegion for a single matched Planet image
and prints whether they are real values or the test fallback.
"""

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
from numpy_efficient_pipeline import TensorFlowFreeDataPipeline
from Auth.AuthGEE import find_pairs, regions, PLANET_COLLECTION_ID

def main():
    pipeline = TensorFlowFreeDataPipeline(cache_dir="./test_fixed_download_cache", tile_size=64)

    region = regions[0]
    print(f"Testing download fix with region: {region['name']} ({region['start']} to {region['end']})")

    pairs = find_pairs(region['aoi'], region['start'], region['end'], PLANET_COLLECTION_ID)
    matched = pairs.get('matched_pairs', [])
    planet_collection = pairs.get('planet_collection')

    if not matched:
        print("No matched pairs found - aborting test")
        return

    first = matched[0]
    planet_id = first['planet_id']
    print(f"Using planet_id: {planet_id}")

    # Create a small tile bounds around AOI centroid
    centroid = region['aoi'].centroid().getInfo()['coordinates']
    lon, lat = centroid[0], centroid[1]
    delta = 0.005  # small bbox (~500m)
    tile_bounds = [lon - delta, lat - delta, lon + delta, lat + delta]

    print("Requesting mean values via pipeline.download_satellite_tile_gee...")

    try:
        tile = pipeline.download_satellite_tile_gee(
            image_collection=planet_collection,
            image_id=planet_id,
            tile_bounds=tile_bounds,
            sensor='planet',
            target_size=16,
            bands=['B1', 'B2', 'B3', 'B4']
        )

        if tile is None:
            print("❌ download returned None - fallback likely triggered")
            return

        # tile contains raw values (not /10000)
        import numpy as np
        hr = np.array(tile, dtype=np.float32)
        print(f"HR tile shape: {hr.shape}")
        print(f"HR min/max: {hr.min():.2f}/{hr.max():.2f}")

        # Check test fallback pattern
        first_pixel = hr[0,0,:]
        test_vals = np.array([300,500,700,600], dtype=np.float32)
        if np.allclose(first_pixel, test_vals, atol=30):
            print("❌ HR matches test fallback values")
        else:
            print("✅ HR looks like real Planet data")

    except Exception as e:
        print(f"Exception during download test: {e}")

if __name__ == '__main__':
    main()
