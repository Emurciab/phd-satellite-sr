#!/usr/bin/env python3
"""
Test script for raw download caching optimization.

This tests that:
1. Raw downloads are cached to disk
2. Cache hits are faster than cache misses
3. Cached data is identical to fresh downloads
4. Cache corruption is handled gracefully
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import numpy as np
import ee
import shutil
from pathlib import Path

# Initialize GEE
credentials = ee.ServiceAccountCredentials(
    'stas-50@remotesensingeduart.iam.gserviceaccount.com',
    '/media/eamurciab/TOSHIBA_EXT1/PhD/SR/Auth/remotesensingeduart-222d07dd6689.json'
)
ee.Initialize(credentials)
print("✓ GEE initialized")

from Auth.AuthGEE import find_pairs, PLANET_COLLECTION_ID, regions
from scripts.preprocessing.generate_sta2_grid_parallel import (
    download_tile_region,
    download_tile_region_cached
)


def test_cache_basic():
    """Test basic cache functionality: miss → hit → verify data"""
    print("\n" + "="*60)
    print("TEST 1: Basic Cache Functionality")
    print("="*60)
    
    # Setup
    cache_dir = Path("./test_cache_raw")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    
    # Get a matched pair
    region_config = regions[0]  # STA-2
    pair_results = find_pairs(
        region_config['aoi'],
        '2024-01-01',
        '2024-06-01',
        PLANET_COLLECTION_ID
    )
    
    matched_pairs = pair_results.get('matched_pairs', [])
    if len(matched_pairs) == 0:
        print("❌ No matched pairs found")
        return False
    
    pair = matched_pairs[0]
    planet_collection = pair_results['planet_collection']
    s2_collection = pair_results['s2_collection']
    
    print(f"   Using pair: Planet {pair['planet_date']}, S2 {pair['s2_date']}")
    
    # Get images
    planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
    s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
    
    # Define tile
    bounds = region_config['aoi'].bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    center_lon = (min(lons) + max(lons)) / 2
    center_lat = (min(lats) + max(lats)) / 2
    tile_size = 0.007
    tile_bounds = [center_lon, center_lat, center_lon + tile_size, center_lat + tile_size]
    
    planet_bands = ['B1', 'B2', 'B3', 'B4']
    sentinel_bands = ['B2', 'B3', 'B4', 'B8A']
    
    # Test 1: Cache MISS (first download)
    print("\n   1a. Testing cache MISS (first download)...")
    t0 = time.time()
    hr1, lr1, was_cached1 = download_tile_region_cached(
        planet_img, s2_img, tile_bounds, planet_bands, sentinel_bands,
        planet_id=pair["planet_id"], s2_id=pair["s2_id"], cache_dir=cache_dir
    )
    time_miss = time.time() - t0
    
    if hr1 is None or lr1 is None:
        print("   ❌ Download failed")
        return False
    
    print(f"      Time: {time_miss:.1f}s")
    print(f"      Was cached: {was_cached1}")
    print(f"      HR shape: {hr1.shape}, LR shape: {lr1.shape}")
    
    if was_cached1:
        print("   ❌ First download should be cache MISS")
        return False
    
    # Test 2: Cache HIT (second download)
    print("\n   1b. Testing cache HIT (second download)...")
    t0 = time.time()
    hr2, lr2, was_cached2 = download_tile_region_cached(
        planet_img, s2_img, tile_bounds, planet_bands, sentinel_bands,
        planet_id=pair["planet_id"], s2_id=pair["s2_id"], cache_dir=cache_dir
    )
    time_hit = time.time() - t0
    
    print(f"      Time: {time_hit:.2f}s")
    print(f"      Was cached: {was_cached2}")
    
    if not was_cached2:
        print("   ❌ Second download should be cache HIT")
        return False
    
    # Verify data is identical
    if not np.allclose(hr1, hr2):
        print("   ❌ HR data mismatch between cache miss and hit")
        return False
    
    if not np.allclose(lr1, lr2):
        print("   ❌ LR data mismatch between cache miss and hit")
        return False
    
    print(f"\n   ✅ Cache working correctly!")
    print(f"      Cache miss: {time_miss:.1f}s")
    print(f"      Cache hit:  {time_hit:.2f}s")
    print(f"      Speedup:    {time_miss/time_hit:.1f}x faster")
    
    # Cleanup
    shutil.rmtree(cache_dir)
    return True


def test_cache_corruption():
    """Test that corrupted cache files are handled gracefully"""
    print("\n" + "="*60)
    print("TEST 2: Cache Corruption Handling")
    print("="*60)
    
    # Setup
    cache_dir = Path("./test_cache_corrupt")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    
    # Get a matched pair
    region_config = regions[0]
    pair_results = find_pairs(
        region_config['aoi'],
        '2024-01-01',
        '2024-06-01',
        PLANET_COLLECTION_ID
    )
    
    matched_pairs = pair_results.get('matched_pairs', [])
    if len(matched_pairs) == 0:
        print("❌ No matched pairs found")
        return False
    
    pair = matched_pairs[0]
    planet_collection = pair_results['planet_collection']
    s2_collection = pair_results['s2_collection']
    
    planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
    s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
    
    bounds = region_config['aoi'].bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    center_lon = (min(lons) + max(lons)) / 2
    center_lat = (min(lats) + max(lats)) / 2
    tile_size = 0.007
    tile_bounds = [center_lon, center_lat, center_lon + tile_size, center_lat + tile_size]
    
    planet_bands = ['B1', 'B2', 'B3', 'B4']
    sentinel_bands = ['B2', 'B3', 'B4', 'B8A']
    
    # First, create a valid cache entry
    print("\n   2a. Creating valid cache entry...")
    hr1, lr1, _ = download_tile_region_cached(
        planet_img, s2_img, tile_bounds, planet_bands, sentinel_bands,
        planet_id=pair["planet_id"], s2_id=pair["s2_id"], cache_dir=cache_dir
    )
    
    if hr1 is None:
        print("   ❌ Initial download failed")
        return False
    
    # Find and corrupt the cache file
    print("   2b. Corrupting cache file...")
    cache_files = list((cache_dir / "raw_downloads").glob("*.npz"))
    if len(cache_files) == 0:
        print("   ❌ No cache files found")
        return False
    
    cache_file = cache_files[0]
    with open(cache_file, 'wb') as f:
        f.write(b"corrupted data")
    
    # Try to load from corrupted cache
    print("   2c. Loading from corrupted cache (should re-download)...")
    t0 = time.time()
    hr2, lr2, was_cached = download_tile_region_cached(
        planet_img, s2_img, tile_bounds, planet_bands, sentinel_bands,
        planet_id=pair["planet_id"], s2_id=pair["s2_id"], cache_dir=cache_dir
    )
    time_redownload = time.time() - t0
    
    if hr2 is None or lr2 is None:
        print("   ❌ Re-download after corruption failed")
        return False
    
    if was_cached:
        print("   ❌ Should have re-downloaded, not used corrupted cache")
        return False
    
    print(f"      Re-download time: {time_redownload:.1f}s")
    print(f"      HR shape: {hr2.shape}, LR shape: {lr2.shape}")
    
    # Verify new cache is valid
    print("   2d. Verifying new cache is valid...")
    hr3, lr3, was_cached3 = download_tile_region_cached(
        planet_img, s2_img, tile_bounds, planet_bands, sentinel_bands,
        planet_id=pair["planet_id"], s2_id=pair["s2_id"], cache_dir=cache_dir
    )
    
    if not was_cached3:
        print("   ❌ New cache should be valid")
        return False
    
    if not np.allclose(hr2, hr3):
        print("   ❌ Data mismatch after re-cache")
        return False
    
    print("   ✅ Corruption handled correctly - re-downloaded and re-cached")
    
    # Cleanup
    shutil.rmtree(cache_dir)
    return True


def test_different_bands():
    """Test that different band configurations create different cache entries"""
    print("\n" + "="*60)
    print("TEST 3: Different Band Configurations")
    print("="*60)
    
    # Setup
    cache_dir = Path("./test_cache_bands")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    
    # Get a matched pair
    region_config = regions[0]
    pair_results = find_pairs(
        region_config['aoi'],
        '2024-01-01',
        '2024-06-01',
        PLANET_COLLECTION_ID
    )
    
    matched_pairs = pair_results.get('matched_pairs', [])
    if len(matched_pairs) == 0:
        print("❌ No matched pairs found")
        return False
    
    pair = matched_pairs[0]
    planet_collection = pair_results['planet_collection']
    s2_collection = pair_results['s2_collection']
    
    planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
    s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
    
    bounds = region_config['aoi'].bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    center_lon = (min(lons) + max(lons)) / 2
    center_lat = (min(lats) + max(lats)) / 2
    tile_size = 0.007
    tile_bounds = [center_lon, center_lat, center_lon + tile_size, center_lat + tile_size]
    
    # Config 1: 4-band
    planet_bands_4 = ['B1', 'B2', 'B3', 'B4']
    sentinel_bands_4 = ['B2', 'B3', 'B4', 'B8A']
    
    print("\n   3a. Downloading with 4-band config...")
    hr4, lr4, _ = download_tile_region_cached(
        planet_img, s2_img, tile_bounds, planet_bands_4, sentinel_bands_4,
        planet_id=pair["planet_id"], s2_id=pair["s2_id"], cache_dir=cache_dir
    )
    
    if hr4 is None:
        print("   ❌ 4-band download failed")
        return False
    
    print(f"      HR shape: {hr4.shape} (should have 4 bands)")
    
    # Config 2: 3-band (subset)
    planet_bands_3 = ['B1', 'B2', 'B3']
    sentinel_bands_3 = ['B2', 'B3', 'B4']
    
    print("\n   3b. Downloading with 3-band config...")
    hr3, lr3, was_cached = download_tile_region_cached(
        planet_img, s2_img, tile_bounds, planet_bands_3, sentinel_bands_3,
        planet_id=pair["planet_id"], s2_id=pair["s2_id"], cache_dir=cache_dir
    )
    
    if hr3 is None:
        print("   ❌ 3-band download failed")
        return False
    
    print(f"      HR shape: {hr3.shape} (should have 3 bands)")
    print(f"      Was cached: {was_cached}")
    
    if was_cached:
        print("   ❌ Different band config should create new cache entry")
        return False
    
    # Count cache files
    cache_files = list((cache_dir / "raw_downloads").glob("*.npz"))
    print(f"\n   Cache files created: {len(cache_files)}")
    
    if len(cache_files) != 2:
        print("   ❌ Should have 2 separate cache files for different band configs")
        return False
    
    print("   ✅ Different band configs create separate cache entries")
    
    # Cleanup
    shutil.rmtree(cache_dir)
    return True


def main():
    print("="*60)
    print("🧪 RAW DOWNLOAD CACHE TESTS")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Basic Cache Functionality", test_cache_basic()))
    results.append(("Cache Corruption Handling", test_cache_corruption()))
    results.append(("Different Band Configurations", test_different_bands()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n   Total: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
