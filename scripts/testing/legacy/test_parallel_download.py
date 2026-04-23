#!/usr/bin/env python3
"""
Test parallel download optimization.
Compares sequential vs parallel download times for 3 frames.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize GEE first
print("Initializing GEE...")
import ee
from Auth.AuthGEE import find_pairs, PLANET_COLLECTION_ID

service_account = 'remotesensingeduart@remotesensingeduart.iam.gserviceaccount.com'
credentials_path = Path(__file__).parent.parent.parent / 'Auth' / 'remotesensingeduart-222d07dd6689.json'
credentials = ee.ServiceAccountCredentials(service_account, str(credentials_path))
ee.Initialize(credentials)
print("GEE initialized!")


def download_planet_tile(planet_img, tile_bounds, planet_bands=['B1', 'B2', 'B3', 'B4'], hr_scale=3):
    """Download Planet tile."""
    t0 = time.time()
    tile_geom = ee.Geometry.Rectangle(tile_bounds)
    hr_selected = planet_img.select(planet_bands).clip(tile_geom).unmask(0)
    hr_resampled = hr_selected.reproject(crs='EPSG:4326', scale=hr_scale)
    hr_rect = hr_resampled.sampleRectangle(region=tile_geom, defaultValue=0)
    hr_info = hr_rect.getInfo()
    
    if not hr_info or 'properties' not in hr_info:
        return None, time.time() - t0
    
    hr_props = hr_info['properties']
    hr_arrays = [np.array(hr_props[b], dtype=np.float32) for b in planet_bands]
    hr_array = np.stack(hr_arrays, axis=0)
    hr_array = np.transpose(hr_array, (1, 2, 0))
    return hr_array, time.time() - t0


def download_s2_tile(s2_img, tile_bounds, sentinel_bands=['B2', 'B3', 'B4', 'B8A'], lr_scale=10):
    """Download Sentinel-2 tile."""
    t0 = time.time()
    tile_geom = ee.Geometry.Rectangle(tile_bounds)
    lr_selected = s2_img.select(sentinel_bands).clip(tile_geom).unmask(0)
    lr_resampled = lr_selected.reproject(crs='EPSG:4326', scale=lr_scale)
    lr_rect = lr_resampled.sampleRectangle(region=tile_geom, defaultValue=0)
    lr_info = lr_rect.getInfo()
    
    if not lr_info or 'properties' not in lr_info:
        return None, time.time() - t0
    
    lr_props = lr_info['properties']
    lr_arrays = [np.array(lr_props[b], dtype=np.float32) for b in sentinel_bands]
    lr_array = np.stack(lr_arrays, axis=0)
    lr_array = np.transpose(lr_array, (1, 2, 0))
    return lr_array, time.time() - t0


def test_sequential_vs_parallel():
    """Test sequential vs parallel download speed."""
    
    print("\n" + "="*70)
    print("🧪 PARALLEL DOWNLOAD OPTIMIZATION TEST")
    print("="*70)
    
    # Get test data - 3 matched pairs
    aoi = ee.Geometry.Polygon([[
        [-80.563, 26.306], [-80.512, 26.306],
        [-80.512, 26.368], [-80.563, 26.368], [-80.563, 26.306]
    ]])
    
    print("\n📡 Finding matched image pairs...")
    result = find_pairs(aoi, "2024-01-01", "2024-06-01", PLANET_COLLECTION_ID)
    pairs = result['matched_pairs']
    
    if len(pairs) < 3:
        print(f"   Only found {len(pairs)} pairs, need 3")
        return False
    
    pairs = pairs[:3]
    print(f"   Using first {len(pairs)} pairs")
    
    # Load collections from result
    planet_collection = result['planet_collection']
    s2_collection = result['s2_collection']
    
    # Test tile bounds (small tile)
    tile_bounds = [-80.540, 26.342, -80.533, 26.349]  # 0.007° tile
    
    print(f"\n📍 Test tile: {tile_bounds}")
    print(f"   Testing {len(pairs)} frames × 2 sources = {len(pairs)*2} downloads")
    
    # =========================================================================
    # TEST 1: SEQUENTIAL DOWNLOADS
    # =========================================================================
    print("\n" + "-"*70)
    print("📥 TEST 1: Sequential downloads (baseline)")
    print("-"*70)
    
    t0_seq = time.time()
    seq_times = []
    
    for i, pair in enumerate(pairs):
        planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
        s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
        
        hr, hr_time = download_planet_tile(planet_img, tile_bounds)
        lr, lr_time = download_s2_tile(s2_img, tile_bounds)
        
        seq_times.append({'planet': hr_time, 's2': lr_time})
        print(f"   Frame {i+1}: Planet={hr_time:.2f}s, S2={lr_time:.2f}s")
        
        if hr is None or lr is None:
            print(f"   ❌ Download failed for frame {i+1}")
            return False
    
    total_seq = time.time() - t0_seq
    print(f"\n   Total sequential time: {total_seq:.2f}s")
    
    # =========================================================================
    # TEST 2: PARALLEL DOWNLOADS (within frame)
    # =========================================================================
    print("\n" + "-"*70)
    print("📥 TEST 2: Parallel Planet+S2 per frame")
    print("-"*70)
    
    t0_par1 = time.time()
    
    for i, pair in enumerate(pairs):
        planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
        s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
        
        t0_frame = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_planet = executor.submit(download_planet_tile, planet_img, tile_bounds)
            future_s2 = executor.submit(download_s2_tile, s2_img, tile_bounds)
            
            hr, hr_time = future_planet.result()
            lr, lr_time = future_s2.result()
        
        frame_time = time.time() - t0_frame
        print(f"   Frame {i+1}: {frame_time:.2f}s (Planet={hr_time:.2f}s, S2={lr_time:.2f}s)")
    
    total_par1 = time.time() - t0_par1
    print(f"\n   Total parallel (per-frame): {total_par1:.2f}s")
    print(f"   Speedup: {total_seq/total_par1:.2f}x")
    
    # =========================================================================
    # TEST 3: FULLY PARALLEL (all frames + all sources)
    # =========================================================================
    print("\n" + "-"*70)
    print("📥 TEST 3: Fully parallel (all 6 downloads at once)")
    print("-"*70)
    
    t0_par2 = time.time()
    
    # Prepare all download tasks
    tasks = []
    for i, pair in enumerate(pairs):
        planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
        s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
        tasks.append(('planet', i, planet_img, tile_bounds))
        tasks.append(('s2', i, s2_img, tile_bounds))
    
    def do_download(task):
        source, idx, img, bounds = task
        t0 = time.time()
        if source == 'planet':
            arr, _ = download_planet_tile(img, bounds)
        else:
            arr, _ = download_s2_tile(img, bounds)
        return (source, idx, arr, time.time() - t0)
    
    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(do_download, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    
    total_par2 = time.time() - t0_par2
    
    # Print individual times
    for source, idx, arr, t in sorted(results, key=lambda x: (x[1], x[0])):
        status = "✓" if arr is not None else "✗"
        print(f"   Frame {idx+1} {source}: {t:.2f}s {status}")
    
    print(f"\n   Total fully parallel: {total_par2:.2f}s")
    print(f"   Speedup vs sequential: {total_seq/total_par2:.2f}x")
    print(f"   Speedup vs per-frame parallel: {total_par1/total_par2:.2f}x")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"""
   Sequential:              {total_seq:.2f}s  (baseline)
   Parallel (per-frame):    {total_par1:.2f}s  ({total_seq/total_par1:.2f}x faster)
   Fully parallel:          {total_par2:.2f}s  ({total_seq/total_par2:.2f}x faster)
   
   🎯 For 3-frame samples:
   - Sequential: ~{total_seq:.0f}s per sample download
   - Parallel:   ~{total_par2:.0f}s per sample download
   
   With coregistration (~0.3s) + harmonization (~12s):
   - Old total: ~{total_seq + 12.3:.0f}s per sample
   - New total: ~{total_par2 + 12.3:.0f}s per sample
   - Improvement: {(total_seq - total_par2):.0f}s faster per sample
""")
    
    return True


if __name__ == "__main__":
    success = test_sequential_vs_parallel()
    sys.exit(0 if success else 1)
