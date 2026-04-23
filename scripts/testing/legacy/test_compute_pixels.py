#!/usr/bin/env python3
"""
Test script to compare ee.data.computePixels() vs sampleRectangle().

This tests:
1. If computePixels() works with our use case
2. Speed comparison between the two methods
3. Data quality comparison
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import numpy as np
import ee
from pathlib import Path

# Initialize GEE
credentials = ee.ServiceAccountCredentials(
    'stas-50@remotesensingeduart.iam.gserviceaccount.com',
    '/media/eamurciab/TOSHIBA_EXT1/PhD/SR/Auth/remotesensingeduart-222d07dd6689.json'
)
ee.Initialize(credentials)
print("✓ GEE initialized")

from Auth.AuthGEE import find_pairs, PLANET_COLLECTION_ID, regions


def download_with_sample_rectangle(img, tile_geom, bands, scale):
    """Download using sampleRectangle (current method)"""
    selected = img.select(bands).clip(tile_geom).unmask(0)
    resampled = selected.reproject(crs='EPSG:4326', scale=scale)
    rect = resampled.sampleRectangle(region=tile_geom, defaultValue=0)
    info = rect.getInfo()
    
    if not info or 'properties' not in info:
        return None
    
    props = info['properties']
    arrays = [np.array(props[b], dtype=np.float32) for b in bands]
    array = np.stack(arrays, axis=0)
    array = np.transpose(array, (1, 2, 0))
    return array


def download_with_compute_pixels(img, tile_geom, bands, scale):
    """Download using computePixels (alternative method)"""
    try:
        selected = img.select(bands).clip(tile_geom).unmask(0)
        resampled = selected.reproject(crs='EPSG:4326', scale=scale)
        
        # Get bounds for computePixels
        bounds = tile_geom.bounds().getInfo()['coordinates'][0]
        lons = [p[0] for p in bounds]
        lats = [p[1] for p in bounds]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Calculate dimensions
        width_deg = max_lon - min_lon
        height_deg = max_lat - min_lat
        # Scale is in meters, convert to approximate pixels
        # 1 degree ≈ 111km at equator
        width_px = int(width_deg * 111000 / scale)
        height_px = int(height_deg * 111000 / scale)
        
        # Use computePixels with NUMPY format
        pixels = ee.data.computePixels({
            'expression': resampled,
            'fileFormat': 'NUMPY_NDARRAY',
            'bandIds': bands,
            'grid': {
                'dimensions': {
                    'width': width_px,
                    'height': height_px
                },
                'affineTransform': {
                    'scaleX': scale / 111000,  # degrees per pixel
                    'shearX': 0,
                    'translateX': min_lon,
                    'shearY': 0,
                    'scaleY': -scale / 111000,  # negative for top-to-bottom
                    'translateY': max_lat
                },
                'crsCode': 'EPSG:4326'
            }
        })
        
        # Convert structured array to regular array (H, W, C)
        # computePixels returns structured array with named fields for each band
        raw_array = np.array(pixels)
        
        # Extract bands from structured array
        band_arrays = []
        for band in bands:
            band_data = raw_array[band].astype(np.float32)
            band_arrays.append(band_data)
        
        # Stack into (H, W, C) format
        array = np.stack(band_arrays, axis=-1)
        return array
        
    except Exception as e:
        print(f"      computePixels error: {e}")
        return None


def download_with_get_pixels(img, tile_geom, bands, scale):
    """Download using getPixels REST API (another alternative)"""
    try:
        selected = img.select(bands).clip(tile_geom).unmask(0)
        resampled = selected.reproject(crs='EPSG:4326', scale=scale)
        
        # Get bounds
        bounds = tile_geom.bounds().getInfo()['coordinates'][0]
        lons = [p[0] for p in bounds]
        lats = [p[1] for p in bounds]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Calculate dimensions
        width_deg = max_lon - min_lon
        height_deg = max_lat - min_lat
        width_px = int(width_deg * 111000 / scale)
        height_px = int(height_deg * 111000 / scale)
        
        # Try getPixels
        result = ee.data.getPixels({
            'assetId': resampled.serialize(),
            'fileFormat': 'NPY',
            'bandIds': bands,
            'grid': {
                'dimensions': {
                    'width': width_px,
                    'height': height_px
                },
                'affineTransform': {
                    'scaleX': scale / 111000,
                    'shearX': 0,
                    'translateX': min_lon,
                    'shearY': 0,
                    'scaleY': -scale / 111000,
                    'translateY': max_lat
                },
                'crsCode': 'EPSG:4326'
            }
        })
        
        return np.load(result)
        
    except Exception as e:
        print(f"      getPixels error: {e}")
        return None


def test_download_methods():
    """Compare different download methods"""
    print("\n" + "="*70)
    print("🧪 DOWNLOAD METHOD COMPARISON")
    print("="*70)
    
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
        return
    
    pair = matched_pairs[0]
    planet_collection = pair_results['planet_collection']
    
    print(f"\n   Using pair: Planet {pair['planet_date']}")
    
    # Get Planet image
    planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
    
    # Define tile
    bounds = region_config['aoi'].bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    center_lon = (min(lons) + max(lons)) / 2
    center_lat = (min(lats) + max(lats)) / 2
    tile_size = 0.007
    tile_bounds = [center_lon, center_lat, center_lon + tile_size, center_lat + tile_size]
    tile_geom = ee.Geometry.Rectangle(tile_bounds)
    
    bands = ['B1', 'B2', 'B3', 'B4']
    scale = 3  # Planet 3m resolution
    
    print(f"   Tile bounds: {tile_bounds}")
    print(f"   Bands: {bands}")
    print(f"   Scale: {scale}m")
    
    # Method 1: sampleRectangle (current)
    print("\n" + "-"*50)
    print("   Method 1: sampleRectangle (current)")
    print("-"*50)
    
    times_sr = []
    for i in range(3):
        t0 = time.time()
        arr_sr = download_with_sample_rectangle(planet_img, tile_geom, bands, scale)
        times_sr.append(time.time() - t0)
        print(f"      Run {i+1}: {times_sr[-1]:.2f}s")
    
    if arr_sr is not None:
        print(f"      Shape: {arr_sr.shape}")
        print(f"      Range: [{arr_sr.min():.1f}, {arr_sr.max():.1f}]")
        print(f"      Avg time: {np.mean(times_sr):.2f}s ± {np.std(times_sr):.2f}s")
    
    # Method 2: computePixels
    print("\n" + "-"*50)
    print("   Method 2: computePixels")
    print("-"*50)
    
    times_cp = []
    arr_cp = None
    for i in range(3):
        t0 = time.time()
        arr_cp = download_with_compute_pixels(planet_img, tile_geom, bands, scale)
        times_cp.append(time.time() - t0)
        if arr_cp is not None:
            print(f"      Run {i+1}: {times_cp[-1]:.2f}s")
        else:
            print(f"      Run {i+1}: FAILED")
            break
    
    if arr_cp is not None:
        print(f"      Shape: {arr_cp.shape}")
        print(f"      Range: [{arr_cp.min():.1f}, {arr_cp.max():.1f}]")
        print(f"      Avg time: {np.mean(times_cp):.2f}s ± {np.std(times_cp):.2f}s")
    else:
        print("      ⚠️ computePixels not working with current setup")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    if arr_sr is not None:
        print(f"\n   sampleRectangle:")
        print(f"      Avg time: {np.mean(times_sr):.2f}s")
        print(f"      Status: ✅ Working")
    
    if arr_cp is not None:
        print(f"\n   computePixels:")
        print(f"      Avg time: {np.mean(times_cp):.2f}s")
        speedup = np.mean(times_sr) / np.mean(times_cp)
        print(f"      Speedup vs sampleRectangle: {speedup:.1f}x")
        print(f"      Status: ✅ Working")
    else:
        print(f"\n   computePixels:")
        print(f"      Status: ❌ Not available or error")
        print(f"      Recommendation: Stick with sampleRectangle")
    
    print("\n" + "="*70)
    print("💡 CONCLUSION")
    print("="*70)
    
    if arr_cp is not None and np.mean(times_cp) < np.mean(times_sr):
        print("   computePixels is faster - consider switching")
    else:
        print("   sampleRectangle remains the best option")
        print("   Focus optimization on:")
        print("   1. Raw download caching (implemented ✅)")
        print("   2. Batch tile downloads")
        print("   3. Parallel processing")


if __name__ == "__main__":
    test_download_methods()
