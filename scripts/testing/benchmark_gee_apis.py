#!/usr/bin/env python3
"""
Benchmark GEE Download APIs: sampleRectangle vs computePixels

This script compares the performance of two GEE pixel retrieval methods:
1. sampleRectangle() - Current implementation (JSON-based, limited)
2. ee.data.computePixels() - Modern API (binary, scalable)

Key differences:
- sampleRectangle: JSON serialization overhead, pixel limits, slower
- computePixels: Binary format (GeoTIFF/NPY), no JSON overhead, much faster

Run: conda run -n tf_gpu python scripts/testing/benchmark_gee_apis.py
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Initialize GEE
from Auth.AuthGEE import ee, find_pairs, PLANET_COLLECTION_ID, regions

# Fallback collection
PLANET_COLLECTION_ID_FALLBACK = 'projects/ee-eeamurciabb/assets/testSTAs'


def get_test_images():
    """Get a test Planet/S2 image pair from STA-2 2020"""
    region_config = next(r for r in regions if r['name'] == 'STA-2')
    
    print("🔍 Finding test image pair...")
    pair_results = find_pairs(
        region_config["aoi"],
        "2020-01-01",
        "2020-03-01",  # Short range for quick test
        PLANET_COLLECTION_ID,
        planet_coll_id_fallback=PLANET_COLLECTION_ID_FALLBACK
    )
    
    matched_pairs = pair_results.get('matched_pairs', [])
    s2_collection = pair_results.get('s2_collection')
    planet_collection = pair_results.get('planet_collection')
    
    if not matched_pairs:
        raise ValueError("No matched pairs found!")
    
    # Get first pair
    pair = matched_pairs[0]
    planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
    s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
    
    print(f"   ✅ Found pair: Planet {pair['planet_date']} / S2 {pair['s2_date']}")
    
    # Get AOI bounds for tile generation
    bounds = region_config["aoi"].bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    # Create test tile in center of AOI
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    tile_size = 0.007  # ~768m at 3m resolution
    
    tile_bounds = [
        center_lon - tile_size/2,
        center_lat - tile_size/2,
        center_lon + tile_size/2,
        center_lat + tile_size/2
    ]
    
    return planet_img, s2_img, tile_bounds, pair


def benchmark_sample_rectangle(planet_img, tile_bounds, bands, scale=3, iterations=5):
    """
    Benchmark current sampleRectangle implementation.
    
    This is the method currently used in generate_sta2_grid_parallel.py
    """
    print(f"\n📊 Benchmarking sampleRectangle() - {iterations} iterations")
    print("─" * 60)
    
    tile_geom = ee.Geometry.Rectangle(tile_bounds)
    times = []
    
    for i in range(iterations):
        start = time.time()
        
        # Current implementation
        selected = planet_img.select(bands).clip(tile_geom).unmask(0)
        resampled = selected.reproject(crs='EPSG:4326', scale=scale)
        rect = resampled.sampleRectangle(region=tile_geom, defaultValue=0)
        info = rect.getInfo()
        
        if info and 'properties' in info:
            props = info['properties']
            arrays = [np.array(props[b], dtype=np.float32) for b in bands]
            result = np.stack(arrays, axis=-1)  # (H, W, C)
            
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Iteration {i+1}: {elapsed:.2f}s - shape: {result.shape}")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"\n   ⏱️ Average: {avg_time:.2f}s ± {std_time:.2f}s")
    
    return avg_time, result


def benchmark_compute_pixels(planet_img, tile_bounds, bands, scale=3, iterations=5):
    """
    Benchmark modern computePixels API.
    
    This is the recommended approach for high-performance pixel retrieval.
    Advantages:
    - Binary format (no JSON overhead)
    - Handles larger requests
    - Direct NumPy array output
    """
    print(f"\n📊 Benchmarking computePixels() - {iterations} iterations")
    print("─" * 60)
    
    tile_geom = ee.Geometry.Rectangle(tile_bounds)
    times = []
    result = None
    
    for i in range(iterations):
        start = time.time()
        
        try:
            # Prepare image
            selected = planet_img.select(bands).clip(tile_geom).unmask(0)
            
            # Get tile bounds for request
            # computePixels expects specific grid parameters
            request = {
                'expression': selected,
                'fileFormat': 'NUMPY_NDARRAY',
                'grid': {
                    'dimensions': {
                        'width': 256,
                        'height': 256
                    },
                    'affineTransform': {
                        'scaleX': scale / 111320,  # degrees per meter at equator (approximate)
                        'shearX': 0,
                        'translateX': tile_bounds[0],
                        'scaleY': -scale / 111320,
                        'shearY': 0,
                        'translateY': tile_bounds[3]  # Top-left Y
                    },
                    'crsCode': 'EPSG:4326'
                }
            }
            
            result_bytes = ee.data.computePixels(request)
            result = np.load(result_bytes, allow_pickle=True)
            
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"   Iteration {i+1}: {elapsed:.2f}s - shape: {result.shape}")
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"   Iteration {i+1}: FAILED ({elapsed:.2f}s) - {e}")
            times.append(elapsed)
    
    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"\n   ⏱️ Average: {avg_time:.2f}s ± {std_time:.2f}s")
        return avg_time, result
    
    return None, None


def benchmark_compute_pixels_geotiff(planet_img, tile_bounds, bands, scale=3, iterations=5):
    """
    Benchmark computePixels with GeoTIFF format (alternative approach).
    
    GeoTIFF format may be more reliable for some use cases.
    """
    import io
    
    print(f"\n📊 Benchmarking computePixels(GeoTIFF) - {iterations} iterations")
    print("─" * 60)
    
    tile_geom = ee.Geometry.Rectangle(tile_bounds)
    times = []
    result = None
    
    for i in range(iterations):
        start = time.time()
        
        try:
            # Prepare image
            selected = planet_img.select(bands).clip(tile_geom).unmask(0)
            resampled = selected.reproject(crs='EPSG:4326', scale=scale)
            
            # Use getPixels (simpler API, automatic grid)
            request = {
                'expression': resampled,
                'fileFormat': 'GEO_TIFF',
                'bandIds': bands,
                'region': tile_geom.getInfo(),
            }
            
            result_bytes = ee.data.computePixels(request)
            
            # Parse GeoTIFF to numpy
            try:
                import rasterio
                from rasterio.io import MemoryFile
                
                with MemoryFile(result_bytes) as memfile:
                    with memfile.open() as dataset:
                        result = dataset.read()  # (C, H, W)
                        result = np.transpose(result, (1, 2, 0))  # (H, W, C)
            except ImportError:
                # Fallback if rasterio not available
                from PIL import Image
                result = np.array(Image.open(io.BytesIO(result_bytes)))
            
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"   Iteration {i+1}: {elapsed:.2f}s - shape: {result.shape}")
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"   Iteration {i+1}: FAILED ({elapsed:.2f}s) - {str(e)[:80]}")
            times.append(elapsed)
    
    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"\n   ⏱️ Average: {avg_time:.2f}s ± {std_time:.2f}s")
        return avg_time, result
    
    return None, None


def benchmark_batched_download(planet_img, tile_bounds, bands, scale=3, grid_size=4, iterations=3):
    """
    Benchmark batched meta-tile download strategy.
    
    Downloads a larger region (e.g., 4x4 tiles) in one request,
    then splits locally with NumPy slicing.
    
    This reduces GEE round-trips dramatically.
    """
    import io
    
    print(f"\n📊 Benchmarking BATCHED download ({grid_size}x{grid_size} tiles) - {iterations} iterations")
    print("─" * 60)
    
    # Calculate meta-tile bounds (covering grid_size x grid_size tiles)
    tile_width = tile_bounds[2] - tile_bounds[0]
    tile_height = tile_bounds[3] - tile_bounds[1]
    
    meta_bounds = [
        tile_bounds[0],  # Start from original tile
        tile_bounds[1],
        tile_bounds[0] + tile_width * grid_size,
        tile_bounds[1] + tile_height * grid_size
    ]
    
    meta_geom = ee.Geometry.Rectangle(meta_bounds)
    times = []
    individual_tiles = []
    
    for i in range(iterations):
        start = time.time()
        
        try:
            # Prepare meta-tile image
            selected = planet_img.select(bands).clip(meta_geom).unmask(0)
            resampled = selected.reproject(crs='EPSG:4326', scale=scale)
            
            # Download entire meta-tile
            request = {
                'expression': resampled,
                'fileFormat': 'GEO_TIFF',
                'bandIds': bands,
                'region': meta_geom.getInfo(),
            }
            
            result_bytes = ee.data.computePixels(request)
            
            # Parse to numpy
            try:
                import rasterio
                from rasterio.io import MemoryFile
                
                with MemoryFile(result_bytes) as memfile:
                    with memfile.open() as dataset:
                        meta_array = dataset.read()  # (C, H, W)
                        meta_array = np.transpose(meta_array, (1, 2, 0))  # (H, W, C)
            except ImportError:
                from PIL import Image
                meta_array = np.array(Image.open(io.BytesIO(result_bytes)))
            
            download_time = time.time() - start
            
            # Split into individual tiles locally (this is basically free)
            split_start = time.time()
            tile_h = meta_array.shape[0] // grid_size
            tile_w = meta_array.shape[1] // grid_size
            
            tiles = []
            for row in range(grid_size):
                for col in range(grid_size):
                    tile = meta_array[
                        row * tile_h:(row + 1) * tile_h,
                        col * tile_w:(col + 1) * tile_w,
                        :
                    ]
                    tiles.append(tile)
            
            split_time = time.time() - split_start
            
            elapsed = time.time() - start
            times.append(elapsed)
            individual_tiles = tiles
            
            print(f"   Iteration {i+1}: {elapsed:.2f}s total (download: {download_time:.2f}s, split: {split_time:.4f}s)")
            print(f"      Meta-tile: {meta_array.shape} → {len(tiles)} tiles of {tiles[0].shape}")
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"   Iteration {i+1}: FAILED ({elapsed:.2f}s) - {str(e)[:80]}")
            times.append(elapsed)
    
    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        time_per_tile = avg_time / (grid_size * grid_size)
        print(f"\n   ⏱️ Average: {avg_time:.2f}s ± {std_time:.2f}s for {grid_size*grid_size} tiles")
        print(f"   ⏱️ Per-tile: {time_per_tile:.3f}s (vs sequential)")
        return avg_time, individual_tiles
    
    return None, None


def main():
    print("=" * 70)
    print("🔬 GEE API BENCHMARK: sampleRectangle vs computePixels")
    print("=" * 70)
    
    # Get test data
    planet_img, s2_img, tile_bounds, pair = get_test_images()
    
    # Detect available bands
    try:
        available_bands = planet_img.bandNames().getInfo()
        has_8band = all(b in available_bands for b in ['B5', 'B6', 'B7', 'B8'])
        if has_8band:
            test_bands = ['B2', 'B4', 'B6', 'B7', 'B8']  # 8-band SuperDove
        else:
            test_bands = ['B1', 'B2', 'B3', 'B4']  # 4-band
        print(f"\n📷 Using bands: {test_bands}")
    except:
        test_bands = ['B1', 'B2', 'B3', 'B4']
        print(f"\n📷 Defaulting to bands: {test_bands}")
    
    print(f"📍 Tile bounds: {tile_bounds}")
    
    # Run benchmarks
    results = {}
    
    # 1. Current implementation (sampleRectangle)
    time_sr, result_sr = benchmark_sample_rectangle(planet_img, tile_bounds, test_bands, iterations=3)
    results['sampleRectangle'] = time_sr
    
    # 2. Modern API (computePixels with NUMPY format)
    time_cp_npy, result_cp_npy = benchmark_compute_pixels(planet_img, tile_bounds, test_bands, iterations=3)
    results['computePixels_numpy'] = time_cp_npy
    
    # 3. computePixels with GeoTIFF format
    time_cp_tiff, result_cp_tiff = benchmark_compute_pixels_geotiff(planet_img, tile_bounds, test_bands, iterations=3)
    results['computePixels_geotiff'] = time_cp_tiff
    
    # 4. Batched download (4x4 meta-tile)
    time_batch, result_batch = benchmark_batched_download(planet_img, tile_bounds, test_bands, grid_size=4, iterations=3)
    results['batched_4x4'] = time_batch
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 BENCHMARK SUMMARY")
    print("=" * 70)
    
    baseline = results.get('sampleRectangle', 1.0)
    
    for name, time_val in results.items():
        if time_val:
            speedup = baseline / time_val if time_val > 0 else 0
            print(f"  {name:25s}: {time_val:.2f}s ({speedup:.2f}x vs baseline)")
        else:
            print(f"  {name:25s}: FAILED")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    
    valid_results = [(k, v) for k, v in results.items() if v is not None]
    best_method = min(valid_results, key=lambda x: x[1]) if valid_results else ("None", 0)
    print(f"  Best method: {best_method[0]} ({best_method[1]:.2f}s)")
    
    if results.get('batched_4x4'):
        per_tile_batch = results['batched_4x4'] / 16
        per_tile_sr = results.get('sampleRectangle', per_tile_batch * 10)
        print(f"\n  Batched approach: {per_tile_batch:.3f}s per tile")
        print(f"  Sequential approach: {per_tile_sr:.2f}s per tile")
        print(f"  Potential speedup: {per_tile_sr / per_tile_batch:.1f}x")


if __name__ == "__main__":
    main()
