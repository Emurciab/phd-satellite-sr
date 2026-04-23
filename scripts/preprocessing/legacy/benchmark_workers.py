#!/usr/bin/env python3
"""
Benchmark different worker configurations to find optimal performance
Tests coverage checking parallelization limits
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import ee
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from Auth.AuthGEE import find_pairs, PLANET_COLLECTION_ID, regions

# Initialize GEE
try:
    ee.Initialize()
except:
    print("Initializing GEE...")

print("="*80)
print("WORKER CONFIGURATION BENCHMARK")
print("="*80)

# System info
cpu_count = psutil.cpu_count(logical=False)  # Physical cores
cpu_count_logical = psutil.cpu_count(logical=True)  # Logical cores (with hyperthreading)
ram_gb = psutil.virtual_memory().total / (1024**3)

print(f"\n📊 SYSTEM SPECIFICATIONS:")
print(f"   Physical CPU cores: {cpu_count}")
print(f"   Logical CPU cores: {cpu_count_logical}")
print(f"   Total RAM: {ram_gb:.1f} GB")
print(f"   Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

# Get test data
print(f"\n🔍 Getting test data...")
region_config = regions[0]
pair_results = find_pairs(
    region_config["aoi"],
    region_config["start"],
    region_config["end"],
    PLANET_COLLECTION_ID
)

matched_pairs = pair_results.get('matched_pairs', [])
planet_collection = pair_results.get('planet_collection')
s2_collection = pair_results.get('s2_collection')

sequence = matched_pairs[:3]
print(f"   Using sequence: {[p['s2_date'] for p in sequence]}")

# Get images
planet_images = []
s2_images = []
for pair in sequence:
    planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
    s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
    planet_images.append(planet_img)
    s2_images.append(s2_img)

# Generate test tiles (smaller set for quick testing)
from generate_sta2_grid_parallel import generate_tile_grid, has_sufficient_coverage

test_tiles = generate_tile_grid(region_config["aoi"], tile_size_deg=0.007, max_tiles=50)
print(f"   Generated {len(test_tiles)} test tiles")

# Function to test
def check_tile_all_frames(tile_bounds, planet_images, s2_images, min_percent=90):
    """Check if tile has coverage in all frames"""
    tile_geom = ee.Geometry.Rectangle(tile_bounds)
    
    for planet_img, s2_img in zip(planet_images, s2_images):
        planet_valid = has_sufficient_coverage(
            planet_img, tile_geom, 'B1',
            min_percent=min_percent, scale=30
        )
        if not planet_valid:
            return False
        
        s2_valid = has_sufficient_coverage(
            s2_img, tile_geom, 'B2',
            min_percent=min_percent, scale=30
        )
        if not s2_valid:
            return False
    
    return True

# Test configurations
worker_configs = [1, 2, 4, 8, 12, 16, 20, 24, 32]

print(f"\n{'='*80}")
print(f"BENCHMARK RESULTS")
print(f"{'='*80}")
print(f"{'Workers':<10} {'Time(s)':<10} {'Tiles/s':<12} {'RAM(GB)':<10} {'CPU%':<10} {'Status':<15}")
print(f"{'-'*80}")

results = []

for num_workers in worker_configs:
    if num_workers > cpu_count_logical * 2:
        print(f"{num_workers:<10} {'SKIP':<10} {'---':<12} {'---':<10} {'---':<10} Too many workers")
        continue
    
    # Monitor system resources
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024**3)
    cpu_percent_before = psutil.cpu_percent(interval=0.1)
    
    try:
        start_time = time.time()
        valid_count = 0
        errors = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(check_tile_all_frames, tile_bounds, planet_images, s2_images): tile_bounds
                for tile_bounds in test_tiles
            }
            
            for future in as_completed(futures):
                try:
                    if future.result():
                        valid_count += 1
                except Exception as e:
                    errors += 1
        
        elapsed = time.time() - start_time
        tiles_per_sec = len(test_tiles) / elapsed
        
        # Check resources after
        ram_after = process.memory_info().rss / (1024**3)
        ram_used = ram_after - ram_before
        cpu_percent_after = psutil.cpu_percent(interval=0.1)
        cpu_avg = (cpu_percent_before + cpu_percent_after) / 2
        
        status = "✓ OK"
        if errors > 0:
            status = f"⚠️ {errors} errors"
        
        print(f"{num_workers:<10} {elapsed:<10.1f} {tiles_per_sec:<12.2f} {ram_used:<10.2f} {cpu_avg:<10.1f} {status:<15}")
        
        results.append({
            'workers': num_workers,
            'time': elapsed,
            'tiles_per_sec': tiles_per_sec,
            'errors': errors,
            'ram_gb': ram_used,
            'cpu_percent': cpu_avg
        })
        
        # Cool down
        time.sleep(2)
        
    except Exception as e:
        print(f"{num_workers:<10} {'ERROR':<10} {'---':<12} {'---':<10} {'---':<10} {str(e)[:30]}")
        continue

# Find optimal configuration
print(f"\n{'='*80}")
print(f"ANALYSIS")
print(f"{'='*80}")

if results:
    # Find fastest
    fastest = min(results, key=lambda x: x['time'])
    print(f"\n⚡ FASTEST: {fastest['workers']} workers")
    print(f"   Time: {fastest['time']:.1f}s")
    print(f"   Throughput: {fastest['tiles_per_sec']:.2f} tiles/s")
    print(f"   RAM usage: {fastest['ram_gb']:.2f} GB")
    print(f"   CPU usage: {fastest['cpu_percent']:.1f}%")
    
    # Find most efficient (best tiles/worker ratio without errors)
    clean_results = [r for r in results if r['errors'] == 0]
    if clean_results:
        efficiency = [(r['tiles_per_sec'] / r['workers'], r) for r in clean_results]
        most_efficient = max(efficiency, key=lambda x: x[0])[1]
        
        print(f"\n🎯 MOST EFFICIENT: {most_efficient['workers']} workers")
        print(f"   Efficiency: {most_efficient['tiles_per_sec']/most_efficient['workers']:.2f} tiles/s/worker")
        print(f"   Time: {most_efficient['time']:.1f}s")
        print(f"   RAM usage: {most_efficient['ram_gb']:.2f} GB")
        
        # Recommended configuration
        print(f"\n💡 RECOMMENDED CONFIGURATION:")
        
        # Balance between speed and efficiency
        if fastest['workers'] <= cpu_count_logical:
            recommended = fastest['workers']
            reason = "Fastest without exceeding logical cores"
        elif most_efficient['workers'] <= cpu_count:
            recommended = most_efficient['workers']
            reason = "Most efficient within physical cores"
        else:
            recommended = min(cpu_count_logical, fastest['workers'])
            reason = "Balanced for your system"
        
        print(f"   Workers: {recommended}")
        print(f"   Reason: {reason}")
        
        # Calculate time savings for full dataset
        baseline_time = results[0]['time'] if results[0]['workers'] == 1 else None
        if baseline_time and fastest['time'] < baseline_time:
            speedup = baseline_time / fastest['time']
            print(f"\n   Speedup vs 1 worker: {speedup:.1f}x faster")
            
            # Estimate for 1440 samples
            est_time_hours = (1440 * fastest['time']) / (len(test_tiles) * 3600)
            print(f"   Estimated time for 1440 samples: ~{est_time_hours:.1f} hours")
        
        # Safety checks
        print(f"\n⚠️  SAFETY CONSIDERATIONS:")
        if fastest['ram_gb'] > ram_gb * 0.5:
            print(f"   ⚠️ High RAM usage ({fastest['ram_gb']:.1f} GB) - may need to reduce workers")
        else:
            print(f"   ✓ RAM usage acceptable")
        
        if fastest['workers'] > cpu_count_logical:
            print(f"   ⚠️ Workers ({fastest['workers']}) exceed logical cores ({cpu_count_logical})")
        else:
            print(f"   ✓ Workers within logical core count")
        
        # GEE rate limits
        print(f"\n📡 GEE RATE LIMIT CONSIDERATIONS:")
        print(f"   GEE concurrent request limit: ~10-50 per project")
        print(f"   Your workers: {recommended}")
        if recommended > 16:
            print(f"   ⚠️ May hit GEE rate limits with {recommended} workers")
            print(f"   Consider using 12-16 workers for GEE operations")
        else:
            print(f"   ✓ Within safe GEE concurrent request limits")

print(f"\n{'='*80}")
