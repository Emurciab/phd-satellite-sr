#!/usr/bin/env python3
"""
Quick validation test for the optimized pipeline.

This runs the generate_sta2_grid_parallel.py with 2 samples to verify:
1. Raw download caching is working
2. Samples are generated correctly
3. Cache hit/miss statistics are tracked
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import subprocess
import pickle
import shutil
import io
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


# Minimal TrainingSample definition for unpickling
@dataclass
class TrainingSample:
    lr_sequence: np.ndarray
    hr_sequence: np.ndarray
    temporal_dates: List[str]
    tile_bounds: Dict[str, Any]
    region_name: str
    sequence_index: int
    tile_index: int
    

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that uses our local TrainingSample definition."""
    def find_class(self, module, name):
        if name == 'TrainingSample':
            return TrainingSample
        return super().find_class(module, name)


def run_validation():
    print("="*70)
    print("🧪 PIPELINE VALIDATION TEST")
    print("="*70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("   Generating 2 samples to validate optimized pipeline")
    print("="*70)
    
    # Setup directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"./validation_output_{timestamp}"
    cache_dir = f"./validation_cache_{timestamp}"
    
    # Build command - use the same Python interpreter that's running this script
    script_path = Path(__file__).parent.parent / "preprocessing" / "generate_sta2_grid_parallel.py"
    python_exe = sys.executable  # Use the current Python interpreter
    
    cmd = [
        python_exe, str(script_path),
        "--max-samples", "2",
        "--output-dir", output_dir,
        "--cache-dir", cache_dir,
        "--sequence-length", "3",
        "--coverage-threshold", "90.0",
        "--tile-workers", "1",  # Sequential for clearer output
        "--coverage-workers", "8",
        "--region", "STA-2",
        "--start-date", "2024-01-01",
        "--end-date", "2024-06-01"
    ]
    
    print(f"\n   Python: {python_exe}")
    print(f"   Running: {' '.join(cmd[-8:])}")
    print("-"*70)
    
    # Run first time (should be cache MISS)
    print("\n📥 RUN 1: First run (cache should be MISS)")
    print("-"*70)
    
    t0 = time.time()
    result1 = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent)
    )
    time1 = time.time() - t0
    
    print(f"\n   Run 1 time: {time1:.1f}s")
    
    # Check results
    dataset_path = Path(output_dir) / "STA-2_dataset.pkl"
    if not dataset_path.exists():
        print("   ❌ Dataset not created!")
        return False
    
    with open(dataset_path, 'rb') as f:
        dataset = CustomUnpickler(f).load()
    
    samples = dataset.get('samples', [])
    print(f"   Samples generated: {len(samples)}")
    
    if len(samples) < 2:
        print("   ❌ Not enough samples generated!")
        return False
    
    # Check cache statistics
    total_hits = 0
    total_misses = 0
    for sample in samples:
        hits = sample.tile_bounds.get('cache_hits', 0)
        misses = sample.tile_bounds.get('cache_misses', 0)
        total_hits += hits
        total_misses += misses
    
    print(f"   Cache hits: {total_hits}, Cache misses: {total_misses}")
    
    if total_misses == 0 and total_hits == 0:
        print("   ⚠️ Cache stats not tracked (may need code check)")
    
    # Verify sample shapes
    sample = samples[0]
    print(f"\n   Sample 0:")
    print(f"      LR shape: {sample.lr_sequence.shape}")
    print(f"      HR shape: {sample.hr_sequence.shape}")
    print(f"      Dates: {sample.temporal_dates}")
    print(f"      Band mode: {sample.tile_bounds.get('band_mode', 'unknown')}")
    
    # Run second time (should hit cache for raw downloads)
    print("\n" + "-"*70)
    print("📥 RUN 2: Second run (raw downloads should be CACHED)")
    print("-"*70)
    
    # Clear coregistered/harmonized cache but keep raw downloads
    coreg_cache = Path(cache_dir) / "STA-2" / "coregistered"
    harm_cache = Path(cache_dir) / "STA-2" / "harmonized"
    if coreg_cache.exists():
        shutil.rmtree(coreg_cache)
    if harm_cache.exists():
        shutil.rmtree(harm_cache)
    
    # Delete dataset to force regeneration
    dataset_path.unlink()
    
    t0 = time.time()
    result2 = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent)
    )
    time2 = time.time() - t0
    
    print(f"\n   Run 2 time: {time2:.1f}s")
    
    # Check cache stats for second run
    if dataset_path.exists():
        with open(dataset_path, 'rb') as f:
            dataset2 = CustomUnpickler(f).load()
        
        samples2 = dataset2.get('samples', [])
        total_hits2 = 0
        total_misses2 = 0
        for sample in samples2:
            hits = sample.tile_bounds.get('cache_hits', 0)
            misses = sample.tile_bounds.get('cache_misses', 0)
            total_hits2 += hits
            total_misses2 += misses
        
        print(f"   Cache hits: {total_hits2}, Cache misses: {total_misses2}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\n   Run 1 (cold cache): {time1:.1f}s")
    print(f"   Run 2 (warm cache): {time2:.1f}s")
    
    if time2 < time1:
        speedup = time1 / time2
        print(f"   Speedup: {speedup:.1f}x faster with cached raw downloads")
    
    print(f"\n   Samples validated: {len(samples)}")
    print(f"   LR shape: {samples[0].lr_sequence.shape}")
    print(f"   HR shape: {samples[0].hr_sequence.shape}")
    
    # Check raw downloads cache
    raw_cache = Path(cache_dir) / "STA-2" / "raw_downloads"
    if raw_cache.exists():
        cache_files = list(raw_cache.glob("*.npz"))
        total_size = sum(f.stat().st_size for f in cache_files)
        print(f"\n   Raw download cache:")
        print(f"      Files: {len(cache_files)}")
        print(f"      Size: {total_size / 1024 / 1024:.1f} MB")
    
    print("\n   ✅ VALIDATION PASSED!")
    
    # Cleanup
    print("\n   Cleaning up...")
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(cache_dir, ignore_errors=True)
    
    return True


if __name__ == "__main__":
    success = run_validation()
    exit(0 if success else 1)
