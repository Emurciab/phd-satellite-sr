#!/usr/bin/env python3
"""
Benchmark test: 100 samples with 8 workers
Tests the optimized parallel download pipeline at scale.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import subprocess
from pathlib import Path
from datetime import datetime

def run_benchmark():
    print("="*70)
    print("🧪 BENCHMARK: 100 SAMPLES WITH 8 WORKERS")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./benchmark_100samples_{timestamp}"
    cache_dir = f"./benchmark_cache_{timestamp}"
    
    print(f"\n   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Output: {output_dir}")
    print(f"   Cache: {cache_dir}")
    print(f"   Target: 100 samples")
    print(f"   Workers: 15 coverage, 8 tile")
    print(f"   Region: STA-2 (2024)")
    
    # Build command - maximize parallelism
    # - 15 coverage workers for fast tile validation
    # - 8 tile workers for parallel sample generation
    cmd = [
        sys.executable,
        "scripts/preprocessing/generate_sta2_grid_parallel.py",
        "--output-dir", output_dir,
        "--cache-dir", cache_dir,
        "--max-samples", "100",
        "--coverage-workers", "15",
        "--tile-workers", "8",
        "--region", "STA-2",
        "--start-date", "2024-01-01",
        "--end-date", "2025-01-01"
    ]
    
    print(f"\n   Command: {' '.join(cmd)}")
    print("\n" + "-"*70)
    print("Running pipeline...")
    print("-"*70 + "\n")
    
    t0 = time.time()
    
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).parent.parent.parent),
        capture_output=False
    )
    
    elapsed = time.time() - t0
    
    print("\n" + "="*70)
    print("📊 BENCHMARK RESULTS")
    print("="*70)
    
    print(f"\n   Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Exit code: {result.returncode}")
    
    # Check output
    output_path = Path(output_dir)
    if output_path.exists():
        pkl_files = list(output_path.glob("*.pkl"))
        if pkl_files:
            import pickle
            from dataclasses import dataclass
            from typing import Dict, Any, List
            import numpy as np
            
            # Minimal TrainingSample for unpickling
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
                def find_class(self, module, name):
                    if name == 'TrainingSample':
                        return TrainingSample
                    return super().find_class(module, name)
            
            total_samples = 0
            for pkl_file in pkl_files:
                try:
                    with open(pkl_file, 'rb') as f:
                        data = CustomUnpickler(f).load()
                    samples = data.get('samples', [])
                    total_samples += len(samples)
                except Exception as e:
                    print(f"   Error reading {pkl_file}: {e}")
            
            print(f"\n   Samples generated: {total_samples}")
            if total_samples > 0:
                print(f"   Time per sample: {elapsed/total_samples:.1f}s")
                print(f"   Samples per minute: {total_samples/(elapsed/60):.1f}")
                
                # Project to full dataset
                print(f"\n   📈 Projected times (at {elapsed/total_samples:.1f}s/sample):")
                for n_samples in [1000, 5000, 10000, 50000]:
                    proj_time = n_samples * (elapsed/total_samples)
                    hours = proj_time / 3600
                    if hours < 24:
                        print(f"      {n_samples:,} samples: {hours:.1f} hours")
                    else:
                        print(f"      {n_samples:,} samples: {hours/24:.1f} days")
    
    print("\n" + "="*70)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
