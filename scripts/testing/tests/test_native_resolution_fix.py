#!/usr/bin/env python3
"""
Test script to verify the native resolution SR fix.

This test generates a few samples and verifies:
1. HR is 256x256 (3m resolution)
2. LR 10m is ~77x77 (10m native resolution)
3. LR 20m is ~38x38 (20m native resolution)

Expected shapes for scale factor 10m/3m = 3.33:
- HR: 256x256 (Planet @ 3m)
- LR 10m: 77x77 (S2 @ 10m native) → round(256/3.33) = 77
- LR 20m: 38x38 (S2 @ 20m native) → round(256/6.67) = 38
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional


# Define TrainingSample for pickle loading (must match original definition)
@dataclass
class TrainingSample:
    """Training sample with temporal sequences and multi-resolution LR bands"""
    lr_sequence: np.ndarray  # (T, H, W, C) - 10m bands at native resolution
    hr_sequence: np.ndarray  # (T, H, W, C)
    temporal_dates: List[str]  # Sentinel-2 dates
    planet_dates: List[str]  # Planet dates (may differ by ±7 days)
    region_name: str
    sequence_id: int
    tile_id: int
    tile_bounds: Dict
    # Solar zenith angles for each frame
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None
    # Multi-resolution: 20m bands at native resolution
    lr_20m_sequence: Optional[np.ndarray] = None
    # Frame quality statistics
    frame_quality: Optional[List[Dict]] = None

def test_scale_factors():
    """Test that scale factor calculations are correct."""
    tile_size_px = 256
    hr_scale = 3.0  # Planet 3m
    lr_scale = 10.0  # S2 10m
    
    scale_factor_10m = lr_scale / hr_scale  # 10/3 = 3.33
    scale_factor_20m = 20.0 / hr_scale      # 20/3 = 6.67
    
    lr_10m_size = int(round(tile_size_px / scale_factor_10m))
    lr_20m_size = int(round(tile_size_px / scale_factor_20m))
    
    print("=" * 60)
    print("Native Resolution SR Fix - Scale Factor Test")
    print("=" * 60)
    print(f"\nScale factors:")
    print(f"  HR scale (Planet): {hr_scale}m")
    print(f"  LR scale (S2 10m): {lr_scale}m")
    print(f"  Scale factor 10m: {scale_factor_10m:.4f} ({lr_scale}/{hr_scale})")
    print(f"  Scale factor 20m: {scale_factor_20m:.4f} (20/{hr_scale})")
    
    print(f"\nExpected tile sizes:")
    print(f"  HR (Planet 3m): {tile_size_px}x{tile_size_px}")
    print(f"  LR 10m (S2): {lr_10m_size}x{lr_10m_size}")
    print(f"  LR 20m (S2): {lr_20m_size}x{lr_20m_size}")
    
    # Verify expected values
    assert lr_10m_size == 77, f"Expected LR 10m size 77, got {lr_10m_size}"
    assert lr_20m_size == 38, f"Expected LR 20m size 38, got {lr_20m_size}"
    
    print("\n✓ Scale factor calculations correct!")
    return True


def test_sample_shapes(dataset_path: str = None):
    """
    Test that generated samples have correct shapes.
    
    Args:
        dataset_path: Path to dataset.pkl with samples
    """
    if dataset_path is None:
        # Try to find a recent dataset
        data_dir = Path(__file__).parent.parent.parent / "data" / "sr_training"
        datasets = list(data_dir.glob("*/dataset.pkl"))
        if not datasets:
            print("\n⚠️ No datasets found. Generate samples first with --max-samples 5")
            return False
        dataset_path = str(max(datasets, key=lambda p: p.stat().st_mtime))
    
    print(f"\n" + "=" * 60)
    print("Sample Shape Verification")
    print("=" * 60)
    print(f"\nLoading dataset: {dataset_path}")
    
    import pickle
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    samples = data.get('samples', [])
    config = data.get('config', {})
    
    print(f"Found {len(samples)} samples")
    print(f"Config: {config}")
    
    if not samples:
        print("⚠️ No samples in dataset")
        return False
    
    # Check first few samples
    all_correct = True
    expected_hr = 256
    expected_lr_10m = 77
    expected_lr_20m = 38
    
    for i, sample in enumerate(samples[:5]):
        print(f"\nSample {i}:")
        
        # Check HR shape
        hr_shape = sample.hr_sequence.shape
        print(f"  HR shape: {hr_shape}")
        if hr_shape[1] != expected_hr or hr_shape[2] != expected_hr:
            print(f"  ❌ Expected HR {expected_hr}x{expected_hr}")
            all_correct = False
        else:
            print(f"  ✓ HR correct")
        
        # Check LR shape
        lr_shape = sample.lr_sequence.shape
        print(f"  LR 10m shape: {lr_shape}")
        if lr_shape[1] == expected_hr:
            print(f"  ❌ LR is {expected_hr}x{expected_hr} - NOT native resolution!")
            print(f"     Should be {expected_lr_10m}x{expected_lr_10m}")
            all_correct = False
        elif lr_shape[1] == expected_lr_10m:
            print(f"  ✓ LR 10m at native resolution!")
        else:
            print(f"  ⚠️ LR has unexpected size: {lr_shape[1]}x{lr_shape[2]}")
        
        # Check LR 20m if available
        if hasattr(sample, 'lr_20m_sequence') and sample.lr_20m_sequence is not None:
            lr_20m_shape = sample.lr_20m_sequence.shape
            print(f"  LR 20m shape: {lr_20m_shape}")
            if lr_20m_shape[1] == expected_lr_20m:
                print(f"  ✓ LR 20m at native resolution!")
            else:
                print(f"  ⚠️ LR 20m unexpected size: {lr_20m_shape[1]}x{lr_20m_shape[2]}")
    
    if all_correct:
        print("\n" + "=" * 60)
        print("✓ All samples have correct native resolution shapes!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Some samples have incorrect shapes - fix needed!")
        print("=" * 60)
    
    return all_correct


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("   NATIVE RESOLUTION SR FIX - VERIFICATION")
    print("=" * 60)
    
    # Test 1: Scale factor calculations
    test_scale_factors()
    
    # Test 2: Check sample shapes if dataset exists
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset.pkl to verify')
    args = parser.parse_args()
    
    test_sample_shapes(args.dataset)
    
    print("\n" + "=" * 60)
    print("To generate new samples with native resolution:")
    print("  cd /media/eamurciab/TOSHIBA_EXT1/PhD/SR")
    print("  conda activate tf_gpu")
    print("  python scripts/preprocessing/generate_sta2_grid_parallel.py \\")
    print("    --region STA-2 --start 2020-01-01 --end 2021-01-01 \\")
    print("    --max-samples 5 --normalize \\")
    print("    --output data/sr_training/test_native_resolution/")
    print("=" * 60)


if __name__ == "__main__":
    main()
