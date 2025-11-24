"""
Validate chunked dataset quality and format
Checks for data integrity, realistic values, and proper shapes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import numpy as np
from pathlib import Path
import pickle

from create_chunked_dataset import load_chunked_dataset


def validate_chunked_dataset(dataset_dir="./chunked_sr_dataset"):
    """
    Comprehensive validation of chunked dataset
    """
    
    print("="*70)
    print("🔍 VALIDATING CHUNKED DATASET")
    print("="*70)
    print(f"Dataset: {dataset_dir}\n")
    
    # Load dataset
    try:
        samples, metadata = load_chunked_dataset(dataset_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False
    
    print(f"\n{'='*70}")
    print("TEST 1: Basic Structure")
    print(f"{'='*70}")
    
    # Check sample count
    expected_samples = metadata['total_samples']
    actual_samples = len(samples)
    
    print(f"Expected samples: {expected_samples}")
    print(f"Actual samples: {actual_samples}")
    
    if actual_samples != expected_samples:
        print(f"⚠️  Sample count mismatch!")
    else:
        print("✅ Sample count matches")
    
    # Check shapes
    print(f"\nExpected LR shape: {metadata['lr_shape']}")
    print(f"Expected HR shape: {metadata['hr_shape']}")
    
    print(f"\n{'='*70}")
    print("TEST 2: Sample Validation")
    print(f"{'='*70}")
    
    # Validate each sample
    valid_samples = 0
    invalid_reasons = []
    
    for i, sample in enumerate(samples):
        try:
            # Check attributes
            if not hasattr(sample, 'lr_sequence'):
                invalid_reasons.append(f"Sample {i}: Missing lr_sequence")
                continue
            if not hasattr(sample, 'hr_sequence'):
                invalid_reasons.append(f"Sample {i}: Missing hr_sequence")
                continue
            
            lr = sample.lr_sequence
            hr = sample.hr_sequence
            
            # Check shapes
            if lr.shape != tuple(metadata['lr_shape']):
                invalid_reasons.append(f"Sample {i}: LR shape mismatch {lr.shape} != {metadata['lr_shape']}")
                continue
            if hr.shape != tuple(metadata['hr_shape']):
                invalid_reasons.append(f"Sample {i}: HR shape mismatch {hr.shape} != {metadata['hr_shape']}")
                continue
            
            # Check for NaN/Inf
            if np.any(np.isnan(lr)) or np.any(np.isinf(lr)):
                invalid_reasons.append(f"Sample {i}: LR contains NaN/Inf")
                continue
            if np.any(np.isnan(hr)) or np.any(np.isinf(hr)):
                invalid_reasons.append(f"Sample {i}: HR contains NaN/Inf")
                continue
            
            valid_samples += 1
            
        except Exception as e:
            invalid_reasons.append(f"Sample {i}: {str(e)}")
    
    print(f"Valid samples: {valid_samples}/{len(samples)}")
    if invalid_reasons:
        print(f"❌ Found {len(invalid_reasons)} invalid samples:")
        for reason in invalid_reasons[:10]:  # Show first 10
            print(f"   - {reason}")
        if len(invalid_reasons) > 10:
            print(f"   ... and {len(invalid_reasons) - 10} more")
    else:
        print("✅ All samples have correct structure")
    
    print(f"\n{'='*70}")
    print("TEST 3: Data Quality (Random Sample)")
    print(f"{'='*70}")
    
    # Analyze a random sample
    import random
    sample_idx = random.randint(0, len(samples) - 1)
    sample = samples[sample_idx]
    
    print(f"\nAnalyzing sample {sample_idx}...")
    
    lr = sample.lr_sequence
    hr = sample.hr_sequence
    
    print(f"\n📊 LR Statistics:")
    print(f"   Shape: {lr.shape}")
    print(f"   Min: {lr.min():.2f}")
    print(f"   Max: {lr.max():.2f}")
    print(f"   Mean: {lr.mean():.2f}")
    print(f"   Std: {lr.std():.2f}")
    
    # Per-band stats
    T, H, W, C = lr.shape
    print(f"\n   Per-band (center frame):")
    center_frame = lr[T // 2]
    for band in range(C):
        band_data = center_frame[:, :, band]
        print(f"      Band {band}: mean={band_data.mean():.1f}, "
              f"std={band_data.std():.1f}, "
              f"min={band_data.min():.0f}, "
              f"max={band_data.max():.0f}")
    
    print(f"\n📊 HR Statistics:")
    print(f"   Shape: {hr.shape}")
    print(f"   Min: {hr.min():.2f}")
    print(f"   Max: {hr.max():.2f}")
    print(f"   Mean: {hr.mean():.2f}")
    print(f"   Std: {hr.std():.2f}")
    
    # Per-band stats
    T, H, W, C = hr.shape
    print(f"\n   Per-band (center frame):")
    center_frame = hr[T // 2]
    for band in range(C):
        band_data = center_frame[:, :, band]
        print(f"      Band {band}: mean={band_data.mean():.1f}, "
              f"std={band_data.std():.1f}, "
              f"min={band_data.min():.0f}, "
              f"max={band_data.max():.0f}")
    
    print(f"\n{'='*70}")
    print("TEST 4: Value Range Check")
    print(f"{'='*70}")
    
    # Check for zeros/fallback patterns
    lr_zeros = np.sum(lr == 0)
    hr_zeros = np.sum(hr == 0)
    lr_total = lr.size
    hr_total = hr.size
    
    lr_zero_pct = (lr_zeros / lr_total) * 100
    hr_zero_pct = (hr_zeros / hr_total) * 100
    
    print(f"\nZero pixels:")
    print(f"   LR: {lr_zeros}/{lr_total} ({lr_zero_pct:.2f}%)")
    print(f"   HR: {hr_zeros}/{hr_total} ({hr_zero_pct:.2f}%)")
    
    if hr_zero_pct > 5.0:
        print(f"   ⚠️  HR has {hr_zero_pct:.1f}% zeros (might indicate masking)")
    else:
        print(f"   ✅ Zero percentage acceptable")
    
    # Check for realistic reflectance values
    # Typical satellite reflectance: 0-10000 (with some outliers)
    lr_valid = np.sum((lr >= 0) & (lr <= 15000))
    hr_valid = np.sum((hr >= 0) & (hr <= 15000))
    
    lr_valid_pct = (lr_valid / lr_total) * 100
    hr_valid_pct = (hr_valid / hr_total) * 100
    
    print(f"\nRealistic reflectance (0-15000):")
    print(f"   LR: {lr_valid}/{lr_total} ({lr_valid_pct:.2f}%)")
    print(f"   HR: {hr_valid}/{hr_total} ({hr_valid_pct:.2f}%)")
    
    if lr_valid_pct < 95 or hr_valid_pct < 95:
        print(f"   ⚠️  Many pixels outside expected range")
    else:
        print(f"   ✅ Values within expected range")
    
    # Check for fallback patterns (e.g., [300, 500, 700, 600])
    fallback_pattern = np.array([300, 500, 700, 600])
    
    # Check center frame of HR
    hr_center = hr[T // 2]
    
    # Sample some pixels
    test_pixels = []
    for _ in range(100):
        y = random.randint(0, hr_center.shape[0] - 1)
        x = random.randint(0, hr_center.shape[1] - 1)
        pixel = hr_center[y, x, :]
        test_pixels.append(pixel)
    
    test_pixels = np.array(test_pixels)
    
    # Check for exact fallback matches
    fallback_matches = np.all(test_pixels == fallback_pattern, axis=1).sum()
    
    print(f"\nFallback pattern check (100 random pixels):")
    print(f"   Matches [300,500,700,600]: {fallback_matches}")
    
    if fallback_matches > 0:
        print(f"   ❌ Found fallback values!")
    else:
        print(f"   ✅ No fallback patterns detected")
    
    print(f"\n{'='*70}")
    print("TEST 5: Spatial Variation")
    print(f"{'='*70}")
    
    # Check for spatial variation (not uniform)
    hr_center = hr[T // 2, :, :, 0]  # First band of center frame
    
    # Compute gradients
    grad_x = np.abs(np.diff(hr_center, axis=1))
    grad_y = np.abs(np.diff(hr_center, axis=0))
    
    mean_grad_x = grad_x.mean()
    mean_grad_y = grad_y.mean()
    
    print(f"\nSpatial gradients (Band 0, center frame):")
    print(f"   Mean gradient X: {mean_grad_x:.2f}")
    print(f"   Mean gradient Y: {mean_grad_y:.2f}")
    
    if mean_grad_x < 1 or mean_grad_y < 1:
        print(f"   ⚠️  Very low gradients (image might be uniform)")
    else:
        print(f"   ✅ Good spatial variation")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    # Overall assessment
    all_passed = (
        actual_samples == expected_samples and
        valid_samples == len(samples) and
        hr_zero_pct < 5.0 and
        hr_valid_pct >= 95 and
        fallback_matches == 0 and
        mean_grad_x >= 1 and
        mean_grad_y >= 1
    )
    
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("   Dataset is ready for training!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("   Review warnings above before training")
    
    print(f"\n{'='*70}\n")
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate chunked dataset')
    parser.add_argument('--dataset-dir', default='./chunked_sr_dataset',
                       help='Path to dataset directory')
    
    args = parser.parse_args()
    
    validate_chunked_dataset(args.dataset_dir)
