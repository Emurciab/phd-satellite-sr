#!/usr/bin/env python3
"""
Dataset validation script - checks structure, metadata, and data quality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

# Must match the dataclass used in generation
@dataclass
class TrainingSample:
    """Training sample with temporal sequences"""
    lr_sequence: np.ndarray  # (T, H, W, C)
    hr_sequence: np.ndarray  # (T, H, W, C)
    temporal_dates: List[str]  # Sentinel-2 dates
    planet_dates: List[str]  # Planet dates (may differ by ±7 days)
    region_name: str
    sequence_id: int
    tile_id: int
    tile_bounds: Dict
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None


def validate_dataset(dataset_path: str):
    """Validate dataset structure, metadata, and data quality."""
    dataset_path = Path(dataset_path)
    print(f"Loading dataset from: {dataset_path}")
    print(f"File size: {dataset_path.stat().st_size / (1024*1024):.2f} MB")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    print("\n" + "="*80)
    print("📊 DATASET STRUCTURE VALIDATION")
    print("="*80)

    print(f"\n📁 Top-level keys: {list(dataset.keys())}")

    # Config info
    config = dataset['config']
    print(f"\n⚙️  CONFIG:")
    for k, v in config.items():
        print(f"   {k}: {v}")

    # Samples overview
    samples = dataset['samples']
    print(f"\n📊 SAMPLES: {len(samples)} total")

    # Check sample structure
    sample = samples[0]
    print(f"\n🔍 SAMPLE 0 STRUCTURE:")
    print(f"   Type: {type(sample).__name__}")
    print(f"   Fields: {list(sample.__class__.__dataclass_fields__.keys())}")

    # Validate each sample
    print(f"\n🔬 VALIDATING ALL {len(samples)} SAMPLES...")
    lr_shapes = []
    hr_shapes = []
    issues = []
    
    for i, s in enumerate(samples):
        lr_shapes.append(s.lr_sequence.shape)
        hr_shapes.append(s.hr_sequence.shape)
        
        # Check for NaN/Inf
        if np.any(np.isnan(s.lr_sequence)):
            issues.append(f"Sample {i}: NaN in lr_sequence")
        if np.any(np.isnan(s.hr_sequence)):
            issues.append(f"Sample {i}: NaN in hr_sequence")
        if np.any(np.isinf(s.lr_sequence)):
            issues.append(f"Sample {i}: Inf in lr_sequence")
        if np.any(np.isinf(s.hr_sequence)):
            issues.append(f"Sample {i}: Inf in hr_sequence")
        
        # Check for all zeros
        if np.all(s.lr_sequence == 0):
            issues.append(f"Sample {i}: all zeros in lr_sequence")
        if np.all(s.hr_sequence == 0):
            issues.append(f"Sample {i}: all zeros in hr_sequence")
        
        # Check shapes match
        if s.lr_sequence.shape[0] != s.hr_sequence.shape[0]:
            issues.append(f"Sample {i}: temporal dimension mismatch")
        if s.lr_sequence.shape[-1] != s.hr_sequence.shape[-1]:
            issues.append(f"Sample {i}: channel mismatch LR={s.lr_sequence.shape[-1]} HR={s.hr_sequence.shape[-1]}")

    # Shape statistics
    unique_lr = list(set(lr_shapes))
    unique_hr = list(set(hr_shapes))
    print(f"\n📐 LR SEQUENCE SHAPES: {len(unique_lr)} unique")
    for shape in sorted(unique_lr):
        count = lr_shapes.count(shape)
        print(f"   {shape}: {count} samples")

    print(f"\n📐 HR SEQUENCE SHAPES: {len(unique_hr)} unique")
    for shape in sorted(unique_hr):
        count = hr_shapes.count(shape)
        print(f"   {shape}: {count} samples")

    # Data quality issues
    if issues:
        print(f"\n⚠️  DATA QUALITY ISSUES: {len(issues)}")
        for issue in issues[:20]:
            print(f"   {issue}")
        if len(issues) > 20:
            print(f"   ... and {len(issues) - 20} more issues")
    else:
        print(f"\n✅ NO DATA QUALITY ISSUES FOUND!")

    # Sample metadata example
    print(f"\n📋 SAMPLE 0 METADATA:")
    print(f"   Region: {sample.region_name}")
    print(f"   Sequence ID: {sample.sequence_id}")
    print(f"   Tile ID: {sample.tile_id}")
    print(f"   S2 Dates: {sample.temporal_dates}")
    print(f"   Planet Dates: {sample.planet_dates}")
    print(f"   S2 Zenith angles: {sample.s2_solar_zenith}")
    print(f"   Planet Zenith angles: {sample.planet_solar_zenith}")
    print(f"   Tile bounds keys: {list(sample.tile_bounds.keys())}")
    print(f"   Band mode: {sample.tile_bounds.get('band_mode', 'N/A')}")
    print(f"   Planet bands: {sample.tile_bounds.get('planet_bands', 'N/A')}")
    print(f"   Sentinel bands: {sample.tile_bounds.get('sentinel_bands', 'N/A')}")
    print(f"   Cache hits/misses: {sample.tile_bounds.get('cache_hits', 'N/A')}/{sample.tile_bounds.get('cache_misses', 'N/A')}")

    # Value statistics across all samples
    print(f"\n📈 VALUE STATISTICS (across all samples):")
    all_lr_min = min(s.lr_sequence.min() for s in samples)
    all_lr_max = max(s.lr_sequence.max() for s in samples)
    all_hr_min = min(s.hr_sequence.min() for s in samples)
    all_hr_max = max(s.hr_sequence.max() for s in samples)
    print(f"   LR global min/max: {all_lr_min:.2f} / {all_lr_max:.2f}")
    print(f"   HR global min/max: {all_hr_min:.2f} / {all_hr_max:.2f}")
    
    # Sample-level stats
    print(f"\n   Sample 0 LR: min={sample.lr_sequence.min():.2f}, max={sample.lr_sequence.max():.2f}, mean={sample.lr_sequence.mean():.2f}")
    print(f"   Sample 0 HR: min={sample.hr_sequence.min():.2f}, max={sample.hr_sequence.max():.2f}, mean={sample.hr_sequence.mean():.2f}")

    # Check resolution ratio
    lr_h, lr_w = sample.lr_sequence.shape[1:3]
    hr_h, hr_w = sample.hr_sequence.shape[1:3]
    print(f"\n🔎 RESOLUTION CHECK (Sample 0):")
    print(f"   LR spatial: {lr_h}x{lr_w}")
    print(f"   HR spatial: {hr_h}x{hr_w}")
    print(f"   Scale factor H: {hr_h/lr_h:.2f}x")
    print(f"   Scale factor W: {hr_w/lr_w:.2f}x")
    print(f"   Temporal frames: {sample.lr_sequence.shape[0]}")
    print(f"   Channels: {sample.lr_sequence.shape[-1]}")

    # Training readiness summary
    print("\n" + "="*80)
    print("🎯 TRAINING READINESS SUMMARY")
    print("="*80)
    print(f"   Total samples: {len(samples)}")
    print(f"   Data quality: {'PASS ✅' if not issues else 'ISSUES ⚠️'}")
    print(f"   Consistent shapes: {'YES ✅' if len(unique_lr) == 1 and len(unique_hr) == 1 else 'VARIABLE ⚠️'}")
    print(f"   Scale factor: ~{hr_h/lr_h:.1f}x (Planet 3m → S2 10m)")
    print(f"   Temporal sequence: {sample.lr_sequence.shape[0]} frames")
    print(f"   Bands: {sample.lr_sequence.shape[-1]} ({sample.tile_bounds.get('sentinel_bands', 'N/A')})")
    print(f"   Ready for training: {'YES ✅' if not issues else 'NEEDS REVIEW ⚠️'}")
    print("="*80)
    
    return dataset, samples, issues


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate SR dataset')
    parser.add_argument('dataset_path', type=str, help='Path to dataset pickle file')
    args = parser.parse_args()
    
    validate_dataset(args.dataset_path)
