#!/usr/bin/env python3
"""
Prepare combined training dataset from ALL STAs and years.

This script:
1. Loads 5-band samples from all available STAs (1E, 1W, 2, 3/4, 5/6)
2. Combines them into a single dataset
3. Creates 80/20 train/test split
4. Saves as memory-mapped numpy arrays for fast training

Output: data/sr_training/all_stas_combined/
    - mmap_train/  (80% of data)
    - mmap_test/   (20% of data)
    - metadata.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import random


@dataclass
class TrainingSample5Band:
    """Placeholder for unpickling"""
    pass


def setup_unpickle_modules():
    """Register modules for unpickling different sample formats"""
    import types
    
    fake_mod = types.ModuleType('convert_cache_to_5band')
    fake_mod.TrainingSample5Band = TrainingSample5Band
    sys.modules['convert_cache_to_5band'] = fake_mod
    
    fake_mod2 = types.ModuleType('scripts.preprocessing.convert_cache_to_5band')
    fake_mod2.TrainingSample5Band = TrainingSample5Band
    sys.modules['scripts.preprocessing.convert_cache_to_5band'] = fake_mod2


def load_dataset(dataset_path: Path) -> List:
    """Load samples from a dataset path (handles both single pkl and chunked format)"""
    setup_unpickle_modules()
    
    samples = []
    
    # Check for chunked format
    metadata_file = dataset_path / 'metadata.pkl'
    if metadata_file.exists():
        chunk_files = sorted(dataset_path.glob('chunk_*.pkl'))
        for chunk_file in chunk_files:
            with open(chunk_file, 'rb') as f:
                chunk_samples = pickle.load(f)
            samples.extend(chunk_samples)
    else:
        # Single file format - try different filenames
        pkl_files = list(dataset_path.glob('*.pkl'))
        if pkl_files:
            for pkl_file in pkl_files:
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, dict):
                        if 'samples' in data:
                            samples.extend(data['samples'])
                        elif 'data' in data:
                            samples.extend(data['data'])
                    elif isinstance(data, list):
                        samples.extend(data)
                except Exception as e:
                    print(f"  Warning: Could not load {pkl_file}: {e}")
    
    return samples


def extract_arrays(sample) -> Optional[Dict]:
    """Extract numpy arrays from a sample object"""
    # Handle different attribute naming conventions
    hr = getattr(sample, 'HR', None) or getattr(sample, 'hr_sequence', None)
    lr_10m = getattr(sample, 'LR_10m', None) or getattr(sample, 'lr_sequence', None)
    lr_20m = getattr(sample, 'LR_20m', None) or getattr(sample, 'lr_20m_sequence', None)
    hr_10m = getattr(sample, 'HR_10m', None) or getattr(sample, 'hr_10m_sequence', None)
    hr_20m = getattr(sample, 'HR_20m', None) or getattr(sample, 'hr_20m_sequence', None)
    
    if hr is None or lr_10m is None:
        return None
    
    return {
        'hr': hr,
        'lr_10m': lr_10m,
        'lr_20m': lr_20m,
        'hr_10m': hr_10m,
        'hr_20m': hr_20m,
    }


def validate_sample(arrays: Dict, expected_seq_len: int = 3) -> bool:
    """Validate sample has expected shapes"""
    hr = arrays['hr']
    lr_10m = arrays['lr_10m']
    
    # Check sequence length
    if hr.shape[0] != expected_seq_len or lr_10m.shape[0] != expected_seq_len:
        return False
    
    # Check spatial dimensions
    if hr.shape[1] != 256 or hr.shape[2] != 256:
        return False
    
    # Check bands (5 for HR, 3 for LR_10m)
    if hr.shape[3] != 5:
        return False
    if lr_10m.shape[3] != 3:
        return False
    
    # Check for NaN/Inf
    if np.any(np.isnan(hr)) or np.any(np.isinf(hr)):
        return False
    if np.any(np.isnan(lr_10m)) or np.any(np.isinf(lr_10m)):
        return False
    
    return True


def save_mmap_dataset(samples: List[Dict], output_dir: Path, name: str):
    """Save samples as memory-mapped numpy arrays"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = len(samples)
    if n_samples == 0:
        print(f"  No samples to save for {name}")
        return
    
    # Get shapes from first sample
    first = samples[0]
    seq_len = first['hr'].shape[0]
    
    print(f"  Allocating arrays for {n_samples} samples...")
    
    # Pre-allocate arrays
    # HR: (N, T, 256, 256, 5)
    hr_shape = (n_samples, seq_len, 256, 256, 5)
    # LR_10m: (N, T, H, W, 3) - typically 77x77 or 78x78
    lr_10m_shape = (n_samples, seq_len) + first['lr_10m'].shape[1:]
    # LR_20m: (N, T, H, W, 2) - typically 38x38 or 39x39
    lr_20m_shape = (n_samples, seq_len) + first['lr_20m'].shape[1:] if first['lr_20m'] is not None else None
    # HR_10m: (N, T, 256, 256, 3)
    hr_10m_shape = (n_samples, seq_len, 256, 256, 3) if first['hr_10m'] is not None else None
    # HR_20m: (N, T, 128, 128, 2)
    hr_20m_shape = (n_samples, seq_len) + first['hr_20m'].shape[1:] if first['hr_20m'] is not None else None
    
    print(f"  HR shape: {hr_shape}")
    print(f"  LR_10m shape: {lr_10m_shape}")
    
    # Create memory-mapped files
    hr_path = output_dir / 'hr.npy'
    lr_path = output_dir / 'lr.npy'
    
    hr_mmap = np.lib.format.open_memmap(hr_path, mode='w+', dtype=np.float32, shape=hr_shape)
    lr_mmap = np.lib.format.open_memmap(lr_path, mode='w+', dtype=np.float32, shape=lr_10m_shape)
    
    lr_20m_mmap = None
    if lr_20m_shape:
        lr_20m_path = output_dir / 'lr_20m.npy'
        lr_20m_mmap = np.lib.format.open_memmap(lr_20m_path, mode='w+', dtype=np.float32, shape=lr_20m_shape)
    
    hr_10m_mmap = None
    if hr_10m_shape:
        hr_10m_path = output_dir / 'hr_10m.npy'
        hr_10m_mmap = np.lib.format.open_memmap(hr_10m_path, mode='w+', dtype=np.float32, shape=hr_10m_shape)
    
    hr_20m_mmap = None
    if hr_20m_shape:
        hr_20m_path = output_dir / 'hr_20m.npy'
        hr_20m_mmap = np.lib.format.open_memmap(hr_20m_path, mode='w+', dtype=np.float32, shape=hr_20m_shape)
    
    # Fill arrays
    print(f"  Writing samples...")
    for i, s in enumerate(samples):
        hr_mmap[i] = s['hr']
        lr_mmap[i] = s['lr_10m']
        
        if lr_20m_mmap is not None and s['lr_20m'] is not None:
            lr_20m_mmap[i] = s['lr_20m']
        
        if hr_10m_mmap is not None and s['hr_10m'] is not None:
            hr_10m_mmap[i] = s['hr_10m']
        
        if hr_20m_mmap is not None and s['hr_20m'] is not None:
            hr_20m_mmap[i] = s['hr_20m']
        
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_samples}")
    
    # Flush to disk
    del hr_mmap, lr_mmap
    if lr_20m_mmap is not None:
        del lr_20m_mmap
    if hr_10m_mmap is not None:
        del hr_10m_mmap
    if hr_20m_mmap is not None:
        del hr_20m_mmap
    
    print(f"  ✅ Saved {n_samples} samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare combined training dataset from all STAs')
    parser.add_argument('--output', type=str, default='data/sr_training/all_stas_combined',
                        help='Output directory')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Fraction for test set (default: 0.2 = 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--max-samples-per-sta', type=int, default=None,
                        help='Limit samples per STA (for testing)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PREPARING COMBINED TRAINING DATASET FROM ALL STAs")
    print("=" * 70)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Define all dataset paths
    datasets_config = [
        # STA-1E
        ('STA-1E', '2023', Path('data/datasets/sta1e_2023_5band')),
        ('STA-1E', '2024', Path('data/datasets/sta1e_2024_5band')),
        # STA-1W
        ('STA-1W', '2023', Path('data/datasets/sta1w_2023_5band')),
        ('STA-1W', '2024', Path('data/datasets/sta1w_2024_5band')),
        # STA-2
        ('STA-2', '2022', Path('data/sr_training/5band_v3/STA2_2022/superdove_5band')),
        ('STA-2', '2023', Path('data/sr_training/5band_v3/STA2_2023/superdove_5band')),
        ('STA-2', '2024', Path('data/sr_training/5band_v3/STA2_2024/superdove_5band')),
        # STA-3/4
        ('STA-3/4', '2023', Path('data/datasets/sta34_2023_5band')),
        ('STA-3/4', '2024', Path('data/datasets/sta34_2024_5band')),
        # STA-5/6
        ('STA-5/6', '2023', Path('data/datasets/sta56_2023_5band')),
        ('STA-5/6', '2024', Path('data/datasets/sta56_2024_5band')),
    ]
    
    # Load all samples
    all_samples = []
    stats = {}
    
    print("\n📂 Loading datasets...")
    for sta, year, path in datasets_config:
        if not path.exists():
            print(f"  ⚠️ {sta} {year}: Not found at {path}")
            continue
        
        print(f"  Loading {sta} {year}...")
        samples = load_dataset(path)
        
        # Extract and validate
        valid_samples = []
        for s in samples:
            arrays = extract_arrays(s)
            if arrays is not None and validate_sample(arrays):
                valid_samples.append(arrays)
        
        # Apply limit if specified
        if args.max_samples_per_sta and len(valid_samples) > args.max_samples_per_sta:
            valid_samples = random.sample(valid_samples, args.max_samples_per_sta)
        
        key = f"{sta}_{year}"
        stats[key] = len(valid_samples)
        all_samples.extend(valid_samples)
        print(f"    ✅ {len(valid_samples)} valid samples")
    
    print(f"\n📊 Total samples: {len(all_samples)}")
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Split 80/20
    n_test = int(len(all_samples) * args.test_split)
    n_train = len(all_samples) - n_test
    
    train_samples = all_samples[:n_train]
    test_samples = all_samples[n_train:]
    
    print(f"\n📊 Split:")
    print(f"  Training: {len(train_samples)} ({100*(1-args.test_split):.0f}%)")
    print(f"  Test:     {len(test_samples)} ({100*args.test_split:.0f}%)")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as mmap
    print(f"\n💾 Saving training data...")
    save_mmap_dataset(train_samples, output_dir / 'mmap_train', 'train')
    
    print(f"\n💾 Saving test data...")
    save_mmap_dataset(test_samples, output_dir / 'mmap_test', 'test')
    
    # Save metadata
    metadata = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'test_samples': len(test_samples),
        'test_split': args.test_split,
        'seed': args.seed,
        'samples_per_dataset': stats,
        'datasets': [f"{sta}_{year}" for sta, year, _ in datasets_config if f"{sta}_{year}" in stats],
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"\nSamples per STA/Year:")
    for key, count in stats.items():
        print(f"  {key}: {count}")


if __name__ == '__main__':
    main()
