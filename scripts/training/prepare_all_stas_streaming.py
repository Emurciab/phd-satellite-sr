#!/usr/bin/env python3
"""
MEMORY-EFFICIENT dataset preparation from ALL STAs.

This script processes one chunk at a time to avoid RAM exhaustion:
1. First pass: Count total samples and collect metadata
2. Pre-allocate mmap files
3. Second pass: Stream data into mmap files chunk by chunk

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
from typing import List, Dict, Optional, Iterator, Tuple
import random
import gc


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


def get_chunk_files(dataset_path: Path) -> List[Path]:
    """Get list of chunk/pkl files to process"""
    metadata_file = dataset_path / 'metadata.pkl'
    if metadata_file.exists():
        # Chunked format
        return sorted(dataset_path.glob('chunk_*.pkl'))
    else:
        # Single file format
        return list(dataset_path.glob('*.pkl'))


def count_samples_in_file(pkl_file: Path) -> int:
    """Count samples in a pickle file without loading all data"""
    setup_unpickle_modules()
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            if 'samples' in data:
                return len(data['samples'])
            elif 'data' in data:
                return len(data['data'])
            return 0
        elif isinstance(data, list):
            return len(data)
        return 0
    except Exception as e:
        print(f"  Warning: Could not count {pkl_file}: {e}")
        return 0


def stream_samples_from_file(pkl_file: Path) -> Iterator:
    """Stream samples from a pickle file one at a time"""
    setup_unpickle_modules()
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            if 'samples' in data:
                samples = data['samples']
            elif 'data' in data:
                samples = data['data']
            else:
                samples = []
        elif isinstance(data, list):
            samples = data
        else:
            samples = []
        
        for sample in samples:
            yield sample
        
        # Clean up
        del data
        del samples
        gc.collect()
        
    except Exception as e:
        print(f"  Warning: Could not load {pkl_file}: {e}")


def extract_arrays(sample, target_shapes: Dict = None) -> Optional[Dict]:
    """Extract numpy arrays from a sample object, resizing if needed"""
    hr = getattr(sample, 'HR', None) or getattr(sample, 'hr_sequence', None)
    lr_10m = getattr(sample, 'LR_10m', None) or getattr(sample, 'lr_sequence', None)
    lr_20m = getattr(sample, 'LR_20m', None) or getattr(sample, 'lr_20m_sequence', None)
    hr_10m = getattr(sample, 'HR_10m', None) or getattr(sample, 'hr_10m_sequence', None)
    hr_20m = getattr(sample, 'HR_20m', None) or getattr(sample, 'hr_20m_sequence', None)
    
    if hr is None or lr_10m is None:
        return None
    
    arrays = {
        'hr': hr,
        'lr_10m': lr_10m,
        'lr_20m': lr_20m,
        'hr_10m': hr_10m,
        'hr_20m': hr_20m,
    }
    
    # Resize arrays if target shapes are specified and don't match
    if target_shapes is not None:
        from scipy.ndimage import zoom
        
        for key in ['lr_10m', 'lr_20m', 'hr_10m', 'hr_20m']:
            if arrays[key] is not None and key in target_shapes:
                current_shape = arrays[key].shape
                target_shape = target_shapes[key]
                
                if current_shape != target_shape:
                    # Calculate zoom factors for spatial dimensions only
                    # Shape is (T, H, W, C)
                    zoom_factors = [
                        target_shape[0] / current_shape[0],  # T
                        target_shape[1] / current_shape[1],  # H
                        target_shape[2] / current_shape[2],  # W
                        1.0  # C (channels stay same)
                    ]
                    
                    # Only resize if spatial dimensions differ
                    if zoom_factors[1] != 1.0 or zoom_factors[2] != 1.0:
                        arrays[key] = zoom(arrays[key], zoom_factors, order=1).astype(np.float32)
    
    return arrays


def get_array_shapes(sample) -> Optional[Dict]:
    """Get array shapes from first valid sample"""
    arrays = extract_arrays(sample)
    if arrays is None:
        return None
    
    shapes = {}
    for key, arr in arrays.items():
        if arr is not None:
            shapes[key] = arr.shape
    return shapes


# Dataset configurations
DATASETS = [
    # STA-1E
    ('STA-1E', '2022', 'data/datasets/sta1e_2022_5band'),
    ('STA-1E', '2023', 'data/datasets/sta1e_2023_5band'),
    ('STA-1E', '2024', 'data/datasets/sta1e_2024_5band'),
    # STA-1W
    ('STA-1W', '2023', 'data/datasets/sta1w_2023_5band'),
    ('STA-1W', '2024', 'data/datasets/sta1w_2024_5band'),
    # STA-2 (in sr_training folder)
    ('STA-2', '2022', 'data/sr_training/5band_v3/STA2_2022/superdove_5band'),
    ('STA-2', '2023', 'data/sr_training/5band_v3/STA2_2023/superdove_5band'),
    ('STA-2', '2024', 'data/sr_training/5band_v3/STA2_2024/superdove_5band'),
    # STA-3/4
    ('STA-3/4', '2023', 'data/datasets/sta34_2023_5band'),
    ('STA-3/4', '2024', 'data/datasets/sta34_2024_5band'),
    # STA-5/6
    ('STA-5/6', '2023', 'data/datasets/sta56_2023_5band'),
    ('STA-5/6', '2024', 'data/datasets/sta56_2024_5band'),
]


def main():
    parser = argparse.ArgumentParser(description='Prepare combined dataset from all STAs (memory-efficient)')
    parser.add_argument('--output', type=str, default='data/sr_training/all_stas_combined',
                        help='Output directory')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Fraction for test set (default: 0.2 = 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only count samples, do not create output')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("MEMORY-EFFICIENT DATASET PREPARATION")
    print("=" * 70)
    
    # ========== PHASE 1: Count samples and get shapes ==========
    print("\n📊 PHASE 1: Counting samples...")
    
    dataset_info = []
    total_samples = 0
    shapes = None
    
    for sta, year, path_str in DATASETS:
        dataset_path = Path(path_str)
        if not dataset_path.exists():
            print(f"  ⚠️  {sta} {year}: NOT FOUND ({path_str})")
            continue
        
        chunk_files = get_chunk_files(dataset_path)
        if not chunk_files:
            print(f"  ⚠️  {sta} {year}: No pickle files found")
            continue
        
        # Count samples
        count = 0
        for chunk_file in chunk_files:
            count += count_samples_in_file(chunk_file)
        
        if count > 0:
            dataset_info.append({
                'sta': sta,
                'year': year,
                'path': path_str,
                'chunk_files': [str(f) for f in chunk_files],
                'count': count
            })
            total_samples += count
            print(f"  ✅ {sta} {year}: {count} samples")
            
            # Get shapes from first file if we don't have them yet
            if shapes is None:
                for sample in stream_samples_from_file(chunk_files[0]):
                    shapes = get_array_shapes(sample)
                    if shapes:
                        break
    
    print(f"\n📈 Total samples found: {total_samples}")
    print(f"   Array shapes: {shapes}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - not creating output files")
        return
    
    if total_samples == 0:
        print("❌ No samples found!")
        return
    
    # ========== PHASE 2: Create train/test indices ==========
    print("\n🎲 PHASE 2: Creating train/test split...")
    
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    
    n_test = int(total_samples * args.test_split)
    n_train = total_samples - n_test
    
    test_indices = set(all_indices[:n_test])
    train_indices = set(all_indices[n_test:])
    
    print(f"   Training: {n_train} samples ({100*(1-args.test_split):.0f}%)")
    print(f"   Test: {n_test} samples ({100*args.test_split:.0f}%)")
    
    # ========== PHASE 3: Create mmap files ==========
    print("\n💾 PHASE 3: Creating memory-mapped files...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / 'mmap_train'
    test_dir = output_dir / 'mmap_test'
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Determine shapes - use first sample to get exact dimensions
    seq_len = shapes['hr'][0]  # Temporal dimension
    
    # Create mmap arrays for training
    train_arrays = {
        'lr': np.lib.format.open_memmap(
            str(train_dir / 'lr.npy'), mode='w+', dtype=np.float32,
            shape=(n_train, seq_len, shapes['lr_10m'][1], shapes['lr_10m'][2], shapes['lr_10m'][3])
        ),
        'lr_20m': np.lib.format.open_memmap(
            str(train_dir / 'lr_20m.npy'), mode='w+', dtype=np.float32,
            shape=(n_train, seq_len, shapes['lr_20m'][1], shapes['lr_20m'][2], shapes['lr_20m'][3])
        ),
        'hr_10m': np.lib.format.open_memmap(
            str(train_dir / 'hr_10m.npy'), mode='w+', dtype=np.float32,
            shape=(n_train, seq_len, shapes['hr_10m'][1], shapes['hr_10m'][2], shapes['hr_10m'][3])
        ),
        'hr_20m': np.lib.format.open_memmap(
            str(train_dir / 'hr_20m.npy'), mode='w+', dtype=np.float32,
            shape=(n_train, seq_len, shapes['hr_20m'][1], shapes['hr_20m'][2], shapes['hr_20m'][3])
        ),
    }
    
    # Create mmap arrays for test
    test_arrays = {
        'lr': np.lib.format.open_memmap(
            str(test_dir / 'lr.npy'), mode='w+', dtype=np.float32,
            shape=(n_test, seq_len, shapes['lr_10m'][1], shapes['lr_10m'][2], shapes['lr_10m'][3])
        ),
        'lr_20m': np.lib.format.open_memmap(
            str(test_dir / 'lr_20m.npy'), mode='w+', dtype=np.float32,
            shape=(n_test, seq_len, shapes['lr_20m'][1], shapes['lr_20m'][2], shapes['lr_20m'][3])
        ),
        'hr_10m': np.lib.format.open_memmap(
            str(test_dir / 'hr_10m.npy'), mode='w+', dtype=np.float32,
            shape=(n_test, seq_len, shapes['hr_10m'][1], shapes['hr_10m'][2], shapes['hr_10m'][3])
        ),
        'hr_20m': np.lib.format.open_memmap(
            str(test_dir / 'hr_20m.npy'), mode='w+', dtype=np.float32,
            shape=(n_test, seq_len, shapes['hr_20m'][1], shapes['hr_20m'][2], shapes['hr_20m'][3])
        ),
    }
    
    print(f"   Created mmap files")
    print(f"   LR shape: {train_arrays['lr'].shape}")
    print(f"   HR_10m shape: {train_arrays['hr_10m'].shape}")
    
    # ========== PHASE 4: Stream data into mmap files ==========
    print("\n📥 PHASE 4: Streaming data to mmap files...")
    
    global_idx = 0
    train_idx = 0
    test_idx = 0
    
    for ds_info in dataset_info:
        print(f"\n   Processing {ds_info['sta']} {ds_info['year']}...")
        
        for chunk_file_str in ds_info['chunk_files']:
            chunk_file = Path(chunk_file_str)
            chunk_name = chunk_file.name
            
            sample_count = 0
            # Define target shapes for resampling (without batch dimension)
            target_shapes = {
                'lr_10m': (shapes['lr_10m'][0], shapes['lr_10m'][1], shapes['lr_10m'][2], shapes['lr_10m'][3]),
                'lr_20m': (shapes['lr_20m'][0], shapes['lr_20m'][1], shapes['lr_20m'][2], shapes['lr_20m'][3]),
                'hr_10m': (shapes['hr_10m'][0], shapes['hr_10m'][1], shapes['hr_10m'][2], shapes['hr_10m'][3]),
                'hr_20m': (shapes['hr_20m'][0], shapes['hr_20m'][1], shapes['hr_20m'][2], shapes['hr_20m'][3]),
            }
            
            for sample in stream_samples_from_file(chunk_file):
                arrays = extract_arrays(sample, target_shapes)
                if arrays is None:
                    global_idx += 1
                    continue
                
                # Determine if this goes to train or test
                if global_idx in test_indices:
                    # Write to test
                    if test_idx < n_test:
                        test_arrays['lr'][test_idx] = arrays['lr_10m']
                        test_arrays['lr_20m'][test_idx] = arrays['lr_20m']
                        test_arrays['hr_10m'][test_idx] = arrays['hr_10m']
                        test_arrays['hr_20m'][test_idx] = arrays['hr_20m']
                        test_idx += 1
                else:
                    # Write to train
                    if train_idx < n_train:
                        train_arrays['lr'][train_idx] = arrays['lr_10m']
                        train_arrays['lr_20m'][train_idx] = arrays['lr_20m']
                        train_arrays['hr_10m'][train_idx] = arrays['hr_10m']
                        train_arrays['hr_20m'][train_idx] = arrays['hr_20m']
                        train_idx += 1
                
                global_idx += 1
                sample_count += 1
            
            print(f"      {chunk_name}: {sample_count} samples (train: {train_idx}, test: {test_idx})")
            gc.collect()
    
    # Flush mmap files
    for arr in train_arrays.values():
        arr.flush()
    for arr in test_arrays.values():
        arr.flush()
    
    # ========== PHASE 5: Save metadata ==========
    print("\n📝 PHASE 5: Saving metadata...")
    
    metadata = {
        'total_samples': total_samples,
        'train_samples': train_idx,
        'test_samples': test_idx,
        'test_split': args.test_split,
        'seed': args.seed,
        'shapes': {k: list(v) for k, v in shapes.items()},
        'datasets': dataset_info,
        'arrays': {
            'train': {
                'lr': str(train_dir / 'lr.npy'),
                'lr_20m': str(train_dir / 'lr_20m.npy'),
                'hr_10m': str(train_dir / 'hr_10m.npy'),
                'hr_20m': str(train_dir / 'hr_20m.npy'),
            },
            'test': {
                'lr': str(test_dir / 'lr.npy'),
                'lr_20m': str(test_dir / 'lr_20m.npy'),
                'hr_10m': str(test_dir / 'hr_10m.npy'),
                'hr_20m': str(test_dir / 'hr_20m.npy'),
            }
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Training samples: {train_idx}")
    print(f"Test samples: {test_idx}")
    print(f"\nFiles created:")
    print(f"  - {train_dir}/lr.npy, lr_20m.npy, hr_10m.npy, hr_20m.npy")
    print(f"  - {test_dir}/lr.npy, lr_20m.npy, hr_10m.npy, hr_20m.npy")
    print(f"  - {output_dir}/metadata.json")


if __name__ == '__main__':
    main()
