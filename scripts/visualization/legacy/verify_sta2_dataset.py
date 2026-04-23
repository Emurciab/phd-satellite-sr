#!/usr/bin/env python3
"""
Verify the quality of generated STA-2 dataset samples.
"""

import os
import sys
import pickle
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'scripts', 'preprocessing'))

def verify_dataset(dataset_path):
    """Verify dataset quality."""
    
    print('=' * 80)
    print('📊 LOADING STA-2 DATASET')
    print('=' * 80)
    
    # Load metadata
    metadata_path = os.path.join(dataset_path, 'metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print('\n📋 METADATA:')
    print(f"  Total samples: {metadata['total_samples']}")
    print(f"  Tile size: {metadata['tile_size']}")
    print(f"  Sequence length: {metadata['sequence_length']}")
    print(f"  LR shape: {metadata['lr_shape']}")
    print(f"  HR shape: {metadata['hr_shape']}")
    print(f"  Creation date: {metadata['creation_date']}")
    
    # Load samples
    samples_path = os.path.join(dataset_path, 'training_samples.pkl')
    file_size = os.path.getsize(samples_path) / (1024**3)
    print(f'\nLoading training_samples.pkl ({file_size:.1f} GB)...')
    
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)
    
    print(f'✅ Loaded {len(samples)} samples\n')
    
    print('=' * 80)
    print('📈 DATASET STATISTICS')
    print('=' * 80)
    
    # Check first sample structure
    sample = samples[0]
    print(f'Sample structure: {type(sample)}')
    print(f'Sample type name: {type(sample).__name__}')
    
    # Handle both dict and TrainingSample objects
    if hasattr(sample, '__dict__'):
        attrs = list(vars(sample).keys())
        print(f'Attributes: {attrs}')
        
        # Use lr_sequence/hr_sequence if available, else lr/hr
        if hasattr(sample, 'lr_sequence'):
            lr = sample.lr_sequence
            hr = sample.hr_sequence
        elif hasattr(sample, 'lr'):
            lr = sample.lr
            hr = sample.hr
        else:
            print(f"Unknown attribute structure!")
            return
    elif isinstance(sample, dict):
        print(f'Keys: {list(sample.keys())}')
        lr = sample['lr']
        hr = sample['hr']
    else:
        print(f"Unknown sample type: {type(sample)}")
        return
    
    print(f'\nLR shape: {lr.shape}')
    print(f'HR shape: {hr.shape}')
    print(f'LR dtype: {lr.dtype}')
    print(f'HR dtype: {hr.dtype}')
    
    # Calculate actual tiling ratio
    ratio = hr.shape[1] / lr.shape[1]
    print(f'\n📐 Tiling ratio: {ratio:.3f}:1 (target: 3.333:1)')
    error = abs(ratio - 3.333) / 3.333 * 100
    print(f'   Error: {error:.2f}%')
    
    # Check for metadata in samples
    if hasattr(sample, 'metadata'):
        meta = sample.metadata
        print(f'\nSample metadata keys: {list(meta.keys()) if isinstance(meta, dict) else list(vars(meta).keys())}')
    elif isinstance(sample, dict) and 'metadata' in sample:
        print(f'\nSample metadata keys: {list(sample["metadata"].keys())}')
    
    print('\n' + '=' * 80)
    print('🔍 SAMPLE QUALITY CHECK (First 10 samples)')
    print('=' * 80)
    
    for i in range(min(10, len(samples))):
        s = samples[i]
        # Handle both object and dict access
        if hasattr(s, 'lr_sequence'):
            lr_data = s.lr_sequence
            hr_data = s.hr_sequence
        elif hasattr(s, 'lr'):
            lr_data = s.lr
            hr_data = s.hr
        else:
            lr_data = s['lr']
            hr_data = s['hr']
        
        # Check for zero/nan values
        lr_zeros = np.sum(lr_data == 0) / lr_data.size * 100
        hr_zeros = np.sum(hr_data == 0) / hr_data.size * 100
        lr_nans = np.sum(np.isnan(lr_data))
        hr_nans = np.sum(np.isnan(hr_data))
        
        # Check value ranges
        lr_min, lr_max = np.min(lr_data), np.max(lr_data)
        hr_min, hr_max = np.min(hr_data), np.max(hr_data)
        
        # Quality assessment
        quality = '✅ GOOD'
        if lr_zeros > 50 or hr_zeros > 50:
            quality = '⚠️  HIGH ZEROS'
        if lr_nans > 0 or hr_nans > 0:
            quality = '❌ HAS NaNs'
        if lr_max == 0 or hr_max == 0:
            quality = '❌ ALL ZEROS'
        
        print(f'Sample {i:3d}: {quality}')
        print(f'  LR: [{lr_min:.4f}, {lr_max:.4f}], zeros: {lr_zeros:.1f}%, shape: {lr_data.shape}')
        print(f'  HR: [{hr_min:.4f}, {hr_max:.4f}], zeros: {hr_zeros:.1f}%, shape: {hr_data.shape}')
    
    # Overall statistics
    print('\n' + '=' * 80)
    print('📊 OVERALL QUALITY SUMMARY (checking all samples...)')
    print('=' * 80)
    
    good_samples = 0
    zero_samples = 0
    nan_samples = 0
    
    for i, s in enumerate(samples):
        if i % 200 == 0:
            print(f'  Checking sample {i}/{len(samples)}...', flush=True)
        
        # Handle both object and dict access
        if hasattr(s, 'lr_sequence'):
            lr_data = s.lr_sequence
            hr_data = s.hr_sequence
        elif hasattr(s, 'lr'):
            lr_data = s.lr
            hr_data = s.hr
        else:
            lr_data = s['lr']
            hr_data = s['hr']
        
        has_nans = np.sum(np.isnan(lr_data)) > 0 or np.sum(np.isnan(hr_data)) > 0
        high_zeros = (np.sum(lr_data == 0) / lr_data.size > 0.5) or (np.sum(hr_data == 0) / hr_data.size > 0.5)
        all_zeros = np.max(lr_data) == 0 or np.max(hr_data) == 0
        
        if has_nans:
            nan_samples += 1
        elif high_zeros or all_zeros:
            zero_samples += 1
        else:
            good_samples += 1
    
    print(f'\n✅ VERIFICATION COMPLETE\n')
    print(f'Total samples: {len(samples)}')
    print(f'✅ Good quality: {good_samples} ({good_samples/len(samples)*100:.1f}%)')
    print(f'⚠️  High zeros: {zero_samples} ({zero_samples/len(samples)*100:.1f}%)')
    print(f'❌ Has NaNs: {nan_samples} ({nan_samples/len(samples)*100:.1f}%)')
    
    # Calculate mean and std for normalization reference
    print('\n' + '=' * 80)
    print('📊 NORMALIZATION STATISTICS (random 100 samples)')
    print('=' * 80)
    
    # Sample 100 random samples for statistics
    indices = np.random.choice(len(samples), min(100, len(samples)), replace=False)
    lr_values = []
    hr_values = []
    
    for idx in indices:
        s = samples[idx]
        if hasattr(s, 'lr_sequence'):
            lr_data = s.lr_sequence
            hr_data = s.hr_sequence
        elif hasattr(s, 'lr'):
            lr_data = s.lr
            hr_data = s.hr
        else:
            lr_data = s['lr']
            hr_data = s['hr']
        
        lr_values.append(lr_data.flatten())
        hr_values.append(hr_data.flatten())
    
    lr_all = np.concatenate(lr_values)
    hr_all = np.concatenate(hr_values)
    
    print(f'LR - Mean: {np.mean(lr_all):.4f}, Std: {np.std(lr_all):.4f}')
    print(f'LR - Min: {np.min(lr_all):.4f}, Max: {np.max(lr_all):.4f}')
    print(f'HR - Mean: {np.mean(hr_all):.4f}, Std: {np.std(hr_all):.4f}')
    print(f'HR - Min: {np.min(hr_all):.4f}, Max: {np.max(hr_all):.4f}')
        
    print('=' * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify STA-2 dataset quality')
    parser.add_argument('--dataset', type=str, 
                       default='./data/datasets/sta2_test',
                       help='Path to dataset directory')
    
    args = parser.parse_args()
    
    verify_dataset(args.dataset)
