"""
Create training dataset using the validated chunked download pipeline
Uses numpy_efficient_pipeline with parallel chunked downloads
"""

import os
import sys
import certifi
import ssl

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import pickle
import numpy as np
from pathlib import Path
from typing import List
from dataclasses import dataclass

from numpy_efficient_pipeline import TensorFlowFreeDataPipeline, TrainingSample
from Auth.AuthGEE import regions


def create_chunked_dataset(
    output_dir="./chunked_sr_dataset",
    num_samples_per_region=50,
    tile_size=256,
    sequence_length=3,
    target_regions=None,
    cache_dir=None
):
    """
    Create SR training dataset using chunked downloads
    
    Args:
        output_dir: Where to save the dataset
        num_samples_per_region: How many samples to generate per region
        tile_size: Tile size (256 = good balance of quality/speed)
        sequence_length: Temporal sequence length (3 = attention mechanism)
        target_regions: List of region indices to process (None = all)
        cache_dir: Custom cache directory (None = use output_dir/cache)
    
    Returns:
        Path to created dataset
    """
    
    print("="*70)
    print("🚀 CREATING SR DATASET WITH CHUNKED DOWNLOADS")
    print("="*70)
    print(f"Output: {output_dir}")
    print(f"Samples per region: {num_samples_per_region}")
    print(f"Tile size: {tile_size}×{tile_size}")
    print(f"Sequence length: {sequence_length} frames")
    print("="*70 + "\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Use custom cache dir or default
    if cache_dir is None:
        cache_dir = str(output_path / "cache")
    
    # Initialize pipeline with chunked downloads
    pipeline = TensorFlowFreeDataPipeline(
        cache_dir=cache_dir,
        tile_size=tile_size,
        overlap=64
    )
    
    # Select regions to process
    if target_regions is not None:
        regions_to_process = [regions[i] for i in target_regions]
    else:
        regions_to_process = regions
    
    # Collect samples from all regions
    all_samples = []
    metadata = {
        'regions': [],
        'total_samples': 0,
        'tile_size': tile_size,
        'sequence_length': sequence_length,
        'lr_shape': None,
        'hr_shape': None,
        'creation_date': None
    }
    
    for region_idx, region in enumerate(regions_to_process):
        print(f"\n{'='*70}")
        print(f"REGION {region_idx+1}/{len(regions_to_process)}: {region['name']}")
        print(f"{'='*70}\n")
        
        region_samples = []
        
        # Process region (generator yields samples as created)
        print(f"Processing region with chunked downloads...")
        sample_generator = pipeline.process_region_numpy(
            region,
            sequence_length=sequence_length,
            use_cache=True
        )
        
        # Collect samples up to limit
        for sample_idx, sample in enumerate(sample_generator):
            if sample_idx >= num_samples_per_region:
                break
            
            region_samples.append(sample)
            
            if (sample_idx + 1) % 10 == 0:
                print(f"   Collected {sample_idx + 1}/{num_samples_per_region} samples...")
        
        print(f"\n✅ Region {region['name']}: {len(region_samples)} samples collected")
        
        # Store region info
        metadata['regions'].append({
            'name': region['name'],
            'sample_count': len(region_samples),
            'start_date': region['start'],
            'end_date': region['end']
        })
        
        all_samples.extend(region_samples)
        
        # Update metadata with shapes from first sample
        if len(region_samples) > 0 and metadata['lr_shape'] is None:
            first_sample = region_samples[0]
            metadata['lr_shape'] = first_sample.lr_sequence.shape
            metadata['hr_shape'] = first_sample.hr_sequence.shape
            print(f"   LR shape: {metadata['lr_shape']}")
            print(f"   HR shape: {metadata['hr_shape']}")
    
    # Final metadata
    metadata['total_samples'] = len(all_samples)
    
    import datetime
    metadata['creation_date'] = datetime.datetime.now().isoformat()
    
    # Save dataset
    print(f"\n{'='*70}")
    print("SAVING DATASET")
    print(f"{'='*70}\n")
    
    samples_file = output_path / "training_samples.pkl"
    metadata_file = output_path / "metadata.pkl"
    
    print(f"💾 Saving {len(all_samples)} samples to {samples_file}...")
    with open(samples_file, 'wb') as f:
        pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"💾 Saving metadata to {metadata_file}...")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Print summary
    print(f"\n{'='*70}")
    print("✅ DATASET CREATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total samples: {len(all_samples)}")
    print(f"LR shape: {metadata['lr_shape']}")
    print(f"HR shape: {metadata['hr_shape']}")
    print(f"Regions processed: {len(metadata['regions'])}")
    for region_info in metadata['regions']:
        print(f"  - {region_info['name']}: {region_info['sample_count']} samples")
    print(f"\nDataset location: {output_dir}")
    print(f"{'='*70}\n")
    
    return output_path


def load_chunked_dataset(dataset_dir="./chunked_sr_dataset"):
    """
    Load the dataset created with chunked downloads
    Compatible with existing training scripts
    """
    dataset_path = Path(dataset_dir)
    
    samples_file = dataset_path / "training_samples.pkl"
    metadata_file = dataset_path / "metadata.pkl"
    
    if not samples_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {samples_file}\n"
            f"Run create_chunked_dataset() first!"
        )
    
    print(f"📂 Loading chunked dataset from {dataset_dir}...")
    
    with open(samples_file, 'rb') as f:
        samples = pickle.load(f)
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"   ✅ Loaded {len(samples)} samples")
    print(f"   LR shape: {metadata['lr_shape']}")
    print(f"   HR shape: {metadata['hr_shape']}")
    print(f"   Created: {metadata['creation_date']}")
    
    return samples, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create SR dataset with chunked downloads')
    parser.add_argument('--output-dir', default='./chunked_sr_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--cache-dir', default=None,
                       help='Cache directory (default: output-dir/cache)')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of samples per region')
    parser.add_argument('--tile-size', type=int, default=256,
                       help='Tile size (256 or 512)')
    parser.add_argument('--sequence-length', type=int, default=3,
                       help='Temporal sequence length (1-5)')
    parser.add_argument('--regions', type=int, nargs='+',
                       help='Region indices to process (e.g., 0 1 2)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with 2 samples from first region')
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("\n🧪 QUICK TEST MODE: 2 samples from first region\n")
        create_chunked_dataset(
            output_dir="./quick_test_dataset",
            num_samples_per_region=2,
            tile_size=256,
            sequence_length=1,
            target_regions=[0],
            cache_dir=args.cache_dir
        )
    else:
        create_chunked_dataset(
            output_dir=args.output_dir,
            num_samples_per_region=args.samples,
            tile_size=args.tile_size,
            sequence_length=args.sequence_length,
            target_regions=args.regions,
            cache_dir=args.cache_dir
        )
    
    print("\n💡 Next steps:")
    print("   1. Verify dataset quality:")
    print("      python validate_chunked_dataset.py")
    print("   2. Train SR model:")
    print("      python train_with_chunked_dataset.py")
    print("   3. For larger dataset, increase --samples:")
    print("      python create_chunked_dataset.py --samples 200")
