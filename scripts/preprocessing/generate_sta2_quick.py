#!/usr/bin/env python3
"""
QUICK dataset generation for testing - generates N samples FAST
Uses smart strategies to minimize download time:
1. Reuses same sequence for multiple tiles (download once, tile many times)
2. Limits to first few sequences only
3. Parallel tile processing from cached sequence
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
from pathlib import Path
import numpy as np

# Import the pipeline
from scripts.preprocessing.numpy_efficient_pipeline import TensorFlowFreeDataPipeline, TrainingSample
from Auth.AuthGEE import regions

def main():
    parser = argparse.ArgumentParser(description='Quick dataset generation for testing')
    parser.add_argument('--max-samples', type=int, default=20,
                       help='Maximum number of samples to generate')
    parser.add_argument('--output-dir', type=str, 
                       default='./data/datasets/sta2_quick',
                       help='Output directory for dataset')
    parser.add_argument('--cache-dir', type=str,
                       default='./cache/sta2_quick',
                       help='Cache directory')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear cache before generation')
    parser.add_argument('--tile-size', type=int, default=256,
                       help='HR tile size')
    parser.add_argument('--sequence-length', type=int, default=3,
                       help='Temporal sequence length')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear cache if requested
    cache_dir = Path(args.cache_dir)
    if args.clear_cache and cache_dir.exists():
        import shutil
        print(f"🗑️  Clearing cache: {cache_dir}")
        shutil.rmtree(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"🚀 QUICK DATASET GENERATION")
    print("=" * 80)
    print(f"Target samples: {args.max_samples}")
    print(f"Strategy: Download minimal sequences, maximize tiles per sequence")
    print(f"Output: {output_dir}")
    print(f"Cache: {cache_dir}")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = TensorFlowFreeDataPipeline(
        cache_dir=str(cache_dir),
        tile_size=args.tile_size,
        overlap=0,  # No overlap for quick generation
        max_cache_gb=5.0
    )
    
    # Use only first region (STA-2)
    region_config = regions[0]
    print(f"\n📍 Processing region: {region_config['name']}")
    
    # STRATEGY: 
    # - Download only 1-2 sequences (minimize download time)
    # - Extract MANY tiles from each sequence to reach target sample count
    # - Each sequence gives us multiple tiles (spatial variation)
    
    samples = []
    samples_needed = args.max_samples
    
    # Calculate tiles needed per sequence
    # Estimate: each sequence can give us ~10-15 tiles from full AOI
    # So for 20 samples, we need 2-3 sequences max
    
    print(f"\n📊 Generating samples...")
    print(f"   Target: {samples_needed} samples")
    print(f"   Strategy: Download few sequences, extract many tiles per sequence")
    
    sample_generator = pipeline.process_region_numpy(
        region_config=region_config,
        sequence_length=args.sequence_length,
        use_cache=True
    )
    
    # Collect samples
    for sample in sample_generator:
        samples.append(sample)
        print(f"   ✅ Sample {len(samples)}/{samples_needed}: "
              f"seq={sample.sequence_id}, tile={sample.tile_id}, "
              f"LR={sample.lr_sequence.shape}, HR={sample.hr_sequence.shape}")
        
        if len(samples) >= samples_needed:
            print(f"\n🎯 Reached target of {samples_needed} samples!")
            break
    
    if len(samples) == 0:
        print("\n❌ No samples generated!")
        return
    
    # Save dataset
    dataset_path = output_dir / 'dataset.pkl'
    print(f"\n💾 Saving {len(samples)} samples to {dataset_path}...")
    
    dataset = {
        'samples': samples,
        'config': {
            'region': region_config['name'],
            'tile_size': args.tile_size,
            'sequence_length': args.sequence_length,
            'num_samples': len(samples)
        }
    }
    
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=4)
    
    print(f"   ✅ Dataset saved: {dataset_path}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("📊 GENERATION STATISTICS")
    print("=" * 80)
    print(f"Total samples: {len(samples)}")
    print(f"LR shape: {samples[0].lr_sequence.shape}")
    print(f"HR shape: {samples[0].hr_sequence.shape}")
    print(f"Temporal length: {len(samples[0].temporal_dates)}")
    
    # Count unique sequences and tiles
    unique_seqs = len(set(s.sequence_id for s in samples))
    unique_tiles = len(set(s.tile_id for s in samples))
    print(f"Unique sequences: {unique_seqs}")
    print(f"Unique tiles: {unique_tiles}")
    print(f"Samples per sequence: {len(samples) / unique_seqs:.1f}")
    
    pipeline.print_stats()
    
    print("\n" + "=" * 80)
    print("✅ QUICK DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nNext step: Visualize with:")
    print(f"  python visualize_20samples.py --dataset-path {dataset_path}")

if __name__ == "__main__":
    main()
