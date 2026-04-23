#!/usr/bin/env python3
"""
ULTRA-FAST dataset generation - downloads ONLY tile regions, not full images
Strategy: Download small 256×256 HR + 77×77 LR tiles directly from GEE
No need to download 3800×5000 full images!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
from pathlib import Path
import numpy as np
import ee
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from Auth.AuthGEE import find_pairs, PLANET_COLLECTION_ID, regions
from scripts.preprocessing.numpy_efficient_pipeline import (
    TrainingSample, coregister_lr_to_hr, harmonize_planet_to_sentinel2
)

def download_tile_region(planet_img, s2_img, tile_bounds, planet_bands, sentinel_bands, 
                         hr_scale=3, lr_scale=10):
    """
    Download a small tile region directly (no full image download)
    
    Args:
        planet_img: ee.Image for Planet
        s2_img: ee.Image for Sentinel-2
        tile_bounds: [min_lon, min_lat, max_lon, max_lat]
        planet_bands: List of band names
        sentinel_bands: List of band names
        hr_scale: Planet resolution (3m)
        lr_scale: Sentinel-2 resolution (10m)
    
    Returns:
        (hr_array, lr_array) or (None, None) if failed
    """
    try:
        tile_geom = ee.Geometry.Rectangle(tile_bounds)
        
        # Download HR tile
        hr_selected = planet_img.select(planet_bands).clip(tile_geom).unmask(0)
        hr_resampled = hr_selected.reproject(crs='EPSG:4326', scale=hr_scale)
        hr_rect = hr_resampled.sampleRectangle(region=tile_geom, defaultValue=0)
        hr_info = hr_rect.getInfo()
        
        if not hr_info or 'properties' not in hr_info:
            return None, None
        
        hr_props = hr_info['properties']
        hr_arrays = [np.array(hr_props[b], dtype=np.float32) for b in planet_bands]
        
        # Check shapes and ensure consistency
        hr_shapes = [arr.shape for arr in hr_arrays]
        if len(set(hr_shapes)) > 1:
            print(f"\n      ⚠️ HR band shape mismatch: {hr_shapes}")
            return None, None
        
        hr_array = np.stack(hr_arrays, axis=0)  # (C, H, W)
        hr_array = np.transpose(hr_array, (1, 2, 0))  # (H, W, C)
        
        # Download LR tile
        # IMPORTANT: Sentinel-2 has mixed resolutions (10m + 20m bands)
        # Resample all to 10m for consistent dimensions
        lr_selected = s2_img.select(sentinel_bands).clip(tile_geom).unmask(0)
        lr_resampled = lr_selected.reproject(crs='EPSG:4326', scale=lr_scale)
        lr_rect = lr_resampled.sampleRectangle(region=tile_geom, defaultValue=0)
        lr_info = lr_rect.getInfo()
        
        if not lr_info or 'properties' not in lr_info:
            return None, None
        
        lr_props = lr_info['properties']
        lr_arrays = [np.array(lr_props[b], dtype=np.float32) for b in sentinel_bands]
        
        # Check shapes and ensure consistency
        lr_shapes = [arr.shape for arr in lr_arrays]
        if len(set(lr_shapes)) > 1:
            print(f"\n      ⚠️ LR band shape mismatch: {lr_shapes}")
            return None, None
        
        lr_array = np.stack(lr_arrays, axis=0)  # (C, H, W)
        lr_array = np.transpose(lr_array, (1, 2, 0))  # (H, W, C)
        
        return hr_array, lr_array
        
    except Exception as e:
        print(f"\n      ⚠️ Tile download failed: {e}")
        return None, None

def generate_tile_bounds(aoi, num_tiles=10):
    """
    Generate random tile bounds within AOI
    
    Args:
        aoi: ee.Geometry
        num_tiles: Number of tiles to generate
    
    Returns:
        List of [min_lon, min_lat, max_lon, max_lat]
    """
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    # Tile size in degrees (approximate)
    # 256 pixels × 3m = 768m ≈ 0.007 degrees
    tile_size_deg = 0.007
    
    # Generate random tiles within bounds
    import random
    tiles = []
    for i in range(num_tiles):
        # Random position within AOI
        tile_min_lon = random.uniform(min_lon, max_lon - tile_size_deg)
        tile_min_lat = random.uniform(min_lat, max_lat - tile_size_deg)
        tiles.append([
            tile_min_lon,
            tile_min_lat,
            tile_min_lon + tile_size_deg,
            tile_min_lat + tile_size_deg
        ])
    
    return tiles

def main():
    parser = argparse.ArgumentParser(description='Ultra-fast dataset generation')
    parser.add_argument('--max-samples', type=int, default=20,
                       help='Maximum number of samples to generate')
    parser.add_argument('--output-dir', type=str, 
                       default='./data/datasets/sta2_ultrafast',
                       help='Output directory for dataset')
    parser.add_argument('--cache-dir', type=str,
                       default='./cache/sta2_ultrafast',
                       help='Cache directory')
    parser.add_argument('--sequence-length', type=int, default=3,
                       help='Temporal sequence length')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"⚡ ULTRA-FAST DATASET GENERATION")
    print("=" * 80)
    print(f"Strategy: Download ONLY tile regions (not full images)")
    print(f"Target: {args.max_samples} samples")
    print(f"Expected time: 5-10 minutes")
    print("=" * 80)
    
    start_time = time.time()
    
    # Initialize GEE
    try:
        ee.Initialize()
    except:
        print("Initializing GEE...")
        from Auth.AuthGEE import ee
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Use only first region (STA-2)
    region_config = regions[0]
    print(f"\n📍 Region: {region_config['name']}")
    
    # Find image pairs
    print(f"\n🔍 Finding image pairs...")
    pair_results = find_pairs(
        region_config["aoi"],
        region_config["start"],
        region_config["end"],
        PLANET_COLLECTION_ID
    )
    
    matched_pairs = pair_results.get('matched_pairs', [])
    s2_collection = pair_results.get('s2_collection')
    planet_collection = pair_results.get('planet_collection')
    
    print(f"   Found {len(matched_pairs)} matched pairs")
    
    if len(matched_pairs) < args.sequence_length:
        print(f"   ❌ Insufficient pairs for sequence length {args.sequence_length}")
        return
    
    # Create ONE sequence (reuse for multiple tiles)
    sequence = matched_pairs[:args.sequence_length]
    print(f"\n📅 Using sequence: {[p['s2_date'] for p in sequence]}")
    
    # Generate random tile bounds
    tiles_per_sequence = args.max_samples // 1  # All from one sequence
    print(f"\n🗺️  Generating {tiles_per_sequence} random tiles...")
    tile_bounds_list = generate_tile_bounds(region_config["aoi"], tiles_per_sequence)
    
    # Band configuration (use 4-band for simplicity)
    planet_bands = ['B1', 'B2', 'B3', 'B4']
    sentinel_bands = ['B2', 'B3', 'B4', 'B8A']
    
    samples = []
    
    print(f"\n⚡ Downloading {args.max_samples} samples...")
    print(f"   Each sample: 3 frames × 2 images = 6 small tile downloads")
    
    for tile_idx, tile_bounds in enumerate(tile_bounds_list):
        print(f"\n   Sample {tile_idx+1}/{tiles_per_sequence}:")
        
        lr_frames = []
        hr_frames = []
        dates = []
        
        # Download all frames for this tile
        for frame_idx, pair in enumerate(sequence):
            print(f"      Frame {frame_idx+1}/3...", end=" ", flush=True)
            
            try:
                # Get images from collections
                planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
                s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
                
                # Download just this tile region
                hr_array, lr_array = download_tile_region(
                    planet_img, s2_img, tile_bounds,
                    planet_bands, sentinel_bands
                )
                
                if hr_array is None or lr_array is None:
                    print("❌ Failed")
                    break
                
                print(f"✓ HR:{hr_array.shape} LR:{lr_array.shape}")
                
                # Apply coregistration and harmonization
                cache_path_coreg = cache_dir / "coregistered" / f"tile{tile_idx}_frame{frame_idx}.npy"
                cache_path_harm = cache_dir / "harmonized" / f"tile{tile_idx}_frame{frame_idx}.npy"
                
                # Coregistration
                hr_array = coregister_lr_to_hr(
                    hr_array=hr_array,
                    lr_array=lr_array,
                    aoi_bounds=None,
                    cache_path=cache_path_coreg
                )
                
                # Harmonization
                hr_array = harmonize_planet_to_sentinel2(
                    hr_array=hr_array,
                    lr_array=lr_array,
                    cache_path=cache_path_harm
                )
                
                # Ensure matching bands
                if lr_array.shape[-1] != hr_array.shape[-1]:
                    min_bands = min(lr_array.shape[-1], hr_array.shape[-1])
                    lr_array = lr_array[:, :, :min_bands]
                    hr_array = hr_array[:, :, :min_bands]
                
                lr_frames.append(lr_array)
                hr_frames.append(hr_array)
                dates.append(pair['s2_date'])
                
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        
        # Create sample if all frames downloaded
        if len(lr_frames) == len(sequence):
            sample = TrainingSample(
                lr_sequence=np.stack(lr_frames, axis=0),
                hr_sequence=np.stack(hr_frames, axis=0),
                temporal_dates=dates,
                region_name=region_config['name'],
                sequence_id=0,
                tile_id=tile_idx,
                tile_bounds={'bounds': tile_bounds}
            )
            samples.append(sample)
            print(f"      ✅ Sample complete")
        else:
            print(f"      ⚠️ Incomplete sample, skipping")
        
        if len(samples) >= args.max_samples:
            break
    
    # Save dataset
    if len(samples) == 0:
        print("\n❌ No samples generated!")
        return
    
    dataset_path = output_dir / 'dataset.pkl'
    print(f"\n💾 Saving {len(samples)} samples...")
    
    dataset = {
        'samples': samples,
        'config': {
            'region': region_config['name'],
            'sequence_length': args.sequence_length,
            'num_samples': len(samples)
        }
    }
    
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=4)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✅ ULTRA-FAST GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Samples generated: {len(samples)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Avg time per sample: {elapsed/len(samples):.1f} seconds")
    print(f"\nDataset saved: {dataset_path}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
