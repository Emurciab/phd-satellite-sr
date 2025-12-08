#!/usr/bin/env python3
"""
Convert existing single-resolution dataset to multi-resolution format.

This script:
1. Loads an existing dataset with B8A interpolated to 10m
2. Downloads only the native 20m B8A data for each tile
3. Creates new multi-resolution samples with:
   - lr_10m: (T, H, W, 3) - B2, B3, B4 from existing data
   - lr_20m: (T, H/2, W/2, 1) - B8A at native 20m (NEW download)
   - hr_10m: (T, H', W', 3) - Planet RGB from existing data  
   - hr_20m: (T, H'/2, W'/2, 1) - Planet NIR downsampled

This is MUCH faster than regenerating everything since we only download 1 band
instead of all bands for each tile.
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
from dataclasses import dataclass
from typing import List, Optional, Dict
import hashlib

from Auth.AuthGEE import find_pairs, PLANET_COLLECTION_ID


@dataclass
class TrainingSample:
    """Training sample with multi-resolution support"""
    # Legacy fields
    lr_sequence: Optional[np.ndarray] = None
    hr_sequence: Optional[np.ndarray] = None
    # Multi-resolution fields
    lr_10m: Optional[np.ndarray] = None
    lr_20m: Optional[np.ndarray] = None
    hr_10m: Optional[np.ndarray] = None
    hr_20m: Optional[np.ndarray] = None
    # Metadata
    temporal_dates: List[str] = None
    planet_dates: List[str] = None
    region_name: str = ""
    sequence_id: int = 0
    tile_id: int = 0
    tile_bounds: Dict = None
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None
    band_mode: str = '4-band'
    is_multi_resolution: bool = False


def download_b8a_20m(s2_id: str, tile_bounds: list, s2_collection) -> Optional[np.ndarray]:
    """
    Download only B8A at native 20m resolution.
    
    Args:
        s2_id: Sentinel-2 image ID (id_pref format)
        tile_bounds: [min_lon, min_lat, max_lon, max_lat]
        s2_collection: ee.ImageCollection for Sentinel-2
    
    Returns:
        np.ndarray of shape (H/2, W/2, 1) or None if failed
    """
    try:
        tile_geom = ee.Geometry.Rectangle(tile_bounds)
        s2_img = s2_collection.filter(ee.Filter.eq("id_pref", s2_id)).first()
        
        # Download B8A at native 20m
        b8a_selected = s2_img.select(['B8A']).clip(tile_geom).unmask(0)
        b8a_resampled = b8a_selected.reproject(crs='EPSG:4326', scale=20)
        b8a_rect = b8a_resampled.sampleRectangle(region=tile_geom, defaultValue=0)
        b8a_info = b8a_rect.getInfo()
        
        if not b8a_info or 'properties' not in b8a_info:
            return None
        
        b8a_array = np.array(b8a_info['properties']['B8A'], dtype=np.float32)
        return b8a_array[:, :, np.newaxis]  # (H/2, W/2, 1)
        
    except Exception as e:
        print(f"      Error downloading B8A: {e}")
        return None


def convert_sample_to_multi_res(
    sample,
    s2_collection,
    cache_dir: Path
) -> Optional[TrainingSample]:
    """
    Convert a single-resolution sample to multi-resolution format.
    
    Args:
        sample: Original TrainingSample with lr_sequence, hr_sequence
        s2_collection: ee.ImageCollection for downloading 20m data
        cache_dir: Directory for caching downloads
    
    Returns:
        New TrainingSample with multi-resolution fields
    """
    # Extract existing data
    lr_seq = sample.lr_sequence  # (T, H, W, C) where C includes B8A interpolated
    hr_seq = sample.hr_sequence  # (T, H', W', C)
    
    if lr_seq is None or hr_seq is None:
        return None
    
    T, H, W, C = lr_seq.shape
    _, H_hr, W_hr, C_hr = hr_seq.shape
    
    # Extract RGB from LR (B2, B3, B4 - first 3 bands)
    lr_10m = lr_seq[:, :, :, :3]  # (T, H, W, 3)
    
    # Extract RGB from HR (B1, B2, B3 - first 3 bands of Planet)
    hr_10m = hr_seq[:, :, :, :3]  # (T, H', W', 3)
    
    # Extract NIR from HR (B4 - 4th band of Planet)
    hr_nir = hr_seq[:, :, :, 3:4] if C_hr >= 4 else None  # (T, H', W', 1)
    
    # Create HR 20m by downsampling NIR
    if hr_nir is not None:
        hr_20m = hr_nir[:, ::2, ::2, :]  # Simple 2x downsample: (T, H'/2, W'/2, 1)
    else:
        hr_20m = None
    
    # Get tile bounds from sample
    bounds = sample.tile_bounds.get('bounds', None) if isinstance(sample.tile_bounds, dict) else None
    if bounds is None:
        print(f"      No tile bounds found")
        return None
    
    # Download native 20m B8A for each frame
    lr_20m_frames = []
    
    # Get S2 IDs from dates (need to match with original data)
    s2_dates = sample.temporal_dates
    
    # Create cache key
    bounds_str = f"{bounds[0]:.6f}_{bounds[1]:.6f}_{bounds[2]:.6f}_{bounds[3]:.6f}"
    
    for t, s2_date in enumerate(s2_dates):
        # Cache path for this frame's 20m data
        cache_key = f"b8a_20m_{s2_date}_{bounds_str}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        cache_path = cache_dir / f"b8a_20m_{cache_hash}.npy"
        
        if cache_path.exists():
            try:
                b8a_20m = np.load(cache_path)
                lr_20m_frames.append(b8a_20m)
                continue
            except:
                cache_path.unlink(missing_ok=True)
        
        # Need to find the S2 image ID for this date
        # Filter collection by date
        s2_img = s2_collection.filter(
            ee.Filter.date(s2_date, ee.Date(s2_date).advance(1, 'day'))
        ).first()
        
        if s2_img is None:
            print(f"      No S2 image found for {s2_date}")
            return None
        
        # Download B8A at 20m
        try:
            tile_geom = ee.Geometry.Rectangle(bounds)
            b8a_selected = s2_img.select(['B8A']).clip(tile_geom).unmask(0)
            b8a_resampled = b8a_selected.reproject(crs='EPSG:4326', scale=20)
            b8a_rect = b8a_resampled.sampleRectangle(region=tile_geom, defaultValue=0)
            b8a_info = b8a_rect.getInfo()
            
            if not b8a_info or 'properties' not in b8a_info:
                return None
            
            b8a_20m = np.array(b8a_info['properties']['B8A'], dtype=np.float32)
            b8a_20m = b8a_20m[:, :, np.newaxis]
            
            # Save to cache
            np.save(cache_path, b8a_20m)
            lr_20m_frames.append(b8a_20m)
            
        except Exception as e:
            print(f"      Error downloading B8A for {s2_date}: {e}")
            return None
    
    # Stack frames
    lr_20m = np.stack(lr_20m_frames, axis=0)  # (T, H/2, W/2, 1)
    
    # Create new multi-resolution sample
    new_sample = TrainingSample(
        lr_sequence=None,
        hr_sequence=None,
        lr_10m=lr_10m,
        lr_20m=lr_20m,
        hr_10m=hr_10m,
        hr_20m=hr_20m,
        temporal_dates=sample.temporal_dates,
        planet_dates=sample.planet_dates,
        region_name=sample.region_name,
        sequence_id=sample.sequence_id,
        tile_id=sample.tile_id,
        tile_bounds=sample.tile_bounds,
        s2_solar_zenith=sample.s2_solar_zenith,
        planet_solar_zenith=sample.planet_solar_zenith,
        band_mode=sample.band_mode if hasattr(sample, 'band_mode') else '4-band',
        is_multi_resolution=True
    )
    
    return new_sample


def convert_dataset(input_path: str, output_path: str, cache_dir: str, max_samples: int = None):
    """
    Convert entire dataset from single-res to multi-res format.
    """
    print("=" * 80)
    print("CONVERTING DATASET TO MULTI-RESOLUTION FORMAT")
    print("=" * 80)
    
    # Initialize GEE
    print("\nInitializing GEE...")
    try:
        from google.oauth2 import service_account
        credentials_path = "Auth/remotesensingeduart-222d07dd6689.json"
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        ee.Initialize(credentials)
    except:
        ee.Initialize()
    
    # Load existing dataset
    print(f"\nLoading existing dataset: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    samples = data.get('samples', [])
    config = data.get('config', {})
    
    print(f"  Found {len(samples)} samples")
    
    if max_samples:
        samples = samples[:max_samples]
        print(f"  Processing first {len(samples)} samples")
    
    # Setup cache
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Get S2 collection
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    
    # Convert samples
    converted_samples = []
    failed = 0
    
    print(f"\nConverting samples...")
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate / 60
            print(f"  Progress: {i+1}/{len(samples)} ({100*(i+1)/len(samples):.1f}%) - "
                  f"Rate: {rate:.2f} samples/sec - ETA: {eta:.1f} min")
        
        try:
            new_sample = convert_sample_to_multi_res(sample, s2_collection, cache_path)
            if new_sample is not None:
                converted_samples.append(new_sample)
            else:
                failed += 1
        except Exception as e:
            print(f"    Error converting sample {i}: {e}")
            failed += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n" + "=" * 80)
    print(f"CONVERSION COMPLETE")
    print(f"=" * 80)
    print(f"  Converted: {len(converted_samples)}/{len(samples)} samples")
    print(f"  Failed: {failed}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Rate: {len(converted_samples)/elapsed:.2f} samples/sec")
    
    # Save converted dataset
    output_data = {
        'samples': converted_samples,
        'config': {
            **config,
            'multi_resolution': True,
            'conversion_time': elapsed,
            'original_samples': len(samples),
            'converted_samples': len(converted_samples)
        }
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\nSaved to: {output_file}")
    
    # Verify output
    print(f"\nVerifying output...")
    if converted_samples:
        s = converted_samples[0]
        print(f"  Sample 0:")
        print(f"    lr_10m: {s.lr_10m.shape if s.lr_10m is not None else None}")
        print(f"    lr_20m: {s.lr_20m.shape if s.lr_20m is not None else None}")
        print(f"    hr_10m: {s.hr_10m.shape if s.hr_10m is not None else None}")
        print(f"    hr_20m: {s.hr_20m.shape if s.hr_20m is not None else None}")
        print(f"    is_multi_resolution: {s.is_multi_resolution}")


def main():
    parser = argparse.ArgumentParser(description='Convert dataset to multi-resolution format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input dataset path (.pkl)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output dataset path (.pkl)')
    parser.add_argument('--cache-dir', type=str, default='./cache/multi_res_conversion',
                        help='Cache directory for 20m downloads')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to convert (for testing)')
    
    args = parser.parse_args()
    
    convert_dataset(args.input, args.output, args.cache_dir, args.max_samples)


if __name__ == '__main__':
    main()
