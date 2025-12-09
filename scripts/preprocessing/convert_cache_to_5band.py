#!/usr/bin/env python3
"""
Convert cached raw downloads to 5-band training data.

This script reads the raw downloaded tiles from cache and creates proper training
samples WITHOUT truncating the HR bands. It preserves the full 5-band HR data
for SuperDove images (B2, B4, B6, B7, B8).

Band Mapping:
=============

SuperDove (8-band sensor, 2023+):
---------------------------------
Planet bands downloaded: B2, B4, B6, B7, B8 (5 bands)
  - B2 (Blue, 465-515nm)      → HR channel 0 → matches S2 B2 (10m)
  - B4 (Green II, 547-583nm)  → HR channel 1 → matches S2 B3 (10m)
  - B6 (Red, 650-680nm)       → HR channel 2 → matches S2 B4 (10m)
  - B7 (Red Edge, 697-713nm)  → HR channel 3 → matches S2 B5 (20m)
  - B8 (NIR, 845-885nm)       → HR channel 4 → matches S2 B8A (20m)

Sentinel-2 bands: B2, B3, B4, B5, B8A
  - LR 10m: B2, B3, B4 (3 channels at 77x77)
  - LR 20m: B5, B8A (2 channels at 38x38)

Dove/PS2 (4-band sensor, pre-2023):
-----------------------------------
Planet bands downloaded: B1, B2, B3, B4 (4 bands)
  - B1 (Blue, 455-515nm)   → HR channel 0 → matches S2 B2 (10m)
  - B2 (Green, 500-590nm)  → HR channel 1 → matches S2 B3 (10m)
  - B3 (Red, 590-670nm)    → HR channel 2 → matches S2 B4 (10m)
  - B4 (NIR, 780-860nm)    → HR channel 3 → matches S2 B8A (20m)

Sentinel-2 bands: B2, B3, B4, B8A
  - LR 10m: B2, B3, B4 (3 channels at 77x77)
  - LR 20m: B8A (1 channel at 38x38)

Output Structure:
=================
For SuperDove (5-band):
  - hr_sequence: (T, 256, 256, 5) - Blue, Green, Red, RedEdge, NIR
  - hr_10m_sequence: (T, 256, 256, 3) - Blue, Green, Red (for 10m SR)
  - hr_20m_sequence: (T, 128, 128, 2) - RedEdge, NIR (for 20m SR, downsampled 2x)
  - lr_sequence: (T, 77, 77, 3) - B2, B3, B4 at native 10m
  - lr_20m_sequence: (T, 38, 38, 2) - B5, B8A at native 20m

For Dove (4-band):
  - hr_sequence: (T, 256, 256, 4) - Blue, Green, Red, NIR
  - hr_10m_sequence: (T, 256, 256, 3) - Blue, Green, Red (for 10m SR)
  - hr_20m_sequence: (T, 128, 128, 1) - NIR only (for 20m SR, downsampled 2x)
  - lr_sequence: (T, 77, 77, 3) - B2, B3, B4 at native 10m
  - lr_20m_sequence: (T, 38, 38, 1) - B8A at native 20m
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def compute_frame_quality_stats(array: np.ndarray, 
                                 zero_threshold: float = 0.05,
                                 outlier_percentile: float = 99.5,
                                 min_valid_fraction: float = 0.90) -> Dict:
    """
    Compute LOCAL quality statistics for a downloaded frame (numpy array).
    
    This catches issues that server-side GEE coverage checking might miss:
    - Pixels filled with 0 from .unmask(0) on cloud/shadow masked areas
    - NaN values from failed computations
    - Outlier values indicating atmospheric contamination (haze, thin clouds)
    
    Args:
        array: numpy array (H, W, C) - single frame
        zero_threshold: threshold below which pixels are considered "zero/bad"
        outlier_percentile: percentile above which pixels are outliers
        min_valid_fraction: minimum valid pixel fraction to pass quality check
    
    Returns:
        Dict with quality metrics and pass/fail flag
    """
    total_pixels = array.size
    
    # Count NaN pixels
    nan_mask = np.isnan(array)
    nan_count = np.sum(nan_mask)
    nan_fraction = nan_count / total_pixels
    
    # Count zero/near-zero pixels (from .unmask(0) on masked areas)
    # Use small threshold to catch near-zeros from interpolation
    zero_mask = np.abs(array) < zero_threshold
    zero_count = np.sum(zero_mask & ~nan_mask)  # Don't double-count NaNs
    zero_fraction = zero_count / total_pixels
    
    # Get valid pixels for outlier detection
    valid_mask = ~nan_mask & ~zero_mask
    valid_pixels = array[valid_mask]
    
    if len(valid_pixels) == 0:
        # All pixels are bad
        return {
            'zero_frac': float(zero_fraction),
            'nan_frac': float(nan_fraction),
            'outlier_frac': 0.0,
            'valid_frac': 0.0,
            'mean_refl': 0.0,
            'std_refl': 0.0,
            'is_good': False
        }
    
    # Detect outliers (extremely high values often indicate atmospheric issues)
    outlier_threshold = np.percentile(valid_pixels, outlier_percentile)
    outlier_mask = (array > outlier_threshold) & valid_mask
    outlier_count = np.sum(outlier_mask)
    outlier_fraction = outlier_count / total_pixels
    
    # Final valid pixels (not nan, not zero, not outlier)
    final_valid_mask = valid_mask & ~outlier_mask
    valid_fraction = np.sum(final_valid_mask) / total_pixels
    
    # Statistics on valid pixels
    final_valid_pixels = array[final_valid_mask]
    mean_refl = float(np.mean(final_valid_pixels)) if len(final_valid_pixels) > 0 else 0.0
    std_refl = float(np.std(final_valid_pixels)) if len(final_valid_pixels) > 0 else 0.0
    
    # Quality decision
    is_good = valid_fraction >= min_valid_fraction
    
    return {
        'zero_frac': float(zero_fraction),
        'nan_frac': float(nan_fraction),
        'outlier_frac': float(outlier_fraction),
        'valid_frac': float(valid_fraction),
        'mean_refl': mean_refl,
        'std_refl': std_refl,
        'is_good': is_good
    }


def compute_combined_frame_quality(lr_frame: np.ndarray, hr_frame: np.ndarray, 
                                    frame_idx: int = 0) -> Dict:
    """
    Compute combined quality stats for a frame (both LR and HR).
    
    Args:
        lr_frame: LR array (H, W, C)
        hr_frame: HR array (H, W, C)
        frame_idx: Frame index for logging
    
    Returns:
        Combined quality dict with both LR and HR stats
    """
    lr_stats = compute_frame_quality_stats(lr_frame)
    hr_stats = compute_frame_quality_stats(hr_frame)
    
    return {
        'frame_idx': frame_idx,
        'lr_zero_frac': lr_stats['zero_frac'],
        'lr_nan_frac': lr_stats['nan_frac'],
        'lr_outlier_frac': lr_stats['outlier_frac'],
        'lr_valid_frac': lr_stats['valid_frac'],
        'hr_zero_frac': hr_stats['zero_frac'],
        'hr_nan_frac': hr_stats['nan_frac'],
        'hr_outlier_frac': hr_stats['outlier_frac'],
        'hr_valid_frac': hr_stats['valid_frac'],
        'is_good': lr_stats['is_good'] and hr_stats['is_good']
    }


@dataclass
class TrainingSample5Band:
    """Training sample with full 5-band HR data for SuperDove images."""
    # Core sequences
    lr_sequence: np.ndarray  # (T, 77, 77, 3) - 10m bands at native resolution
    hr_sequence: np.ndarray  # (T, 256, 256, C) - Full HR (5 for SuperDove, 4 for Dove)
    lr_20m_sequence: Optional[np.ndarray] = None  # (T, 38, 38, C_20m) - 20m bands at native
    
    # Split HR for multi-resolution training
    hr_10m_sequence: Optional[np.ndarray] = None  # (T, 256, 256, 3) - RGB bands only
    hr_20m_sequence: Optional[np.ndarray] = None  # (T, 128, 128, C_20m) - RedEdge/NIR at 6m
    
    # Metadata
    temporal_dates: List[str] = field(default_factory=list)
    planet_dates: List[str] = field(default_factory=list)
    region_name: str = ''
    sequence_id: int = 0
    tile_id: int = 0
    tile_bounds: Dict = field(default_factory=dict)
    
    # Band configuration
    sensor_type: str = ''  # 'superdove' or 'dove'
    planet_bands: List[str] = field(default_factory=list)
    sentinel_bands: List[str] = field(default_factory=list)
    
    # Solar zenith angles
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None
    
    # Quality metrics
    frame_quality: Optional[List[Dict]] = None


def detect_sensor_type(planet_bands: List[str]) -> str:
    """
    Detect sensor type based on Planet band names.
    
    SuperDove (8-band): Uses B2, B4, B6, B7, B8 naming
    Dove/PS2 (4-band): Uses B1, B2, B3, B4 naming
    """
    planet_bands_set = set(planet_bands)
    
    # SuperDove indicators: B6, B7, B8 are unique to 8-band
    if 'B7' in planet_bands_set or 'B8' in planet_bands_set:
        return 'superdove'
    # Dove indicators: B1 with B3 (not B6)
    elif 'B1' in planet_bands_set and 'B3' in planet_bands_set:
        return 'dove'
    else:
        # Fallback based on band count
        return 'superdove' if len(planet_bands) >= 5 else 'dove'


def get_band_mapping(sensor_type: str) -> Dict:
    """
    Get band mapping for each sensor type.
    
    Returns dict with:
    - hr_to_s2_10m: mapping HR channels to S2 10m channels
    - hr_to_s2_20m: mapping HR channels to S2 20m channels
    - hr_10m_indices: HR channels that correspond to 10m bands
    - hr_20m_indices: HR channels that correspond to 20m bands
    """
    if sensor_type == 'superdove':
        return {
            'planet_bands': ['B2', 'B4', 'B6', 'B7', 'B8'],
            's2_bands_10m': ['B2', 'B3', 'B4'],
            's2_bands_20m': ['B5', 'B8A'],
            # HR channel indices for 10m bands (Blue, Green, Red)
            'hr_10m_indices': [0, 1, 2],
            # HR channel indices for 20m bands (RedEdge, NIR)
            'hr_20m_indices': [3, 4],
            # Spectral correspondence
            'hr_to_s2': {
                0: ('B2', '10m'),   # Planet B2 (Blue) → S2 B2
                1: ('B3', '10m'),   # Planet B4 (Green) → S2 B3
                2: ('B4', '10m'),   # Planet B6 (Red) → S2 B4
                3: ('B5', '20m'),   # Planet B7 (RedEdge) → S2 B5
                4: ('B8A', '20m'),  # Planet B8 (NIR) → S2 B8A
            }
        }
    else:  # dove
        return {
            'planet_bands': ['B1', 'B2', 'B3', 'B4'],
            's2_bands_10m': ['B2', 'B3', 'B4'],
            's2_bands_20m': ['B8A'],
            # HR channel indices for 10m bands (Blue, Green, Red)
            'hr_10m_indices': [0, 1, 2],
            # HR channel indices for 20m bands (NIR only)
            'hr_20m_indices': [3],
            # Spectral correspondence
            'hr_to_s2': {
                0: ('B2', '10m'),   # Planet B1 (Blue) → S2 B2
                1: ('B3', '10m'),   # Planet B2 (Green) → S2 B3
                2: ('B4', '10m'),   # Planet B3 (Red) → S2 B4
                3: ('B8A', '20m'),  # Planet B4 (NIR) → S2 B8A
            }
        }


def downsample_hr_for_20m(hr_array: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Downsample HR array by factor for 20m band targets.
    
    HR at 3m → target at 6m (factor 2) to match 20m/10m = 2x ratio
    Input: (H, W, C) at 3m → Output: (H/2, W/2, C) at 6m
    """
    from scipy.ndimage import zoom
    
    # Use area averaging (order=1 with anti-aliasing effect)
    # For (256, 256, C) → (128, 128, C)
    h, w, c = hr_array.shape
    new_h, new_w = h // factor, w // factor
    
    # Simple block averaging for clean downsampling
    result = hr_array.reshape(new_h, factor, new_w, factor, c).mean(axis=(1, 3))
    return result.astype(hr_array.dtype)


def process_cached_tile(npz_path: Path, apply_coregistration: bool = False,
                        apply_harmonization: bool = False) -> Optional[Dict]:
    """
    Process a single cached tile and extract band data.
    
    Args:
        npz_path: Path to cached .npz file
        apply_coregistration: Whether to apply coregistration (requires additional cache)
        apply_harmonization: Whether to apply harmonization (requires additional cache)
    
    Returns:
        Dict with processed data or None if failed
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # Extract arrays
        hr_array = data['hr_array']
        lr_10m = data['lr_10m'] if 'lr_10m' in data else data.get('lr_array')
        lr_20m = data['lr_20m'] if 'lr_20m' in data else None
        
        # Extract band names
        planet_bands = list(data['planet_bands']) if 'planet_bands' in data else None
        s2_bands = list(data['s2_bands']) if 's2_bands' in data else None
        
        if planet_bands is None or hr_array is None or lr_10m is None:
            return None
        
        # Detect sensor type
        sensor_type = detect_sensor_type(planet_bands)
        band_mapping = get_band_mapping(sensor_type)
        
        # Validate shapes
        expected_hr_bands = len(band_mapping['planet_bands'])
        if hr_array.shape[-1] != expected_hr_bands:
            # Band count mismatch - might be truncated data
            return None
        
        # Extract 10m and 20m HR bands
        hr_10m_indices = band_mapping['hr_10m_indices']
        hr_20m_indices = band_mapping['hr_20m_indices']
        
        hr_10m = hr_array[:, :, hr_10m_indices]  # (256, 256, 3)
        hr_20m_full = hr_array[:, :, hr_20m_indices]  # (256, 256, 1 or 2)
        
        # Downsample 20m HR bands from 3m to 6m (factor 2)
        # This creates proper HR targets for 20m→6m super-resolution
        hr_20m = downsample_hr_for_20m(hr_20m_full, factor=2)  # (128, 128, 1 or 2)
        
        # Compute frame quality stats (useful for filtering during training)
        frame_quality = compute_frame_quality_stats(lr_10m)
        hr_quality = compute_frame_quality_stats(hr_array)
        
        # Combined quality (frame is good only if both LR and HR are good)
        combined_quality = {
            'lr_zero_frac': frame_quality['zero_frac'],
            'lr_nan_frac': frame_quality['nan_frac'],
            'lr_outlier_frac': frame_quality['outlier_frac'],
            'lr_valid_frac': frame_quality['valid_frac'],
            'hr_zero_frac': hr_quality['zero_frac'],
            'hr_nan_frac': hr_quality['nan_frac'],
            'hr_outlier_frac': hr_quality['outlier_frac'],
            'hr_valid_frac': hr_quality['valid_frac'],
            'is_good': frame_quality['is_good'] and hr_quality['is_good']
        }
        
        return {
            'hr_full': hr_array,       # Full HR (256, 256, 4 or 5)
            'hr_10m': hr_10m,          # RGB HR (256, 256, 3)
            'hr_20m': hr_20m,          # RedEdge/NIR HR at 6m (128, 128, 1 or 2)
            'lr_10m': lr_10m,          # S2 10m (77, 77, 3)
            'lr_20m': lr_20m,          # S2 20m (38, 38, 1 or 2)
            'sensor_type': sensor_type,
            'planet_bands': planet_bands,
            's2_bands': s2_bands,
            'band_mapping': band_mapping,
            'cache_file': npz_path.name,
            'frame_quality': combined_quality  # NEW: quality metadata
        }
        
    except Exception as e:
        print(f"Error processing {npz_path.name}: {e}")
        return None


def group_tiles_by_sequence(cache_dir: Path) -> Dict[str, List[Path]]:
    """
    Group cached tiles by their sequence/date.
    
    Cache files are named with hash of: planet_id|s2_id|tile_bounds|bands
    We need to group files that belong to the same temporal sequence.
    """
    # For now, return all files as individual items
    # Proper grouping would require parsing the original sequence info
    npz_files = list(cache_dir.glob('*.npz'))
    return {'all': npz_files}


def create_training_samples_from_cache(
    cache_dir: Path,
    output_dir: Path,
    max_samples: Optional[int] = None,
    sequence_length: int = 3,
    filter_sensor: Optional[str] = None,  # 'superdove', 'dove', or None for both
    normalize: bool = True,
    scale_factor: float = 10000.0
) -> List[TrainingSample5Band]:
    """
    Create training samples from cached raw downloads.
    
    Args:
        cache_dir: Directory with cached .npz files
        output_dir: Output directory for processed dataset
        max_samples: Maximum number of samples to create
        sequence_length: Temporal sequence length (default 3)
        filter_sensor: Filter by sensor type ('superdove', 'dove', or None)
        normalize: Whether to normalize to [0, 1]
        scale_factor: Normalization scale factor
    
    Returns:
        List of TrainingSample5Band objects
    """
    print(f"\n{'='*80}")
    print("CONVERTING CACHED DATA TO 5-BAND TRAINING SAMPLES")
    print(f"{'='*80}")
    print(f"Cache directory: {cache_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Filter sensor: {filter_sensor or 'all'}")
    print(f"Normalize: {normalize}")
    print(f"{'='*80}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all cached files
    npz_files = sorted(cache_dir.glob('*.npz'))
    print(f"Found {len(npz_files)} cached tiles")
    
    # Process tiles and group by sensor type
    superdove_tiles = []
    dove_tiles = []
    errors = 0
    
    print("\nProcessing tiles...")
    start_time = time.time()
    
    for i, npz_path in enumerate(npz_files):
        if max_samples and (len(superdove_tiles) + len(dove_tiles)) >= max_samples * sequence_length:
            break
            
        result = process_cached_tile(npz_path)
        
        if result is None:
            errors += 1
            continue
        
        if result['sensor_type'] == 'superdove':
            if filter_sensor is None or filter_sensor == 'superdove':
                superdove_tiles.append(result)
        else:
            if filter_sensor is None or filter_sensor == 'dove':
                dove_tiles.append(result)
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {i+1}/{len(npz_files)} tiles "
                  f"(SuperDove: {len(superdove_tiles)}, Dove: {len(dove_tiles)}, "
                  f"Errors: {errors}) - {elapsed:.1f}s")
    
    print(f"\nProcessing complete:")
    print(f"  SuperDove tiles: {len(superdove_tiles)}")
    print(f"  Dove tiles: {len(dove_tiles)}")
    print(f"  Errors: {errors}")
    
    # Create samples by grouping tiles into sequences
    samples = []
    
    def create_samples_from_tiles(tiles: List[Dict], sensor_name: str) -> List[TrainingSample5Band]:
        """Create temporal sequences from tile list."""
        tile_samples = []
        
        # For simplicity, create non-overlapping sequences
        # In production, you'd group by actual temporal dates
        n_sequences = len(tiles) // sequence_length
        
        if max_samples:
            n_sequences = min(n_sequences, max_samples)
        
        print(f"\nCreating {n_sequences} {sensor_name} sequences...")
        
        for seq_idx in range(n_sequences):
            start_idx = seq_idx * sequence_length
            seq_tiles = tiles[start_idx:start_idx + sequence_length]
            
            if len(seq_tiles) != sequence_length:
                continue
            
            # Stack frames
            hr_full_stack = np.stack([t['hr_full'] for t in seq_tiles], axis=0)
            hr_10m_stack = np.stack([t['hr_10m'] for t in seq_tiles], axis=0)
            hr_20m_stack = np.stack([t['hr_20m'] for t in seq_tiles], axis=0)
            lr_10m_stack = np.stack([t['lr_10m'] for t in seq_tiles], axis=0)
            
            # Stack 20m LR if available
            lr_20m_stack = None
            if all(t['lr_20m'] is not None for t in seq_tiles):
                lr_20m_stack = np.stack([t['lr_20m'] for t in seq_tiles], axis=0)
            
            # Normalize if requested
            if normalize:
                hr_full_stack = np.clip(hr_full_stack / scale_factor, 0, 1).astype(np.float32)
                hr_10m_stack = np.clip(hr_10m_stack / scale_factor, 0, 1).astype(np.float32)
                hr_20m_stack = np.clip(hr_20m_stack / scale_factor, 0, 1).astype(np.float32)
                lr_10m_stack = np.clip(lr_10m_stack / scale_factor, 0, 1).astype(np.float32)
                if lr_20m_stack is not None:
                    lr_20m_stack = np.clip(lr_20m_stack / scale_factor, 0, 1).astype(np.float32)
            
            # Collect frame quality for each frame in sequence
            frame_quality_list = []
            for frame_idx, tile in enumerate(seq_tiles):
                fq = tile.get('frame_quality', {})
                fq['frame_idx'] = frame_idx
                frame_quality_list.append(fq)
            
            sample = TrainingSample5Band(
                lr_sequence=lr_10m_stack,
                hr_sequence=hr_full_stack,
                lr_20m_sequence=lr_20m_stack,
                hr_10m_sequence=hr_10m_stack,
                hr_20m_sequence=hr_20m_stack,
                sensor_type=sensor_name,
                planet_bands=seq_tiles[0]['planet_bands'],
                sentinel_bands=seq_tiles[0]['s2_bands'],
                sequence_id=seq_idx,
                tile_id=seq_idx,
                region_name='STA-2',
                frame_quality=frame_quality_list  # NEW: preserve quality metadata
            )
            tile_samples.append(sample)
            
            if (seq_idx + 1) % 100 == 0:
                print(f"  Created {seq_idx + 1}/{n_sequences} sequences")
        
        return tile_samples
    
    # Create samples for each sensor type
    if superdove_tiles:
        samples.extend(create_samples_from_tiles(superdove_tiles, 'superdove'))
    if dove_tiles:
        samples.extend(create_samples_from_tiles(dove_tiles, 'dove'))
    
    print(f"\nTotal samples created: {len(samples)}")
    
    # Save dataset
    if samples:
        # Separate by sensor type for easier training
        superdove_samples = [s for s in samples if s.sensor_type == 'superdove']
        dove_samples = [s for s in samples if s.sensor_type == 'dove']
        
        # Save SuperDove samples (5-band)
        if superdove_samples:
            superdove_output = output_dir / 'superdove_5band'
            superdove_output.mkdir(parents=True, exist_ok=True)
            
            dataset_superdove = {
                'samples': superdove_samples,
                'metadata': {
                    'sensor_type': 'superdove',
                    'n_samples': len(superdove_samples),
                    'hr_bands': 5,
                    'hr_10m_bands': 3,
                    'hr_20m_bands': 2,
                    'lr_10m_bands': 3,
                    'lr_20m_bands': 2,
                    'planet_bands': ['B2', 'B4', 'B6', 'B7', 'B8'],
                    's2_bands_10m': ['B2', 'B3', 'B4'],
                    's2_bands_20m': ['B5', 'B8A'],
                    'normalized': normalize,
                    'scale_factor': scale_factor,
                    'band_mapping': {
                        'hr_ch0': 'Blue (Planet B2 → S2 B2)',
                        'hr_ch1': 'Green (Planet B4 → S2 B3)',
                        'hr_ch2': 'Red (Planet B6 → S2 B4)',
                        'hr_ch3': 'RedEdge (Planet B7 → S2 B5)',
                        'hr_ch4': 'NIR (Planet B8 → S2 B8A)',
                    }
                }
            }
            
            with open(superdove_output / 'dataset.pkl', 'wb') as f:
                pickle.dump(dataset_superdove, f)
            
            print(f"\nSaved {len(superdove_samples)} SuperDove samples to {superdove_output}")
            print(f"  HR shape: {superdove_samples[0].hr_sequence.shape}")
            print(f"  HR 10m shape: {superdove_samples[0].hr_10m_sequence.shape}")
            print(f"  HR 20m shape: {superdove_samples[0].hr_20m_sequence.shape}")
            print(f"  LR 10m shape: {superdove_samples[0].lr_sequence.shape}")
            print(f"  LR 20m shape: {superdove_samples[0].lr_20m_sequence.shape if superdove_samples[0].lr_20m_sequence is not None else 'None'}")
        
        # Save Dove samples (4-band)
        if dove_samples:
            dove_output = output_dir / 'dove_4band'
            dove_output.mkdir(parents=True, exist_ok=True)
            
            dataset_dove = {
                'samples': dove_samples,
                'metadata': {
                    'sensor_type': 'dove',
                    'n_samples': len(dove_samples),
                    'hr_bands': 4,
                    'hr_10m_bands': 3,
                    'hr_20m_bands': 1,
                    'lr_10m_bands': 3,
                    'lr_20m_bands': 1,
                    'planet_bands': ['B1', 'B2', 'B3', 'B4'],
                    's2_bands_10m': ['B2', 'B3', 'B4'],
                    's2_bands_20m': ['B8A'],
                    'normalized': normalize,
                    'scale_factor': scale_factor,
                    'band_mapping': {
                        'hr_ch0': 'Blue (Planet B1 → S2 B2)',
                        'hr_ch1': 'Green (Planet B2 → S2 B3)',
                        'hr_ch2': 'Red (Planet B3 → S2 B4)',
                        'hr_ch3': 'NIR (Planet B4 → S2 B8A)',
                    }
                }
            }
            
            with open(dove_output / 'dataset.pkl', 'wb') as f:
                pickle.dump(dataset_dove, f)
            
            print(f"\nSaved {len(dove_samples)} Dove samples to {dove_output}")
            print(f"  HR shape: {dove_samples[0].hr_sequence.shape}")
            print(f"  HR 10m shape: {dove_samples[0].hr_10m_sequence.shape}")
            print(f"  HR 20m shape: {dove_samples[0].hr_20m_sequence.shape}")
            print(f"  LR 10m shape: {dove_samples[0].lr_sequence.shape}")
            print(f"  LR 20m shape: {dove_samples[0].lr_20m_sequence.shape if dove_samples[0].lr_20m_sequence is not None else 'None'}")
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Convert cached raw downloads to 5-band training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Band Mapping Reference:
=======================

SuperDove (8-band, 2023+):
  Planet B2 (Blue)      → HR ch0 → S2 B2 (10m)
  Planet B4 (Green)     → HR ch1 → S2 B3 (10m)
  Planet B6 (Red)       → HR ch2 → S2 B4 (10m)
  Planet B7 (Red Edge)  → HR ch3 → S2 B5 (20m)
  Planet B8 (NIR)       → HR ch4 → S2 B8A (20m)

Dove (4-band, pre-2023):
  Planet B1 (Blue)      → HR ch0 → S2 B2 (10m)
  Planet B2 (Green)     → HR ch1 → S2 B3 (10m)
  Planet B3 (Red)       → HR ch2 → S2 B4 (10m)
  Planet B4 (NIR)       → HR ch3 → S2 B8A (20m)
        """
    )
    
    parser.add_argument('--cache-dir', type=str,
                        default='./cache/sta2_grid_parallel/raw_downloads',
                        help='Cache directory with raw .npz files')
    parser.add_argument('--output-dir', type=str,
                        default='./data/sr_training/5band_converted',
                        help='Output directory for converted dataset')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per sensor type')
    parser.add_argument('--sequence-length', type=int, default=3,
                        help='Temporal sequence length')
    parser.add_argument('--sensor', type=str, default=None,
                        choices=['superdove', 'dove'],
                        help='Filter by sensor type (default: both)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Keep raw reflectance values (0-10000)')
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    
    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        sys.exit(1)
    
    samples = create_training_samples_from_cache(
        cache_dir=cache_dir,
        output_dir=output_dir,
        max_samples=args.max_samples,
        sequence_length=args.sequence_length,
        filter_sensor=args.sensor,
        normalize=not args.no_normalize
    )
    
    print(f"\n{'='*80}")
    print("CONVERSION COMPLETE")
    print(f"{'='*80}")
    print(f"Total samples: {len(samples)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
