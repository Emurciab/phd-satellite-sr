#!/usr/bin/env python3
"""
STA-1E GRID-BASED PARALLEL dataset generation with coverage pre-checking

This is a direct copy of generate_sta2_grid_parallel.py adapted for STA-1E.
It generates 4-band or 8-band data dynamically based on sensor type (Dove vs SuperDove).

Features:
- Grid-based tile sampling (systematic coverage)
- Coverage pre-checking (avoid Planet mosaic gaps)
- Parallel sequence processing
- Direct tile downloads (no full image waste)
- Dynamic band detection (4-band Dove vs 8-band SuperDove)

To generate 5-band data for SR training:
1. Run this script to generate the base dataset (4/8-band)
2. Run convert_dataset_to_5band_sta1e.py to filter and convert to 5-band format
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
import hashlib
import io
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from Auth.AuthGEE import find_pairs, PLANET_COLLECTION_ID, regions

# Additional Planet collections for maximum data coverage
PLANET_COLLECTION_ID_FALLBACK = 'projects/ee-eeamurciabb/assets/testSTAs'
PLANET_COLLECTION_ID_ADDITIONAL = 'projects/stasadditionalfree/assets/testSTAs_ic'


@dataclass
class TrainingSample:
    """Training sample with temporal sequences and multi-resolution LR bands"""
    lr_sequence: np.ndarray  # (T, H, W, C) - 10m bands at native resolution
    hr_sequence: np.ndarray  # (T, H, W, C)
    temporal_dates: List[str]  # Sentinel-2 dates
    planet_dates: List[str]  # Planet dates (may differ by ±7 days)
    region_name: str
    sequence_id: int
    tile_id: int
    tile_bounds: Dict
    # Solar zenith angles for each frame (optimal ~45° for aquatic vegetation)
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None
    # Multi-resolution: 20m bands stored at native resolution (half spatial size of 10m)
    lr_20m_sequence: Optional[np.ndarray] = None  # (T, H/2, W/2, C_20m) - B8A etc at native 20m
    # LOCAL-SIDE frame quality statistics (computed AFTER download)
    frame_quality: Optional[List[Dict]] = None


@dataclass
class FrameQualityStats:
    """Quality statistics for a single frame (computed locally from numpy arrays)"""
    zero_fraction: float
    nan_fraction: float
    outlier_fraction: float
    valid_fraction: float
    mean_reflectance: float
    std_reflectance: float
    is_good: bool
    
    def to_dict(self) -> Dict:
        return {
            'zero_frac': self.zero_fraction,
            'nan_frac': self.nan_fraction,
            'outlier_frac': self.outlier_fraction,
            'valid_frac': self.valid_fraction,
            'mean_refl': self.mean_reflectance,
            'std_refl': self.std_reflectance,
            'is_good': self.is_good
        }


def normalize_sample(sample: TrainingSample, scale_factor: float = 10000.0) -> TrainingSample:
    """
    Normalize sample arrays from raw reflectance (0-10000) to [0, 1] range.
    """
    sample.lr_sequence = np.clip(sample.lr_sequence / scale_factor, 0, 1).astype(np.float32)
    sample.hr_sequence = np.clip(sample.hr_sequence / scale_factor, 0, 1).astype(np.float32)
    
    if sample.lr_20m_sequence is not None:
        sample.lr_20m_sequence = np.clip(sample.lr_20m_sequence / scale_factor, 0, 1).astype(np.float32)
    
    return sample


def compute_frame_quality_stats(array: np.ndarray, 
                                 zero_threshold: float = 0.05,
                                 outlier_percentile: float = 99.5,
                                 min_valid_fraction: float = 0.90) -> FrameQualityStats:
    """
    Compute LOCAL quality statistics for a downloaded frame (numpy array).
    """
    total_pixels = array.size
    
    # Count NaN pixels
    nan_mask = np.isnan(array)
    nan_count = np.sum(nan_mask)
    nan_fraction = nan_count / total_pixels
    
    # Count zero/near-zero pixels
    zero_mask = np.abs(array) < zero_threshold
    zero_count = np.sum(zero_mask & ~nan_mask)
    zero_fraction = zero_count / total_pixels
    
    # Get valid pixels for outlier detection
    valid_mask = ~nan_mask & ~zero_mask
    valid_pixels = array[valid_mask]
    
    if len(valid_pixels) == 0:
        return FrameQualityStats(
            zero_fraction=zero_fraction,
            nan_fraction=nan_fraction,
            outlier_fraction=0.0,
            valid_fraction=0.0,
            mean_reflectance=0.0,
            std_reflectance=0.0,
            is_good=False
        )
    
    # Detect outliers
    outlier_threshold = np.percentile(valid_pixels, outlier_percentile)
    outlier_mask = (array > outlier_threshold) & valid_mask
    outlier_count = np.sum(outlier_mask)
    outlier_fraction = outlier_count / total_pixels
    
    # Final valid pixels
    final_valid_mask = valid_mask & ~outlier_mask
    valid_fraction = np.sum(final_valid_mask) / total_pixels
    
    # Statistics
    final_valid_pixels = array[final_valid_mask]
    mean_refl = float(np.mean(final_valid_pixels)) if len(final_valid_pixels) > 0 else 0.0
    std_refl = float(np.std(final_valid_pixels)) if len(final_valid_pixels) > 0 else 0.0
    
    is_good = valid_fraction >= min_valid_fraction
    
    return FrameQualityStats(
        zero_fraction=float(zero_fraction),
        nan_fraction=float(nan_fraction),
        outlier_fraction=float(outlier_fraction),
        valid_fraction=float(valid_fraction),
        mean_reflectance=mean_refl,
        std_reflectance=std_refl,
        is_good=is_good
    )


# =============================================================================
# GEE DOWNLOAD FUNCTIONS
# =============================================================================

def _compute_pixels_download(img: ee.Image, bounds: List[float], bands: List[str],
                             width: int = 256, height: int = 256) -> Optional[np.ndarray]:
    """
    Download image region using ee.data.computePixels (5x faster than sampleRectangle).
    """
    try:
        selected = img.select(bands).unmask(0)
        
        request = {
            'expression': selected,
            'fileFormat': 'NPY',
            'grid': {
                'dimensions': {'width': width, 'height': height},
                'affineTransform': {
                    'scaleX': (bounds[2] - bounds[0]) / width,
                    'shearX': 0,
                    'translateX': bounds[0],
                    'scaleY': -(bounds[3] - bounds[1]) / height,
                    'shearY': 0,
                    'translateY': bounds[3]
                },
                'crsCode': 'EPSG:4326'
            }
        }
        
        result_bytes = ee.data.computePixels(request)
        arr = np.load(io.BytesIO(result_bytes))
        result = np.stack([arr[b].astype(np.float32) for b in bands], axis=-1)
        
        return result
        
    except Exception as e:
        return None


def download_all_frames_batched(sequence: List[Dict], tile_bounds: List[float],
                                planet_collection, s2_collection,
                                planet_bands: List[str], sentinel_bands: List[str],
                                cache_dir: Path, tile_size_px: int = 256) -> Tuple[Dict, Dict]:
    """
    Download all frames for a tile using batched computePixels.
    """
    results = {}
    cache_stats = {'hits': 0, 'misses': 0}
    
    # Check cache first for all frames
    frames_to_download = []
    for frame_idx, pair in enumerate(sequence):
        cache_key_str = f"{pair['planet_id']}|{pair['s2_id']}|{tile_bounds}|{planet_bands}|{sentinel_bands}"
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:16]
        cache_path = Path(cache_dir) / "raw_downloads" / f"tile_{cache_hash}.npz"
        
        if cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=True)
                if 'lr_10m' in cached:
                    lr_data = {
                        'bands_10m': cached['lr_10m'],
                        'bands_20m': cached['lr_20m'] if 'lr_20m' in cached else None,
                        'band_names_10m': list(cached['band_names_10m']) if 'band_names_10m' in cached else None,
                        'band_names_20m': list(cached['band_names_20m']) if 'band_names_20m' in cached else None
                    }
                else:
                    lr_data = {
                        'bands_10m': cached['lr_array'],
                        'bands_20m': None,
                        'band_names_10m': None,
                        'band_names_20m': None
                    }
                results[frame_idx] = {
                    'hr': cached['hr_array'],
                    'lr': lr_data,
                    'pair': pair
                }
                cache_stats['hits'] += 1
                continue
            except:
                pass
        
        frames_to_download.append({
            'frame_idx': frame_idx,
            'pair': pair,
            'cache_path': cache_path
        })
        cache_stats['misses'] += 1
    
    if not frames_to_download:
        return results, cache_stats
    
    # Download missing frames using computePixels in parallel
    def download_frame_batched(task):
        pair = task['pair']
        frame_idx = task['frame_idx']
        
        try:
            planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
            s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
            
            # Resolution calculations
            scale_factor_10m = 10.0 / 3.0
            scale_factor_20m = 20.0 / 3.0
            lr_10m_size = int(round(tile_size_px / scale_factor_10m))
            lr_20m_size = int(round(tile_size_px / scale_factor_20m))
            
            # Download HR (Planet)
            hr_array = _compute_pixels_download(
                planet_img, tile_bounds, planet_bands,
                width=tile_size_px, height=tile_size_px
            )
            
            # Download S2 at native resolutions
            bands_10m = [b for b in sentinel_bands if b in ['B2', 'B3', 'B4', 'B8']]
            bands_20m = [b for b in sentinel_bands if b in ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']]
            
            lr_data = {}
            
            if bands_10m:
                lr_10m = _compute_pixels_download(
                    s2_img, tile_bounds, bands_10m,
                    width=lr_10m_size, height=lr_10m_size
                )
                if lr_10m is not None:
                    lr_data['bands_10m'] = lr_10m
                    lr_data['band_names_10m'] = bands_10m
            
            if bands_20m:
                lr_20m = _compute_pixels_download(
                    s2_img, tile_bounds, bands_20m,
                    width=lr_20m_size, height=lr_20m_size
                )
                if lr_20m is not None:
                    lr_data['bands_20m'] = lr_20m
                    lr_data['band_names_20m'] = bands_20m
            
            if hr_array is not None and lr_data:
                # Save to cache
                task['cache_path'].parent.mkdir(parents=True, exist_ok=True)
                try:
                    save_dict = {
                        'hr_array': hr_array,
                        'planet_bands': planet_bands,
                        's2_bands': sentinel_bands
                    }
                    if 'bands_10m' in lr_data:
                        save_dict['lr_10m'] = lr_data['bands_10m']
                        save_dict['band_names_10m'] = lr_data.get('band_names_10m', [])
                    if 'bands_20m' in lr_data:
                        save_dict['lr_20m'] = lr_data['bands_20m']
                        save_dict['band_names_20m'] = lr_data.get('band_names_20m', [])
                    np.savez_compressed(task['cache_path'], **save_dict)
                except:
                    pass
                
                return frame_idx, hr_array, lr_data, pair
                    
        except Exception as e:
            pass
        
        return frame_idx, None, None, task['pair']
    
    # Use 20 parallel workers for GEE
    max_workers = min(20, len(frames_to_download))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_frame_batched, task): task for task in frames_to_download}
        
        for future in as_completed(futures):
            try:
                frame_idx, hr_array, lr_data, pair = future.result()
                if hr_array is not None and lr_data is not None:
                    results[frame_idx] = {'hr': hr_array, 'lr': lr_data, 'pair': pair}
            except Exception as e:
                pass
    
    return results, cache_stats


# =============================================================================
# COVERAGE CHECKING
# =============================================================================

def has_sufficient_coverage(img: ee.Image, geom: ee.Geometry, band_name: str, 
                           min_percent: float = 90.0, scale: int = 30) -> bool:
    """Check if image has sufficient valid pixel coverage over geometry."""
    try:
        valid_mask = img.select(band_name).mask()
        stats = valid_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale,
            maxPixels=1e6,
            bestEffort=True
        )
        valid_fraction = stats.get(band_name).getInfo()
        if valid_fraction is None:
            return False
        return valid_fraction * 100.0 >= min_percent
    except:
        return False


def generate_tile_grid(aoi, tile_size_deg=0.007, max_tiles=None):
    """Generate systematic grid of tiles covering the AOI."""
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    num_cols = int(np.ceil((max_lon - min_lon) / tile_size_deg))
    num_rows = int(np.ceil((max_lat - min_lat) / tile_size_deg))
    
    print(f"   Grid dimensions: {num_rows} rows × {num_cols} cols = {num_rows * num_cols} potential tiles")
    
    tiles = []
    for row in range(num_rows):
        for col in range(num_cols):
            tile_min_lon = min_lon + col * tile_size_deg
            tile_min_lat = min_lat + row * tile_size_deg
            tile_max_lon = min(tile_min_lon + tile_size_deg, max_lon)
            tile_max_lat = min(tile_min_lat + tile_size_deg, max_lat)
            
            tiles.append([tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat])
            
            if max_tiles and len(tiles) >= max_tiles:
                break
        if max_tiles and len(tiles) >= max_tiles:
            break
    
    return tiles


# =============================================================================
# COREGISTRATION AND HARMONIZATION
# =============================================================================

def coregister_lr_to_hr_wrapper(hr_array, lr_array, aoi_bounds, cache_path):
    """Simple coregistration wrapper with caching."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        return np.load(cache_path)
    
    try:
        from scripts.preprocessing.numpy_efficient_pipeline import coregister_lr_to_hr
        hr_coreg = coregister_lr_to_hr(hr_array, lr_array, aoi_bounds)
        np.save(cache_path, hr_coreg)
        return hr_coreg
    except Exception as e:
        np.save(cache_path, hr_array)
        return hr_array


def harmonize_planet_to_sentinel2_wrapper(hr_array, lr_array, cache_path):
    """Simple harmonization wrapper with caching."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        return np.load(cache_path)
    
    try:
        from scripts.preprocessing.numpy_efficient_pipeline import harmonize_planet_to_sentinel2
        hr_harm = harmonize_planet_to_sentinel2(hr_array, lr_array)
        np.save(cache_path, hr_harm)
        return hr_harm
    except Exception as e:
        np.save(cache_path, hr_array)
        return hr_array


# =============================================================================
# TILE PROCESSING
# =============================================================================

def process_tile(tile_info):
    """
    Process a single tile across all temporal frames.
    """
    tile_idx = tile_info['tile_idx']
    tile_bounds = tile_info['tile_bounds']
    sequence = tile_info['sequence']
    planet_collection = tile_info['planet_collection']
    s2_collection = tile_info['s2_collection']
    default_planet_bands = tile_info['planet_bands']
    default_sentinel_bands = tile_info['sentinel_bands']
    date_band_config = tile_info.get('date_band_config', {})
    zenith_lookup = tile_info.get('zenith_lookup', {'s2': {}, 'planet': {}})
    cache_dir = tile_info['cache_dir']
    region_name = tile_info['region_name']
    seq_idx = tile_info['seq_idx']
    
    lr_frames = []
    hr_frames = []
    dates = []
    s2_zenith_angles = []
    planet_zenith_angles = []
    
    # Determine sequence band strategy before downloading
    sequence_band_modes = []
    for pair in sequence:
        frame_config = date_band_config.get(pair['planet_date'], {'mode': '4-band'})
        sequence_band_modes.append(frame_config['mode'])
    
    # Determine unified strategy
    if all(mode == '8-band' for mode in sequence_band_modes):
        sequence_mode = '8-band'
        sequence_planet_bands = ['B2', 'B4', 'B6', 'B7', 'B8']
        sequence_sentinel_bands = ['B2', 'B3', 'B4', 'B5', 'B8A']
    else:
        sequence_mode = '4-band'
        sequence_planet_bands = default_planet_bands
        sequence_sentinel_bands = default_sentinel_bands
    
    # Download all frames
    downloaded_frames, cache_stats = download_all_frames_batched(
        sequence=sequence,
        tile_bounds=tile_bounds,
        planet_collection=planet_collection,
        s2_collection=s2_collection,
        planet_bands=sequence_planet_bands,
        sentinel_bands=sequence_sentinel_bands,
        cache_dir=cache_dir,
        tile_size_px=256
    )
    
    if len(downloaded_frames) != len(sequence):
        return None
    
    # Process downloaded frames
    all_frames_ok = True
    lr_20m_frames = []
    frame_quality_stats = []
    
    for frame_idx in range(len(sequence)):
        if frame_idx not in downloaded_frames:
            all_frames_ok = False
            break
            
        frame_data = downloaded_frames[frame_idx]
        hr_array = frame_data['hr']
        lr_data = frame_data['lr']
        pair = frame_data['pair']
        
        # Extract arrays
        if isinstance(lr_data, dict):
            lr_array_10m = lr_data.get('bands_10m')
            lr_array_20m = lr_data.get('bands_20m')
        else:
            lr_array_10m = lr_data
            lr_array_20m = None
        
        if lr_array_10m is None:
            all_frames_ok = False
            break
        
        try:
            # Cache keys
            cache_key_coreg = f"{pair['planet_id']}|{pair['s2_id']}|{tile_bounds}|coreg"
            cache_key_harm = f"{pair['planet_id']}|{pair['s2_id']}|{tile_bounds}|harm"
            cache_hash_coreg = hashlib.md5(cache_key_coreg.encode()).hexdigest()[:16]
            cache_hash_harm = hashlib.md5(cache_key_harm.encode()).hexdigest()[:16]
            cache_path_coreg = cache_dir / "coregistered" / f"tile_{cache_hash_coreg}.npy"
            cache_path_harm = cache_dir / "harmonized" / f"tile_{cache_hash_harm}.npy"
            
            # Geographic coordinates
            aoi_bounds_coords = [
                [tile_bounds[0], tile_bounds[1]],
                [tile_bounds[2], tile_bounds[1]],
                [tile_bounds[2], tile_bounds[3]],
                [tile_bounds[0], tile_bounds[3]]
            ]
            
            # Coregistration (uses first 3 bands of HR)
            hr_for_coreg = hr_array[:, :, :min(3, hr_array.shape[-1])]
            lr_for_coreg = lr_array_10m[:, :, :min(3, lr_array_10m.shape[-1])]
            
            hr_array = coregister_lr_to_hr_wrapper(
                hr_array=hr_array,
                lr_array=lr_for_coreg,
                aoi_bounds=aoi_bounds_coords,
                cache_path=cache_path_coreg
            )
            
            # Harmonization (only applies to the first 3 bands that match LR 10m)
            hr_array = harmonize_planet_to_sentinel2_wrapper(
                hr_array=hr_array,
                lr_array=lr_for_coreg,
                cache_path=cache_path_harm
            )
            
            # NOTE: LR and HR have DIFFERENT band counts in 8-band mode:
            # - HR: 5 bands (B2, B4, B6, B7, B8)
            # - LR 10m: 3 bands (B2, B3, B4)
            # - LR 20m: 2 bands (B5, B8A)
            # We keep all bands - no truncation!
            lr_array = lr_array_10m
            
            lr_20m_frames.append(lr_array_20m)
            
            # Zenith angles from lookup
            s2_zenith = zenith_lookup.get('s2', {}).get(pair['s2_id'])
            planet_zenith = zenith_lookup.get('planet', {}).get(pair['planet_id'])
            s2_zenith_angles.append(s2_zenith)
            planet_zenith_angles.append(planet_zenith)
            
            # Quality stats
            lr_stats = compute_frame_quality_stats(lr_array)
            hr_stats = compute_frame_quality_stats(hr_array)
            frame_quality_stats.append({
                'frame_idx': frame_idx,
                'lr_valid_frac': lr_stats.valid_fraction,
                'hr_valid_frac': hr_stats.valid_fraction,
                'is_good': lr_stats.is_good and hr_stats.is_good
            })
            
            lr_frames.append(lr_array)
            hr_frames.append(hr_array)
            dates.append((pair['s2_date'], pair['planet_date']))
            
        except Exception as e:
            all_frames_ok = False
            break
    
    if all_frames_ok and len(lr_frames) == len(sequence):
        # Stack 20m bands if available
        lr_20m_stack = None
        if lr_20m_frames and all(f is not None for f in lr_20m_frames):
            try:
                lr_20m_stack = np.stack(lr_20m_frames, axis=0)
            except:
                pass
        
        sample = TrainingSample(
            lr_sequence=np.stack(lr_frames, axis=0),
            hr_sequence=np.stack(hr_frames, axis=0),
            temporal_dates=[d[0] for d in dates],
            planet_dates=[d[1] for d in dates],
            region_name=region_name,
            sequence_id=seq_idx,
            tile_id=tile_idx,
            tile_bounds={
                'bounds': tile_bounds, 
                'band_mode': sequence_mode,
                'planet_bands': sequence_planet_bands,
                'sentinel_bands': sequence_sentinel_bands
            },
            s2_solar_zenith=s2_zenith_angles,
            planet_solar_zenith=planet_zenith_angles,
            lr_20m_sequence=lr_20m_stack,
            frame_quality=frame_quality_stats
        )
        return sample
    else:
        return None


def process_single_sequence(sequence_info):
    """Process a single sequence with multiple tiles in parallel."""
    sequence = sequence_info['sequence']
    valid_tiles = sequence_info['valid_tiles']
    planet_collection = sequence_info['planet_collection']
    s2_collection = sequence_info['s2_collection']
    planet_bands = sequence_info['planet_bands']
    sentinel_bands = sequence_info['sentinel_bands']
    date_band_config = sequence_info.get('date_band_config', {})
    zenith_lookup = sequence_info.get('zenith_lookup', {'s2': {}, 'planet': {}})
    cache_dir = sequence_info['cache_dir']
    region_name = sequence_info['region_name']
    seq_idx = sequence_info['seq_idx']
    max_samples_per_seq = sequence_info['max_samples_per_seq']
    tile_workers = sequence_info.get('tile_workers', 1)
    
    samples = []
    
    print(f"\n📦 Processing sequence {seq_idx} with {len(valid_tiles)} valid tiles...")
    print(f"   Using {tile_workers} parallel worker(s) for tile processing")
    
    tiles_to_process = valid_tiles[:max_samples_per_seq]
    tile_infos = []
    
    for tile_idx, tile_bounds in enumerate(tiles_to_process):
        tile_info = {
            'tile_idx': tile_idx,
            'tile_bounds': tile_bounds,
            'sequence': sequence,
            'planet_collection': planet_collection,
            's2_collection': s2_collection,
            'planet_bands': planet_bands,
            'sentinel_bands': sentinel_bands,
            'date_band_config': date_band_config,
            'zenith_lookup': zenith_lookup,
            'cache_dir': cache_dir,
            'region_name': region_name,
            'seq_idx': seq_idx,
        }
        tile_infos.append(tile_info)
    
    # Process tiles in parallel
    if tile_workers > 1:
        with ThreadPoolExecutor(max_workers=tile_workers) as executor:
            future_to_tile = {
                executor.submit(process_tile, tile_info): (tile_info['tile_idx'], tile_info['tile_bounds'])
                for tile_info in tile_infos
            }
            
            processed = 0
            for future in as_completed(future_to_tile):
                tile_idx, tile_bounds = future_to_tile[future]
                processed += 1
                
                try:
                    sample = future.result()
                    if sample is not None:
                        samples.append(sample)
                        if processed % 10 == 0 or processed == len(tiles_to_process):
                            print(f"   Progress: {processed}/{len(tiles_to_process)}, {len(samples)} valid")
                except Exception as e:
                    pass
    else:
        for tile_idx, tile_info in enumerate(tile_infos):
            try:
                sample = process_tile(tile_info)
                if sample is not None:
                    samples.append(sample)
            except:
                pass
    
    print(f"   Sequence {seq_idx}: Generated {len(samples)} samples")
    return samples


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='STA-1E Grid-based parallel dataset generation')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to generate')
    parser.add_argument('--output-dir', type=str, 
                       default='./data/datasets/sta1e_grid_parallel',
                       help='Output directory for dataset')
    parser.add_argument('--cache-dir', type=str,
                       default='./cache/sta1e_grid_parallel',
                       help='Cache directory')
    parser.add_argument('--sequence-length', type=int, default=3,
                       help='Temporal sequence length')
    parser.add_argument('--coverage-threshold', type=float, default=90.0,
                       help='Minimum coverage percentage for valid tiles')
    parser.add_argument('--tile-workers', type=int, default=20,
                       help='Number of parallel workers for tile processing')
    parser.add_argument('--coverage-workers', type=int, default=16,
                       help='Number of parallel workers for coverage checking')
    parser.add_argument('--year', type=str, default=None,
                       help='Specific year to process (filters regions by start date)')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize data to [0, 1] range (default: True)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                       help='Keep raw reflectance values')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"🌐 STA-1E GRID-BASED PARALLEL DATASET GENERATION")
    print("=" * 80)
    print(f"\n⚙️  CONFIGURATION:")
    print("─" * 80)
    print(f"  Strategy:               Grid-based systematic tile sampling")
    print(f"  Coverage pre-check:     {args.coverage_threshold}% minimum (across all frames)")
    print(f"  Tile workers:           {args.tile_workers} parallel worker(s) per sequence")
    print(f"  Coverage workers:       {args.coverage_workers} parallel workers")
    print(f"  Sequence length:        {args.sequence_length} temporal frames")
    print(f"  Year filter:            {args.year if args.year else 'All years'}")
    print(f"  Normalization:          {'[0, 1]' if args.normalize else 'Raw (0-10000)'}")
    print(f"\n🎯 TARGET:")
    print("─" * 80)
    print(f"  Max samples:            {args.max_samples if args.max_samples else 'No limit'}")
    print(f"  Output directory:       {args.output_dir}")
    print(f"  Cache directory:        {args.cache_dir}")
    print("=" * 80)
    
    start_time = time.time()
    
    # Profiling
    profile_times = {
        'find_pairs': 0,
        'zenith_extraction': 0,
        'band_detection': 0,
        'tile_grid': 0,
        'coverage_check': 0,
        'tile_downloads': 0,
        'total': 0
    }
    
    # Initialize GEE
    try:
        ee.Initialize()
    except:
        print("Initializing GEE...")
        from Auth.AuthGEE import ee
    
    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter regions for STA-1E
    sta1e_regions = [r for r in regions if r['name'] == 'STA-1E']
    
    if args.year:
        sta1e_regions = [r for r in sta1e_regions if r['start'].startswith(args.year)]
    
    if not sta1e_regions:
        print(f"❌ No STA-1E regions found for year {args.year}")
        print(f"   Available STA-1E periods:")
        for r in regions:
            if r['name'] == 'STA-1E':
                print(f"     - {r['start']} to {r['end']}")
        return
    
    print(f"\n📍 Found {len(sta1e_regions)} STA-1E time periods to process")
    for r in sta1e_regions:
        print(f"   - {r['start']} to {r['end']}")
    
    all_samples = []
    
    for region_idx, region_config in enumerate(sta1e_regions):
        print(f"\n{'='*80}")
        print(f"PROCESSING {region_config['name']} ({region_config['start']} to {region_config['end']})")
        print(f"{'='*80}")
        
        start_date = region_config["start"]
        end_date = region_config["end"]
        
        # Find image pairs
        print(f"\n🔍 Finding image pairs...")
        print(f"   Checking ALL Planet repositories for maximum coverage...")
        
        t_find_pairs = time.time()
        
        # Cache for pairs
        pairs_cache_key = f"{region_config['name']}|{start_date}|{end_date}"
        pairs_cache_hash = hashlib.md5(pairs_cache_key.encode()).hexdigest()[:16]
        pairs_cache_file = cache_dir / f"matched_pairs_{pairs_cache_hash}.pkl"
        
        cached_matched_pairs = None
        if pairs_cache_file.exists():
            try:
                with open(pairs_cache_file, 'rb') as f:
                    cached_matched_pairs = pickle.load(f)
                print(f"   ✅ Loaded {len(cached_matched_pairs)} cached matched pairs")
            except:
                cached_matched_pairs = None
        
        # Call find_pairs to get collections
        pair_results = find_pairs(
            region_config["aoi"],
            start_date,
            end_date,
            PLANET_COLLECTION_ID,
            planet_coll_id_fallback=PLANET_COLLECTION_ID_FALLBACK,
            planet_coll_id_additional=PLANET_COLLECTION_ID_ADDITIONAL
        )
        
        if cached_matched_pairs is not None:
            matched_pairs = cached_matched_pairs
        else:
            matched_pairs = pair_results.get('matched_pairs', [])
            try:
                with open(pairs_cache_file, 'wb') as f:
                    pickle.dump(matched_pairs, f)
                print(f"   ✅ Cached {len(matched_pairs)} matched pairs")
            except:
                pass
        
        s2_collection = pair_results.get('s2_collection')
        planet_collection = pair_results.get('planet_collection')
        
        profile_times['find_pairs'] += time.time() - t_find_pairs
        print(f"   Found {len(matched_pairs)} matched pairs")
        
        if len(matched_pairs) < args.sequence_length:
            print(f"   ⚠️ Insufficient pairs for sequence length {args.sequence_length}")
            continue
        
        # Batched zenith extraction
        print(f"\n☀️  BATCHED SOLAR ZENITH EXTRACTION")
        
        zenith_cache_file = cache_dir / f"zenith_data_{pairs_cache_hash}.pkl"
        zenith_lookup = None
        
        if zenith_cache_file.exists():
            try:
                with open(zenith_cache_file, 'rb') as f:
                    zenith_lookup = pickle.load(f)
                print(f"   ✅ Loaded cached zenith data")
            except:
                zenith_lookup = None
        
        if zenith_lookup is None:
            print(f"   Extracting zenith angles...")
            t_zenith = time.time()
            
            try:
                s2_ids = [p['s2_id'] for p in matched_pairs]
                s2_filtered = s2_collection.filter(ee.Filter.inList('id_pref', s2_ids))
                s2_zenith_list = s2_filtered.aggregate_array('MEAN_SOLAR_ZENITH_ANGLE').getInfo()
                s2_id_list = s2_filtered.aggregate_array('id_pref').getInfo()
                
                planet_ids = [p['planet_id'] for p in matched_pairs]
                planet_filtered = planet_collection.filter(ee.Filter.inList('id_pref', planet_ids))
                planet_elevation_list = planet_filtered.aggregate_array('sun_elevation').getInfo()
                planet_id_list = planet_filtered.aggregate_array('id_pref').getInfo()
                
                s2_zenith_map = dict(zip(s2_id_list, s2_zenith_list))
                planet_zenith_map = {}
                for pid, elev in zip(planet_id_list, planet_elevation_list):
                    if elev is not None:
                        planet_zenith_map[pid] = 90.0 - float(elev)
                    else:
                        planet_zenith_map[pid] = None
                
                zenith_lookup = {'s2': s2_zenith_map, 'planet': planet_zenith_map}
                
                with open(zenith_cache_file, 'wb') as f:
                    pickle.dump(zenith_lookup, f)
                
                profile_times['zenith_extraction'] += time.time() - t_zenith
                print(f"   ✓ Zenith extraction: {time.time() - t_zenith:.1f}s")
                
            except Exception as e:
                print(f"   ⚠️ Zenith extraction failed: {e}")
                zenith_lookup = {'s2': {}, 'planet': {}}
        
        # Dynamic band detection
        print(f"\n🔬 DYNAMIC BAND DETECTION")
        
        dates_hash = hashlib.md5("_".join(sorted([p['planet_date'] for p in matched_pairs])).encode()).hexdigest()[:16]
        band_cache_file = cache_dir / f"band_detection_{dates_hash}.pkl"
        
        if band_cache_file.exists():
            print(f"   ✅ Found cached band detection results!")
            with open(band_cache_file, 'rb') as f:
                date_band_config = pickle.load(f)
            modes_8band = sum(1 for cfg in date_band_config.values() if cfg['mode'] == '8-band')
            modes_4band = len(date_band_config) - modes_8band
            print(f"   📊 {modes_8band} dates with 8-band, {modes_4band} with 4-band")
        else:
            print(f"   Detecting band availability...")
            t_detect = time.time()
            
            all_planet_dates = list(set([p['planet_date'] for p in matched_pairs]))
            date_band_config = {}
            
            def detect_bands_for_date(pdate):
                try:
                    planet_img = planet_collection.filter(ee.Filter.eq('date_key', pdate)).first()
                    available_bands = planet_img.bandNames().getInfo()
                    
                    has_extended_bands = any(band in available_bands for band in ['B5', 'B6', 'B7', 'B8'])
                    
                    if has_extended_bands and all(band in available_bands for band in ['B5', 'B6', 'B7', 'B8']):
                        return pdate, {
                            'mode': '8-band',
                            'planet_bands': ['B2', 'B4', 'B6', 'B7', 'B8'],
                            'sentinel_bands': ['B2', 'B3', 'B4', 'B5', 'B8A']
                        }
                    else:
                        return pdate, {
                            'mode': '4-band',
                            'planet_bands': ['B1', 'B2', 'B3', 'B4'],
                            'sentinel_bands': ['B2', 'B3', 'B4', 'B8A']
                        }
                except Exception as e:
                    return pdate, {
                        'mode': '4-band',
                        'planet_bands': ['B1', 'B2', 'B3', 'B4'],
                        'sentinel_bands': ['B2', 'B3', 'B4', 'B8A']
                    }
            
            with ThreadPoolExecutor(max_workers=min(20, len(all_planet_dates))) as executor:
                futures = {executor.submit(detect_bands_for_date, pdate): pdate for pdate in all_planet_dates}
                completed = 0
                for future in as_completed(futures):
                    pdate, config = future.result()
                    date_band_config[pdate] = config
                    completed += 1
                    if completed % 10 == 0 or completed == len(all_planet_dates):
                        print(f"   Progress: {completed}/{len(all_planet_dates)} dates checked")
            
            with open(band_cache_file, 'wb') as f:
                pickle.dump(date_band_config, f)
            
            modes_8band = sum(1 for cfg in date_band_config.values() if cfg['mode'] == '8-band')
            modes_4band = len(all_planet_dates) - modes_8band
            profile_times['band_detection'] += time.time() - t_detect
            print(f"   📊 {modes_8band} dates with 8-band, {modes_4band} with 4-band")
        
        # Default bands
        planet_bands = ['B1', 'B2', 'B3', 'B4']
        sentinel_bands = ['B2', 'B3', 'B4', 'B8A']
        
        # Process sequences
        max_possible_sequences = len(matched_pairs) - args.sequence_length + 1
        remaining = (args.max_samples - len(all_samples)) if args.max_samples else max_possible_sequences
        num_sequences = min(remaining, max_possible_sequences)
        
        print(f"\n📅 Processing up to {num_sequences} sequences")
        
        for seq_idx in range(num_sequences):
            if args.max_samples and len(all_samples) >= args.max_samples:
                break
            
            sequence = matched_pairs[seq_idx:seq_idx + args.sequence_length]
            
            if len(sequence) < args.sequence_length:
                break
            
            print(f"\n{'='*60}")
            print(f"SEQUENCE {seq_idx + 1}/{num_sequences}")
            print(f"Dates: {[p['s2_date'] for p in sequence]}")
            print(f"{'='*60}")
            
            # Get images for coverage checking
            print(f"\n📅 Getting images for coverage check...")
            planet_images = []
            s2_images = []
            
            for frame_idx, pair in enumerate(sequence):
                planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
                s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
                planet_images.append(planet_img)
                s2_images.append(s2_img)
            
            # Cache coverage results
            sequence_dates = tuple(sorted([p['s2_date'] for p in sequence]))
            dates_str = "_".join(sequence_dates)
            cache_hash = hashlib.md5(dates_str.encode()).hexdigest()[:16]
            coverage_cache_file = cache_dir / f"coverage_{cache_hash}.pkl"
            
            if coverage_cache_file.exists():
                print(f"\n   ✅ Found cached coverage results!")
                with open(coverage_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    valid_tiles = cached_data['valid_tiles']
                    total_checked = cached_data.get('total_checked', len(valid_tiles))
                print(f"   ⚡ Loaded {len(valid_tiles)}/{total_checked} valid tiles from cache")
            else:
                # Generate grid and check coverage
                print(f"\n🗺️  Generating tile grid...")
                t_tile_grid = time.time()
                all_tiles = generate_tile_grid(region_config["aoi"], tile_size_deg=0.007)
                profile_times['tile_grid'] += time.time() - t_tile_grid
                
                print(f"\n🔍 Batched coverage check...")
                t_coverage = time.time()
                valid_tiles = []
                
                try:
                    # Create tile FeatureCollection
                    tile_features = []
                    for idx, tile_bounds in enumerate(all_tiles):
                        tile_geom = ee.Geometry.Rectangle(tile_bounds)
                        tile_features.append(ee.Feature(tile_geom, {
                            'tile_idx': idx,
                            'bounds': tile_bounds
                        }))
                    tile_fc = ee.FeatureCollection(tile_features)
                    
                    # Check all images in parallel
                    all_images = planet_images + s2_images
                    image_names = [f"planet_{i}" for i in range(len(planet_images))] + \
                                 [f"s2_{i}" for i in range(len(s2_images))]
                    
                    tile_coverage = {idx: {} for idx in range(len(all_tiles))}
                    
                    def check_image_coverage(img_data):
                        img, img_name = img_data
                        band_name = 'B1' if 'planet' in img_name else 'B2'
                        valid_mask = img.select(band_name).mask()
                        coverage_results = valid_mask.reduceRegions(
                            collection=tile_fc,
                            reducer=ee.Reducer.mean(),
                            scale=30
                        ).getInfo()
                        return img_name, coverage_results
                    
                    with ThreadPoolExecutor(max_workers=len(all_images)) as executor:
                        futures = [executor.submit(check_image_coverage, (img, name)) 
                                  for img, name in zip(all_images, image_names)]
                        
                        for future in as_completed(futures):
                            img_name, coverage_results = future.result()
                            for feature in coverage_results['features']:
                                tile_idx = feature['properties']['tile_idx']
                                coverage = feature['properties'].get('mean', 0) or 0
                                tile_coverage[tile_idx][img_name] = coverage * 100
                    
                    # Filter valid tiles
                    for idx, tile_bounds in enumerate(all_tiles):
                        coverages = tile_coverage[idx]
                        if all(cov >= args.coverage_threshold for cov in coverages.values()):
                            valid_tiles.append(tile_bounds)
                    
                    print(f"   ✓ Valid tiles: {len(valid_tiles)}/{len(all_tiles)}")
                    
                except Exception as e:
                    print(f"   ❌ Batched coverage failed: {e}")
                    print(f"   ⚠️ Falling back to per-tile checking...")
                    
                    def check_tile_all_frames(tile_bounds):
                        tile_geom = ee.Geometry.Rectangle(tile_bounds)
                        for planet_img, s2_img in zip(planet_images, s2_images):
                            if not has_sufficient_coverage(planet_img, tile_geom, 'B1', args.coverage_threshold):
                                return False
                            if not has_sufficient_coverage(s2_img, tile_geom, 'B2', args.coverage_threshold):
                                return False
                        return True
                    
                    with ThreadPoolExecutor(max_workers=args.coverage_workers) as executor:
                        futures = {executor.submit(check_tile_all_frames, tb): tb for tb in all_tiles}
                        for future in as_completed(futures):
                            if future.result():
                                valid_tiles.append(futures[future])
                    
                    print(f"   ✓ Valid tiles (fallback): {len(valid_tiles)}/{len(all_tiles)}")
                
                profile_times['coverage_check'] += time.time() - t_coverage
                
                # Cache results
                cache_data = {
                    'valid_tiles': valid_tiles,
                    'total_checked': len(all_tiles),
                    'sequence_dates': sequence_dates,
                    'coverage_threshold': args.coverage_threshold
                }
                with open(coverage_cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
            
            if not valid_tiles:
                print(f"   ⚠️ No valid tiles, skipping sequence")
                continue
            
            # Calculate samples for this sequence
            remaining_samples = (args.max_samples - len(all_samples)) if args.max_samples else len(valid_tiles)
            max_samples_this_seq = min(len(valid_tiles), remaining_samples)
            
            print(f"\n⚡ Processing {max_samples_this_seq} tiles...")
            
            t_downloads = time.time()
            
            sequence_info = {
                'sequence': sequence,
                'valid_tiles': valid_tiles,
                'planet_collection': planet_collection,
                's2_collection': s2_collection,
                'planet_bands': planet_bands,
                'sentinel_bands': sentinel_bands,
                'date_band_config': date_band_config,
                'zenith_lookup': zenith_lookup,
                'cache_dir': cache_dir,
                'region_name': region_config['name'],
                'seq_idx': seq_idx,
                'max_samples_per_seq': max_samples_this_seq,
                'tile_workers': args.tile_workers,
            }
            
            samples = process_single_sequence(sequence_info)
            
            profile_times['tile_downloads'] += time.time() - t_downloads
            
            all_samples.extend(samples)
            
            print(f"   Total samples so far: {len(all_samples)}")
            
            if args.max_samples and len(all_samples) >= args.max_samples:
                break
    
    # Save dataset
    if len(all_samples) == 0:
        print("\n❌ No samples generated!")
        return
    
    # Normalize
    if args.normalize:
        print(f"\n🔄 Normalizing {len(all_samples)} samples to [0, 1] range...")
        for sample in all_samples:
            normalize_sample(sample)
        print("   ✓ Normalization complete")
    
    dataset_path = output_dir / 'dataset.pkl'
    print(f"\n💾 Saving {len(all_samples)} samples...")
    
    dataset = {
        'samples': all_samples,
        'config': {
            'region': 'STA-1E',
            'sequence_length': args.sequence_length,
            'num_samples': len(all_samples),
            'coverage_threshold': args.coverage_threshold,
            'grid_based': True,
            'normalized': args.normalize,
            'scale_factor': 10000.0 if args.normalize else None
        }
    }
    
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=4)
    
    profile_times['total'] = time.time() - start_time
    elapsed = profile_times['total']
    
    print("\n" + "=" * 80)
    print("✅ STA-1E GRID-BASED PARALLEL GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Samples generated: {len(all_samples)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes ({elapsed:.1f}s)")
    if len(all_samples) > 0:
        print(f"Avg time per sample: {elapsed/len(all_samples):.1f} seconds")
    
    # Profiling summary
    print("\n" + "─" * 80)
    print("⏱️  PROFILING BREAKDOWN:")
    print("─" * 80)
    print(f"  find_pairs():      {profile_times['find_pairs']:6.1f}s ({100*profile_times['find_pairs']/elapsed:5.1f}%)")
    print(f"  zenith_extract:    {profile_times['zenith_extraction']:6.1f}s ({100*profile_times['zenith_extraction']/elapsed:5.1f}%)")
    print(f"  band_detection:    {profile_times['band_detection']:6.1f}s ({100*profile_times['band_detection']/elapsed:5.1f}%)")
    print(f"  tile_grid:         {profile_times['tile_grid']:6.1f}s ({100*profile_times['tile_grid']/elapsed:5.1f}%)")
    print(f"  coverage_check:    {profile_times['coverage_check']:6.1f}s ({100*profile_times['coverage_check']/elapsed:5.1f}%)")
    print(f"  tile_downloads:    {profile_times['tile_downloads']:6.1f}s ({100*profile_times['tile_downloads']/elapsed:5.1f}%)")
    print("─" * 80)
    
    print(f"\nDataset saved: {dataset_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
