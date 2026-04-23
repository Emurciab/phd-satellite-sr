#!/usr/bin/env python3
"""
GRID-BASED PARALLEL dataset generation with coverage pre-checking
Features:
- Grid-based tile sampling (systematic coverage)
- Coverage pre-checking (avoid Planet mosaic gaps)
- Parallel sequence processing
- Direct tile downloads (no full image waste)
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from Auth.AuthGEE import find_pairs, PLANET_COLLECTION_ID, regions

# Secondary Planet collection for more data
PLANET_COLLECTION_ID_FALLBACK = 'projects/ee-eeamurciabb/assets/testSTAs'


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
    s2_solar_zenith: Optional[List[float]] = None   # Sentinel-2 MEAN_SOLAR_ZENITH_ANGLE per frame
    planet_solar_zenith: Optional[List[float]] = None  # PlanetScope zenith (90 - sun_elevation) per frame
    # Multi-resolution: 20m bands stored at native resolution (half spatial size of 10m)
    lr_20m_sequence: Optional[np.ndarray] = None  # (T, H/2, W/2, C_20m) - B8A etc at native 20m
    # LOCAL-SIDE frame quality statistics (computed AFTER download)
    # These flag frames with atmospheric issues (zeros from masked pixels, outliers, etc.)
    frame_quality: Optional[List[Dict]] = None  # Per-frame quality stats: {zero_frac, nan_frac, outlier_frac, valid_frac, is_good}


@dataclass
class FrameQualityStats:
    """Quality statistics for a single frame (computed locally from numpy arrays)"""
    zero_fraction: float      # Fraction of pixels that are exactly 0 (masked/bad)
    nan_fraction: float       # Fraction of pixels that are NaN
    outlier_fraction: float   # Fraction of pixels with extreme values (atmospheric issues)
    valid_fraction: float     # Fraction of "good" pixels (1 - zero - nan - outlier)
    mean_reflectance: float   # Mean of valid pixels (sanity check)
    std_reflectance: float    # Std of valid pixels
    is_good: bool             # True if frame passes quality thresholds
    
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
    
    Satellite data typically comes in surface reflectance * 10000 format.
    This normalizes to [0, 1] range for neural network training.
    
    Args:
        sample: TrainingSample with raw reflectance values
        scale_factor: Division factor (default 10000 for surface reflectance)
    
    Returns:
        TrainingSample with normalized arrays (in-place modification)
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
        FrameQualityStats with quality metrics and pass/fail flag
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
        return FrameQualityStats(
            zero_fraction=zero_fraction,
            nan_fraction=nan_fraction,
            outlier_fraction=0.0,
            valid_fraction=0.0,
            mean_reflectance=0.0,
            std_reflectance=0.0,
            is_good=False
        )
    
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
    
    return FrameQualityStats(
        zero_fraction=float(zero_fraction),
        nan_fraction=float(nan_fraction),
        outlier_fraction=float(outlier_fraction),
        valid_fraction=float(valid_fraction),
        mean_reflectance=mean_refl,
        std_reflectance=std_refl,
        is_good=is_good
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
        # IMPORTANT: Sentinel-2 has mixed resolutions (10m RGB + 20m NIR)
        # Resample all to 10m for consistent dimensions
        lr_selected = s2_img.select(sentinel_bands).clip(tile_geom).unmask(0)
        lr_resampled = lr_selected.reproject(crs='EPSG:4326', scale=lr_scale)
        lr_rect = lr_resampled.sampleRectangle(region=tile_geom, defaultValue=0)
        lr_info = lr_rect.getInfo()
        
        if not lr_info or 'properties' not in lr_info:
            return None, None
        
        lr_props = lr_info['properties']
        lr_arrays = [np.array(lr_props[b], dtype=np.float32) for b in sentinel_bands]
        
        # Check shapes
        lr_shapes = [arr.shape for arr in lr_arrays]
        if len(set(lr_shapes)) > 1:
            print(f"\n      ⚠️ LR band shape mismatch: {lr_shapes}")
            return None, None
        
        lr_array = np.stack(lr_arrays, axis=0)  # (C, H, W)
        lr_array = np.transpose(lr_array, (1, 2, 0))  # (H, W, C)
        
        return hr_array, lr_array
        
    except Exception as e:
        print(f"\n      Download error: {e}")
        return None, None


def download_tile_region_cached(planet_img, s2_img, tile_bounds, planet_bands, sentinel_bands,
                                planet_id, s2_id, cache_dir, hr_scale=3, lr_scale=10):
    """
    Download tile region with caching for 1073x speedup on re-runs.
    
    Cache key is based on: {planet_id}|{s2_id}|{bounds}|{bands}
    """
    import hashlib
    
    # Create stable cache key
    cache_key_str = f"{planet_id}|{s2_id}|{tile_bounds}|{planet_bands}|{sentinel_bands}"
    cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:16]
    
    cache_path = Path(cache_dir) / "raw_downloads" / f"tile_{cache_hash}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check cache first
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            return cached['hr_array'], cached['lr_array'], True  # True = cache hit
        except Exception as e:
            pass  # Cache corrupted, re-download
    
    # Download fresh
    hr_array, lr_array = download_tile_region(
        planet_img, s2_img, tile_bounds, planet_bands, sentinel_bands,
        hr_scale, lr_scale
    )
    
    # Save to cache if successful
    if hr_array is not None and lr_array is not None:
        try:
            np.savez_compressed(cache_path, hr_array=hr_array, lr_array=lr_array,
                              planet_bands=planet_bands, s2_bands=sentinel_bands)
        except Exception as e:
            pass  # Cache save failed, not critical
    
    return hr_array, lr_array, False  # False = cache miss


def _download_planet_tile(planet_img, tile_bounds, planet_bands, hr_scale=3):
    """Download just the Planet tile (for parallel downloading)"""
    try:
        tile_geom = ee.Geometry.Rectangle(tile_bounds)
        hr_selected = planet_img.select(planet_bands).clip(tile_geom).unmask(0)
        hr_resampled = hr_selected.reproject(crs='EPSG:4326', scale=hr_scale)
        hr_rect = hr_resampled.sampleRectangle(region=tile_geom, defaultValue=0)
        hr_info = hr_rect.getInfo()
        
        if not hr_info or 'properties' not in hr_info:
            return None
        
        hr_props = hr_info['properties']
        hr_arrays = [np.array(hr_props[b], dtype=np.float32) for b in planet_bands]
        
        hr_shapes = [arr.shape for arr in hr_arrays]
        if len(set(hr_shapes)) > 1:
            return None
        
        hr_array = np.stack(hr_arrays, axis=0)
        hr_array = np.transpose(hr_array, (1, 2, 0))
        return hr_array
    except Exception as e:
        return None


def _download_s2_tile(s2_img, tile_bounds, sentinel_bands, lr_scale=10):
    """
    Download Sentinel-2 tile at NATIVE resolutions (no upsampling).
    Downloads 10m and 20m bands in PARALLEL to maintain speed.
    
    Returns:
        Dict with 'bands_10m' and 'bands_20m' arrays at native resolutions,
        or None if failed
    """
    from concurrent.futures import ThreadPoolExecutor
    
    try:
        tile_geom = ee.Geometry.Rectangle(tile_bounds)
        
        # Separate 10m and 20m bands
        bands_10m = [b for b in sentinel_bands if b in ['B2', 'B3', 'B4', 'B8']]  # Native 10m
        bands_20m = [b for b in sentinel_bands if b in ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']]  # Native 20m
        
        result = {}
        
        def download_10m_bands():
            """Download 10m bands at native resolution"""
            if not bands_10m:
                return None, None
            try:
                lr_10m_selected = s2_img.select(bands_10m).clip(tile_geom).unmask(0)
                lr_10m_resampled = lr_10m_selected.reproject(crs='EPSG:4326', scale=10)
                lr_10m_rect = lr_10m_resampled.sampleRectangle(region=tile_geom, defaultValue=0)
                lr_10m_info = lr_10m_rect.getInfo()
                
                if lr_10m_info and 'properties' in lr_10m_info:
                    lr_10m_props = lr_10m_info['properties']
                    lr_10m_arrays = [np.array(lr_10m_props[b], dtype=np.float32) for b in bands_10m]
                    if len(set(arr.shape for arr in lr_10m_arrays)) <= 1:
                        return np.transpose(np.stack(lr_10m_arrays, axis=0), (1, 2, 0)), bands_10m
                return None, None
            except:
                return None, None
        
        def download_20m_bands():
            """Download 20m bands at native resolution (NO upsampling!)"""
            if not bands_20m:
                return None, None
            try:
                lr_20m_selected = s2_img.select(bands_20m).clip(tile_geom).unmask(0)
                lr_20m_resampled = lr_20m_selected.reproject(crs='EPSG:4326', scale=20)  # Native 20m!
                lr_20m_rect = lr_20m_resampled.sampleRectangle(region=tile_geom, defaultValue=0)
                lr_20m_info = lr_20m_rect.getInfo()
                
                if lr_20m_info and 'properties' in lr_20m_info:
                    lr_20m_props = lr_20m_info['properties']
                    lr_20m_arrays = [np.array(lr_20m_props[b], dtype=np.float32) for b in bands_20m]
                    if len(set(arr.shape for arr in lr_20m_arrays)) <= 1:
                        return np.transpose(np.stack(lr_20m_arrays, axis=0), (1, 2, 0)), bands_20m
                return None, None
            except:
                return None, None
        
        # Download both resolutions in PARALLEL (maintains speed!)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_10m = executor.submit(download_10m_bands)
            future_20m = executor.submit(download_20m_bands)
            
            arr_10m, names_10m = future_10m.result()
            arr_20m, names_20m = future_20m.result()
        
        # Build result dict
        if arr_10m is not None:
            result['bands_10m'] = arr_10m
            result['band_names_10m'] = names_10m
        if arr_20m is not None:
            result['bands_20m'] = arr_20m
            result['band_names_20m'] = names_20m
        
        # Return result if we got at least the 10m bands
        if 'bands_10m' in result:
            return result
        return None
        
    except Exception as e:
        return None


def download_all_frames_parallel(sequence, tile_bounds, planet_collection, s2_collection,
                                  planet_bands, sentinel_bands, cache_dir, max_workers=6):
    """
    Download all frames for a tile in parallel (3.16x speedup).
    
    Downloads Planet and S2 for all frames concurrently instead of sequentially.
    """
    import hashlib
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = {}
    cache_stats = {'hits': 0, 'misses': 0}
    
    # Prepare download tasks
    tasks = []
    for frame_idx, pair in enumerate(sequence):
        # Create cache key for this frame
        cache_key_str = f"{pair['planet_id']}|{pair['s2_id']}|{tile_bounds}|{planet_bands}|{sentinel_bands}"
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:16]
        cache_path = Path(cache_dir) / "raw_downloads" / f"tile_{cache_hash}.npz"
        
        tasks.append({
            'frame_idx': frame_idx,
            'pair': pair,
            'cache_path': cache_path,
            'tile_bounds': tile_bounds,
            'planet_bands': planet_bands,
            'sentinel_bands': sentinel_bands
        })
    
    # Check cache first for all frames
    frames_to_download = []
    for task in tasks:
        if task['cache_path'].exists():
            try:
                cached = np.load(task['cache_path'], allow_pickle=True)
                # Handle both old format (lr_array) and new format (lr_10m, lr_20m)
                if 'lr_10m' in cached:
                    # New multi-resolution format
                    lr_data = {
                        'bands_10m': cached['lr_10m'],
                        'bands_20m': cached['lr_20m'] if 'lr_20m' in cached else None,
                        'band_names_10m': list(cached['band_names_10m']) if 'band_names_10m' in cached else None,
                        'band_names_20m': list(cached['band_names_20m']) if 'band_names_20m' in cached else None
                    }
                else:
                    # Old format - treat as all 10m (upsampled) for backward compatibility
                    lr_data = {
                        'bands_10m': cached['lr_array'],
                        'bands_20m': None,
                        'band_names_10m': None,
                        'band_names_20m': None
                    }
                results[task['frame_idx']] = {
                    'hr': cached['hr_array'],
                    'lr': lr_data,
                    'pair': task['pair']
                }
                cache_stats['hits'] += 1
                continue
            except:
                pass
        frames_to_download.append(task)
        cache_stats['misses'] += 1
    
    if not frames_to_download:
        # All frames cached!
        return results, cache_stats
    
    # Download missing frames in parallel
    def download_frame(task):
        pair = task['pair']
        planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
        s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
        
        # Download both in parallel using nested ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as inner_executor:
            hr_future = inner_executor.submit(_download_planet_tile, planet_img, 
                                              task['tile_bounds'], task['planet_bands'])
            lr_future = inner_executor.submit(_download_s2_tile, s2_img,
                                              task['tile_bounds'], task['sentinel_bands'])
            hr_array = hr_future.result()
            lr_data = lr_future.result()  # Now returns dict with bands_10m, bands_20m
        
        # Save to cache with multi-resolution format
        if hr_array is not None and lr_data is not None:
            task['cache_path'].parent.mkdir(parents=True, exist_ok=True)
            try:
                save_dict = {
                    'hr_array': hr_array,
                    'planet_bands': task['planet_bands'],
                    's2_bands': task['sentinel_bands']
                }
                # Save 10m bands
                if 'bands_10m' in lr_data and lr_data['bands_10m'] is not None:
                    save_dict['lr_10m'] = lr_data['bands_10m']
                    save_dict['band_names_10m'] = lr_data.get('band_names_10m', [])
                # Save 20m bands
                if 'bands_20m' in lr_data and lr_data['bands_20m'] is not None:
                    save_dict['lr_20m'] = lr_data['bands_20m']
                    save_dict['band_names_20m'] = lr_data.get('band_names_20m', [])
                np.savez_compressed(task['cache_path'], **save_dict)
            except:
                pass
        
        return task['frame_idx'], hr_array, lr_data, pair
    
    with ThreadPoolExecutor(max_workers=min(max_workers, len(frames_to_download))) as executor:
        futures = {executor.submit(download_frame, task): task for task in frames_to_download}
        
        for future in as_completed(futures):
            try:
                frame_idx, hr_array, lr_data, pair = future.result()
                if hr_array is not None and lr_data is not None:
                    results[frame_idx] = {'hr': hr_array, 'lr': lr_data, 'pair': pair}
            except Exception as e:
                pass
    
    return results, cache_stats


# =============================================================================
# OPTIMIZED BATCHED DOWNLOADS using ee.data.computePixels
# =============================================================================
# This is 5-40x faster than sampleRectangle by:
# 1. Using binary NPY format instead of JSON serialization
# 2. Downloading meta-tiles (e.g., 4x4 grid) in single requests
# 3. Splitting tiles locally with NumPy (essentially free)
# =============================================================================

import io

def _compute_pixels_download(img: ee.Image, bounds: List[float], bands: List[str],
                             width: int = 256, height: int = 256) -> Optional[np.ndarray]:
    """
    Download image region using ee.data.computePixels (5x faster than sampleRectangle).
    
    Args:
        img: ee.Image to download
        bounds: [min_lon, min_lat, max_lon, max_lat]
        bands: List of band names
        width: Output width in pixels
        height: Output height in pixels
    
    Returns:
        numpy array (H, W, C) or None if failed
    """
    try:
        # Prepare image
        selected = img.select(bands).unmask(0)
        
        # Build request with explicit grid
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
                    'translateY': bounds[3]  # Top-left Y
                },
                'crsCode': 'EPSG:4326'
            }
        }
        
        # Download binary data
        result_bytes = ee.data.computePixels(request)
        
        # Parse NPY format (structured array with band names as fields)
        arr = np.load(io.BytesIO(result_bytes))
        
        # Convert structured array to regular (H, W, C) array
        result = np.stack([arr[b].astype(np.float32) for b in bands], axis=-1)
        
        return result
        
    except Exception as e:
        return None


def download_metatile_batched(planet_img: ee.Image, s2_img: ee.Image,
                              tile_bounds_list: List[List[float]],
                              planet_bands: List[str], sentinel_bands: List[str],
                              hr_scale: float = 3.0, lr_scale: float = 10.0,
                              tile_size_px: int = 256) -> Dict[int, Dict]:
    """
    Download multiple tiles as a single meta-tile and split locally.
    
    This is the KEY optimization: instead of N separate GEE requests,
    we download one large region and split with NumPy (essentially free).
    
    Args:
        planet_img: Planet ee.Image
        s2_img: Sentinel-2 ee.Image  
        tile_bounds_list: List of tile bounds [[min_lon, min_lat, max_lon, max_lat], ...]
        planet_bands: Planet band names
        sentinel_bands: S2 band names (will download 10m and 20m separately)
        hr_scale: Planet resolution (meters)
        lr_scale: Sentinel-2 base resolution (meters)
        tile_size_px: Size of each tile in pixels
    
    Returns:
        Dict mapping tile_idx -> {'hr': array, 'lr': dict with bands_10m, bands_20m}
    """
    if not tile_bounds_list:
        return {}
    
    # Calculate meta-tile bounding box
    all_lons = [b[0] for b in tile_bounds_list] + [b[2] for b in tile_bounds_list]
    all_lats = [b[1] for b in tile_bounds_list] + [b[3] for b in tile_bounds_list]
    meta_bounds = [min(all_lons), min(all_lats), max(all_lons), max(all_lats)]
    
    # Calculate grid dimensions
    tile_width = tile_bounds_list[0][2] - tile_bounds_list[0][0]
    tile_height = tile_bounds_list[0][3] - tile_bounds_list[0][1]
    
    n_cols = int(round((meta_bounds[2] - meta_bounds[0]) / tile_width))
    n_rows = int(round((meta_bounds[3] - meta_bounds[1]) / tile_height))
    
    meta_width = n_cols * tile_size_px
    meta_height = n_rows * tile_size_px
    
    # =========================================================================
    # NATIVE RESOLUTION DOWNLOAD - Key SR fix!
    # Download LR at native 10m resolution instead of upsampling to 3m
    # Scale factor: 10m / 3m ≈ 3.33, so LR pixels = HR pixels / 3.33
    # For 256px HR tile: LR 10m ≈ 77px, LR 20m ≈ 38px
    # =========================================================================
    scale_factor_10m = lr_scale / hr_scale  # 10m / 3m = 3.33
    scale_factor_20m = 20.0 / hr_scale       # 20m / 3m = 6.67
    
    # LR tile sizes at native resolution (must be integers)
    lr_10m_tile_size = int(round(tile_size_px / scale_factor_10m))  # ~77 for 256px HR
    lr_20m_tile_size = int(round(tile_size_px / scale_factor_20m))  # ~38 for 256px HR
    
    # Meta-tile sizes for each resolution
    lr_10m_meta_width = n_cols * lr_10m_tile_size
    lr_10m_meta_height = n_rows * lr_10m_tile_size
    lr_20m_meta_width = n_cols * lr_20m_tile_size
    lr_20m_meta_height = n_rows * lr_20m_tile_size
    
    results = {}
    
    try:
        # Download Planet meta-tile (HR at 3m native resolution)
        hr_array = _compute_pixels_download(
            planet_img, meta_bounds, planet_bands,
            width=meta_width, height=meta_height
        )
        
        # Download S2 10m bands at NATIVE 10m resolution (NOT upsampled to 3m!)
        bands_10m = [b for b in sentinel_bands if b in ['B2', 'B3', 'B4', 'B8']]
        bands_20m = [b for b in sentinel_bands if b in ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']]
        
        lr_10m_array = None
        lr_20m_array = None
        
        if bands_10m:
            # 10m bands: download at native 10m resolution (~77px per tile)
            lr_10m_array = _compute_pixels_download(
                s2_img, meta_bounds, bands_10m,
                width=lr_10m_meta_width, height=lr_10m_meta_height
            )
        
        if bands_20m:
            # 20m bands: download at native 20m resolution (~38px per tile)
            lr_20m_array = _compute_pixels_download(
                s2_img, meta_bounds, bands_20m,
                width=lr_20m_meta_width, height=lr_20m_meta_height
            )
        
        if hr_array is None:
            return {}
        
        # Split meta-tiles into individual tiles
        for tile_idx, tile_bounds in enumerate(tile_bounds_list):
            # Calculate tile position in meta-tile grid
            col = int(round((tile_bounds[0] - meta_bounds[0]) / tile_width))
            row = int(round((meta_bounds[3] - tile_bounds[3]) / tile_height))  # Y is inverted
            
            # Extract HR tile (256x256 at 3m)
            hr_tile = hr_array[
                row * tile_size_px:(row + 1) * tile_size_px,
                col * tile_size_px:(col + 1) * tile_size_px,
                :
            ]
            
            # Extract LR tiles at NATIVE resolution
            lr_data = {}
            if lr_10m_array is not None:
                # Extract 10m tile (~77x77 native resolution)
                lr_data['bands_10m'] = lr_10m_array[
                    row * lr_10m_tile_size:(row + 1) * lr_10m_tile_size,
                    col * lr_10m_tile_size:(col + 1) * lr_10m_tile_size,
                    :
                ]
                lr_data['band_names_10m'] = bands_10m
            
            if lr_20m_array is not None:
                # Extract 20m tile (~38x38 native resolution)
                lr_data['bands_20m'] = lr_20m_array[
                    row * lr_20m_tile_size:(row + 1) * lr_20m_tile_size,
                    col * lr_20m_tile_size:(col + 1) * lr_20m_tile_size,
                    :
                ]
                lr_data['band_names_20m'] = bands_20m
            
            if hr_tile.shape[0] == tile_size_px and hr_tile.shape[1] == tile_size_px:
                results[tile_idx] = {'hr': hr_tile, 'lr': lr_data}
        
    except Exception as e:
        print(f"      ⚠️ Meta-tile download error: {e}")
    
    return results


def download_all_frames_batched(sequence: List[Dict], tile_bounds: List[float],
                                planet_collection, s2_collection,
                                planet_bands: List[str], sentinel_bands: List[str],
                                cache_dir: Path, tile_size_px: int = 256,
                                batch_size: int = 1) -> Tuple[Dict, Dict]:
    """
    Download all frames for a tile using batched computePixels.
    
    This is the optimized version of download_all_frames_parallel that uses
    ee.data.computePixels instead of sampleRectangle for ~5x speedup.
    
    Args:
        sequence: List of pair dicts with planet_id, s2_id
        tile_bounds: [min_lon, min_lat, max_lon, max_lat]
        planet_collection: ee.ImageCollection
        s2_collection: ee.ImageCollection
        planet_bands: List of Planet band names
        sentinel_bands: List of S2 band names
        cache_dir: Cache directory
        tile_size_px: Tile size in pixels
        batch_size: Number of frames to batch (1 = no batching across frames)
    
    Returns:
        (results_dict, cache_stats_dict)
    """
    import hashlib
    
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
    
    # Download missing frames using computePixels IN PARALLEL (20 workers for GEE)
    def download_frame_batched(task):
        """Download a single frame using computePixels"""
        pair = task['pair']
        frame_idx = task['frame_idx']
        
        try:
            planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
            s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
            
            # =========================================================================
            # NATIVE RESOLUTION DOWNLOAD - Key SR fix!
            # HR (Planet @ 3m): tile_size_px (256)
            # LR 10m (S2 @ 10m): tile_size_px / scale_factor_10m (~77)
            # LR 20m (S2 @ 20m): tile_size_px / scale_factor_20m (~38)
            # =========================================================================
            scale_factor_10m = 10.0 / 3.0  # 10m / 3m = 3.33
            scale_factor_20m = 20.0 / 3.0  # 20m / 3m = 6.67
            lr_10m_size = int(round(tile_size_px / scale_factor_10m))  # ~77
            lr_20m_size = int(round(tile_size_px / scale_factor_20m))  # ~38
            
            # Download HR (Planet) at 3m native resolution
            hr_array = _compute_pixels_download(
                planet_img, tile_bounds, planet_bands,
                width=tile_size_px, height=tile_size_px
            )
            
            # Download S2 at NATIVE resolutions (not upsampled!)
            bands_10m = [b for b in sentinel_bands if b in ['B2', 'B3', 'B4', 'B8']]
            bands_20m = [b for b in sentinel_bands if b in ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']]
            
            lr_data = {}
            
            if bands_10m:
                # 10m bands at native 10m resolution (~77x77)
                lr_10m = _compute_pixels_download(
                    s2_img, tile_bounds, bands_10m,
                    width=lr_10m_size, height=lr_10m_size
                )
                if lr_10m is not None:
                    lr_data['bands_10m'] = lr_10m
                    lr_data['band_names_10m'] = bands_10m
            
            if bands_20m:
                # 20m bands at native 20m resolution (~38x38)
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
    
    # Use 20 parallel workers for GEE (optimal for API rate limits)
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


def has_sufficient_coverage(img: ee.Image, geom: ee.Geometry, band_name: str, 
                           min_percent: float = 90.0, scale: int = 30) -> bool:
    """
    Check if image has sufficient valid pixel coverage over geometry.
    
    Args:
        img: ee.Image to check
        geom: ee.Geometry region to check
        band_name: Band name to check for valid pixels
        min_percent: Minimum percentage of valid pixels required (0-100)
        scale: Scale in meters for reduction (larger = faster but less accurate)
    
    Returns:
        True if coverage >= min_percent, False otherwise
    """
    try:
        # Get valid pixel mask (non-masked pixels)
        valid_mask = img.select(band_name).mask()
        
        # Calculate mean of mask (1 = valid, 0 = masked/nodata)
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
        
        valid_percent = valid_fraction * 100.0
        return valid_percent >= min_percent
    except Exception as e:
        return False


def generate_tile_grid(aoi, tile_size_deg=0.007, max_tiles=None):
    """
    Generate systematic grid of tiles covering the AOI.
    
    Args:
        aoi: ee.Geometry for area of interest
        tile_size_deg: Tile size in degrees (~0.007 = 768m at 3m resolution)
        max_tiles: Maximum number of tiles to generate (None = all)
    
    Returns:
        List of [min_lon, min_lat, max_lon, max_lat]
    """
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    # Calculate grid dimensions
    num_cols = int(np.ceil((max_lon - min_lon) / tile_size_deg))
    num_rows = int(np.ceil((max_lat - min_lat) / tile_size_deg))
    
    print(f"   Grid dimensions: {num_rows} rows × {num_cols} cols = {num_rows * num_cols} potential tiles")
    
    # Generate grid tiles
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


def check_tile_coverage(planet_img: ee.Image, s2_img: ee.Image, 
                       tile_bounds: List, min_percent: float = 90.0) -> bool:
    """
    Check if a single tile has sufficient coverage.
    """
    tile_geom = ee.Geometry.Rectangle(tile_bounds)
    
    # Check Planet coverage (this is where gaps occur)
    planet_valid = has_sufficient_coverage(
        planet_img, tile_geom, 'B1', 
        min_percent=min_percent, scale=30
    )
    
    if not planet_valid:
        return False
    
    # Check Sentinel-2 coverage
    s2_valid = has_sufficient_coverage(
        s2_img, tile_geom, 'B2',
        min_percent=min_percent, scale=30
    )
    
    return s2_valid


def filter_valid_tiles(planet_img: ee.Image, s2_img: ee.Image, 
                      tile_bounds_list: List, min_percent: float = 90.0,
                      max_workers: int = 4) -> List:
    """
    Filter tiles to keep only those with sufficient coverage.
    Uses parallel processing for faster checking.
    
    Args:
        planet_img: Planet image
        s2_img: Sentinel-2 image  
        tile_bounds_list: List of tile bounds
        min_percent: Minimum coverage percentage
        max_workers: Number of parallel workers
    
    Returns:
        List of valid tile bounds with sufficient coverage
    """
    print(f"\n🔍 Checking coverage for {len(tile_bounds_list)} tiles (parallel: {max_workers} workers)...")
    
    valid_tiles = []
    checked = 0
    
    # Check tiles in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_tile = {
            executor.submit(check_tile_coverage, planet_img, s2_img, tile_bounds, min_percent): tile_bounds
            for tile_bounds in tile_bounds_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_tile):
            tile_bounds = future_to_tile[future]
            checked += 1
            
            try:
                is_valid = future.result()
                if is_valid:
                    valid_tiles.append(tile_bounds)
                
                if checked % 20 == 0 or checked == len(tile_bounds_list):
                    print(f"   Progress: {checked}/{len(tile_bounds_list)} checked, {len(valid_tiles)} valid ({100*len(valid_tiles)/checked:.1f}%)")
            except Exception as e:
                print(f"   Error checking tile: {e}")
    
    print(f"   ✓ Final: {len(valid_tiles)}/{len(tile_bounds_list)} tiles have sufficient coverage ({100*len(valid_tiles)/len(tile_bounds_list):.1f}%)")
    return valid_tiles


def coregister_lr_to_hr_wrapper(hr_array, lr_array, aoi_bounds, cache_path):
    """Simple coregistration wrapper"""
    from scripts.preprocessing.numpy_efficient_pipeline import coregister_lr_to_hr
    
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        return np.load(cache_path)
    
    try:
        hr_coreg = coregister_lr_to_hr(hr_array, lr_array, aoi_bounds)
        np.save(cache_path, hr_coreg)
        return hr_coreg
    except Exception as e:
        print(f"\n      AROSICS failed: {e}")
        # Fallback: just return HR as-is
        np.save(cache_path, hr_array)
        return hr_array


def harmonize_planet_to_sentinel2_wrapper(hr_array, lr_array, cache_path):
    """Simple harmonization wrapper"""
    from scripts.preprocessing.numpy_efficient_pipeline import harmonize_planet_to_sentinel2
    
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        return np.load(cache_path)
    
    try:
        hr_harm = harmonize_planet_to_sentinel2(hr_array, lr_array)
        np.save(cache_path, hr_harm)
        return hr_harm
    except Exception as e:
        print(f"\n      Harmonization warning: {e}")
        np.save(cache_path, hr_array)
        return hr_array


def get_solar_zenith_angles(planet_img: ee.Image, s2_img: ee.Image) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract solar zenith angles from Sentinel-2 and PlanetScope images.
    
    Solar zenith angle is important for aquatic vegetation analysis:
    - Optimal angle ~45° minimizes sun glint and maximizes water-leaving radiance
    - Can be used to penalize samples with extreme angles during training
    
    Args:
        planet_img: ee.Image for PlanetScope
        s2_img: ee.Image for Sentinel-2
    
    Returns:
        Tuple of (s2_zenith, planet_zenith) in degrees, or None if unavailable
    """
    s2_zenith = None
    planet_zenith = None
    
    try:
        # Sentinel-2: MEAN_SOLAR_ZENITH_ANGLE is directly available
        s2_zenith = s2_img.get('MEAN_SOLAR_ZENITH_ANGLE').getInfo()
        if s2_zenith is not None:
            s2_zenith = float(s2_zenith)
    except Exception as e:
        pass
    
    try:
        # PlanetScope: sun_elevation is available, zenith = 90 - elevation
        # Property name varies: 'sun_elevation', 'view:sun_elevation', 'sun_elevation_angle'
        sun_elevation = None
        
        # Try different property names used by Planet
        for prop_name in ['sun_elevation', 'view:sun_elevation', 'sun_elevation_angle']:
            try:
                sun_elevation = planet_img.get(prop_name).getInfo()
                if sun_elevation is not None:
                    break
            except:
                continue
        
        if sun_elevation is not None:
            # Convert elevation to zenith: zenith = 90 - elevation
            planet_zenith = 90.0 - float(sun_elevation)
    except Exception as e:
        pass
    
    return s2_zenith, planet_zenith


def process_tile(tile_info):
    """
    Process a single tile across all temporal frames.
    This is the worker function for parallel tile processing.
    
    Args:
        tile_info: Dict containing:
            - tile_idx: Tile index
            - tile_bounds: Geographic bounds [minLon, minLat, maxLon, maxLat]
            - sequence: List of matched pairs
            - planet_collection: Planet image collection
            - s2_collection: Sentinel-2 image collection
            - planet_bands: List of Planet band names (default fallback)
            - sentinel_bands: List of Sentinel-2 band names (default fallback)
            - date_band_config: Dict mapping planet_date -> band configuration
            - cache_dir: Cache directory path
            - region_name: Region name
            - seq_idx: Sequence index
    
    Returns:
        TrainingSample object or None if processing failed
    """
    tile_idx = tile_info['tile_idx']
    tile_bounds = tile_info['tile_bounds']
    sequence = tile_info['sequence']
    planet_collection = tile_info['planet_collection']
    s2_collection = tile_info['s2_collection']
    default_planet_bands = tile_info['planet_bands']
    default_sentinel_bands = tile_info['sentinel_bands']
    date_band_config = tile_info.get('date_band_config', {})  # Dynamic band config per date
    zenith_lookup = tile_info.get('zenith_lookup', {'s2': {}, 'planet': {}})  # Pre-extracted zenith angles
    cache_dir = tile_info['cache_dir']
    region_name = tile_info['region_name']
    seq_idx = tile_info['seq_idx']
    
    lr_frames = []
    hr_frames = []
    dates = []
    s2_zenith_angles = []      # Solar zenith angles for Sentinel-2
    planet_zenith_angles = []  # Solar zenith angles for PlanetScope
    
    # =========================================================================
    # DETERMINE SEQUENCE BAND STRATEGY BEFORE DOWNLOADING
    # =========================================================================
    # Check band modes for ALL frames in this sequence to decide on uniform strategy
    # If any frame is 4-band, use 4-band for entire sequence (temporal consistency)
    # This prevents data loss from post-download truncation
    sequence_band_modes = []
    for pair in sequence:
        frame_config = date_band_config.get(pair['planet_date'], {'mode': '4-band'})
        sequence_band_modes.append(frame_config['mode'])
    
    # Determine unified strategy for this sequence
    if all(mode == '8-band' for mode in sequence_band_modes):
        # All frames are 8-band SuperDove - use extended band configuration
        # Correct spectral matching (per user specification):
        # S2 B2 (Blue 10m)     ↔ Planet B2 (Blue)
        # S2 B3 (Green 10m)    ↔ Planet B4 (Green)  
        # S2 B4 (Red 10m)      ↔ Planet B6 (Red)
        # S2 B5 (RedEdge 20m)  ↔ Planet B7 (Red Edge)
        # S2 B8A (NIR 20m)     ↔ Planet B8 (NIR)
        sequence_mode = '8-band'
        sequence_planet_bands = ['B2', 'B4', 'B6', 'B7', 'B8']  # 5 bands
        sequence_sentinel_bands = ['B2', 'B3', 'B4', 'B5', 'B8A']  # 3×10m + 2×20m
    else:
        # Mixed or all 4-band - use 4-band for temporal consistency
        sequence_mode = '4-band'
        sequence_planet_bands = default_planet_bands  # ['B1', 'B2', 'B3', 'B4']
        sequence_sentinel_bands = default_sentinel_bands  # ['B2', 'B3', 'B4', 'B8A']
    
    # =========================================================================
    # PARALLEL DOWNLOAD ALL FRAMES (3.16x speedup with batched: ~15x)
    # =========================================================================
    # Download all frames - use batched computePixels if enabled (~5x faster)
    use_batched = tile_info.get('use_batched', True)
    
    if use_batched:
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
    else:
        downloaded_frames, cache_stats = download_all_frames_parallel(
            sequence=sequence,
            tile_bounds=tile_bounds,
            planet_collection=planet_collection,
            s2_collection=s2_collection,
            planet_bands=sequence_planet_bands,
            sentinel_bands=sequence_sentinel_bands,
            cache_dir=cache_dir,
            max_workers=6
        )
    
    # Check if all frames downloaded successfully
    if len(downloaded_frames) != len(sequence):
        return None
    
    # Process downloaded frames (coregistration, harmonization, zenith angles)
    all_frames_ok = True
    lr_20m_frames = []  # Store 20m bands separately
    
    for frame_idx in range(len(sequence)):
        if frame_idx not in downloaded_frames:
            all_frames_ok = False
            break
            
        frame_data = downloaded_frames[frame_idx]
        hr_array = frame_data['hr']
        lr_data = frame_data['lr']
        pair = frame_data['pair']
        
        # Extract 10m and 20m arrays from lr_data
        if isinstance(lr_data, dict):
            lr_array_10m = lr_data.get('bands_10m')
            lr_array_20m = lr_data.get('bands_20m')
            band_names_10m = lr_data.get('band_names_10m', [])
            band_names_20m = lr_data.get('band_names_20m', [])
        else:
            # Backward compatibility: old format was just numpy array
            lr_array_10m = lr_data
            lr_array_20m = None
            band_names_10m = []
            band_names_20m = []
        
        if lr_array_10m is None:
            all_frames_ok = False
            break
        
        try:
            # Apply coregistration and harmonization using 10m bands
            # FIX: Use stable cache key based on tile_bounds + pair IDs, NOT tile_idx
            # This prevents cache mismatches when tile processing order changes between runs
            import hashlib
            cache_key_coreg = f"{pair['planet_id']}|{pair['s2_id']}|{tile_bounds}|coreg"
            cache_key_harm = f"{pair['planet_id']}|{pair['s2_id']}|{tile_bounds}|harm"
            cache_hash_coreg = hashlib.md5(cache_key_coreg.encode()).hexdigest()[:16]
            cache_hash_harm = hashlib.md5(cache_key_harm.encode()).hexdigest()[:16]
            cache_path_coreg = cache_dir / "coregistered" / f"tile_{cache_hash_coreg}.npy"
            cache_path_harm = cache_dir / "harmonized" / f"tile_{cache_hash_harm}.npy"
            
            # Convert tile_bounds [minLon, minLat, maxLon, maxLat] to aoi_bounds format
            # AROSICS expects list of [lon, lat] coordinate pairs
            aoi_bounds_coords = [
                [tile_bounds[0], tile_bounds[1]],  # bottom-left
                [tile_bounds[2], tile_bounds[1]],  # bottom-right
                [tile_bounds[2], tile_bounds[3]],  # top-right
                [tile_bounds[0], tile_bounds[3]]   # top-left
            ]
            
            # Coregistration with REAL geographic coordinates (uses 10m bands)
            hr_array = coregister_lr_to_hr_wrapper(
                hr_array=hr_array,
                lr_array=lr_array_10m,
                aoi_bounds=aoi_bounds_coords,
                cache_path=cache_path_coreg
            )
            
            # Harmonization (uses 10m bands)
            hr_array = harmonize_planet_to_sentinel2_wrapper(
                hr_array=hr_array,
                lr_array=lr_array_10m,
                cache_path=cache_path_harm
            )
            
            # NOTE: LR and HR have DIFFERENT band counts in 8-band mode:
            # - HR: 5 bands (B2, B4, B6, B7, B8)
            # - LR 10m: 3 bands (B2, B3, B4)
            # - LR 20m: 2 bands (B5, B8A)
            # We keep all bands - no truncation! The model handles different input/output channels.
            lr_array = lr_array_10m
            
            # Store 20m bands separately (will be saved in sample)
            lr_20m_frames.append(lr_array_20m)
            
            # Extract solar zenith angles from PRE-COMPUTED lookup (8x faster!)
            # No more slow getInfo() calls - already extracted in batched zenith step
            s2_zenith = zenith_lookup.get('s2', {}).get(pair['s2_id'])
            planet_zenith = zenith_lookup.get('planet', {}).get(pair['planet_id'])
            s2_zenith_angles.append(s2_zenith)
            planet_zenith_angles.append(planet_zenith)
            
            lr_frames.append(lr_array)
            hr_frames.append(hr_array)
            dates.append((pair['s2_date'], pair['planet_date']))
            
        except Exception as e:
            import traceback
            print(f"\n      ❌ Frame {frame_idx} processing error: {e}")
            print(f"         HR shape: {hr_array.shape if hr_array is not None else 'None'}")
            print(f"         LR 10m shape: {lr_array_10m.shape if lr_array_10m is not None else 'None'}")
            traceback.print_exc()
            all_frames_ok = False
            break
    
    # Create sample if all frames downloaded
    if all_frames_ok and len(lr_frames) == len(sequence):
        # All frames should have uniform band counts now (determined before download)
        # Verify this is the case (sanity check)
        lr_band_counts = [f.shape[-1] for f in lr_frames]
        hr_band_counts = [f.shape[-1] for f in hr_frames]
        
        if len(set(lr_band_counts)) > 1 or len(set(hr_band_counts)) > 1:
            # This shouldn't happen with the uniform strategy, but log if it does
            print(f"\n      ⚠️ Unexpected band count mismatch: LR={lr_band_counts}, HR={hr_band_counts}")
            return None
        
        # =====================================================================
        # LOCAL-SIDE FRAME QUALITY STATISTICS (catches atmospheric issues)
        # =====================================================================
        # Compute quality stats for each frame AFTER download
        # This catches issues that server-side GEE coverage checking might miss
        frame_quality_stats = []
        all_frames_good = True
        
        for frame_idx, (lr_frame, hr_frame) in enumerate(zip(lr_frames, hr_frames)):
            # Compute quality stats for both LR and HR, use worst case
            lr_stats = compute_frame_quality_stats(lr_frame)
            hr_stats = compute_frame_quality_stats(hr_frame)
            
            # Combine stats (take minimum valid fraction, flag if either is bad)
            combined_stats = {
                'frame_idx': frame_idx,
                'lr_zero_frac': lr_stats.zero_fraction,
                'lr_nan_frac': lr_stats.nan_fraction,
                'lr_outlier_frac': lr_stats.outlier_fraction,
                'lr_valid_frac': lr_stats.valid_fraction,
                'hr_zero_frac': hr_stats.zero_fraction,
                'hr_nan_frac': hr_stats.nan_fraction,
                'hr_outlier_frac': hr_stats.outlier_fraction,
                'hr_valid_frac': hr_stats.valid_fraction,
                'is_good': lr_stats.is_good and hr_stats.is_good
            }
            frame_quality_stats.append(combined_stats)
            
            if not combined_stats['is_good']:
                all_frames_good = False
        
        # Separate S2 and Planet dates
        s2_dates = [d[0] for d in dates]
        planet_dates = [d[1] for d in dates]
        
        # Stack 20m bands if available (all frames should have same structure)
        lr_20m_stack = None
        if lr_20m_frames and all(f is not None for f in lr_20m_frames):
            try:
                lr_20m_stack = np.stack(lr_20m_frames, axis=0)
            except:
                lr_20m_stack = None  # Different shapes, skip 20m bands
        
        sample = TrainingSample(
            lr_sequence=np.stack(lr_frames, axis=0),
            hr_sequence=np.stack(hr_frames, axis=0),
            temporal_dates=s2_dates,
            planet_dates=planet_dates,
            region_name=region_name,
            sequence_id=seq_idx,
            tile_id=tile_idx,
            tile_bounds={
                'bounds': tile_bounds, 
                'band_mode': sequence_mode,  # '8-band' or '4-band'
                'planet_bands': sequence_planet_bands,
                'sentinel_bands': sequence_sentinel_bands
            },
            s2_solar_zenith=s2_zenith_angles,
            planet_solar_zenith=planet_zenith_angles,
            lr_20m_sequence=lr_20m_stack,  # 20m bands at native resolution
            frame_quality=frame_quality_stats  # LOCAL quality stats per frame
        )
        return sample
    else:
        return None


def process_single_sequence(sequence_info):
    """
    Process a single sequence with multiple tiles in parallel.
    
    Args:
        sequence_info: Dict with sequence data and configuration
    
    Returns:
        List of TrainingSample objects
    """
    sequence = sequence_info['sequence']
    valid_tiles = sequence_info['valid_tiles']
    planet_collection = sequence_info['planet_collection']
    s2_collection = sequence_info['s2_collection']
    planet_bands = sequence_info['planet_bands']
    sentinel_bands = sequence_info['sentinel_bands']
    date_band_config = sequence_info.get('date_band_config', {})  # Dynamic band config per date
    zenith_lookup = sequence_info.get('zenith_lookup', {'s2': {}, 'planet': {}})  # Pre-extracted zenith angles
    cache_dir = sequence_info['cache_dir']
    region_name = sequence_info['region_name']
    seq_idx = sequence_info['seq_idx']
    max_samples_per_seq = sequence_info['max_samples_per_seq']
    tile_workers = sequence_info.get('tile_workers', 1)
    use_batched = sequence_info.get('use_batched', True)  # Use batched computePixels by default
    
    samples = []
    
    print(f"\n📦 Processing sequence {seq_idx} with {len(valid_tiles)} valid tiles...")
    print(f"   Using {tile_workers} parallel worker(s) for tile processing")
    
    # Prepare tile info for parallel processing
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
            'date_band_config': date_band_config,  # Dynamic band config per date
            'zenith_lookup': zenith_lookup,  # Pre-extracted zenith angles (8x faster)
            'cache_dir': cache_dir,
            'region_name': region_name,
            'seq_idx': seq_idx,
            'use_batched': use_batched  # Use batched computePixels (~5x faster)
        }
        tile_infos.append(tile_info)
    
    # Process tiles in parallel
    if tile_workers > 1:
        # Parallel processing with ThreadPoolExecutor (not ProcessPoolExecutor)
        # ThreadPoolExecutor avoids pickling issues with ee.Image objects
        with ThreadPoolExecutor(max_workers=tile_workers) as executor:
            # Submit all tile processing tasks
            future_to_tile = {
                executor.submit(process_tile, tile_info): (tile_info['tile_idx'], tile_info['tile_bounds'])
                for tile_info in tile_infos
            }
            
            # Collect results as they complete
            processed = 0
            for future in as_completed(future_to_tile):
                tile_idx, tile_bounds = future_to_tile[future]
                processed += 1
                
                try:
                    sample = future.result()
                    if sample is not None:
                        samples.append(sample)
                        print(f"   Tile {tile_idx+1}/{len(tiles_to_process)}: ✓ ({processed}/{len(tiles_to_process)} done)")
                    else:
                        print(f"   Tile {tile_idx+1}/{len(tiles_to_process)}: ✗ ({processed}/{len(tiles_to_process)} done)")
                except Exception as e:
                    print(f"   Tile {tile_idx+1}/{len(tiles_to_process)}: ✗ Error: {e} ({processed}/{len(tiles_to_process)} done)")
    else:
        # Sequential processing (for debugging or when tile_workers=1)
        for tile_idx, tile_info in enumerate(tile_infos):
            print(f"   Tile {tile_idx+1}/{len(tiles_to_process)}...", end=" ", flush=True)
            try:
                sample = process_tile(tile_info)
                if sample is not None:
                    samples.append(sample)
                    print("✓")
                else:
                    print("✗")
            except Exception as e:
                print(f"✗ Error: {e}")
    
    print(f"   Sequence {seq_idx}: Generated {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description='Grid-based parallel dataset generation')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to generate (default: no limit, use all available)')
    parser.add_argument('--output-dir', type=str, 
                       default='./data/datasets/sta2_grid_parallel',
                       help='Output directory for dataset')
    parser.add_argument('--cache-dir', type=str,
                       default='./cache/sta2_grid_parallel',
                       help='Cache directory')
    parser.add_argument('--sequence-length', type=int, default=3,
                       help='Temporal sequence length')
    parser.add_argument('--coverage-threshold', type=float, default=90.0,
                       help='Minimum coverage percentage for valid tiles')
    parser.add_argument('--tile-workers', type=int, default=20,
                       help='Number of parallel workers for tile processing within each sequence')
    parser.add_argument('--coverage-workers', type=int, default=16,
                       help='Number of parallel workers for coverage checking')
    parser.add_argument('--region', type=str, default='all',
                       help='Region to process: "all" for all regions, or specific name (STA-2, STA-1E, STA-1W, STA-3/4, STA-5/6)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for image search (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for image search (YYYY-MM-DD)')
    parser.add_argument('--use-batched', action='store_true', default=True,
                       help='Use batched computePixels for ~5x faster downloads (default: True)')
    parser.add_argument('--no-batched', dest='use_batched', action='store_false',
                       help='Disable batched downloads (use legacy sampleRectangle)')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize data to [0, 1] range for training (default: True)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                       help='Keep raw reflectance values (0-10000 scale)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"🌐 GRID-BASED PARALLEL DATASET GENERATION")
    print("=" * 80)
    print(f"\n⚙️  CONFIGURATION:")
    print("─" * 80)
    print(f"  Strategy:               Grid-based systematic tile sampling")
    print(f"  Coverage pre-check:     {args.coverage_threshold}% minimum (across all frames)")
    print(f"  Tile workers:           {args.tile_workers} parallel worker(s) per sequence")
    print(f"  Coverage workers:       {args.coverage_workers} parallel workers")
    print(f"  Direct tile download:   YES (no full image waste)")
    print(f"  Download method:        {'computePixels BATCHED (~5x faster)' if args.use_batched else 'sampleRectangle (legacy)'}")
    print(f"  Coregistration:         AROSICS + Phase correlation fallback")
    print(f"  Harmonization:          Histogram matching (grid-based)")
    print(f"  Sentinel-2 bands:       B2,B3,B4 (10m) + B8A (20m→10m resampled)")
    print(f"  Planet bands:           B1,B2,B3,B4 (3m, 4-band harmonized)")
    print(f"  Sequence length:        {args.sequence_length} temporal frames")
    print(f"\n🎯 TARGET:")
    print("─" * 80)
    print(f"  Max samples:            {args.max_samples}")
    print(f"  Output directory:       {args.output_dir}")
    print(f"  Cache directory:        {args.cache_dir}")
    print("=" * 80)
    
    start_time = time.time()
    
    # =========================================================================
    # PROFILING: Track time spent on each major task
    # =========================================================================
    profile_times = {
        'find_pairs': 0,
        'zenith_extraction': 0,
        'band_detection': 0,
        'tile_grid': 0,
        'coverage_check': 0,
        'tile_downloads': 0,
        'local_quality_check': 0,
        'total': 0
    }
    
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
    
    # Select region to process (single region at a time for speed)
    region_config = None
    for r in regions:
        if r['name'] == args.region:
            region_config = r
            break
    
    if region_config is None:
        print(f"❌ Region '{args.region}' not found. Available regions:")
        for r in regions:
            print(f"   - {r['name']} ({r['start']} to {r['end']})")
        return
    
    # Use command-line dates if provided, otherwise use region defaults
    start_date = args.start_date if args.start_date else region_config["start"]
    end_date = args.end_date if args.end_date else region_config["end"]
    
    print(f"\n📍 Region: {region_config['name']}")
    print(f"   Date range: {start_date} to {end_date}")
    
    # Find image pairs (with caching for matched_pairs metadata)
    print(f"\n🔍 Finding image pairs...")
    print(f"   Checking BOTH Planet repositories for maximum coverage...")
    
    t_find_pairs = time.time()  # PROFILING
    
    # Check for cached pairs metadata
    import hashlib
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
    
    # Always call find_pairs to get collections (needed for image access)
    # but use cached matched_pairs if available
    pair_results = find_pairs(
        region_config["aoi"],
        start_date,
        end_date,
        PLANET_COLLECTION_ID,
        planet_coll_id_fallback=PLANET_COLLECTION_ID_FALLBACK  # Use BOTH repositories
    )
    
    if cached_matched_pairs is not None:
        matched_pairs = cached_matched_pairs
    else:
        matched_pairs = pair_results.get('matched_pairs', [])
        # Cache the matched pairs for future runs
        try:
            with open(pairs_cache_file, 'wb') as f:
                pickle.dump(matched_pairs, f)
            print(f"   ✅ Cached {len(matched_pairs)} matched pairs for future runs")
        except:
            pass
    
    s2_collection = pair_results.get('s2_collection')
    planet_collection = pair_results.get('planet_collection')
    
    profile_times['find_pairs'] = time.time() - t_find_pairs  # PROFILING
    print(f"   Found {len(matched_pairs)} matched pairs")
    print(f"   ⏱️  find_pairs took {profile_times['find_pairs']:.1f}s")
    
    if len(matched_pairs) < args.sequence_length:
        print(f"   ❌ Insufficient pairs for sequence length {args.sequence_length}")
        return
    
    # =========================================================================
    # BATCHED ZENITH EXTRACTION (8x faster than per-frame extraction)
    # =========================================================================
    # Extract ALL solar zenith angles upfront using aggregate_array()
    # This is a FIXED COST (~99s) regardless of number of samples
    # Previously: ~47s per frame × N frames = O(N) time
    # Now: ~99s total for ALL frames = O(1) time
    print(f"\n☀️  BATCHED SOLAR ZENITH EXTRACTION")
    print("─" * 80)
    
    # Check for cached zenith data
    zenith_cache_file = cache_dir / f"zenith_data_{pairs_cache_hash}.pkl"
    zenith_lookup = None
    
    if zenith_cache_file.exists():
        try:
            with open(zenith_cache_file, 'rb') as f:
                zenith_lookup = pickle.load(f)
            print(f"   ✅ Loaded cached zenith data for {len(zenith_lookup.get('s2', {}))} S2 + {len(zenith_lookup.get('planet', {}))} Planet images")
        except:
            zenith_lookup = None
    
    if zenith_lookup is None:
        print(f"   Extracting zenith angles for {len(matched_pairs)} matched pairs...")
        t_zenith = time.time()
        
        try:
            # Get all S2 zenith angles in ONE call
            s2_ids = [p['s2_id'] for p in matched_pairs]
            s2_filtered = s2_collection.filter(ee.Filter.inList('id_pref', s2_ids))
            s2_zenith_list = s2_filtered.aggregate_array('MEAN_SOLAR_ZENITH_ANGLE').getInfo()
            s2_id_list = s2_filtered.aggregate_array('id_pref').getInfo()
            print(f"   ✓ S2 zenith: {len(s2_id_list)} images in {time.time()-t_zenith:.1f}s")
            
            # Get all Planet zenith angles in ONE call
            t_planet = time.time()
            planet_ids = [p['planet_id'] for p in matched_pairs]
            planet_filtered = planet_collection.filter(ee.Filter.inList('id_pref', planet_ids))
            planet_elevation_list = planet_filtered.aggregate_array('sun_elevation').getInfo()
            planet_id_list = planet_filtered.aggregate_array('id_pref').getInfo()
            print(f"   ✓ Planet zenith: {len(planet_id_list)} images in {time.time()-t_planet:.1f}s")
            
            # Build lookup dicts
            s2_zenith_map = dict(zip(s2_id_list, s2_zenith_list))
            planet_zenith_map = {}
            for pid, elev in zip(planet_id_list, planet_elevation_list):
                if elev is not None:
                    planet_zenith_map[pid] = 90.0 - float(elev)  # Convert elevation to zenith
                else:
                    planet_zenith_map[pid] = None
            
            zenith_lookup = {
                's2': s2_zenith_map,
                'planet': planet_zenith_map
            }
            
            # Cache for future runs
            with open(zenith_cache_file, 'wb') as f:
                pickle.dump(zenith_lookup, f)
            print(f"   ✅ Cached zenith data: {zenith_cache_file.name}")
            
            profile_times['zenith_extraction'] = time.time() - t_zenith
            print(f"   ⏱️  Zenith extraction took {profile_times['zenith_extraction']:.1f}s (FIXED COST)")
            
        except Exception as e:
            print(f"   ⚠️ Batched zenith extraction failed: {e}")
            print(f"   Will fall back to per-frame extraction (slower)")
            zenith_lookup = {'s2': {}, 'planet': {}}
    else:
        profile_times['zenith_extraction'] = 0  # Cached
    
    print("─" * 80)
    
    # Calculate how many sequences we need - use ALL available sequences up to max
    max_possible_sequences = len(matched_pairs) - args.sequence_length + 1
    num_sequences = min(args.max_samples, max_possible_sequences) if args.max_samples else max_possible_sequences
    
    print(f"\n📅 Creating {num_sequences} sequences of {args.sequence_length} frames each")
    
    # =========================================================================
    # DYNAMIC BAND DETECTION (with caching and parallelization)
    # =========================================================================
    # Detect 4-band vs 8-band Planet images for each unique date
    # Now with disk caching and parallel API calls for ~6x speedup
    print(f"\n🔬 DYNAMIC BAND DETECTION")
    print("─" * 80)
    
    # Check for cached band detection results
    import hashlib
    dates_hash = hashlib.md5("_".join(sorted([p['planet_date'] for p in matched_pairs])).encode()).hexdigest()[:16]
    band_cache_file = cache_dir / f"band_detection_{dates_hash}.pkl"
    
    if band_cache_file.exists():
        print(f"   ✅ Found cached band detection results!")
        with open(band_cache_file, 'rb') as f:
            date_band_config = pickle.load(f)
        modes_8band = sum(1 for cfg in date_band_config.values() if cfg['mode'] == '8-band')
        modes_4band = len(date_band_config) - modes_8band
        print(f"   ⚡ Loaded {len(date_band_config)} dates from cache")
        print(f"   📊 {modes_8band} dates with 8-band, {modes_4band} with 4-band")
        t_detect = 0
    else:
        print(f"   Detecting band availability for all unique Planet dates...")
        t_detect = time.time()
        
        # Get all unique Planet dates from matched pairs
        all_planet_dates = list(set([p['planet_date'] for p in matched_pairs]))
        print(f"   📊 Checking {len(all_planet_dates)} unique Planet dates for 8-band capability...")
        
        # Dictionary to store band configuration per date
        date_band_config = {}
        
        # Parallel band detection function
        def detect_bands_for_date(pdate):
            try:
                planet_img = planet_collection.filter(ee.Filter.eq('date_key', pdate)).first()
                available_bands = planet_img.bandNames().getInfo()
                
                # Check for extended bands (8-band SuperDove)
                has_extended_bands = any(band in available_bands for band in ['B5', 'B6', 'B7', 'B8'])
                
                if has_extended_bands and all(band in available_bands for band in ['B5', 'B6', 'B7', 'B8']):
                    # 8-band SuperDove - correct spectral matching:
                    # S2 B2↔Planet B2, S2 B3↔Planet B4, S2 B4↔Planet B6
                    # S2 B5↔Planet B7, S2 B8A↔Planet B8
                    return pdate, {
                        'mode': '8-band',
                        'planet_bands': ['B2', 'B4', 'B6', 'B7', 'B8'],  # 5 bands
                        'sentinel_bands': ['B2', 'B3', 'B4', 'B5', 'B8A']  # 3×10m + 2×20m
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
        
        # Parallel detection with ThreadPoolExecutor (20 workers optimal for GEE)
        with ThreadPoolExecutor(max_workers=min(20, len(all_planet_dates))) as executor:
            futures = {executor.submit(detect_bands_for_date, pdate): pdate for pdate in all_planet_dates}
            completed = 0
            for future in as_completed(futures):
                pdate, config = future.result()
                date_band_config[pdate] = config
                completed += 1
                if completed % 10 == 0 or completed == len(all_planet_dates):
                    print(f"   Progress: {completed}/{len(all_planet_dates)} dates checked")
        
        # Save to cache
        with open(band_cache_file, 'wb') as f:
            pickle.dump(date_band_config, f)
        print(f"   ✅ Cache saved: {band_cache_file.name}")
        
        # Report detection results
        modes_8band = sum(1 for cfg in date_band_config.values() if cfg['mode'] == '8-band')
        modes_4band = len(all_planet_dates) - modes_8band
        profile_times['band_detection'] = time.time() - t_detect  # PROFILING
        print(f"   ✅ Band detection complete in {profile_times['band_detection']:.1f}s")
        print(f"   📊 Found {modes_8band} dates with 8-band, {modes_4band} with 4-band")
    
    print("─" * 80)
    
    # Default bands (used for coverage checking - use minimal common set)
    planet_bands = ['B1', 'B2', 'B3', 'B4']
    sentinel_bands = ['B2', 'B3', 'B4', 'B8A']
    
    all_samples = []
    
    # Process sequences
    for seq_idx in range(num_sequences):
        t_seq_start = time.time()  # PROFILING - per sequence timing
        
        sequence = matched_pairs[seq_idx:seq_idx + args.sequence_length]
        
        if len(sequence) < args.sequence_length:
            break
        
        print(f"\n{'='*80}")
        print(f"SEQUENCE {seq_idx + 1}/{num_sequences}")
        print(f"Dates: {[p['s2_date'] for p in sequence]}")
        print(f"{'='*80}")
        
        # Check temporal distances between consecutive frames
        from datetime import datetime
        print(f"\n⏱️  Checking temporal distances...")
        has_large_gaps = False
        
        for i in range(len(sequence) - 1):
            date1 = datetime.strptime(sequence[i]['s2_date'], '%Y-%m-%d')
            date2 = datetime.strptime(sequence[i+1]['s2_date'], '%Y-%m-%d')
            days_diff = (date2 - date1).days
            
            if days_diff > 30:
                print(f"   ⚠️  WARNING: {sequence[i]['s2_date']} → {sequence[i+1]['s2_date']}: {days_diff} days (>{30} days)")
                has_large_gaps = True
            else:
                print(f"   ✓ {sequence[i]['s2_date']} → {sequence[i+1]['s2_date']}: {days_diff} days")
        
        if has_large_gaps:
            print(f"\n   🚨 SEQUENCE HAS LARGE TEMPORAL GAPS (>1 month)")
            print(f"   This may affect temporal attention learning!")
        else:
            print(f"\n   ✅ All temporal gaps ≤30 days (good for temporal learning)")
        
        # Get images for coverage checking
        # IMPORTANT: Check coverage across ALL frames in sequence to find common valid areas
        print(f"\n📅 Getting images from all {len(sequence)} frames for coverage checking...")
        planet_images = []
        s2_images = []
        
        for frame_idx, pair in enumerate(sequence):
            planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
            s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
            planet_images.append(planet_img)
            s2_images.append(s2_img)
            print(f"   Frame {frame_idx + 1}: Planet {pair['planet_id']}, S2 {pair['s2_id']}")
        
        # Check if we have cached coverage results for this sequence
        # This saves ~600 GEE API calls per sequence (200 tiles × 3 frames × 2 images/frame)
        sequence_dates = tuple(sorted([p['s2_date'] for p in sequence]))
        # Use stable hash based on concatenated dates
        import hashlib
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
            print(f"   ⚡ Saved ~{total_checked * len(sequence) * 2} GEE API calls!")
        else:
            # Check coverage for ALL frames to find intersection
            # IMPORTANT: A tile is only valid if ALL frames have sufficient coverage
            print(f"\n🔍 Checking coverage across ALL {len(sequence)} frames...")
            
            # First, check overall coverage for each frame
            for frame_idx, (planet_img, pair) in enumerate(zip(planet_images, sequence)):
                test_geom = region_config["aoi"]
                try:
                    valid_mask = planet_img.select('B1').mask()
                    stats = valid_mask.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=test_geom,
                        scale=100,  # Coarse for speed
                        maxPixels=1e6,
                        bestEffort=True
                    )
                    coverage = stats.get('B1').getInfo() or 0
                    print(f"   Frame {frame_idx + 1} ({pair['s2_date']}): {coverage*100:.1f}% overall coverage")
                except:
                    print(f"   Frame {frame_idx + 1} ({pair['s2_date']}): Coverage check failed")
            
            # Generate grid tiles - no limit, use all tiles
            print(f"\n🗺️  Generating tile grid...")
            t_tile_grid = time.time()  # PROFILING
            all_tiles = generate_tile_grid(region_config["aoi"], tile_size_deg=0.007, max_tiles=None)
            profile_times['tile_grid'] += time.time() - t_tile_grid  # PROFILING
            print(f"   ⏱️  Tile grid generation: {time.time() - t_tile_grid:.2f}s")
            
            # =========================================================================
            # BATCHED COVERAGE CHECK (OPTIMIZED - 6 API calls instead of 1200+)
            # =========================================================================
            # Instead of checking each tile individually (1200+ getInfo calls),
            # we use reduceRegions() to check ALL tiles in ONE GEE call per image.
            # This reduces API calls from O(tiles * frames * 2) to O(frames * 2)
            # =========================================================================
            print(f"\n🔍 Filtering tiles with coverage in ALL {len(sequence)} frames (intersection)...")
            print(f"   This ensures temporal consistency - all frames have data at each tile")
            print(f"   ⚡ OPTIMIZED: Using batched reduceRegions() - ~{len(sequence)*2} API calls instead of ~{len(all_tiles)*len(sequence)*2}")
            
            t_coverage = time.time()  # PROFILING
            valid_tiles = []
            
            try:
                # 1. Create a FeatureCollection of tile geometries with their bounds as properties
                tile_features = []
                for idx, tile_bounds in enumerate(all_tiles):
                    tile_geom = ee.Geometry.Rectangle(tile_bounds)
                    tile_features.append(ee.Feature(tile_geom, {
                        'tile_idx': idx,
                        'bounds': tile_bounds
                    }))
                tile_fc = ee.FeatureCollection(tile_features)
                
                # 2. For each image, compute coverage per tile using reduceRegions IN PARALLEL
                # This replaces 200+ individual getInfo() calls with ONE reduceRegions() call per image
                # AND runs all images in parallel for maximum speed
                all_images = planet_images + s2_images
                image_names = [f"planet_{i}" for i in range(len(planet_images))] + \
                             [f"s2_{i}" for i in range(len(s2_images))]
                
                # Dictionary to store coverage results: tile_idx -> {image_name: coverage}
                tile_coverage = {idx: {} for idx in range(len(all_tiles))}
                
                print(f"   Checking coverage for {len(all_images)} images across {len(all_tiles)} tiles...")
                print(f"   ⚡ Running {len(all_images)} reduceRegions() calls IN PARALLEL...")
                
                def check_image_coverage(img_data):
                    """Check coverage for one image across all tiles (one reduceRegions call)"""
                    img, img_name = img_data
                    band_name = 'B1' if 'planet' in img_name else 'B2'
                    valid_mask = img.select(band_name).mask()
                    
                    # Use reduceRegions to get mean coverage for ALL tiles at once
                    coverage_results = valid_mask.reduceRegions(
                        collection=tile_fc,
                        reducer=ee.Reducer.mean(),
                        scale=30
                    ).getInfo()
                    
                    return img_name, coverage_results
                
                # Run all coverage checks in parallel (6 parallel GEE calls)
                completed_count = 0
                with ThreadPoolExecutor(max_workers=len(all_images)) as executor:
                    futures = [executor.submit(check_image_coverage, (img, name)) 
                              for img, name in zip(all_images, image_names)]
                    
                    for future in as_completed(futures):
                        img_name, coverage_results = future.result()
                        
                        # Store results
                        for feature in coverage_results['features']:
                            tile_idx = feature['properties']['tile_idx']
                            coverage = feature['properties'].get('mean', 0) or 0
                            tile_coverage[tile_idx][img_name] = coverage * 100  # Convert to percentage
                        
                        completed_count += 1
                        print(f"   ✓ {img_name}: coverage check complete ({completed_count}/{len(all_images)})")
                
                # 3. Filter tiles where ALL images have sufficient coverage
                coverage_threshold = args.coverage_threshold
                for idx, tile_bounds in enumerate(all_tiles):
                    coverages = tile_coverage[idx]
                    
                    # Check if all images meet the threshold
                    all_valid = all(cov >= coverage_threshold for cov in coverages.values())
                    
                    if all_valid:
                        valid_tiles.append(tile_bounds)
                
                print(f"   ✓ Valid tiles (ALL frames): {len(valid_tiles)}/{len(all_tiles)} ({100*len(valid_tiles)/len(all_tiles):.1f}%)")
                
            except Exception as e:
                print(f"   ❌ Batched coverage check failed: {e}")
                print(f"   ⚠️ Falling back to slower per-tile checking...")
                
                # Fallback to original slow method
                valid_tiles = []
                checked = 0
                
                with ThreadPoolExecutor(max_workers=args.coverage_workers) as executor:
                    def check_tile_all_frames(tile_bounds):
                        tile_geom = ee.Geometry.Rectangle(tile_bounds)
                        for planet_img, s2_img in zip(planet_images, s2_images):
                            planet_valid = has_sufficient_coverage(
                                planet_img, tile_geom, 'B1',
                                min_percent=args.coverage_threshold, scale=30
                            )
                            if not planet_valid:
                                return False
                            s2_valid = has_sufficient_coverage(
                                s2_img, tile_geom, 'B2',
                                min_percent=args.coverage_threshold, scale=30
                            )
                            if not s2_valid:
                                return False
                        return True
                    
                    future_to_tile = {
                        executor.submit(check_tile_all_frames, tile_bounds): tile_bounds
                        for tile_bounds in all_tiles
                    }
                    
                    for future in as_completed(future_to_tile):
                        tile_bounds = future_to_tile[future]
                        checked += 1
                        try:
                            if future.result():
                                valid_tiles.append(tile_bounds)
                            if checked % 20 == 0 or checked == len(all_tiles):
                                print(f"   Progress: {checked}/{len(all_tiles)} checked, {len(valid_tiles)} valid")
                        except:
                            pass
                
                print(f"   ✓ Valid tiles (ALL frames): {len(valid_tiles)}/{len(all_tiles)} ({100*len(valid_tiles)/len(all_tiles):.1f}%)")
            
            # PROFILING - end coverage check timing
            profile_times['coverage_check'] += time.time() - t_coverage
            print(f"   ⏱️  Coverage check took {time.time() - t_coverage:.1f}s")
            
            # Save coverage results to cache for future runs
            cache_data = {
                'valid_tiles': valid_tiles,
                'total_checked': len(all_tiles),
                'sequence_dates': sequence_dates,
                'coverage_threshold': args.coverage_threshold
            }
            with open(coverage_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"   ✅ Cache saved: {coverage_cache_file.name}")
        
        if len(valid_tiles) == 0:
            print(f"   ⚠️ No valid tiles found for this sequence, skipping")
            continue
        
        # Calculate samples for this sequence
        remaining_samples = args.max_samples - len(all_samples)
        max_samples_this_seq = min(len(valid_tiles), remaining_samples)
        
        print(f"\n⚡ Processing {max_samples_this_seq} tiles from this sequence...")
        
        t_downloads = time.time()  # PROFILING - downloads start
        
        # Prepare sequence info
        sequence_info = {
            'sequence': sequence,
            'valid_tiles': valid_tiles,
            'planet_collection': planet_collection,
            's2_collection': s2_collection,
            'planet_bands': planet_bands,
            'sentinel_bands': sentinel_bands,
            'date_band_config': date_band_config,  # Dynamic band config per date
            'zenith_lookup': zenith_lookup,  # Pre-extracted zenith angles (8x faster)
            'cache_dir': cache_dir,
            'region_name': region_config['name'],
            'seq_idx': seq_idx,
            'max_samples_per_seq': max_samples_this_seq,
            'tile_workers': args.tile_workers,
            'use_batched': args.use_batched  # Use batched computePixels (~5x faster)
        }
        
        # Process this sequence
        samples = process_single_sequence(sequence_info)
        
        # PROFILING - downloads end
        profile_times['tile_downloads'] += time.time() - t_downloads
        
        all_samples.extend(samples)
        
        # PROFILING - per sequence summary
        seq_time = time.time() - t_seq_start
        print(f"\n   ✓ Sequence {seq_idx + 1} complete: {len(samples)} samples generated")
        print(f"   ⏱️  Sequence {seq_idx + 1} took {seq_time:.1f}s total")
        print(f"   Total samples so far: {len(all_samples)}/{args.max_samples}")
        
        if len(all_samples) >= args.max_samples:
            break
    
    # Save dataset
    if len(all_samples) == 0:
        print("\n❌ No samples generated!")
        return
    
    # Normalize samples to [0, 1] range for neural network training
    if args.normalize:
        print(f"\n🔄 Normalizing {len(all_samples)} samples to [0, 1] range...")
        for i, sample in enumerate(all_samples):
            normalize_sample(sample)  # In-place normalization
        print("   ✓ Normalization complete")
    else:
        print(f"\n⚠️  Skipping normalization (raw reflectance values preserved)")
    
    dataset_path = output_dir / 'dataset.pkl'
    print(f"\n💾 Saving {len(all_samples)} samples...")
    
    dataset = {
        'samples': all_samples,
        'config': {
            'region': region_config['name'],
            'sequence_length': args.sequence_length,
            'num_samples': len(all_samples),
            'coverage_threshold': args.coverage_threshold,
            'grid_based': True,
            'normalized': args.normalize,  # True if data is in [0, 1] range
            'scale_factor': 10000.0 if args.normalize else None  # For denormalization if needed
        }
    }
    
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=4)
    
    profile_times['total'] = time.time() - start_time
    elapsed = profile_times['total']
    
    print("\n" + "=" * 80)
    print("✅ GRID-BASED PARALLEL GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Samples generated: {len(all_samples)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes ({elapsed:.1f}s)")
    print(f"Avg time per sample: {elapsed/len(all_samples):.1f} seconds")
    
    # =========================================================================
    # PROFILING SUMMARY
    # =========================================================================
    print("\n" + "─" * 80)
    print("⏱️  PROFILING BREAKDOWN:")
    print("─" * 80)
    print(f"  find_pairs():      {profile_times['find_pairs']:6.1f}s ({100*profile_times['find_pairs']/elapsed:5.1f}%)")
    print(f"  zenith_extract:    {profile_times['zenith_extraction']:6.1f}s ({100*profile_times['zenith_extraction']/elapsed:5.1f}%)")
    print(f"  band_detection:    {profile_times['band_detection']:6.1f}s ({100*profile_times['band_detection']/elapsed:5.1f}%)")
    print(f"  tile_grid:         {profile_times['tile_grid']:6.1f}s ({100*profile_times['tile_grid']/elapsed:5.1f}%)")
    print(f"  coverage_check:    {profile_times['coverage_check']:6.1f}s ({100*profile_times['coverage_check']/elapsed:5.1f}%)")
    print(f"  tile_downloads:    {profile_times['tile_downloads']:6.1f}s ({100*profile_times['tile_downloads']/elapsed:5.1f}%)")
    other_time = elapsed - sum([profile_times['find_pairs'], profile_times['zenith_extraction'],
                                profile_times['band_detection'], profile_times['tile_grid'], 
                                profile_times['coverage_check'], profile_times['tile_downloads']])
    print(f"  other:             {other_time:6.1f}s ({100*other_time/elapsed:5.1f}%)")
    print("─" * 80)
    print(f"  TOTAL:             {elapsed:6.1f}s (100.0%)")
    print("─" * 80)
    
    # THEORY CHECK: Fixed costs vs variable costs
    fixed_costs = profile_times['find_pairs'] + profile_times['zenith_extraction'] + profile_times['band_detection'] + profile_times['coverage_check']
    variable_costs = profile_times['tile_downloads']
    print(f"\n📊 FIXED vs VARIABLE COST ANALYSIS:")
    print(f"   Fixed costs (same for 3 or 100 samples):   {fixed_costs:.1f}s ({100*fixed_costs/elapsed:.1f}%)")
    print(f"   Variable costs (scales with sample count): {variable_costs:.1f}s ({100*variable_costs/elapsed:.1f}%)")
    if fixed_costs > variable_costs:
        print(f"   ⚠️  FIXED COSTS DOMINATE - generating more samples is almost free!")
    
    print(f"\nDataset saved: {dataset_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
