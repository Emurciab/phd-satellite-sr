#!/usr/bin/env python3
"""
STA-1E GRID-BASED PARALLEL dataset generation with INTEGRATED 5-BAND OUTPUT
=============================================================================

This script generates training data for STA-1E directly in 5-band format,
eliminating the need for a separate conversion step.

Key Features:
- Grid-based tile sampling (systematic coverage)
- Coverage pre-checking (avoid Planet mosaic gaps)
- Parallel sequence processing
- Direct 5-band output (SuperDove: B2,B4,B6,B7,B8 → S2 B2,B3,B4,B5,B8A)
- Multi-resolution output (10m + 20m bands at native resolution)
- Frame quality tracking

Output Format (TrainingSample5Band):
- lr_sequence: (T, 77, 77, 3) - S2 10m bands at native resolution
- hr_sequence: (T, 256, 256, 5) - Planet 5-band at 3m
- lr_20m_sequence: (T, 38, 38, 2) - S2 20m bands at native
- hr_10m_sequence: (T, 256, 256, 3) - Planet RGB only
- hr_20m_sequence: (T, 128, 128, 2) - Planet RedEdge/NIR at 6m
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
import io
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from Auth.AuthGEE import (
    find_pairs, 
    PLANET_COLLECTION_ID, 
    PLANET_COLLECTION_ID_FALLBACK,
    PLANET_COLLECTION_ID_ADDITIONAL,
    regions
)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TrainingSample5Band:
    """Training sample with full 5-band HR data for SuperDove images."""
    # Core sequences
    lr_sequence: np.ndarray  # (T, 77, 77, 3) - 10m bands at native resolution
    hr_sequence: np.ndarray  # (T, 256, 256, 5) - Full HR (5 for SuperDove)
    lr_20m_sequence: Optional[np.ndarray] = None  # (T, 38, 38, 2) - 20m bands at native
    
    # Split HR for multi-resolution training
    hr_10m_sequence: Optional[np.ndarray] = None  # (T, 256, 256, 3) - RGB bands only
    hr_20m_sequence: Optional[np.ndarray] = None  # (T, 128, 128, 2) - RedEdge/NIR at 6m
    
    # Metadata
    temporal_dates: List[str] = field(default_factory=list)
    planet_dates: List[str] = field(default_factory=list)
    region_name: str = ''
    sequence_id: int = 0
    tile_id: int = 0
    tile_bounds: Dict = field(default_factory=dict)
    
    # Band configuration
    sensor_type: str = 'superdove'
    planet_bands: List[str] = field(default_factory=lambda: ['B2', 'B4', 'B6', 'B7', 'B8'])
    sentinel_bands: List[str] = field(default_factory=lambda: ['B2', 'B3', 'B4', 'B5', 'B8A'])
    
    # Solar zenith angles
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None
    
    # Quality metrics
    frame_quality: Optional[List[Dict]] = None


@dataclass
class FrameQualityStats:
    """Quality statistics for a single frame"""
    zero_fraction: float
    nan_fraction: float
    outlier_fraction: float
    valid_fraction: float
    mean_reflectance: float
    std_reflectance: float
    is_good: bool


# ============================================================================
# QUALITY FUNCTIONS
# ============================================================================

def compute_frame_quality_stats(array: np.ndarray, 
                                 zero_threshold: float = 0.05,
                                 outlier_percentile: float = 99.5,
                                 max_valid_reflectance: float = 10000.0) -> FrameQualityStats:
    """
    Compute quality statistics for a single frame.
    
    This catches issues that server-side GEE coverage checking misses:
    - Masked/nodata pixels that come through as zeros
    - Extreme outliers from atmospheric contamination
    - NaN values from processing errors
    """
    total_pixels = array.size
    
    # Count zeros (masked pixels often come as 0)
    zero_mask = np.abs(array) < zero_threshold
    zero_count = np.sum(zero_mask)
    zero_fraction = zero_count / total_pixels
    
    # Count NaNs
    nan_mask = np.isnan(array)
    nan_count = np.sum(nan_mask)
    nan_fraction = nan_count / total_pixels
    
    # Valid pixels for statistics
    valid_mask = ~zero_mask & ~nan_mask
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
    
    # Compute outlier threshold
    p99 = np.percentile(valid_pixels, outlier_percentile)
    outlier_threshold = min(p99 * 1.5, max_valid_reflectance)
    
    # Count outliers (extreme values from atmospheric issues)
    outlier_mask = valid_pixels > outlier_threshold
    outlier_count = np.sum(outlier_mask)
    outlier_fraction = outlier_count / total_pixels
    
    # Final valid fraction
    valid_fraction = 1.0 - zero_fraction - nan_fraction - outlier_fraction
    
    # Statistics on valid pixels
    mean_refl = np.mean(valid_pixels[~outlier_mask]) if np.sum(~outlier_mask) > 0 else 0.0
    std_refl = np.std(valid_pixels[~outlier_mask]) if np.sum(~outlier_mask) > 0 else 0.0
    
    # Quality thresholds
    is_good = (
        zero_fraction < 0.10 and     # <10% zeros
        nan_fraction < 0.01 and      # <1% NaN
        outlier_fraction < 0.05 and  # <5% outliers
        valid_fraction > 0.85        # >85% valid
    )
    
    return FrameQualityStats(
        zero_fraction=zero_fraction,
        nan_fraction=nan_fraction,
        outlier_fraction=outlier_fraction,
        valid_fraction=valid_fraction,
        mean_reflectance=mean_refl,
        std_reflectance=std_refl,
        is_good=is_good
    )


def compute_combined_frame_quality(lr_frame: np.ndarray, hr_frame: np.ndarray, 
                                   frame_idx: int = 0) -> Dict:
    """Compute combined quality stats for both LR and HR frames."""
    lr_stats = compute_frame_quality_stats(lr_frame)
    hr_stats = compute_frame_quality_stats(hr_frame)
    
    return {
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


# ============================================================================
# GEE DOWNLOAD FUNCTIONS
# ============================================================================

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


def download_frame_5band(planet_img: ee.Image, s2_img: ee.Image,
                         tile_bounds: List[float], tile_size_px: int = 256,
                         cache_path: Optional[Path] = None) -> Optional[Dict]:
    """
    Download Planet and Sentinel-2 data for one frame in 5-band format.
    
    Downloads:
    - Planet: B2, B4, B6, B7, B8 (5 bands at 3m)
    - S2 10m: B2, B3, B4 (3 bands at native 10m)
    - S2 20m: B5, B8A (2 bands at native 20m)
    """
    # Check cache first
    if cache_path and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=True)
            return {
                'hr': cached['hr_array'],
                'lr_10m': cached['lr_10m'],
                'lr_20m': cached['lr_20m']
            }
        except:
            pass
    
    # Resolution calculations
    scale_factor_10m = 10.0 / 3.0  # 3.33
    scale_factor_20m = 20.0 / 3.0  # 6.67
    lr_10m_size = int(round(tile_size_px / scale_factor_10m))  # ~77
    lr_20m_size = int(round(tile_size_px / scale_factor_20m))  # ~38
    
    # Planet 5-band configuration
    planet_bands = ['B2', 'B4', 'B6', 'B7', 'B8']
    s2_bands_10m = ['B2', 'B3', 'B4']
    s2_bands_20m = ['B5', 'B8A']
    
    try:
        # Download Planet HR (256x256 at 3m)
        hr_array = _compute_pixels_download(
            planet_img, tile_bounds, planet_bands,
            width=tile_size_px, height=tile_size_px
        )
        
        # Download S2 10m bands (~77x77)
        lr_10m = _compute_pixels_download(
            s2_img, tile_bounds, s2_bands_10m,
            width=lr_10m_size, height=lr_10m_size
        )
        
        # Download S2 20m bands (~38x38)
        lr_20m = _compute_pixels_download(
            s2_img, tile_bounds, s2_bands_20m,
            width=lr_20m_size, height=lr_20m_size
        )
        
        if hr_array is None or lr_10m is None or lr_20m is None:
            return None
        
        # Cache the result
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path,
                hr_array=hr_array,
                lr_10m=lr_10m,
                lr_20m=lr_20m,
                planet_bands=planet_bands,
                s2_bands_10m=s2_bands_10m,
                s2_bands_20m=s2_bands_20m
            )
        
        return {
            'hr': hr_array,
            'lr_10m': lr_10m,
            'lr_20m': lr_20m
        }
        
    except Exception as e:
        return None


def download_all_frames_5band(sequence: List[Dict], tile_bounds: List[float],
                              planet_collection, s2_collection,
                              cache_dir: Path, tile_size_px: int = 256) -> Tuple[Dict, Dict]:
    """
    Download all frames for a tile in 5-band format with parallel workers.
    """
    results = {}
    cache_stats = {'hits': 0, 'misses': 0}
    
    planet_bands = ['B2', 'B4', 'B6', 'B7', 'B8']
    s2_bands = ['B2', 'B3', 'B4', 'B5', 'B8A']
    
    # Prepare tasks
    tasks = []
    for frame_idx, pair in enumerate(sequence):
        cache_key_str = f"{pair['planet_id']}|{pair['s2_id']}|{tile_bounds}|5band"
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:16]
        cache_path = cache_dir / "raw_downloads_5band" / f"tile_{cache_hash}.npz"
        
        tasks.append({
            'frame_idx': frame_idx,
            'pair': pair,
            'cache_path': cache_path
        })
    
    # Check cache first
    frames_to_download = []
    for task in tasks:
        if task['cache_path'].exists():
            try:
                cached = np.load(task['cache_path'], allow_pickle=True)
                results[task['frame_idx']] = {
                    'hr': cached['hr_array'],
                    'lr_10m': cached['lr_10m'],
                    'lr_20m': cached['lr_20m'],
                    'pair': task['pair']
                }
                cache_stats['hits'] += 1
                continue
            except:
                pass
        frames_to_download.append(task)
        cache_stats['misses'] += 1
    
    if not frames_to_download:
        return results, cache_stats
    
    # Download missing frames in parallel
    def download_frame(task):
        pair = task['pair']
        planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
        s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
        
        frame_data = download_frame_5band(
            planet_img, s2_img, tile_bounds,
            tile_size_px=256, cache_path=task['cache_path']
        )
        
        return task['frame_idx'], frame_data, pair
    
    max_workers = min(20, len(frames_to_download))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_frame, task): task for task in frames_to_download}
        
        for future in as_completed(futures):
            try:
                frame_idx, frame_data, pair = future.result()
                if frame_data is not None:
                    results[frame_idx] = {
                        'hr': frame_data['hr'],
                        'lr_10m': frame_data['lr_10m'],
                        'lr_20m': frame_data['lr_20m'],
                        'pair': pair
                    }
            except Exception as e:
                pass
    
    return results, cache_stats


# ============================================================================
# TILE PROCESSING
# ============================================================================

def downsample_hr_for_20m(hr_array: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample HR array by factor for 20m band targets."""
    h, w, c = hr_array.shape
    new_h = h // factor
    new_w = w // factor
    
    h_trim = new_h * factor
    w_trim = new_w * factor
    trimmed = hr_array[:h_trim, :w_trim, :]
    
    result = trimmed.reshape(new_h, factor, new_w, factor, c).mean(axis=(1, 3))
    return result.astype(hr_array.dtype)


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


def has_sufficient_coverage(img: ee.Image, geom: ee.Geometry, band_name: str, 
                           min_percent: float = 90.0, scale: int = 30) -> bool:
    """
    Check if image has sufficient valid pixel coverage over geometry.
    """
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


def filter_tiles_batched(planet_images: List[ee.Image], s2_images: List[ee.Image],
                        tile_bounds_list: List, min_percent: float = 90.0,
                        max_workers: int = 16) -> List:
    """
    Filter tiles using batched reduceRegions() for efficiency.
    
    OPTIMIZED: Uses ~(num_frames * 2) API calls instead of ~(num_tiles * num_frames * 2)
    For 90 tiles, 3 frames: 6 calls instead of 540 calls
    
    Args:
        planet_images: List of Planet ee.Image objects (one per frame)
        s2_images: List of Sentinel-2 ee.Image objects (one per frame)
        tile_bounds_list: List of tile bounds [[min_lon, min_lat, max_lon, max_lat], ...]
        min_percent: Minimum valid pixel percentage required
        max_workers: Number of parallel workers for coverage checking
    
    Returns:
        List of tile bounds that have sufficient coverage in ALL frames
    """
    num_frames = len(planet_images)
    num_tiles = len(tile_bounds_list)
    
    print(f"\n🔍 Filtering tiles with coverage in ALL {num_frames} frames (intersection)...")
    print(f"   This ensures temporal consistency - all frames have data at each tile")
    print(f"   ⚡ OPTIMIZED: Using batched reduceRegions() - ~{num_frames * 2} API calls instead of ~{num_tiles * num_frames * 2}")
    
    valid_tiles = []
    
    try:
        # 1. Create a FeatureCollection of tile geometries with their bounds as properties
        tile_features = []
        for idx, tile_bounds in enumerate(tile_bounds_list):
            tile_geom = ee.Geometry.Rectangle(tile_bounds)
            tile_features.append(ee.Feature(tile_geom, {
                'tile_idx': idx,
                'bounds': tile_bounds
            }))
        tile_fc = ee.FeatureCollection(tile_features)
        
        # 2. For each image, compute coverage per tile using reduceRegions IN PARALLEL
        # This replaces N individual getInfo() calls with ONE reduceRegions() call per image
        all_images = planet_images + s2_images
        image_names = [f"planet_{i}" for i in range(len(planet_images))] + \
                     [f"s2_{i}" for i in range(len(s2_images))]
        
        # Dictionary to store coverage results: tile_idx -> {image_name: coverage}
        tile_coverage = {idx: {} for idx in range(num_tiles)}
        
        print(f"   Checking coverage for {len(all_images)} images across {num_tiles} tiles...")
        print(f"   ⚡ Running {len(all_images)} reduceRegions() calls IN PARALLEL...")
        
        def check_image_coverage(img_data):
            """Check coverage for one image across all tiles (one reduceRegions call)"""
            img, img_name = img_data
            # Use B1 for Planet (works for both 4-band and 8-band), B2 for S2
            band_name = 'B1' if 'planet' in img_name else 'B2'
            valid_mask = img.select(band_name).mask()
            
            # Use reduceRegions to get mean coverage for ALL tiles at once
            coverage_results = valid_mask.reduceRegions(
                collection=tile_fc,
                reducer=ee.Reducer.mean(),
                scale=30
            ).getInfo()
            
            return img_name, coverage_results
        
        # Run all coverage checks in parallel
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
        for idx, tile_bounds in enumerate(tile_bounds_list):
            coverages = tile_coverage[idx]
            
            # Check if all images meet the threshold
            all_valid = all(cov >= min_percent for cov in coverages.values())
            
            if all_valid:
                valid_tiles.append(tile_bounds)
        
        print(f"   ✓ Valid tiles (ALL frames): {len(valid_tiles)}/{num_tiles} ({100*len(valid_tiles)/max(1, num_tiles):.1f}%)")
        
    except Exception as e:
        print(f"   ❌ Batched coverage check failed: {e}")
        print(f"   ⚠️ Falling back to slower per-tile checking...")
        
        # Fallback to slower per-tile method
        valid_tiles = []
        checked = 0
        
        def check_tile_all_frames(tile_bounds):
            tile_geom = ee.Geometry.Rectangle(tile_bounds)
            for planet_img, s2_img in zip(planet_images, s2_images):
                # Use B1 for Planet (works for both 4-band and 8-band), B2 for S2
                planet_valid = has_sufficient_coverage(
                    planet_img, tile_geom, 'B1',
                    min_percent=min_percent, scale=30
                )
                if not planet_valid:
                    return False
                s2_valid = has_sufficient_coverage(
                    s2_img, tile_geom, 'B2',
                    min_percent=min_percent, scale=30
                )
                if not s2_valid:
                    return False
            return True
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tile = {
                executor.submit(check_tile_all_frames, tile_bounds): tile_bounds
                for tile_bounds in tile_bounds_list
            }
            
            for future in as_completed(future_to_tile):
                tile_bounds = future_to_tile[future]
                checked += 1
                try:
                    if future.result():
                        valid_tiles.append(tile_bounds)
                    if checked % 20 == 0 or checked == num_tiles:
                        print(f"   Progress: {checked}/{num_tiles} checked, {len(valid_tiles)} valid")
                except:
                    pass
        
        print(f"   ✓ Valid tiles (ALL frames, fallback): {len(valid_tiles)}/{num_tiles} ({100*len(valid_tiles)/max(1, num_tiles):.1f}%)")
    
    return valid_tiles


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


def process_tile_5band(tile_info: Dict) -> Optional[TrainingSample5Band]:
    """
    Process a single tile across all temporal frames with 5-band output.
    """
    tile_idx = tile_info['tile_idx']
    tile_bounds = tile_info['tile_bounds']
    sequence = tile_info['sequence']
    planet_collection = tile_info['planet_collection']
    s2_collection = tile_info['s2_collection']
    zenith_lookup = tile_info.get('zenith_lookup', {'s2': {}, 'planet': {}})
    cache_dir = tile_info['cache_dir']
    region_name = tile_info['region_name']
    seq_idx = tile_info['seq_idx']
    normalize = tile_info.get('normalize', True)
    scale_factor = tile_info.get('scale_factor', 10000.0)
    
    # Download all frames in 5-band format
    downloaded_frames, cache_stats = download_all_frames_5band(
        sequence=sequence,
        tile_bounds=tile_bounds,
        planet_collection=planet_collection,
        s2_collection=s2_collection,
        cache_dir=cache_dir,
        tile_size_px=256
    )
    
    if len(downloaded_frames) != len(sequence):
        return None
    
    # Process frames
    hr_frames = []
    hr_10m_frames = []
    hr_20m_frames = []
    lr_10m_frames = []
    lr_20m_frames = []
    dates = []
    s2_zenith_angles = []
    planet_zenith_angles = []
    frame_quality_list = []
    
    all_frames_ok = True
    
    for frame_idx in range(len(sequence)):
        if frame_idx not in downloaded_frames:
            all_frames_ok = False
            break
        
        frame_data = downloaded_frames[frame_idx]
        hr_array = frame_data['hr']  # (256, 256, 5)
        lr_10m = frame_data['lr_10m']  # (77, 77, 3)
        lr_20m = frame_data['lr_20m']  # (38, 38, 2)
        pair = frame_data['pair']
        
        try:
            # Cache keys based on tile_bounds + pair IDs (stable)
            cache_key_coreg = f"{pair['planet_id']}|{pair['s2_id']}|{tile_bounds}|coreg_5b"
            cache_key_harm = f"{pair['planet_id']}|{pair['s2_id']}|{tile_bounds}|harm_5b"
            cache_hash_coreg = hashlib.md5(cache_key_coreg.encode()).hexdigest()[:16]
            cache_hash_harm = hashlib.md5(cache_key_harm.encode()).hexdigest()[:16]
            cache_path_coreg = cache_dir / "coregistered" / f"tile_{cache_hash_coreg}.npy"
            cache_path_harm = cache_dir / "harmonized" / f"tile_{cache_hash_harm}.npy"
            
            # Geographic coordinates for coregistration
            aoi_bounds_coords = [
                [tile_bounds[0], tile_bounds[1]],
                [tile_bounds[2], tile_bounds[1]],
                [tile_bounds[2], tile_bounds[3]],
                [tile_bounds[0], tile_bounds[3]]
            ]
            
            # Coregistration using 10m bands (first 3 channels of HR)
            hr_for_coreg = hr_array[:, :, :3]  # RGB only
            hr_coreg = coregister_lr_to_hr_wrapper(hr_for_coreg, lr_10m, aoi_bounds_coords, cache_path_coreg)
            
            # Apply same transform to all HR bands
            if hr_coreg.shape != hr_for_coreg.shape:
                hr_coreg = hr_for_coreg
            
            # Harmonization using 10m bands
            hr_harm = harmonize_planet_to_sentinel2_wrapper(hr_coreg, lr_10m, cache_path_harm)
            
            # Reconstruct full 5-band HR
            hr_full = hr_array.copy()
            hr_full[:, :, :3] = hr_harm  # Apply harmonized RGB
            
            # Split into 10m and 20m bands
            hr_10m = hr_full[:, :, :3]  # B2, B4, B6 → S2 B2, B3, B4
            hr_20m = hr_full[:, :, 3:]  # B7, B8 → S2 B5, B8A
            
            # Downsample 20m HR from 256 to 128 (3m→6m)
            hr_20m_downsampled = downsample_hr_for_20m(hr_20m, factor=2)
            
            # Compute frame quality
            fq = compute_combined_frame_quality(lr_10m, hr_full, frame_idx)
            frame_quality_list.append(fq)
            
            # Get zenith angles
            s2_zenith = zenith_lookup.get('s2', {}).get(pair['s2_id'])
            planet_zenith = zenith_lookup.get('planet', {}).get(pair['planet_id'])
            s2_zenith_angles.append(s2_zenith)
            planet_zenith_angles.append(planet_zenith)
            
            hr_frames.append(hr_full)
            hr_10m_frames.append(hr_10m)
            hr_20m_frames.append(hr_20m_downsampled)
            lr_10m_frames.append(lr_10m)
            lr_20m_frames.append(lr_20m)
            dates.append((pair['s2_date'], pair['planet_date']))
            
        except Exception as e:
            all_frames_ok = False
            break
    
    if not all_frames_ok or len(hr_frames) != len(sequence):
        return None
    
    # Stack into sequences
    hr_sequence = np.stack(hr_frames, axis=0)
    hr_10m_sequence = np.stack(hr_10m_frames, axis=0)
    hr_20m_sequence = np.stack(hr_20m_frames, axis=0)
    lr_10m_sequence = np.stack(lr_10m_frames, axis=0)
    lr_20m_sequence = np.stack(lr_20m_frames, axis=0)
    
    # Normalize to [0, 1]
    if normalize:
        hr_sequence = np.clip(hr_sequence / scale_factor, 0, 1).astype(np.float32)
        hr_10m_sequence = np.clip(hr_10m_sequence / scale_factor, 0, 1).astype(np.float32)
        hr_20m_sequence = np.clip(hr_20m_sequence / scale_factor, 0, 1).astype(np.float32)
        lr_10m_sequence = np.clip(lr_10m_sequence / scale_factor, 0, 1).astype(np.float32)
        lr_20m_sequence = np.clip(lr_20m_sequence / scale_factor, 0, 1).astype(np.float32)
    
    # Separate dates
    s2_dates = [d[0] for d in dates]
    planet_dates = [d[1] for d in dates]
    
    return TrainingSample5Band(
        lr_sequence=lr_10m_sequence,
        hr_sequence=hr_sequence,
        lr_20m_sequence=lr_20m_sequence,
        hr_10m_sequence=hr_10m_sequence,
        hr_20m_sequence=hr_20m_sequence,
        temporal_dates=s2_dates,
        planet_dates=planet_dates,
        region_name=region_name,
        sequence_id=seq_idx,
        tile_id=tile_idx,
        tile_bounds={'bounds': tile_bounds, 'band_mode': '5-band'},
        sensor_type='superdove',
        planet_bands=['B2', 'B4', 'B6', 'B7', 'B8'],
        sentinel_bands=['B2', 'B3', 'B4', 'B5', 'B8A'],
        s2_solar_zenith=s2_zenith_angles,
        planet_solar_zenith=planet_zenith_angles,
        frame_quality=frame_quality_list
    )


def process_single_sequence(sequence_info: Dict) -> List[TrainingSample5Band]:
    """Process a single sequence with multiple tiles in parallel."""
    sequence = sequence_info['sequence']
    valid_tiles = sequence_info['valid_tiles']
    planet_collection = sequence_info['planet_collection']
    s2_collection = sequence_info['s2_collection']
    zenith_lookup = sequence_info.get('zenith_lookup', {'s2': {}, 'planet': {}})
    cache_dir = sequence_info['cache_dir']
    region_name = sequence_info['region_name']
    seq_idx = sequence_info['seq_idx']
    max_samples_per_seq = sequence_info['max_samples_per_seq']
    tile_workers = sequence_info.get('tile_workers', 1)
    normalize = sequence_info.get('normalize', True)
    
    samples = []
    
    print(f"\n📦 Processing sequence {seq_idx} with {len(valid_tiles)} valid tiles...")
    print(f"   Using {tile_workers} parallel worker(s)")
    
    tiles_to_process = valid_tiles[:max_samples_per_seq]
    tile_infos = []
    
    for tile_idx, tile_bounds in enumerate(tiles_to_process):
        tile_info = {
            'tile_idx': tile_idx,
            'tile_bounds': tile_bounds,
            'sequence': sequence,
            'planet_collection': planet_collection,
            's2_collection': s2_collection,
            'zenith_lookup': zenith_lookup,
            'cache_dir': cache_dir,
            'region_name': region_name,
            'seq_idx': seq_idx,
            'normalize': normalize
        }
        tile_infos.append(tile_info)
    
    # Process tiles in parallel
    if tile_workers > 1:
        with ThreadPoolExecutor(max_workers=tile_workers) as executor:
            future_to_tile = {
                executor.submit(process_tile_5band, tile_info): (tile_info['tile_idx'], tile_info['tile_bounds'])
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
                        print(f"   Tile {tile_idx+1}/{len(tiles_to_process)}: ✓ ({processed}/{len(tiles_to_process)} done)")
                    else:
                        print(f"   Tile {tile_idx+1}/{len(tiles_to_process)}: ✗ ({processed}/{len(tiles_to_process)} done)")
                except Exception as e:
                    print(f"   Tile {tile_idx+1}/{len(tiles_to_process)}: ✗ Error: {e}")
    else:
        for tile_idx, tile_info in enumerate(tile_infos):
            print(f"   Tile {tile_idx+1}/{len(tiles_to_process)}...", end=" ", flush=True)
            try:
                sample = process_tile_5band(tile_info)
                if sample is not None:
                    samples.append(sample)
                    print("✓")
                else:
                    print("✗")
            except Exception as e:
                print(f"✗ Error: {e}")
    
    print(f"   Sequence {seq_idx}: Generated {len(samples)} samples")
    return samples


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='STA-1E Grid-based 5-band dataset generation')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to generate')
    parser.add_argument('--output-dir', type=str, 
                       default='./data/datasets/sta1e_5band',
                       help='Output directory')
    parser.add_argument('--cache-dir', type=str,
                       default='./cache/sta1e_5band',
                       help='Cache directory')
    parser.add_argument('--sequence-length', type=int, default=3,
                       help='Temporal sequence length')
    parser.add_argument('--coverage-threshold', type=float, default=90.0,
                       help='Minimum coverage percentage')
    parser.add_argument('--tile-workers', type=int, default=20,
                       help='Parallel workers for tile processing')
    parser.add_argument('--coverage-workers', type=int, default=16,
                       help='Number of parallel workers for coverage checking')
    parser.add_argument('--year', type=str, default=None,
                       help='Specific year to process (2020-2024)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                       help='Keep raw reflectance values')
    parser.set_defaults(normalize=True)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🌐 STA-1E GRID-BASED 5-BAND DATASET GENERATION")
    print("=" * 80)
    print(f"\n⚙️  CONFIGURATION:")
    print("─" * 80)
    print(f"  Output format:         5-band SuperDove (B2,B4,B6,B7,B8)")
    print(f"  HR resolution:         256×256 @ 3m (RGB + RedEdge + NIR)")
    print(f"  LR 10m resolution:     77×77 @ 10m (B2,B3,B4)")
    print(f"  LR 20m resolution:     38×38 @ 20m (B5,B8A)")
    print(f"  Normalization:         {'[0, 1]' if args.normalize else 'Raw (0-10000)'}")
    print(f"  Sequence length:       {args.sequence_length} frames")
    print(f"  Tile workers:          {args.tile_workers}")
    print(f"  Coverage workers:      {args.coverage_workers}")
    print(f"  Coverage threshold:    {args.coverage_threshold}%")
    print("=" * 80)
    
    start_time = time.time()
    
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
        return
    
    print(f"\n📍 Found {len(sta1e_regions)} STA-1E time periods to process")
    
    all_samples = []
    
    for region_idx, region_config in enumerate(sta1e_regions):
        print(f"\n{'='*80}")
        print(f"PROCESSING {region_config['name']} ({region_config['start']} to {region_config['end']})")
        print(f"{'='*80}")
        
        # Find image pairs using ALL available collections
        print(f"\n🔍 Finding image pairs...")
        
        # Cache key for pairs
        pairs_cache_key = f"{region_config['name']}|{region_config['start']}|{region_config['end']}|5band"
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
        
        # Build collections (always needed for downloads)
        # If pairs are cached, we can skip full find_pairs and just build collections
        if cached_matched_pairs is not None:
            # Just build collections without expensive pair matching
            planet_collection = ee.ImageCollection(PLANET_COLLECTION_ID).merge(
                ee.ImageCollection(PLANET_COLLECTION_ID_FALLBACK)
            ).merge(
                ee.ImageCollection(PLANET_COLLECTION_ID_ADDITIONAL)
            ).filterBounds(region_config["aoi"]).map(lambda img: img.set('id_pref', img.id()))
            
            s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(
                region_config["aoi"]
            ).filterDate(region_config["start"], region_config["end"]).map(
                lambda img: img.set('id_pref', img.id())
            )
            
            matched_pairs = cached_matched_pairs
            print(f"   ✅ Using cached pairs (skipped expensive pair matching)")
        else:
            # Full pair matching needed
            pair_results = find_pairs(
                region_config["aoi"],
                region_config["start"],
                region_config["end"],
                PLANET_COLLECTION_ID,
                planet_coll_id_fallback=PLANET_COLLECTION_ID_FALLBACK,
                planet_coll_id_additional=PLANET_COLLECTION_ID_ADDITIONAL
            )
            
            matched_pairs = pair_results.get('matched_pairs', [])
            s2_collection = pair_results.get('s2_collection')
            planet_collection = pair_results.get('planet_collection')
            
            # Cache pairs for next time
            try:
                with open(pairs_cache_file, 'wb') as f:
                    pickle.dump(matched_pairs, f)
            except:
                pass
        
        print(f"   Found {len(matched_pairs)} matched pairs")
        
        if len(matched_pairs) < args.sequence_length:
            print(f"   ⚠️ Insufficient pairs, skipping this period")
            continue
        
        # Zenith extraction
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
                    
            except Exception as e:
                print(f"   ⚠️ Zenith extraction failed: {e}")
                zenith_lookup = {'s2': {}, 'planet': {}}
        
        # Generate sequences
        max_sequences = len(matched_pairs) - args.sequence_length + 1
        num_sequences = min(args.max_samples or max_sequences, max_sequences)
        
        print(f"\n📅 Processing {num_sequences} sequences...")
        
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
            
            # Generate tile grid
            all_tiles = generate_tile_grid(region_config["aoi"], tile_size_deg=0.007)
            
            # Coverage checking with caching (cache key based on ALL sequence dates)
            sequence_dates = tuple(sorted([p['s2_date'] for p in sequence]))
            dates_str = "_".join(sequence_dates)
            coverage_cache_hash = hashlib.md5(dates_str.encode()).hexdigest()[:16]
            coverage_cache_file = cache_dir / f"coverage_{coverage_cache_hash}.pkl"
            
            if coverage_cache_file.exists():
                try:
                    with open(coverage_cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        # Handle both old and new cache formats
                        if isinstance(cached_data, dict):
                            valid_tiles = cached_data.get('valid_tiles', cached_data)
                            total_checked = cached_data.get('total_checked', len(valid_tiles))
                        else:
                            valid_tiles = cached_data
                            total_checked = len(valid_tiles)
                    print(f"   ✅ Loaded {len(valid_tiles)}/{total_checked} cached valid tiles")
                except:
                    valid_tiles = None
            else:
                valid_tiles = None
            
            if valid_tiles is None:
                # Use batched coverage checking across ALL frames
                valid_tiles = filter_tiles_batched(
                    planet_images, s2_images, all_tiles,
                    min_percent=args.coverage_threshold,
                    max_workers=args.coverage_workers
                )
                
                # Save coverage results to cache for future runs
                cache_data = {
                    'valid_tiles': valid_tiles,
                    'total_checked': len(all_tiles),
                    'sequence_dates': sequence_dates,
                    'coverage_threshold': args.coverage_threshold
                }
                try:
                    with open(coverage_cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)
                    print(f"   ✅ Cache saved: {coverage_cache_file.name}")
                except:
                    pass
            
            if not valid_tiles:
                print(f"   ⚠️ No valid tiles found, skipping sequence")
                continue
            
            remaining = args.max_samples - len(all_samples) if args.max_samples else len(valid_tiles)
            max_samples_this_seq = min(len(valid_tiles), remaining)
            
            sequence_info = {
                'sequence': sequence,
                'valid_tiles': valid_tiles[:max_samples_this_seq],
                'planet_collection': planet_collection,
                's2_collection': s2_collection,
                'zenith_lookup': zenith_lookup,
                'cache_dir': cache_dir,
                'region_name': region_config['name'],
                'seq_idx': seq_idx,
                'max_samples_per_seq': max_samples_this_seq,
                'tile_workers': args.tile_workers,
                'normalize': args.normalize
            }
            
            samples = process_single_sequence(sequence_info)
            all_samples.extend(samples)
            
            print(f"   Total samples so far: {len(all_samples)}")
    
    # Save dataset
    if len(all_samples) == 0:
        print("\n❌ No samples generated!")
        return
    
    dataset_path = output_dir / 'dataset_5band.pkl'
    print(f"\n💾 Saving {len(all_samples)} samples to {dataset_path}...")
    
    dataset = {
        'samples': all_samples,
        'metadata': {
            'region': 'STA-1E',
            'sensor_type': 'superdove',
            'n_samples': len(all_samples),
            'sequence_length': args.sequence_length,
            'hr_bands': 5,
            'hr_10m_bands': 3,
            'hr_20m_bands': 2,
            'lr_10m_bands': 3,
            'lr_20m_bands': 2,
            'planet_bands': ['B2', 'B4', 'B6', 'B7', 'B8'],
            's2_bands_10m': ['B2', 'B3', 'B4'],
            's2_bands_20m': ['B5', 'B8A'],
            'normalized': args.normalize,
            'scale_factor': 10000.0,
            'hr_tile_size': 256,
            'lr_10m_tile_size': 77,
            'lr_20m_tile_size': 38
        }
    }
    
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=4)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✅ STA-1E 5-BAND GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Samples generated: {len(all_samples)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Avg time per sample: {elapsed/len(all_samples):.1f} seconds")
    print(f"Dataset saved: {dataset_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
