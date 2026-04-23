#!/usr/bin/env python3
"""
Download 2025-2026 PlanetScope + Sentinel-2 inference data from GEE.

This script downloads NEW image data from GEE repositories
`projects/staslatest/assets/STAs` and `projects/stasadditionalfree/assets/testSTAs_ic`
for all STAs (STA-1E, STA-1W, STA-2, STA-3/4, STA-5/6).

This data is for INFERENCE ONLY — to super-resolve, reconstruct full images,
and compare SR vs Sentinel-2 vs PlanetScope for downstream classification.

Output format:
  - TrainingSample (same as grid training scripts, needs convert_cache_to_5band.py)
  - Same grid dimensions as training data (0.007° tiles)
  - Adaptive band handling: 8-band SuperDove or 4-band Dove per sequence
  - 256×256 HR, 77×77 LR 10m, 38×38 LR 20m

Usage:
    # Download all STAs for 2025
    python scripts/preprocessing/download_inference_data.py --year 2025

    # Download only STA-2 for 2025
    python scripts/preprocessing/download_inference_data.py --year 2025 --sta STA-2

    # List available STAs
    python scripts/preprocessing/download_inference_data.py --list-stas
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
import hashlib
import io
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import ee

from Auth.AuthGEE import find_pairs

# GEE repositories with 2025-2026 PlanetScope data
PLANET_COLLECTION_LATEST = 'projects/staslatest/assets/STAs'
PLANET_COLLECTION_ADDITIONAL = 'projects/stasadditionalfree/assets/testSTAs_ic'


# =============================================================================
# AOI DEFINITIONS — same geometries as training data
# =============================================================================

def get_inference_regions(year: str = "2025", chunk_months: int = 4) -> List[Dict]:
    """
    Region definitions for 2025-2026 inference data.
    Same AOIs as training data, new time periods.
    Splits the year into shorter chunks to stay within GEE memory limits.
    """
    aois = {
        "STA-2": ee.Geometry.Polygon([[
            [-80.61495827, 26.43609898],
            [-80.53742967, 26.33549131],
            [-80.48100774, 26.42148366],
            [-80.544935, 26.42144199],
            [-80.54505831, 26.43527516],
            [-80.61739151, 26.43780818],
            [-80.61495827, 26.43609898]
        ]]),
        "STA-1E": ee.Geometry.Polygon([[
            [-80.36388849, 26.6843562],
            [-80.36395033, 26.67959061],
            [-80.34775062, 26.64813207],
            [-80.33282065, 26.63369505],
            [-80.31717686, 26.62427804],
            [-80.316679, 26.65008888],
            [-80.30003641, 26.65027958],
            [-80.29854709, 26.68272326],
            [-80.36388849, 26.6843562]
        ]]),
        "STA-1W": ee.Geometry.Polygon([[
            [-80.45309541, 26.68276268],
            [-80.45309541, 26.57253594],
            [-80.37008749, 26.67612901],
            [-80.37428635, 26.69141472],
            [-80.45406437, 26.69112633],
            [-80.45309541, 26.68276268]
        ]]),
        "STA-3/4": ee.Geometry.Polygon([[
            [-80.81055951, 26.43631098],
            [-80.68733359, 26.44017758],
            [-80.6888542, 26.47943094],
            [-80.66261472, 26.48062023],
            [-80.64501101, 26.47140293],
            [-80.5389865, 26.33573306],
            [-80.54198348, 26.33350413],
            [-80.77568272, 26.33197003],
            [-80.79271676, 26.35535459],
            [-80.81055951, 26.43631098]
        ]]),
        "STA-5/6": ee.Geometry.Polygon([[
            [-80.94783564, 26.46274662],
            [-80.94748873, 26.40605488],
            [-80.88007816, 26.32885213],
            [-80.88198692, 26.46306834],
            [-80.94783564, 26.46274662]
        ]]),
    }

    # Split year into chunks to stay within GEE memory limits
    from datetime import datetime as _dt
    start = _dt.strptime(f"{year}-01-01", "%Y-%m-%d")
    end = _dt.strptime(f"{int(year) + 1}-01-01", "%Y-%m-%d")

    chunks = []
    current = start
    while current < end:
        chunk_end_month = current.month + chunk_months
        chunk_end_year = current.year + (chunk_end_month - 1) // 12
        chunk_end_month = ((chunk_end_month - 1) % 12) + 1
        chunk_end = _dt(chunk_end_year, chunk_end_month, 1)
        if chunk_end > end:
            chunk_end = end
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end

    regions = []
    for name, aoi in aois.items():
        for chunk_start, chunk_end in chunks:
            regions.append({
                "name": name,
                "aoi": aoi,
                "start": chunk_start,
                "end": chunk_end,
            })

    return regions


# =============================================================================
# DATACLASSES
# =============================================================================

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
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None
    lr_20m_sequence: Optional[np.ndarray] = None  # (T, H/2, W/2, C_20m)
    frame_quality: Optional[List[Dict]] = None


def normalize_sample(sample: TrainingSample, scale_factor: float = 10000.0) -> TrainingSample:
    """Normalize sample arrays from raw reflectance (0-10000) to [0, 1] range."""
    sample.lr_sequence = np.clip(sample.lr_sequence / scale_factor, 0, 1).astype(np.float32)
    sample.hr_sequence = np.clip(sample.hr_sequence / scale_factor, 0, 1).astype(np.float32)
    if sample.lr_20m_sequence is not None:
        sample.lr_20m_sequence = np.clip(sample.lr_20m_sequence / scale_factor, 0, 1).astype(np.float32)
    return sample


@dataclass
class FrameQualityStats:
    """Quality statistics for a single frame."""
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


# =============================================================================
# QUALITY STATS
# =============================================================================

def compute_frame_quality_stats(array: np.ndarray,
                                zero_threshold: float = 0.05,
                                outlier_percentile: float = 99.5,
                                min_valid_fraction: float = 0.90) -> FrameQualityStats:
    """Compute local quality statistics for a downloaded frame."""
    total_pixels = array.size

    nan_mask = np.isnan(array)
    nan_fraction = np.sum(nan_mask) / total_pixels

    zero_mask = np.abs(array) < zero_threshold
    zero_fraction = np.sum(zero_mask & ~nan_mask) / total_pixels

    valid_mask = ~nan_mask & ~zero_mask
    valid_pixels = array[valid_mask]

    if len(valid_pixels) == 0:
        return FrameQualityStats(zero_fraction=float(zero_fraction),
                                 nan_fraction=float(nan_fraction),
                                 outlier_fraction=0.0, valid_fraction=0.0,
                                 mean_reflectance=0.0, std_reflectance=0.0,
                                 is_good=False)

    outlier_threshold = np.percentile(valid_pixels, outlier_percentile)
    outlier_mask = (array > outlier_threshold) & valid_mask
    outlier_fraction = np.sum(outlier_mask) / total_pixels

    final_valid_mask = valid_mask & ~outlier_mask
    valid_fraction = np.sum(final_valid_mask) / total_pixels

    final_valid_pixels = array[final_valid_mask]
    mean_refl = float(np.mean(final_valid_pixels)) if len(final_valid_pixels) > 0 else 0.0
    std_refl = float(np.std(final_valid_pixels)) if len(final_valid_pixels) > 0 else 0.0

    return FrameQualityStats(
        zero_fraction=float(zero_fraction),
        nan_fraction=float(nan_fraction),
        outlier_fraction=float(outlier_fraction),
        valid_fraction=float(valid_fraction),
        mean_reflectance=mean_refl,
        std_reflectance=std_refl,
        is_good=(valid_fraction >= min_valid_fraction)
    )


# =============================================================================
# GEE DOWNLOAD
# =============================================================================

def _compute_pixels_download(img: ee.Image, bounds: List[float], bands: List[str],
                             width: int = 256, height: int = 256) -> Optional[np.ndarray]:
    """Download image region using ee.data.computePixels."""
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
        return np.stack([arr[b].astype(np.float32) for b in bands], axis=-1)
    except Exception:
        return None


def download_all_frames_batched(sequence: List[Dict], tile_bounds: List[float],
                                planet_collection, s2_collection,
                                planet_bands: List[str], sentinel_bands: List[str],
                                cache_dir: Path, tile_size_px: int = 256) -> Tuple[Dict, Dict]:
    """Download all frames for a tile with caching."""
    results = {}
    cache_stats = {'hits': 0, 'misses': 0}

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
                results[frame_idx] = {'hr': cached['hr_array'], 'lr': lr_data, 'pair': pair}
                cache_stats['hits'] += 1
                continue
            except Exception:
                pass

        frames_to_download.append({'frame_idx': frame_idx, 'pair': pair, 'cache_path': cache_path})
        cache_stats['misses'] += 1

    if not frames_to_download:
        return results, cache_stats

    def download_frame_batched(task):
        pair = task['pair']
        frame_idx = task['frame_idx']
        try:
            planet_img = planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
            s2_img = s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()

            scale_factor_10m = 10.0 / 3.0
            scale_factor_20m = 20.0 / 3.0
            lr_10m_size = int(round(tile_size_px / scale_factor_10m))
            lr_20m_size = int(round(tile_size_px / scale_factor_20m))

            hr_array = _compute_pixels_download(planet_img, tile_bounds, planet_bands,
                                                width=tile_size_px, height=tile_size_px)

            bands_10m = [b for b in sentinel_bands if b in ['B2', 'B3', 'B4', 'B8']]
            bands_20m = [b for b in sentinel_bands if b in ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']]
            lr_data = {}

            if bands_10m:
                lr_10m = _compute_pixels_download(s2_img, tile_bounds, bands_10m,
                                                  width=lr_10m_size, height=lr_10m_size)
                if lr_10m is not None:
                    lr_data['bands_10m'] = lr_10m
                    lr_data['band_names_10m'] = bands_10m

            if bands_20m:
                lr_20m = _compute_pixels_download(s2_img, tile_bounds, bands_20m,
                                                  width=lr_20m_size, height=lr_20m_size)
                if lr_20m is not None:
                    lr_data['bands_20m'] = lr_20m
                    lr_data['band_names_20m'] = bands_20m

            if hr_array is not None and lr_data:
                task['cache_path'].parent.mkdir(parents=True, exist_ok=True)
                try:
                    save_dict = {'hr_array': hr_array, 'planet_bands': planet_bands, 's2_bands': sentinel_bands}
                    if 'bands_10m' in lr_data:
                        save_dict['lr_10m'] = lr_data['bands_10m']
                        save_dict['band_names_10m'] = lr_data.get('band_names_10m', [])
                    if 'bands_20m' in lr_data:
                        save_dict['lr_20m'] = lr_data['bands_20m']
                        save_dict['band_names_20m'] = lr_data.get('band_names_20m', [])
                    np.savez_compressed(task['cache_path'], **save_dict)
                except Exception:
                    pass
                return frame_idx, hr_array, lr_data, pair

        except Exception:
            pass
        return frame_idx, None, None, task['pair']

    max_workers = min(20, len(frames_to_download))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_frame_batched, task): task for task in frames_to_download}
        for future in as_completed(futures):
            try:
                frame_idx, hr_array, lr_data, pair = future.result()
                if hr_array is not None and lr_data is not None:
                    results[frame_idx] = {'hr': hr_array, 'lr': lr_data, 'pair': pair}
            except Exception:
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
    except Exception:
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

    print(f"   Grid dimensions: {num_rows} rows x {num_cols} cols = {num_rows * num_cols} potential tiles")

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
    """Coregistration wrapper with caching."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return np.load(cache_path)

    try:
        from scripts.preprocessing.numpy_efficient_pipeline import coregister_lr_to_hr
        hr_coreg = coregister_lr_to_hr(hr_array, lr_array, aoi_bounds)
        np.save(cache_path, hr_coreg)
        return hr_coreg
    except Exception:
        np.save(cache_path, hr_array)
        return hr_array


def harmonize_planet_to_sentinel2_wrapper(hr_array, lr_array, cache_path):
    """Harmonization wrapper with caching."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return np.load(cache_path)

    try:
        from scripts.preprocessing.numpy_efficient_pipeline import harmonize_planet_to_sentinel2
        hr_harm = harmonize_planet_to_sentinel2(hr_array, lr_array)
        np.save(cache_path, hr_harm)
        return hr_harm
    except Exception:
        np.save(cache_path, hr_array)
        return hr_array


# =============================================================================
# TILE PROCESSING
# =============================================================================

def process_tile(tile_info) -> Optional[TrainingSample]:
    """
    Process a single tile across all temporal frames.
    Downloads, coregisters, harmonizes, and creates a sample.
    Uses adaptive band strategy (4-band or 8-band) per sequence.
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
    lr_20m_frames = []
    dates = []
    s2_zenith_angles = []
    planet_zenith_angles = []
    frame_quality_stats = []

    # Determine sequence band strategy BEFORE downloading
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

    lr_frames = []
    hr_frames = []
    lr_20m_frames = []
    dates = []
    s2_zenith_angles = []
    planet_zenith_angles = []
    frame_quality_stats = []

    for frame_idx in range(len(sequence)):
        if frame_idx not in downloaded_frames:
            return None

        frame_data = downloaded_frames[frame_idx]
        hr_array = frame_data['hr']
        lr_data = frame_data['lr']
        pair = frame_data['pair']

        lr_array_10m = lr_data.get('bands_10m') if isinstance(lr_data, dict) else lr_data
        lr_array_20m = lr_data.get('bands_20m') if isinstance(lr_data, dict) else None

        if lr_array_10m is None:
            return None

        try:
            # Cache keys for coreg/harmonization
            cache_key_coreg = f"{pair['planet_id']}|{pair['s2_id']}|{tile_bounds}|coreg_inf"
            cache_key_harm = f"{pair['planet_id']}|{pair['s2_id']}|{tile_bounds}|harm_inf"
            cache_hash_coreg = hashlib.md5(cache_key_coreg.encode()).hexdigest()[:16]
            cache_hash_harm = hashlib.md5(cache_key_harm.encode()).hexdigest()[:16]
            cache_path_coreg = cache_dir / "coregistered" / f"tile_{cache_hash_coreg}.npy"
            cache_path_harm = cache_dir / "harmonized" / f"tile_{cache_hash_harm}.npy"

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

            hr_array = harmonize_planet_to_sentinel2_wrapper(
                hr_array=hr_array,
                lr_array=lr_for_coreg,
                cache_path=cache_path_harm
            )

            # Zenith angles
            s2_zenith = zenith_lookup.get('s2', {}).get(pair['s2_id'])
            planet_zenith = zenith_lookup.get('planet', {}).get(pair['planet_id'])
            s2_zenith_angles.append(s2_zenith)
            planet_zenith_angles.append(planet_zenith)

            # Quality stats
            lr_stats = compute_frame_quality_stats(lr_array_10m)
            hr_stats = compute_frame_quality_stats(hr_array)
            frame_quality_stats.append({
                'frame_idx': frame_idx,
                'lr_valid_frac': lr_stats.valid_fraction,
                'hr_valid_frac': hr_stats.valid_fraction,
                'is_good': lr_stats.is_good and hr_stats.is_good
            })

            lr_frames.append(lr_array_10m)
            hr_frames.append(hr_array)
            lr_20m_frames.append(lr_array_20m)
            dates.append((pair['s2_date'], pair['planet_date']))

        except Exception as e:
            print(f"      Tile {tile_idx} frame {frame_idx} error: {e}")
            return None

    if len(lr_frames) != len(sequence):
        return None

    # Stack 20m bands if available
    lr_20m_stack = None
    if lr_20m_frames and all(f is not None for f in lr_20m_frames):
        try:
            lr_20m_stack = np.stack(lr_20m_frames, axis=0)
        except Exception:
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


def process_single_sequence(sequence_info) -> List[TrainingSample]:
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

    print(f"\n   Processing sequence {seq_idx} with {len(valid_tiles)} valid tiles...")

    tiles_to_process = valid_tiles[:max_samples_per_seq]
    tile_infos = []

    for tile_idx, tile_bounds in enumerate(tiles_to_process):
        tile_infos.append({
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
        })

    if tile_workers > 1:
        with ThreadPoolExecutor(max_workers=tile_workers) as executor:
            future_to_tile = {
                executor.submit(process_tile, ti): (ti['tile_idx'], ti['tile_bounds'])
                for ti in tile_infos
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
                            print(f"      Progress: {processed}/{len(tiles_to_process)}, {len(samples)} valid")
                except Exception:
                    pass
    else:
        for tile_info in tile_infos:
            try:
                sample = process_tile(tile_info)
                if sample is not None:
                    samples.append(sample)
            except Exception:
                pass

    print(f"   Sequence {seq_idx}: Generated {len(samples)} samples")
    return samples


# =============================================================================
# BAND DETECTION
# =============================================================================

def detect_band_config(planet_collection, matched_pairs, cache_dir):
    """
    Check which dates have SuperDove 8-band or Dove 4-band data.
    Returns dict: planet_date -> {'mode': '8-band'|'4-band', 'planet_bands': [...], 'sentinel_bands': [...]}
    """
    dates_hash = hashlib.md5(
        "_".join(sorted([p['planet_date'] for p in matched_pairs])).encode()
    ).hexdigest()[:16]
    band_cache_file = cache_dir / f"band_detection_inf_{dates_hash}.pkl"

    if band_cache_file.exists():
        with open(band_cache_file, 'rb') as f:
            date_band_config = pickle.load(f)
        # Handle legacy string-only cache format
        if date_band_config and isinstance(next(iter(date_band_config.values())), str):
            band_cache_file.unlink()
        else:
            modes_8band = sum(1 for cfg in date_band_config.values() if cfg['mode'] == '8-band')
            print(f"   Loaded cached band detection: {modes_8band} 8-band, "
                  f"{len(date_band_config) - modes_8band} 4-band")
            return date_band_config

    all_planet_dates = list(set([p['planet_date'] for p in matched_pairs]))
    date_band_config = {}

    def detect_bands_for_date(pdate):
        try:
            planet_img = planet_collection.filter(ee.Filter.eq('date_key', pdate)).first()
            available_bands = planet_img.bandNames().getInfo()
            has_extended = any(band in available_bands for band in ['B5', 'B6', 'B7', 'B8'])
            if has_extended and all(band in available_bands for band in ['B5', 'B6', 'B7', 'B8']):
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
        except Exception:
            return pdate, {
                'mode': '4-band',
                'planet_bands': ['B1', 'B2', 'B3', 'B4'],
                'sentinel_bands': ['B2', 'B3', 'B4', 'B8A']
            }

    with ThreadPoolExecutor(max_workers=min(20, len(all_planet_dates))) as executor:
        futures = {executor.submit(detect_bands_for_date, pd): pd for pd in all_planet_dates}
        completed = 0
        for future in as_completed(futures):
            pdate, config = future.result()
            date_band_config[pdate] = config
            completed += 1
            if completed % 10 == 0 or completed == len(all_planet_dates):
                print(f"      Band detection: {completed}/{len(all_planet_dates)}")

    with open(band_cache_file, 'wb') as f:
        pickle.dump(date_band_config, f)

    modes_8band = sum(1 for cfg in date_band_config.values() if cfg['mode'] == '8-band')
    print(f"   Band detection: {modes_8band} 8-band, {len(date_band_config) - modes_8band} 4-band")
    return date_band_config


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Download 2025-2026 inference data from GEE')
    parser.add_argument('--year', type=str, default='2025',
                        help='Year to process (default: 2025)')
    parser.add_argument('--sta', type=str, default=None,
                        help='Specific STA to process (e.g. STA-2). Default: all STAs')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per STA')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./cache/inference_{year})')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Cache directory (default: ./cache/inference_{year}_cache)')
    parser.add_argument('--sequence-length', type=int, default=3,
                        help='Temporal sequence length (default: 3)')
    parser.add_argument('--coverage-threshold', type=float, default=65.0,
                        help='Min coverage percentage for valid tiles (default: 65)')
    parser.add_argument('--tile-workers', type=int, default=20,
                        help='Parallel workers for tile processing (default: 20)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize data to [0, 1] (default: True)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Keep raw reflectance values')
    parser.add_argument('--list-stas', action='store_true',
                        help='List available STAs and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without downloading')
    parser.add_argument('--s2-cloud-pct-max', type=float, default=60,
                        help='Max S2 scene-level CLOUDY_PIXEL_PERCENTAGE (default: 60)')
    parser.add_argument('--s2-aoi-cloud-frac-max', type=float, default=0.25,
                        help='Max S2 AOI cloud fraction (default: 0.25)')
    parser.add_argument('--planet-q-cloud-max', type=float, default=25,
                        help='Max Planet cloud cover percentage (default: 25)')

    args = parser.parse_args()

    # Defaults
    if args.output_dir is None:
        args.output_dir = f'./cache/inference_{args.year}'
    if args.cache_dir is None:
        args.cache_dir = f'./cache/inference_{args.year}_cache'

    # List STAs mode
    if args.list_stas:
        print("Available STAs:")
        seen = {}
        for r in get_inference_regions(args.year):
            name = r['name']
            if name not in seen:
                seen[name] = []
            seen[name].append(f"{r['start']} to {r['end']}")
        for name, periods in seen.items():
            print(f"  - {name}  ({len(periods)} chunks: {', '.join(periods)})")
        return

    print("=" * 80)
    print(f"INFERENCE DATA DOWNLOAD — {args.year}")
    print("=" * 80)
    print(f"  Planet collections: {PLANET_COLLECTION_LATEST}")
    print(f"                      {PLANET_COLLECTION_ADDITIONAL}")
    print(f"  Year:               {args.year}")
    print(f"  STA filter:         {args.sta if args.sta else 'All STAs'}")
    print(f"  Sequence length:    {args.sequence_length}")
    print(f"  Coverage threshold: {args.coverage_threshold}%")
    print(f"  S2 cloud pct max:   {args.s2_cloud_pct_max}%")
    print(f"  S2 AOI cloud max:   {args.s2_aoi_cloud_frac_max}")
    print(f"  Planet q_cloud max: {args.planet_q_cloud_max}%")
    print(f"  Tile workers:       {args.tile_workers}")
    print(f"  Normalize:          {'[0, 1]' if args.normalize else 'Raw (0-10000)'}")
    print(f"  Output:             {args.output_dir}")
    print(f"  Cache:              {args.cache_dir}")
    print("=" * 80)

    start_time = time.time()

    # Initialize GEE
    try:
        ee.Initialize(project='remotesensingeduart')
    except Exception:
        print("Initializing GEE...")
        from Auth.AuthGEE import ee as _ee

    # Get regions
    regions = get_inference_regions(args.year)
    if args.sta:
        regions = [r for r in regions if r['name'] == args.sta]
        if not regions:
            print(f"No region found for '{args.sta}'. Available STAs:")
            for r in get_inference_regions(args.year):
                print(f"  - {r['name']}")
            return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_samples = 0

    for region_config in regions:
        sta_name = region_config['name']
        sta_safe = sta_name.replace('/', '_')
        sta_output_dir = output_dir / sta_safe
        sta_output_dir.mkdir(parents=True, exist_ok=True)
        sta_cache_dir = cache_dir / sta_safe
        sta_cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"  {sta_name}  ({region_config['start']} to {region_config['end']})")
        print(f"{'=' * 70}")

        # Find image pairs — merge latest + additional collections
        print(f"\n   Finding image pairs from {PLANET_COLLECTION_LATEST} + {PLANET_COLLECTION_ADDITIONAL}...")

        pairs_cache_key = f"{sta_name}|{region_config['start']}|{region_config['end']}|latest+additional|cf{args.s2_aoi_cloud_frac_max}|pq{args.planet_q_cloud_max}"
        pairs_cache_hash = hashlib.md5(pairs_cache_key.encode()).hexdigest()[:16]
        pairs_cache_file = sta_cache_dir / f"matched_pairs_{pairs_cache_hash}.pkl"

        cached_matched_pairs = None
        if pairs_cache_file.exists():
            try:
                with open(pairs_cache_file, 'rb') as f:
                    cached_matched_pairs = pickle.load(f)
                print(f"   Loaded {len(cached_matched_pairs)} cached matched pairs")
            except Exception:
                cached_matched_pairs = None

        try:
            pair_results = find_pairs(
                region_config["aoi"],
                region_config["start"],
                region_config["end"],
                PLANET_COLLECTION_LATEST,
                planet_coll_id_additional=PLANET_COLLECTION_ADDITIONAL,
                s2_cloud_pct_max=args.s2_cloud_pct_max,
                s2_aoi_cloud_frac_max=args.s2_aoi_cloud_frac_max,
                planet_q_cloud_max=args.planet_q_cloud_max,
                lightweight=True,
            )
        except Exception as e:
            print(f"   ERROR finding pairs for {sta_name}: {e}")
            traceback.print_exc()
            continue

        if cached_matched_pairs is not None:
            matched_pairs = cached_matched_pairs
        else:
            matched_pairs = pair_results.get('matched_pairs', [])
            try:
                with open(pairs_cache_file, 'wb') as f:
                    pickle.dump(matched_pairs, f)
                print(f"   Cached {len(matched_pairs)} matched pairs")
            except Exception:
                pass

        s2_collection = pair_results.get('s2_collection')
        planet_collection = pair_results.get('planet_collection')

        print(f"   Found {len(matched_pairs)} matched pairs")

        if len(matched_pairs) < args.sequence_length:
            print(f"   Insufficient pairs for sequence length {args.sequence_length}, skipping")
            continue

        # Band detection — adaptive 4-band / 8-band
        print(f"\n   Detecting band availability...")
        date_band_config = detect_band_config(planet_collection, matched_pairs, sta_cache_dir)

        modes_8band = sum(1 for cfg in date_band_config.values() if cfg['mode'] == '8-band')
        print(f"   8-band pairs: {modes_8band} / {len(matched_pairs)} total")

        if len(matched_pairs) < args.sequence_length:
            print(f"   Insufficient pairs, skipping {sta_name}")
            continue

        # Default bands (used for 4-band fallback)
        planet_bands = ['B1', 'B2', 'B3', 'B4']
        sentinel_bands = ['B2', 'B3', 'B4', 'B8A']

        # Solar zenith extraction
        print(f"\n   Extracting solar zenith angles...")
        zenith_cache_file = sta_cache_dir / f"zenith_data_{pairs_cache_hash}.pkl"
        zenith_lookup = None

        if zenith_cache_file.exists():
            try:
                with open(zenith_cache_file, 'rb') as f:
                    zenith_lookup = pickle.load(f)
                print(f"   Loaded cached zenith data")
            except Exception:
                zenith_lookup = None

        if zenith_lookup is None:
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
                print(f"   Zenith extraction complete")

            except Exception as e:
                print(f"   Zenith extraction failed: {e}")
                zenith_lookup = {'s2': {}, 'planet': {}}

        if args.dry_run:
            max_seqs = len(matched_pairs) - args.sequence_length + 1
            print(f"\n   [DRY RUN] Would process up to {max_seqs} sequences for {sta_name}")
            print(f"   [DRY RUN] Dates: {matched_pairs[0]['s2_date']} to {matched_pairs[-1]['s2_date']}")
            continue

        # Process sequences
        max_possible_sequences = len(matched_pairs) - args.sequence_length + 1
        remaining = (args.max_samples - total_samples) if args.max_samples else max_possible_sequences
        num_sequences = min(remaining, max_possible_sequences)

        print(f"\n   Processing up to {num_sequences} sequences")

        # Checkpoint directory for per-sequence results
        seq_checkpoint_dir = sta_cache_dir / "seq_checkpoints"
        seq_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        sta_samples = []

        for seq_idx in range(num_sequences):
            if args.max_samples and total_samples + len(sta_samples) >= args.max_samples:
                break

            sequence = matched_pairs[seq_idx:seq_idx + args.sequence_length]
            if len(sequence) < args.sequence_length:
                break

            print(f"\n   --- Sequence {seq_idx + 1}/{num_sequences} ---")
            print(f"   Dates: {[p['s2_date'] for p in sequence]}")

            # Check for sequence checkpoint — skip if already processed
            seq_dates_key = "_".join(sorted([p['s2_date'] for p in sequence]))
            seq_ckpt_hash = hashlib.md5(seq_dates_key.encode()).hexdigest()[:16]
            seq_ckpt_file = seq_checkpoint_dir / f"seq_{seq_idx}_{seq_ckpt_hash}.pkl"

            if seq_ckpt_file.exists():
                try:
                    with open(seq_ckpt_file, 'rb') as f:
                        cached_samples = pickle.load(f)
                    sta_samples.extend(cached_samples)
                    print(f"   Restored {len(cached_samples)} samples from checkpoint")
                    print(f"   {sta_name} samples so far: {len(sta_samples)}")
                    continue
                except Exception:
                    pass  # Corrupted checkpoint, reprocess

            # Get images for coverage checking
            planet_images = []
            s2_images = []
            for pair in sequence:
                planet_images.append(
                    planet_collection.filter(ee.Filter.eq("id_pref", pair["planet_id"])).first()
                )
                s2_images.append(
                    s2_collection.filter(ee.Filter.eq("id_pref", pair["s2_id"])).first()
                )

            # Cache coverage results
            sequence_dates = tuple(sorted([p['s2_date'] for p in sequence]))
            dates_str = "_".join(sequence_dates)
            cache_hash = hashlib.md5(dates_str.encode()).hexdigest()[:16]
            coverage_cache_file = sta_cache_dir / f"coverage_{cache_hash}.pkl"

            if coverage_cache_file.exists():
                with open(coverage_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    valid_tiles = cached_data['valid_tiles']
                    total_checked = cached_data.get('total_checked', len(valid_tiles))
                print(f"   Loaded {len(valid_tiles)}/{total_checked} valid tiles from cache")
            else:
                # Generate grid and check coverage
                print(f"   Generating tile grid...")
                all_tiles = generate_tile_grid(region_config["aoi"], tile_size_deg=0.007)

                print(f"   Batched coverage check...")
                valid_tiles = []

                try:
                    tile_features = []
                    for idx, tile_bounds in enumerate(all_tiles):
                        tile_geom = ee.Geometry.Rectangle(tile_bounds)
                        tile_features.append(ee.Feature(tile_geom, {
                            'tile_idx': idx,
                            'bounds': tile_bounds
                        }))
                    tile_fc = ee.FeatureCollection(tile_features)

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

                    for idx, tile_bounds in enumerate(all_tiles):
                        coverages = tile_coverage[idx]
                        if all(cov >= args.coverage_threshold for cov in coverages.values()):
                            valid_tiles.append(tile_bounds)

                    print(f"   Valid tiles: {len(valid_tiles)}/{len(all_tiles)}")

                except Exception as e:
                    print(f"   Batched coverage failed: {e}")
                    print(f"   Falling back to per-tile checking...")

                    def check_tile_all_frames(tile_bounds):
                        tile_geom = ee.Geometry.Rectangle(tile_bounds)
                        for planet_img, s2_img in zip(planet_images, s2_images):
                            if not has_sufficient_coverage(planet_img, tile_geom, 'B1',
                                                          args.coverage_threshold):
                                return False
                            if not has_sufficient_coverage(s2_img, tile_geom, 'B2',
                                                          args.coverage_threshold):
                                return False
                        return True

                    with ThreadPoolExecutor(max_workers=16) as executor:
                        futures = {executor.submit(check_tile_all_frames, tb): tb for tb in all_tiles}
                        for future in as_completed(futures):
                            if future.result():
                                valid_tiles.append(futures[future])

                    print(f"   Valid tiles (fallback): {len(valid_tiles)}/{len(all_tiles)}")

                # Cache
                with open(coverage_cache_file, 'wb') as f:
                    pickle.dump({
                        'valid_tiles': valid_tiles,
                        'total_checked': len(all_tiles),
                        'sequence_dates': sequence_dates,
                        'coverage_threshold': args.coverage_threshold
                    }, f)

            if not valid_tiles:
                print(f"   No valid tiles, skipping sequence")
                continue

            remaining_samples = (args.max_samples - total_samples - len(sta_samples)) \
                if args.max_samples else len(valid_tiles)
            max_samples_this_seq = min(len(valid_tiles), remaining_samples)

            print(f"   Processing {max_samples_this_seq} tiles...")

            sequence_info = {
                'sequence': sequence,
                'valid_tiles': valid_tiles,
                'planet_collection': planet_collection,
                's2_collection': s2_collection,
                'planet_bands': planet_bands,
                'sentinel_bands': sentinel_bands,
                'date_band_config': date_band_config,
                'zenith_lookup': zenith_lookup,
                'cache_dir': sta_cache_dir,
                'region_name': sta_name,
                'seq_idx': seq_idx,
                'max_samples_per_seq': max_samples_this_seq,
                'tile_workers': args.tile_workers,
            }

            samples = process_single_sequence(sequence_info)

            # Save sequence checkpoint
            try:
                with open(seq_ckpt_file, 'wb') as f:
                    pickle.dump(samples, f, protocol=4)
            except Exception as e:
                print(f"   Warning: failed to save sequence checkpoint: {e}")

            sta_samples.extend(samples)
            print(f"   {sta_name} samples so far: {len(sta_samples)}")

            if args.max_samples and total_samples + len(sta_samples) >= args.max_samples:
                break

        # Normalize
        if args.normalize and sta_samples:
            print(f"\n   Normalizing {len(sta_samples)} samples to [0, 1]...")
            for sample in sta_samples:
                normalize_sample(sample)

        # Save per-STA dataset
        if sta_samples:
            dataset = {
                'samples': sta_samples,
                'config': {
                    'region': sta_name,
                    'year': args.year,
                    'sequence_length': args.sequence_length,
                    'num_samples': len(sta_samples),
                    'coverage_threshold': args.coverage_threshold,
                    'grid_based': True,
                    'normalized': args.normalize,
                    'scale_factor': 10000.0 if args.normalize else None,
                    'planet_collections': [PLANET_COLLECTION_LATEST, PLANET_COLLECTION_ADDITIONAL],
                    'inference_only': True,
                    's2_cloud_pct_max': args.s2_cloud_pct_max,
                    's2_aoi_cloud_frac_max': args.s2_aoi_cloud_frac_max,
                    'planet_q_cloud_max': args.planet_q_cloud_max,
                }
            }

            dataset_path = sta_output_dir / 'dataset.pkl'
            print(f"\n   Saving {len(sta_samples)} samples to {dataset_path}")
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset, f, protocol=4)

            total_samples += len(sta_samples)
        else:
            print(f"\n   No samples generated for {sta_name}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"INFERENCE DATA DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"  Total samples:  {total_samples}")
    print(f"  Time elapsed:   {elapsed / 60:.1f} minutes")
    print(f"  Output:         {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
