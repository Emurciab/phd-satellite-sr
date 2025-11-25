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


@dataclass
class TrainingSample:
    """Training sample with temporal sequences"""
    lr_sequence: np.ndarray  # (T, H, W, C)
    hr_sequence: np.ndarray  # (T, H, W, C)
    temporal_dates: List[str]  # Sentinel-2 dates
    planet_dates: List[str]  # Planet dates (may differ by ±7 days)
    region_name: str
    sequence_id: int
    tile_id: int
    tile_bounds: Dict


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
            - planet_bands: List of Planet band names
            - sentinel_bands: List of Sentinel-2 band names
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
    planet_bands = tile_info['planet_bands']
    sentinel_bands = tile_info['sentinel_bands']
    cache_dir = tile_info['cache_dir']
    region_name = tile_info['region_name']
    seq_idx = tile_info['seq_idx']
    
    lr_frames = []
    hr_frames = []
    dates = []
    
    # Download all frames for this tile
    all_frames_ok = True
    for frame_idx, pair in enumerate(sequence):
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
                all_frames_ok = False
                break
            
            # Apply coregistration and harmonization
            cache_path_coreg = cache_dir / "coregistered" / f"seq{seq_idx}_tile{tile_idx}_frame{frame_idx}.npy"
            cache_path_harm = cache_dir / "harmonized" / f"seq{seq_idx}_tile{tile_idx}_frame{frame_idx}.npy"
            
            # Convert tile_bounds [minLon, minLat, maxLon, maxLat] to aoi_bounds format
            # AROSICS expects list of [lon, lat] coordinate pairs
            aoi_bounds_coords = [
                [tile_bounds[0], tile_bounds[1]],  # bottom-left
                [tile_bounds[2], tile_bounds[1]],  # bottom-right
                [tile_bounds[2], tile_bounds[3]],  # top-right
                [tile_bounds[0], tile_bounds[3]]   # top-left
            ]
            
            # Coregistration with REAL geographic coordinates
            hr_array = coregister_lr_to_hr_wrapper(
                hr_array=hr_array,
                lr_array=lr_array,
                aoi_bounds=aoi_bounds_coords,
                cache_path=cache_path_coreg
            )
            
            # Harmonization
            hr_array = harmonize_planet_to_sentinel2_wrapper(
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
            dates.append((pair['s2_date'], pair['planet_date']))
            
        except Exception as e:
            all_frames_ok = False
            break
    
    # Create sample if all frames downloaded
    if all_frames_ok and len(lr_frames) == len(sequence):
        # Separate S2 and Planet dates
        s2_dates = [d[0] for d in dates]
        planet_dates = [d[1] for d in dates]
        
        sample = TrainingSample(
            lr_sequence=np.stack(lr_frames, axis=0),
            hr_sequence=np.stack(hr_frames, axis=0),
            temporal_dates=s2_dates,
            planet_dates=planet_dates,
            region_name=region_name,
            sequence_id=seq_idx,
            tile_id=tile_idx,
            tile_bounds={'bounds': tile_bounds}
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
    cache_dir = sequence_info['cache_dir']
    region_name = sequence_info['region_name']
    seq_idx = sequence_info['seq_idx']
    max_samples_per_seq = sequence_info['max_samples_per_seq']
    tile_workers = sequence_info.get('tile_workers', 1)
    
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
            'cache_dir': cache_dir,
            'region_name': region_name,
            'seq_idx': seq_idx
        }
        tile_infos.append(tile_info)
    
    # Process tiles in parallel
    if tile_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=tile_workers) as executor:
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
    parser.add_argument('--max-samples', type=int, default=20,
                       help='Maximum number of samples to generate')
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
    parser.add_argument('--tile-workers', type=int, default=1,
                       help='Number of parallel workers for tile processing within each sequence')
    parser.add_argument('--coverage-workers', type=int, default=8,
                       help='Number of parallel workers for coverage checking')
    
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
    
    # Calculate how many sequences we need
    num_sequences = max(1, (args.max_samples + 9) // 10)  # ~10 samples per sequence
    num_sequences = min(num_sequences, len(matched_pairs) - args.sequence_length + 1)
    
    print(f"\n📅 Creating {num_sequences} sequences of {args.sequence_length} frames each")
    
    # Band configuration (use 4-band for simplicity)
    planet_bands = ['B1', 'B2', 'B3', 'B4']
    sentinel_bands = ['B2', 'B3', 'B4', 'B8A']
    
    all_samples = []
    
    # Process sequences
    for seq_idx in range(num_sequences):
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
            
            # Generate grid tiles
            print(f"\n🗺️  Generating tile grid...")
            all_tiles = generate_tile_grid(region_config["aoi"], tile_size_deg=0.007, max_tiles=200)
            
            # Filter tiles to keep only those with coverage in ALL frames (intersection)
            print(f"\n🔍 Filtering tiles with coverage in ALL {len(sequence)} frames (intersection)...")
            print(f"   This ensures temporal consistency - all frames have data at each tile")
            
            valid_tiles = []
            checked = 0
            
            with ThreadPoolExecutor(max_workers=args.coverage_workers) as executor:
                # For each tile, check if ALL frames have sufficient coverage
                def check_tile_all_frames(tile_bounds):
                    tile_geom = ee.Geometry.Rectangle(tile_bounds)
                    
                    # Check coverage on ALL frames
                    for planet_img, s2_img in zip(planet_images, s2_images):
                        # Check Planet
                        planet_valid = has_sufficient_coverage(
                            planet_img, tile_geom, 'B1',
                            min_percent=args.coverage_threshold, scale=30
                        )
                        if not planet_valid:
                            return False  # Reject tile if any frame has insufficient coverage
                        
                        # Check Sentinel-2
                        s2_valid = has_sufficient_coverage(
                            s2_img, tile_geom, 'B2',
                            min_percent=args.coverage_threshold, scale=30
                        )
                        if not s2_valid:
                            return False
                    
                    # All frames have sufficient coverage
                    return True
                
                # Submit all tasks
                future_to_tile = {
                    executor.submit(check_tile_all_frames, tile_bounds): tile_bounds
                    for tile_bounds in all_tiles
                }
                
                # Collect results
                for future in as_completed(future_to_tile):
                    tile_bounds = future_to_tile[future]
                    checked += 1
                    
                    try:
                        is_valid = future.result()
                        if is_valid:
                            valid_tiles.append(tile_bounds)
                        
                        if checked % 20 == 0 or checked == len(all_tiles):
                            print(f"   Progress: {checked}/{len(all_tiles)} checked, {len(valid_tiles)} valid ({100*len(valid_tiles)/checked:.1f}%)")
                    except Exception as e:
                        pass
            
            print(f"   ✓ Valid tiles (ALL frames): {len(valid_tiles)}/{len(all_tiles)} ({100*len(valid_tiles)/len(all_tiles):.1f}%)")
            
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
        
        # Prepare sequence info
        sequence_info = {
            'sequence': sequence,
            'valid_tiles': valid_tiles,
            'planet_collection': planet_collection,
            's2_collection': s2_collection,
            'planet_bands': planet_bands,
            'sentinel_bands': sentinel_bands,
            'cache_dir': cache_dir,
            'region_name': region_config['name'],
            'seq_idx': seq_idx,
            'max_samples_per_seq': max_samples_this_seq,
            'tile_workers': args.tile_workers
        }
        
        # Process this sequence
        samples = process_single_sequence(sequence_info)
        all_samples.extend(samples)
        
        print(f"\n   ✓ Sequence {seq_idx + 1} complete: {len(samples)} samples generated")
        print(f"   Total samples so far: {len(all_samples)}/{args.max_samples}")
        
        if len(all_samples) >= args.max_samples:
            break
    
    # Save dataset
    if len(all_samples) == 0:
        print("\n❌ No samples generated!")
        return
    
    dataset_path = output_dir / 'dataset.pkl'
    print(f"\n💾 Saving {len(all_samples)} samples...")
    
    dataset = {
        'samples': all_samples,
        'config': {
            'region': region_config['name'],
            'sequence_length': args.sequence_length,
            'num_samples': len(all_samples),
            'coverage_threshold': args.coverage_threshold,
            'grid_based': True
        }
    }
    
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=4)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✅ GRID-BASED PARALLEL GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Samples generated: {len(all_samples)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Avg time per sample: {elapsed/len(all_samples):.1f} seconds")
    print(f"\nDataset saved: {dataset_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
