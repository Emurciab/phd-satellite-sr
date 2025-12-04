#!/usr/bin/env python3
"""
Estimate processing time for all STAs based on:
- AOI size and tile grid
- Number of matched pairs per year
- Average valid tile ratio (~29% based on STA-2 test)
- Optimized pipeline: ~20s per sample
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import ee
from pathlib import Path

# Initialize GEE
print("Initializing GEE...")
service_account = 'remotesensingeduart@remotesensingeduart.iam.gserviceaccount.com'
credentials_path = Path(__file__).parent.parent.parent / 'Auth' / 'remotesensingeduart-222d07dd6689.json'
credentials = ee.ServiceAccountCredentials(service_account, str(credentials_path))
ee.Initialize(credentials)

from Auth.AuthGEE import regions, find_pairs, PLANET_COLLECTION_ID

# Constants based on our testing
TILE_SIZE_DEG = 0.007  # ~256px at 3m
TIME_PER_SAMPLE_SEC = 20  # With parallel downloads
SEQUENCE_LENGTH = 3
VALID_TILE_RATIO = 0.29  # ~29% tiles pass coverage check (from STA-2 test)


def estimate_grid_tiles(aoi):
    """Calculate grid dimensions for an AOI."""
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    min_lon = min(p[0] for p in bounds)
    max_lon = max(p[0] for p in bounds)
    min_lat = min(p[1] for p in bounds)
    max_lat = max(p[1] for p in bounds)
    
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    
    cols = int(lon_range / TILE_SIZE_DEG) + 1
    rows = int(lat_range / TILE_SIZE_DEG) + 1
    
    return {
        'rows': rows,
        'cols': cols,
        'total_tiles': rows * cols,
        'lon_range': lon_range,
        'lat_range': lat_range,
        'area_deg2': lon_range * lat_range
    }


def count_matched_pairs(aoi, start_date, end_date, planet_coll_id):
    """Count matched pairs for a date range."""
    try:
        result = find_pairs(aoi, start_date, end_date, planet_coll_id)
        return len(result['matched_pairs'])
    except Exception as e:
        print(f"      Error: {e}")
        return 0


def format_time(seconds):
    """Format seconds to human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        hours = seconds / 3600
        if hours < 24:
            return f"{hours:.1f}h"
        else:
            return f"{hours/24:.1f}d"


def main():
    print("\n" + "="*80)
    print("📊 PROCESSING TIME ESTIMATION FOR ALL STAs")
    print("="*80)
    print(f"\nParameters:")
    print(f"  - Tile size: {TILE_SIZE_DEG}° (~256px @ 3m)")
    print(f"  - Time per sample: {TIME_PER_SAMPLE_SEC}s (with parallel downloads)")
    print(f"  - Sequence length: {SEQUENCE_LENGTH} frames")
    print(f"  - Expected valid tile ratio: {VALID_TILE_RATIO*100:.0f}%")
    
    # Group regions by name
    sta_groups = {}
    for reg in regions:
        name = reg['name']
        if name not in sta_groups:
            sta_groups[name] = []
        sta_groups[name].append(reg)
    
    print(f"\nFound {len(sta_groups)} unique STAs: {list(sta_groups.keys())}")
    
    # Calculate for each STA
    results = []
    total_samples = 0
    total_time = 0
    
    for sta_name, sta_regions in sta_groups.items():
        print(f"\n{'='*80}")
        print(f"📍 {sta_name}")
        print(f"{'='*80}")
        
        # Get AOI info from first region (same for all years)
        aoi = sta_regions[0]['aoi']
        planet_coll = sta_regions[0].get('planet_collection_id', PLANET_COLLECTION_ID)
        
        grid = estimate_grid_tiles(aoi)
        print(f"\n   AOI Size: {grid['lon_range']:.4f}° × {grid['lat_range']:.4f}° = {grid['area_deg2']:.4f} deg²")
        print(f"   Grid: {grid['rows']} rows × {grid['cols']} cols = {grid['total_tiles']} tiles")
        print(f"   Est. valid tiles: ~{int(grid['total_tiles'] * VALID_TILE_RATIO)} ({VALID_TILE_RATIO*100:.0f}%)")
        
        sta_total_pairs = 0
        sta_total_sequences = 0
        year_details = []
        
        print(f"\n   📅 Year-by-year matched pairs:")
        for reg in sta_regions:
            year_start = reg['start'][:4]
            print(f"      {year_start}: ", end="", flush=True)
            
            pairs = count_matched_pairs(aoi, reg['start'], reg['end'], planet_coll)
            sequences = max(0, pairs - SEQUENCE_LENGTH + 1)  # Sliding window
            
            print(f"{pairs} pairs → {sequences} sequences")
            sta_total_pairs += pairs
            sta_total_sequences += sequences
            year_details.append({'year': year_start, 'pairs': pairs, 'sequences': sequences})
        
        # Calculate samples and time
        valid_tiles = int(grid['total_tiles'] * VALID_TILE_RATIO)
        samples_per_sequence = valid_tiles  # Each sequence can generate 1 sample per valid tile
        total_sta_samples = sta_total_sequences * samples_per_sequence
        sta_time_sec = total_sta_samples * TIME_PER_SAMPLE_SEC
        
        print(f"\n   📊 Estimates:")
        print(f"      Total sequences (5 years): {sta_total_sequences}")
        print(f"      Samples per sequence: ~{valid_tiles} (valid tiles)")
        print(f"      Total samples: {sta_total_sequences} × {valid_tiles} = {total_sta_samples:,}")
        print(f"      Processing time: {total_sta_samples} × {TIME_PER_SAMPLE_SEC}s = {format_time(sta_time_sec)}")
        
        results.append({
            'name': sta_name,
            'grid': grid,
            'years': len(sta_regions),
            'total_pairs': sta_total_pairs,
            'total_sequences': sta_total_sequences,
            'valid_tiles': valid_tiles,
            'total_samples': total_sta_samples,
            'time_sec': sta_time_sec
        })
        
        total_samples += total_sta_samples
        total_time += sta_time_sec
    
    # Summary table
    print("\n" + "="*80)
    print("📊 SUMMARY TABLE")
    print("="*80)
    print(f"\n{'STA':<12} {'Grid':<12} {'Sequences':<10} {'Valid Tiles':<12} {'Samples':<12} {'Time':<10}")
    print("-"*80)
    
    for r in results:
        grid_str = f"{r['grid']['rows']}×{r['grid']['cols']}"
        print(f"{r['name']:<12} {grid_str:<12} {r['total_sequences']:<10} {r['valid_tiles']:<12} {r['total_samples']:<12,} {format_time(r['time_sec']):<10}")
    
    print("-"*80)
    print(f"{'TOTAL':<12} {'':<12} {'':<10} {'':<12} {total_samples:<12,} {format_time(total_time):<10}")
    
    # Time breakdown
    print("\n" + "="*80)
    print("⏱️  TIME ESTIMATES")
    print("="*80)
    print(f"\n   Total samples: {total_samples:,}")
    print(f"   Time at {TIME_PER_SAMPLE_SEC}s/sample: {format_time(total_time)}")
    print(f"\n   With 24/7 processing:")
    print(f"      Days needed: {total_time/86400:.1f}")
    
    print(f"\n   Parallelization options:")
    print(f"      2 workers: {format_time(total_time/2)}")
    print(f"      4 workers: {format_time(total_time/4)}")
    print(f"      8 workers: {format_time(total_time/8)}")
    
    return results


if __name__ == "__main__":
    main()
