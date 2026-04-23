"""
AuthGEESTAs34.py - Specialized GEE authentication and pair-finding for STA-3/4

This module handles the large AOI + dense collection issue for STA-3/4 by:
1. Using LIGHTWEIGHT pair-finding (no reduceRegion operations)
2. Deferring adaptive 8-band/4-band mosaicking to TILE DOWNLOAD TIME

The key insight: 
- planet_best_per_day_adaptive runs 2 reduceRegion per date over the WHOLE AOI
- For STA-3/4's large AOI + many dates, this causes GEE rate limits
- But at tile scale (768m x 768m), these operations are fast and rarely needed
- Final data quality is IDENTICAL - we just make the mosaic decision per-tile

Strategy:
1. Pair-finding: Use simple _best_per_day (picks single best image per date)
2. Return ALL images for each date (not just the mosaic) in the collection
3. Tile download: Apply adaptive mosaic logic for each small tile
4. Keep the EXACT same filtering logic as AuthGEE.py

Usage:
    from Auth.AuthGEESTAs34 import find_pairs_sta34
    
    # Instead of find_pairs(), use:
    pair_results = find_pairs_sta34(aoi, start_date, end_date, ...)
    
    # The planet_collection_full contains ALL filtered images (for tile-level mosaicking)
    # The planet_collection contains best-per-day (for pair IDs)
"""

import ee
import re
from datetime import datetime as _dt
from datetime import timedelta
from typing import Dict, List, Tuple
import time as _time_module

# Import everything from AuthGEE to keep same filtering
from Auth.AuthGEE import (
    credentials, ee,
    PLANET_COLLECTION_ID, PLANET_COLLECTION_ID_FALLBACK, PLANET_COLLECTION_ID_ADDITIONAL,
    S2_COLLECTION_ID, regions,
    _num_prop_or_default, _add_pref_id, _add_common_fields,
    _add_s2_quality, _add_planet_quality, _add_s2_score, _add_planet_score,
    _best_per_day, make_add_udm_mask_and_score_for_aoi,
    planet_best_per_day_adaptive, make_add_aoi_cloud_fraction,
    make_tag_has_data, _match_dates_with_tolerance_nn,
    harmonize_to_4bands  # For tile-level adaptive mosaicking
)


def get_tile_adaptive_mosaic(planet_collection_full: ee.ImageCollection, 
                             date_key: str, 
                             tile_geom: ee.Geometry) -> ee.Image:
    """
    Get adaptive mosaic for a specific tile and date.
    
    This implements the SAME logic as planet_best_per_day_adaptive but for a small tile,
    making the reduceRegion operations fast and lightweight.
    
    The logic:
    1. Filter collection to this date
    2. Check if 8-band images cover the tile well (95%+)
    3. If yes: use 8-band quality mosaic
    4. If no: harmonize all to 4-band and mosaic
    
    Args:
        planet_collection_full: Full filtered Planet collection (all images, not just best-per-day)
        date_key: Date string in 'YYYY-MM-DD' format
        tile_geom: ee.Geometry for the tile bounds
    
    Returns:
        ee.Image: Adaptive mosaic for this tile and date
    """
    # Filter to this date
    day_col = planet_collection_full.filter(ee.Filter.eq('date_key', date_key))
    
    # Tag each image with its band type
    def tag_band_type(img):
        is_8band = img.bandNames().contains('B5')
        return img.set('band_type', ee.Algorithms.If(is_8band, '8band', '4band'))
    
    day_col = day_col.map(tag_band_type)
    
    # Split by band type
    day_4band = day_col.filter(ee.Filter.eq('band_type', '4band'))
    day_8band = day_col.filter(ee.Filter.eq('band_type', '8band'))
    
    count_4 = day_4band.size()
    count_8 = day_8band.size()
    
    # Check 8-band coverage for this TILE (fast since tile is small!)
    def get_8band_coverage():
        mosaic_8 = day_8band.qualityMosaic('qscore')
        valid_count = mosaic_8.select('qscore').reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=tile_geom,
            scale=10,  # Smaller scale OK for tile
            maxPixels=1e6,
            tileScale=1
        ).get('qscore')
        
        total_pixels = ee.Image.constant(1).clip(tile_geom).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=tile_geom,
            scale=10,
            maxPixels=1e6,
            tileScale=1
        ).get('constant')
        
        coverage_frac = ee.Number(valid_count).divide(ee.Number(total_pixels))
        return coverage_frac.gte(0.95)
    
    # Build mosaic options
    def make_8band_mosaic():
        return day_8band.qualityMosaic('qscore').set('strategy', '8band_only')
    
    def make_mixed_mosaic():
        return day_8band.map(harmonize_to_4bands).merge(day_4band).qualityMosaic('qscore').set('strategy', 'mixed_4band')
    
    def make_4band_mosaic():
        return day_4band.qualityMosaic('qscore').set('strategy', '4band_only')
    
    # Decision tree
    result = ee.Algorithms.If(
        count_8.gt(0),
        ee.Algorithms.If(
            get_8band_coverage(),
            make_8band_mosaic(),
            ee.Algorithms.If(
                count_4.gt(0),
                make_mixed_mosaic(),
                day_8band.qualityMosaic('qscore').set('strategy', '8band_only_partial')
            )
        ),
        make_4band_mosaic()
    )
    
    return ee.Image(result)


def _split_date_range_into_chunks(start_date: str, end_date: str, chunk_months: int = 6) -> List[Tuple[str, str]]:
    """
    Split a date range into chunks of specified months.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        chunk_months: Number of months per chunk (default 6)
    
    Returns:
        List of (chunk_start, chunk_end) tuples
    """
    start = _dt.strptime(start_date, "%Y-%m-%d")
    end = _dt.strptime(end_date, "%Y-%m-%d")
    
    chunks = []
    current = start
    
    while current < end:
        # Calculate chunk end (add chunk_months months)
        chunk_end_month = current.month + chunk_months
        chunk_end_year = current.year + (chunk_end_month - 1) // 12
        chunk_end_month = ((chunk_end_month - 1) % 12) + 1
        
        chunk_end = _dt(chunk_end_year, chunk_end_month, 1)
        
        # Don't go past the overall end date
        if chunk_end > end:
            chunk_end = end
        
        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d")
        ))
        
        current = chunk_end
    
    return chunks


def _getInfo_with_retry(ee_obj, max_retries=8, initial_delay=15, label="data"):
    """Get info from EE object with exponential backoff on rate limiting."""
    print(f"      Fetching {label}...", end=" ", flush=True)
    for attempt in range(max_retries):
        try:
            result = ee_obj.getInfo()
            print(f"✓", flush=True)
            return result
        except Exception as e:
            if "Too many concurrent aggregations" in str(e) and attempt < max_retries - 1:
                delay = min(initial_delay * (2 ** attempt), 300)  # Cap at 5 minutes
                print(f"\n   ⏳ GEE rate limit hit, waiting {delay}s (attempt {attempt + 1}/{max_retries})...", flush=True)
                _time_module.sleep(delay)
            else:
                print(f"✗", flush=True)
                raise


def find_pairs_for_chunk(aoi, start_date: str, end_date: str, 
                         planet_coll_id: str, planet_coll_id_fallback: str = None,
                         planet_coll_id_additional: str = None,
                         chunk_label: str = "") -> Dict:
    """
    Find pairs for a single date chunk using LIGHTWEIGHT approach.
    
    KEY CHANGE: Uses _best_per_day instead of planet_best_per_day_adaptive
    - NO reduceRegion operations during pair-finding
    - Returns both best-per-day collection (for pair IDs) AND full filtered collection
    - Adaptive mosaicking happens at tile download time
    
    Returns dict with dates, IDs, scores for both S2 and Planet.
    """
    print(f"\n   📅 Processing chunk: {chunk_label} ({start_date} to {end_date})")
    
    # S2 processing (same as find_pairs)
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
        .map(make_add_aoi_cloud_fraction(aoi))
        .filter(ee.Filter.notNull(['AOI_CLOUD_FRAC']))
        .filter(ee.Filter.lt('AOI_CLOUD_FRAC', 0.1))
        .map(_add_common_fields)
        .map(_add_s2_quality)
        .map(_add_s2_score)
        .map(_add_pref_id))
    s2_best = _best_per_day(s2)
    
    # Fetch S2 data first
    _time_module.sleep(2)
    s2_dates = _getInfo_with_retry(s2_best.aggregate_array('date_key'), label=f"S2 dates ({chunk_label})")
    s2_ids = _getInfo_with_retry(s2_best.aggregate_array('id_pref'), label=f"S2 IDs ({chunk_label})")
    
    # Planet processing - LIGHTWEIGHT approach (no reduceRegion!)
    def prep_planet_collection(coll_id, conf_threshold=60):
        """Prepare Planet collection with same filtering as AuthGEE but NO reduceRegion."""
        return (ee.ImageCollection(coll_id)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .map(_add_common_fields)
            .map(_add_planet_quality)
            .map(_add_planet_score)
            .filter(ee.Filter.lt('q_cloud', 20))
            .map(make_add_udm_mask_and_score_for_aoi(aoi, conf_threshold=conf_threshold))
            .map(_add_pref_id))
    
    def process_single_planet_repo(coll_id, repo_name):
        """Process a single Planet repo using LIGHTWEIGHT _best_per_day (no reduceRegion)."""
        try:
            pl_prepped = prep_planet_collection(coll_id, conf_threshold=60)
            size = pl_prepped.size().getInfo()
            if size == 0:
                print(f"      {repo_name}: 0 images", flush=True)
                return [], [], [], None, None
            
            print(f"      {repo_name}: {size} images, using lightweight best-per-day...", flush=True)
            
            # KEY CHANGE: Use simple _best_per_day instead of planet_best_per_day_adaptive
            # This avoids ALL reduceRegion operations during pair-finding!
            pl_best = _best_per_day(pl_prepped)
            
            # Fetch with retries - now this is FAST (no reduceRegion)
            _time_module.sleep(1)
            dates = _getInfo_with_retry(pl_best.aggregate_array('date_key'), label=f"{repo_name} dates")
            _time_module.sleep(1)
            ids = _getInfo_with_retry(pl_best.aggregate_array('id_pref'), label=f"{repo_name} IDs")
            _time_module.sleep(1)
            scores = _getInfo_with_retry(pl_best.aggregate_array('score'), label=f"{repo_name} scores")
            
            print(f"      ✓ {repo_name}: {len(dates)} dates processed (lightweight)", flush=True)
            
            # Return BOTH best-per-day (for pair IDs) AND full filtered collection (for tile mosaicking)
            return dates, ids, scores, pl_best, pl_prepped
            
        except Exception as e:
            print(f"      {repo_name}: Failed - {e}", flush=True)
            return [], [], [], None, None
    
    # Process each repo
    print(f"   🔄 Processing Planet repos (LIGHTWEIGHT - no reduceRegion)", flush=True)
    
    all_pl_dates = []
    all_pl_ids = []
    all_pl_scores = []
    all_pl_best_collections = []  # Best-per-day for pair matching
    all_pl_full_collections = []  # Full filtered for tile-level mosaicking
    
    for coll_id, repo_name in [(planet_coll_id, "primary"), 
                               (planet_coll_id_fallback, "secondary"),
                               (planet_coll_id_additional, "additional")]:
        if coll_id:
            dates, ids, scores, best_coll, full_coll = process_single_planet_repo(coll_id, repo_name)
            if dates:
                all_pl_dates.extend(dates)
                all_pl_ids.extend(ids)
                all_pl_scores.extend(scores if scores else [0] * len(dates))
                if best_coll:
                    all_pl_best_collections.append(best_coll)
                if full_coll:
                    all_pl_full_collections.append(full_coll)
            # Brief pause between repos
            _time_module.sleep(2)
    
    if not all_pl_dates:
        print(f"      ⚠️ No Planet data for this chunk", flush=True)
        return {
            's2_dates': s2_dates, 's2_ids': s2_ids,
            'pl_dates': [], 'pl_ids': [], 'pl_scores': [],
            's2_collection': s2_best, 'planet_collection': None,
            'planet_collection_full': None
        }
    
    # CLIENT-SIDE MERGE: Keep best image per date across repos
    date_to_best = {}
    for date, img_id, score in zip(all_pl_dates, all_pl_ids, all_pl_scores):
        score_val = score if score is not None else float('inf')
        if date not in date_to_best or score_val < date_to_best[date][1]:
            date_to_best[date] = (img_id, score_val)
    
    pl_dates = sorted(date_to_best.keys())
    pl_ids = [date_to_best[d][0] for d in pl_dates]
    pl_scores = [date_to_best[d][1] for d in pl_dates]
    
    # Merge best-per-day collections (for pair IDs)
    if len(all_pl_best_collections) == 1:
        pl_best = all_pl_best_collections[0]
    elif len(all_pl_best_collections) > 1:
        pl_best = all_pl_best_collections[0]
        for coll in all_pl_best_collections[1:]:
            pl_best = pl_best.merge(coll)
    else:
        pl_best = None
    
    # Merge FULL filtered collections (for tile-level adaptive mosaicking)
    if len(all_pl_full_collections) == 1:
        pl_full = all_pl_full_collections[0]
    elif len(all_pl_full_collections) > 1:
        pl_full = all_pl_full_collections[0]
        for coll in all_pl_full_collections[1:]:
            pl_full = pl_full.merge(coll)
    else:
        pl_full = None
    
    print(f"      ✓ Chunk complete: {len(s2_dates)} S2 dates, {len(pl_dates)} Planet dates", flush=True)
    
    return {
        's2_dates': s2_dates,
        's2_ids': s2_ids,
        'pl_dates': pl_dates,
        'pl_ids': pl_ids,
        'pl_scores': pl_scores,
        's2_collection': s2_best,
        'planet_collection': pl_best,
        'planet_collection_full': pl_full  # NEW: Full collection for tile-level mosaicking
    }


def find_pairs_sta34(aoi, start_date: str, end_date: str,
                     planet_coll_id: str, planet_coll_id_fallback: str = None,
                     planet_coll_id_additional: str = None,
                     chunk_months: int = 3) -> Dict:
    """
    Find image pairs for STA-3/4 using LIGHTWEIGHT pair-finding.
    
    This is a drop-in replacement for find_pairs() that:
    1. Uses _best_per_day instead of planet_best_per_day_adaptive (NO reduceRegion!)
    2. Returns planet_collection_full for tile-level adaptive mosaicking
    3. Same filtering logic - adaptive quality selection happens at tile download time
    
    Args:
        aoi: Earth Engine geometry for area of interest
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        planet_coll_id: Primary Planet collection ID
        planet_coll_id_fallback: Secondary/fallback Planet collection ID
        planet_coll_id_additional: Additional Planet collection ID
        chunk_months: Number of months per processing chunk (default 3)
    
    Returns:
        Dictionary with same format as find_pairs() PLUS:
        - planet_collection_full: All filtered Planet images (for tile-level mosaicking)
    """
    print(f"\n🔄 STA-3/4 LIGHTWEIGHT PAIR FINDING")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Strategy: _best_per_day for pairs, adaptive mosaicking at tile download")
    
    # Split into chunks
    chunks = _split_date_range_into_chunks(start_date, end_date, chunk_months)
    print(f"   Processing in {len(chunks)} chunks of {chunk_months} months each")
    
    # Process each chunk
    all_s2_dates = []
    all_s2_ids = []
    all_pl_dates = []
    all_pl_ids = []
    all_pl_scores = []
    chunk_collections = {'s2': [], 'planet': [], 'planet_full': []}
    
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_label = f"Chunk {i+1}/{len(chunks)}"
        
        result = find_pairs_for_chunk(
            aoi, chunk_start, chunk_end,
            planet_coll_id, planet_coll_id_fallback, planet_coll_id_additional,
            chunk_label=chunk_label
        )
        
        all_s2_dates.extend(result['s2_dates'])
        all_s2_ids.extend(result['s2_ids'])
        all_pl_dates.extend(result['pl_dates'])
        all_pl_ids.extend(result['pl_ids'])
        all_pl_scores.extend(result['pl_scores'])
        
        if result['s2_collection']:
            chunk_collections['s2'].append(result['s2_collection'])
        if result['planet_collection']:
            chunk_collections['planet'].append(result['planet_collection'])
        if result.get('planet_collection_full'):
            chunk_collections['planet_full'].append(result['planet_collection_full'])
        
        # Brief pause between chunks
        if i < len(chunks) - 1:
            print(f"\n   ⏸️  Pausing 3s between chunks...")
            _time_module.sleep(3)
    
    # Remove duplicates (same date might appear in overlapping processing)
    # For Planet: keep best score per date
    print(f"\n   🔗 Merging chunk results...")
    
    s2_map = {}
    for date, img_id in zip(all_s2_dates, all_s2_ids):
        if date not in s2_map:
            s2_map[date] = img_id
    
    pl_map = {}
    pl_score_map = {}
    for date, img_id, score in zip(all_pl_dates, all_pl_ids, all_pl_scores):
        if date not in pl_map or (score is not None and score < pl_score_map.get(date, float('inf'))):
            pl_map[date] = img_id
            pl_score_map[date] = score if score is not None else float('inf')
    
    # Final unique lists
    s2_dates = sorted(s2_map.keys())
    s2_ids = [s2_map[d] for d in s2_dates]
    pl_dates = sorted(pl_map.keys())
    pl_ids = [pl_map[d] for d in pl_dates]
    
    print(f"   ✅ Merged: {len(s2_dates)} unique S2 dates, {len(pl_dates)} unique Planet dates")
    
    # Merge collections
    if chunk_collections['s2']:
        s2_best = chunk_collections['s2'][0]
        for coll in chunk_collections['s2'][1:]:
            s2_best = s2_best.merge(coll)
    else:
        s2_best = None
    
    if chunk_collections['planet']:
        pl_best = chunk_collections['planet'][0]
        for coll in chunk_collections['planet'][1:]:
            pl_best = pl_best.merge(coll)
    else:
        pl_best = None
    
    # Merge FULL Planet collections (for tile-level mosaicking)
    if chunk_collections['planet_full']:
        pl_full = chunk_collections['planet_full'][0]
        for coll in chunk_collections['planet_full'][1:]:
            pl_full = pl_full.merge(coll)
    else:
        pl_full = None
    
    # Match dates (same logic as find_pairs)
    s2_map_final = {d: i for d, i in zip(s2_dates, s2_ids)}
    pl_map_final = {d: i for d, i in zip(pl_dates, pl_ids)}
    
    matches, planet_only_dates, s2_only_dates = _match_dates_with_tolerance_nn(
        s2_map_final, pl_map_final, days_tol=7
    )
    
    matched = [{
        's2_date': s2_d,
        'planet_date': pl_d,
        's2_id': s2_map_final[s2_d],
        'planet_id': pl_map_final[pl_d]
    } for s2_d, pl_d in matches]
    
    planet_unmatched_ids = [pl_map_final[d] for d in planet_only_dates]
    s2_unmatched_ids = [s2_map_final[d] for d in s2_only_dates]
    
    rows = max(len(planet_only_dates), len(s2_only_dates))
    unmatched_table = []
    for i in range(rows):
        unmatched_table.append({
            'planet_date': planet_only_dates[i] if i < len(planet_only_dates) else '',
            's2_date': s2_only_dates[i] if i < len(s2_only_dates) else ''
        })
    
    print(f"   ✅ Final: {len(matched)} matched pairs")
    
    return {
        'matched_pairs': matched,
        'planet_unmatched': planet_unmatched_ids,
        's2_unmatched': s2_unmatched_ids,
        'unmatched_table': unmatched_table,
        'planet_unmatched_dates': planet_only_dates,
        's2_unmatched_dates': s2_only_dates,
        's2_collection': s2_best,
        'planet_collection': pl_best,
        'planet_collection_full': pl_full,  # NEW: All filtered images for tile-level mosaicking
        's2_all_dates': s2_dates,
        's2_all_ids': s2_ids,
        'planet_all_dates': pl_dates,
        'planet_all_ids': pl_ids
    }


# Convenience alias
find_pairs = find_pairs_sta34
