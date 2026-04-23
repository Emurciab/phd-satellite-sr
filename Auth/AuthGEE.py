# from coreg.coreg import harmonize_images, coregister_images
import ee
import pandas as pd
import re
from datetime import datetime, timedelta
from datetime import datetime as _dt
from datetime import datetime, timedelta
from typing import Tuple, Dict, List

import os as _os
_auth_dir = _os.path.dirname(_os.path.abspath(__file__))
credentials = ee.ServiceAccountCredentials(
    'stas-50@remotesensingeduart.iam.gserviceaccount.com',
    _os.path.join(_auth_dir, 'remotesensingeduart-222d07dd6689.json')
)
ee.Initialize(credentials, project='remotesensingeduart')

# Planet imagery collections
PLANET_COLLECTION_ID = 'projects/remotesensingeduart/assets/testSTAs'
PLANET_COLLECTION_ID_FALLBACK = 'projects/ee-eeamurciabb/assets/testSTAs'
PLANET_COLLECTION_ID_ADDITIONAL = 'projects/stasadditionalfree/assets/testSTAs_ic'

# Sentinel-2 collection
S2_COLLECTION_ID = 'COPERNICUS/S2_SR_HARMONIZED'

# --------- Helpers (fixed) ---------
def _num_prop_or_default(img, prop, default_val):
    d = ee.Dictionary(img.toDictionary())
    return ee.Number(ee.Algorithms.If(d.contains(prop), d.get(prop), default_val))

def _add_pref_id(img):
    # Prefer system:id; fallback to system:index
    sid = ee.String(
        ee.Algorithms.If(img.get('system:id'), img.get('system:id'), img.get('system:index'))
    )
    return img.set({'id_pref': sid})

def _add_common_fields(img):
    d = ee.Date(img.get('system:time_start'))
    hour = ee.Number.parse(d.format('H'))
    minute = ee.Number.parse(d.format('m'))
    dist_noon = hour.add(minute.divide(60)).subtract(12).abs()  # 0..~12
    return img.set({
        'date_key': d.format('YYYY-MM-dd'),
        'dist_noon': dist_noon,
        'time_start': d.millis()
    })

def _add_s2_quality(img):
    # 0..100 (lower is better)
    q = _num_prop_or_default(img, 'CLOUDY_PIXEL_PERCENTAGE', 100)
    return img.set({'q_cloud': q})

def _add_planet_quality(img):
    # Prefer 'cloud_cover' (0..1) -> %, else 'cloud_percent' (0..100), else 100
    d = ee.Dictionary(img.toDictionary())
    q = ee.Number(
        ee.Algorithms.If(
            d.contains('cloud_cover'),
            ee.Number(img.get('cloud_cover')).multiply(100),
            ee.Algorithms.If(
                d.contains('cloud_percent'),
                ee.Number(img.get('cloud_percent')),
                100
            )
        )
    )
    # default sun_elevation if missing (keeps code robust)
    sun_el = _num_prop_or_default(img, 'sun_elevation', 45)
    return img.set({'q_cloud': q, 'sun_elevation_safe': sun_el})

# ---- composite scores so we can sort by one field ----
def _add_s2_score(img):
    q = ee.Number(img.get('q_cloud'))                 
    dist_noon = ee.Number(img.get('dist_noon'))       
    t = ee.Number(img.get('time_start')).divide(1e15)
    score = q.multiply(1000).add(dist_noon).add(t)
    return img.set({'score': score})

def _add_planet_score(img):
    q = ee.Number(img.get('q_cloud'))                  
    sun_el = ee.Number(img.get('sun_elevation_safe'))  
    dist_noon = ee.Number(img.get('dist_noon'))        
    score = (
        q.multiply(1000)
         .add(ee.Number(100).subtract(sun_el).divide(100))
         .add(dist_noon)
    )
    return img.set({'score': score})

def _best_per_day(col):
    dates = col.aggregate_array('date_key').distinct()
    def pick(d):
        d = ee.String(d)
        daycol = col.filter(ee.Filter.eq('date_key', d)).sort('score', True)
        return ee.Image(daycol.first()).set('date_key', d)
    return ee.ImageCollection(dates.map(pick))

def make_add_udm_mask_and_score_for_aoi(aoi, conf_threshold=80):
    def _fn(img):
        # UDM bands
        q1_clear = img.select('Q1')      # 0/1
        q2_snow  = img.select('Q2')      # 0/1
        q3_shadow= img.select('Q3')      # 0/1
        q4_lhaze = img.select('Q4')      # 0/1
        q5_hhaze = img.select('Q5')      # 0/1
        q6_cloud = img.select('Q6')      # 0/1
        q7_conf  = img.select('Q7')      # 0..100
        q8_unuse = img.select('Q8')      # 0/1

        # AOI mask (1 only inside AOI)
        aoi_mask = ee.Image.constant(1).clip(aoi).mask()

        # Base keep: clear & not unusable & confident enough
        base_keep = (
                   q1_clear.eq(1)
                   .And(q8_unuse.eq(0))
                   .And(q7_conf.gte(conf_threshold))
                    )           

        # Combined mask inside AOI
        valid_mask = base_keep.updateMask(aoi_mask)

        # Quality score (0..1): just normalized confidence of the classes we keep
        qscore = q7_conf.divide(100).rename('qscore').updateMask(valid_mask)

        # Apply same mask + clip to AOI
        img_masked = img.clip(aoi).updateMask(valid_mask)

        return img_masked.addBands(qscore)
    return _fn

def add_valid_area_props(img, aoi, scale=50, tileScale=2):
    m = img.select('qscore').mask()
    px   = m.reduceRegion(ee.Reducer.count(), aoi, scale,
                          maxPixels=1e9, tileScale=tileScale).get('qscore')
    area = ee.Image.pixelArea().updateMask(m) \
             .reduceRegion(ee.Reducer.sum(), aoi, scale,
                           maxPixels=1e9, tileScale=tileScale).get('area')
    frac = ee.Number(area).divide(aoi.area(scale))
    return img.set({'valid_px': px, 'valid_area_m2': area, 'valid_frac': frac})


def make_tag_has_data(aoi, scale=50, tileScale=2):
    def _fn(img):
        # count = number of unmasked qscore pixels (mask != 0)
        count = img.select('qscore').reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e8,
            tileScale=tileScale
        ).get('qscore')
        return img.set('has_data', count)
    return _fn


def planet_best_per_day(col):
    dates = col.aggregate_array('date_key').distinct()
    def _pick(d):
        d = ee.String(d)
        day_col = col.filter(ee.Filter.eq('date_key', d))
        # Get the first image to copy properties from
        first_img = ee.Image(day_col.first())
        # Create the mosaic
        mosaic = day_col.qualityMosaic('qscore')
        # Copy over important properties from the first image
        return mosaic.copyProperties(first_img, ['date_key', 'id_pref', 'time_start'])
    return ee.ImageCollection(dates.map(_pick))

def harmonize_to_4bands(img):
    """
    Harmonize Planet image to 4 bands (keeps only B1-B4 + UDM + qscore).
    """
    spectral_bands = ['B1', 'B2', 'B3', 'B4']
    udm_bands = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
    qscore_band = ['qscore']
    
    bands_to_keep = spectral_bands + udm_bands + qscore_band
    return img.select(bands_to_keep)

def planet_best_per_day_adaptive(col, aoi):
    """
    For each date:
    1. Check if 8-band images alone can cover AOI with good quality
    2. If yes: use only 8-band images (keep all 8 bands)
    3. If no: harmonize all to 4 bands and mosaic together
    """
    # Tag each image with its band count
    def tag_band_type(img):
        is_8band = img.bandNames().contains('B5')
        return img.set('band_type', ee.Algorithms.If(is_8band, '8band', '4band'))
    
    col_tagged = col.map(tag_band_type)
    
    # Split by band type
    col_4band = col_tagged.filter(ee.Filter.eq('band_type', '4band'))
    col_8band = col_tagged.filter(ee.Filter.eq('band_type', '8band'))
    
    dates = col_tagged.aggregate_array('date_key').distinct()
    
    def _pick(d):
        d = ee.String(d)
        
        # Get images for this date
        day_4band = col_4band.filter(ee.Filter.eq('date_key', d))
        day_8band = col_8band.filter(ee.Filter.eq('date_key', d))
        
        count_4 = day_4band.size()
        count_8 = day_8band.size()
        
        # Get first image to copy properties from
        first_img = ee.Image(col_tagged.filter(ee.Filter.eq('date_key', d)).first())
        
        # Helper: check if 8-band coverage is sufficient
        def get_8band_coverage():
            mosaic_8 = day_8band.qualityMosaic('qscore')
            valid_count = mosaic_8.select('qscore').reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=aoi,
                scale=50,
                maxPixels=1e8,
                tileScale=2
            ).get('qscore')
            
            total_pixels = ee.Image.constant(1).clip(aoi).reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=aoi,
                scale=50,
                maxPixels=1e8,
                tileScale=2
            ).get('constant')
            
            coverage_frac = ee.Number(valid_count).divide(ee.Number(total_pixels))
            return coverage_frac.gte(0.95)
        
        # Decision logic with proper ee.Image casting
        def make_8band_mosaic():
            return day_8band.qualityMosaic('qscore').set('strategy', '8band_only')
        
        def make_mixed_mosaic():
            return day_8band.map(harmonize_to_4bands).merge(day_4band).qualityMosaic('qscore').set('strategy', 'mixed_4band')
        
        def make_8band_partial_mosaic():
            return day_8band.qualityMosaic('qscore').set('strategy', '8band_only_partial')
        
        def make_4band_mosaic():
            return day_4band.qualityMosaic('qscore').set('strategy', '4band_only')
        
        # Build the decision tree
        result = ee.Algorithms.If(
            count_8.gt(0),
            # We have 8-band images
            ee.Algorithms.If(
                get_8band_coverage(),
                # Case 1: 8-band covers well
                make_8band_mosaic(),
                # Case 2: 8-band insufficient
                ee.Algorithms.If(
                    count_4.gt(0),
                    # Mix 8-band and 4-band
                    make_mixed_mosaic(),
                    # Only 8-band available (with gaps)
                    make_8band_partial_mosaic()
                )
            ),
            # Case 3: No 8-band images
            make_4band_mosaic()
        )
        
        # Cast to ee.Image and copy properties
        result_img = ee.Image(result)
        return result_img.copyProperties(first_img, ['date_key', 'id_pref', 'time_start'])
    
    return ee.ImageCollection(dates.map(_pick))


def make_add_aoi_cloud_fraction(aoi):
    def _fn(img):
        scl = img.select('SCL')
        cloudmask = scl.eq(9).Or(scl.eq(10))  # 9=cloud, 10=cirrus
        cloud_frac = cloudmask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=50,
            maxPixels=1e8
        ).get('SCL')
        return img.set('AOI_CLOUD_FRAC', cloud_frac)
    return _fn

def extract_date(id_str):
    if not id_str:
        return ''
    m = re.search(r'(\d{8})', id_str)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"  # format YYYY-MM-DD
    return ''

def _match_dates_with_tolerance_nn(s2_map, pl_map, days_tol=12):
    """
    Match S2 and Planet dates with improved strategy:
    1. FIRST: Match exact same-day pairs (distance = 0)
    2. THEN: Match remaining using nearest-neighbor
    This prevents early S2 dates from "stealing" Planet dates that could be exact matches later.
    """
    to_dt = lambda s: _dt.strptime(s, "%Y-%m-%d")

    s2_dates = sorted(s2_map.keys())
    pl_dates = sorted(pl_map.keys())
    used_pl = set()
    used_s2 = set()
    matched = []

    # PHASE 1: Exact same-day matches first
    for s2d in s2_dates:
        if s2d in pl_dates and s2d not in used_pl:
            matched.append((s2d, s2d))
            used_pl.add(s2d)
            used_s2.add(s2d)
    
    exact_matches = len(matched)

    # PHASE 2: Nearest-neighbor for remaining
    for s2d in s2_dates:
        if s2d in used_s2:
            continue
        sdt = to_dt(s2d)
        candidates = []
        for pld in pl_dates:
            if pld in used_pl:
                continue
            delta = abs((to_dt(pld) - sdt).days)
            if delta <= days_tol:
                candidates.append((delta, pld))

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            _, best_pld = candidates[0]
            matched.append((s2d, best_pld))
            used_pl.add(best_pld)
            used_s2.add(s2d)

    pl_only = [d for d in pl_dates if d not in used_pl]
    s2_only = [d for d in s2_dates if d not in used_s2]

    # DEBUG: summarize
    print(f"[matcher] {exact_matches} exact + {len(matched)-exact_matches} nn = {len(matched)} matched, {len(pl_only)} PL-only, {len(s2_only)} S2-only")
    return matched, pl_only, s2_only


# --------- Main pairing function (uses scores) ----------
def find_pairs(aoi, start_date, end_date, planet_coll_id, planet_coll_id_fallback=None, planet_coll_id_additional=None,
               s2_cloud_pct_max=60, s2_aoi_cloud_frac_max=0.1, planet_q_cloud_max=20,
               lightweight=False):

    if lightweight:
        # Lightweight mode: skip heavy AOI-level reduceRegion operations.
        # Uses scene-level CLOUDY_PIXEL_PERCENTAGE only (no per-image AOI cloud fraction).
        # Uses simple _best_per_day instead of planet_best_per_day_adaptive.
        # Suitable when tile-level coverage checks handle quality downstream.
        print("   [lightweight] Skipping AOI-level cloud fraction and adaptive mosaicking")
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', s2_cloud_pct_max))
            .map(_add_common_fields)
            .map(_add_s2_quality)
            .map(_add_s2_score)
            .map(_add_pref_id))
        s2_best = _best_per_day(s2)
    else:
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', s2_cloud_pct_max))
            .map(make_add_aoi_cloud_fraction(aoi))
            .filter(ee.Filter.notNull(['AOI_CLOUD_FRAC']))         # guard against null
            .filter(ee.Filter.lt('AOI_CLOUD_FRAC', s2_aoi_cloud_frac_max))
            .map(_add_common_fields)
            .map(_add_s2_quality)
            .map(_add_s2_score)
            .map(_add_pref_id))
        s2_best = _best_per_day(s2)


    # MERGE ALL REPOSITORIES to get maximum coverage
    def prep_planet_collection(coll_id, conf_threshold=60):
        """Prepare a Planet collection with all filters and scores"""
        return (ee.ImageCollection(coll_id)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .map(_add_common_fields)  
            .map(_add_planet_quality) 
            .map(_add_planet_score)
            .filter(ee.Filter.lt('q_cloud', planet_q_cloud_max))                 
            .map(make_add_udm_mask_and_score_for_aoi(aoi, conf_threshold=conf_threshold))
            .map(_add_pref_id))
    
    # Try primary collection
    try:
        pl_primary = prep_planet_collection(planet_coll_id, conf_threshold=60)
        primary_size = pl_primary.size().getInfo()
    except Exception as _e:
        print(f"   [warn] Primary collection failed: {_e}")
        primary_size = 0
        pl_primary = None
    
    # Try fallback/secondary collection if provided
    secondary_size = 0
    pl_secondary = None
    if planet_coll_id_fallback:
        try:
            pl_secondary = prep_planet_collection(planet_coll_id_fallback, conf_threshold=60)
            secondary_size = pl_secondary.size().getInfo()
        except Exception as _e:
            print(f"   [warn] Fallback collection failed: {_e}")
            secondary_size = 0
            pl_secondary = None
    
    # Try additional/tertiary collection if provided
    tertiary_size = 0
    pl_tertiary = None
    if planet_coll_id_additional:
        try:
            pl_tertiary = prep_planet_collection(planet_coll_id_additional, conf_threshold=60)
            tertiary_size = pl_tertiary.size().getInfo()
        except Exception as _e:
            print(f"   [warn] Additional collection failed: {_e}")
            tertiary_size = 0
            pl_tertiary = None
    
    # MERGE all collections that have data
    collections_to_merge = []
    sizes = []
    if primary_size > 0:
        collections_to_merge.append(pl_primary)
        sizes.append(f"primary={primary_size}")
    if secondary_size > 0:
        collections_to_merge.append(pl_secondary)
        sizes.append(f"secondary={secondary_size}")
    if tertiary_size > 0:
        collections_to_merge.append(pl_tertiary)
        sizes.append(f"additional={tertiary_size}")
    
    if len(collections_to_merge) == 0:
        raise ValueError(f"No Planet data found in any collection for {start_date} to {end_date}")
    elif len(collections_to_merge) == 1:
        print(f"   Using single collection: {sizes[0]} images")
        pl_prepped = collections_to_merge[0]
    else:
        print(f"   ✅ MERGING {len(collections_to_merge)} repositories: {', '.join(sizes)} images")
        pl_prepped = collections_to_merge[0]
        for coll in collections_to_merge[1:]:
            pl_prepped = pl_prepped.merge(coll)
        
    print(f"   Processing merged Planet collection...")
    if lightweight:
        # Simple best-per-day using quality mosaic (not single strips)
        pl_best = planet_best_per_day(pl_prepped)
        print(f"   [lightweight] Using quality-mosaic best-per-day (no adaptive mosaicking)")
    else:
        pl_best = (planet_best_per_day_adaptive(pl_prepped, aoi)
            .map(make_tag_has_data(aoi, scale=50, tileScale=2))
            .filter(ee.Filter.gt('has_data', 0))
        )
    print(f"   Planet processing pipeline defined, fetching data...")

    # Dates - with retry logic for GEE rate limiting
    import time as _time_module
    def _getInfo_with_retry(ee_obj, max_retries=5, initial_delay=10, label="data"):
        """Get info from EE object with exponential backoff on rate limiting."""
        print(f"      Fetching {label}...", end=" ", flush=True)
        for attempt in range(max_retries):
            try:
                result = ee_obj.getInfo()
                print(f"✓")
                return result
            except Exception as e:
                if "Too many concurrent aggregations" in str(e) and attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)  # Exponential backoff
                    print(f"\n   ⏳ GEE rate limit hit, waiting {delay}s (attempt {attempt + 1}/{max_retries})...")
                    _time_module.sleep(delay)
                else:
                    print(f"✗")
                    raise
    
    # Fetch S2 data first (lighter computation)
    s2_dates = _getInfo_with_retry(s2_best.aggregate_array('date_key'), label="S2 dates")
    s2_ids   = _getInfo_with_retry(s2_best.aggregate_array('id_pref'), label="S2 IDs")

    # Small delay to let GEE aggregations complete before heavy Planet fetch
    _time_module.sleep(2)
    
    # Planet fetch triggers heavy computations from planet_best_per_day_adaptive
    pl_dates = _getInfo_with_retry(pl_best.aggregate_array('date_key'), label="Planet dates (heavy)")
    _time_module.sleep(1)  # Brief pause between heavy calls
    pl_ids   = _getInfo_with_retry(pl_best.aggregate_array('id_pref'), label="Planet IDs")



    def _to_dt(s): return _dt.strptime(s, "%Y-%m-%d")

    def _count_near_hits(s2_dates, pl_dates, days_tol=7):
        hits = 0
        for sd in s2_dates:
            sdt = _to_dt(sd)
            for pd in pl_dates:
                if abs((_to_dt(pd) - sdt).days) <= days_tol:
                    hits += 1
                    break
        return hits

    # print('Potential matches within ±7d:', _count_near_hits(s2_dates, pl_dates, 7))

    def _to_dt(s): return _dt.strptime(s, "%Y-%m-%d")

    # print("First S2 vs first PL:",
        # s2_dates[0], pl_dates[0],
        # "diff days =", (_to_dt(pl_dates[0]) - _to_dt(s2_dates[0])).days)

    # Show the 3 closest PL dates for the first 3 S2 dates
    for sd in s2_dates[:3]:
        ranked = sorted([(pd, abs((_to_dt(pd) - _to_dt(sd)).days)) for pd in pl_dates],
                        key=lambda x: x[1])[:3]
        # print("Closest PL to", sd, "=>", ranked)


    # print('S2 days:', len(s2_dates), 'range:', min(s2_dates), '→', max(s2_dates))
    # print('PL days:', len(pl_dates), 'range:', (min(pl_dates) if pl_dates else '∅'),
        # '→', (max(pl_dates) if pl_dates else '∅'))
    # print('S2 sample:', s2_dates[:10])
    # print('PL sample:', pl_dates[:10])

    s2_map = {d: i for d, i in zip(s2_dates, s2_ids)}
    pl_map = {d: i for d, i in zip(pl_dates, pl_ids)}

    matches, planet_only_dates, s2_only_dates = _match_dates_with_tolerance_nn(s2_map, pl_map, days_tol=7)
    # print(f"NN matched: {len(matches)} | PL-only: {len(planet_only_dates)} | S2-only: {len(s2_only_dates)}")
    # if matches[:5]:
        # print("Sample pairs:", matches[:5])


    matched = [{
        's2_date': s2_d,
        'planet_date': pl_d,
        's2_id': s2_map[s2_d],
        'planet_id': pl_map[pl_d]
    } for s2_d, pl_d in matches]

    # Unmatched IDs (as before)
    planet_unmatched_ids = [pl_map[d] for d in planet_only_dates]
    s2_unmatched_ids = [s2_map[d] for d in s2_only_dates]

    # ---- NEW: 2-column table of unmatched dates ----
    # Rows are independent lists aligned by index; blanks for shorter side.
    rows = max(len(planet_only_dates), len(s2_only_dates))
    unmatched_table = []
    for i in range(rows):
        unmatched_table.append({
            'planet_date': planet_only_dates[i] if i < len(planet_only_dates) else '',
            's2_date':     s2_only_dates[i]     if i < len(s2_only_dates)     else ''
        })

    return {
        'matched_pairs': matched,
        'planet_unmatched': planet_unmatched_ids,
        's2_unmatched': s2_unmatched_ids,
        'unmatched_table': unmatched_table,
        'planet_unmatched_dates': planet_only_dates,
        's2_unmatched_dates': s2_only_dates,
        's2_collection': s2_best,      
        'planet_collection': pl_best,
        # Full lists for export
        's2_all_dates': s2_dates,
        's2_all_ids': s2_ids,
        'planet_all_dates': pl_dates,
        'planet_all_ids': pl_ids
    }


def export_image_inventory(pair_results: Dict, output_path: str, 
                           include_metadata: bool = True,
                           region_name: str = None) -> Dict[str, str]:
    """
    Export inventory of available S2 and PlanetScope images to CSV/Excel files.
    
    Creates three files:
    1. {output_path}_s2_inventory.csv - All available Sentinel-2 images
    2. {output_path}_planet_inventory.csv - All available PlanetScope images  
    3. {output_path}_matched_pairs.csv - Matched image pairs
    4. {output_path}_summary.xlsx - Excel workbook with all sheets (if openpyxl available)
    
    Args:
        pair_results: Dictionary returned by find_pairs()
        output_path: Base path for output files (without extension)
        include_metadata: Whether to fetch additional metadata from GEE (slower)
        region_name: Optional region name to include in output
    
    Returns:
        Dictionary with paths to created files
    """
    import os
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    created_files = {}
    
    # =========================================================================
    # 1. Sentinel-2 Inventory
    # =========================================================================
    s2_data = []
    s2_dates = pair_results.get('s2_all_dates', [])
    s2_ids = pair_results.get('s2_all_ids', [])
    matched_s2_dates = set(p['s2_date'] for p in pair_results.get('matched_pairs', []))
    
    for date, img_id in zip(s2_dates, s2_ids):
        row = {
            'date': date,
            'image_id': img_id,
            'matched': 'Yes' if date in matched_s2_dates else 'No',
            'sensor': 'Sentinel-2'
        }
        
        # Fetch additional metadata if requested
        if include_metadata and pair_results.get('s2_collection'):
            try:
                s2_col = pair_results['s2_collection']
                img = s2_col.filter(ee.Filter.eq('id_pref', img_id)).first()
                props = img.toDictionary(['q_cloud', 'dist_noon', 'CLOUDY_PIXEL_PERCENTAGE', 
                                          'MEAN_SOLAR_ZENITH_ANGLE', 'AOI_CLOUD_FRAC']).getInfo()
                row['cloud_pct'] = props.get('CLOUDY_PIXEL_PERCENTAGE')
                row['aoi_cloud_frac'] = props.get('AOI_CLOUD_FRAC')
                row['solar_zenith'] = props.get('MEAN_SOLAR_ZENITH_ANGLE')
                row['dist_noon'] = props.get('dist_noon')
                row['q_cloud'] = props.get('q_cloud')
            except Exception as e:
                pass  # Skip metadata if fetch fails
        
        if region_name:
            row['region'] = region_name
            
        s2_data.append(row)
    
    s2_df = pd.DataFrame(s2_data)
    s2_csv_path = f"{output_path}_s2_inventory.csv"
    s2_df.to_csv(s2_csv_path, index=False)
    created_files['s2_inventory'] = s2_csv_path
    print(f"✅ Saved S2 inventory: {s2_csv_path} ({len(s2_data)} images)")
    
    # =========================================================================
    # 2. PlanetScope Inventory
    # =========================================================================
    planet_data = []
    planet_dates = pair_results.get('planet_all_dates', [])
    planet_ids = pair_results.get('planet_all_ids', [])
    matched_planet_dates = set(p['planet_date'] for p in pair_results.get('matched_pairs', []))
    
    for date, img_id in zip(planet_dates, planet_ids):
        row = {
            'date': date,
            'image_id': img_id,
            'matched': 'Yes' if date in matched_planet_dates else 'No',
            'sensor': 'PlanetScope'
        }
        
        # Fetch additional metadata if requested
        if include_metadata and pair_results.get('planet_collection'):
            try:
                pl_col = pair_results['planet_collection']
                img = pl_col.filter(ee.Filter.eq('id_pref', img_id)).first()
                # Try different property names for Planet metadata
                props = img.toDictionary().getInfo()
                row['cloud_pct'] = props.get('q_cloud')
                row['sun_elevation'] = props.get('sun_elevation') or props.get('sun_elevation_safe')
                if row['sun_elevation']:
                    row['solar_zenith'] = 90.0 - float(row['sun_elevation'])
                row['dist_noon'] = props.get('dist_noon')
                row['band_mode'] = props.get('band_mode', 'unknown')
                row['aoi_clear_frac'] = props.get('aoi_clear_frac')
            except Exception as e:
                pass  # Skip metadata if fetch fails
        
        if region_name:
            row['region'] = region_name
            
        planet_data.append(row)
    
    planet_df = pd.DataFrame(planet_data)
    planet_csv_path = f"{output_path}_planet_inventory.csv"
    planet_df.to_csv(planet_csv_path, index=False)
    created_files['planet_inventory'] = planet_csv_path
    print(f"✅ Saved Planet inventory: {planet_csv_path} ({len(planet_data)} images)")
    
    # =========================================================================
    # 3. Matched Pairs
    # =========================================================================
    matched_data = []
    for pair in pair_results.get('matched_pairs', []):
        row = {
            's2_date': pair['s2_date'],
            'planet_date': pair['planet_date'],
            's2_id': pair['s2_id'],
            'planet_id': pair['planet_id'],
            'date_diff_days': abs((datetime.strptime(pair['s2_date'], '%Y-%m-%d') - 
                                   datetime.strptime(pair['planet_date'], '%Y-%m-%d')).days)
        }
        if region_name:
            row['region'] = region_name
        matched_data.append(row)
    
    matched_df = pd.DataFrame(matched_data)
    matched_csv_path = f"{output_path}_matched_pairs.csv"
    matched_df.to_csv(matched_csv_path, index=False)
    created_files['matched_pairs'] = matched_csv_path
    print(f"✅ Saved matched pairs: {matched_csv_path} ({len(matched_data)} pairs)")
    
    # =========================================================================
    # 4. Unmatched Images
    # =========================================================================
    unmatched_data = []
    for date in pair_results.get('s2_unmatched_dates', []):
        unmatched_data.append({'date': date, 'sensor': 'Sentinel-2', 'status': 'No Planet match'})
    for date in pair_results.get('planet_unmatched_dates', []):
        unmatched_data.append({'date': date, 'sensor': 'PlanetScope', 'status': 'No S2 match'})
    
    if unmatched_data:
        unmatched_df = pd.DataFrame(unmatched_data)
        unmatched_csv_path = f"{output_path}_unmatched.csv"
        unmatched_df.to_csv(unmatched_csv_path, index=False)
        created_files['unmatched'] = unmatched_csv_path
        print(f"✅ Saved unmatched images: {unmatched_csv_path} ({len(unmatched_data)} images)")
    
    # =========================================================================
    # 5. Summary Excel (if openpyxl available)
    # =========================================================================
    try:
        excel_path = f"{output_path}_summary.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total S2 Images',
                    'Total Planet Images', 
                    'Matched Pairs',
                    'Unmatched S2',
                    'Unmatched Planet',
                    'Match Rate (S2)',
                    'Match Rate (Planet)',
                    'Date Range Start',
                    'Date Range End'
                ],
                'Value': [
                    len(s2_dates),
                    len(planet_dates),
                    len(pair_results.get('matched_pairs', [])),
                    len(pair_results.get('s2_unmatched_dates', [])),
                    len(pair_results.get('planet_unmatched_dates', [])),
                    f"{100 * len(matched_s2_dates) / max(len(s2_dates), 1):.1f}%",
                    f"{100 * len(matched_planet_dates) / max(len(planet_dates), 1):.1f}%",
                    min(s2_dates + planet_dates) if (s2_dates or planet_dates) else 'N/A',
                    max(s2_dates + planet_dates) if (s2_dates or planet_dates) else 'N/A'
                ]
            }
            if region_name:
                summary_data['Metric'].insert(0, 'Region')
                summary_data['Value'].insert(0, region_name)
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            s2_df.to_excel(writer, sheet_name='S2_Inventory', index=False)
            planet_df.to_excel(writer, sheet_name='Planet_Inventory', index=False)
            matched_df.to_excel(writer, sheet_name='Matched_Pairs', index=False)
            if unmatched_data:
                unmatched_df.to_excel(writer, sheet_name='Unmatched', index=False)
        
        created_files['excel_summary'] = excel_path
        print(f"✅ Saved Excel summary: {excel_path}")
    except ImportError:
        print("⚠️ openpyxl not installed, skipping Excel export. Install with: pip install openpyxl")
    except Exception as e:
        print(f"⚠️ Excel export failed: {e}")
    
    # Print summary
    print(f"\n📊 INVENTORY SUMMARY:")
    print(f"   S2 images: {len(s2_dates)} ({len(matched_s2_dates)} matched)")
    print(f"   Planet images: {len(planet_dates)} ({len(matched_planet_dates)} matched)")
    print(f"   Matched pairs: {len(pair_results.get('matched_pairs', []))}")
    
    return created_files


def get_image_from_collection(image_collection, image_id: str, id_field: str = 'id_pref'):
    """
    Retrieve a single image from an existing ImageCollection by matching a property.
    
    Args:
        image_collection: ee.ImageCollection to search
        image_id: The image ID to match
        id_field: Property field to match against (default: 'id_pref')
    
    Returns:
        ee.Image or None if not found
    """
    try:
        filtered = image_collection.filter(ee.Filter.eq(id_field, image_id))
        
        # Check if any images found
        size = filtered.size().getInfo()
        if size == 0:
            print(f"⚠️ No image found with {id_field}: {image_id}")
            return None
        
        return ee.Image(filtered.first())
    
    except Exception as e:
        print(f"❌ Error retrieving image {image_id}: {e}")
        return None
    

def get_paired_images_from_collections(matched_pairs: List[Dict],
                                        s2_collection,
                                        planet_collection,
                                        aoi = None) -> List[Dict]:
    """
    Retrieve paired Sentinel-2 and PlanetScope images from pre-processed collections.
    This preserves all bands and properties added during preprocessing (UDM, qscore, etc.)
    
    Args:
        matched_pairs: List of dicts with 's2_id', 'planet_id', 's2_date', 'planet_date'
        s2_collection: Pre-processed S2 ImageCollection (s2_best)
        planet_collection: Pre-processed Planet ImageCollection (pl_best)
        aoi: Optional AOI to clip images
    
    Returns:
        List of dicts with 's2_img', 'planet_img', and metadata
    """
    paired_images = []
    
    for i, pair in enumerate(matched_pairs):
        print(f"\n🔄 Retrieving pair {i+1}/{len(matched_pairs)}...")
        print(f"   S2: {pair['s2_date']} | Planet: {pair['planet_date']}")
        
        # Get Sentinel-2 image from s2_best collection
        s2_img = get_image_from_collection(s2_collection, pair['s2_id'], 'id_pref')
        
        # Get PlanetScope image from pl_best collection  
        planet_img = get_image_from_collection(planet_collection, pair['planet_id'], 'id_pref')
        
        if s2_img is None or planet_img is None:
            print(f"   ⚠️ Skipping pair {i+1} - image retrieval failed")
            continue
        
        # Optional: clip to AOI
        if aoi is not None:
            s2_img = s2_img.clip(aoi)
            planet_img = planet_img.clip(aoi)
        
        paired_images.append({
            's2_img': s2_img,
            'planet_img': planet_img,
            's2_id': pair['s2_id'],
            'planet_id': pair['planet_id'],
            's2_date': pair['s2_date'],
            'planet_date': pair['planet_date'],
            'pair_index': i
        })
        
        print(f"   ✅ Retrieved pair {i+1}")
    
    print(f"\n✅ Total pairs retrieved: {len(paired_images)}/{len(matched_pairs)}")
    return paired_images

def download_paired_images_as_arrays(paired_images: List[Dict],
                                     aoi,
                                     scale: int = 10,
                                     s2_bands: List[str] = None,
                                     planet_bands: List[str] = None) -> List[Dict]:
    """
    Download paired images as numpy arrays for processing.
    
    Args:
        paired_images: List from get_paired_images_from_collections()
        aoi: Area of interest geometry
        scale: Resolution in meters
        s2_bands: List of S2 bands (default: ['B2','B3','B4','B8'])
        planet_bands: List of Planet bands (default: ['B1','B2','B3','B4'])
    
    Returns:
        List of dicts with numpy arrays and metadata
    """
    if s2_bands is None:
        s2_bands = ['B2', 'B3', 'B4', 'B8A']  # Blue, Green, Red, NIR
    
    if planet_bands is None:
        planet_bands = ['B1', 'B2', 'B3', 'B4']  # Blue, Green, Red, NIR
    
    results = []
    
    for pair in paired_images:
        print(f"\n⬇️ Downloading pair {pair['pair_index']+1}...")
        print(f"   S2: {pair['s2_date']} | Planet: {pair['planet_date']}")
        
        try:
            # Select bands
            s2_selected = pair['s2_img'].select(s2_bands)
            planet_selected = pair['planet_img'].select(planet_bands)
            
            # Download S2
            s2_region = s2_selected.sampleRectangle(
                region=aoi,
                defaultValue=0
            )
            s2_dict = s2_region.getInfo()
            
            # Convert to numpy array (bands, height, width)
            s2_np = np.stack([
                np.array(s2_dict['properties'][band]) 
                for band in s2_bands
            ], axis=0)  # Shape: (bands, height, width)
            
            # Download Planet
            planet_region = planet_selected.sampleRectangle(
                region=aoi,
                defaultValue=0
            )
            planet_dict = planet_region.getInfo()
            
            planet_np = np.stack([
                np.array(planet_dict['properties'][band]) 
                for band in planet_bands
            ], axis=0)  # Shape: (bands, height, width)
            
            results.append({
                's2_array': s2_np,
                'planet_array': planet_np,
                's2_date': pair['s2_date'],
                'planet_date': pair['planet_date'],
                's2_id': pair['s2_id'],
                'planet_id': pair['planet_id'],
                'pair_index': pair['pair_index']
            })
            
            print(f"   ✅ S2 shape: {s2_np.shape}, Planet shape: {planet_np.shape}")
            
        except Exception as e:
            print(f"   ❌ Error downloading pair {pair['pair_index']+1}: {e}")
            continue
    
    return results


def harmonize_images_patched(ref, coreg, patch_size=256, overlap=32, verbose=False):
    """
    Harmonize coregistered image to reference using patch-based min-max matching.
    
    Args:
        ref: Reference image (height, width, bands) - S2
        coreg: Coregistered target image (height, width, bands) - Planet
        patch_size: Size of square patches
        overlap: Overlap between patches for smooth blending
        verbose: Print progress
    
    Returns:
        Harmonized image (height, width, bands)
    """
    def minmax_match(src_band, ref_band, eps=1e-5):
        """Match histogram statistics of source to reference."""
        src_min, src_max = np.nanpercentile(src_band, (2, 98))
        ref_min, ref_max = np.nanpercentile(ref_band, (2, 98))
        norm = (src_band - src_min) / max((src_max - src_min), eps)
        return norm * max((ref_max - ref_min), eps) + ref_min
    
    height, width, n_bands = coreg.shape
    
    # Ensure reference has same dimensions
    if ref.shape != coreg.shape:
        min_h = min(ref.shape[0], coreg.shape[0])
        min_w = min(ref.shape[1], coreg.shape[1])
        ref = ref[:min_h, :min_w, :]
        coreg = coreg[:min_h, :min_w, :]
        height, width = min_h, min_w
    
    # Initialize output
    harmonized = np.zeros_like(coreg, dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    
    # Calculate patch positions
    step = patch_size - overlap
    n_patches_h = int(np.ceil((height - overlap) / step))
    n_patches_w = int(np.ceil((width - overlap) / step))
    
    if verbose:
        print(f"   Processing {n_patches_h}x{n_patches_w} = {n_patches_h*n_patches_w} patches")
    
    patch_count = 0
    total_patches = n_patches_h * n_patches_w
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Calculate patch boundaries
            start_h = i * step
            start_w = j * step
            end_h = min(start_h + patch_size, height)
            end_w = min(start_w + patch_size, width)
            
            # Extract patches
            ref_patch = ref[start_h:end_h, start_w:end_w, :]
            coreg_patch = coreg[start_h:end_h, start_w:end_w, :]
            
            # Harmonize each band in the patch
            harm_patch = np.zeros_like(coreg_patch, dtype=np.float32)
            for b in range(n_bands):
                harm_patch[:, :, b] = minmax_match(
                    coreg_patch[:, :, b], 
                    ref_patch[:, :, b]
                )
            
            # Create weight mask for smooth blending (distance from patch center)
            patch_h, patch_w = harm_patch.shape[:2]
            y_center, x_center = patch_h / 2, patch_w / 2
            y_grid, x_grid = np.ogrid[:patch_h, :patch_w]
            
            # Gaussian-like weights (higher at center, lower at edges)
            dist_from_center = np.sqrt(
                ((y_grid - y_center) / (patch_h / 2)) ** 2 + 
                ((x_grid - x_center) / (patch_w / 2)) ** 2
            )
            weights = np.exp(-dist_from_center)  # Gaussian falloff
            
            # Accumulate weighted patches
            harmonized[start_h:end_h, start_w:end_w, :] += harm_patch * weights[:, :, np.newaxis]
            weight_map[start_h:end_h, start_w:end_w] += weights
            
            patch_count += 1
            if verbose and patch_count % 10 == 0:
                print(f"   Processed {patch_count}/{total_patches} patches ({100*patch_count/total_patches:.1f}%)")
    
    # Normalize by weights (avoid division by zero)
    weight_map = np.maximum(weight_map, 1e-10)
    harmonized = harmonized / weight_map[:, :, np.newaxis]
    
    if verbose:
        print(f"   ✅ Completed all {total_patches} patches")
    
    return harmonized


def harmonize_images_grid(ref, coreg, grid_rows=4, grid_cols=4, verbose=False):
    """
    Harmonize coregistered image using a regular grid of patches.
    Simpler alternative without overlap blending.
    
    Args:
        ref: Reference image (height, width, bands) - S2
        coreg: Coregistered target image (height, width, bands) - Planet
        grid_rows: Number of rows in grid
        grid_cols: Number of columns in grid
        verbose: Print progress
    
    Returns:
        Harmonized image (height, width, bands)
    """
    def minmax_match(src_band, ref_band, eps=1e-5):
        """Match histogram statistics of source to reference."""
        src_min, src_max = np.nanpercentile(src_band, (2, 98))
        ref_min, ref_max = np.nanpercentile(ref_band, (2, 98))
        norm = (src_band - src_min) / max((src_max - src_min), eps)
        return norm * max((ref_max - ref_min), eps) + ref_min
    
    height, width, n_bands = coreg.shape
    
    # Ensure reference has same dimensions
    if ref.shape != coreg.shape:
        min_h = min(ref.shape[0], coreg.shape[0])
        min_w = min(ref.shape[1], coreg.shape[1])
        ref = ref[:min_h, :min_w, :]
        coreg = coreg[:min_h, :min_w, :]
        height, width = min_h, min_w
    
    # Initialize output
    harmonized = np.zeros_like(coreg, dtype=np.float32)
    
    # Calculate patch dimensions
    patch_h = height // grid_rows
    patch_w = width // grid_cols
    
    if verbose:
        print(f"   Processing {grid_rows}x{grid_cols} = {grid_rows*grid_cols} patches")
        print(f"   Patch size: {patch_h}x{patch_w}")
    
    patch_count = 0
    total_patches = grid_rows * grid_cols
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            # Calculate patch boundaries
            start_h = i * patch_h
            start_w = j * patch_w
            
            # Last patches extend to image edge
            if i == grid_rows - 1:
                end_h = height
            else:
                end_h = start_h + patch_h
                
            if j == grid_cols - 1:
                end_w = width
            else:
                end_w = start_w + patch_w
            
            # Extract patches
            ref_patch = ref[start_h:end_h, start_w:end_w, :]
            coreg_patch = coreg[start_h:end_h, start_w:end_w, :]
            
            # Harmonize each band in the patch
            for b in range(n_bands):
                harmonized[start_h:end_h, start_w:end_w, b] = minmax_match(
                    coreg_patch[:, :, b], 
                    ref_patch[:, :, b]
                )
            
            patch_count += 1
            if verbose:
                print(f"   Processed patch [{i},{j}] ({patch_count}/{total_patches})")
    
    if verbose:
        print(f"   ✅ Completed all {total_patches} patches")
    
    return harmonized


def process_paired_images(paired_arrays: List[Dict], 
                          harmonization_method: str = 'patched',
                          patch_size: int = 256,
                          overlap: int = 32,
                          grid_rows: int = 4,
                          grid_cols: int = 4,
                          r_bmatch: int = 3,
                          s_bmatch: int = 3,
                          max_shift: int = 10,
                          verbose: bool = True,
                          verbose_img: bool = False) -> List[Dict]:
    """
    Coregister and harmonize paired images using patch-based processing.
    
    Args:
        paired_arrays: List from download_paired_images_as_arrays()
        harmonization_method: 'patched' (overlapping) or 'grid' (non-overlapping)
        patch_size: Size of patches for 'patched' method
        overlap: Overlap between patches for 'patched' method
        grid_rows: Number of rows for 'grid' method
        grid_cols: Number of columns for 'grid' method
        r_bmatch: Reference band index for coregistration (0-indexed)
        s_bmatch: Source band index for coregistration (0-indexed)
        max_shift: Maximum allowed shift in pixels
        verbose: Print progress
        verbose_img: Show visualizations
    
    Returns:
        List of processed results with harmonized images
    """
    results = []
    
    for arr_pair in paired_arrays:
        print(f"\n🔧 Processing pair {arr_pair['pair_index']+1}...")
        print(f"   S2: {arr_pair['s2_date']} | Planet: {arr_pair['planet_date']}")
        
        try:
            # Get arrays
            s2_array = arr_pair['s2_array']      # Shape: (bands, height, width)
            planet_array = arr_pair['planet_array']  # Shape: (bands, height, width)
            
            # Transpose to (height, width, bands) for processing
            s2_hwc = np.transpose(s2_array, (1, 2, 0))  # (H, W, C)
            planet_hwc = np.transpose(planet_array, (1, 2, 0))  # (H, W, C)
            
            if verbose:
                print(f"   Reference (S2) shape: {s2_hwc.shape}")
                print(f"   Target (Planet) shape: {planet_hwc.shape}")
            
            # Step 1: Coregister Planet to S2
            print("   📍 Coregistering...")
            coreg_obj, planet_coreg = coregister_images(
                ref=s2_hwc,
                tar=planet_hwc,
                r_bmatch=r_bmatch,
                s_bmatch=s_bmatch,
                max_shift=max_shift,
                nodata=(0, 0)
            )
            
            if verbose:
                print(f"   ✅ Coregistration complete. Coreg shape: {planet_coreg.shape}")
            
            # Step 2: Harmonize using patch-based method
            print(f"   🎨 Harmonizing (method: {harmonization_method})...")
            
            if harmonization_method == 'patched':
                harmonized = harmonize_images_patched(
                    ref=s2_hwc,
                    coreg=planet_coreg,
                    patch_size=patch_size,
                    overlap=overlap,
                    verbose=verbose
                )
            elif harmonization_method == 'grid':
                harmonized = harmonize_images_grid(
                    ref=s2_hwc,
                    coreg=planet_coreg,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    verbose=verbose
                )
            else:
                raise ValueError(f"Unknown harmonization method: {harmonization_method}")
            
            if verbose:
                print(f"   ✅ Harmonization complete. Harmonized shape: {harmonized.shape}")
            
            # Transpose back to (bands, height, width)
            harmonized_chw = np.transpose(harmonized, (2, 0, 1))
            planet_coreg_chw = np.transpose(planet_coreg, (2, 0, 1))
            
            # Calculate quality metrics
            if verbose:
                print("   📊 Calculating quality metrics...")
                mse = np.mean((s2_hwc - harmonized) ** 2)
                if mse > 0:
                    psnr = 10 * np.log10((np.max(s2_hwc) ** 2) / mse)
                else:
                    psnr = float('inf')
                print(f"   PSNR: {psnr:.2f} dB")
                
                try:
                    from skimage.metrics import structural_similarity as ssim
                    ssim_value = ssim(
                        s2_hwc, 
                        harmonized, 
                        data_range=s2_hwc.max() - s2_hwc.min(),
                        channel_axis=2
                    )
                    print(f"   SSIM: {ssim_value:.4f}")
                except ImportError:
                    print("   SSIM calculation requires scikit-image")
            
            results.append({
                's2_array': s2_array,
                'planet_original': planet_array,
                'planet_coregistered': planet_coreg_chw,
                'planet_harmonized': harmonized_chw,
                's2_date': arr_pair['s2_date'],
                'planet_date': arr_pair['planet_date'],
                's2_id': arr_pair['s2_id'],
                'planet_id': arr_pair['planet_id'],
                'pair_index': arr_pair['pair_index'],
                'coreg_object': coreg_obj
            })
            
            print(f"   ✅ Pair {arr_pair['pair_index']+1} processed successfully!")
            
        except Exception as e:
            print(f"   ❌ Error processing pair {arr_pair['pair_index']+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def save_processed_results(processed_results: List[Dict], 
                           output_dir: str,
                           region_name: str):
    """
    Save processed results as GeoTIFFs.
    
    Args:
        processed_results: List from process_paired_images()
        output_dir: Output directory path
        region_name: Region name for filename
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for result in processed_results:
        pair_idx = result['pair_index']
        s2_date = result['s2_date'].replace('-', '')
        planet_date = result['planet_date'].replace('-', '')
        
        # Save S2 reference
        s2_path = os.path.join(
            output_dir, 
            f"{region_name}_S2_{s2_date}_pair{pair_idx}_ref.tif"
        )
        save_array_as_geotiff(result['s2_array'], s2_path)
        
        # Save Planet original
        planet_orig_path = os.path.join(
            output_dir, 
            f"{region_name}_Planet_{planet_date}_pair{pair_idx}_original.tif"
        )
        save_array_as_geotiff(result['planet_original'], planet_orig_path)
        
        # Save Planet coregistered
        planet_coreg_path = os.path.join(
            output_dir, 
            f"{region_name}_Planet_{planet_date}_pair{pair_idx}_coregistered.tif"
        )
        save_array_as_geotiff(result['planet_coregistered'], planet_coreg_path)
        
        # Save Planet harmonized
        planet_harm_path = os.path.join(
            output_dir, 
            f"{region_name}_Planet_{planet_date}_pair{pair_idx}_harmonized.tif"
        )
        save_array_as_geotiff(result['planet_harmonized'], planet_harm_path)
        
        print(f"✅ Saved pair {pair_idx} to {output_dir}")


def save_array_as_geotiff(array, filepath):
    """Save numpy array as GeoTIFF."""
    from rasterio.transform import from_bounds
    
    # Assume array is (bands, height, width)
    bands, height, width = array.shape
    
    # Create a simple transform (you may want to use actual georeferencing)
    transform = from_bounds(0, 0, width, height, width, height)
    
    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=array.dtype,
        transform=transform
    ) as dst:
        dst.write(array)




regions = [
    {
        "name": "STA-2",
        "aoi": ee.Geometry.Polygon([[
            [-80.61495827,26.43609898],
            [-80.53742967,26.33549131],
            [-80.48100774,26.42148366],
            [-80.544935,26.42144199],
            [-80.54505831,26.43527516],
            [-80.61739151,26.43780818],
            [-80.61495827,26.43609898]
        ]]),
        "start": "2020-01-01",
        "end": "2021-01-01"
    },
    {
        "name": "STA-2",
        "aoi": ee.Geometry.Polygon([[
            [-80.61495827,26.43609898],
            [-80.53742967,26.33549131],
            [-80.48100774,26.42148366],
            [-80.544935,26.42144199],
            [-80.54505831,26.43527516],
            [-80.61739151,26.43780818],
            [-80.61495827,26.43609898]
        ]]),
        "start": "2021-01-01",
        "end": "2022-01-01"
    },
    {
        "name": "STA-2",
        "aoi": ee.Geometry.Polygon([[
            [-80.61495827,26.43609898],
            [-80.53742967,26.33549131],
            [-80.48100774,26.42148366],
            [-80.544935,26.42144199],
            [-80.54505831,26.43527516],
            [-80.61739151,26.43780818],
            [-80.61495827,26.43609898]
        ]]),
        "start": "2022-01-01",
        "end": "2023-01-01"
    },
    {
        "name": "STA-2",
        "aoi": ee.Geometry.Polygon([[
            [-80.61495827,26.43609898],
            [-80.53742967,26.33549131],
            [-80.48100774,26.42148366],
            [-80.544935,26.42144199],
            [-80.54505831,26.43527516],
            [-80.61739151,26.43780818],
            [-80.61495827,26.43609898]
        ]]),
        "start": "2023-01-01",
        "end": "2024-01-01"
    },
    {
        "name": "STA-2",
        "aoi": ee.Geometry.Polygon([[
            [-80.61495827,26.43609898],
            [-80.53742967,26.33549131],
            [-80.48100774,26.42148366],
            [-80.544935,26.42144199],
            [-80.54505831,26.43527516],
            [-80.61739151,26.43780818],
            [-80.61495827,26.43609898]
        ]]),
        "start": "2024-01-01",
        "end": "2025-01-01"
    },

    {
        "name": "STA-1E",
        "aoi": ee.Geometry.Polygon([[
            [-80.36388849,26.6843562],
            [-80.36395033,26.67959061],
            [-80.34775062,26.64813207],
            [-80.33282065,26.63369505],
            [-80.31717686,26.62427804],
            [-80.316679,26.65008888],
            [-80.30003641,26.65027958],
            [-80.29854709,26.68272326],
            [-80.36388849,26.6843562]
        ]]),
        "start": "2020-01-01",
        "end": "2021-01-01"
    },
    {
        "name": "STA-1E",
        "aoi": ee.Geometry.Polygon([[
            [-80.36388849,26.6843562],
            [-80.36395033,26.67959061],
            [-80.34775062,26.64813207],
            [-80.33282065,26.63369505],
            [-80.31717686,26.62427804],
            [-80.316679,26.65008888],
            [-80.30003641,26.65027958],
            [-80.29854709,26.68272326],
            [-80.36388849,26.6843562]
        ]]),
        "start": "2021-01-01",
        "end": "2022-01-01"
    },
    {
        "name": "STA-1E",
        "aoi": ee.Geometry.Polygon([[
            [-80.36388849,26.6843562],
            [-80.36395033,26.67959061],
            [-80.34775062,26.64813207],
            [-80.33282065,26.63369505],
            [-80.31717686,26.62427804],
            [-80.316679,26.65008888],
            [-80.30003641,26.65027958],
            [-80.29854709,26.68272326],
            [-80.36388849,26.6843562]
        ]]),
        "start": "2022-01-01",
        "end": "2023-01-01"
    },
    {
        "name": "STA-1E",
        "aoi": ee.Geometry.Polygon([[
            [-80.36388849,26.6843562],
            [-80.36395033,26.67959061],
            [-80.34775062,26.64813207],
            [-80.33282065,26.63369505],
            [-80.31717686,26.62427804],
            [-80.316679,26.65008888],
            [-80.30003641,26.65027958],
            [-80.29854709,26.68272326],
            [-80.36388849,26.6843562]
        ]]),
        "start": "2023-01-01",
        "end": "2024-01-01"
    },
    {
        "name": "STA-1E",
        "aoi": ee.Geometry.Polygon([[
            [-80.36388849,26.6843562],
            [-80.36395033,26.67959061],
            [-80.34775062,26.64813207],
            [-80.33282065,26.63369505],
            [-80.31717686,26.62427804],
            [-80.316679,26.65008888],
            [-80.30003641,26.65027958],
            [-80.29854709,26.68272326],
            [-80.36388849,26.6843562]
        ]]),
        "start": "2024-01-01",
        "end": "2025-01-01"
    },

    {
        "name": "STA-1W",
        "aoi": ee.Geometry.Polygon([[
            [-80.45309541,26.68276268],
            [-80.45309541,26.57253594],
            [-80.37008749,26.67612901],
            [-80.37428635,26.69141472],
            [-80.45406437,26.69112633],
            [-80.45309541,26.68276268]
        ]]),
        "start": "2020-01-01",
        "end": "2021-01-01"
    },
    {
        "name": "STA-1W",
        "aoi": ee.Geometry.Polygon([[
            [-80.45309541,26.68276268],
            [-80.45309541,26.57253594],
            [-80.37008749,26.67612901],
            [-80.37428635,26.69141472],
            [-80.45406437,26.69112633],
            [-80.45309541,26.68276268]
        ]]),
        "start": "2021-01-01",
        "end": "2022-01-01"
    },
    {
        "name": "STA-1W",
        "aoi": ee.Geometry.Polygon([[
            [-80.45309541,26.68276268],
            [-80.45309541,26.57253594],
            [-80.37008749,26.67612901],
            [-80.37428635,26.69141472],
            [-80.45406437,26.69112633],
            [-80.45309541,26.68276268]
        ]]),
        "start": "2022-01-01",
        "end": "2023-01-01"
    },
    {
        "name": "STA-1W",
        "aoi": ee.Geometry.Polygon([[
            [-80.45309541,26.68276268],
            [-80.45309541,26.57253594],
            [-80.37008749,26.67612901],
            [-80.37428635,26.69141472],
            [-80.45406437,26.69112633],
            [-80.45309541,26.68276268]
        ]]),
        "start": "2023-01-01",
        "end": "2024-01-01"
    },
    {
        "name": "STA-1W",
        "aoi": ee.Geometry.Polygon([[
            [-80.45309541,26.68276268],
            [-80.45309541,26.57253594],
            [-80.37008749,26.67612901],
            [-80.37428635,26.69141472],
            [-80.45406437,26.69112633],
            [-80.45309541,26.68276268]
        ]]),
        "start": "2024-01-01",
        "end": "2025-01-01"
    },
    {
        "name": "STA-3/4",
        "aoi": ee.Geometry.Polygon([[
            [-80.81055951,26.43631098],
            [-80.68733359,26.44017758],
            [-80.6888542,26.47943094],
            [-80.66261472,26.48062023],
            [-80.64501101,26.47140293],
            [-80.5389865,26.33573306],
            [-80.54198348,26.33350413],
            [-80.77568272,26.33197003],
            [-80.79271676,26.35535459],
            [-80.81055951,26.43631098]
        ]]),
        "start": "2020-01-01",
        "end": "2021-01-01"
    },
    {
        "name": "STA-3/4",
        "aoi": ee.Geometry.Polygon([[
            [-80.81055951,26.43631098],
            [-80.68733359,26.44017758],
            [-80.6888542,26.47943094],
            [-80.66261472,26.48062023],
            [-80.64501101,26.47140293],
            [-80.5389865,26.33573306],
            [-80.54198348,26.33350413],
            [-80.77568272,26.33197003],
            [-80.79271676,26.35535459],
            [-80.81055951,26.43631098]
        ]]),
        "start": "2021-01-01",
        "end": "2022-01-01"
    },
    {
        "name": "STA-3/4",
        "aoi": ee.Geometry.Polygon([[
            [-80.81055951,26.43631098],
            [-80.68733359,26.44017758],
            [-80.6888542,26.47943094],
            [-80.66261472,26.48062023],
            [-80.64501101,26.47140293],
            [-80.5389865,26.33573306],
            [-80.54198348,26.33350413],
            [-80.77568272,26.33197003],
            [-80.79271676,26.35535459],
            [-80.81055951,26.43631098]
        ]]),
        "start": "2022-01-01",
        "end": "2023-01-01"
    },
    {
        "name": "STA-3/4",
        "aoi": ee.Geometry.Polygon([[
            [-80.81055951,26.43631098],
            [-80.68733359,26.44017758],
            [-80.6888542,26.47943094],
            [-80.66261472,26.48062023],
            [-80.64501101,26.47140293],
            [-80.5389865,26.33573306],
            [-80.54198348,26.33350413],
            [-80.77568272,26.33197003],
            [-80.79271676,26.35535459],
            [-80.81055951,26.43631098]
        ]]),
        "start": "2023-01-01",
        "end": "2024-01-01"
    },
    {
        "name": "STA-3/4",
        "aoi": ee.Geometry.Polygon([[
            [-80.81055951,26.43631098],
            [-80.68733359,26.44017758],
            [-80.6888542,26.47943094],
            [-80.66261472,26.48062023],
            [-80.64501101,26.47140293],
            [-80.5389865,26.33573306],
            [-80.54198348,26.33350413],
            [-80.77568272,26.33197003],
            [-80.79271676,26.35535459],
            [-80.81055951,26.43631098]
        ]]),
        "start": "2024-01-01",
        "end": "2025-01-01"
    },
    {
        "name": "STA-5/6",
        "aoi": ee.Geometry.Polygon([[
            [-80.94783564,26.46274662],
            [-80.94748873,26.40605488],
            [-80.88007816,26.32885213],
            [-80.88198692,26.46306834],
            [-80.94783564,26.46274662]
        ]]),
        "start": "2020-01-01",
        "end": "2021-01-01",
        "planet_collection_id": 'projects/ee-eeamurciabb/assets/testSTAs'
    },
    {
        "name": "STA-5/6",
        "aoi": ee.Geometry.Polygon([[
            [-80.94783564,26.46274662],
            [-80.94748873,26.40605488],
            [-80.88007816,26.32885213],
            [-80.88198692,26.46306834],
            [-80.94783564,26.46274662]
        ]]),
        "start": "2021-01-01",
        "end": "2022-01-01",
        "planet_collection_id": 'projects/ee-eeamurciabb/assets/testSTAs'
    },

    {
        "name": "STA-5/6",
        "aoi": ee.Geometry.Polygon([[
            [-80.94783564,26.46274662],
            [-80.94748873,26.40605488],
            [-80.88007816,26.32885213],
            [-80.88198692,26.46306834],
            [-80.94783564,26.46274662]
        ]]),
        "start": "2022-01-01",
        "end": "2023-01-01",
        "planet_collection_id": 'projects/ee-eeamurciabb/assets/testSTAs'
    },
    {
        "name": "STA-5/6",
        "aoi": ee.Geometry.Polygon([[
            [-80.94783564,26.46274662],
            [-80.94748873,26.40605488],
            [-80.88007816,26.32885213],
            [-80.88198692,26.46306834],
            [-80.94783564,26.46274662]
        ]]),
        "start": "2023-01-01",
        "end": "2024-01-01",
        "planet_collection_id": 'projects/ee-eeamurciabb/assets/testSTAs'
    },
    {
        "name": "STA-5/6",
        "aoi": ee.Geometry.Polygon([[
            [-80.94783564,26.46274662],
            [-80.94748873,26.40605488],
            [-80.88007816,26.32885213],
            [-80.88198692,26.46306834],
            [-80.94783564,26.46274662]
        ]]),
        "start": "2024-01-01",
        "end": "2025-01-01",
        "planet_collection_id": 'projects/ee-eeamurciabb/assets/testSTAs'
    },                           

]

# # ---- Container to accumulate results ----
# batch_results = []


# for reg in regions:
#     print(f"\n🔹 Processing {reg['name']} ({reg['start']} → {reg['end']}) ...")
#     try:
#         planet_coll = reg.get("planet_collection_id", PLANET_COLLECTION_ID)
#         out = find_pairs(reg["aoi"], reg["start"], reg["end"], planet_coll)

#         # Extract IDs
#         planet_ids = out['planet_unmatched']
#         s2_ids = out['s2_unmatched']
#         matched = out['matched_pairs']

#         # Extract and format dates
#         planet_dates = [extract_date(i) for i in planet_ids]
#         s2_dates = [extract_date(i) for i in s2_ids]

#         paired_s2_dates     = [m['s2_date'] for m in matched]
#         paired_planet_dates = [m['planet_date'] for m in matched]

#         # Align list lengths (pad with blanks so we can zip)
#         rows = max(len(planet_dates), len(s2_dates), len(paired_s2_dates), len(paired_planet_dates))        
#         planet_dates += [''] * (rows - len(planet_dates))
#         s2_dates     += [''] * (rows - len(s2_dates))
#         paired_s2_dates += [''] * (rows - len(paired_s2_dates))
#         paired_planet_dates += [''] * (rows - len(paired_planet_dates))

#         # Add rows
#         for p, s, papl, pas2 in zip(planet_dates, s2_dates, paired_planet_dates, paired_s2_dates):
#             batch_results.append({
#                 "Region": reg["name"],
#                 "Start": reg["start"],
#                 "End": reg["end"],
#                 "Planet_date": p,
#                 "Sentinel_date": s,
#                 "Paired_planet_date": papl,
#                 "Paired_s2_dates": pas2
#             })

#         print(f"   ✅ Done: {len(out['matched_pairs'])} matched | "
#               f"{len(planet_ids)} Planet only | {len(s2_ids)} S2 only")

#     except Exception as e:
#         print(f"   ⚠️ Error in {reg['name']}: {e}")



# # ---- Convert all results to a single DataFrame ----
# df_all = pd.DataFrame(batch_results)
# print("\n📋 Combined Table:")
# print(df_all)

# # (optional) save to CSV
# outpath = "/home/eamurciab/planet_s2_batches_b.csv"  # fallback path
# try:
#     df_all.to_csv("/media/eamurciab/TOSHIBA_EXT1/PhD/SR/Auth/planet_s2_batches_b.csv", index=False)
# except OSError:
#     print("⚠️ External drive failed, saving to home directory instead...")
#     df_all.to_csv(outpath, index=False)
#     print(f"✅ Saved to {outpath}")




# for reg in regions:
#     planet_coll = reg.get("planet_collection_id", PLANET_COLLECTION_ID)
#     out = find_pairs(reg["aoi"], reg["start"], reg["end"], planet_coll)
    
#     matched = out['matched_pairs']
#     s2_best = out['s2_collection']
#     pl_best = out['planet_collection']
    
#     if len(matched) > 0:
#         # Get paired images
#         paired_imgs = get_paired_images_from_collections(
#             matched, s2_best, pl_best, aoi=reg["aoi"]
#         )
        
#         # Download as arrays
#         paired_arrays = download_paired_images_as_arrays(
#             paired_imgs, reg["aoi"], scale=10
#         )
        
#         # Process with patch-based harmonization
#         processed = process_paired_images(
#             paired_arrays,
#             harmonization_method='patched',  # or 'grid'
#             patch_size=256,
#             overlap=32,
#             r_bmatch=3,  # NIR band
#             s_bmatch=3,
#             max_shift=10,
#             verbose=True
#         )
        
#         # Save results
#         save_processed_results(
#             processed,
#             "/path/to/output",
#             f"{reg['name']}_{reg['start'][:4]}"
#         )

# Example usage for efficient pipeline integration:
# 
# import sys
# import os
# 
# # Get the absolute path of the current file's directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# 
# # Construct the path to the parent directory
# parent_dir = os.path.join(current_dir, '..')
# 
# # Add the parent directory to sys.path
# sys.path.insert(0, parent_dir)
# 
# from numpy_efficient_pipeline import TensorFlowFreeDataPipeline, NumpyTrainingDataGenerator
# 
# # Initialize pipeline (works in ANY Python environment)
# pipeline = TensorFlowFreeDataPipeline(
#     cache_dir="./sr_numpy_cache",
#     tile_size=256,  # Start small, adjust based on memory
#     overlap=32
# )
# 
# data_gen = NumpyTrainingDataGenerator(
#     pipeline=pipeline,
#     batch_size=2,  # Start small for testing
#     sequence_length=3,
#     normalize=True,
#     augment=True
# )