import ee
import pandas as pd
import re
from datetime import datetime, timedelta
from datetime import datetime as _dt
from datetime import datetime, timedelta

credentials = ee.ServiceAccountCredentials(
    'stas-50@remotesensingeduart.iam.gserviceaccount.com',
    '/media/eamurciab/TOSHIBA_EXT1/PhD/SR/Auth/remotesensingeduart-222d07dd6689.json'
)
ee.Initialize(credentials)

PLANET_COLLECTION_ID = 'projects/remotesensingeduart/assets/testSTAs'
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

# def make_tag_has_data(aoi, scale=50, tileScale=2):
#     def _fn(img):
#         # any non-zero qscore pixel inside AOI?
#         has = img.select('qscore').reduceRegion(
#             reducer=ee.Reducer.max(),
#             geometry=aoi,
#             scale=scale,
#             maxPixels=1e8,
#             tileScale=tileScale
#         ).get('qscore')
#         return img.set('has_data', has)
#     return _fn

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
    to_dt = lambda s: _dt.strptime(s, "%Y-%m-%d")

    s2_dates = sorted(s2_map.keys())
    pl_dates = sorted(pl_map.keys())
    used_pl = set()
    matched = []

    # DEBUG: show the first S2 date’s candidate count
    if s2_dates and pl_dates:
        s0 = s2_dates[0]
        cand0 = [pd for pd in pl_dates if abs((to_dt(pd) - to_dt(s0)).days) <= days_tol]
        print(f"[matcher] first S2 {s0} has {len(cand0)} PL candidates within ±{days_tol}d")

    for s2d in s2_dates:
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

    pl_only = [d for d in pl_dates if d not in used_pl]
    s2_only = [d for d in s2_dates if d not in {m[0] for m in matched}]

    # DEBUG: summarize
    print(f"[matcher] nn result: {len(matched)} matched, {len(pl_only)} PL-only, {len(s2_only)} S2-only")
    return matched, pl_only, s2_only

# --------- Main pairing function (uses scores) ----------
def find_pairs(aoi, start_date, end_date, planet_coll_id, planet_coll_id_fallback=None):

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
        .map(make_add_aoi_cloud_fraction(aoi))
        .filter(ee.Filter.notNull(['AOI_CLOUD_FRAC']))         # guard against null
        .filter(ee.Filter.lt('AOI_CLOUD_FRAC', 0.1))
        .map(_add_common_fields)
        .map(_add_s2_quality)
        .map(_add_s2_score)
        .map(_add_pref_id))
    s2_best = _best_per_day(s2)


    try:
        pl_prepped = (ee.ImageCollection(planet_coll_id)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .map(_add_common_fields)  
            .map(_add_planet_quality) 
            .map(_add_planet_score)
            .filter(ee.Filter.lt('q_cloud', 20))                 
            .map(make_add_udm_mask_and_score_for_aoi(aoi, conf_threshold=80))
            .map(_add_pref_id))
        
        # Test if collection has data
        size = pl_prepped.size().getInfo()
        if size == 0 and planet_coll_id_fallback:
            print(f"   Primary collection empty, trying fallback...")
            raise ValueError("Empty collection, try fallback")
    except Exception as e:
        if planet_coll_id_fallback:
            print(f"   Using fallback collection: {planet_coll_id_fallback}")
            pl_prepped = (ee.ImageCollection(planet_coll_id_fallback)
                .filterBounds(aoi)
                .filterDate(start_date, end_date)
                .map(_add_common_fields)  
                .map(_add_planet_quality) 
                .map(_add_planet_score)
                .filter(ee.Filter.lt('q_cloud', 20))                 
                .map(make_add_udm_mask_and_score_for_aoi(aoi, conf_threshold=80))
                .map(_add_pref_id))
        else:
            raise e
        
    # print('Planet raw (date-filtered) size:', ee.Number(pl_prepped.size()).getInfo())
    # first_pl = ee.Image(pl_prepped.first())
    
    # print('First Planet band names:', first_pl.bandNames().getInfo())

    # # After you compute pl_best but before filtering has_data:
    # test_best = planet_best_per_day(pl_prepped)
    # print('Planet best-per-day (unfiltered) size:', ee.Number(test_best.size()).getInfo())

    # # After you filter with has_data:
    # pl_best_dbg = test_best.map(make_tag_has_data(aoi, 50, 2))
    # print('Planet best-per-day with any AOI pixels:',
    #     ee.Number(pl_best_dbg.filter(ee.Filter.gt('has_data', 0)).size()).getInfo())

    pl_best = (planet_best_per_day_adaptive(pl_prepped, aoi)  # << CHANGED - now needs aoi
        .map(make_tag_has_data(aoi, scale=50, tileScale=2))
        .filter(ee.Filter.gt('has_data', 0))
    )

    # pl_best = (planet_best_per_day(pl_prepped)
    #     .map(make_tag_has_data(aoi, scale=50, tileScale=2))
    #     .filter(ee.Filter.gt('has_data', 0))
    #     )

    # pl_best = (pl_best
    #         .map(lambda im: add_valid_area_props(im, aoi, scale=50, tileScale=2))
    #         .filter(ee.Filter.gt('valid_px', 0)))           # drop AOI-empty images

    # Dates
    s2_dates = s2_best.aggregate_array('date_key').getInfo()
    s2_ids   = s2_best.aggregate_array('id_pref').getInfo()

    pl_dates = pl_best.aggregate_array('date_key').getInfo()
    pl_ids   = pl_best.aggregate_array('id_pref').getInfo()



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

    print('Potential matches within ±7d:', _count_near_hits(s2_dates, pl_dates, 7))

    def _to_dt(s): return _dt.strptime(s, "%Y-%m-%d")

    print("First S2 vs first PL:",
        s2_dates[0], pl_dates[0],
        "diff days =", (_to_dt(pl_dates[0]) - _to_dt(s2_dates[0])).days)

    # Show the 3 closest PL dates for the first 3 S2 dates
    for sd in s2_dates[:3]:
        ranked = sorted([(pd, abs((_to_dt(pd) - _to_dt(sd)).days)) for pd in pl_dates],
                        key=lambda x: x[1])[:3]
        print("Closest PL to", sd, "=>", ranked)


    print('S2 days:', len(s2_dates), 'range:', min(s2_dates), '→', max(s2_dates))
    print('PL days:', len(pl_dates), 'range:', (min(pl_dates) if pl_dates else '∅'),
        '→', (max(pl_dates) if pl_dates else '∅'))
    print('S2 sample:', s2_dates[:10])
    print('PL sample:', pl_dates[:10])

    s2_map = {d: i for d, i in zip(s2_dates, s2_ids)}
    pl_map = {d: i for d, i in zip(pl_dates, pl_ids)}

    # both_dates   = sorted(set(s2_map).intersection(pl_map))
    # planet_only_dates  = sorted(set(pl_map).difference(s2_map))
    # s2_only_dates      = sorted(set(s2_map).difference(pl_map))

    # matched = [{'date': d, 's2_id': s2_map[d], 'planet_id': pl_map[d]} for d in both_dates]

    matches, planet_only_dates, s2_only_dates = _match_dates_with_tolerance_nn(s2_map, pl_map, days_tol=7)
    print(f"NN matched: {len(matches)} | PL-only: {len(planet_only_dates)} | S2-only: {len(s2_only_dates)}")
    if matches[:5]:
        print("Sample pairs:", matches[:5])


    matched = [{
        's2_date': s2_d,
        'planet_date': pl_d,
        's2_id': s2_map[s2_d],
        'planet_id': pl_map[pl_d]
    } for s2_d, pl_d in matches]

    # Unmatched IDs (as before)
    planet_unmatched_ids = [pl_map[d] for d in planet_only_dates]
    s2_unmatched_ids     = [s2_map[d] for d in s2_only_dates]

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
        'unmatched_table': unmatched_table,          # << new
        'planet_unmatched_dates': planet_only_dates, # optional convenience
        's2_unmatched_dates': s2_only_dates          # optional convenience
    }


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

# ---- Container to accumulate results ----
batch_results = []


for reg in regions:
    print(f"\n🔹 Processing {reg['name']} ({reg['start']} → {reg['end']}) ...")
    try:
        planet_coll = reg.get("planet_collection_id", PLANET_COLLECTION_ID)
        out = find_pairs(reg["aoi"], reg["start"], reg["end"], planet_coll)

        # Extract IDs
        planet_ids = out['planet_unmatched']
        s2_ids = out['s2_unmatched']
        matched = out['matched_pairs']

        # Extract and format dates
        planet_dates = [extract_date(i) for i in planet_ids]
        s2_dates = [extract_date(i) for i in s2_ids]

        paired_s2_dates     = [m['s2_date'] for m in matched]
        paired_planet_dates = [m['planet_date'] for m in matched]

        # Align list lengths (pad with blanks so we can zip)
        rows = max(len(planet_dates), len(s2_dates), len(paired_s2_dates), len(paired_planet_dates))        
        planet_dates += [''] * (rows - len(planet_dates))
        s2_dates     += [''] * (rows - len(s2_dates))
        paired_s2_dates += [''] * (rows - len(paired_s2_dates))
        paired_planet_dates += [''] * (rows - len(paired_planet_dates))

        # Add rows
        for p, s, papl, pas2 in zip(planet_dates, s2_dates, paired_planet_dates, paired_s2_dates):
            batch_results.append({
                "Region": reg["name"],
                "Start": reg["start"],
                "End": reg["end"],
                "Planet_date": p,
                "Sentinel_date": s,
                "Paired_planet_date": papl,
                "Paired_s2_dates": pas2
            })

        print(f"   ✅ Done: {len(out['matched_pairs'])} matched | "
              f"{len(planet_ids)} Planet only | {len(s2_ids)} S2 only")

    except Exception as e:
        print(f"   ⚠️ Error in {reg['name']}: {e}")



# ---- Convert all results to a single DataFrame ----
df_all = pd.DataFrame(batch_results)
print("\n📋 Combined Table:")
print(df_all)

# (optional) save to CSV
outpath = "/home/eamurciab/planet_s2_batches_b.csv"  # fallback path
try:
    df_all.to_csv("/media/eamurciab/TOSHIBA_EXT1/PhD/SR/Auth/planet_s2_batches_b.csv", index=False)
except OSError:
    print("⚠️ External drive failed, saving to home directory instead...")
    df_all.to_csv(outpath, index=False)
    print(f"✅ Saved to {outpath}")