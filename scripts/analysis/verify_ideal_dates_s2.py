#!/usr/bin/env python3
"""
Verify that Sentinel-2 images exist and pass quality filters
on/near the 'ideal_dates_needed' for new PlanetScope acquisitions.

This helps confirm that acquiring new Planet images on those dates
would actually result in usable training pairs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import ee
import pandas as pd
from datetime import datetime as dt, timedelta
from pathlib import Path

from Auth.AuthGEE import (
    regions, _add_common_fields, make_add_aoi_cloud_fraction
)

# Initialize GEE
try:
    ee.Initialize()
except:
    from Auth.AuthGEE import initialize_gee
    initialize_gee()


def check_s2_availability(aoi, target_date_str, tolerance_days=7):
    """
    Check if good-quality S2 images exist within ±tolerance_days of target date.
    
    Returns:
        dict with S2 availability info
    """
    target_date = dt.strptime(target_date_str, '%Y-%m-%d')
    start = (target_date - timedelta(days=tolerance_days)).strftime('%Y-%m-%d')
    end = (target_date + timedelta(days=tolerance_days + 1)).strftime('%Y-%m-%d')
    
    # Get S2 images with quality info (before filtering)
    s2_all = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start, end)
        .map(_add_common_fields)
        .map(make_add_aoi_cloud_fraction(aoi)))
    
    # Get info
    try:
        info = s2_all.aggregate_array('date_key').zip(
            s2_all.aggregate_array('CLOUDY_PIXEL_PERCENTAGE')
        ).zip(
            s2_all.aggregate_array('AOI_CLOUD_FRAC')
        ).getInfo()
        
        results = []
        for item in info:
            (date_key, cloud_pct), aoi_cloud = item
            # Check if passes quality filters
            passes_cloud = cloud_pct < 60
            passes_aoi = aoi_cloud is not None and aoi_cloud < 0.1
            passes_all = passes_cloud and passes_aoi
            
            results.append({
                'date': date_key,
                'cloud_pct': cloud_pct,
                'aoi_cloud_frac': aoi_cloud,
                'passes_cloud_filter': passes_cloud,
                'passes_aoi_filter': passes_aoi,
                'passes_all': passes_all
            })
        
        return {
            'target_date': target_date_str,
            's2_found': len(results),
            's2_passing': sum(1 for r in results if r['passes_all']),
            'details': results
        }
    except Exception as e:
        return {
            'target_date': target_date_str,
            'error': str(e),
            's2_found': 0,
            's2_passing': 0,
            'details': []
        }


def main():
    # Load recommendations
    rec_path = Path('/media/eamurciab/TOSHIBA_EXT1/PhD/SR/data/analysis/all_regions_recommendations.csv')
    
    if not rec_path.exists():
        print("❌ Recommendations file not found. Run analyze_all_regions_gaps.py first.")
        return
    
    rec_df = pd.read_csv(rec_path)
    
    print("=" * 100)
    print("VERIFYING S2 AVAILABILITY FOR IDEAL PLANETSCOPE ACQUISITION DATES")
    print("=" * 100)
    
    # Build region lookup by name
    region_lookup = {}
    for r in regions:
        if r['name'] not in region_lookup:
            region_lookup[r['name']] = r['aoi']
    
    # Collect results
    verification_results = []
    
    # Process only gaps that need NEW acquisitions (can_fill_with_existing == False)
    needs_new = rec_df[rec_df['can_fill_with_existing'] == False].copy()
    
    print(f"\n📋 Found {len(needs_new)} gaps that need NEW PlanetScope acquisitions")
    print(f"   (Skipping {len(rec_df) - len(needs_new)} gaps that can be filled with existing data)\n")
    
    for idx, row in needs_new.iterrows():
        region_name = row['region']
        gap_start = row['gap_start']
        gap_end = row['gap_end']
        
        # Parse ideal dates (stored as string representation of list)
        ideal_dates_str = row['ideal_dates_needed']
        ideal_dates = eval(ideal_dates_str) if isinstance(ideal_dates_str, str) else []
        
        if not ideal_dates:
            continue
            
        aoi = region_lookup.get(region_name)
        if aoi is None:
            print(f"⚠️  Region '{region_name}' not found in lookup")
            continue
        
        print(f"\n🔍 {region_name}: Gap {gap_start} → {gap_end} ({row['gap_days']} days)")
        print(f"   Checking {len(ideal_dates)} ideal dates: {ideal_dates}")
        
        for ideal_date in ideal_dates:
            result = check_s2_availability(aoi, ideal_date, tolerance_days=7)
            
            status = "✅" if result['s2_passing'] > 0 else "❌"
            print(f"   {status} {ideal_date}: {result['s2_passing']}/{result['s2_found']} S2 images pass quality")
            
            if result['details']:
                for d in result['details']:
                    flag = "✓" if d['passes_all'] else "✗"
                    print(f"      {flag} {d['date']}: cloud={d['cloud_pct']:.1f}%, aoi_frac={d['aoi_cloud_frac']}")
            
            verification_results.append({
                'region': region_name,
                'gap_start': gap_start,
                'gap_end': gap_end,
                'gap_days': row['gap_days'],
                'ideal_date': ideal_date,
                's2_found_within_7d': result['s2_found'],
                's2_passing_quality': result['s2_passing'],
                'would_be_useful': result['s2_passing'] > 0
            })
    
    # Save results
    output_path = Path('/media/eamurciab/TOSHIBA_EXT1/PhD/SR/data/analysis/ideal_dates_s2_verification.csv')
    ver_df = pd.DataFrame(verification_results)
    ver_df.to_csv(output_path, index=False)
    
    # Summary
    print(f"\n\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    
    total_ideal = len(verification_results)
    useful = sum(1 for r in verification_results if r['would_be_useful'])
    
    print(f"\n📊 Out of {total_ideal} ideal acquisition dates:")
    print(f"   ✅ {useful} have good S2 within ±7 days (WORTH ACQUIRING)")
    print(f"   ❌ {total_ideal - useful} have NO good S2 nearby (not useful)")
    
    print(f"\n💾 Detailed results saved to: {output_path}")
    
    # List the useful dates
    if useful > 0:
        print(f"\n🛰️  RECOMMENDED NEW PLANETSCOPE ACQUISITIONS:")
        useful_dates = ver_df[ver_df['would_be_useful'] == True]
        for _, row in useful_dates.iterrows():
            print(f"   • {row['region']}: {row['ideal_date']} (fills {row['gap_days']}-day gap)")


if __name__ == '__main__':
    main()
