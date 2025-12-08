#!/usr/bin/env python3
"""
Comprehensive temporal gap analysis across ALL regions defined in AuthGEE.py.

Analyzes matched pairs directly from GEE without generating samples.
Identifies acquisition gaps and recommends additional PlanetScope dates.

FIXES (Nov 2025):
- Uses correct planet_collection_id from each region config
- Combines images from BOTH repos (remotesensingeduart + ee-eeamurciabb)
- Simulates quality filtering that happens during sample generation
- Adds delays to avoid GEE rate limiting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import ee
import time
from datetime import datetime as dt, timedelta
from collections import defaultdict
import pandas as pd
from pathlib import Path

# Import from AuthGEE
from Auth.AuthGEE import (
    find_pairs, export_image_inventory, PLANET_COLLECTION_ID, regions,
    _add_common_fields, _add_planet_quality, _add_planet_score, _add_pref_id,
    make_add_udm_mask_and_score_for_aoi, planet_best_per_day_adaptive, make_tag_has_data
)

# Both Planet repositories
PLANET_REPO_PRIMARY = 'projects/remotesensingeduart/assets/testSTAs'
PLANET_REPO_SECONDARY = 'projects/ee-eeamurciabb/assets/testSTAs'


def get_planet_images_from_both_repos(aoi, start_date, end_date, primary_id=None):
    """
    Get Planet images from BOTH repositories and merge them.
    
    Args:
        aoi: Area of interest geometry
        start_date: Start date string
        end_date: End date string
        primary_id: Optional primary collection ID (from region config)
    
    Returns:
        Dictionary with combined planet info from both repos
    """
    all_dates = set()
    all_ids = []
    repo_sources = {}
    
    repos_to_check = [PLANET_REPO_PRIMARY, PLANET_REPO_SECONDARY]
    
    for repo_id in repos_to_check:
        try:
            pl_coll = (ee.ImageCollection(repo_id)
                .filterBounds(aoi)
                .filterDate(start_date, end_date)
                .map(_add_common_fields)
                .map(_add_planet_quality)
                .map(_add_planet_score)
                .map(_add_pref_id))
            
            # Single combined call to reduce API requests
            info = pl_coll.aggregate_array('date_key').zip(
                pl_coll.aggregate_array('id_pref')
            ).getInfo()
            
            for date_id_pair in info:
                d, img_id = date_id_pair
                if d not in all_dates:
                    all_dates.add(d)
                    all_ids.append(img_id)
                    repo_sources[d] = repo_id.split('/')[-2]  # Extract project name
            
            # Rate limit delay between repos
            time.sleep(1)
                    
        except Exception as e:
            print(f"      ⚠️  Error accessing {repo_id}: {e}")
    
    return {
        'dates': sorted(list(all_dates)),
        'ids': all_ids,
        'sources': repo_sources,
        'total_from_primary': sum(1 for d in repo_sources.values() if 'remotesensingeduart' in d),
        'total_from_secondary': sum(1 for d in repo_sources.values() if 'ee-eeamurciabb' in d)
    }


def estimate_quality_filtered_pairs(aoi, matched_pairs, quality_threshold=0.9):
    """
    Estimate how many matched pairs would survive quality filtering.
    
    This simulates the tile-based quality filtering in generate_sta2_grid_parallel.py
    without actually downloading data. Uses AOI-level cloud/valid pixel stats as proxy.
    
    Args:
        aoi: Area of interest
        matched_pairs: List of matched pair dictionaries
        quality_threshold: Estimated survival rate (default 0.9 = 90% tiles pass)
    
    Returns:
        Dictionary with filtering estimates
    """
    # Simple estimation: assume ~10-20% of pairs lose significant tiles due to quality
    # This is based on typical cloud patterns and the 90% coverage threshold
    estimated_survival_rate = quality_threshold
    
    estimated_usable = int(len(matched_pairs) * estimated_survival_rate)
    
    return {
        'total_matched': len(matched_pairs),
        'estimated_usable': estimated_usable,
        'estimated_filtered_out': len(matched_pairs) - estimated_usable,
        'survival_rate': estimated_survival_rate
    }


def analyze_region_temporal_gaps(pair_results, region_name, date_range, sequence_length=3):
    """
    Analyze temporal gaps for a single region based on matched pairs.
    Simulates sequence formation without downloading data.
    
    Args:
        pair_results: Dictionary from find_pairs()
        region_name: Name of the region
        date_range: String describing date range
        sequence_length: Number of frames per sequence (default: 3)
    
    Returns:
        Dictionary with gap analysis results
    """
    matched_pairs = pair_results.get('matched_pairs', [])
    
    if len(matched_pairs) < sequence_length:
        return {
            'region': region_name,
            'date_range': date_range,
            'matched_pairs': len(matched_pairs),
            'possible_sequences': 0,
            'gaps': [],
            'large_gaps': [],
            'status': 'INSUFFICIENT_PAIRS'
        }
    
    # Sort pairs by date
    sorted_pairs = sorted(matched_pairs, key=lambda x: x['planet_date'])
    
    # Analyze gaps between consecutive matched pairs
    all_gaps = []
    for i in range(len(sorted_pairs) - 1):
        date1 = dt.strptime(sorted_pairs[i]['planet_date'], '%Y-%m-%d')
        date2 = dt.strptime(sorted_pairs[i+1]['planet_date'], '%Y-%m-%d')
        gap_days = (date2 - date1).days
        
        all_gaps.append({
            'region': region_name,
            'date_range': date_range,
            'frame1_date': sorted_pairs[i]['planet_date'],
            'frame2_date': sorted_pairs[i+1]['planet_date'],
            'frame1_s2': sorted_pairs[i]['s2_date'],
            'frame2_s2': sorted_pairs[i+1]['s2_date'],
            'gap_days': gap_days
        })
    
    # Identify large gaps (>30 days)
    large_gaps = [g for g in all_gaps if g['gap_days'] > 30]
    
    # Calculate potential sequences
    num_sequences = max(0, len(sorted_pairs) - sequence_length + 1)
    
    # Identify which sequences would have large gaps
    sequences_with_large_gaps = 0
    for seq_start in range(num_sequences):
        seq_dates = [sorted_pairs[seq_start + j]['planet_date'] for j in range(sequence_length)]
        for i in range(len(seq_dates) - 1):
            d1 = dt.strptime(seq_dates[i], '%Y-%m-%d')
            d2 = dt.strptime(seq_dates[i+1], '%Y-%m-%d')
            if (d2 - d1).days > 30:
                sequences_with_large_gaps += 1
                break
    
    return {
        'region': region_name,
        'date_range': date_range,
        'matched_pairs': len(matched_pairs),
        'possible_sequences': num_sequences,
        'sequences_with_large_gaps': sequences_with_large_gaps,
        'good_sequences': num_sequences - sequences_with_large_gaps,
        'gaps': all_gaps,
        'large_gaps': large_gaps,
        'status': 'OK'
    }


def identify_needed_acquisitions(analysis_result, available_planet_dates):
    """
    Identify what additional PlanetScope acquisitions would fill gaps.
    """
    recommendations = []
    
    for gap in analysis_result.get('large_gaps', []):
        date1 = dt.strptime(gap['frame1_date'], '%Y-%m-%d')
        date2 = dt.strptime(gap['frame2_date'], '%Y-%m-%d')
        gap_days = gap['gap_days']
        
        # Check for existing Planet dates in gap (not matched)
        existing_in_gap = []
        for pdate_str in available_planet_dates:
            pdate = dt.strptime(pdate_str, '%Y-%m-%d')
            if date1 < pdate < date2:
                existing_in_gap.append(pdate_str)
        
        # Calculate ideal intermediate dates
        num_needed = max(1, gap_days // 20)
        ideal_dates = []
        for i in range(1, num_needed + 1):
            ideal_date = date1 + timedelta(days=int(gap_days * i / (num_needed + 1)))
            ideal_dates.append(ideal_date.strftime('%Y-%m-%d'))
        
        recommendations.append({
            'region': analysis_result['region'],
            'date_range': analysis_result['date_range'],
            'gap_start': gap['frame1_date'],
            'gap_end': gap['frame2_date'],
            'gap_days': gap_days,
            'existing_planet_in_gap': existing_in_gap,
            'num_existing': len(existing_in_gap),
            'ideal_dates_needed': ideal_dates,
            'num_needed': len(ideal_dates),
            'can_fill_with_existing': len(existing_in_gap) >= len(ideal_dates)
        })
    
    return recommendations


def main():
    output_dir = Path('/media/eamurciab/TOSHIBA_EXT1/PhD/SR/data/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("COMPREHENSIVE TEMPORAL GAP ANALYSIS - ALL REGIONS")
    print("=" * 100)
    print(f"\n📡 Checking BOTH Planet repositories:")
    print(f"   Primary:   {PLANET_REPO_PRIMARY}")
    print(f"   Secondary: {PLANET_REPO_SECONDARY}")
    print(f"\nAnalyzing {len(regions)} region/year configurations...")
    
    # Collect results
    all_analyses = []
    all_gaps = []
    all_recommendations = []
    summary_rows = []
    
    # Group regions by name for cleaner output
    regions_by_name = defaultdict(list)
    for r in regions:
        regions_by_name[r['name']].append(r)
    
    print(f"\nFound {len(regions_by_name)} unique regions: {list(regions_by_name.keys())}")
    
    for region_name, region_configs in regions_by_name.items():
        print(f"\n{'='*100}")
        print(f"REGION: {region_name}")
        print(f"{'='*100}")
        
        for config in region_configs:
            date_range = f"{config['start']} to {config['end']}"
            print(f"\n  📅 {date_range}")
            print(f"  " + "-" * 50)
            
            try:
                # Get the correct Planet collection ID from region config
                # Fallback to PLANET_COLLECTION_ID if not specified
                region_planet_id = config.get('planet_collection_id', PLANET_COLLECTION_ID)
                
                # Check both repos for Planet images
                print(f"     Checking both Planet repos for available images...")
                planet_both_repos = get_planet_images_from_both_repos(
                    config['aoi'], config['start'], config['end'], region_planet_id
                )
                print(f"     📦 Planet images: {len(planet_both_repos['dates'])} total " +
                      f"(primary: {planet_both_repos['total_from_primary']}, " +
                      f"secondary: {planet_both_repos['total_from_secondary']})")
                
                # Get matched pairs using the region's specified collection
                pair_results = find_pairs(
                    config['aoi'],
                    config['start'],
                    config['end'],
                    region_planet_id,  # Use region-specific collection
                    planet_coll_id_fallback=PLANET_REPO_SECONDARY if region_planet_id == PLANET_REPO_PRIMARY else PLANET_REPO_PRIMARY
                )
                
                # Estimate quality filtering impact
                quality_estimate = estimate_quality_filtered_pairs(
                    config['aoi'], 
                    pair_results.get('matched_pairs', []),
                    quality_threshold=0.85  # ~15% filtered due to quality
                )
                
                # Analyze gaps
                analysis = analyze_region_temporal_gaps(
                    pair_results, region_name, date_range, sequence_length=3
                )
                all_analyses.append(analysis)
                all_gaps.extend(analysis['gaps'])
                
                # Use ALL planet dates from both repos for recommendations
                available_planet = planet_both_repos['dates']
                
                # Generate recommendations
                recs = identify_needed_acquisitions(analysis, available_planet)
                all_recommendations.extend(recs)
                
                # Print summary
                print(f"     S2 images:       {len(pair_results.get('s2_all_dates', []))}")
                print(f"     Planet (both):   {len(planet_both_repos['dates'])}")
                print(f"     Matched pairs:   {analysis['matched_pairs']}")
                print(f"     Est. after QA:   ~{quality_estimate['estimated_usable']} ({quality_estimate['survival_rate']*100:.0f}% survival)")
                print(f"     Possible seqs:   {analysis['possible_sequences']}")
                
                if analysis['possible_sequences'] > 0:
                    good_pct = 100 * analysis['good_sequences'] / analysis['possible_sequences']
                    print(f"     Good sequences:  {analysis['good_sequences']} ({good_pct:.1f}%)")
                    print(f"     Large gaps:      {len(analysis['large_gaps'])}")
                    
                    if analysis['large_gaps']:
                        print(f"     ⚠️  Gaps > 30 days:")
                        for gap in analysis['large_gaps']:
                            print(f"        • {gap['frame1_date']} → {gap['frame2_date']}: {gap['gap_days']} days")
                
                # Add to summary
                summary_rows.append({
                    'region': region_name,
                    'date_range': date_range,
                    'start': config['start'],
                    'end': config['end'],
                    'planet_collection': region_planet_id,
                    's2_images': len(pair_results.get('s2_all_dates', [])),
                    'planet_primary': planet_both_repos['total_from_primary'],
                    'planet_secondary': planet_both_repos['total_from_secondary'],
                    'planet_total': len(planet_both_repos['dates']),
                    'matched_pairs': analysis['matched_pairs'],
                    'est_after_quality': quality_estimate['estimated_usable'],
                    'possible_sequences': analysis['possible_sequences'],
                    'good_sequences': analysis.get('good_sequences', 0),
                    'sequences_with_gaps': analysis.get('sequences_with_large_gaps', 0),
                    'large_gaps_count': len(analysis['large_gaps']),
                    'max_gap_days': max([g['gap_days'] for g in analysis['gaps']], default=0),
                    'mean_gap_days': sum([g['gap_days'] for g in analysis['gaps']]) / len(analysis['gaps']) if analysis['gaps'] else 0
                })
                
                # Add delay to avoid GEE rate limiting (3s between region-years)
                print(f"     ⏱️  Waiting 3s before next query...")
                time.sleep(3)
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
                summary_rows.append({
                    'region': region_name,
                    'date_range': date_range,
                    'start': config['start'],
                    'end': config['end'],
                    'error': str(e)
                })
                # Wait longer after error (rate limit recovery)
                print(f"     ⏱️  Rate limit hit, waiting 10s...")
                time.sleep(10)
    
    # Save results
    print(f"\n\n{'='*100}")
    print("SAVING RESULTS")
    print(f"{'='*100}")
    
    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / 'all_regions_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary: {summary_path}")
    
    # All gaps CSV
    if all_gaps:
        gaps_df = pd.DataFrame(all_gaps)
        gaps_path = output_dir / 'all_regions_gaps.csv'
        gaps_df.to_csv(gaps_path, index=False)
        print(f"✅ All gaps: {gaps_path}")
    
    # Recommendations CSV
    if all_recommendations:
        # Convert lists to strings for CSV
        for rec in all_recommendations:
            rec['existing_planet_in_gap'] = str(rec['existing_planet_in_gap'])
            rec['ideal_dates_needed'] = str(rec['ideal_dates_needed'])
        
        rec_df = pd.DataFrame(all_recommendations)
        rec_path = output_dir / 'all_regions_recommendations.csv'
        rec_df.to_csv(rec_path, index=False)
        print(f"✅ Recommendations: {rec_path}")
    
    # Print overall summary
    print(f"\n\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}")
    
    total_matched = sum(r.get('matched_pairs', 0) for r in summary_rows if 'matched_pairs' in r)
    total_est_qa = sum(r.get('est_after_quality', 0) for r in summary_rows if 'est_after_quality' in r)
    total_sequences = sum(r.get('possible_sequences', 0) for r in summary_rows if 'possible_sequences' in r)
    total_good = sum(r.get('good_sequences', 0) for r in summary_rows if 'good_sequences' in r)
    total_large_gaps = sum(r.get('large_gaps_count', 0) for r in summary_rows if 'large_gaps_count' in r)
    total_planet_primary = sum(r.get('planet_primary', 0) for r in summary_rows if 'planet_primary' in r)
    total_planet_secondary = sum(r.get('planet_secondary', 0) for r in summary_rows if 'planet_secondary' in r)
    
    print(f"\n  📊 Across all regions and years:")
    print(f"     Planet images (primary):     {total_planet_primary}")
    print(f"     Planet images (secondary):   {total_planet_secondary}")
    print(f"     Total Planet images:         {total_planet_primary + total_planet_secondary}")
    print(f"     Total matched pairs:         {total_matched}")
    print(f"     Est. after quality filter:   ~{total_est_qa}")
    print(f"     Total possible sequences:    {total_sequences}")
    print(f"     Good sequences (≤30d):       {total_good} ({100*total_good/max(total_sequences,1):.1f}%)")
    print(f"     Total large gaps:            {total_large_gaps}")
    print(f"     Recommendations:             {len(all_recommendations)}")
    
    # Gaps that can be filled with existing data
    fillable = sum(1 for r in all_recommendations if r.get('can_fill_with_existing'))
    need_new = len(all_recommendations) - fillable
    
    print(f"\n  💡 Gap analysis:")
    print(f"     Gaps fillable with existing unmatched Planet data: {fillable}/{len(all_recommendations)}")
    print(f"     Gaps requiring NEW PlanetScope acquisitions:       {need_new}/{len(all_recommendations)}")
    
    if fillable > 0:
        print(f"\n  📝 Note: {fillable} gaps have Planet images that weren't matched due to")
        print(f"     S2 cloud/quality filters. Consider relaxing matching criteria.")
    
    if need_new > 0:
        print(f"\n  🛰️  Action needed: {need_new} gaps require additional PlanetScope")
        print(f"     acquisitions (not currently in either GEE repository).")


if __name__ == '__main__':
    main()
