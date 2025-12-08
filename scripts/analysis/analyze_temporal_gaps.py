#!/usr/bin/env python3
"""
Analyze temporal gaps in generated sequences and identify what additional
PlanetScope acquisition dates would help create more temporally coherent sequences.

Goal: Sequences with < 30 days between consecutive frames for better temporal learning.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pickle
from pathlib import Path
from datetime import datetime as dt, timedelta
from collections import defaultdict
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

# Define TrainingSample for unpickling
@dataclass
class TrainingSample:
    """Training sample with temporal sequences - needed for unpickling"""
    lr_sequence: np.ndarray
    hr_sequence: np.ndarray
    temporal_dates: List[str]
    planet_dates: List[str]
    region_name: str
    sequence_id: int = 0
    tile_id: int = 0
    tile_bounds: Dict = None
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None

# Import for getting available images
from Auth.AuthGEE import find_pairs, PLANET_COLLECTION_ID, regions


def analyze_sequences(dataset_path: str):
    """Analyze temporal gaps in generated sequences."""
    
    print("=" * 80)
    print("TEMPORAL GAP ANALYSIS FOR SR TRAINING SEQUENCES")
    print("=" * 80)
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    samples = dataset['samples']
    print(f"   Total samples: {len(samples)}")
    
    # Analyze temporal gaps
    all_gaps = []
    large_gap_sequences = []
    sequences_by_dates = defaultdict(list)
    
    for sample in samples:
        # Get dates for this sequence
        s2_dates = sample.temporal_dates  # Sentinel-2 dates
        planet_dates = sample.planet_dates  # Planet dates
        
        # Use the key dates (either works, they're matched)
        dates_key = tuple(sorted(s2_dates))
        sequences_by_dates[dates_key].append(sample)
        
        # Calculate gaps between consecutive frames
        for i in range(len(planet_dates) - 1):
            date1 = dt.strptime(planet_dates[i], '%Y-%m-%d')
            date2 = dt.strptime(planet_dates[i+1], '%Y-%m-%d')
            gap_days = abs((date2 - date1).days)
            all_gaps.append({
                'sequence_id': sample.sequence_id,
                'tile_id': sample.tile_id,
                'frame1_date': planet_dates[i],
                'frame2_date': planet_dates[i+1],
                'gap_days': gap_days,
                's2_dates': s2_dates,
                'planet_dates': planet_dates
            })
            
            if gap_days > 30:
                large_gap_sequences.append({
                    'sequence_id': sample.sequence_id,
                    'planet_dates': planet_dates,
                    's2_dates': s2_dates,
                    'gap_days': gap_days,
                    'gap_between': f"{planet_dates[i]} → {planet_dates[i+1]}"
                })
    
    # Summary statistics
    gaps_df = pd.DataFrame(all_gaps)
    
    print(f"\n📊 GAP STATISTICS:")
    print("-" * 80)
    print(f"   Total gaps analyzed: {len(gaps_df)}")
    print(f"   Mean gap: {gaps_df['gap_days'].mean():.1f} days")
    print(f"   Median gap: {gaps_df['gap_days'].median():.1f} days")
    print(f"   Min gap: {gaps_df['gap_days'].min()} days")
    print(f"   Max gap: {gaps_df['gap_days'].max()} days")
    print(f"   Std dev: {gaps_df['gap_days'].std():.1f} days")
    
    # Distribution
    print(f"\n📈 GAP DISTRIBUTION:")
    print("-" * 80)
    bins = [0, 7, 14, 21, 30, 60, 90, 180, 365]
    for i in range(len(bins)-1):
        count = len(gaps_df[(gaps_df['gap_days'] >= bins[i]) & (gaps_df['gap_days'] < bins[i+1])])
        pct = 100 * count / len(gaps_df)
        bar = '█' * int(pct / 2)
        print(f"   {bins[i]:3d}-{bins[i+1]:3d} days: {count:5d} ({pct:5.1f}%) {bar}")
    
    # Large gaps (> 30 days)
    large_gaps = gaps_df[gaps_df['gap_days'] > 30]
    print(f"\n⚠️  LARGE GAPS (>30 days): {len(large_gaps)} gaps")
    print("-" * 80)
    
    if len(large_gaps) > 0:
        # Group by unique gap
        unique_large_gaps = large_gaps.drop_duplicates(subset=['frame1_date', 'frame2_date'])
        print(f"   Unique large gaps: {len(unique_large_gaps)}")
        
        for _, row in unique_large_gaps.iterrows():
            print(f"   • {row['frame1_date']} → {row['frame2_date']}: {row['gap_days']} days")
    
    return gaps_df, sequences_by_dates


def identify_needed_dates(gaps_df, pair_results):
    """Identify what additional PlanetScope dates would fill large gaps."""
    
    print(f"\n" + "=" * 80)
    print("RECOMMENDED ADDITIONAL PLANETSCOPE ACQUISITIONS")
    print("=" * 80)
    
    # Get current Planet dates
    available_planet_dates = set(pair_results.get('planet_all_dates', []))
    matched_planet_dates = set(p['planet_date'] for p in pair_results.get('matched_pairs', []))
    
    print(f"\n📅 Current inventory:")
    print(f"   Available Planet dates: {len(available_planet_dates)}")
    print(f"   Matched Planet dates: {len(matched_planet_dates)}")
    
    # Find large gaps
    large_gaps = gaps_df[gaps_df['gap_days'] > 30].drop_duplicates(
        subset=['frame1_date', 'frame2_date']
    )
    
    recommendations = []
    
    print(f"\n🎯 ANALYSIS OF EACH LARGE GAP:")
    print("-" * 80)
    
    for _, row in large_gaps.iterrows():
        date1 = dt.strptime(row['frame1_date'], '%Y-%m-%d')
        date2 = dt.strptime(row['frame2_date'], '%Y-%m-%d')
        gap_days = row['gap_days']
        
        # Calculate ideal intermediate dates (every ~15 days)
        num_needed = max(1, gap_days // 20)  # Need a date every ~20 days
        
        print(f"\n   Gap: {row['frame1_date']} → {row['frame2_date']} ({gap_days} days)")
        
        # Check if there are any existing Planet dates in this gap
        existing_in_gap = []
        for pdate_str in available_planet_dates:
            pdate = dt.strptime(pdate_str, '%Y-%m-%d')
            if date1 < pdate < date2:
                existing_in_gap.append(pdate_str)
        
        if existing_in_gap:
            print(f"   📍 Existing Planet dates in gap (NOT MATCHED): {existing_in_gap}")
            print(f"      → These could be used if matching criteria are relaxed")
        
        # Suggest ideal dates
        ideal_dates = []
        for i in range(1, num_needed + 1):
            ideal_date = date1 + timedelta(days=int(gap_days * i / (num_needed + 1)))
            ideal_dates.append(ideal_date.strftime('%Y-%m-%d'))
        
        print(f"   🎯 Ideal acquisition windows:")
        for ideal in ideal_dates:
            ideal_dt = dt.strptime(ideal, '%Y-%m-%d')
            window_start = (ideal_dt - timedelta(days=7)).strftime('%Y-%m-%d')
            window_end = (ideal_dt + timedelta(days=7)).strftime('%Y-%m-%d')
            print(f"      • Target: {ideal} (window: {window_start} to {window_end})")
            
            recommendations.append({
                'gap_start': row['frame1_date'],
                'gap_end': row['frame2_date'],
                'gap_days': gap_days,
                'target_date': ideal,
                'window_start': window_start,
                'window_end': window_end,
                'existing_in_gap': str(existing_in_gap)  # Convert list to string for CSV
            })
    
    import pandas
    return pandas.DataFrame(recommendations)


def main():
    # Paths
    dataset_path = Path('/media/eamurciab/TOSHIBA_EXT1/PhD/SR/data/datasets/sta2_1000samples_parallel/dataset.pkl')
    output_dir = Path('/media/eamurciab/TOSHIBA_EXT1/PhD/SR/data/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Checking for 400-sample dataset...")
        dataset_path = Path('/media/eamurciab/TOSHIBA_EXT1/PhD/SR/data/datasets/sta2_400samples_parallel/dataset.pkl')
        if not dataset_path.exists():
            print(f"❌ No dataset found!")
            return
    
    # Analyze sequences
    gaps_df, sequences_by_dates = analyze_sequences(str(dataset_path))
    
    # Get available images from GEE
    print(f"\n🔍 Fetching available images from GEE...")
    region = regions[0]
    pair_results = find_pairs(
        region['aoi'],
        region['start'],
        region['end'],
        PLANET_COLLECTION_ID
    )
    
    # Identify needed dates
    recommendations = identify_needed_dates(gaps_df, pair_results)
    
    # Save results
    print(f"\n💾 SAVING RESULTS:")
    print("-" * 80)
    
    # Save gaps analysis
    gaps_csv = output_dir / 'temporal_gaps_analysis.csv'
    gaps_df.to_csv(gaps_csv, index=False)
    print(f"   ✅ Gaps analysis: {gaps_csv}")
    
    # Save recommendations
    if len(recommendations) > 0:
        rec_csv = output_dir / 'recommended_acquisitions.csv'
        recommendations.to_csv(rec_csv, index=False)
        print(f"   ✅ Recommendations: {rec_csv}")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    good_gaps = len(gaps_df[gaps_df['gap_days'] <= 30])
    total_gaps = len(gaps_df)
    print(f"   Gaps ≤ 30 days: {good_gaps}/{total_gaps} ({100*good_gaps/total_gaps:.1f}%)")
    print(f"   Gaps > 30 days: {total_gaps - good_gaps}/{total_gaps} ({100*(total_gaps-good_gaps)/total_gaps:.1f}%)")
    
    if len(recommendations) > 0:
        print(f"\n   📋 {len(recommendations)} additional acquisition windows recommended")
        print(f"   📋 See: {rec_csv}")


if __name__ == '__main__':
    main()
