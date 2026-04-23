#!/usr/bin/env python3
"""
Add Quality Metadata to Existing Dataset

This script retroactively computes and adds quality statistics to training samples
that don't have frame_quality metadata populated.

Quality metrics computed per frame:
- zero_fraction: Fraction of zero/masked pixels (from cloud/shadow masking)
- nan_fraction: Fraction of NaN pixels
- outlier_fraction: Fraction of outlier pixels (extreme values)
- valid_fraction: Fraction of valid usable pixels
- is_good: Boolean flag if valid_fraction >= threshold

Usage:
    python add_quality_metadata.py --dataset path/to/dataset.pkl
    python add_quality_metadata.py --dataset path/to/dataset.pkl --threshold 0.85 --inplace
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from tqdm import tqdm


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


def compute_frame_quality_stats(
    array: np.ndarray,
    zero_threshold: float = 0.05,
    outlier_percentile: float = 99.5,
    min_valid_fraction: float = 0.90
) -> FrameQualityStats:
    """
    Compute quality statistics for a single frame.
    
    Args:
        array: numpy array (H, W, C) - single frame
        zero_threshold: threshold below which pixels are considered "zero/bad"
        outlier_percentile: percentile above which pixels are outliers
        min_valid_fraction: minimum valid pixel fraction to pass quality check
    
    Returns:
        FrameQualityStats with quality metrics
    """
    total_pixels = array.size
    
    # Count NaN pixels
    nan_mask = np.isnan(array)
    nan_count = np.sum(nan_mask)
    nan_fraction = nan_count / total_pixels
    
    # Count zero/near-zero pixels
    zero_mask = np.abs(array) < zero_threshold
    zero_count = np.sum(zero_mask & ~nan_mask)
    zero_fraction = zero_count / total_pixels
    
    # Get valid pixels for outlier detection
    valid_mask = ~nan_mask & ~zero_mask
    valid_pixels = array[valid_mask]
    
    if len(valid_pixels) == 0:
        return FrameQualityStats(
            zero_fraction=float(zero_fraction),
            nan_fraction=float(nan_fraction),
            outlier_fraction=0.0,
            valid_fraction=0.0,
            mean_reflectance=0.0,
            std_reflectance=0.0,
            is_good=False
        )
    
    # Detect outliers
    outlier_threshold = np.percentile(valid_pixels, outlier_percentile)
    outlier_mask = (array > outlier_threshold) & valid_mask
    outlier_count = np.sum(outlier_mask)
    outlier_fraction = outlier_count / total_pixels
    
    # Final valid pixels
    final_valid_mask = valid_mask & ~outlier_mask
    valid_fraction = np.sum(final_valid_mask) / total_pixels
    
    # Statistics on valid pixels
    final_valid_pixels = array[final_valid_mask]
    mean_refl = float(np.mean(final_valid_pixels)) if len(final_valid_pixels) > 0 else 0.0
    std_refl = float(np.std(final_valid_pixels)) if len(final_valid_pixels) > 0 else 0.0
    
    # Quality decision
    is_good = valid_fraction >= min_valid_fraction
    
    return FrameQualityStats(
        zero_fraction=float(zero_fraction),
        nan_fraction=float(nan_fraction),
        outlier_fraction=float(outlier_fraction),
        valid_fraction=float(valid_fraction),
        mean_reflectance=mean_refl,
        std_reflectance=std_refl,
        is_good=is_good
    )


def compute_sample_quality(
    sample,
    min_valid_fraction: float = 0.90
) -> List[Dict]:
    """
    Compute quality for all frames in a sample.
    
    Computes quality for both LR and HR, using the worst case between them.
    
    Args:
        sample: TrainingSample5Band object
        min_valid_fraction: minimum valid fraction threshold
    
    Returns:
        List of quality dicts, one per temporal frame
    """
    if sample.lr_sequence is None:
        return None
    
    T = sample.lr_sequence.shape[0]
    frame_qualities = []
    
    for t in range(T):
        # Compute quality for LR (10m)
        lr_stats = compute_frame_quality_stats(
            sample.lr_sequence[t],
            min_valid_fraction=min_valid_fraction
        )
        
        # Compute quality for HR (10m resolution)
        if sample.hr_10m_sequence is not None:
            hr_stats = compute_frame_quality_stats(
                sample.hr_10m_sequence[t],
                min_valid_fraction=min_valid_fraction
            )
        elif sample.hr_sequence is not None:
            hr_stats = compute_frame_quality_stats(
                sample.hr_sequence[t],
                min_valid_fraction=min_valid_fraction
            )
        else:
            hr_stats = lr_stats
        
        # Use worst case between LR and HR
        combined_quality = {
            'lr_zero_frac': lr_stats.zero_fraction,
            'lr_nan_frac': lr_stats.nan_fraction,
            'lr_valid_frac': lr_stats.valid_fraction,
            'lr_is_good': lr_stats.is_good,
            'hr_zero_frac': hr_stats.zero_fraction,
            'hr_nan_frac': hr_stats.nan_fraction,
            'hr_valid_frac': hr_stats.valid_fraction,
            'hr_is_good': hr_stats.is_good,
            # Combined metrics (use minimum valid fraction)
            'valid_frac': min(lr_stats.valid_fraction, hr_stats.valid_fraction),
            'is_good': lr_stats.is_good and hr_stats.is_good
        }
        
        frame_qualities.append(combined_quality)
    
    return frame_qualities


def add_quality_to_dataset(
    dataset_path: str,
    output_path: Optional[str] = None,
    min_valid_fraction: float = 0.90,
    inplace: bool = False
) -> Dict:
    """
    Add quality metadata to all samples in dataset.
    
    Args:
        dataset_path: Path to dataset.pkl
        output_path: Path for output (if None, adds '_with_quality' suffix)
        min_valid_fraction: Quality threshold
        inplace: If True, overwrite original file
    
    Returns:
        Statistics about the quality distribution
    """
    print(f"Loading dataset from {dataset_path}...")
    
    # Need to register the class for unpickling
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_cache_to_5band import TrainingSample5Band
    import __main__
    __main__.TrainingSample5Band = TrainingSample5Band
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    samples = data['samples']
    print(f"  Loaded {len(samples)} samples")
    
    # Statistics
    stats = {
        'total_samples': len(samples),
        'total_frames': 0,
        'good_frames': 0,
        'bad_frames': 0,
        'samples_all_good': 0,
        'samples_all_bad': 0,
        'samples_mixed': 0,
        'quality_distribution': []
    }
    
    # Process each sample
    print(f"Computing quality metadata (threshold={min_valid_fraction})...")
    for sample in tqdm(samples):
        quality = compute_sample_quality(sample, min_valid_fraction)
        
        if quality is None:
            continue
        
        # Update sample
        sample.frame_quality = quality
        
        # Update stats
        good_in_sample = sum(1 for q in quality if q['is_good'])
        bad_in_sample = len(quality) - good_in_sample
        
        stats['total_frames'] += len(quality)
        stats['good_frames'] += good_in_sample
        stats['bad_frames'] += bad_in_sample
        
        if good_in_sample == len(quality):
            stats['samples_all_good'] += 1
        elif good_in_sample == 0:
            stats['samples_all_bad'] += 1
        else:
            stats['samples_mixed'] += 1
        
        # Track quality distribution
        for q in quality:
            stats['quality_distribution'].append(q['valid_frac'])
    
    # Compute distribution stats
    if stats['quality_distribution']:
        q_dist = np.array(stats['quality_distribution'])
        stats['valid_frac_mean'] = float(np.mean(q_dist))
        stats['valid_frac_std'] = float(np.std(q_dist))
        stats['valid_frac_median'] = float(np.median(q_dist))
        stats['valid_frac_min'] = float(np.min(q_dist))
        stats['valid_frac_max'] = float(np.max(q_dist))
        # Histogram bins
        stats['quality_histogram'] = {
            '0.0-0.5': int(np.sum(q_dist < 0.5)),
            '0.5-0.7': int(np.sum((q_dist >= 0.5) & (q_dist < 0.7))),
            '0.7-0.8': int(np.sum((q_dist >= 0.7) & (q_dist < 0.8))),
            '0.8-0.9': int(np.sum((q_dist >= 0.8) & (q_dist < 0.9))),
            '0.9-1.0': int(np.sum(q_dist >= 0.9))
        }
    
    # Remove raw distribution for cleaner output
    del stats['quality_distribution']
    
    # Save
    if inplace:
        output_path = dataset_path
    elif output_path is None:
        p = Path(dataset_path)
        output_path = str(p.parent / f"{p.stem}_with_quality{p.suffix}")
    
    print(f"\nSaving to {output_path}...")
    data['samples'] = samples
    data['quality_stats'] = stats
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    return stats


def print_quality_report(stats: Dict):
    """Print formatted quality report."""
    print("\n" + "=" * 60)
    print("QUALITY METADATA REPORT")
    print("=" * 60)
    
    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"Total frames:  {stats['total_frames']}")
    
    good_pct = stats['good_frames'] / stats['total_frames'] * 100 if stats['total_frames'] > 0 else 0
    print(f"\nFrame Quality:")
    print(f"  Good frames: {stats['good_frames']} ({good_pct:.1f}%)")
    print(f"  Bad frames:  {stats['bad_frames']} ({100-good_pct:.1f}%)")
    
    print(f"\nSample Distribution:")
    print(f"  All frames good: {stats['samples_all_good']}")
    print(f"  All frames bad:  {stats['samples_all_bad']}")
    print(f"  Mixed quality:   {stats['samples_mixed']}")
    
    if 'valid_frac_mean' in stats:
        print(f"\nValid Fraction Statistics:")
        print(f"  Mean:   {stats['valid_frac_mean']:.3f}")
        print(f"  Std:    {stats['valid_frac_std']:.3f}")
        print(f"  Median: {stats['valid_frac_median']:.3f}")
        print(f"  Min:    {stats['valid_frac_min']:.3f}")
        print(f"  Max:    {stats['valid_frac_max']:.3f}")
    
    if 'quality_histogram' in stats:
        print(f"\nQuality Distribution (valid fraction):")
        for range_str, count in stats['quality_histogram'].items():
            pct = count / stats['total_frames'] * 100 if stats['total_frames'] > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  {range_str}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Add quality metadata to dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset.pkl')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: adds _with_quality suffix)')
    parser.add_argument('--threshold', type=float, default=0.90,
                        help='Minimum valid fraction threshold (default: 0.90)')
    parser.add_argument('--inplace', action='store_true',
                        help='Overwrite original file')
    
    args = parser.parse_args()
    
    stats = add_quality_to_dataset(
        args.dataset,
        args.output,
        args.threshold,
        args.inplace
    )
    
    print_quality_report(stats)
    
    print("\n✓ Quality metadata added successfully!")
    print(f"  Use sample.frame_quality[t]['is_good'] to filter frames during training")


if __name__ == '__main__':
    main()
