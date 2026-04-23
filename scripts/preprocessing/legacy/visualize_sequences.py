#!/usr/bin/env python3
"""
Visualize temporal sequences from the grid-based parallel dataset.
Shows all 3 temporal frames from each sequence side-by-side.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import argparse
from pathlib import Path


@dataclass
class TrainingSample:
    """Training sample with temporal sequences - must match the original definition"""
    lr_sequence: np.ndarray  # (T, H, W, C)
    hr_sequence: np.ndarray  # (T, H, W, C)
    temporal_dates: List[str]  # Sentinel-2 dates
    planet_dates: List[str] = None  # Planet dates (may differ by ±7 days) - optional for backwards compatibility
    region_name: str = None
    sequence_id: int = None
    tile_id: int = None
    tile_bounds: Dict = None


def normalize_for_display(img, percentile_clip=2):
    """
    Normalize image for display using percentile clipping
    
    Args:
        img: numpy array (H, W, C) or (H, W)
        percentile_clip: percentile to clip at both ends
    
    Returns:
        Normalized image in [0, 1]
    """
    img = img.copy().astype(np.float32)
    
    # Clip outliers
    p_low = np.percentile(img, percentile_clip)
    p_high = np.percentile(img, 100 - percentile_clip)
    
    img = np.clip(img, p_low, p_high)
    
    # Normalize to [0, 1]
    img_min = img.min()
    img_max = img.max()
    
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    
    return img


def create_rgb_composite(img_4band, bands=[2, 1, 0]):
    """
    Create RGB composite from 4-band image
    
    Args:
        img_4band: (H, W, 4) array - B, G, R, NIR
        bands: Which bands to use for RGB [R_idx, G_idx, B_idx]
    
    Returns:
        (H, W, 3) RGB image
    """
    if img_4band.shape[-1] == 4:
        # Select RGB bands (default: R=B3, G=B2, B=B1)
        rgb = img_4band[:, :, bands]
    else:
        rgb = img_4band
    
    return rgb


def visualize_sequence(sample, sample_idx, save_dir=None, show=True):
    """
    Visualize a single temporal sequence with all 3 frames
    
    Args:
        sample: TrainingSample object
        sample_idx: Index of the sample
        save_dir: Directory to save the figure (optional)
        show: Whether to display the figure
    """
    n_frames = sample.lr_sequence.shape[0]
    
    # Create figure with 2 rows (LR and HR) and n_frames columns
    fig, axes = plt.subplots(2, n_frames, figsize=(5*n_frames, 10))
    
    if n_frames == 1:
        axes = axes.reshape(2, 1)
    
    # Create title with both S2 and Planet dates if available
    if hasattr(sample, 'planet_dates') and sample.planet_dates:
        date_info = ' → '.join([f"S2:{s2}/PL:{pl}" for s2, pl in zip(sample.temporal_dates, sample.planet_dates)])
    else:
        date_info = ' → '.join(sample.temporal_dates)
    
    fig.suptitle(
        f"Sample {sample_idx} - {sample.region_name} (Seq {sample.sequence_id}, Tile {sample.tile_id})\n"
        f"{date_info}",
        fontsize=14, fontweight='bold'
    )
    
    for frame_idx in range(n_frames):
        s2_date = sample.temporal_dates[frame_idx]
        planet_date = sample.planet_dates[frame_idx] if hasattr(sample, 'planet_dates') and sample.planet_dates else s2_date
        
        # LR frame (Sentinel-2)
        lr_frame = sample.lr_sequence[frame_idx]
        lr_rgb = create_rgb_composite(lr_frame, bands=[2, 1, 0])  # R, G, B
        lr_display = normalize_for_display(lr_rgb, percentile_clip=2)
        
        axes[0, frame_idx].imshow(lr_display)
        axes[0, frame_idx].set_title(f"LR (S2) Frame {frame_idx+1}\n{s2_date}\n{lr_frame.shape[0]}×{lr_frame.shape[1]}px", 
                                     fontsize=11)
        axes[0, frame_idx].axis('off')
        
        # Add value range info
        axes[0, frame_idx].text(0.02, 0.98, 
                               f"Range: [{lr_frame.min():.0f}, {lr_frame.max():.0f}]",
                               transform=axes[0, frame_idx].transAxes,
                               fontsize=8, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # HR frame (Planet)
        hr_frame = sample.hr_sequence[frame_idx]
        hr_rgb = create_rgb_composite(hr_frame, bands=[2, 1, 0])  # R, G, B
        hr_display = normalize_for_display(hr_rgb, percentile_clip=2)
        
        axes[1, frame_idx].imshow(hr_display)
        axes[1, frame_idx].set_title(f"HR (Planet) Frame {frame_idx+1}\n{planet_date}\n{hr_frame.shape[0]}×{hr_frame.shape[1]}px", 
                                     fontsize=11)
        axes[1, frame_idx].axis('off')
        
        # Add value range info
        axes[1, frame_idx].text(0.02, 0.98, 
                               f"Range: [{hr_frame.min():.0f}, {hr_frame.max():.0f}]",
                               transform=axes[1, frame_idx].transAxes,
                               fontsize=8, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Sentinel-2 (10m)', 
                   transform=axes[0, 0].transAxes,
                   fontsize=12, fontweight='bold',
                   rotation=90, verticalalignment='center')
    
    axes[1, 0].text(-0.1, 0.5, 'Planet (3m)', 
                   transform=axes[1, 0].transAxes,
                   fontsize=12, fontweight='bold',
                   rotation=90, verticalalignment='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_dir:
        save_path = Path(save_dir) / f"sequence_{sample_idx:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize temporal sequences from dataset')
    parser.add_argument('--dataset-path', type=str, 
                       default='./data/datasets/sta2_grid_parallel/dataset.pkl',
                       help='Path to dataset pickle file')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='Maximum number of samples to visualize')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save figures (optional)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display figures (only save)')
    parser.add_argument('--sample-indices', type=str, default=None,
                       help='Specific sample indices to visualize (comma-separated, e.g., "0,5,10")')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TEMPORAL SEQUENCE VISUALIZATION")
    print("="*80)
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {args.dataset_path}")
    with open(args.dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both dict and list formats
    if isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
        config = data.get('config', {})
        print(f"✓ Loaded {len(samples)} samples")
        print(f"📋 Config: {config}")
    else:
        samples = data
        print(f"✓ Loaded {len(samples)} samples")
    
    # Determine which samples to visualize
    if args.sample_indices:
        indices = [int(i) for i in args.sample_indices.split(',')]
        indices = [i for i in indices if 0 <= i < len(samples)]
        print(f"📊 Visualizing specific samples: {indices}")
    else:
        indices = list(range(min(args.max_samples, len(samples))))
        print(f"📊 Visualizing first {len(indices)} samples")
    
    # Create save directory if needed
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving figures to: {args.save_dir}")
    
    # Visualize each sample
    print(f"\n{'='*80}")
    print("VISUALIZING SEQUENCES")
    print(f"{'='*80}\n")
    
    for idx in indices:
        sample = samples[idx]
        print(f"Sample {idx}:")
        print(f"   Region: {sample.region_name}")
        print(f"   Sequence ID: {sample.sequence_id}")
        print(f"   Tile ID: {sample.tile_id}")
        print(f"   Dates: {' → '.join(sample.temporal_dates)}")
        print(f"   LR shape: {sample.lr_sequence.shape}")
        print(f"   HR shape: {sample.hr_sequence.shape}")
        
        visualize_sequence(
            sample, 
            idx, 
            save_dir=args.save_dir, 
            show=not args.no_show
        )
        print()
    
    print(f"{'='*80}")
    print(f"✓ Visualization complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
