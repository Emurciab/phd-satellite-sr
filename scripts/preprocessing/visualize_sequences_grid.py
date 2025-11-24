#!/usr/bin/env python3
"""
Create a compact grid visualization showing multiple sequences at once.
Useful for getting an overview of the dataset quality.
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
    """Training sample with temporal sequences"""
    lr_sequence: np.ndarray
    hr_sequence: np.ndarray
    temporal_dates: List[str]  # Sentinel-2 dates
    planet_dates: List[str] = None  # Planet dates (may differ by ±7 days) - optional for backwards compatibility
    region_name: str = None
    sequence_id: int = None
    tile_id: int = None
    tile_bounds: Dict = None


def normalize_for_display(img, percentile_clip=2):
    """Normalize image for display using percentile clipping"""
    img = img.copy().astype(np.float32)
    p_low = np.percentile(img, percentile_clip)
    p_high = np.percentile(img, 100 - percentile_clip)
    img = np.clip(img, p_low, p_high)
    
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    
    return img


def create_rgb_composite(img_4band, bands=[2, 1, 0]):
    """Create RGB composite from 4-band image"""
    if img_4band.shape[-1] == 4:
        rgb = img_4band[:, :, bands]
    else:
        rgb = img_4band
    return rgb


def visualize_grid(samples, max_sequences=12, save_path=None, show=True):
    """
    Create a grid visualization of multiple sequences
    
    Args:
        samples: List of TrainingSample objects
        max_sequences: Maximum number of sequences to show
        save_path: Path to save the figure (optional)
        show: Whether to display the figure
    """
    n_sequences = min(len(samples), max_sequences)
    n_frames = samples[0].lr_sequence.shape[0]
    
    # Create figure: rows = n_sequences, columns = n_frames * 2 (LR and HR)
    fig, axes = plt.subplots(n_sequences, n_frames * 2, 
                            figsize=(4 * n_frames * 2, 3 * n_sequences))
    
    if n_sequences == 1:
        axes = axes.reshape(1, -1)
    
    # Create title with date information
    if hasattr(samples[0], 'planet_dates') and samples[0].planet_dates:
        date_info = ' | '.join([f"S2:{s2}/PL:{pl}" for s2, pl in zip(samples[0].temporal_dates, samples[0].planet_dates)])
    else:
        date_info = ' → '.join(samples[0].temporal_dates)
    
    fig.suptitle(
        f"Temporal Sequences Overview - First {n_sequences} Samples\n"
        f"Region: {samples[0].region_name} | {date_info}",
        fontsize=14, fontweight='bold', y=0.995
    )
    
    for seq_idx in range(n_sequences):
        sample = samples[seq_idx]
        
        for frame_idx in range(n_frames):
            # LR frame (left columns)
            lr_frame = sample.lr_sequence[frame_idx]
            lr_rgb = create_rgb_composite(lr_frame, bands=[2, 1, 0])
            lr_display = normalize_for_display(lr_rgb, percentile_clip=2)
            
            ax_lr = axes[seq_idx, frame_idx * 2]
            ax_lr.imshow(lr_display)
            ax_lr.axis('off')
            
            # Add title only for first row
            if seq_idx == 0:
                s2_date = sample.temporal_dates[frame_idx]
                ax_lr.set_title(f"LR-T{frame_idx+1}\nS2:{s2_date}", 
                              fontsize=9, fontweight='bold')
            
            # Add sample ID on the left
            if frame_idx == 0:
                ax_lr.text(-0.15, 0.5, f"Sample {seq_idx}\nTile {sample.tile_id}", 
                          transform=ax_lr.transAxes,
                          fontsize=8, verticalalignment='center',
                          rotation=90, fontweight='bold')
            
            # HR frame (right columns)
            hr_frame = sample.hr_sequence[frame_idx]
            hr_rgb = create_rgb_composite(hr_frame, bands=[2, 1, 0])
            hr_display = normalize_for_display(hr_rgb, percentile_clip=2)
            
            ax_hr = axes[seq_idx, frame_idx * 2 + 1]
            ax_hr.imshow(hr_display)
            ax_hr.axis('off')
            
            # Add title only for first row
            if seq_idx == 0:
                if hasattr(sample, 'planet_dates') and sample.planet_dates:
                    planet_date = sample.planet_dates[frame_idx]
                else:
                    planet_date = sample.temporal_dates[frame_idx]
                ax_hr.set_title(f"HR-T{frame_idx+1}\nPL:{planet_date}", 
                              fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0.01, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create grid visualization of sequences')
    parser.add_argument('--dataset-path', type=str, 
                       default='./data/datasets/sta2_grid_parallel/dataset.pkl',
                       help='Path to dataset pickle file')
    parser.add_argument('--max-sequences', type=int, default=12,
                       help='Maximum number of sequences to show in grid')
    parser.add_argument('--save-path', type=str, 
                       default='./visualizations/sequences_grid_overview.png',
                       help='Path to save the grid visualization')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display figure (only save)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TEMPORAL SEQUENCES GRID VISUALIZATION")
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
    
    # Create save directory if needed
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create grid visualization
    print(f"\n📊 Creating grid with first {min(args.max_sequences, len(samples))} sequences...")
    visualize_grid(
        samples, 
        max_sequences=args.max_sequences,
        save_path=args.save_path,
        show=not args.no_show
    )
    
    print(f"\n{'='*80}")
    print(f"✓ Grid visualization complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
