#!/usr/bin/env python3
"""
Visualize all 100 samples from the STA-2 2020 dataset.
Creates a grid showing LR and HR for each sample.
"""

import sys
sys.path.insert(0, 'scripts/preprocessing')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from generate_sta2_grid_parallel import TrainingSample

def normalize_for_display(img, percentile_low=2, percentile_high=98):
    """Normalize image to 0-1 range using percentile clipping."""
    img = np.clip(img, 0, None)  # Remove negatives
    
    if img.max() == img.min():
        return np.zeros_like(img)
    
    # Use percentile for better contrast
    low = np.percentile(img, percentile_low)
    high = np.percentile(img, percentile_high)
    
    if high == low:
        return np.zeros_like(img)
    
    img = (img - low) / (high - low)
    return np.clip(img, 0, 1)

def create_rgb(img_array, bands=[2, 1, 0]):
    """Create RGB from bands (default: B4, B3, B2 for true color)."""
    # img_array shape: (H, W, C)
    if img_array.shape[-1] >= 3:
        rgb = img_array[:, :, bands[:3]]
    else:
        rgb = np.stack([img_array[:, :, 0]] * 3, axis=-1)
    return normalize_for_display(rgb)

def main():
    # Load dataset
    dataset_path = Path('data/sr_training/sta2_2020_100samples/dataset.pkl')
    print(f"Loading {dataset_path}...")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    samples = dataset['samples']
    config = dataset['config']
    
    print(f"Loaded {len(samples)} samples")
    print(f"Config: {config}")
    
    # Create output directory
    output_dir = Path('visualizations/sta2_2020_100samples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. Overview grid: 10x10 thumbnails (first frame only)
    # =========================================================================
    print("\n📊 Creating overview grid (100 samples)...")
    
    n_samples = len(samples)
    n_cols = 10
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 2))
    fig.suptitle(f'STA-2 2020 Dataset: {n_samples} Samples (HR Frame 1)', fontsize=16)
    
    for idx, sample in enumerate(samples):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Show first HR frame as RGB
        hr_frame = sample.hr_sequence[0]  # First temporal frame
        rgb = create_rgb(hr_frame)
        
        ax.imshow(rgb)
        ax.axis('off')
        ax.set_title(f'{idx+1}', fontsize=8)
    
    # Hide empty axes
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    overview_path = output_dir / 'overview_100_samples.png'
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {overview_path}")
    
    # =========================================================================
    # 2. LR vs HR comparison grid (first 25 samples)
    # =========================================================================
    print("\n📊 Creating LR vs HR comparison (first 25 samples)...")
    
    n_compare = min(25, n_samples)
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    fig.suptitle(f'LR (Sentinel-2) vs HR (Planet) Comparison - First {n_compare} Samples', fontsize=14)
    
    for idx in range(n_compare):
        row = idx // 5
        col_lr = (idx % 5) * 2
        col_hr = col_lr + 1
        
        sample = samples[idx]
        
        # LR (Sentinel-2)
        lr_frame = sample.lr_sequence[0]
        lr_rgb = create_rgb(lr_frame)
        axes[row, col_lr].imshow(lr_rgb)
        axes[row, col_lr].axis('off')
        if row == 0:
            axes[row, col_lr].set_title('LR', fontsize=10)
        
        # HR (Planet)
        hr_frame = sample.hr_sequence[0]
        hr_rgb = create_rgb(hr_frame)
        axes[row, col_hr].imshow(hr_rgb)
        axes[row, col_hr].axis('off')
        if row == 0:
            axes[row, col_hr].set_title('HR', fontsize=10)
    
    plt.tight_layout()
    compare_path = output_dir / 'lr_hr_comparison_25.png'
    plt.savefig(compare_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {compare_path}")
    
    # =========================================================================
    # 3. Temporal sequences (first 10 samples, all 3 frames)
    # =========================================================================
    print("\n📊 Creating temporal sequence visualization (10 samples × 3 frames)...")
    
    n_temporal = min(10, n_samples)
    fig, axes = plt.subplots(n_temporal, 6, figsize=(18, n_temporal * 2.5))
    fig.suptitle('Temporal Sequences: LR (S2) and HR (Planet) for 3 Frames', fontsize=14)
    
    for idx in range(n_temporal):
        sample = samples[idx]
        dates = sample.temporal_dates
        
        for frame_idx in range(3):
            # LR
            lr_frame = sample.lr_sequence[frame_idx]
            lr_rgb = create_rgb(lr_frame)
            axes[idx, frame_idx * 2].imshow(lr_rgb)
            axes[idx, frame_idx * 2].axis('off')
            if idx == 0:
                axes[idx, frame_idx * 2].set_title(f'LR T{frame_idx+1}', fontsize=10)
            if frame_idx == 0:
                axes[idx, frame_idx * 2].set_ylabel(f'Sample {idx+1}', fontsize=10)
            
            # HR
            hr_frame = sample.hr_sequence[frame_idx]
            hr_rgb = create_rgb(hr_frame)
            axes[idx, frame_idx * 2 + 1].imshow(hr_rgb)
            axes[idx, frame_idx * 2 + 1].axis('off')
            if idx == 0:
                axes[idx, frame_idx * 2 + 1].set_title(f'HR T{frame_idx+1}', fontsize=10)
    
    plt.tight_layout()
    temporal_path = output_dir / 'temporal_sequences_10.png'
    plt.savefig(temporal_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {temporal_path}")
    
    # =========================================================================
    # 4. Statistics summary
    # =========================================================================
    print("\n📊 Computing dataset statistics...")
    
    lr_mins, lr_maxs, lr_means = [], [], []
    hr_mins, hr_maxs, hr_means = [], [], []
    
    for sample in samples:
        lr_mins.append(sample.lr_sequence.min())
        lr_maxs.append(sample.lr_sequence.max())
        lr_means.append(sample.lr_sequence.mean())
        hr_mins.append(sample.hr_sequence.min())
        hr_maxs.append(sample.hr_sequence.max())
        hr_means.append(sample.hr_sequence.mean())
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   LR (Sentinel-2):")
    print(f"      Min: {np.min(lr_mins):.2f}, Max: {np.max(lr_maxs):.2f}")
    print(f"      Mean of means: {np.mean(lr_means):.2f}")
    print(f"   HR (Planet):")
    print(f"      Min: {np.min(hr_mins):.2f}, Max: {np.max(hr_maxs):.2f}")
    print(f"      Mean of means: {np.mean(hr_means):.2f}")
    
    # =========================================================================
    # 5. Individual sample pages (all 100)
    # =========================================================================
    print("\n📊 Creating individual sample visualizations...")
    
    individual_dir = output_dir / 'individual'
    individual_dir.mkdir(exist_ok=True)
    
    for idx, sample in enumerate(samples):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        dates = sample.temporal_dates
        
        fig.suptitle(f'Sample {idx+1}: {sample.region_name} - Dates: {dates}', fontsize=12)
        
        for frame_idx in range(3):
            # LR
            lr_frame = sample.lr_sequence[frame_idx]
            lr_rgb = create_rgb(lr_frame)
            axes[0, frame_idx].imshow(lr_rgb)
            axes[0, frame_idx].set_title(f'LR (S2) - {dates[frame_idx]}')
            axes[0, frame_idx].axis('off')
            
            # HR
            hr_frame = sample.hr_sequence[frame_idx]
            hr_rgb = create_rgb(hr_frame)
            axes[1, frame_idx].imshow(hr_rgb)
            axes[1, frame_idx].set_title(f'HR (Planet) - {dates[frame_idx]}')
            axes[1, frame_idx].axis('off')
        
        plt.tight_layout()
        individual_path = individual_dir / f'sample_{idx+1:03d}.png'
        plt.savefig(individual_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if (idx + 1) % 20 == 0:
            print(f"   ✓ Saved {idx + 1}/{len(samples)} individual samples")
    
    print(f"   ✓ All {len(samples)} individual samples saved to {individual_dir}")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print(f"   - overview_100_samples.png")
    print(f"   - lr_hr_comparison_25.png")
    print(f"   - temporal_sequences_10.png")
    print(f"   - individual/sample_001.png ... sample_{len(samples):03d}.png")

if __name__ == '__main__':
    main()
