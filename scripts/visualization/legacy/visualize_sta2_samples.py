#!/usr/bin/env python3
"""
Visualize STA-2 dataset sample pairs (LR and HR sequences).
Shows both good quality and high-zero samples for comparison.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'scripts', 'preprocessing'))


def normalize_for_display(img, percentile=98, clip_negatives=True):
    """
    Normalize image for display using percentile stretching.
    
    Bands are: [Blue, Green, Red, NIR]
    For both Sentinel-2 (B2,B3,B4,B8A) and Planet (B1,B2,B3,B4)
    """
    if img.max() == 0:
        return img
    
    # Use RGB bands only (first 3 bands: Blue, Green, Red)
    if img.shape[-1] >= 3:
        rgb = img[..., :3].astype(np.float32)
    else:
        rgb = img.astype(np.float32)
    
    # Clip negative values from coregistration interpolation
    if clip_negatives:
        rgb = np.clip(rgb, 0, None)
    
    # Normalize ALL RGB bands together (not per-band) for better color balance
    p2, p98 = np.percentile(rgb, (2, percentile))
    rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    return rgb_norm


def assess_sample_quality(lr_data, hr_data):
    """Assess sample quality and return classification."""
    lr_zeros = np.sum(lr_data == 0) / lr_data.size * 100
    hr_zeros = np.sum(hr_data == 0) / hr_data.size * 100
    lr_nans = np.sum(np.isnan(lr_data))
    hr_nans = np.sum(np.isnan(hr_data))
    
    if lr_nans > 0 or hr_nans > 0:
        return 'nan', lr_zeros, hr_zeros
    elif lr_zeros > 50 or hr_zeros > 50:
        return 'high_zeros', lr_zeros, hr_zeros
    else:
        return 'good', lr_zeros, hr_zeros


def visualize_samples(dataset_path, output_dir, num_good=6, num_high_zeros=6):
    """Visualize sample pairs from the dataset."""
    
    print('=' * 80)
    print('📊 VISUALIZING STA-2 DATASET SAMPLES')
    print('=' * 80)
    
    # Load samples
    samples_path = os.path.join(dataset_path, 'training_samples.pkl')
    print(f'\nLoading samples from: {samples_path}')
    
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)
    
    print(f'✅ Loaded {len(samples)} samples\n')
    
    # Categorize samples
    good_samples = []
    high_zero_samples = []
    
    print('Categorizing samples...')
    for i, s in enumerate(samples):
        if i % 200 == 0:
            print(f'  Checked {i}/{len(samples)}...', flush=True)
        
        # Get data
        if hasattr(s, 'lr_sequence'):
            lr_data = s.lr_sequence
            hr_data = s.hr_sequence
        elif hasattr(s, 'lr'):
            lr_data = s.lr
            hr_data = s.hr
        else:
            lr_data = s['lr']
            hr_data = s['hr']
        
        quality, lr_zeros, hr_zeros = assess_sample_quality(lr_data, hr_data)
        
        if quality == 'good':
            good_samples.append((i, s, lr_zeros, hr_zeros))
        elif quality == 'high_zeros':
            high_zero_samples.append((i, s, lr_zeros, hr_zeros))
    
    print(f'\n✅ Found {len(good_samples)} good samples')
    print(f'⚠️  Found {len(high_zero_samples)} high-zero samples\n')
    
    # Select samples to visualize
    np.random.seed(42)
    
    good_indices = np.random.choice(len(good_samples), min(num_good, len(good_samples)), replace=False)
    selected_good = [good_samples[i] for i in good_indices]
    
    high_zero_indices = np.random.choice(len(high_zero_samples), min(num_high_zeros, len(high_zero_samples)), replace=False)
    selected_high_zeros = [high_zero_samples[i] for i in high_zero_indices]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize good samples
    print('=' * 80)
    print('📸 VISUALIZING GOOD QUALITY SAMPLES')
    print('=' * 80)
    
    for idx, (sample_idx, sample, lr_zeros, hr_zeros) in enumerate(selected_good):
        print(f'\nVisualizing good sample {idx+1}/{len(selected_good)} (dataset index: {sample_idx})')
        visualize_single_sample(
            sample, sample_idx, lr_zeros, hr_zeros,
            os.path.join(output_dir, f'good_sample_{idx+1:02d}_idx{sample_idx}.png'),
            'GOOD QUALITY'
        )
    
    # Visualize high-zero samples
    print('\n' + '=' * 80)
    print('📸 VISUALIZING HIGH-ZERO SAMPLES')
    print('=' * 80)
    
    for idx, (sample_idx, sample, lr_zeros, hr_zeros) in enumerate(selected_high_zeros):
        print(f'\nVisualizing high-zero sample {idx+1}/{len(selected_high_zeros)} (dataset index: {sample_idx})')
        visualize_single_sample(
            sample, sample_idx, lr_zeros, hr_zeros,
            os.path.join(output_dir, f'highzero_sample_{idx+1:02d}_idx{sample_idx}.png'),
            'HIGH ZEROS'
        )
    
    print('\n' + '=' * 80)
    print('✅ VISUALIZATION COMPLETE')
    print('=' * 80)
    print(f'\nSaved {len(selected_good) + len(selected_high_zeros)} visualizations to:')
    print(f'  {output_dir}')
    print(f'\n  Good quality: {len(selected_good)} samples')
    print(f'  High zeros: {len(selected_high_zeros)} samples')


def visualize_single_sample(sample, sample_idx, lr_zeros, hr_zeros, output_path, quality_label):
    """Visualize a single sample with LR and HR sequences."""
    
    # Get data
    if hasattr(sample, 'lr_sequence'):
        lr_seq = sample.lr_sequence
        hr_seq = sample.hr_sequence
        temporal_dates = sample.temporal_dates if hasattr(sample, 'temporal_dates') else None
        region_name = sample.region_name if hasattr(sample, 'region_name') else 'N/A'
        tile_id = sample.tile_id if hasattr(sample, 'tile_id') else 'N/A'
    elif hasattr(sample, 'lr'):
        lr_seq = sample.lr
        hr_seq = sample.hr
        temporal_dates = None
        region_name = 'N/A'
        tile_id = 'N/A'
    else:
        lr_seq = sample['lr']
        hr_seq = sample['hr']
        temporal_dates = sample.get('temporal_dates', None)
        region_name = sample.get('region_name', 'N/A')
        tile_id = sample.get('tile_id', 'N/A')
    
    # Create figure
    n_frames = lr_seq.shape[0]
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(2, n_frames + 1, figure=fig, hspace=0.3, wspace=0.15)
    
    # Add title
    title = f'{quality_label} Sample #{sample_idx} - Region: {region_name}, Tile: {tile_id}\n'
    title += f'RGB Composite (Blue, Green, Red bands) - LR zeros: {lr_zeros:.1f}%, HR zeros: {hr_zeros:.1f}%'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Plot LR sequence
    for t in range(n_frames):
        ax = fig.add_subplot(gs[0, t])
        lr_frame = lr_seq[t]
        lr_display = normalize_for_display(lr_frame)
        
        ax.imshow(lr_display)
        date_str = f'T{t+1}'
        if temporal_dates and len(temporal_dates) > t:
            date_str = f'T{t+1}: {temporal_dates[t]}'
        ax.set_title(f'LR {date_str}\n{lr_frame.shape[0]}×{lr_frame.shape[1]}', fontsize=10)
        ax.axis('off')
        
        # Add zero percentage
        frame_zeros = np.sum(lr_frame == 0) / lr_frame.size * 100
        ax.text(0.02, 0.98, f'{frame_zeros:.1f}% zeros', 
                transform=ax.transAxes, fontsize=8, color='white',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Add LR label
    ax = fig.add_subplot(gs[0, n_frames])
    ax.text(0.5, 0.5, 'LOW\nRESOLUTION\n(Sentinel-2)\n10m', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes, rotation=0)
    ax.axis('off')
    
    # Plot HR sequence
    for t in range(n_frames):
        ax = fig.add_subplot(gs[1, t])
        hr_frame = hr_seq[t]
        hr_display = normalize_for_display(hr_frame)
        
        ax.imshow(hr_display)
        date_str = f'T{t+1}'
        if temporal_dates and len(temporal_dates) > t:
            date_str = f'T{t+1}: {temporal_dates[t]}'
        ax.set_title(f'HR {date_str}\n{hr_frame.shape[0]}×{hr_frame.shape[1]}', fontsize=10)
        ax.axis('off')
        
        # Add zero percentage
        frame_zeros = np.sum(hr_frame == 0) / hr_frame.size * 100
        ax.text(0.02, 0.98, f'{frame_zeros:.1f}% zeros', 
                transform=ax.transAxes, fontsize=8, color='white',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Add HR label
    ax = fig.add_subplot(gs[1, n_frames])
    ax.text(0.5, 0.5, 'HIGH\nRESOLUTION\n(Planet)\n3m', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes, rotation=0)
    ax.axis('off')
    
    # Add statistics box
    stats_text = f'Dataset Statistics:\n'
    stats_text += f'LR shape: {lr_seq.shape}\n'
    stats_text += f'HR shape: {hr_seq.shape}\n'
    stats_text += f'Resolution ratio: {hr_seq.shape[1] / lr_seq.shape[1]:.2f}:1\n'
    stats_text += f'LR range: [{lr_seq.min():.1f}, {lr_seq.max():.1f}]\n'
    stats_text += f'HR range: [{hr_seq.min():.1f}, {hr_seq.max():.1f}]\n\n'
    stats_text += f'Band Order (both sensors):\n'
    stats_text += f'  Band 0: Blue\n'
    stats_text += f'  Band 1: Green\n'
    stats_text += f'  Band 2: Red\n'
    stats_text += f'  Band 3: NIR (not shown)\n\n'
    stats_text += f'Sentinel-2: B2,B3,B4,B8A\n'
    stats_text += f'Planet: B1,B2,B3,B4'
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  ✅ Saved: {output_path}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize STA-2 dataset samples')
    parser.add_argument('--dataset', type=str, 
                       default='./data/datasets/sta2_test',
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str,
                       default='./tests/visualizations/sta2_samples',
                       help='Output directory for visualizations')
    parser.add_argument('--num-good', type=int, default=6,
                       help='Number of good quality samples to visualize')
    parser.add_argument('--num-high-zeros', type=int, default=6,
                       help='Number of high-zero samples to visualize')
    
    args = parser.parse_args()
    
    visualize_samples(args.dataset, args.output, args.num_good, args.num_high_zeros)
