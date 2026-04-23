#!/usr/bin/env python3
"""
Interactive visualization of dataset samples.
Shows LR, LR upsampled, and HR for multiple samples.
"""

import sys
sys.path.insert(0, 'scripts/preprocessing')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom

def normalize_for_display(img, percentile=98):
    """Normalize image for display"""
    if np.all(img == 0):
        return img
    p = np.percentile(img[img > 0], percentile) if np.any(img > 0) else 1.0
    if p == 0:
        p = 1.0
    return np.clip(img / p, 0, 1)

def visualize_samples(dataset_path, sample_indices=None, num_samples=12, save_path=None):
    """Visualize multiple samples from the dataset"""
    
    print("="*80)
    print("🎨 VISUALIZING DATASET SAMPLES")
    print("="*80)
    print()
    
    # Load dataset
    print("📂 Loading dataset...")
    with open(dataset_path / 'training_samples.pkl', 'rb') as f:
        data = pickle.load(f)
    
    total_samples = len(data)
    print(f"✅ Loaded {total_samples} samples")
    print()
    
    # Select samples to visualize
    if sample_indices is None:
        # Show evenly distributed samples
        step = max(1, total_samples // num_samples)
        sample_indices = list(range(0, min(total_samples, num_samples * step), step))[:num_samples]
    
    print(f"📊 Visualizing samples: {sample_indices}")
    print()
    
    # Helper function to get sequences
    def get_sequences(idx):
        sample = data[idx]
        lr_seq = sample.lr_sequence if hasattr(sample, 'lr_sequence') else sample[0]
        hr_seq = sample.hr_sequence if hasattr(sample, 'hr_sequence') else sample[1]
        return lr_seq, hr_seq
    
    # Get sample info
    lr_seq, hr_seq = get_sequences(0)
    lr_size = lr_seq.shape[2]
    hr_size = hr_seq.shape[2]
    ratio = hr_size / lr_size
    
    print(f"📐 Image dimensions:")
    print(f"   LR: {lr_size}×{lr_size}")
    print(f"   HR: {hr_size}×{hr_size}")
    print(f"   Ratio: {ratio:.2f}:1")
    print()
    
    # Create visualization grid
    n_samples = len(sample_indices)
    n_cols = 3  # LR, LR upsampled, HR
    n_rows = n_samples
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Dataset Samples - {n_samples} Examples (Middle Frame of Sequence)', 
                 fontsize=16, y=0.999)
    
    print("🖼️  Processing samples...")
    
    for row_idx, sample_idx in enumerate(sample_indices):
        lr_seq, hr_seq = get_sequences(sample_idx)
        
        # Use middle frame (index 1 of 3-frame sequence)
        lr_frame = lr_seq[1, :, :, :3]  # RGB channels only
        hr_frame = hr_seq[1, :, :, :3]  # RGB channels only
        
        # Check if sample is valid (not all zeros)
        is_valid = np.any(lr_frame > 0) and np.any(hr_frame > 0)
        status = "✅" if is_valid else "⚠️ (masked)"
        
        # Normalize for display
        lr_display = normalize_for_display(lr_frame)
        hr_display = normalize_for_display(hr_frame)
        
        # Upsample LR to HR size for comparison
        lr_upsampled = zoom(lr_display, (ratio, ratio, 1), order=1)
        
        # Plot LR
        axes[row_idx, 0].imshow(lr_display)
        axes[row_idx, 0].set_title(f'Sample {sample_idx} - LR ({lr_size}×{lr_size}) {status}')
        axes[row_idx, 0].axis('off')
        
        # Plot LR upsampled
        axes[row_idx, 1].imshow(lr_upsampled)
        axes[row_idx, 1].set_title(f'LR Upsampled ({hr_size}×{hr_size})')
        axes[row_idx, 1].axis('off')
        
        # Plot HR
        axes[row_idx, 2].imshow(hr_display)
        axes[row_idx, 2].set_title(f'HR Target ({hr_size}×{hr_size})')
        axes[row_idx, 2].axis('off')
        
        # Add statistics as text
        if is_valid:
            lr_mean = np.mean(lr_frame)
            hr_mean = np.mean(hr_frame)
            stats_text = f'LR mean: {lr_mean:.1f}, HR mean: {hr_mean:.1f}'
            axes[row_idx, 0].text(0.02, 0.02, stats_text, transform=axes[row_idx, 0].transAxes,
                                 fontsize=8, verticalalignment='bottom', 
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        print(f"   ✓ Sample {sample_idx:3d} {status}")
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        output_path = Path(save_path)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved visualization to: {output_path}")
    else:
        output_path = dataset_path / f'samples_visualization_{n_samples}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved visualization to: {output_path}")
    
    plt.close()
    
    # Create temporal sequence visualization for first sample
    print("\n🎬 Creating temporal sequence visualization...")
    
    sample_idx = sample_indices[0]
    lr_seq, hr_seq = get_sequences(sample_idx)
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    fig2.suptitle(f'Temporal Sequence - Sample {sample_idx} (3 Frames)', fontsize=14)
    
    for t in range(3):
        lr_frame = lr_seq[t, :, :, :3]
        hr_frame = hr_seq[t, :, :, :3]
        
        lr_display = normalize_for_display(lr_frame)
        hr_display = normalize_for_display(hr_frame)
        
        axes2[0, t].imshow(lr_display)
        axes2[0, t].set_title(f'LR Frame {t+1}/3')
        axes2[0, t].axis('off')
        
        axes2[1, t].imshow(hr_display)
        axes2[1, t].set_title(f'HR Frame {t+1}/3')
        axes2[1, t].axis('off')
    
    plt.tight_layout()
    
    temporal_path = dataset_path / f'temporal_sequence_sample_{sample_idx}.png'
    plt.savefig(temporal_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved temporal sequence to: {temporal_path}")
    plt.close()
    
    print()
    print("="*80)
    print("✅ VISUALIZATION COMPLETE")
    print("="*80)
    print()
    print("💡 Files created:")
    print(f"   - {output_path}")
    print(f"   - {temporal_path}")
    print()


def visualize_specific_samples():
    """Interactive function to visualize specific samples"""
    dataset_path = Path('./data/datasets/sr_production_100')
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    print("="*80)
    print("🎨 DATASET SAMPLE VISUALIZER")
    print("="*80)
    print()
    
    # Load dataset to get count
    with open(dataset_path / 'training_samples.pkl', 'rb') as f:
        data = pickle.load(f)
    
    total_samples = len(data)
    print(f"Dataset contains {total_samples} samples (0-{total_samples-1})")
    print()
    print("Options:")
    print("  1) View 12 evenly distributed samples")
    print("  2) View first 20 samples")
    print("  3) View last 20 samples")
    print("  4) View random 15 samples")
    print("  5) View specific sample indices")
    print()
    
    choice = input("Select option (1-5) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        visualize_samples(dataset_path, num_samples=12)
    elif choice == "2":
        indices = list(range(min(20, total_samples)))
        visualize_samples(dataset_path, sample_indices=indices)
    elif choice == "3":
        indices = list(range(max(0, total_samples - 20), total_samples))
        visualize_samples(dataset_path, sample_indices=indices)
    elif choice == "4":
        import random
        indices = sorted(random.sample(range(total_samples), min(15, total_samples)))
        visualize_samples(dataset_path, sample_indices=indices)
    elif choice == "5":
        indices_str = input(f"Enter sample indices (comma-separated, 0-{total_samples-1}): ").strip()
        try:
            indices = [int(x.strip()) for x in indices_str.split(',')]
            indices = [i for i in indices if 0 <= i < total_samples]
            if indices:
                visualize_samples(dataset_path, sample_indices=indices)
            else:
                print("❌ No valid indices provided")
        except ValueError:
            print("❌ Invalid input format")
    else:
        print("❌ Invalid option")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize dataset samples')
    parser.add_argument('--dataset', type=str, default='./data/datasets/sr_production_100',
                       help='Path to dataset directory')
    parser.add_argument('--samples', type=str, default=None,
                       help='Comma-separated sample indices (e.g., 0,5,10,15)')
    parser.add_argument('--num', type=int, default=12,
                       help='Number of evenly distributed samples to show (if --samples not specified)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        exit(1)
    
    if args.interactive:
        visualize_specific_samples()
    else:
        if args.samples:
            indices = [int(x.strip()) for x in args.samples.split(',')]
            visualize_samples(dataset_path, sample_indices=indices)
        else:
            visualize_samples(dataset_path, num_samples=args.num)
