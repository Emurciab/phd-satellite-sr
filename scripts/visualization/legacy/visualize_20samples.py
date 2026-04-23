#!/usr/bin/env python3
"""
Visualize 20 samples with harmonization from the generated dataset.
Creates side-by-side LR/HR comparisons with spectral statistics.
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def normalize_for_display(img, percentile=2):
    """Normalize image for display using percentile clipping"""
    if img.max() == img.min():
        return img
    
    p_low = np.percentile(img, percentile)
    p_high = np.percentile(img, 100 - percentile)
    
    img_norm = np.clip(img, p_low, p_high)
    img_norm = (img_norm - p_low) / (p_high - p_low)
    
    return img_norm

def get_rgb_from_sample(sample_frame, is_lr=False):
    """
    Extract RGB from sample frame.
    For 4-band: B,G,R,NIR → take indices [2,1,0] for RGB
    """
    # Sample frame is (H, W, C)
    if sample_frame.shape[2] >= 3:
        # For 4-band BGRA format: B=0, G=1, R=2, A=3
        # We want RGB order, so [2, 1, 0] = [R, G, B]
        rgb = sample_frame[:, :, [2, 1, 0]]
    else:
        # Fallback: use first 3 bands as-is
        rgb = sample_frame[:, :, :3]
    
    return rgb

def compute_spectral_stats(lr_frame, hr_frame):
    """Compute per-band statistics"""
    stats = []
    num_bands = min(lr_frame.shape[2], hr_frame.shape[2])
    
    for band_idx in range(num_bands):
        lr_mean = lr_frame[:, :, band_idx].mean()
        hr_mean = hr_frame[:, :, band_idx].mean()
        diff = abs(lr_mean - hr_mean)
        stats.append({
            'band': band_idx,
            'lr_mean': lr_mean,
            'hr_mean': hr_mean,
            'diff': diff
        })
    
    return stats

def visualize_sample(sample, output_path, sample_idx):
    """Create visualization for a single sample (first frame only)"""
    
    # Get first frame from temporal sequence
    lr_frame = sample.lr_sequence[0]  # (H, W, C)
    hr_frame = sample.hr_sequence[0]  # (H, W, C)
    
    # Extract RGB
    lr_rgb = get_rgb_from_sample(lr_frame, is_lr=True)
    hr_rgb = get_rgb_from_sample(hr_frame, is_lr=False)
    
    # Normalize for display
    lr_rgb_norm = normalize_for_display(lr_rgb)
    hr_rgb_norm = normalize_for_display(hr_rgb)
    
    # Compute spectral statistics
    stats = compute_spectral_stats(lr_frame, hr_frame)
    avg_diff = np.mean([s['diff'] for s in stats])
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # LR image
    axes[0].imshow(lr_rgb_norm)
    axes[0].set_title(f'LR (Sentinel-2)\n{lr_frame.shape[0]}×{lr_frame.shape[1]}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # HR image
    axes[1].imshow(hr_rgb_norm)
    axes[1].set_title(f'HR (Planet - Harmonized)\n{hr_frame.shape[0]}×{hr_frame.shape[1]}', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Statistics
    axes[2].axis('off')
    stats_text = f"""Sample #{sample_idx + 1}
    
Region: {sample.region_name}
Date: {sample.temporal_dates[0]}
Sequence ID: {sample.sequence_id}
Tile ID: {sample.tile_id}

LR shape: {sample.lr_sequence.shape}
HR shape: {sample.hr_sequence.shape}

Spectral Statistics (Frame 1):
─────────────────────────────
"""
    
    for s in stats:
        band_name = ['Blue', 'Green', 'Red', 'NIR'][s['band']] if s['band'] < 4 else f"B{s['band']}"
        stats_text += f"{band_name:5s}: LR={s['lr_mean']:6.1f} HR={s['hr_mean']:6.1f} Δ={s['diff']:5.1f}\n"
    
    stats_text += f"\nAvg Spectral Diff: {avg_diff:.1f}"
    
    # Color code based on harmonization quality
    if avg_diff < 100:
        quality = "✅ Excellent"
        color = 'green'
    elif avg_diff < 500:
        quality = "🟡 Good"
        color = 'orange'
    else:
        quality = "⚠️  Needs Review"
        color = 'red'
    
    stats_text += f"\nHarmonization: {quality}"
    
    axes[2].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', color=color if avg_diff >= 100 else 'black')
    
    plt.suptitle(f'Sample {sample_idx + 1}/{20} - Spectral Harmonization Applied', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return avg_diff

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize dataset samples')
    parser.add_argument('--dataset-path', type=str, 
                       default='./data/datasets/sta2_20samples_harmonized/dataset.pkl',
                       help='Path to dataset.pkl file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 VISUALIZING SAMPLES WITH HARMONIZATION")
    print("=" * 80)
    
    # Dataset path
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found at: {dataset_path}")
        print("\n💡 Please wait for dataset generation to complete first.")
        print("   Check: tail -f dataset_20samples_generation.log")
        return
    
    # Create output directory
    output_dir = Path('./data/datasets/sta2_20samples_harmonized/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output directory next to dataset
    output_dir = dataset_path.parent / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset_dict = pickle.load(f)
    
    # Handle both formats: dict with 'samples' key or direct list
    if isinstance(dataset_dict, dict) and 'samples' in dataset_dict:
        samples = dataset_dict['samples']
    else:
        samples = dataset_dict
    
    print(f"   ✅ Loaded {len(samples)} samples")
    
    # Visualize each sample
    print(f"\n🎨 Generating visualizations...")
    
    spectral_diffs = []
    
    for idx, sample in enumerate(samples):
        output_path = output_dir / f'sample_{idx+1:02d}.png'
        
        print(f"   [{idx+1}/{len(samples)}] {sample.region_name} - {sample.temporal_dates[0]}...", end=" ")
        
        try:
            avg_diff = visualize_sample(sample, output_path, idx)
            spectral_diffs.append(avg_diff)
            print(f"✓ (Δ={avg_diff:.1f})")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary statistics
    print(f"\n" + "=" * 80)
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)
    
    if spectral_diffs:
        print(f"\nSpectral Difference (lower = better harmonization):")
        print(f"   Mean:   {np.mean(spectral_diffs):.1f}")
        print(f"   Median: {np.median(spectral_diffs):.1f}")
        print(f"   Min:    {np.min(spectral_diffs):.1f}")
        print(f"   Max:    {np.max(spectral_diffs):.1f}")
        print(f"   Std:    {np.std(spectral_diffs):.1f}")
        
        # Quality distribution
        excellent = sum(1 for d in spectral_diffs if d < 100)
        good = sum(1 for d in spectral_diffs if 100 <= d < 500)
        needs_review = sum(1 for d in spectral_diffs if d >= 500)
        
        print(f"\nQuality Distribution:")
        print(f"   ✅ Excellent (<100):   {excellent}/{len(spectral_diffs)} ({excellent/len(spectral_diffs)*100:.1f}%)")
        print(f"   🟡 Good (100-500):     {good}/{len(spectral_diffs)} ({good/len(spectral_diffs)*100:.1f}%)")
        print(f"   ⚠️  Needs Review (>500): {needs_review}/{len(spectral_diffs)} ({needs_review/len(spectral_diffs)*100:.1f}%)")
    
    print(f"\n✅ Visualizations saved to: {output_dir}")
    print(f"   Total files: {len(list(output_dir.glob('*.png')))}")
    
    # Create summary plot
    if len(spectral_diffs) > 0:
        print(f"\n📊 Creating summary plot...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(spectral_diffs, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(100, color='green', linestyle='--', label='Excellent threshold')
        ax1.axvline(500, color='orange', linestyle='--', label='Good threshold')
        ax1.set_xlabel('Average Spectral Difference', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_title('Distribution of Spectral Differences', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2.scatter(range(len(spectral_diffs)), spectral_diffs, 
                   c=['green' if d < 100 else 'orange' if d < 500 else 'red' for d in spectral_diffs],
                   s=100, alpha=0.6, edgecolors='black')
        ax2.axhline(100, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(500, color='orange', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Sample Index', fontsize=12)
        ax2.set_ylabel('Average Spectral Difference', fontsize=12)
        ax2.set_title('Spectral Difference per Sample', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Harmonization Quality Assessment (20 Samples)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        summary_path = output_dir / 'summary_harmonization.png'
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Summary plot saved: {summary_path}")
    
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
