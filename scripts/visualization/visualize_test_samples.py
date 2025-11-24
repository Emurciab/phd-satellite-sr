#!/usr/bin/env python3
"""
Visualize test samples to check tiling fix and coregistration quality
Shows HR, LR (upscaled), difference, and overlay for visual inspection
"""

import sys
import os

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "preprocessing"))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

def load_dataset(dataset_path):
    """Load the test dataset"""
    dataset_path = Path(dataset_path)
    
    # Load samples
    samples_file = dataset_path / "training_samples.pkl"
    with open(samples_file, 'rb') as f:
        samples = pickle.load(f)
    
    # Load metadata
    metadata_file = dataset_path / "metadata.pkl"
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"✅ Loaded {len(samples)} samples")
    print(f"   LR shape: {metadata['lr_shape']}")
    print(f"   HR shape: {metadata['hr_shape']}")
    print(f"   Tiling ratio: {metadata['hr_shape'][1]} / {metadata['lr_shape'][1]} = {metadata['hr_shape'][1]/metadata['lr_shape'][1]:.2f}:1")
    
    return samples, metadata


def normalize_for_display(img):
    """Normalize image for display (0-1 range)"""
    img_float = img.astype(np.float32)
    # Use percentile normalization to handle outliers
    p2, p98 = np.percentile(img_float, (2, 98))
    img_norm = np.clip((img_float - p2) / (p98 - p2 + 1e-8), 0, 1)
    return img_norm


def calculate_alignment_metrics(hr, lr_upscaled):
    """Calculate alignment quality metrics"""
    # Use RGB bands only
    hr_rgb = hr[:, :, :3]
    lr_rgb = lr_upscaled[:, :, :3]
    
    # Normalize
    hr_norm = normalize_for_display(hr_rgb)
    lr_norm = normalize_for_display(lr_rgb)
    
    # Calculate metrics
    ssim_val = ssim(hr_norm, lr_norm, channel_axis=2, data_range=1.0)
    mse_val = mse(hr_norm, lr_norm)
    
    # Edge alignment (gradient correlation)
    from scipy.ndimage import sobel
    hr_edges = np.sqrt(sobel(hr_norm[:,:,0], axis=0)**2 + sobel(hr_norm[:,:,0], axis=1)**2)
    lr_edges = np.sqrt(sobel(lr_norm[:,:,0], axis=0)**2 + sobel(lr_norm[:,:,0], axis=1)**2)
    edge_corr = np.corrcoef(hr_edges.flatten(), lr_edges.flatten())[0, 1]
    
    return {
        'ssim': ssim_val,
        'mse': mse_val,
        'edge_correlation': edge_corr
    }


def visualize_sample(sample, sample_idx, output_dir):
    """
    Create comprehensive visualization for one sample
    Shows temporal sequence with alignment analysis
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    lr_seq = sample.lr_sequence  # (T, H, W, C)
    hr_seq = sample.hr_sequence  # (T, H, W, C)
    dates = sample.temporal_dates
    
    T = lr_seq.shape[0]
    
    print(f"\n{'='*70}")
    print(f"Sample {sample_idx}: {sample.region_name}")
    print(f"{'='*70}")
    print(f"Dates: {dates}")
    print(f"LR shape: {lr_seq.shape}")
    print(f"HR shape: {hr_seq.shape}")
    
    # Create figure with subplots for each frame
    fig = plt.figure(figsize=(20, 4.5*T))
    
    for t in range(T):
        lr = lr_seq[t]  # (H, W, C)
        hr = hr_seq[t]  # (H, W, C)
        
        # Upscale LR to HR size for comparison
        lr_upscaled = resize(lr, hr.shape[:2], order=1, preserve_range=True, anti_aliasing=True)
        
        # Calculate metrics
        metrics = calculate_alignment_metrics(hr, lr_upscaled)
        
        print(f"\n  Frame {t+1} ({dates[t]}):")
        print(f"    SSIM: {metrics['ssim']:.4f}")
        print(f"    MSE: {metrics['mse']:.6f}")
        print(f"    Edge correlation: {metrics['edge_correlation']:.4f}")
        
        # Normalize for display
        hr_disp = normalize_for_display(hr[:, :, :3])
        lr_disp = normalize_for_display(lr_upscaled[:, :, :3])
        
        # Calculate difference
        diff = np.abs(hr_disp - lr_disp).mean(axis=2)
        
        # Create overlay (checkerboard pattern)
        overlay = hr_disp.copy()
        block_size = 32
        for i in range(0, overlay.shape[0], block_size*2):
            for j in range(0, overlay.shape[1], block_size*2):
                overlay[i:i+block_size, j:j+block_size] = lr_disp[i:i+block_size, j:j+block_size]
                if i+block_size < overlay.shape[0] and j+block_size < overlay.shape[1]:
                    overlay[i+block_size:i+block_size*2, j+block_size:j+block_size*2] = \
                        lr_disp[i+block_size:i+block_size*2, j+block_size:j+block_size*2]
        
        # Row for this frame
        row_base = t * 1
        
        # Plot HR
        ax1 = plt.subplot(T, 5, t*5 + 1)
        ax1.imshow(hr_disp)
        ax1.set_title(f"HR (Planet)\n{dates[t]}\n{hr.shape[0]}×{hr.shape[1]}", fontsize=10)
        ax1.axis('off')
        
        # Plot LR (upscaled)
        ax2 = plt.subplot(T, 5, t*5 + 2)
        ax2.imshow(lr_disp)
        ax2.set_title(f"LR (Sentinel-2)\nUpscaled to HR size\n{lr.shape[0]}×{lr.shape[1]} → {lr_upscaled.shape[0]}×{lr_upscaled.shape[1]}", 
                     fontsize=10)
        ax2.axis('off')
        
        # Plot difference
        ax3 = plt.subplot(T, 5, t*5 + 3)
        im3 = ax3.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        ax3.set_title(f"Difference Map\nMSE: {metrics['mse']:.6f}", fontsize=10)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Plot overlay
        ax4 = plt.subplot(T, 5, t*5 + 4)
        ax4.imshow(overlay)
        ax4.set_title(f"Checkerboard Overlay\nSSIM: {metrics['ssim']:.4f}", fontsize=10)
        ax4.axis('off')
        
        # Plot edge alignment
        ax5 = plt.subplot(T, 5, t*5 + 5)
        from scipy.ndimage import sobel
        hr_edges = np.sqrt(sobel(hr_disp[:,:,0], axis=0)**2 + sobel(hr_disp[:,:,0], axis=1)**2)
        lr_edges = np.sqrt(sobel(lr_disp[:,:,0], axis=0)**2 + sobel(lr_disp[:,:,0], axis=1)**2)
        edge_overlay = np.stack([hr_edges, lr_edges, np.zeros_like(hr_edges)], axis=-1)
        edge_overlay = edge_overlay / (edge_overlay.max() + 1e-8)
        ax5.imshow(edge_overlay)
        ax5.set_title(f"Edge Overlay (R=HR, G=LR)\nCorr: {metrics['edge_correlation']:.4f}", fontsize=10)
        ax5.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f"sample_{sample_idx:03d}_alignment.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  💾 Saved: {output_file}")
    plt.close()


def create_summary_visualization(samples, output_dir):
    """Create summary showing one frame from each sample"""
    output_dir = Path(output_dir)
    
    n_samples = len(samples)
    
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: All {n_samples} samples (middle frame)")
    print(f"{'='*70}")
    
    for idx, sample in enumerate(samples):
        lr_seq = sample.lr_sequence
        hr_seq = sample.hr_sequence
        dates = sample.temporal_dates
        
        # Use middle frame
        t = lr_seq.shape[0] // 2
        lr = lr_seq[t]
        hr = hr_seq[t]
        
        # Upscale LR
        lr_upscaled = resize(lr, hr.shape[:2], order=1, preserve_range=True, anti_aliasing=True)
        
        # Calculate metrics
        metrics = calculate_alignment_metrics(hr, lr_upscaled)
        
        # Normalize
        hr_disp = normalize_for_display(hr[:, :, :3])
        lr_disp = normalize_for_display(lr_upscaled[:, :, :3])
        diff = np.abs(hr_disp - lr_disp).mean(axis=2)
        
        # Create overlay
        overlay = hr_disp * 0.5 + lr_disp * 0.5
        
        # Plot
        axes[idx, 0].imshow(hr_disp)
        axes[idx, 0].set_title(f"Sample {idx}\nHR {hr.shape[0]}×{hr.shape[1]}", fontsize=9)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(lr_disp)
        axes[idx, 1].set_title(f"LR {lr.shape[0]}×{lr.shape[1]}\n→ {lr_upscaled.shape[0]}×{lr_upscaled.shape[1]}", fontsize=9)
        axes[idx, 1].axis('off')
        
        im = axes[idx, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[idx, 2].set_title(f"Difference\nMSE: {metrics['mse']:.5f}", fontsize=9)
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title(f"Overlay\nSSIM: {metrics['ssim']:.3f}", fontsize=9)
        axes[idx, 3].axis('off')
        
        # Tiling visualization
        from scipy.ndimage import sobel
        hr_edges = np.sqrt(sobel(hr_disp[:,:,0], axis=0)**2 + sobel(hr_disp[:,:,0], axis=1)**2)
        lr_edges = np.sqrt(sobel(lr_disp[:,:,0], axis=0)**2 + sobel(lr_disp[:,:,0], axis=1)**2)
        edge_overlay = np.stack([hr_edges, lr_edges, np.zeros_like(hr_edges)], axis=-1)
        edge_overlay = edge_overlay / (edge_overlay.max() + 1e-8)
        axes[idx, 4].imshow(edge_overlay)
        axes[idx, 4].set_title(f"Edges\nCorr: {metrics['edge_correlation']:.3f}", fontsize=9)
        axes[idx, 4].axis('off')
        
        print(f"  Sample {idx}: SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.6f}, EdgeCorr={metrics['edge_correlation']:.4f}")
    
    plt.tight_layout()
    
    output_file = output_dir / "summary_all_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  💾 Saved summary: {output_file}")
    plt.close()


def main():
    dataset_path = "./data/datasets/sr_test_fixed_tiling"
    output_dir = "./visualizations_test_fixed_tiling"
    
    print("="*70)
    print("🔍 VISUAL QUALITY CHECK: Fixed Tiling + Coregistration")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Load dataset
    samples, metadata = load_dataset(dataset_path)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Visualize each sample
    for idx, sample in enumerate(samples):
        visualize_sample(sample, idx, output_dir)
    
    # Create summary
    create_summary_visualization(samples, output_dir)
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE")
    print("="*70)
    print(f"Generated {len(samples)} detailed visualizations + 1 summary")
    print(f"Output directory: {output_dir}")
    print("\nCheck the images to verify:")
    print("  1. ✅ Geographic features align (buildings, roads, fields)")
    print("  2. ✅ Edges match between HR and LR")
    print("  3. ✅ Checkerboard pattern shows minimal discontinuities")
    print("  4. ✅ Difference map shows low values (dark areas)")
    print("="*70)


if __name__ == "__main__":
    main()
