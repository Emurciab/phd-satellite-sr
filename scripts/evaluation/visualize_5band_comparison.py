#!/usr/bin/env python3
"""
Visualize 5-Band Super-Resolution Results

Compares:
1. Multi-Resolution model (10m→3m at 260×260, 20m→6m at 130×130)
2. Bilinear interpolation baseline

Uses held-out test data (unseen during training) to evaluate generalization.
"""

import sys
import os

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts' / 'training'))
sys.path.insert(0, str(project_root / 'scripts' / 'preprocessing'))
sys.path.insert(0, str(project_root / 'scripts' / 'training' / 'models'))

from satellite_sr_model import MultiResolutionSatelliteSR


def load_model(checkpoint_path, scale_factor_10m=3.33, scale_factor_20m=3.33):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model = MultiResolutionSatelliteSR(
        in_channels_10m=3,
        in_channels_20m=2,
        base_channels=64,
        num_encoder_blocks=4,
        num_frames=3,
        scale_factor_10m=scale_factor_10m,
        scale_factor_20m=scale_factor_20m
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def prepare_sample_mmap(lr_10m, lr_20m, hr_10m, hr_20m, sequence_length=3):
    """Prepare a sample from mmap arrays for inference.
    
    Args:
        lr_10m: (T, H, W, C) - LR 10m bands
        lr_20m: (T, H, W, C) - LR 20m bands
        hr_10m: (T, H, W, C) - HR 10m bands (ground truth)
        hr_20m: (T, H, W, C) - HR 20m bands (ground truth)
        sequence_length: Number of temporal frames
    """
    # Convert to tensors: (T, H, W, C) -> (1, T, C, H, W)
    lr_10m_t = torch.from_numpy(lr_10m).permute(0, 3, 1, 2).unsqueeze(0).float()
    lr_20m_t = torch.from_numpy(lr_20m).permute(0, 3, 1, 2).unsqueeze(0).float()
    
    # Ground truth for middle frame
    mid_idx = sequence_length // 2
    gt_10m = hr_10m[mid_idx]  # (H, W, C)
    gt_20m = hr_20m[mid_idx]  # (H/2, W/2, C)
    
    # LR for display (middle frame)
    lr_10m_display = lr_10m[mid_idx]  # (H, W, C)
    lr_20m_display = lr_20m[mid_idx]  # (H/2, W/2, C)
    
    return {
        'lr_10m': lr_10m_t,
        'lr_20m': lr_20m_t,
        'lr_10m_display': lr_10m_display,
        'lr_20m_display': lr_20m_display,
        'gt_10m': gt_10m,
        'gt_20m': gt_20m,
    }


# =============================================================================
# QUALITY METRICS
# =============================================================================

def compute_psnr(pred, target):
    """Compute PSNR between prediction and target."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    max_val = max(pred.max(), target.max(), 1.0)
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_rmse(pred, target):
    """Compute Root Mean Square Error."""
    return np.sqrt(np.mean((pred - target) ** 2))


def compute_ssim(pred, target, win_size=7):
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted image (H, W) or (H, W, C)
        target: Target image (H, W) or (H, W, C)
        win_size: Window size for local statistics
    
    Returns:
        SSIM value between -1 and 1 (1 = identical)
    """
    # Handle multi-channel images
    if pred.ndim == 3:
        ssim_values = []
        for c in range(pred.shape[-1]):
            ssim_values.append(compute_ssim(pred[:, :, c], target[:, :, c], win_size))
        return np.mean(ssim_values)
    
    # Constants to stabilize division
    C1 = (0.01 * 1.0) ** 2  # Assuming data range is 0-1
    C2 = (0.03 * 1.0) ** 2
    
    # Compute local means using uniform filter
    from scipy.ndimage import uniform_filter
    
    mu_pred = uniform_filter(pred.astype(np.float64), size=win_size, mode='reflect')
    mu_target = uniform_filter(target.astype(np.float64), size=win_size, mode='reflect')
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = uniform_filter(pred.astype(np.float64) ** 2, size=win_size, mode='reflect') - mu_pred_sq
    sigma_target_sq = uniform_filter(target.astype(np.float64) ** 2, size=win_size, mode='reflect') - mu_target_sq
    sigma_pred_target = uniform_filter(pred.astype(np.float64) * target.astype(np.float64), size=win_size, mode='reflect') - mu_pred_target
    
    # SSIM formula
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    
    ssim_map = numerator / (denominator + 1e-10)
    return np.mean(ssim_map)


def compute_scc(pred, target):
    """
    Compute Spatial Correlation Coefficient (SCC).
    
    Measures the correlation of high-frequency spatial details.
    
    Args:
        pred: Predicted image (H, W) or (H, W, C)
        target: Target image (H, W) or (H, W, C)
    
    Returns:
        SCC value between -1 and 1 (1 = perfect correlation)
    """
    # Handle multi-channel images
    if pred.ndim == 3:
        scc_values = []
        for c in range(pred.shape[-1]):
            scc_values.append(compute_scc(pred[:, :, c], target[:, :, c]))
        return np.mean(scc_values)
    
    # Apply Laplacian filter to extract high-frequency components
    from scipy.ndimage import laplace
    
    pred_hf = laplace(pred.astype(np.float64))
    target_hf = laplace(target.astype(np.float64))
    
    # Compute correlation coefficient
    pred_flat = pred_hf.flatten()
    target_flat = target_hf.flatten()
    
    # Remove mean
    pred_centered = pred_flat - np.mean(pred_flat)
    target_centered = target_flat - np.mean(target_flat)
    
    # Correlation coefficient
    numerator = np.sum(pred_centered * target_centered)
    denominator = np.sqrt(np.sum(pred_centered ** 2) * np.sum(target_centered ** 2))
    
    if denominator < 1e-10:
        return 1.0 if numerator < 1e-10 else 0.0
    
    return numerator / denominator


def compute_ergas(pred, target, scale_ratio):
    """
    Compute ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse).
    
    A global quality metric for super-resolution that accounts for scale factor.
    Lower is better (0 = perfect reconstruction).
    
    Args:
        pred: Predicted image (H, W, C)
        target: Target image (H, W, C)
        scale_ratio: Upscaling factor (e.g., 3.33 for 10m→3m)
    
    Returns:
        ERGAS value (lower is better, typically < 3 is good)
    """
    if pred.ndim == 2:
        pred = pred[:, :, np.newaxis]
        target = target[:, :, np.newaxis]
    
    num_bands = pred.shape[-1]
    sum_term = 0.0
    
    for b in range(num_bands):
        pred_band = pred[:, :, b]
        target_band = target[:, :, b]
        
        rmse_band = np.sqrt(np.mean((pred_band - target_band) ** 2))
        mean_target = np.mean(target_band)
        
        if mean_target > 1e-10:
            sum_term += (rmse_band / mean_target) ** 2
        else:
            sum_term += 0.0
    
    ergas = 100 * (1 / scale_ratio) * np.sqrt(sum_term / num_bands)
    return ergas


def compute_sid(pred, target, epsilon=1e-10):
    """
    Compute Spectral Information Divergence (SID).
    
    SID measures the spectral similarity between two images based on 
    information theory. It uses the KL divergence to compare the 
    probability distributions of spectral signatures.
    
    Lower is better (0 = identical spectral signatures).
    
    Args:
        pred: Predicted image (H, W, C) - C is number of spectral bands
        target: Target image (H, W, C)
        epsilon: Small value to avoid log(0)
    
    Returns:
        SID value (lower is better, 0 = perfect spectral match)
    """
    if pred.ndim == 2:
        # Single channel - SID not meaningful
        return 0.0
    
    # Ensure positive values and normalize per-pixel to get probability distributions
    pred = np.clip(pred, epsilon, None)
    target = np.clip(target, epsilon, None)
    
    # Normalize each pixel's spectrum to sum to 1 (probability distribution)
    pred_sum = np.sum(pred, axis=-1, keepdims=True)
    target_sum = np.sum(target, axis=-1, keepdims=True)
    
    p = pred / (pred_sum + epsilon)  # Predicted probability distribution
    q = target / (target_sum + epsilon)  # Target probability distribution
    
    # KL divergence: D_KL(p||q) = sum(p * log(p/q))
    # SID = D_KL(p||q) + D_KL(q||p) (symmetric)
    
    kl_pq = np.sum(p * np.log((p + epsilon) / (q + epsilon)), axis=-1)
    kl_qp = np.sum(q * np.log((q + epsilon) / (p + epsilon)), axis=-1)
    
    sid = kl_pq + kl_qp  # Per-pixel SID
    
    # Return mean SID across all pixels
    return np.mean(sid)


def compute_all_metrics(pred, target, scale_ratio=3.33):
    """
    Compute all quality metrics.
    
    Args:
        pred: Predicted image (H, W) or (H, W, C)
        target: Target image (H, W) or (H, W, C)
        scale_ratio: Upscaling factor for ERGAS computation
    
    Returns:
        Dictionary with all metrics
    """
    return {
        'psnr': compute_psnr(pred, target),
        'ssim': compute_ssim(pred, target),
        'scc': compute_scc(pred, target),
        'rmse': compute_rmse(pred, target),
        'ergas': compute_ergas(pred, target, scale_ratio),
        'sid': compute_sid(pred, target)
    }


def normalize_for_display(img, percentile=98):
    """Normalize image for display using percentile clipping."""
    img = img.copy()
    vmin = np.percentile(img, 100 - percentile)
    vmax = np.percentile(img, percentile)
    img = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return img


def create_rgb_composite(bands_10m):
    """Create RGB composite from 10m bands (Blue, Green, Red order)."""
    # bands_10m: (H, W, 3) with Blue, Green, Red
    rgb = bands_10m[:, :, [2, 1, 0]]  # Reorder to RGB
    return normalize_for_display(rgb)


def create_false_color(bands_10m, bands_20m):
    """Create false color composite (NIR, Red, Green)."""
    nir = bands_20m[:, :, 1]
    red = bands_10m[:, :, 2]
    green = bands_10m[:, :, 1]
    
    # Handle size mismatch by upsampling 20m to match 10m
    if nir.shape[0] != red.shape[0]:
        scale = red.shape[0] / nir.shape[0]
        nir = zoom(nir, scale, order=1)
    
    false_color = np.stack([nir, red, green], axis=-1)
    return normalize_for_display(false_color)


def visualize_comparison_single_model(sample_data, model, sample_idx, save_path=None):
    """Create visualization comparing SR model vs bilinear baseline."""
    device = next(model.parameters()).device
    
    # Get dimensions
    gt_10m = sample_data['gt_10m']
    gt_20m = sample_data['gt_20m']
    target_h_10m, target_w_10m = gt_10m.shape[:2]
    target_h_20m, target_w_20m = gt_20m.shape[:2]
    
    lr_10m_display = sample_data['lr_10m_display']
    lr_20m_display = sample_data['lr_20m_display']
    
    # Run SR inference with target sizes
    with torch.no_grad():
        lr_10m = sample_data['lr_10m'].to(device)
        lr_20m = sample_data['lr_20m'].to(device)
        
        sr_10m, sr_20m = model(
            lr_10m, lr_20m,
            target_size_10m=(target_h_10m, target_w_10m),
            target_size_20m=(target_h_20m, target_w_20m)
        )
        sr_10m = sr_10m.cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, C)
        sr_20m = sr_20m.cpu().numpy()[0].transpose(1, 2, 0)
    
    # Compute bilinear baseline (upscale LR to HR size)
    bilinear_10m = F.interpolate(
        torch.tensor(lr_10m_display).permute(2, 0, 1).unsqueeze(0).float(),
        size=(target_h_10m, target_w_10m),
        mode='bilinear', align_corners=False
    )[0].permute(1, 2, 0).numpy()
    
    bilinear_20m = F.interpolate(
        torch.tensor(lr_20m_display).permute(2, 0, 1).unsqueeze(0).float(),
        size=(target_h_20m, target_w_20m),
        mode='bilinear', align_corners=False
    )[0].permute(1, 2, 0).numpy()
    
    # Compute metrics
    metrics_sr_10m = compute_all_metrics(sr_10m, gt_10m, scale_ratio=3.33)
    metrics_sr_20m = compute_all_metrics(sr_20m, gt_20m, scale_ratio=3.33)
    metrics_bilinear_10m = compute_all_metrics(bilinear_10m, gt_10m, scale_ratio=3.33)
    metrics_bilinear_20m = compute_all_metrics(bilinear_20m, gt_20m, scale_ratio=3.33)
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    
    # Title with key metrics
    fig.suptitle(f'Super-Resolution Comparison - Sample {sample_idx}\n'
                 f'Bilinear: 10m={metrics_bilinear_10m["psnr"]:.2f}dB, 20m={metrics_bilinear_20m["psnr"]:.2f}dB | '
                 f'SR Model: 10m={metrics_sr_10m["psnr"]:.2f}dB, 20m={metrics_sr_20m["psnr"]:.2f}dB | '
                 f'Improvement: 10m={metrics_sr_10m["psnr"]-metrics_bilinear_10m["psnr"]:+.2f}dB, 20m={metrics_sr_20m["psnr"]-metrics_bilinear_20m["psnr"]:+.2f}dB',
                 fontsize=12, fontweight='bold')
    
    # Row 1: RGB composites
    ax1 = fig.add_subplot(4, 5, 1)
    lr_rgb = create_rgb_composite(lr_10m_display)
    ax1.imshow(lr_rgb)
    ax1.set_title(f'LR Input\n({lr_10m_display.shape[0]}×{lr_10m_display.shape[1]})')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(4, 5, 2)
    ax2.imshow(create_rgb_composite(bilinear_10m))
    ax2.set_title(f'Bilinear\n({target_h_10m}×{target_w_10m})')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(4, 5, 3)
    ax3.imshow(create_rgb_composite(sr_10m))
    ax3.set_title(f'SR Model\n({target_h_10m}×{target_w_10m})')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(4, 5, 4)
    ax4.imshow(create_rgb_composite(gt_10m))
    ax4.set_title(f'Ground Truth\n({target_h_10m}×{target_w_10m})')
    ax4.axis('off')
    
    # Difference maps
    ax5 = fig.add_subplot(4, 5, 5)
    diff_bilinear = np.abs(bilinear_10m - gt_10m).mean(axis=-1)
    diff_sr = np.abs(sr_10m - gt_10m).mean(axis=-1)
    diff_comparison = diff_bilinear - diff_sr  # Positive = SR better
    im = ax5.imshow(diff_comparison, cmap='RdBu', vmin=-0.05, vmax=0.05)
    ax5.set_title('Error Diff\n(Blue=SR better)')
    ax5.axis('off')
    plt.colorbar(im, ax=ax5, fraction=0.046)
    
    # Row 2: Individual 10m bands
    band_names_10m = ['Blue', 'Green', 'Red']
    for i, name in enumerate(band_names_10m):
        ax = fig.add_subplot(4, 5, 6 + i)
        comparison = np.hstack([
            normalize_for_display(bilinear_10m[:, :, i:i+1].repeat(3, axis=-1)),
            normalize_for_display(sr_10m[:, :, i:i+1].repeat(3, axis=-1)),
            normalize_for_display(gt_10m[:, :, i:i+1].repeat(3, axis=-1))
        ])
        ax.imshow(comparison, cmap='gray')
        ax.set_title(f'{name}\nBilinear | SR | GT')
        ax.axis('off')
    
    # Row 2: PSNR improvement info
    ax = fig.add_subplot(4, 5, 9)
    ax.axis('off')
    ax.text(0.5, 0.5, f'10m Bands:\n\n'
            f'Bilinear PSNR: {metrics_bilinear_10m["psnr"]:.2f} dB\n'
            f'SR Model PSNR: {metrics_sr_10m["psnr"]:.2f} dB\n'
            f'Improvement: {metrics_sr_10m["psnr"]-metrics_bilinear_10m["psnr"]:+.2f} dB\n\n'
            f'Bilinear SSIM: {metrics_bilinear_10m["ssim"]:.4f}\n'
            f'SR Model SSIM: {metrics_sr_10m["ssim"]:.4f}',
            ha='center', va='center', fontsize=10, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax = fig.add_subplot(4, 5, 10)
    ax.axis('off')
    ax.text(0.5, 0.5, f'20m Bands:\n\n'
            f'Bilinear PSNR: {metrics_bilinear_20m["psnr"]:.2f} dB\n'
            f'SR Model PSNR: {metrics_sr_20m["psnr"]:.2f} dB\n'
            f'Improvement: {metrics_sr_20m["psnr"]-metrics_bilinear_20m["psnr"]:+.2f} dB\n\n'
            f'Bilinear SSIM: {metrics_bilinear_20m["ssim"]:.4f}\n'
            f'SR Model SSIM: {metrics_sr_20m["ssim"]:.4f}',
            ha='center', va='center', fontsize=10, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Row 3: 20m bands (RedEdge, NIR)
    band_names_20m = ['RedEdge', 'NIR']
    for i, name in enumerate(band_names_20m):
        ax = fig.add_subplot(4, 5, 11 + i)
        comparison = np.hstack([
            normalize_for_display(bilinear_20m[:, :, i:i+1].repeat(3, axis=-1)),
            normalize_for_display(sr_20m[:, :, i:i+1].repeat(3, axis=-1)),
            normalize_for_display(gt_20m[:, :, i:i+1].repeat(3, axis=-1))
        ])
        ax.imshow(comparison, cmap='gray')
        ax.set_title(f'{name}\nBilinear | SR | GT')
        ax.axis('off')
    
    # False color composite
    ax = fig.add_subplot(4, 5, 13)
    ax.imshow(create_false_color(bilinear_10m, bilinear_20m))
    ax.set_title('False Color (NIR-R-G)\nBilinear')
    ax.axis('off')
    
    ax = fig.add_subplot(4, 5, 14)
    ax.imshow(create_false_color(sr_10m, sr_20m))
    ax.set_title('False Color (NIR-R-G)\nSR Model')
    ax.axis('off')
    
    ax = fig.add_subplot(4, 5, 15)
    ax.imshow(create_false_color(gt_10m, gt_20m))
    ax.set_title('False Color (NIR-R-G)\nGround Truth')
    ax.axis('off')
    
    # Row 4: Zoomed regions
    zoom_size = 64
    center_h = target_h_10m // 2
    center_w = target_w_10m // 2
    slice_h = slice(center_h - zoom_size//2, center_h + zoom_size//2)
    slice_w = slice(center_w - zoom_size//2, center_w + zoom_size//2)
    
    ax = fig.add_subplot(4, 5, 16)
    ax.imshow(create_rgb_composite(bilinear_10m[slice_h, slice_w]))
    ax.set_title('Zoomed Bilinear\n(64×64 crop)')
    ax.axis('off')
    
    ax = fig.add_subplot(4, 5, 17)
    ax.imshow(create_rgb_composite(sr_10m[slice_h, slice_w]))
    ax.set_title('Zoomed SR\n(64×64 crop)')
    ax.axis('off')
    
    ax = fig.add_subplot(4, 5, 18)
    ax.imshow(create_rgb_composite(gt_10m[slice_h, slice_w]))
    ax.set_title('Zoomed GT\n(64×64 crop)')
    ax.axis('off')
    
    # Per-band PSNR bar chart
    ax = fig.add_subplot(4, 5, 19)
    bands = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR']
    
    psnr_sr = [
        compute_psnr(sr_10m[:, :, 0], gt_10m[:, :, 0]),
        compute_psnr(sr_10m[:, :, 1], gt_10m[:, :, 1]),
        compute_psnr(sr_10m[:, :, 2], gt_10m[:, :, 2]),
        compute_psnr(sr_20m[:, :, 0], gt_20m[:, :, 0]),
        compute_psnr(sr_20m[:, :, 1], gt_20m[:, :, 1]),
    ]
    psnr_bilinear = [
        compute_psnr(bilinear_10m[:, :, 0], gt_10m[:, :, 0]),
        compute_psnr(bilinear_10m[:, :, 1], gt_10m[:, :, 1]),
        compute_psnr(bilinear_10m[:, :, 2], gt_10m[:, :, 2]),
        compute_psnr(bilinear_20m[:, :, 0], gt_20m[:, :, 0]),
        compute_psnr(bilinear_20m[:, :, 1], gt_20m[:, :, 1]),
    ]
    
    x = np.arange(len(bands))
    width = 0.35
    ax.bar(x - width/2, psnr_bilinear, width, label='Bilinear', color='gray', alpha=0.7)
    ax.bar(x + width/2, psnr_sr, width, label='SR Model', color='steelblue')
    ax.set_ylabel('PSNR (dB)')
    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=45)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title('Per-Band PSNR')
    ax.set_ylim([20, 50])
    ax.grid(axis='y', alpha=0.3)
    
    # Summary text
    ax = fig.add_subplot(4, 5, 20)
    ax.axis('off')
    
    summary_text = (
        f"═══ QUALITY METRICS ═══\n\n"
        f"10m Bands:\n"
        f"  Bilinear: PSNR={metrics_bilinear_10m['psnr']:.2f}dB\n"
        f"            SSIM={metrics_bilinear_10m['ssim']:.4f}\n"
        f"            ERGAS={metrics_bilinear_10m['ergas']:.2f}\n"
        f"  SR Model: PSNR={metrics_sr_10m['psnr']:.2f}dB\n"
        f"            SSIM={metrics_sr_10m['ssim']:.4f}\n"
        f"            ERGAS={metrics_sr_10m['ergas']:.2f}\n\n"
        f"20m Bands:\n"
        f"  Bilinear: PSNR={metrics_bilinear_20m['psnr']:.2f}dB\n"
        f"            SSIM={metrics_bilinear_20m['ssim']:.4f}\n"
        f"  SR Model: PSNR={metrics_sr_20m['psnr']:.2f}dB\n"
        f"            SSIM={metrics_sr_20m['ssim']:.4f}\n"
    )
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close(fig)
    
    return {
        'psnr_sr_10m': metrics_sr_10m['psnr'],
        'psnr_sr_20m': metrics_sr_20m['psnr'],
        'psnr_bilinear_10m': metrics_bilinear_10m['psnr'],
        'psnr_bilinear_20m': metrics_bilinear_20m['psnr'],
        'psnr_sr_bands': psnr_sr,
        'psnr_bilinear_bands': psnr_bilinear,
        'metrics_sr_10m': metrics_sr_10m,
        'metrics_sr_20m': metrics_sr_20m,
        'metrics_bilinear_10m': metrics_bilinear_10m,
        'metrics_bilinear_20m': metrics_bilinear_20m,
    }


def main():
    """Main evaluation function using mmap test data."""
    # Paths
    test_dir = project_root / 'data' / 'sr_training' / '5band_v3' / 'combined' / 'mmap_test'
    checkpoint_path = project_root / 'checkpoints' / '5band_v3_goodonly' / 'best_model.pt'
    output_dir = project_root / 'results' / 'sr_5band_evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("5-BAND SUPER-RESOLUTION MODEL EVALUATION")
    print("=" * 70)
    
    # Check files exist
    if not test_dir.exists():
        print(f"Error: Test data not found at {test_dir}")
        return
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load test data from mmap
    print("\nLoading test data...")
    lr_10m_all = np.load(test_dir / 'lr.npy', mmap_mode='r')
    lr_20m_all = np.load(test_dir / 'lr_20m.npy', mmap_mode='r')
    hr_10m_all = np.load(test_dir / 'hr_10m.npy', mmap_mode='r')
    hr_20m_all = np.load(test_dir / 'hr_20m.npy', mmap_mode='r')
    
    n_samples = lr_10m_all.shape[0]
    print(f"  Test samples: {n_samples}")
    print(f"  LR 10m shape: {lr_10m_all.shape}")
    print(f"  HR 10m shape: {hr_10m_all.shape}")
    
    # Load model
    print("\nLoading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    model, checkpoint = load_model(checkpoint_path, scale_factor_10m=3.33, scale_factor_20m=3.33)
    model = model.to(device)
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    print(f"  Training val_loss: {checkpoint['val_loss']:.4f}")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS AND METRICS")
    print("=" * 70)
    
    np.random.seed(42)
    num_visualizations = min(10, n_samples)  # Save visualizations for 10 samples
    viz_indices = np.random.choice(n_samples, num_visualizations, replace=False)
    viz_indices_set = set(viz_indices)
    
    all_results = []
    
    for i in range(n_samples):
        save_viz = i in viz_indices_set
        
        # Prepare sample
        lr_10m = np.array(lr_10m_all[i])
        lr_20m = np.array(lr_20m_all[i])
        hr_10m = np.array(hr_10m_all[i])
        hr_20m = np.array(hr_20m_all[i])
        
        sample_data = prepare_sample_mmap(lr_10m, lr_20m, hr_10m, hr_20m)
        
        if save_viz:
            save_path = output_dir / f'comparison_sample_{i}.png'
            print(f"\n[{i+1}/{n_samples}] Processing sample {i} (saving visualization)...")
        else:
            save_path = None
            if (i+1) % 20 == 0:
                print(f"[{i+1}/{n_samples}] Processing samples...")
        
        results = visualize_comparison_single_model(sample_data, model, i, save_path)
        all_results.append(results)
        
        if save_viz:
            improvement_10m = results['psnr_sr_10m'] - results['psnr_bilinear_10m']
            improvement_20m = results['psnr_sr_20m'] - results['psnr_bilinear_20m']
            print(f"  SR Model:  10m={results['psnr_sr_10m']:.2f}dB, 20m={results['psnr_sr_20m']:.2f}dB")
            print(f"  Bilinear:  10m={results['psnr_bilinear_10m']:.2f}dB, 20m={results['psnr_bilinear_20m']:.2f}dB")
            print(f"  Improvement: 10m={improvement_10m:+.2f}dB, 20m={improvement_20m:+.2f}dB")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print(f"SUMMARY STATISTICS ({len(all_results)} Test Samples)")
    print("=" * 70)
    
    # Compute averages
    metric_names = ['psnr', 'ssim', 'scc', 'rmse', 'ergas', 'sid']
    
    avg_metrics = {
        'sr_10m': {m: np.mean([r['metrics_sr_10m'][m] for r in all_results]) for m in metric_names},
        'sr_20m': {m: np.mean([r['metrics_sr_20m'][m] for r in all_results]) for m in metric_names},
        'bilinear_10m': {m: np.mean([r['metrics_bilinear_10m'][m] for r in all_results]) for m in metric_names},
        'bilinear_20m': {m: np.mean([r['metrics_bilinear_20m'][m] for r in all_results]) for m in metric_names},
    }
    
    std_metrics = {
        'sr_10m': {m: np.std([r['metrics_sr_10m'][m] for r in all_results]) for m in metric_names},
        'sr_20m': {m: np.std([r['metrics_sr_20m'][m] for r in all_results]) for m in metric_names},
        'bilinear_10m': {m: np.std([r['metrics_bilinear_10m'][m] for r in all_results]) for m in metric_names},
        'bilinear_20m': {m: np.std([r['metrics_bilinear_20m'][m] for r in all_results]) for m in metric_names},
    }
    
    print(f"\n{'─' * 90}")
    print("BILINEAR BASELINE vs SR MODEL:")
    print(f"{'─' * 90}")
    
    # Print comprehensive metrics table
    print(f"\n{'METRIC':<10} {'Bilinear 10m':>14} {'SR 10m':>14} {'Gain 10m':>12} {'Bilinear 20m':>14} {'SR 20m':>14} {'Gain 20m':>12}")
    print(f"{'─' * 90}")
    
    for metric in metric_names:
        bil_10m = avg_metrics['bilinear_10m'][metric]
        sr_10m = avg_metrics['sr_10m'][metric]
        bil_20m = avg_metrics['bilinear_20m'][metric]
        sr_20m = avg_metrics['sr_20m'][metric]
        
        # Compute gain (positive = SR better)
        if metric in ['psnr', 'ssim', 'scc']:  # Higher is better
            gain_10m = sr_10m - bil_10m
            gain_20m = sr_20m - bil_20m
        else:  # Lower is better (rmse, ergas, sid)
            gain_10m = bil_10m - sr_10m
            gain_20m = bil_20m - sr_20m
        
        if metric == 'psnr':
            print(f"{metric.upper():<10} {bil_10m:>14.2f} {sr_10m:>14.2f} {gain_10m:>+12.2f} {bil_20m:>14.2f} {sr_20m:>14.2f} {gain_20m:>+12.2f}")
        else:
            print(f"{metric.upper():<10} {bil_10m:>14.4f} {sr_10m:>14.4f} {gain_10m:>+12.4f} {bil_20m:>14.4f} {sr_20m:>14.4f} {gain_20m:>+12.4f}")
    
    print(f"{'─' * 90}")
    
    # Per-band PSNR summary
    print(f"\n{'─' * 90}")
    print("PER-BAND PSNR COMPARISON")
    print(f"{'─' * 90}")
    bands = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR']
    for i, band in enumerate(bands):
        avg_sr = np.mean([r['psnr_sr_bands'][i] for r in all_results])
        avg_bilinear = np.mean([r['psnr_bilinear_bands'][i] for r in all_results])
        std_sr = np.std([r['psnr_sr_bands'][i] for r in all_results])
        gain = avg_sr - avg_bilinear
        winner = "SR" if gain > 0 else "Bilinear"
        print(f"  {band:8s}: Bilinear={avg_bilinear:.2f}dB, SR={avg_sr:.2f}dB (±{std_sr:.2f}) → {winner} ({gain:+.2f}dB)")
    
    # Overall summary
    print(f"\n{'═' * 90}")
    print("OVERALL ASSESSMENT")
    print(f"{'═' * 90}")
    
    avg_psnr_gain_10m = avg_metrics['sr_10m']['psnr'] - avg_metrics['bilinear_10m']['psnr']
    avg_psnr_gain_20m = avg_metrics['sr_20m']['psnr'] - avg_metrics['bilinear_20m']['psnr']
    avg_ssim_gain_10m = avg_metrics['sr_10m']['ssim'] - avg_metrics['bilinear_10m']['ssim']
    avg_ssim_gain_20m = avg_metrics['sr_20m']['ssim'] - avg_metrics['bilinear_20m']['ssim']
    
    print(f"\n  10m Bands: SR {'improves' if avg_psnr_gain_10m > 0 else 'degrades'} over bilinear by {abs(avg_psnr_gain_10m):.2f} dB PSNR, {abs(avg_ssim_gain_10m):.4f} SSIM")
    print(f"  20m Bands: SR {'improves' if avg_psnr_gain_20m > 0 else 'degrades'} over bilinear by {abs(avg_psnr_gain_20m):.2f} dB PSNR, {abs(avg_ssim_gain_20m):.4f} SSIM")
    
    if avg_psnr_gain_10m < 0 or avg_psnr_gain_20m < 0:
        print(f"\n  ⚠️  WARNING: Model is performing WORSE than bilinear interpolation!")
        print(f"      This indicates a potential issue with training or evaluation.")
    else:
        print(f"\n  ✅ Model is improving over bilinear baseline.")
    
    print(f"\n{'═' * 90}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'═' * 90}")


if __name__ == '__main__':
    main()
