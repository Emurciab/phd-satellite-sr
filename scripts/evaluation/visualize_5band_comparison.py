#!/usr/bin/env python3
"""
Visualize 5-Band Super-Resolution Results from Both Models

Compares:
1. Multi-Resolution model (10m→3m at 256×256, 20m→6m at 128×128)
2. Uniform 3m model (all bands→3m at 256×256)

Uses validation data (unseen during training) to evaluate generalization.
"""

import sys
import os

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

import pickle
import numpy as np
import torch
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
from convert_cache_to_5band import TrainingSample5Band  # Needed for pickle


def load_model(checkpoint_path, scale_factor_10m=3.32, scale_factor_20m=3.37):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
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
    return model


def prepare_sample(sample, sequence_length=3):
    """Prepare a sample for inference."""
    # Get arrays - (T, H, W, C) format
    lr_10m = sample.lr_sequence.copy()
    lr_20m = sample.lr_20m_sequence.copy()
    hr_10m = sample.hr_10m_sequence.copy()
    hr_20m = sample.hr_20m_sequence.copy()
    hr_full = sample.hr_sequence.copy()  # Full 5-band at 3m
    
    # Handle temporal dimension
    T = lr_10m.shape[0]
    if T > sequence_length:
        start = (T - sequence_length) // 2  # Center crop
        lr_10m = lr_10m[start:start + sequence_length]
        lr_20m = lr_20m[start:start + sequence_length]
        hr_10m = hr_10m[start:start + sequence_length]
        hr_20m = hr_20m[start:start + sequence_length]
        hr_full = hr_full[start:start + sequence_length]
    elif T < sequence_length:
        while lr_10m.shape[0] < sequence_length:
            lr_10m = np.concatenate([lr_10m, lr_10m[-1:]], axis=0)
            lr_20m = np.concatenate([lr_20m, lr_20m[-1:]], axis=0)
            hr_10m = np.concatenate([hr_10m, hr_10m[-1:]], axis=0)
            hr_20m = hr_20m[start:start + sequence_length]
            hr_full = np.concatenate([hr_full, hr_full[-1:]], axis=0)
    
    # Convert to tensors: (T, H, W, C) -> (1, T, C, H, W)
    lr_10m_t = torch.from_numpy(lr_10m).permute(0, 3, 1, 2).unsqueeze(0).float()
    lr_20m_t = torch.from_numpy(lr_20m).permute(0, 3, 1, 2).unsqueeze(0).float()
    
    # Ground truth for middle frame
    mid_idx = sequence_length // 2
    gt_10m = hr_10m[mid_idx]  # (256, 256, 3)
    gt_20m = hr_20m[mid_idx]  # (128, 128, 2)
    gt_full = hr_full[mid_idx]  # (256, 256, 5)
    
    # LR for display (middle frame)
    lr_10m_display = lr_10m[mid_idx]  # (77, 77, 3)
    lr_20m_display = lr_20m[mid_idx]  # (38, 38, 2)
    
    return {
        'lr_10m': lr_10m_t,
        'lr_20m': lr_20m_t,
        'lr_10m_display': lr_10m_display,
        'lr_20m_display': lr_20m_display,
        'gt_10m': gt_10m,
        'gt_20m': gt_20m,
        'gt_full': gt_full
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


def visualize_comparison(sample_data, model_multires, model_uniform, sample_idx, save_path=None):
    """Create visualization comparing both models."""
    device = next(model_multires.parameters()).device
    
    # Run inference
    with torch.no_grad():
        lr_10m = sample_data['lr_10m'].to(device)
        lr_20m = sample_data['lr_20m'].to(device)
        
        # Multi-resolution model - specify target sizes to get 256×256 / 128×128
        sr_10m_multi, sr_20m_multi = model_multires(lr_10m, lr_20m, 
                                                     target_size_10m=(256, 256),
                                                     target_size_20m=(128, 128))
        sr_10m_multi = sr_10m_multi.cpu().numpy()[0].transpose(1, 2, 0)  # (256, 256, 3)
        sr_20m_multi = sr_20m_multi.cpu().numpy()[0].transpose(1, 2, 0)  # (128, 128, 2)
        
        # Uniform 3m model - specify target sizes to get 256×256 for both
        sr_10m_uniform, sr_20m_uniform = model_uniform(lr_10m, lr_20m,
                                                        target_size_10m=(256, 256),
                                                        target_size_20m=(256, 256))
        sr_10m_uniform = sr_10m_uniform.cpu().numpy()[0].transpose(1, 2, 0)  # (256, 256, 3)
        sr_20m_uniform = sr_20m_uniform.cpu().numpy()[0].transpose(1, 2, 0)  # (256, 256, 2)
    
    # Get ground truth and LR
    gt_10m = sample_data['gt_10m']
    gt_20m = sample_data['gt_20m']
    gt_full = sample_data['gt_full']
    lr_10m_display = sample_data['lr_10m_display']
    lr_20m_display = sample_data['lr_20m_display']
    
    # For uniform model, compare 20m at full resolution
    gt_20m_full = gt_full[:, :, 3:5]  # RedEdge, NIR at 256×256
    
    # Compute BILINEAR BASELINE
    # Upscale LR to HR size using bilinear interpolation
    bilinear_10m = zoom(lr_10m_display, (256/77, 256/77, 1), order=1)  # (256, 256, 3)
    bilinear_20m_multi = zoom(lr_20m_display, (128/38, 128/38, 1), order=1)  # (128, 128, 2) for multi-res comparison
    bilinear_20m_uniform = zoom(lr_20m_display, (256/38, 256/38, 1), order=1)  # (256, 256, 2) for uniform comparison
    
    # Compute ALL metrics for bilinear baseline
    metrics_bilinear_10m = compute_all_metrics(bilinear_10m, gt_10m, scale_ratio=3.33)
    metrics_bilinear_20m_multi = compute_all_metrics(bilinear_20m_multi, gt_20m, scale_ratio=3.33)
    metrics_bilinear_20m_uniform = compute_all_metrics(bilinear_20m_uniform, gt_20m_full, scale_ratio=6.74)
    
    # Compute ALL metrics for both models
    # Scale factors: 10m bands at 3.33×, 20m bands at 3.33× (multi) or 6.74× (uniform)
    metrics_multi_10m = compute_all_metrics(sr_10m_multi, gt_10m, scale_ratio=3.33)
    metrics_multi_20m = compute_all_metrics(sr_20m_multi, gt_20m, scale_ratio=3.33)
    metrics_uniform_10m = compute_all_metrics(sr_10m_uniform, gt_10m, scale_ratio=3.33)
    metrics_uniform_20m = compute_all_metrics(sr_20m_uniform, gt_20m_full, scale_ratio=6.74)
    
    # Extract PSNR for backward compatibility
    psnr_multi_10m = metrics_multi_10m['psnr']
    psnr_multi_20m = metrics_multi_20m['psnr']
    psnr_uniform_10m = metrics_uniform_10m['psnr']
    psnr_uniform_20m = metrics_uniform_20m['psnr']
    psnr_bilinear_10m = metrics_bilinear_10m['psnr']
    
    # Create figure - larger to accommodate metrics table
    fig = plt.figure(figsize=(24, 18))
    
    # Title with key metrics
    fig.suptitle(f'Super-Resolution Comparison - Sample {sample_idx}\n'
                 f'Bilinear PSNR: 10m={psnr_bilinear_10m:.2f}dB | '
                 f'Multi-Res PSNR: 10m={psnr_multi_10m:.2f}dB, 20m={psnr_multi_20m:.2f}dB | '
                 f'Uniform 3m PSNR: 10m={psnr_uniform_10m:.2f}dB, 20m={psnr_uniform_20m:.2f}dB\n'
                 f'Multi-Res SSIM: 10m={metrics_multi_10m["ssim"]:.4f}, 20m={metrics_multi_20m["ssim"]:.4f} | '
                 f'Uniform 3m SSIM: 10m={metrics_uniform_10m["ssim"]:.4f}, 20m={metrics_uniform_20m["ssim"]:.4f}',
                 fontsize=12, fontweight='bold')
    
    # Row 1: RGB composites
    ax1 = fig.add_subplot(4, 5, 1)
    lr_rgb = create_rgb_composite(lr_10m_display)
    ax1.imshow(lr_rgb)
    ax1.set_title(f'LR Input\n({lr_10m_display.shape[0]}×{lr_10m_display.shape[1]})')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(4, 5, 2)
    ax2.imshow(create_rgb_composite(sr_10m_multi))
    ax2.set_title(f'Multi-Res SR\n(256×256)')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(4, 5, 3)
    ax3.imshow(create_rgb_composite(sr_10m_uniform))
    ax3.set_title(f'Uniform 3m SR\n(256×256)')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(4, 5, 4)
    ax4.imshow(create_rgb_composite(gt_10m))
    ax4.set_title(f'Ground Truth\n(256×256)')
    ax4.axis('off')
    
    # Difference map for 10m
    ax5 = fig.add_subplot(4, 5, 5)
    diff_multi = np.abs(sr_10m_multi - gt_10m).mean(axis=-1)
    diff_uniform = np.abs(sr_10m_uniform - gt_10m).mean(axis=-1)
    diff_comparison = diff_uniform - diff_multi  # Positive = multi-res better
    im = ax5.imshow(diff_comparison, cmap='RdBu', vmin=-0.05, vmax=0.05)
    ax5.set_title('Error Diff\n(Blue=Multi better)')
    ax5.axis('off')
    plt.colorbar(im, ax=ax5, fraction=0.046)
    
    # Row 2: Individual 10m bands
    band_names_10m = ['Blue', 'Green', 'Red']
    for i, name in enumerate(band_names_10m):
        ax = fig.add_subplot(4, 5, 6 + i)
        comparison = np.hstack([
            normalize_for_display(sr_10m_multi[:, :, i:i+1].repeat(3, axis=-1)),
            normalize_for_display(sr_10m_uniform[:, :, i:i+1].repeat(3, axis=-1)),
            normalize_for_display(gt_10m[:, :, i:i+1].repeat(3, axis=-1))
        ])
        ax.imshow(comparison, cmap='gray')
        ax.set_title(f'{name}\nMulti | Uniform | GT')
        ax.axis('off')
    
    # Row 2 continued: Bilinear upsampled LR for reference
    ax = fig.add_subplot(4, 5, 9)
    lr_bilinear = zoom(lr_10m_display, (256/77, 256/77, 1), order=1)
    ax.imshow(create_rgb_composite(lr_bilinear))
    ax.set_title('Bilinear Upscale\n(baseline)')
    ax.axis('off')
    
    # PSNR vs Bilinear
    psnr_bilinear = compute_psnr(lr_bilinear, gt_10m)
    ax = fig.add_subplot(4, 5, 10)
    ax.axis('off')
    ax.text(0.5, 0.5, f'Bilinear PSNR:\n{psnr_bilinear:.2f} dB\n\n'
            f'Multi-Res gain:\n+{psnr_multi_10m - psnr_bilinear:.2f} dB\n\n'
            f'Uniform gain:\n+{psnr_uniform_10m - psnr_bilinear:.2f} dB',
            ha='center', va='center', fontsize=11, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Row 3: 20m bands comparison (RedEdge)
    ax = fig.add_subplot(4, 5, 11)
    sr_20m_multi_up = zoom(sr_20m_multi[:, :, 0], 2, order=1)  # Upsample to 256×256
    comparison_re = np.hstack([
        normalize_for_display(sr_20m_multi_up[:, :, np.newaxis].repeat(3, axis=-1)),
        normalize_for_display(sr_20m_uniform[:, :, 0:1].repeat(3, axis=-1)),
        normalize_for_display(gt_20m_full[:, :, 0:1].repeat(3, axis=-1))
    ])
    ax.imshow(comparison_re, cmap='gray')
    ax.set_title('RedEdge (B5)\nMulti(↑2×) | Uniform | GT')
    ax.axis('off')
    
    # NIR band
    ax = fig.add_subplot(4, 5, 12)
    sr_20m_multi_up_nir = zoom(sr_20m_multi[:, :, 1], 2, order=1)
    comparison_nir = np.hstack([
        normalize_for_display(sr_20m_multi_up_nir[:, :, np.newaxis].repeat(3, axis=-1)),
        normalize_for_display(sr_20m_uniform[:, :, 1:2].repeat(3, axis=-1)),
        normalize_for_display(gt_20m_full[:, :, 1:2].repeat(3, axis=-1))
    ])
    ax.imshow(comparison_nir, cmap='gray')
    ax.set_title('NIR (B8A)\nMulti(↑2×) | Uniform | GT')
    ax.axis('off')
    
    # Row 3 continued: False color composites
    ax = fig.add_subplot(4, 5, 13)
    ax.imshow(create_false_color(sr_10m_multi, sr_20m_multi))
    ax.set_title('False Color (NIR-R-G)\nMulti-Res')
    ax.axis('off')
    
    ax = fig.add_subplot(4, 5, 14)
    ax.imshow(create_false_color(sr_10m_uniform, sr_20m_uniform))
    ax.set_title('False Color (NIR-R-G)\nUniform 3m')
    ax.axis('off')
    
    ax = fig.add_subplot(4, 5, 15)
    ax.imshow(create_false_color(gt_10m, gt_20m_full))
    ax.set_title('False Color (NIR-R-G)\nGround Truth')
    ax.axis('off')
    
    # Row 4: Zoomed regions
    zoom_size = 64
    center = 128
    slice_h = slice(center - zoom_size//2, center + zoom_size//2)
    slice_w = slice(center - zoom_size//2, center + zoom_size//2)
    
    ax = fig.add_subplot(4, 5, 16)
    ax.imshow(create_rgb_composite(sr_10m_multi[slice_h, slice_w]))
    ax.set_title('Zoomed Multi-Res\n(64×64 crop)')
    ax.axis('off')
    
    ax = fig.add_subplot(4, 5, 17)
    ax.imshow(create_rgb_composite(sr_10m_uniform[slice_h, slice_w]))
    ax.set_title('Zoomed Uniform 3m\n(64×64 crop)')
    ax.axis('off')
    
    ax = fig.add_subplot(4, 5, 18)
    ax.imshow(create_rgb_composite(gt_10m[slice_h, slice_w]))
    ax.set_title('Zoomed GT\n(64×64 crop)')
    ax.axis('off')
    
    # Per-band PSNR bar chart
    ax = fig.add_subplot(4, 5, 19)
    bands = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR']
    
    psnr_multi = [
        compute_psnr(sr_10m_multi[:, :, 0], gt_10m[:, :, 0]),
        compute_psnr(sr_10m_multi[:, :, 1], gt_10m[:, :, 1]),
        compute_psnr(sr_10m_multi[:, :, 2], gt_10m[:, :, 2]),
        compute_psnr(sr_20m_multi[:, :, 0], gt_20m[:, :, 0]),
        compute_psnr(sr_20m_multi[:, :, 1], gt_20m[:, :, 1]),
    ]
    psnr_uniform = [
        compute_psnr(sr_10m_uniform[:, :, 0], gt_10m[:, :, 0]),
        compute_psnr(sr_10m_uniform[:, :, 1], gt_10m[:, :, 1]),
        compute_psnr(sr_10m_uniform[:, :, 2], gt_10m[:, :, 2]),
        compute_psnr(sr_20m_uniform[:, :, 0], gt_20m_full[:, :, 0]),
        compute_psnr(sr_20m_uniform[:, :, 1], gt_20m_full[:, :, 1]),
    ]
    
    # Per-band PSNR for bilinear
    psnr_bilinear = [
        compute_psnr(bilinear_10m[:, :, 0], gt_10m[:, :, 0]),
        compute_psnr(bilinear_10m[:, :, 1], gt_10m[:, :, 1]),
        compute_psnr(bilinear_10m[:, :, 2], gt_10m[:, :, 2]),
        compute_psnr(bilinear_20m_uniform[:, :, 0], gt_20m_full[:, :, 0]),
        compute_psnr(bilinear_20m_uniform[:, :, 1], gt_20m_full[:, :, 1]),
    ]
    
    x = np.arange(len(bands))
    width = 0.25
    ax.bar(x - width, psnr_bilinear, width, label='Bilinear', color='gray', alpha=0.7)
    ax.bar(x, psnr_multi, width, label='Multi-Res', color='steelblue')
    ax.bar(x + width, psnr_uniform, width, label='Uniform 3m', color='darkorange')
    ax.set_ylabel('PSNR (dB)')
    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=45)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title('Per-Band PSNR')
    ax.set_ylim([20, 50])
    ax.grid(axis='y', alpha=0.3)
    
    # Summary text with ALL metrics including bilinear
    ax = fig.add_subplot(4, 5, 20)
    ax.axis('off')
    
    # Create comprehensive metrics summary
    summary_text = (
        f"═══ QUALITY METRICS ═══\n\n"
        f"Bilinear Baseline:\n"
        f"  10m bands:\n"
        f"    PSNR:  {metrics_bilinear_10m['psnr']:.2f} dB\n"
        f"    SSIM:  {metrics_bilinear_10m['ssim']:.4f}\n"
        f"    SCC:   {metrics_bilinear_10m['scc']:.4f}\n"
        f"    RMSE:  {metrics_bilinear_10m['rmse']:.4f}\n"
        f"    ERGAS: {metrics_bilinear_10m['ergas']:.2f}\n"
        f"  20m bands:\n"
        f"    PSNR:  {metrics_bilinear_20m_uniform['psnr']:.2f} dB\n"
        f"    SSIM:  {metrics_bilinear_20m_uniform['ssim']:.4f}\n"
        f"    ERGAS: {metrics_bilinear_20m_uniform['ergas']:.2f}\n\n"
        f"Multi-Resolution Model:\n"
        f"  10m: PSNR={metrics_multi_10m['psnr']:.2f}dB\n"
        f"       SSIM={metrics_multi_10m['ssim']:.4f}\n"
        f"  20m: PSNR={metrics_multi_20m['psnr']:.2f}dB\n"
        f"       SSIM={metrics_multi_20m['ssim']:.4f}\n\n"
        f"Uniform 3m Model:\n"
        f"  10m: PSNR={metrics_uniform_10m['psnr']:.2f}dB\n"
        f"       SSIM={metrics_uniform_10m['ssim']:.4f}\n"
        f"  20m: PSNR={metrics_uniform_20m['psnr']:.2f}dB\n"
        f"       SSIM={metrics_uniform_20m['ssim']:.4f}\n"
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
        # PSNR
        'psnr_multi_10m': psnr_multi_10m,
        'psnr_multi_20m': psnr_multi_20m,
        'psnr_uniform_10m': psnr_uniform_10m,
        'psnr_uniform_20m': psnr_uniform_20m,
        'psnr_multi_bands': psnr_multi,
        'psnr_uniform_bands': psnr_uniform,
        'psnr_bilinear_bands': psnr_bilinear,
        # Full metrics
        'metrics_multi_10m': metrics_multi_10m,
        'metrics_multi_20m': metrics_multi_20m,
        'metrics_uniform_10m': metrics_uniform_10m,
        'metrics_uniform_20m': metrics_uniform_20m,
        'metrics_bilinear_10m': metrics_bilinear_10m,
        'metrics_bilinear_20m': metrics_bilinear_20m_uniform,
    }


def main():
    # Paths
    data_path = project_root / 'data' / 'sr_training' / '5band_full' / 'superdove_5band' / 'dataset.pkl'
    checkpoint_multires = project_root / 'checkpoints' / '5band_multires_full' / 'best_model.pt'
    checkpoint_uniform = project_root / 'checkpoints' / '5band_uniform_3m_full' / 'best_model.pt'
    output_dir = project_root / 'results' / 'sr_5band_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("5-BAND SUPER-RESOLUTION MODEL COMPARISON")
    print("=" * 70)
    
    # Check files exist
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        return
    if not checkpoint_multires.exists():
        print(f"Error: Multi-res checkpoint not found at {checkpoint_multires}")
        return
    if not checkpoint_uniform.exists():
        print(f"Error: Uniform checkpoint not found at {checkpoint_uniform}")
        return
    
    # Load data
    print("\nLoading validation data...")
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    samples = dataset['samples']
    
    # Filter valid samples
    valid_samples = [s for s in samples if 
                    s.lr_sequence is not None and 
                    s.lr_20m_sequence is not None and
                    s.hr_10m_sequence is not None and
                    s.hr_20m_sequence is not None and
                    s.hr_sequence is not None]
    
    print(f"  Total samples: {len(samples)}")
    print(f"  Valid samples: {len(valid_samples)}")
    
    # Use validation split (last 20%)
    val_start = int(len(valid_samples) * 0.8)
    val_samples = valid_samples[val_start:]
    print(f"  Validation samples: {len(val_samples)}")
    
    # Load models
    print("\nLoading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    model_multires = load_model(checkpoint_multires, scale_factor_10m=3.32, scale_factor_20m=3.37)
    model_multires = model_multires.to(device)
    print(f"  Multi-res model loaded ({sum(p.numel() for p in model_multires.parameters()):,} params)")
    
    model_uniform = load_model(checkpoint_uniform, scale_factor_10m=3.32, scale_factor_20m=6.74)
    model_uniform = model_uniform.to(device)
    print(f"  Uniform 3m model loaded ({sum(p.numel() for p in model_uniform.parameters()):,} params)")
    
    # Visualize random samples
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    np.random.seed(42)
    num_samples_to_evaluate = 100  # Evaluate metrics on 100 samples
    num_visualizations_to_save = 10  # But only save 10 visualizations
    
    # Select samples for evaluation
    num_to_use = min(num_samples_to_evaluate, len(val_samples))
    sample_indices = np.random.choice(len(val_samples), num_to_use, replace=False)
    
    # Select which ones to save visualizations for
    viz_indices = set(sample_indices[:num_visualizations_to_save])
    
    all_results = []
    
    for i, idx in enumerate(sample_indices):
        save_viz = idx in viz_indices
        progress = f"[{i+1}/{num_to_use}]"
        
        sample = val_samples[idx]
        sample_data = prepare_sample(sample)
        
        if save_viz:
            save_path = output_dir / f'comparison_sample_{idx}.png'
            print(f"\n{progress} Processing sample {idx} (saving visualization)...")
        else:
            save_path = None
            if (i+1) % 20 == 0:
                print(f"{progress} Processing samples...")
        
        results = visualize_comparison(sample_data, model_multires, model_uniform, idx, save_path)
        all_results.append(results)
        
        if save_viz:
            print(f"  Multi-Res: 10m={results['psnr_multi_10m']:.2f}dB, 20m={results['psnr_multi_20m']:.2f}dB")
            print(f"  Uniform:   10m={results['psnr_uniform_10m']:.2f}dB, 20m={results['psnr_uniform_20m']:.2f}dB")
            print(f"  Bilinear:  10m={results['metrics_bilinear_10m']['psnr']:.2f}dB, 20m={results['metrics_bilinear_20m']['psnr']:.2f}dB")
    
    print(f"\nEvaluated {len(all_results)} samples, saved {len(viz_indices)} visualizations")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print(f"SUMMARY STATISTICS ({len(all_results)} Validation Samples)")
    print("=" * 70)
    
    # Compute averages for all metrics
    metric_names = ['psnr', 'ssim', 'scc', 'rmse', 'ergas', 'sid']
    
    avg_metrics = {
        'multi_10m': {m: np.mean([r['metrics_multi_10m'][m] for r in all_results]) for m in metric_names},
        'multi_20m': {m: np.mean([r['metrics_multi_20m'][m] for r in all_results]) for m in metric_names},
        'uniform_10m': {m: np.mean([r['metrics_uniform_10m'][m] for r in all_results]) for m in metric_names},
        'uniform_20m': {m: np.mean([r['metrics_uniform_20m'][m] for r in all_results]) for m in metric_names},
        'bilinear_10m': {m: np.mean([r['metrics_bilinear_10m'][m] for r in all_results]) for m in metric_names},
        'bilinear_20m': {m: np.mean([r['metrics_bilinear_20m'][m] for r in all_results]) for m in metric_names},
    }
    
    print(f"\n{'─' * 90}")
    print("BASELINE (Bilinear Interpolation) vs SR Models:")
    print(f"{'─' * 90}")
    print(f"  Bilinear 10m avg: PSNR={avg_metrics['bilinear_10m']['psnr']:.2f}dB, SSIM={avg_metrics['bilinear_10m']['ssim']:.4f}, SID={avg_metrics['bilinear_10m']['sid']:.4f}")
    print(f"  Bilinear 20m avg: PSNR={avg_metrics['bilinear_20m']['psnr']:.2f}dB, SSIM={avg_metrics['bilinear_20m']['ssim']:.4f}, SID={avg_metrics['bilinear_20m']['sid']:.4f}")
    
    # Print comprehensive metrics table
    print(f"\n{'─' * 105}")
    print(f"{'METRIC':<10} {'Bilinear 10m':>12} {'Bilinear 20m':>12} {'Multi-Res 10m':>14} {'Multi-Res 20m':>14} {'Uniform 10m':>12} {'Uniform 20m':>12}")
    print(f"{'─' * 105}")
    
    # PSNR (higher is better)
    print(f"{'PSNR (dB)':<10} {avg_metrics['bilinear_10m']['psnr']:>12.2f} {avg_metrics['bilinear_20m']['psnr']:>12.2f} "
          f"{avg_metrics['multi_10m']['psnr']:>14.2f} {avg_metrics['multi_20m']['psnr']:>14.2f} "
          f"{avg_metrics['uniform_10m']['psnr']:>12.2f} {avg_metrics['uniform_20m']['psnr']:>12.2f}")
    
    # SSIM (higher is better, range 0-1)
    print(f"{'SSIM':<10} {avg_metrics['bilinear_10m']['ssim']:>12.4f} {avg_metrics['bilinear_20m']['ssim']:>12.4f} "
          f"{avg_metrics['multi_10m']['ssim']:>14.4f} {avg_metrics['multi_20m']['ssim']:>14.4f} "
          f"{avg_metrics['uniform_10m']['ssim']:>12.4f} {avg_metrics['uniform_20m']['ssim']:>12.4f}")
    
    # SCC (higher is better, range -1 to 1)
    print(f"{'SCC':<10} {avg_metrics['bilinear_10m']['scc']:>12.4f} {avg_metrics['bilinear_20m']['scc']:>12.4f} "
          f"{avg_metrics['multi_10m']['scc']:>14.4f} {avg_metrics['multi_20m']['scc']:>14.4f} "
          f"{avg_metrics['uniform_10m']['scc']:>12.4f} {avg_metrics['uniform_20m']['scc']:>12.4f}")
    
    # RMSE (lower is better)
    print(f"{'RMSE':<10} {avg_metrics['bilinear_10m']['rmse']:>12.4f} {avg_metrics['bilinear_20m']['rmse']:>12.4f} "
          f"{avg_metrics['multi_10m']['rmse']:>14.4f} {avg_metrics['multi_20m']['rmse']:>14.4f} "
          f"{avg_metrics['uniform_10m']['rmse']:>12.4f} {avg_metrics['uniform_20m']['rmse']:>12.4f}")
    
    # ERGAS (lower is better)
    print(f"{'ERGAS':<10} {avg_metrics['bilinear_10m']['ergas']:>12.2f} {avg_metrics['bilinear_20m']['ergas']:>12.2f} "
          f"{avg_metrics['multi_10m']['ergas']:>14.2f} {avg_metrics['multi_20m']['ergas']:>14.2f} "
          f"{avg_metrics['uniform_10m']['ergas']:>12.2f} {avg_metrics['uniform_20m']['ergas']:>12.2f}")
    
    # SID (lower is better)
    print(f"{'SID':<10} {avg_metrics['bilinear_10m']['sid']:>12.4f} {avg_metrics['bilinear_20m']['sid']:>12.4f} "
          f"{avg_metrics['multi_10m']['sid']:>14.4f} {avg_metrics['multi_20m']['sid']:>14.4f} "
          f"{avg_metrics['uniform_10m']['sid']:>12.4f} {avg_metrics['uniform_20m']['sid']:>12.4f}")
    
    print(f"{'─' * 105}")
    
    # Winner analysis - now including bilinear
    print(f"\n{'─' * 105}")
    print("WINNER ANALYSIS (Bilinear vs Multi-Res vs Uniform)")
    print(f"{'─' * 105}")
    
    # For 10m bands
    print("\n10m Bands:")
    for metric in metric_names:
        bilinear_val = avg_metrics['bilinear_10m'][metric]
        multi_val = avg_metrics['multi_10m'][metric]
        uniform_val = avg_metrics['uniform_10m'][metric]
        
        # Determine winner (PSNR, SSIM, SCC: higher is better; RMSE, ERGAS, SID: lower is better)
        if metric in ['psnr', 'ssim', 'scc']:
            vals = {'Bilinear': bilinear_val, 'Multi-Res': multi_val, 'Uniform': uniform_val}
            winner = max(vals, key=vals.get)
            best_val = vals[winner]
            second_best = sorted(vals.values(), reverse=True)[1]
            gain = best_val - second_best
        else:  # rmse, ergas, sid (lower is better)
            vals = {'Bilinear': bilinear_val, 'Multi-Res': multi_val, 'Uniform': uniform_val}
            winner = min(vals, key=vals.get)
            best_val = vals[winner]
            second_best = sorted(vals.values())[1]
            gain = second_best - best_val
        
        print(f"  {metric.upper():<6}: {winner} (gain: {gain:.4f})")
    
    # For 20m bands
    print("\n20m Bands:")
    for metric in metric_names:
        bilinear_val = avg_metrics['bilinear_20m'][metric]
        multi_val = avg_metrics['multi_20m'][metric]
        uniform_val = avg_metrics['uniform_20m'][metric]
        
        if metric in ['psnr', 'ssim', 'scc']:
            vals = {'Bilinear': bilinear_val, 'Multi-Res': multi_val, 'Uniform': uniform_val}
            winner = max(vals, key=vals.get)
            best_val = vals[winner]
            second_best = sorted(vals.values(), reverse=True)[1]
            gain = best_val - second_best
        else:  # rmse, ergas, sid (lower is better)
            vals = {'Bilinear': bilinear_val, 'Multi-Res': multi_val, 'Uniform': uniform_val}
            winner = min(vals, key=vals.get)
            best_val = vals[winner]
            second_best = sorted(vals.values())[1]
            gain = second_best - best_val
        
        print(f"  {metric.upper():<6}: {winner} (gain: {gain:.4f})")
    
    # Per-band PSNR summary
    print(f"\n{'─' * 90}")
    print("PER-BAND PSNR COMPARISON")
    print(f"{'─' * 105}")
    bands = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR']
    for i, band in enumerate(bands):
        avg_multi = np.mean([r['psnr_multi_bands'][i] for r in all_results])
        avg_uniform = np.mean([r['psnr_uniform_bands'][i] for r in all_results])
        winner = "Multi-Res" if avg_multi > avg_uniform else "Uniform"
        diff = abs(avg_multi - avg_uniform)
        print(f"  {band:8s}: Multi={avg_multi:.2f}dB, Uniform={avg_uniform:.2f}dB → {winner} (+{diff:.2f}dB)")
    
    # SR improvement over bilinear
    print(f"\n{'─' * 105}")
    print("SR IMPROVEMENT OVER BILINEAR BASELINE")
    print(f"{'─' * 105}")
    
    print("\n10m Bands:")
    for metric in metric_names:
        bilinear_val = avg_metrics['bilinear_10m'][metric]
        multi_val = avg_metrics['multi_10m'][metric]
        uniform_val = avg_metrics['uniform_10m'][metric]
        
        if metric in ['psnr', 'ssim', 'scc']:
            multi_gain = multi_val - bilinear_val
            uniform_gain = uniform_val - bilinear_val
            print(f"  {metric.upper():<6}: Multi-Res {multi_gain:+.4f}, Uniform {uniform_gain:+.4f}")
        else:  # rmse, ergas, sid (lower is better, so positive gain means improvement)
            multi_gain = bilinear_val - multi_val
            uniform_gain = bilinear_val - uniform_val
            print(f"  {metric.upper():<6}: Multi-Res {multi_gain:+.4f}, Uniform {uniform_gain:+.4f}")
    
    print("\n20m Bands:")
    for metric in metric_names:
        bilinear_val = avg_metrics['bilinear_20m'][metric]
        multi_val = avg_metrics['multi_20m'][metric]
        uniform_val = avg_metrics['uniform_20m'][metric]
        
        if metric in ['psnr', 'ssim', 'scc']:
            multi_gain = multi_val - bilinear_val
            uniform_gain = uniform_val - bilinear_val
            print(f"  {metric.upper():<6}: Multi-Res {multi_gain:+.4f}, Uniform {uniform_gain:+.4f}")
        else:  # rmse, ergas, sid (lower is better)
            multi_gain = bilinear_val - multi_val
            uniform_gain = bilinear_val - uniform_val
            print(f"  {metric.upper():<6}: Multi-Res {multi_gain:+.4f}, Uniform {uniform_gain:+.4f}")
    
    print(f"\n{'═' * 105}")
    print(f"Results saved to: {output_dir}")
    print(f"{'═' * 105}")


if __name__ == '__main__':
    main()
