#!/usr/bin/env python3
"""
Comprehensive test script for all trained SR models.

Evaluates with full metrics suite:
- PSNR (Peak Signal-to-Noise Ratio)
- RMSE (Root Mean Square Error)
- SSIM (Structural Similarity Index)
- SCC (Spatial Correlation Coefficient)
- ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)
- SID (Spectral Information Divergence)

Tests:
1. V3 (trained on full dataset)
2. V3 (trained on filtered dataset)
3. MultiRes (trained on full dataset)
4. MultiRes (trained on filtered dataset)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, List
import json
from datetime import datetime
from scipy.ndimage import uniform_filter, laplace

# Import models
from models.satellite_sr_model_v3 import MultiResolutionSatelliteSRv3
from models.satellite_sr_model_v2 import MultiResolutionSatelliteSRv2

# ============================================================
# Quality Metrics (from visualize_5band_comparison.py)
# ============================================================

def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute PSNR between prediction and target."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    max_val = max(pred.max(), target.max(), 1.0)
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Root Mean Square Error."""
    return np.sqrt(np.mean((pred - target) ** 2))


def compute_ssim(pred: np.ndarray, target: np.ndarray, win_size: int = 7) -> float:
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


def compute_scc(pred: np.ndarray, target: np.ndarray) -> float:
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


def compute_ergas(pred: np.ndarray, target: np.ndarray, scale_ratio: float) -> float:
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


def compute_sid(pred: np.ndarray, target: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Spectral Information Divergence (SID).
    
    SID measures the spectral similarity between two images based on 
    information theory. Lower is better (0 = identical spectral signatures).
    
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


def compute_all_metrics(pred: np.ndarray, target: np.ndarray, scale_ratio: float = 3.33) -> Dict[str, float]:
    """Compute all quality metrics."""
    return {
        'psnr': compute_psnr(pred, target),
        'rmse': compute_rmse(pred, target),
        'ssim': compute_ssim(pred, target),
        'scc': compute_scc(pred, target),
        'ergas': compute_ergas(pred, target, scale_ratio),
        'sid': compute_sid(pred, target)
    }


# ============================================================
# Data Loading
# ============================================================

def load_test_data(test_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load test data from mmap files."""
    test_dir = Path(test_dir)
    
    lr = np.load(test_dir / 'lr.npy', mmap_mode='r')
    lr_20m = np.load(test_dir / 'lr_20m.npy', mmap_mode='r')
    hr_10m = np.load(test_dir / 'hr_10m.npy', mmap_mode='r')
    hr_20m = np.load(test_dir / 'hr_20m.npy', mmap_mode='r')
    
    print(f"Loaded test data:")
    print(f"  LR 10m: {lr.shape}")
    print(f"  LR 20m: {lr_20m.shape}")
    print(f"  HR 10m: {hr_10m.shape}")
    print(f"  HR 20m: {hr_20m.shape}")
    
    return lr, lr_20m, hr_10m, hr_20m


# ============================================================
# Model Loading
# ============================================================

def load_v3_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load V3 model from checkpoint."""
    model = MultiResolutionSatelliteSRv3(
        base_channels=48,
        num_encoder_blocks=3,
        in_channels_10m=3,
        in_channels_20m=2,
        num_frames=3
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def load_multires_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load MultiRes model from checkpoint."""
    model = MultiResolutionSatelliteSRv2(
        base_channels=64,
        num_encoder_blocks=4,
        in_channels_10m=3,
        in_channels_20m=2,
        num_frames=3
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


# ============================================================
# Inference
# ============================================================

def bilinear_baseline(lr_10m: np.ndarray, lr_20m: np.ndarray, 
                      target_h: int, target_w: int,
                      target_h_20m: int, target_w_20m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bilinear upscaling baseline."""
    # Convert to torch for interpolation
    lr_10m_t = torch.from_numpy(lr_10m.copy()).float()
    lr_20m_t = torch.from_numpy(lr_20m.copy()).float()
    
    # Handle different input shapes: (T, H, W, C)
    if lr_10m_t.ndim == 4:
        # Use middle frame
        mid = lr_10m_t.shape[0] // 2
        lr_10m_t = lr_10m_t[mid].permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    elif lr_10m_t.ndim == 3:
        lr_10m_t = lr_10m_t.permute(2, 0, 1).unsqueeze(0)
    
    if lr_20m_t.ndim == 4:
        mid = lr_20m_t.shape[0] // 2
        lr_20m_t = lr_20m_t[mid].permute(2, 0, 1).unsqueeze(0)
    elif lr_20m_t.ndim == 3:
        lr_20m_t = lr_20m_t.permute(2, 0, 1).unsqueeze(0)
    
    # Bilinear upscale
    hr_10m_bilinear = F.interpolate(lr_10m_t, size=(target_h, target_w), 
                                     mode='bilinear', align_corners=False)
    hr_20m_bilinear = F.interpolate(lr_20m_t, size=(target_h_20m, target_w_20m), 
                                     mode='bilinear', align_corners=False)
    
    # Convert back to numpy (H, W, C)
    hr_10m_np = hr_10m_bilinear.squeeze(0).permute(1, 2, 0).numpy()
    hr_20m_np = hr_20m_bilinear.squeeze(0).permute(1, 2, 0).numpy()
    
    return hr_10m_np, hr_20m_np


def run_inference(model: torch.nn.Module, lr_10m: np.ndarray, lr_20m: np.ndarray,
                  device: torch.device, target_size_10m: Tuple[int, int] = None,
                  target_size_20m: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Run model inference on a single sample."""
    with torch.no_grad():
        # Prepare input: (T, H, W, C) -> (1, T, C, H, W)
        lr_10m_t = torch.from_numpy(lr_10m.copy()).float().permute(0, 3, 1, 2).unsqueeze(0).to(device)
        lr_20m_t = torch.from_numpy(lr_20m.copy()).float().permute(0, 3, 1, 2).unsqueeze(0).to(device)
        
        # Forward pass
        sr_10m, sr_20m = model(lr_10m_t, lr_20m_t)
        
        # Resize to target if needed
        if target_size_10m is not None:
            sr_10m = F.interpolate(sr_10m, size=target_size_10m, mode='bilinear', align_corners=False)
        if target_size_20m is not None:
            sr_20m = F.interpolate(sr_20m, size=target_size_20m, mode='bilinear', align_corners=False)
        
        # Convert to numpy: (1, C, H, W) -> (H, W, C)
        sr_10m_np = sr_10m.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_20m_np = sr_20m.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        return sr_10m_np, sr_20m_np


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model: torch.nn.Module, 
                   lr_10m_data: np.ndarray, lr_20m_data: np.ndarray,
                   hr_10m_data: np.ndarray, hr_20m_data: np.ndarray,
                   device: torch.device,
                   model_name: str,
                   scale_10m: float = 3.33,
                   scale_20m: float = 3.33) -> Dict:
    """Evaluate model on full test set with all metrics."""
    
    n_samples = len(lr_10m_data)
    print(f"\nEvaluating {model_name} on {n_samples} test samples...")
    
    # Metrics accumulators for model
    metrics_10m = {k: [] for k in ['psnr', 'rmse', 'ssim', 'scc', 'ergas', 'sid']}
    metrics_20m = {k: [] for k in ['psnr', 'rmse', 'ssim', 'scc', 'ergas', 'sid']}
    
    # Metrics accumulators for bilinear
    metrics_bilinear_10m = {k: [] for k in ['psnr', 'rmse', 'ssim', 'scc', 'ergas', 'sid']}
    metrics_bilinear_20m = {k: [] for k in ['psnr', 'rmse', 'ssim', 'scc', 'ergas', 'sid']}
    
    for i in range(n_samples):
        # Get sample
        lr_10m = lr_10m_data[i]  # (T, H, W, C)
        lr_20m = lr_20m_data[i]
        hr_10m_raw = hr_10m_data[i]  # Could be (T, H, W, C) or (H, W, C)
        hr_20m_raw = hr_20m_data[i]
        
        # Extract middle frame from HR if temporal dimension exists
        if hr_10m_raw.ndim == 4:  # (T, H, W, C)
            mid = hr_10m_raw.shape[0] // 2
            hr_10m = hr_10m_raw[mid]  # (H, W, C)
        else:
            hr_10m = hr_10m_raw
            
        if hr_20m_raw.ndim == 4:
            mid = hr_20m_raw.shape[0] // 2
            hr_20m = hr_20m_raw[mid]
        else:
            hr_20m = hr_20m_raw
        
        target_h, target_w = hr_10m.shape[:2]
        target_h_20m, target_w_20m = hr_20m.shape[:2]
        
        # Model prediction (resize to match target)
        sr_10m, sr_20m = run_inference(model, lr_10m, lr_20m, device,
                                        target_size_10m=(target_h, target_w),
                                        target_size_20m=(target_h_20m, target_w_20m))
        
        # Clip to valid range
        sr_10m = np.clip(sr_10m, 0, 1)
        sr_20m = np.clip(sr_20m, 0, 1)
        
        # Bilinear baseline
        bl_10m, bl_20m = bilinear_baseline(lr_10m, lr_20m, target_h, target_w,
                                            target_h_20m, target_w_20m)
        bl_10m = np.clip(bl_10m, 0, 1)
        bl_20m = np.clip(bl_20m, 0, 1)
        
        # Compute all metrics for model
        m_10m = compute_all_metrics(sr_10m, hr_10m, scale_10m)
        m_20m = compute_all_metrics(sr_20m, hr_20m, scale_20m)
        
        for k in metrics_10m:
            metrics_10m[k].append(m_10m[k])
            metrics_20m[k].append(m_20m[k])
        
        # Compute all metrics for bilinear
        b_10m = compute_all_metrics(bl_10m, hr_10m, scale_10m)
        b_20m = compute_all_metrics(bl_20m, hr_20m, scale_20m)
        
        for k in metrics_bilinear_10m:
            metrics_bilinear_10m[k].append(b_10m[k])
            metrics_bilinear_20m[k].append(b_20m[k])
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{n_samples}")
    
    # Aggregate results
    results = {
        'model_name': model_name,
        'n_samples': n_samples,
        'model': {
            '10m': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics_10m.items()},
            '20m': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics_20m.items()},
        },
        'bilinear': {
            '10m': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics_bilinear_10m.items()},
            '20m': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics_bilinear_20m.items()},
        },
        'improvement': {
            '10m': {k: float(np.mean(metrics_10m[k]) - np.mean(metrics_bilinear_10m[k])) for k in metrics_10m},
            '20m': {k: float(np.mean(metrics_20m[k]) - np.mean(metrics_bilinear_20m[k])) for k in metrics_20m},
        }
    }
    
    # Compute combined metrics
    results['model']['combined'] = {}
    results['bilinear']['combined'] = {}
    results['improvement']['combined'] = {}
    
    for k in metrics_10m:
        model_combined = np.mean(metrics_10m[k] + metrics_20m[k])
        bilinear_combined = np.mean(metrics_bilinear_10m[k] + metrics_bilinear_20m[k])
        results['model']['combined'][k] = float(model_combined)
        results['bilinear']['combined'][k] = float(bilinear_combined)
        results['improvement']['combined'][k] = float(model_combined - bilinear_combined)
    
    return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print(f"\n{'='*80}")
    print(f"Results: {results['model_name']}")
    print(f"{'='*80}")
    print(f"Test samples: {results['n_samples']}")
    
    print(f"\n--- Model Performance ---")
    m = results['model']
    print(f"  10m bands:")
    print(f"    PSNR={m['10m']['psnr']['mean']:.2f}±{m['10m']['psnr']['std']:.2f} dB | "
          f"RMSE={m['10m']['rmse']['mean']:.4f} | "
          f"SSIM={m['10m']['ssim']['mean']:.4f}")
    print(f"    SCC={m['10m']['scc']['mean']:.4f} | "
          f"ERGAS={m['10m']['ergas']['mean']:.2f} | "
          f"SID={m['10m']['sid']['mean']:.4f}")
    
    print(f"  20m bands:")
    print(f"    PSNR={m['20m']['psnr']['mean']:.2f}±{m['20m']['psnr']['std']:.2f} dB | "
          f"RMSE={m['20m']['rmse']['mean']:.4f} | "
          f"SSIM={m['20m']['ssim']['mean']:.4f}")
    print(f"    SCC={m['20m']['scc']['mean']:.4f} | "
          f"ERGAS={m['20m']['ergas']['mean']:.2f} | "
          f"SID={m['20m']['sid']['mean']:.4f}")
    
    print(f"\n--- Bilinear Baseline ---")
    b = results['bilinear']
    print(f"  10m bands:")
    print(f"    PSNR={b['10m']['psnr']['mean']:.2f} dB | RMSE={b['10m']['rmse']['mean']:.4f} | "
          f"SSIM={b['10m']['ssim']['mean']:.4f}")
    print(f"    SCC={b['10m']['scc']['mean']:.4f} | ERGAS={b['10m']['ergas']['mean']:.2f} | "
          f"SID={b['10m']['sid']['mean']:.4f}")
    print(f"  20m bands:")
    print(f"    PSNR={b['20m']['psnr']['mean']:.2f} dB | RMSE={b['20m']['rmse']['mean']:.4f} | "
          f"SSIM={b['20m']['ssim']['mean']:.4f}")
    print(f"    SCC={b['20m']['scc']['mean']:.4f} | ERGAS={b['20m']['ergas']['mean']:.2f} | "
          f"SID={b['20m']['sid']['mean']:.4f}")
    
    print(f"\n--- Improvement over Bilinear ---")
    imp = results['improvement']
    print(f"  10m: PSNR {imp['10m']['psnr']:+.2f} dB | RMSE {imp['10m']['rmse']:+.4f} | "
          f"SSIM {imp['10m']['ssim']:+.4f} | SCC {imp['10m']['scc']:+.4f}")
    print(f"  20m: PSNR {imp['20m']['psnr']:+.2f} dB | RMSE {imp['20m']['rmse']:+.4f} | "
          f"SSIM {imp['20m']['ssim']:+.4f} | SCC {imp['20m']['scc']:+.4f}")


def print_summary_table(all_results: List[Dict]):
    """Print summary comparison table."""
    print("\n" + "="*100)
    print("SUMMARY COMPARISON (Test Set: 111 samples)")
    print("="*100)
    
    # Header
    print(f"\n{'Model':<28} | {'PSNR':<8} | {'RMSE':<8} | {'SSIM':<8} | {'SCC':<8} | {'ERGAS':<8} | {'SID':<8}")
    print("-"*100)
    
    # Model results
    for r in all_results:
        name = r['model_name']
        m = r['model']['combined']
        print(f"{name:<28} | {m['psnr']:<8.2f} | {m['rmse']:<8.4f} | {m['ssim']:<8.4f} | "
              f"{m['scc']:<8.4f} | {m['ergas']:<8.2f} | {m['sid']:<8.4f}")
    
    # Bilinear baseline (from first result)
    if all_results:
        b = all_results[0]['bilinear']['combined']
        print(f"{'Bilinear Baseline':<28} | {b['psnr']:<8.2f} | {b['rmse']:<8.4f} | {b['ssim']:<8.4f} | "
              f"{b['scc']:<8.4f} | {b['ergas']:<8.2f} | {b['sid']:<8.4f}")
    
    # Improvement table
    print(f"\n{'='*100}")
    print("IMPROVEMENT OVER BILINEAR")
    print("="*100)
    print(f"\n{'Model':<28} | {'ΔPSNR':<8} | {'ΔRMSE':<8} | {'ΔSSIM':<8} | {'ΔSCC':<8} | {'ΔERGAS':<8} | {'ΔSID':<8}")
    print("-"*100)
    
    for r in all_results:
        name = r['model_name']
        imp = r['improvement']['combined']
        # Note: For RMSE, ERGAS, SID - negative improvement is better (lower is better)
        print(f"{name:<28} | {imp['psnr']:+8.2f} | {imp['rmse']:+8.4f} | {imp['ssim']:+8.4f} | "
              f"{imp['scc']:+8.4f} | {imp['ergas']:+8.2f} | {imp['sid']:+8.4f}")
    
    # Best model summary
    print(f"\n{'='*100}")
    print("BEST MODEL PER METRIC (higher is better for PSNR, SSIM, SCC; lower for RMSE, ERGAS, SID)")
    print("="*100)
    
    metrics_higher_better = ['psnr', 'ssim', 'scc']
    metrics_lower_better = ['rmse', 'ergas', 'sid']
    
    for metric in metrics_higher_better + metrics_lower_better:
        if metric in metrics_higher_better:
            best_idx = np.argmax([r['model']['combined'][metric] for r in all_results])
        else:
            best_idx = np.argmin([r['model']['combined'][metric] for r in all_results])
        
        best_name = all_results[best_idx]['model_name']
        best_val = all_results[best_idx]['model']['combined'][metric]
        print(f"  {metric.upper():<6}: {best_name:<28} ({best_val:.4f})")


# ============================================================
# Main
# ============================================================

def main():
    print("="*100)
    print("Comprehensive Model Evaluation with Full Metrics Suite")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Paths
    base_dir = Path('/media/eamurciab/TOSHIBA_EXT1/PhD/SR')
    test_dir = base_dir / 'data/sr_training/5band_v3/combined/mmap_test'
    checkpoint_dir = base_dir / 'checkpoints'
    
    # Models to evaluate
    models_config = [
        {
            'name': 'V3 (Full Dataset)',
            'checkpoint': checkpoint_dir / '5band_v3_perceptual_full/best_model.pt',
            'loader': load_v3_model,
        },
        {
            'name': 'V3 (Filtered Dataset)',
            'checkpoint': checkpoint_dir / '5band_v3_perceptual_filtered/best_model.pt',
            'loader': load_v3_model,
        },
        {
            'name': 'MultiRes (Full Dataset)',
            'checkpoint': checkpoint_dir / 'multires_perceptual_full/best_model.pt',
            'loader': load_multires_model,
        },
        {
            'name': 'MultiRes (Filtered Dataset)',
            'checkpoint': checkpoint_dir / 'multires_perceptual_filtered/best_model.pt',
            'loader': load_multires_model,
        },
    ]
    
    # Load test data
    lr_10m, lr_20m, hr_10m, hr_20m = load_test_data(test_dir)
    
    # Evaluate each model
    all_results = []
    
    for config in models_config:
        if not config['checkpoint'].exists():
            print(f"\n⚠️  Checkpoint not found: {config['checkpoint']}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Loading: {config['name']}")
        print(f"Checkpoint: {config['checkpoint']}")
        
        model = config['loader'](str(config['checkpoint']), device)
        results = evaluate_model(
            model, lr_10m, lr_20m, hr_10m, hr_20m, device, config['name']
        )
        all_results.append(results)
        print_results(results)
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Summary comparison
    print_summary_table(all_results)
    
    # Save results
    results_path = base_dir / 'test_results_full_metrics.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == '__main__':
    main()
