#!/usr/bin/env python3
"""
Test script for the temporal supervision model.

Evaluates with full metrics suite (same as test_models_full_metrics.py):
- PSNR, RMSE, SSIM, SCC, ERGAS, SID
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple
import json
from datetime import datetime
from scipy.ndimage import uniform_filter, laplace

# Import model
from scripts.training.models.satellite_sr_model_v2 import MultiResolutionSatelliteSRv2


# ============================================================
# Quality Metrics (copied from test_models_full_metrics.py)
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
    """Compute Structural Similarity Index (SSIM)."""
    if pred.ndim == 3:
        ssim_values = []
        for c in range(pred.shape[-1]):
            ssim_values.append(compute_ssim(pred[:, :, c], target[:, :, c], win_size))
        return np.mean(ssim_values)
    
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    mu_pred = uniform_filter(pred.astype(np.float64), size=win_size, mode='reflect')
    mu_target = uniform_filter(target.astype(np.float64), size=win_size, mode='reflect')
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = uniform_filter(pred.astype(np.float64) ** 2, size=win_size, mode='reflect') - mu_pred_sq
    sigma_target_sq = uniform_filter(target.astype(np.float64) ** 2, size=win_size, mode='reflect') - mu_target_sq
    sigma_pred_target = uniform_filter(pred.astype(np.float64) * target.astype(np.float64), size=win_size, mode='reflect') - mu_pred_target
    
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    
    ssim_map = numerator / (denominator + 1e-10)
    return np.mean(ssim_map)


def compute_scc(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Spatial Correlation Coefficient (SCC)."""
    if pred.ndim == 3:
        scc_values = []
        for c in range(pred.shape[-1]):
            scc_values.append(compute_scc(pred[:, :, c], target[:, :, c]))
        return np.mean(scc_values)
    
    pred_lap = laplace(pred.astype(np.float64))
    target_lap = laplace(target.astype(np.float64))
    
    pred_lap_flat = pred_lap.flatten()
    target_lap_flat = target_lap.flatten()
    
    if np.std(pred_lap_flat) < 1e-10 or np.std(target_lap_flat) < 1e-10:
        return 0.0
    
    correlation = np.corrcoef(pred_lap_flat, target_lap_flat)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def compute_ergas(pred: np.ndarray, target: np.ndarray, scale_ratio: float = 3.33) -> float:
    """Compute ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)."""
    if pred.ndim == 2:
        pred = pred[:, :, np.newaxis]
        target = target[:, :, np.newaxis]
    
    n_bands = pred.shape[-1]
    sum_term = 0.0
    
    for b in range(n_bands):
        pred_band = pred[:, :, b]
        target_band = target[:, :, b]
        
        rmse_band = np.sqrt(np.mean((pred_band - target_band) ** 2))
        mean_target = np.mean(target_band)
        
        if mean_target > 1e-10:
            sum_term += (rmse_band / mean_target) ** 2
        else:
            sum_term += rmse_band ** 2
    
    ergas = 100 * scale_ratio * np.sqrt(sum_term / n_bands)
    return ergas


def compute_sid(pred: np.ndarray, target: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute Spectral Information Divergence (SID)."""
    if pred.ndim == 2:
        return 0.0
    
    pred = np.clip(pred, epsilon, None)
    target = np.clip(target, epsilon, None)
    
    pred_sum = np.sum(pred, axis=-1, keepdims=True)
    target_sum = np.sum(target, axis=-1, keepdims=True)
    
    p = pred / (pred_sum + epsilon)
    q = target / (target_sum + epsilon)
    
    kl_pq = np.sum(p * np.log((p + epsilon) / (q + epsilon)), axis=-1)
    kl_qp = np.sum(q * np.log((q + epsilon) / (p + epsilon)), axis=-1)
    
    sid = kl_pq + kl_qp
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


def load_temporal_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load temporal model from checkpoint."""
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


def evaluate_model(model, lr, lr_20m, hr_10m, hr_20m, device, batch_size=4):
    """Evaluate model on test data."""
    model.eval()
    n_samples = lr.shape[0]
    
    all_metrics_10m = []
    all_metrics_20m = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_size_actual = end_idx - i
            
            # Load batch
            batch_lr = torch.from_numpy(lr[i:end_idx].copy()).permute(0, 1, 4, 2, 3).float().to(device)
            batch_lr_20m = torch.from_numpy(lr_20m[i:end_idx].copy()).permute(0, 1, 4, 2, 3).float().to(device)
            batch_hr_10m = torch.from_numpy(hr_10m[i:end_idx].copy()).float()
            batch_hr_20m = torch.from_numpy(hr_20m[i:end_idx].copy()).float()
            
            # Get middle frame target
            mid_idx = batch_hr_10m.shape[1] // 2
            target_10m = batch_hr_10m[:, mid_idx]  # (B, H, W, C)
            target_20m = batch_hr_20m[:, mid_idx]
            
            # Model forward
            target_size_10m = (target_10m.shape[1], target_10m.shape[2])
            target_size_20m = (target_20m.shape[1], target_20m.shape[2])
            
            pred_10m, pred_20m = model(batch_lr, batch_lr_20m, 
                                        target_size_10m=target_size_10m,
                                        target_size_20m=target_size_20m)
            
            # Ensure output matches target size
            if pred_10m.shape[2:] != target_size_10m:
                pred_10m = F.interpolate(pred_10m, size=target_size_10m, mode='bilinear', align_corners=False)
            if pred_20m.shape[2:] != target_size_20m:
                pred_20m = F.interpolate(pred_20m, size=target_size_20m, mode='bilinear', align_corners=False)
            
            # Convert to numpy (B, C, H, W) -> (B, H, W, C)
            pred_10m_np = pred_10m.cpu().numpy().transpose(0, 2, 3, 1)
            pred_20m_np = pred_20m.cpu().numpy().transpose(0, 2, 3, 1)
            target_10m_np = target_10m.numpy()
            target_20m_np = target_20m.numpy()
            
            # Compute metrics for each sample
            for j in range(batch_size_actual):
                metrics_10m = compute_all_metrics(pred_10m_np[j], target_10m_np[j], scale_ratio=3.33)
                metrics_20m = compute_all_metrics(pred_20m_np[j], target_20m_np[j], scale_ratio=6.67)
                all_metrics_10m.append(metrics_10m)
                all_metrics_20m.append(metrics_20m)
            
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {end_idx}/{n_samples} samples...")
    
    # Average metrics
    avg_10m = {k: np.mean([m[k] for m in all_metrics_10m]) for k in all_metrics_10m[0].keys()}
    avg_20m = {k: np.mean([m[k] for m in all_metrics_20m]) for k in all_metrics_20m[0].keys()}
    
    return avg_10m, avg_20m


def main():
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    TEST_DATA = PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap_test'
    CHECKPOINT = PROJECT_ROOT / 'checkpoints/multires_temporal_full/best_model.pt'
    
    print("="*60)
    print("Testing Temporal Supervision Model")
    print("="*60)
    
    # Check paths
    if not TEST_DATA.exists():
        print(f"ERROR: Test data not found at {TEST_DATA}")
        return
    if not CHECKPOINT.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
        return
    
    print(f"Test data: {TEST_DATA}")
    print(f"Checkpoint: {CHECKPOINT}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading test data...")
    lr, lr_20m, hr_10m, hr_20m = load_test_data(str(TEST_DATA))
    print(f"Test samples: {lr.shape[0]}")
    
    # Load model
    print("\nLoading temporal model...")
    model = load_temporal_model(str(CHECKPOINT), device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics_10m, metrics_20m = evaluate_model(model, lr, lr_20m, hr_10m, hr_20m, device)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS: MultiRes Temporal Supervision (Full Dataset)")
    print("="*60)
    
    print("\n10m Branch (RGB-like bands):")
    print(f"  PSNR:  {metrics_10m['psnr']:.4f} dB")
    print(f"  RMSE:  {metrics_10m['rmse']:.6f}")
    print(f"  SSIM:  {metrics_10m['ssim']:.4f}")
    print(f"  SCC:   {metrics_10m['scc']:.4f}")
    print(f"  ERGAS: {metrics_10m['ergas']:.4f}")
    print(f"  SID:   {metrics_10m['sid']:.6f}")
    
    print("\n20m Branch (RedEdge/NIR bands):")
    print(f"  PSNR:  {metrics_20m['psnr']:.4f} dB")
    print(f"  RMSE:  {metrics_20m['rmse']:.6f}")
    print(f"  SSIM:  {metrics_20m['ssim']:.4f}")
    print(f"  SCC:   {metrics_20m['scc']:.4f}")
    print(f"  ERGAS: {metrics_20m['ergas']:.4f}")
    print(f"  SID:   {metrics_20m['sid']:.6f}")
    
    # Combined average
    avg_psnr = (metrics_10m['psnr'] + metrics_20m['psnr']) / 2
    avg_ssim = (metrics_10m['ssim'] + metrics_20m['ssim']) / 2
    avg_ergas = (metrics_10m['ergas'] + metrics_20m['ergas']) / 2
    avg_sid = (metrics_10m['sid'] + metrics_20m['sid']) / 2
    
    print("\nCombined Average:")
    print(f"  Avg PSNR:  {avg_psnr:.4f} dB")
    print(f"  Avg SSIM:  {avg_ssim:.4f}")
    print(f"  Avg ERGAS: {avg_ergas:.4f}")
    print(f"  Avg SID:   {avg_sid:.6f}")
    
    # Save results
    results = {
        'model': 'MultiRes Temporal Supervision (Full)',
        'checkpoint': str(CHECKPOINT),
        'test_samples': lr.shape[0],
        'timestamp': datetime.now().isoformat(),
        '10m_branch': metrics_10m,
        '20m_branch': metrics_20m,
        'combined': {
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'avg_ergas': avg_ergas,
            'avg_sid': avg_sid
        }
    }
    
    results_path = PROJECT_ROOT / 'test_results_temporal.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == '__main__':
    main()
