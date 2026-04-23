#!/usr/bin/env python3
"""
Comprehensive test script for all trained SR models.

Evaluates:
1. V3 (trained on full dataset)
2. V3 (trained on filtered dataset)
3. MultiRes (trained on full dataset)
4. MultiRes (trained on filtered dataset)

Metrics:
- PSNR (per band and average)
- SSIM (per band and average)
- SAM (Spectral Angle Mapper)
- Comparison vs bilinear baseline
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

# Import models
from models.satellite_sr_model_v3 import MultiResolutionSatelliteSRv3
from models.satellite_sr_model_v2 import MultiResolutionSatelliteSRv2

# ============================================================
# Metrics
# ============================================================

def psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """Calculate PSNR between prediction and target."""
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(max_val ** 2 / mse)


def ssim_single(pred: np.ndarray, target: np.ndarray, win_size: int = 7) -> float:
    """Calculate SSIM for single channel images."""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Simple SSIM implementation
    mu_x = np.mean(pred)
    mu_y = np.mean(target)
    sigma_x = np.std(pred)
    sigma_y = np.std(target)
    sigma_xy = np.mean((pred - mu_x) * (target - mu_y))
    
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))
    return ssim


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate average SSIM across all channels."""
    if pred.ndim == 2:
        return ssim_single(pred, target)
    
    ssim_vals = []
    for c in range(pred.shape[-1]):
        ssim_vals.append(ssim_single(pred[..., c], target[..., c]))
    return np.mean(ssim_vals)


def spectral_angle_mapper(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Spectral Angle Mapper (SAM) in degrees."""
    # Reshape to (pixels, channels)
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    
    # Compute dot product and norms
    dot = np.sum(pred_flat * target_flat, axis=1)
    norm_pred = np.linalg.norm(pred_flat, axis=1)
    norm_target = np.linalg.norm(target_flat, axis=1)
    
    # Avoid division by zero
    valid = (norm_pred > 1e-8) & (norm_target > 1e-8)
    cos_angle = np.clip(dot[valid] / (norm_pred[valid] * norm_target[valid]), -1, 1)
    
    # Convert to degrees
    angles = np.arccos(cos_angle) * 180 / np.pi
    return np.mean(angles)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute all metrics."""
    return {
        'psnr': psnr(pred, target),
        'ssim': ssim(pred, target),
        'sam': spectral_angle_mapper(pred, target)
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
                      target_h: int, target_w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bilinear upscaling baseline."""
    # Convert to torch for interpolation
    lr_10m_t = torch.from_numpy(lr_10m).float()
    lr_20m_t = torch.from_numpy(lr_20m).float()
    
    # Handle different input shapes
    if lr_10m_t.ndim == 3:  # (H, W, C)
        lr_10m_t = lr_10m_t.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    elif lr_10m_t.ndim == 4:  # (T, H, W, C)
        # Use middle frame
        mid = lr_10m_t.shape[0] // 2
        lr_10m_t = lr_10m_t[mid].permute(2, 0, 1).unsqueeze(0)
    
    if lr_20m_t.ndim == 3:
        lr_20m_t = lr_20m_t.permute(2, 0, 1).unsqueeze(0)
    elif lr_20m_t.ndim == 4:
        mid = lr_20m_t.shape[0] // 2
        lr_20m_t = lr_20m_t[mid].permute(2, 0, 1).unsqueeze(0)
    
    # Bilinear upscale
    hr_10m_bilinear = F.interpolate(lr_10m_t, size=(target_h, target_w), 
                                     mode='bilinear', align_corners=False)
    hr_20m_bilinear = F.interpolate(lr_20m_t, size=(target_h // 2, target_w // 2), 
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
                   model_name: str) -> Dict:
    """Evaluate model on full test set."""
    
    n_samples = len(lr_10m_data)
    print(f"\nEvaluating {model_name} on {n_samples} test samples...")
    
    # Metrics accumulators
    metrics_10m = {'psnr': [], 'ssim': [], 'sam': []}
    metrics_20m = {'psnr': [], 'ssim': [], 'sam': []}
    metrics_bilinear_10m = {'psnr': [], 'ssim': [], 'sam': []}
    metrics_bilinear_20m = {'psnr': [], 'ssim': [], 'sam': []}
    
    # Per-band PSNR for detailed analysis
    psnr_per_band_10m = {f'B{i}': [] for i in range(3)}  # B, G, R
    psnr_per_band_20m = {f'B{i}': [] for i in range(2)}  # RE, NIR
    
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
        bl_10m, bl_20m = bilinear_baseline(lr_10m, lr_20m, target_h, target_w)
        bl_10m = np.clip(bl_10m, 0, 1)
        bl_20m = np.clip(bl_20m, 0, 1)
        
        # Compute metrics for model
        m_10m = compute_metrics(sr_10m, hr_10m)
        m_20m = compute_metrics(sr_20m, hr_20m)
        
        for k in metrics_10m:
            metrics_10m[k].append(m_10m[k])
            metrics_20m[k].append(m_20m[k])
        
        # Compute metrics for bilinear
        b_10m = compute_metrics(bl_10m, hr_10m)
        b_20m = compute_metrics(bl_20m, hr_20m)
        
        for k in metrics_bilinear_10m:
            metrics_bilinear_10m[k].append(b_10m[k])
            metrics_bilinear_20m[k].append(b_20m[k])
        
        # Per-band PSNR
        for b in range(3):
            psnr_per_band_10m[f'B{b}'].append(psnr(sr_10m[..., b], hr_10m[..., b]))
        for b in range(2):
            psnr_per_band_20m[f'B{b}'].append(psnr(sr_20m[..., b], hr_20m[..., b]))
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{n_samples}")
    
    # Aggregate results
    results = {
        'model_name': model_name,
        'n_samples': n_samples,
        'model': {
            '10m': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in metrics_10m.items()},
            '20m': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in metrics_20m.items()},
            'avg_psnr': np.mean(metrics_10m['psnr'] + metrics_20m['psnr']),
            'avg_ssim': np.mean(metrics_10m['ssim'] + metrics_20m['ssim']),
        },
        'bilinear': {
            '10m': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in metrics_bilinear_10m.items()},
            '20m': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in metrics_bilinear_20m.items()},
            'avg_psnr': np.mean(metrics_bilinear_10m['psnr'] + metrics_bilinear_20m['psnr']),
            'avg_ssim': np.mean(metrics_bilinear_10m['ssim'] + metrics_bilinear_20m['ssim']),
        },
        'improvement': {
            'psnr_10m': np.mean(metrics_10m['psnr']) - np.mean(metrics_bilinear_10m['psnr']),
            'psnr_20m': np.mean(metrics_20m['psnr']) - np.mean(metrics_bilinear_20m['psnr']),
            'ssim_10m': np.mean(metrics_10m['ssim']) - np.mean(metrics_bilinear_10m['ssim']),
            'ssim_20m': np.mean(metrics_20m['ssim']) - np.mean(metrics_bilinear_20m['ssim']),
        },
        'per_band_psnr': {
            '10m': {k: np.mean(v) for k, v in psnr_per_band_10m.items()},
            '20m': {k: np.mean(v) for k, v in psnr_per_band_20m.items()},
        }
    }
    
    return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print(f"\n{'='*70}")
    print(f"Results: {results['model_name']}")
    print(f"{'='*70}")
    print(f"Test samples: {results['n_samples']}")
    
    print(f"\n--- Model Performance ---")
    print(f"  10m bands: PSNR={results['model']['10m']['psnr']['mean']:.2f}±{results['model']['10m']['psnr']['std']:.2f} dB, "
          f"SSIM={results['model']['10m']['ssim']['mean']:.4f}, "
          f"SAM={results['model']['10m']['sam']['mean']:.2f}°")
    print(f"  20m bands: PSNR={results['model']['20m']['psnr']['mean']:.2f}±{results['model']['20m']['psnr']['std']:.2f} dB, "
          f"SSIM={results['model']['20m']['ssim']['mean']:.4f}, "
          f"SAM={results['model']['20m']['sam']['mean']:.2f}°")
    print(f"  Average:   PSNR={results['model']['avg_psnr']:.2f} dB, SSIM={results['model']['avg_ssim']:.4f}")
    
    print(f"\n--- Bilinear Baseline ---")
    print(f"  10m bands: PSNR={results['bilinear']['10m']['psnr']['mean']:.2f} dB, "
          f"SSIM={results['bilinear']['10m']['ssim']['mean']:.4f}")
    print(f"  20m bands: PSNR={results['bilinear']['20m']['psnr']['mean']:.2f} dB, "
          f"SSIM={results['bilinear']['20m']['ssim']['mean']:.4f}")
    print(f"  Average:   PSNR={results['bilinear']['avg_psnr']:.2f} dB")
    
    print(f"\n--- Improvement over Bilinear ---")
    print(f"  10m: PSNR +{results['improvement']['psnr_10m']:.2f} dB, SSIM +{results['improvement']['ssim_10m']:.4f}")
    print(f"  20m: PSNR +{results['improvement']['psnr_20m']:.2f} dB, SSIM +{results['improvement']['ssim_20m']:.4f}")
    
    print(f"\n--- Per-Band PSNR ---")
    print(f"  10m: B={results['per_band_psnr']['10m']['B0']:.2f}, G={results['per_band_psnr']['10m']['B1']:.2f}, R={results['per_band_psnr']['10m']['B2']:.2f}")
    print(f"  20m: RE={results['per_band_psnr']['20m']['B0']:.2f}, NIR={results['per_band_psnr']['20m']['B1']:.2f}")


# ============================================================
# Main
# ============================================================

def main():
    print("="*70)
    print("Comprehensive Model Evaluation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
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
        
        print(f"\n{'='*70}")
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
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"\n{'Model':<30} {'PSNR 10m':<12} {'PSNR 20m':<12} {'PSNR Avg':<12} {'vs Bilinear':<12}")
    print("-"*78)
    
    for r in all_results:
        psnr_10m = r['model']['10m']['psnr']['mean']
        psnr_20m = r['model']['20m']['psnr']['mean']
        psnr_avg = r['model']['avg_psnr']
        improvement = psnr_avg - r['bilinear']['avg_psnr']
        print(f"{r['model_name']:<30} {psnr_10m:<12.2f} {psnr_20m:<12.2f} {psnr_avg:<12.2f} +{improvement:<11.2f}")
    
    # Bilinear baseline (from first result)
    if all_results:
        bl = all_results[0]['bilinear']
        print(f"{'Bilinear Baseline':<30} {bl['10m']['psnr']['mean']:<12.2f} {bl['20m']['psnr']['mean']:<12.2f} {bl['avg_psnr']:<12.2f} {'—':<12}")
    
    # Save results
    results_path = base_dir / 'test_results_comprehensive.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == '__main__':
    main()
