#!/usr/bin/env python3
"""
Training script for Native Resolution Super-Resolution.

This script properly handles the native resolution data:
- LR (Sentinel-2 @ 10m): 77×77 pixels
- HR (Planet @ 3m): 256×256 pixels
- Scale factor: 3.33× (10m → 3m)

Key fixes from previous training:
1. Passes target_size=(256, 256) to model
2. Properly handles the 77→256 upscaling
3. No interpolation of HR target (preserves full resolution)
"""

import os
import sys
import pickle
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add paths
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / 'models'))
sys.path.insert(0, str(script_dir.parent / 'preprocessing'))

from dataclasses import field
from typing import Any

# Define TrainingSample for pickle loading - must match saved structure
@dataclass
class TrainingSample:
    """Training sample with temporal sequences and multi-resolution LR bands"""
    lr_sequence: np.ndarray = None
    hr_sequence: np.ndarray = None
    lr_20m_sequence: np.ndarray = None
    tile_id: str = ''
    sequence_id: str = ''
    region_name: str = ''
    temporal_dates: List[str] = field(default_factory=list)
    tile_bounds: Dict = field(default_factory=dict)
    lr_10m_coregistered: np.ndarray = None
    lr_20m_coregistered: np.ndarray = None
    lr_10m_coregistered_harmonized: np.ndarray = None
    lr_20m_coregistered_harmonized: np.ndarray = None
    planet_dates: List[str] = field(default_factory=list)
    frame_quality: Any = None
    planet_solar_zenith: Any = None
    s2_solar_zenith: Any = None

# Inject TrainingSample into __main__ for pickle to find it
import types
_main_module = types.ModuleType('__main__')
_main_module.TrainingSample = TrainingSample
sys.modules['__main__'] = _main_module

from satellite_sr_model import SingleResolutionSatelliteSR


# =============================================================================
# DATASET
# =============================================================================

class NativeResolutionSRDataset(Dataset):
    """
    Dataset for native resolution SR training.
    
    Expects:
    - LR: (T, 77, 77, C) @ 10m
    - HR: (T, 256, 256, C) @ 3m
    """
    
    def __init__(
        self,
        pickle_paths: List[str],
        sequence_length: int = 3,
        augment: bool = True,
        use_full_images: bool = True,  # Use full images (77 LR -> 256 HR)
    ):
        self.sequence_length = sequence_length
        self.augment = augment
        self.use_full_images = use_full_images
        
        # Load all samples from multiple pickle files
        self.samples = []
        for path in pickle_paths:
            print(f"Loading {path}...")
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.samples.extend(data.get('samples', []))
        
        print(f"Total samples: {len(self.samples)}")
        
        # Verify native resolution and get exact sizes
        self.lr_size = None
        self.hr_size = None
        if self.samples:
            s = self.samples[0]
            lr_shape = s.lr_sequence.shape
            hr_shape = s.hr_sequence.shape
            self.lr_size = (lr_shape[1], lr_shape[2])
            self.hr_size = (hr_shape[1], hr_shape[2])
            print(f"LR shape: {lr_shape} (expected T, 77, 77, C)")
            print(f"HR shape: {hr_shape} (expected T, 256, 256, C)")
            
            # Compute actual scale factor
            self.scale_factor = hr_shape[1] / lr_shape[1]
            print(f"Scale factor: {self.scale_factor:.2f}×")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert (T, H, W, C) → (T, C, H, W)
        lr = sample.lr_sequence.copy().transpose(0, 3, 1, 2).astype(np.float32)
        hr = sample.hr_sequence.copy().transpose(0, 3, 1, 2).astype(np.float32)
        
        # Select frames
        T = lr.shape[0]
        if T > self.sequence_length:
            start = random.randint(0, T - self.sequence_length)
            lr = lr[start:start + self.sequence_length]
            hr = hr[start:start + self.sequence_length]
        elif T < self.sequence_length:
            # Repeat last frame
            while lr.shape[0] < self.sequence_length:
                lr = np.concatenate([lr, lr[-1:]], axis=0)
                hr = np.concatenate([hr, hr[-1:]], axis=0)
        
        # Augmentation (no cropping - use full images)
        if self.augment:
            lr, hr = self._augment(lr, hr)
        
        result = {
            'lr': torch.from_numpy(lr),
            'hr': torch.from_numpy(hr)
        }
        
        # Include SZA if available (use Sentinel-2 SZA as reference)
        # Use middle frame's SZA to match HR target
        if hasattr(sample, 's2_solar_zenith') and sample.s2_solar_zenith is not None:
            sza_val = sample.s2_solar_zenith
            if isinstance(sza_val, (list, tuple, np.ndarray)):
                # Use middle frame SZA
                mid_idx = len(sza_val) // 2
                result['sza'] = float(sza_val[mid_idx])
            else:
                result['sza'] = float(sza_val)
        
        return result
    
    def _augment(self, lr, hr):
        """Random augmentation (flip, rotate)."""
        # Random horizontal flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=3).copy()
            hr = np.flip(hr, axis=3).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=2).copy()
            hr = np.flip(hr, axis=2).copy()
        
        # Random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            lr = np.rot90(lr, k, axes=(2, 3)).copy()
            hr = np.rot90(hr, k, axes=(2, 3)).copy()
        
        return lr, hr


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class SRLoss(nn.Module):
    """Simple combined loss for super-resolution (L1 + Gradient)."""
    
    def __init__(self, l1_weight=1.0, ssim_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target, sza=None):
        """
        Args:
            pred: (B, C, H, W) SR output
            target: (B, C, H, W) HR ground truth
            sza: Unused (for API compatibility with SpectralFidelityLoss)
        """
        # L1 loss
        l1_loss = self.l1(pred, target)
        
        # Simple SSIM-like loss (structural similarity via gradient)
        ssim_loss = self._gradient_loss(pred, target)
        
        return self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
    
    def _gradient_loss(self, pred, target):
        """Gradient-based structural loss."""
        # Sobel gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        return self.l1(pred_dx, target_dx) + self.l1(pred_dy, target_dy)


class SpectralFidelityLoss(nn.Module):
    """
    Multi-component loss for satellite SR with optional solar zenith angle weighting.
    
    Loss = SZA_weight × (α_pixel × (L1 + 0.1×MSE) + α_spectral × SAM + α_gradient × Gradient)
    
    The loss can be weighted by a Gaussian penalty based on deviation from optimal
    solar zenith angle (45°) for aquatic vegetation analysis:
        w(SZA) = exp(-(SZA - SZA_optimal)^2 / (2 * sigma^2))
    """
    
    def __init__(self, alpha_pixel=1.0, alpha_spectral=0.1, alpha_gradient=0.01,
                 sza_optimal=45.0, sza_sigma=30.0, use_sza_weighting=False):
        """
        Args:
            alpha_pixel: Weight for pixel-level loss (L1 + 0.1×MSE)
            alpha_spectral: Weight for spectral angle loss (SAM)
            alpha_gradient: Weight for gradient/edge loss
            sza_optimal: Optimal solar zenith angle in degrees (default: 45°)
            sza_sigma: Standard deviation for Gaussian penalty (default: 30°)
            use_sza_weighting: Whether to apply SZA weighting (default: False)
        """
        super().__init__()
        self.alpha_pixel = alpha_pixel
        self.alpha_spectral = alpha_spectral
        self.alpha_gradient = alpha_gradient
        self.sza_optimal = sza_optimal
        self.sza_sigma = sza_sigma
        self.use_sza_weighting = use_sza_weighting
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def compute_sza_weight(self, sza):
        """
        Compute Gaussian weight based on solar zenith angle deviation.
        
        Args:
            sza: Solar zenith angle in degrees, or None if unavailable
        
        Returns:
            Weight in range (0, 1], where 1.0 = optimal angle
        """
        if sza is None or not self.use_sza_weighting:
            return 1.0
        
        deviation = sza - self.sza_optimal
        weight = np.exp(-(deviation ** 2) / (2 * self.sza_sigma ** 2))
        return float(weight)
    
    def spectral_angle_loss(self, pred, target):
        """
        Spectral Angle Mapper (SAM) loss.
        Measures spectral fidelity by computing angle between spectral vectors.
        
        Args:
            pred: (B, C, H, W) predicted
            target: (B, C, H, W) target
        
        Returns:
            Scalar loss (1 - mean cosine similarity)
        """
        # Normalize along channel dimension
        pred_norm = F.normalize(pred, dim=1, eps=1e-8)
        target_norm = F.normalize(target, dim=1, eps=1e-8)
        
        # Cosine similarity per pixel
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)  # (B, H, W)
        
        # SAM loss: 1 - mean cosine similarity
        return 1.0 - cos_sim.mean()
    
    def gradient_loss(self, pred, target):
        """Gradient loss for edge preservation."""
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        return self.l1_loss(pred_dx, target_dx) + self.l1_loss(pred_dy, target_dy)
        
    def forward(self, pred, target, sza=None):
        """
        Args:
            pred: (B, C, H, W) predicted SR
            target: (B, C, H, W) ground truth HR
            sza: Solar zenith angle (optional). Can be:
                 - float: Single angle for entire batch
                 - List[float]: Per-sample angles
                 - None: No SZA weighting applied
        
        Returns:
            total_loss: Scalar loss value
        """
        # Pixel-level loss (L1 + 0.1×MSE)
        pixel_loss = self.l1_loss(pred, target) + 0.1 * self.mse_loss(pred, target)
        
        # Spectral angle loss (SAM)
        spectral_loss = self.spectral_angle_loss(pred, target)
        
        # Gradient loss (edge preservation)
        gradient_loss = self.gradient_loss(pred, target)
        
        # Combine losses
        base_loss = (self.alpha_pixel * pixel_loss + 
                     self.alpha_spectral * spectral_loss + 
                     self.alpha_gradient * gradient_loss)
        
        # Apply SZA weighting if enabled
        if sza is not None and self.use_sza_weighting:
            if isinstance(sza, (list, tuple)):
                weights = [self.compute_sza_weight(s) for s in sza]
                sza_weight = np.mean(weights)
            else:
                sza_weight = self.compute_sza_weight(sza)
        else:
            sza_weight = 1.0
        
        return sza_weight * base_loss


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

# Band names for reporting (S2 10m bands used in training)
BAND_NAMES = ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)']


def compute_psnr(pred, target):
    """Compute PSNR between prediction and target."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_psnr_per_band(pred, target):
    """Compute PSNR for each band separately.
    
    Args:
        pred: (B, C, H, W) predicted SR
        target: (B, C, H, W) ground truth HR
    
    Returns:
        List of PSNR values, one per band
    """
    num_bands = pred.shape[1]
    psnr_per_band = []
    
    for b in range(num_bands):
        mse = F.mse_loss(pred[:, b:b+1], target[:, b:b+1]).item()
        if mse < 1e-10:
            psnr_per_band.append(100.0)
        else:
            psnr_per_band.append(10 * np.log10(1.0 / mse))
    
    return psnr_per_band


def compute_ssim(pred, target, window_size=11):
    """Compute SSIM between prediction and target."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Simple mean-based SSIM approximation
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
    
    ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
           ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    
    return ssim.mean().item()


def compute_ssim_per_band(pred, target, window_size=11):
    """Compute SSIM for each band separately.
    
    Args:
        pred: (B, C, H, W) predicted SR
        target: (B, C, H, W) ground truth HR
    
    Returns:
        List of SSIM values, one per band
    """
    num_bands = pred.shape[1]
    ssim_per_band = []
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    for b in range(num_bands):
        pred_b = pred[:, b:b+1]
        target_b = target[:, b:b+1]
        
        mu_pred = F.avg_pool2d(pred_b, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target_b, window_size, stride=1, padding=window_size//2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d(pred_b ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target_b ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred_b * target_b, window_size, stride=1, padding=window_size//2) - mu_pred_target
        
        ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        ssim_per_band.append(ssim.mean().item())
    
    return ssim_per_band


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, target_size):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        lr = batch['lr'].to(device)  # (B, T, C, H_lr, W_lr)
        hr = batch['hr'].to(device)  # (B, T, C, H_hr, W_hr)
        
        # Get SZA if available (for SpectralFidelityLoss)
        sza = batch.get('sza', None)
        if sza is not None:
            sza = sza.tolist() if torch.is_tensor(sza) else list(sza)
        
        # Use middle frame as HR target
        hr_target = hr[:, hr.shape[1]//2]  # (B, C, H_hr, W_hr)
        
        optimizer.zero_grad()
        
        with autocast():
            # Forward with TARGET SIZE specified!
            sr = model(lr, target_size=target_size)  # (B, C, H_hr, W_hr)
            
            # Loss - no interpolation needed, sizes should match
            assert sr.shape == hr_target.shape, f"Size mismatch: SR={sr.shape}, HR={hr_target.shape}"
            loss = criterion(sr, hr_target, sza)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device, target_size):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            hr_target = hr[:, hr.shape[1]//2]
            
            # Get SZA if available
            sza = batch.get('sza', None)
            if sza is not None:
                sza = sza.tolist() if torch.is_tensor(sza) else list(sza)
            
            with autocast():
                sr = model(lr, target_size=target_size)
                loss = criterion(sr, hr_target, sza)
            
            # Metrics
            psnr = compute_psnr(sr.float(), hr_target.float())
            ssim = compute_ssim(sr.float(), hr_target.float())
            
            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += ssim
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches
    }


def validate_with_per_band_metrics(model, dataloader, criterion, device, target_size):
    """Validate model with per-band metrics reporting."""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    
    # Per-band accumulators
    num_bands = None
    psnr_per_band_accum = None
    ssim_per_band_accum = None
    
    with torch.no_grad():
        for batch in dataloader:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            hr_target = hr[:, hr.shape[1]//2]
            
            # Get SZA if available
            sza = batch.get('sza', None)
            if sza is not None:
                sza = sza.tolist() if torch.is_tensor(sza) else list(sza)
            
            with autocast():
                sr = model(lr, target_size=target_size)
                loss = criterion(sr, hr_target, sza)
            
            # Overall metrics
            psnr = compute_psnr(sr.float(), hr_target.float())
            ssim = compute_ssim(sr.float(), hr_target.float())
            
            # Per-band metrics
            psnr_bands = compute_psnr_per_band(sr.float(), hr_target.float())
            ssim_bands = compute_ssim_per_band(sr.float(), hr_target.float())
            
            # Initialize accumulators on first batch
            if num_bands is None:
                num_bands = len(psnr_bands)
                psnr_per_band_accum = [0.0] * num_bands
                ssim_per_band_accum = [0.0] * num_bands
            
            # Accumulate per-band metrics
            for b in range(num_bands):
                psnr_per_band_accum[b] += psnr_bands[b]
                ssim_per_band_accum[b] += ssim_bands[b]
            
            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += ssim
            num_batches += 1
    
    # Average per-band metrics
    psnr_per_band_avg = [p / num_batches for p in psnr_per_band_accum] if psnr_per_band_accum else []
    ssim_per_band_avg = [s / num_batches for s in ssim_per_band_accum] if ssim_per_band_accum else []
    
    return {
        'loss': total_loss / num_batches,
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches,
        'psnr_per_band': psnr_per_band_avg,
        'ssim_per_band': ssim_per_band_avg
    }


def save_visualization(model, dataloader, device, target_size, output_path, num_samples=5):
    """Save visualization of SR results."""
    import matplotlib.pyplot as plt
    
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # Get middle LR and HR frames
            lr_vis = lr[0, lr.shape[1]//2].cpu().numpy()  # (C, H_lr, W_lr)
            hr_vis = hr[0, hr.shape[1]//2].cpu().numpy()  # (C, H_hr, W_hr)
            
            # Generate SR
            with autocast():
                sr = model(lr[:1], target_size=target_size)
            sr_vis = sr[0].cpu().float().numpy()  # (C, H_hr, W_hr)
            
            # Upsample LR for comparison
            lr_up = F.interpolate(
                torch.from_numpy(lr_vis).unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            )[0].numpy()
            
            # Convert to RGB (take first 3 bands, transpose to H,W,C)
            lr_rgb = np.clip(lr_up[:3].transpose(1, 2, 0), 0, 1)
            sr_rgb = np.clip(sr_vis[:3].transpose(1, 2, 0), 0, 1)
            hr_rgb = np.clip(hr_vis[:3].transpose(1, 2, 0), 0, 1)
            
            # Difference map
            diff = np.abs(sr_rgb - hr_rgb).mean(axis=-1)
            
            # Plot
            axes[i, 0].imshow(lr_rgb)
            axes[i, 0].set_title(f'LR (bicubic upsampled)')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(sr_rgb)
            axes[i, 1].set_title(f'SR (model output)')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(hr_rgb)
            axes[i, 2].set_title(f'HR (ground truth)')
            axes[i, 2].axis('off')
            
            im = axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=0.1)
            axes[i, 3].set_title(f'|SR - HR| (error)')
            axes[i, 3].axis('off')
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
            
            # Compute metrics for this sample
            psnr = compute_psnr(
                torch.from_numpy(sr_vis).unsqueeze(0),
                torch.from_numpy(hr_vis).unsqueeze(0)
            )
            axes[i, 1].set_title(f'SR (PSNR: {psnr:.2f} dB)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Native Resolution SR Training')
    parser.add_argument('--loss', type=str, default='simple', choices=['simple', 'spectral'],
                        help='Loss function: "simple" (L1+Gradient) or "spectral" (L1+MSE+SAM+Gradient)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--use-sza', action='store_true', help='Use solar zenith angle weighting')
    args = parser.parse_args()
    
    # Configuration
    config = {
        # Data
        'data_dir': 'data/sr_training',
        'years': [2020, 2021, 2022, 2023, 2024],
        
        # Model
        'in_channels': 3,
        'out_channels': 3,
        'base_channels': 64,
        'num_encoder_blocks': 4,
        'num_frames': 3,
        
        # Training
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'val_split': 0.1,
        
        # Loss function
        'loss_type': args.loss,
        'use_sza_weighting': args.use_sza,
        
        # Output - include loss type in directory name
        'output_dir': f'outputs/native_res_training_{args.loss}_loss',
        'save_every': 10,
    }
    
    print("=" * 70)
    print("Native Resolution SR Training")
    print("=" * 70)
    print(f"Loss function: {config['loss_type'].upper()}")
    if config['loss_type'] == 'simple':
        print("  Formula: L1 + 0.5×Gradient")
    else:
        print("  Formula: L1 + 0.1×MSE + 0.1×SAM + 0.01×Gradient")
        print(f"  SZA weighting: {'Enabled' if config['use_sza_weighting'] else 'Disabled'}")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load datasets
    pickle_paths = []
    for year in config['years']:
        path = Path(config['data_dir']) / f'sta2_{year}_native_res' / 'dataset.pkl'
        if path.exists():
            pickle_paths.append(str(path))
            print(f"  Found: {path}")
        else:
            print(f"  Missing: {path}")
    
    if not pickle_paths:
        print("ERROR: No datasets found!")
        return
    
    # Create dataset
    full_dataset = NativeResolutionSRDataset(
        pickle_paths=pickle_paths,
        sequence_length=config['num_frames'],
        augment=True
    )
    
    # Get actual sizes from dataset
    lr_size = full_dataset.lr_size  # (77, 77)
    hr_size = full_dataset.hr_size  # (256, 256)
    scale_factor = full_dataset.scale_factor  # 3.32
    target_size = hr_size
    
    print(f"LR size: {lr_size[0]}×{lr_size[1]} (10m)")
    print(f"HR size: {hr_size[0]}×{hr_size[1]} (3m)")
    print(f"Scale factor: {scale_factor:.2f}×")
    print("=" * 70)
    
    # Split train/val
    n_val = int(len(full_dataset) * config['val_split'])
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Model - use scale_factor from dataset
    model = SingleResolutionSatelliteSR(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        base_channels=config['base_channels'],
        num_encoder_blocks=config['num_encoder_blocks'],
        num_frames=config['num_frames'],
        scale_factor=scale_factor  # Use actual scale from dataset
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function selection
    if config['loss_type'] == 'spectral':
        criterion = SpectralFidelityLoss(
            alpha_pixel=1.0,       # L1 + 0.1×MSE
            alpha_spectral=0.1,    # SAM
            alpha_gradient=0.01,   # Gradient
            use_sza_weighting=config['use_sza_weighting']
        )
        print("Loss: SpectralFidelityLoss (L1 + 0.1×MSE + 0.1×SAM + 0.01×Gradient)")
    else:
        criterion = SRLoss(l1_weight=1.0, ssim_weight=0.5)
        print("Loss: SRLoss (L1 + 0.5×Gradient)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=1e-6
    )
    scaler = GradScaler()
    
    # Output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_psnr = 0
    
    print("\nStarting training...")
    for epoch in range(config['num_epochs']):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, target_size
        )
        
        # Validate with per-band metrics
        val_metrics = validate_with_per_band_metrics(model, val_loader, criterion, device, target_size)
        
        # Update scheduler
        scheduler.step()
        
        # Log overall metrics
        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_metrics['loss']:.4f}, "
              f"PSNR={val_metrics['psnr']:.2f} dB, "
              f"SSIM={val_metrics['ssim']:.4f}")
        
        # Log per-band metrics every 10 epochs or on first/last epoch
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == config['num_epochs'] - 1:
            psnr_bands = val_metrics['psnr_per_band']
            ssim_bands = val_metrics['ssim_per_band']
            
            print(f"  Per-band PSNR: ", end="")
            for b, (name, psnr_b) in enumerate(zip(BAND_NAMES, psnr_bands)):
                print(f"{name}={psnr_b:.2f}", end="  " if b < len(psnr_bands)-1 else "\n")
            
            print(f"  Per-band SSIM: ", end="")
            for b, (name, ssim_b) in enumerate(zip(BAND_NAMES, ssim_bands)):
                print(f"{name}={ssim_b:.4f}", end="  " if b < len(ssim_bands)-1 else "\n")
        
        # Save best model
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'psnr_per_band': val_metrics['psnr_per_band'],
                'ssim_per_band': val_metrics['ssim_per_band'],
                'config': config
            }, output_dir / 'best_model.pth')
            print(f"  → New best PSNR: {best_psnr:.2f} dB")
        
        # Save checkpoint and visualization
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': val_metrics['psnr'],
                'psnr_per_band': val_metrics['psnr_per_band'],
                'ssim_per_band': val_metrics['ssim_per_band'],
                'config': config
            }, output_dir / f'checkpoint_epoch{epoch+1}.pth')
            
            save_visualization(
                model, val_loader, device, target_size,
                output_dir / f'visualization_epoch{epoch+1}.png'
            )
    
    # Final per-band metrics summary
    print(f"\n{'='*70}")
    print("FINAL PER-BAND METRICS")
    print(f"{'='*70}")
    val_metrics = validate_with_per_band_metrics(model, val_loader, criterion, device, target_size)
    print(f"{'Band':<15} {'PSNR (dB)':<12} {'SSIM':<10}")
    print("-" * 40)
    for b, name in enumerate(BAND_NAMES):
        if b < len(val_metrics['psnr_per_band']):
            print(f"{name:<15} {val_metrics['psnr_per_band'][b]:<12.2f} {val_metrics['ssim_per_band'][b]:<10.4f}")
    print("-" * 40)
    print(f"{'Overall':<15} {val_metrics['psnr']:<12.2f} {val_metrics['ssim']:<10.4f}")
    print(f"{'='*70}")
    
    print(f"\nTraining complete! Best PSNR: {best_psnr:.2f} dB")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    main()
