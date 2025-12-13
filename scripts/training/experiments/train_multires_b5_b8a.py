#!/usr/bin/env python3
"""
Multi-Resolution SR Training with B5 and B8A bands.

Uses 2023-2024 data which have:
- LR 10m: (T, 77, 77, 3) - B2, B3, B4
- LR 20m: (T, 38, 38, 2) - B5, B8A
- HR: (T, 256, 256, 3) - Planet RGB at 3m

Trains with SpectralFidelityLoss including SAM for spectral consistency.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent / 'models'))

import pickle
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


# =============================================================================
# TRAINING SAMPLE DATACLASS
# =============================================================================

@dataclass
class TrainingSample:
    lr_sequence: np.ndarray = None      # (T, 77, 77, 3) - B2, B3, B4
    hr_sequence: np.ndarray = None      # (T, 256, 256, 3) - Planet RGB
    lr_20m_sequence: np.ndarray = None  # (T, 38, 38, 2) - B5, B8A
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

# Inject TrainingSample into __main__ for pickle
import types
_main_module = types.ModuleType('__main__')
_main_module.TrainingSample = TrainingSample
sys.modules['__main__'] = _main_module

from satellite_sr_model import MultiResolutionSatelliteSR


# =============================================================================
# BAND NAMES
# =============================================================================

BAND_NAMES_10M = ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)']
BAND_NAMES_20M = ['B5 (Red Edge)', 'B8A (NIR)']


# =============================================================================
# DATASET
# =============================================================================

class MultiResolutionSRDataset(Dataset):
    """
    Dataset for multi-resolution SR training with B5 and B8A bands.
    
    Expects samples with:
    - LR 10m: (T, 77, 77, 3) @ 10m - B2, B3, B4
    - LR 20m: (T, 38, 38, 2) @ 20m - B5, B8A  
    - HR: (T, 256, 256, 3) @ 3m - Planet RGB
    """
    
    def __init__(
        self,
        pickle_paths: List[str],
        sequence_length: int = 3,
        augment: bool = True,
        require_20m_bands: int = 2  # Require B5 + B8A
    ):
        self.sequence_length = sequence_length
        self.augment = augment
        self.require_20m_bands = require_20m_bands
        
        # Load samples that have the required 20m bands
        self.samples = []
        for path in pickle_paths:
            print(f"Loading {path}...")
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            for s in data.get('samples', []):
                # Check if sample has 20m data with required bands
                if (hasattr(s, 'lr_20m_sequence') and 
                    s.lr_20m_sequence is not None and
                    s.lr_20m_sequence.shape[-1] >= require_20m_bands):
                    self.samples.append(s)
        
        print(f"Total samples with B5+B8A: {len(self.samples)}")
        
        # Get sizes from first sample
        self.lr_10m_size = None
        self.lr_20m_size = None
        self.hr_size = None
        
        if self.samples:
            s = self.samples[0]
            lr_10m_shape = s.lr_sequence.shape
            lr_20m_shape = s.lr_20m_sequence.shape
            hr_shape = s.hr_sequence.shape
            
            self.lr_10m_size = (lr_10m_shape[1], lr_10m_shape[2])  # (77, 77)
            self.lr_20m_size = (lr_20m_shape[1], lr_20m_shape[2])  # (38, 38)
            self.hr_size = (hr_shape[1], hr_shape[2])              # (256, 256)
            
            print(f"LR 10m shape: {lr_10m_shape} (T, H, W, C)")
            print(f"LR 20m shape: {lr_20m_shape} (T, H, W, C)")
            print(f"HR shape: {hr_shape} (T, H, W, C)")
            
            # Scale factors
            self.scale_10m = hr_shape[1] / lr_10m_shape[1]  # 256/77 ≈ 3.32
            self.scale_20m = (hr_shape[1] // 2) / lr_20m_shape[1]  # 128/38 ≈ 3.37
            print(f"Scale 10m: {self.scale_10m:.2f}×, Scale 20m: {self.scale_20m:.2f}×")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert (T, H, W, C) → (T, C, H, W) for PyTorch
        lr_10m = sample.lr_sequence.copy().transpose(0, 3, 1, 2).astype(np.float32)
        lr_20m = sample.lr_20m_sequence.copy().transpose(0, 3, 1, 2).astype(np.float32)
        hr = sample.hr_sequence.copy().transpose(0, 3, 1, 2).astype(np.float32)
        
        # Select frames
        T = lr_10m.shape[0]
        if T > self.sequence_length:
            start = random.randint(0, T - self.sequence_length)
            lr_10m = lr_10m[start:start + self.sequence_length]
            lr_20m = lr_20m[start:start + self.sequence_length]
            hr = hr[start:start + self.sequence_length]
        elif T < self.sequence_length:
            # Repeat last frame
            while lr_10m.shape[0] < self.sequence_length:
                lr_10m = np.concatenate([lr_10m, lr_10m[-1:]], axis=0)
                lr_20m = np.concatenate([lr_20m, lr_20m[-1:]], axis=0)
                hr = np.concatenate([hr, hr[-1:]], axis=0)
        
        # Augmentation
        if self.augment:
            lr_10m, lr_20m, hr = self._augment(lr_10m, lr_20m, hr)
        
        result = {
            'lr_10m': torch.from_numpy(lr_10m),  # (T, 3, 77, 77)
            'lr_20m': torch.from_numpy(lr_20m),  # (T, 2, 38, 38)
            'hr': torch.from_numpy(hr)           # (T, 3, 256, 256)
        }
        
        # SZA for weighting
        if hasattr(sample, 's2_solar_zenith') and sample.s2_solar_zenith is not None:
            sza_val = sample.s2_solar_zenith
            if isinstance(sza_val, (list, tuple, np.ndarray)):
                mid_idx = len(sza_val) // 2
                result['sza'] = float(sza_val[mid_idx])
            else:
                result['sza'] = float(sza_val)
        
        return result
    
    def _augment(self, lr_10m, lr_20m, hr):
        """Synchronized augmentation for all inputs."""
        # Random horizontal flip
        if random.random() > 0.5:
            lr_10m = np.flip(lr_10m, axis=3).copy()
            lr_20m = np.flip(lr_20m, axis=3).copy()
            hr = np.flip(hr, axis=3).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            lr_10m = np.flip(lr_10m, axis=2).copy()
            lr_20m = np.flip(lr_20m, axis=2).copy()
            hr = np.flip(hr, axis=2).copy()
        
        # Random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            lr_10m = np.rot90(lr_10m, k, axes=(2, 3)).copy()
            lr_20m = np.rot90(lr_20m, k, axes=(2, 3)).copy()
            hr = np.rot90(hr, k, axes=(2, 3)).copy()
        
        return lr_10m, lr_20m, hr


# =============================================================================
# SPECTRAL FIDELITY LOSS
# =============================================================================

class SpectralFidelityLoss(nn.Module):
    """
    Multi-component loss for satellite SR with spectral consistency.
    
    Loss = SZA_weight × (α_pixel × (L1 + 0.1×MSE) + α_spectral × SAM + α_gradient × Gradient)
    """
    
    def __init__(self, alpha_pixel=1.0, alpha_spectral=0.1, alpha_gradient=0.01,
                 sza_optimal=45.0, sza_sigma=30.0, use_sza_weighting=False):
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
        if sza is None or not self.use_sza_weighting:
            return 1.0
        deviation = sza - self.sza_optimal
        weight = np.exp(-(deviation ** 2) / (2 * self.sza_sigma ** 2))
        return float(weight)
    
    def spectral_angle_loss(self, pred, target):
        """SAM loss - measures spectral fidelity."""
        pred_norm = F.normalize(pred, dim=1, eps=1e-8)
        target_norm = F.normalize(target, dim=1, eps=1e-8)
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        return 1.0 - cos_sim.mean()
    
    def gradient_loss(self, pred, target):
        """Edge preservation loss."""
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return self.l1_loss(pred_dx, target_dx) + self.l1_loss(pred_dy, target_dy)
    
    def forward(self, pred, target, sza=None):
        # Pixel loss
        pixel_loss = self.l1_loss(pred, target) + 0.1 * self.mse_loss(pred, target)
        
        # Spectral angle loss (only if channels > 1)
        if pred.shape[1] > 1:
            spectral_loss = self.spectral_angle_loss(pred, target)
        else:
            spectral_loss = torch.tensor(0.0, device=pred.device)
        
        # Gradient loss
        gradient_loss = self.gradient_loss(pred, target)
        
        # Combine
        base_loss = (self.alpha_pixel * pixel_loss + 
                     self.alpha_spectral * spectral_loss + 
                     self.alpha_gradient * gradient_loss)
        
        # SZA weighting
        if sza is not None and self.use_sza_weighting:
            if isinstance(sza, (list, tuple)):
                weights = [self.compute_sza_weight(s) for s in sza]
                sza_weight = np.mean(weights)
            else:
                sza_weight = self.compute_sza_weight(sza)
        else:
            sza_weight = 1.0
        
        return sza_weight * base_loss


class MultiResolutionSpectralLoss(nn.Module):
    """
    Combined loss for multi-resolution training.
    
    Computes loss for both 10m and 20m branches.
    """
    
    def __init__(self, weight_10m=1.0, weight_20m=0.5, use_sza_weighting=False):
        super().__init__()
        self.weight_10m = weight_10m
        self.weight_20m = weight_20m
        
        # Separate loss functions for each resolution
        self.loss_10m = SpectralFidelityLoss(
            alpha_pixel=1.0, alpha_spectral=0.1, alpha_gradient=0.01,
            use_sza_weighting=use_sza_weighting
        )
        self.loss_20m = SpectralFidelityLoss(
            alpha_pixel=1.0, alpha_spectral=0.1, alpha_gradient=0.01,
            use_sza_weighting=use_sza_weighting
        )
    
    def forward(self, sr_10m, sr_20m, hr_10m, hr_20m, sza=None):
        """
        Args:
            sr_10m: (B, 3, H, W) - predicted 10m bands
            sr_20m: (B, 2, H/2, W/2) - predicted 20m bands (B5, B8A)
            hr_10m: (B, 3, H, W) - target 10m bands
            hr_20m: (B, 2, H/2, W/2) - target 20m bands
            sza: Solar zenith angles
        """
        loss_10m = self.loss_10m(sr_10m, hr_10m, sza)
        loss_20m = self.loss_20m(sr_20m, hr_20m, sza)
        
        total_loss = self.weight_10m * loss_10m + self.weight_20m * loss_20m
        
        return total_loss, loss_10m, loss_20m


# =============================================================================
# METRICS
# =============================================================================

def compute_psnr(pred, target, max_val=1.0):
    """Compute PSNR between prediction and target."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 10 * torch.log10(max_val ** 2 / mse).item()


def compute_psnr_per_band(pred, target, max_val=1.0):
    """Compute PSNR for each band separately."""
    psnr_per_band = []
    for b in range(pred.shape[1]):
        mse = torch.mean((pred[:, b] - target[:, b]) ** 2)
        if mse == 0:
            psnr_per_band.append(100.0)
        else:
            psnr_per_band.append(10 * torch.log10(max_val ** 2 / mse).item())
    return psnr_per_band


def compute_ssim_per_band(pred, target, window_size=11):
    """Compute SSIM for each band."""
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


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, 
                target_size_10m, target_size_20m):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_loss_10m = 0
    total_loss_20m = 0
    num_batches = 0
    
    for batch in dataloader:
        lr_10m = batch['lr_10m'].to(device)  # (B, T, 3, H_10m, W_10m)
        lr_20m = batch['lr_20m'].to(device)  # (B, T, 2, H_20m, W_20m)
        hr = batch['hr'].to(device)          # (B, T, 3, H_hr, W_hr)
        
        sza = batch.get('sza', None)
        if sza is not None:
            sza = sza.tolist() if torch.is_tensor(sza) else list(sza)
        
        # Use middle frame as HR target
        hr_target = hr[:, hr.shape[1]//2]  # (B, 3, 256, 256)
        
        # Create "HR" target for 20m bands
        # Since we don't have real HR for B5/B8A, we use upsampled LR as pseudo-target
        lr_20m_mid = lr_20m[:, lr_20m.shape[1]//2]  # (B, 2, 38, 38)
        hr_20m_target = F.interpolate(lr_20m_mid, size=target_size_20m, 
                                       mode='bilinear', align_corners=False)
        
        optimizer.zero_grad()
        
        with autocast():
            # Forward
            sr_10m, sr_20m = model(lr_10m, lr_20m, 
                                   target_size_10m=target_size_10m,
                                   target_size_20m=target_size_20m)
            
            # Loss
            loss, loss_10m, loss_20m = criterion(sr_10m, sr_20m, hr_target, hr_20m_target, sza)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_loss_10m += loss_10m.item()
        total_loss_20m += loss_20m.item()
        num_batches += 1
    
    return (total_loss / num_batches, 
            total_loss_10m / num_batches, 
            total_loss_20m / num_batches)


def validate(model, dataloader, criterion, device, target_size_10m, target_size_20m):
    """Validate model with per-band metrics."""
    model.eval()
    
    total_loss = 0
    total_psnr_10m = 0
    total_psnr_20m = 0
    num_batches = 0
    
    psnr_10m_bands = [0.0, 0.0, 0.0]  # B2, B3, B4
    psnr_20m_bands = [0.0, 0.0]       # B5, B8A
    ssim_10m_bands = [0.0, 0.0, 0.0]
    ssim_20m_bands = [0.0, 0.0]
    
    with torch.no_grad():
        for batch in dataloader:
            lr_10m = batch['lr_10m'].to(device)
            lr_20m = batch['lr_20m'].to(device)
            hr = batch['hr'].to(device)
            
            sza = batch.get('sza', None)
            if sza is not None:
                sza = sza.tolist() if torch.is_tensor(sza) else list(sza)
            
            hr_target = hr[:, hr.shape[1]//2]
            lr_20m_mid = lr_20m[:, lr_20m.shape[1]//2]
            hr_20m_target = F.interpolate(lr_20m_mid, size=target_size_20m,
                                          mode='bilinear', align_corners=False)
            
            with autocast():
                sr_10m, sr_20m = model(lr_10m, lr_20m,
                                       target_size_10m=target_size_10m,
                                       target_size_20m=target_size_20m)
                loss, _, _ = criterion(sr_10m, sr_20m, hr_target, hr_20m_target, sza)
            
            # Per-band metrics for 10m
            psnr_10m = compute_psnr_per_band(sr_10m.float(), hr_target.float())
            ssim_10m = compute_ssim_per_band(sr_10m.float(), hr_target.float())
            
            # Per-band metrics for 20m
            psnr_20m = compute_psnr_per_band(sr_20m.float(), hr_20m_target.float())
            ssim_20m = compute_ssim_per_band(sr_20m.float(), hr_20m_target.float())
            
            for i in range(3):
                psnr_10m_bands[i] += psnr_10m[i]
                ssim_10m_bands[i] += ssim_10m[i]
            for i in range(2):
                psnr_20m_bands[i] += psnr_20m[i]
                ssim_20m_bands[i] += ssim_20m[i]
            
            total_loss += loss.item()
            total_psnr_10m += np.mean(psnr_10m)
            total_psnr_20m += np.mean(psnr_20m)
            num_batches += 1
    
    # Average
    psnr_10m_bands = [p / num_batches for p in psnr_10m_bands]
    psnr_20m_bands = [p / num_batches for p in psnr_20m_bands]
    ssim_10m_bands = [s / num_batches for s in ssim_10m_bands]
    ssim_20m_bands = [s / num_batches for s in ssim_20m_bands]
    
    return {
        'loss': total_loss / num_batches,
        'psnr_10m': total_psnr_10m / num_batches,
        'psnr_20m': total_psnr_20m / num_batches,
        'psnr_10m_bands': psnr_10m_bands,
        'psnr_20m_bands': psnr_20m_bands,
        'ssim_10m_bands': ssim_10m_bands,
        'ssim_20m_bands': ssim_20m_bands
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Resolution SR Training with B5/B8A')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--use-sza', action='store_true', help='Use SZA weighting')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_dir': 'data/sr_training',
        'years': [2023, 2024],  # Only years with B5 + B8A
        
        'base_channels': 64,
        'num_encoder_blocks': 4,
        'num_frames': 3,
        
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'val_split': 0.1,
        
        'use_sza_weighting': args.use_sza,
        'output_dir': 'outputs/multires_b5_b8a_training',
        'save_every': 10,
    }
    
    print("=" * 70)
    print("Multi-Resolution SR Training (B5 + B8A)")
    print("=" * 70)
    print(f"Years: {config['years']} (have B5 + B8A)")
    print(f"Loss: SpectralFidelityLoss (L1 + 0.1×MSE + 0.1×SAM + 0.01×Gradient)")
    print(f"SZA weighting: {'Enabled' if config['use_sza_weighting'] else 'Disabled'}")
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
    full_dataset = MultiResolutionSRDataset(
        pickle_paths=pickle_paths,
        sequence_length=config['num_frames'],
        augment=True,
        require_20m_bands=2  # Require B5 + B8A
    )
    
    # Get sizes
    lr_10m_size = full_dataset.lr_10m_size  # (77, 77)
    lr_20m_size = full_dataset.lr_20m_size  # (38, 38)
    hr_size = full_dataset.hr_size          # (256, 256)
    
    # Target sizes for outputs
    target_size_10m = hr_size               # (256, 256) for 10m bands
    target_size_20m = (hr_size[0]//2, hr_size[1]//2)  # (128, 128) for 20m bands
    
    print(f"Target 10m size: {target_size_10m}")
    print(f"Target 20m size: {target_size_20m}")
    print("=" * 70)
    
    # Split dataset
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
    
    # Model
    model = MultiResolutionSatelliteSR(
        base_channels=config['base_channels'],
        num_encoder_blocks=config['num_encoder_blocks'],
        num_frames=config['num_frames'],
        scale_factor_10m=full_dataset.scale_10m,
        scale_factor_20m=full_dataset.scale_20m,
        use_cross_resolution_fusion=True,
        in_channels_10m=3,   # B2, B3, B4
        in_channels_20m=2,   # B5, B8A
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss
    criterion = MultiResolutionSpectralLoss(
        weight_10m=1.0,
        weight_20m=0.5,
        use_sza_weighting=config['use_sza_weighting']
    )
    print("Loss: MultiResolutionSpectralLoss (10m + 0.5×20m)")
    
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
        train_loss, train_loss_10m, train_loss_20m = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            target_size_10m, target_size_20m
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device,
                               target_size_10m, target_size_20m)
        
        scheduler.step()
        
        # Log
        avg_psnr = (val_metrics['psnr_10m'] + val_metrics['psnr_20m']) / 2
        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Train Loss={train_loss:.4f} (10m={train_loss_10m:.4f}, 20m={train_loss_20m:.4f}), "
              f"Val Loss={val_metrics['loss']:.4f}, "
              f"PSNR 10m={val_metrics['psnr_10m']:.2f} dB, "
              f"PSNR 20m={val_metrics['psnr_20m']:.2f} dB")
        
        # Per-band metrics every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == config['num_epochs'] - 1:
            print("  10m bands:")
            for b, (name, psnr, ssim) in enumerate(zip(
                BAND_NAMES_10M, val_metrics['psnr_10m_bands'], val_metrics['ssim_10m_bands']
            )):
                print(f"    {name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
            
            print("  20m bands:")
            for b, (name, psnr, ssim) in enumerate(zip(
                BAND_NAMES_20M, val_metrics['psnr_20m_bands'], val_metrics['ssim_20m_bands']
            )):
                print(f"    {name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'config': config
            }, output_dir / 'best_model.pt')
            print(f"  → New best avg PSNR: {best_psnr:.2f} dB")
        
        # Periodic checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    print(f"\nTraining complete! Best avg PSNR: {best_psnr:.2f} dB")
    print(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    main()
