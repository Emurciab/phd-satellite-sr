#!/usr/bin/env python3
"""
Training script for 5-band UNIFORM 3m resolution super-resolution.

This script trains a model that super-resolves ALL bands to 3m resolution:
1. 10m bands (B2, B3, B4): 77×77 → 256×256 (3.33× upscaling)
2. 20m bands (B5, B8A):    38×38 → 256×256 (6.74× upscaling) ← KEY DIFFERENCE

Both branches output at the SAME 3m resolution (256×256), enabling:
- Unified pixel-aligned 5-band output
- Direct band stacking for vegetation indices
- Consistent spatial resolution for all spectral analysis

HR targets from Planet SuperDove (all at 3m):
- 10m HR: Planet B2, B4, B6 (Blue, Green, Red) at 3m → 256×256
- 20m HR: Planet B7, B8 (RedEdge, NIR) at 3m → 256×256

Comparison with train_5band_multires.py:
┌──────────────────┬─────────────────────┬─────────────────────┐
│                  │  train_5band_multires │ train_5band_uniform_3m │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 10m bands output │  256×256 (3m)       │  256×256 (3m)       │
│ 20m bands output │  128×128 (6m)       │  256×256 (3m)       │
│ 20m scale factor │  3.33×              │  6.74×              │
│ Output alignment │  Different sizes    │  Pixel-aligned      │
└──────────────────┴─────────────────────┴─────────────────────┘
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Import model and loss
from models.satellite_sr_model import MultiResolutionSatelliteSR, SpectralFidelityLoss


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Early stopping to halt training when validation metrics stop improving.
    
    Monitors multiple metrics with configurable patience and minimum delta.
    Supports both 'min' mode (loss) and 'max' mode (PSNR, SSIM).
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = 'min',
        metric_name: str = 'val_loss'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric_name = metric_name
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
        self.history = []
    
    def __call__(self, value: float, epoch: int) -> bool:
        self.history.append(value)
        
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def status(self) -> str:
        return f"{self.metric_name}: best={self.best_value:.4f} @ epoch {self.best_epoch+1}, patience={self.counter}/{self.patience}"


class ConvergenceTracker:
    """Track multiple metrics for convergence analysis."""
    
    def __init__(
        self,
        patience_loss: int = 15,
        patience_psnr: int = 20,
        min_delta_loss: float = 1e-4,
        min_delta_psnr: float = 0.1,
        plateau_window: int = 10,
        plateau_threshold: float = 0.05
    ):
        self.es_loss = EarlyStopping(patience_loss, min_delta_loss, 'min', 'val_loss')
        self.es_psnr = EarlyStopping(patience_psnr, min_delta_psnr, 'max', 'avg_psnr')
        
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'psnr_10m': [],
            'psnr_20m': [],
            'avg_psnr': [],
        }
    
    def update(self, epoch: int, train_loss: float, val_loss: float, 
               psnr_10m: float, psnr_20m: float) -> Dict:
        avg_psnr = (psnr_10m + psnr_20m) / 2
        
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['psnr_10m'].append(psnr_10m)
        self.metrics_history['psnr_20m'].append(psnr_20m)
        self.metrics_history['avg_psnr'].append(avg_psnr)
        
        loss_stop = self.es_loss(val_loss, epoch)
        psnr_stop = self.es_psnr(avg_psnr, epoch)
        plateau = self._check_plateau()
        
        should_stop = False
        stop_reason = None
        
        if loss_stop and psnr_stop:
            should_stop = True
            stop_reason = "Both loss and PSNR stopped improving"
        elif plateau and (self.es_loss.counter >= 5 or self.es_psnr.counter >= 5):
            should_stop = True
            stop_reason = f"Metrics plateaued (variation < {self.plateau_threshold*100:.1f}%)"
        
        return {
            'should_stop': should_stop,
            'stop_reason': stop_reason,
            'status': {
                'loss': self.es_loss.status(),
                'psnr': self.es_psnr.status(),
                'plateau': plateau,
                'best_loss': self.es_loss.best_value,
                'best_psnr': self.es_psnr.best_value,
            }
        }
    
    def _check_plateau(self) -> bool:
        if len(self.metrics_history['val_loss']) < self.plateau_window:
            return False
        
        recent_loss = self.metrics_history['val_loss'][-self.plateau_window:]
        recent_psnr = self.metrics_history['avg_psnr'][-self.plateau_window:]
        
        loss_cv = np.std(recent_loss) / (np.mean(recent_loss) + 1e-8)
        psnr_cv = np.std(recent_psnr) / (np.mean(recent_psnr) + 1e-8)
        
        return loss_cv < self.plateau_threshold and psnr_cv < self.plateau_threshold
    
    def get_summary(self) -> str:
        return (
            f"\nConvergence Summary:\n"
            f"  Best val_loss: {self.es_loss.best_value:.4f} @ epoch {self.es_loss.best_epoch+1}\n"
            f"  Best avg_PSNR: {self.es_psnr.best_value:.2f} dB @ epoch {self.es_psnr.best_epoch+1}\n"
            f"  Total epochs: {len(self.metrics_history['val_loss'])}\n"
            f"  Loss patience used: {self.es_loss.counter}/{self.es_loss.patience}\n"
            f"  PSNR patience used: {self.es_psnr.counter}/{self.es_psnr.patience}"
        )


class FiveBandUniform3mDataset(Dataset):
    """Dataset for 5-band SuperDove data with ALL bands at 3m resolution.
    
    Key difference from FiveBandDataset: HR targets for 20m bands are at full
    256×256 resolution (3m) instead of 128×128 (6m).
    
    Supports full temporal sequences (T=3) for temporal attention.
    """
    
    def __init__(self, samples: List, sequence_length: int = 3, augment: bool = True):
        """
        Args:
            samples: List of TrainingSample5Band objects
            sequence_length: Number of temporal frames (default 3)
            augment: Whether to apply data augmentation
        """
        # Filter out samples with None values
        self.samples = [s for s in samples if self._is_valid_sample(s)]
        self.sequence_length = sequence_length
        self.augment = augment
        print(f"Uniform3mDataset: {len(self.samples)}/{len(samples)} samples valid (filtered {len(samples) - len(self.samples)} with None values)")
    
    @staticmethod
    def _is_valid_sample(sample):
        """Check if sample has all required non-None arrays."""
        return (sample.lr_sequence is not None and 
                sample.lr_20m_sequence is not None and
                sample.hr_10m_sequence is not None and
                sample.hr_20m_sequence is not None and
                sample.hr_sequence is not None)  # Also need full 5-band HR
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get arrays - (T, H, W, C) format from cache
        lr_10m = sample.lr_sequence.copy()        # (T, 77, 77, 3)
        lr_20m = sample.lr_20m_sequence.copy()    # (T, 38, 38, 2)
        hr_10m = sample.hr_10m_sequence.copy()    # (T, 256, 256, 3)
        
        # KEY DIFFERENCE: Get full-resolution HR for 20m bands (not downsampled)
        # hr_20m_sequence is at 128×128 (6m), but we need 256×256 (3m)
        # We extract from the full hr_sequence which has all bands at 3m
        hr_full = sample.hr_sequence.copy()       # (T, 256, 256, 5) - full 5-band at 3m
        hr_20m = hr_full[:, :, :, 3:5]            # (T, 256, 256, 2) - RedEdge+NIR at 3m
        
        # Handle temporal dimension
        T = lr_10m.shape[0]
        if T > self.sequence_length:
            # Random crop in time
            start = np.random.randint(0, T - self.sequence_length + 1)
            lr_10m = lr_10m[start:start + self.sequence_length]
            lr_20m = lr_20m[start:start + self.sequence_length]
            hr_10m = hr_10m[start:start + self.sequence_length]
            hr_20m = hr_20m[start:start + self.sequence_length]
        elif T < self.sequence_length:
            # Repeat last frame to reach sequence_length
            while lr_10m.shape[0] < self.sequence_length:
                lr_10m = np.concatenate([lr_10m, lr_10m[-1:]], axis=0)
                lr_20m = np.concatenate([lr_20m, lr_20m[-1:]], axis=0)
                hr_10m = np.concatenate([hr_10m, hr_10m[-1:]], axis=0)
                hr_20m = np.concatenate([hr_20m, hr_20m[-1:]], axis=0)
        
        # Apply augmentation (consistent across all arrays)
        if self.augment:
            lr_10m, lr_20m, hr_10m, hr_20m = self._augment(lr_10m, lr_20m, hr_10m, hr_20m)
        
        # Convert to torch tensors: (T, H, W, C) -> (T, C, H, W)
        lr_10m = torch.from_numpy(lr_10m).permute(0, 3, 1, 2).float()  # (T, 3, 77, 77)
        lr_20m = torch.from_numpy(lr_20m).permute(0, 3, 1, 2).float()  # (T, 2, 38, 38)
        
        # For HR targets, use middle frame (consistent with temporal attention output)
        mid_idx = self.sequence_length // 2
        hr_10m = torch.from_numpy(hr_10m[mid_idx]).permute(2, 0, 1).float()  # (3, 256, 256)
        hr_20m = torch.from_numpy(hr_20m[mid_idx]).permute(2, 0, 1).float()  # (2, 256, 256) ← SAME SIZE!
        
        return {
            'lr_10m': lr_10m,      # (T, 3, 77, 77)
            'lr_20m': lr_20m,      # (T, 2, 38, 38)
            'hr_10m': hr_10m,      # (3, 256, 256)
            'hr_20m': hr_20m,      # (2, 256, 256) ← NOW SAME SIZE AS HR_10M!
        }
    
    def _augment(self, lr_10m, lr_20m, hr_10m, hr_20m):
        """Apply consistent augmentation across all arrays."""
        import random
        
        # Random horizontal flip
        if random.random() > 0.5:
            lr_10m = np.flip(lr_10m, axis=2).copy()  # Flip W dimension
            lr_20m = np.flip(lr_20m, axis=2).copy()
            hr_10m = np.flip(hr_10m, axis=2).copy()
            hr_20m = np.flip(hr_20m, axis=2).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            lr_10m = np.flip(lr_10m, axis=1).copy()  # Flip H dimension
            lr_20m = np.flip(lr_20m, axis=1).copy()
            hr_10m = np.flip(hr_10m, axis=1).copy()
            hr_20m = np.flip(hr_20m, axis=1).copy()
        
        # Random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            lr_10m = np.rot90(lr_10m, k, axes=(1, 2)).copy()
            lr_20m = np.rot90(lr_20m, k, axes=(1, 2)).copy()
            hr_10m = np.rot90(hr_10m, k, axes=(1, 2)).copy()
            hr_20m = np.rot90(hr_20m, k, axes=(1, 2)).copy()
        
        return lr_10m, lr_20m, hr_10m, hr_20m


class MultiResSpectralLoss(nn.Module):
    """
    Combined loss for multi-resolution SR with spectral fidelity.
    
    Uses the same loss components as train_native_resolution.py:
    - L1 + MSE pixel loss
    - Spectral Angle Mapper (SAM) loss
    - Gradient/edge preservation loss
    
    Applied to both 10m and 20m branches with configurable weights.
    """
    
    def __init__(
        self,
        weight_10m: float = 1.0,
        weight_20m: float = 1.0,
        alpha_pixel: float = 1.0,
        alpha_spectral: float = 0.1,
        alpha_gradient: float = 0.01
    ):
        super().__init__()
        self.weight_10m = weight_10m
        self.weight_20m = weight_20m
        self.alpha_pixel = alpha_pixel
        self.alpha_spectral = alpha_spectral
        self.alpha_gradient = alpha_gradient
        
        # Base losses
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Sobel filters for edge detection (registered as buffers for automatic GPU transfer)
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def spectral_angle_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Spectral Angle Mapper (SAM) loss.
        Measures spectral fidelity by computing angle between spectral vectors.
        """
        pred_norm = nn.functional.normalize(pred, dim=1, eps=1e-8)
        target_norm = nn.functional.normalize(target, dim=1, eps=1e-8)
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        return 1.0 - cos_sim.mean()
    
    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gradient loss for edge preservation using Sobel filters."""
        pred_edges = self._compute_edges(pred)
        target_edges = self._compute_edges(target)
        return self.l1_loss(pred_edges, target_edges)
    
    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel filters."""
        B, C, H, W = x.shape
        edges = []
        for c in range(C):
            channel = x[:, c:c+1, :, :]
            edge_x = nn.functional.conv2d(channel, self.sobel_x, padding=1)
            edge_y = nn.functional.conv2d(channel, self.sobel_y, padding=1)
            edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
            edges.append(edge_mag)
        return torch.cat(edges, dim=1)
    
    def _compute_branch_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute full spectral fidelity loss for a single branch."""
        pixel_loss = self.l1_loss(pred, target) + 0.1 * self.mse_loss(pred, target)
        sam_loss = self.spectral_angle_loss(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        return self.alpha_pixel * pixel_loss + self.alpha_spectral * sam_loss + self.alpha_gradient * grad_loss
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m):
        loss_10m = self._compute_branch_loss(pred_10m, target_10m)
        loss_20m = self._compute_branch_loss(pred_20m, target_20m)
        total_loss = self.weight_10m * loss_10m + self.weight_20m * loss_20m
        return total_loss, loss_10m, loss_20m


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate PSNR between prediction and target."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 1.0
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr.item()


def calculate_psnr_per_band(pred: torch.Tensor, target: torch.Tensor, band_names: List[str]) -> dict:
    """Calculate PSNR for each band."""
    results = {}
    for i, name in enumerate(band_names):
        mse = torch.mean((pred[:, i:i+1] - target[:, i:i+1]) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(1.0) - 10 * np.log10(mse.item())
        results[name] = psnr
    return results


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_loss_10m = 0
    total_loss_20m = 0
    
    for batch in dataloader:
        lr_10m = batch['lr_10m'].to(device)
        lr_20m = batch['lr_20m'].to(device)
        hr_10m = batch['hr_10m'].to(device)
        hr_20m = batch['hr_20m'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - BOTH branches output 256×256
        pred_10m, pred_20m = model(
            lr_10m, lr_20m,
            target_size_10m=(hr_10m.shape[2], hr_10m.shape[3]),  # (256, 256)
            target_size_20m=(hr_20m.shape[2], hr_20m.shape[3])   # (256, 256) ← SAME!
        )
        
        # Calculate loss
        loss, loss_10m, loss_20m = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_10m += loss_10m.item()
        total_loss_20m += loss_20m.item()
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_loss_10m / n_batches, total_loss_20m / n_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_psnr_10m = 0
    total_psnr_20m = 0
    
    band_psnr_10m = {'Blue': 0, 'Green': 0, 'Red': 0}
    band_psnr_20m = {'RedEdge': 0, 'NIR': 0}
    
    with torch.no_grad():
        for batch in dataloader:
            lr_10m = batch['lr_10m'].to(device)
            lr_20m = batch['lr_20m'].to(device)
            hr_10m = batch['hr_10m'].to(device)
            hr_20m = batch['hr_20m'].to(device)
            
            pred_10m, pred_20m = model(
                lr_10m, lr_20m,
                target_size_10m=(hr_10m.shape[2], hr_10m.shape[3]),
                target_size_20m=(hr_20m.shape[2], hr_20m.shape[3])
            )
            
            loss, _, _ = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
            total_loss += loss.item()
            
            total_psnr_10m += calculate_psnr(pred_10m, hr_10m)
            total_psnr_20m += calculate_psnr(pred_20m, hr_20m)
            
            psnr_10m = calculate_psnr_per_band(pred_10m, hr_10m, ['Blue', 'Green', 'Red'])
            psnr_20m = calculate_psnr_per_band(pred_20m, hr_20m, ['RedEdge', 'NIR'])
            
            for k, v in psnr_10m.items():
                band_psnr_10m[k] += v
            for k, v in psnr_20m.items():
                band_psnr_20m[k] += v
    
    n_batches = len(dataloader)
    
    for k in band_psnr_10m:
        band_psnr_10m[k] /= n_batches
    for k in band_psnr_20m:
        band_psnr_20m[k] /= n_batches
    
    return {
        'loss': total_loss / n_batches,
        'psnr_10m': total_psnr_10m / n_batches,
        'psnr_20m': total_psnr_20m / n_batches,
        'band_psnr_10m': band_psnr_10m,
        'band_psnr_20m': band_psnr_20m,
    }


def main():
    parser = argparse.ArgumentParser(description='Train 5-band UNIFORM 3m resolution SR model')
    parser.add_argument('--data-dir', type=str, 
                        default='./data/sr_training/5band_test/superdove_5band',
                        help='Path to 5-band dataset')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/5band_uniform_3m',
                        help='Directory to save checkpoints')
    parser.add_argument('--num-frames', type=int, default=3,
                        help='Number of temporal frames')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    
    # Early stopping arguments
    parser.add_argument('--patience-loss', type=int, default=15,
                        help='Early stopping patience for validation loss')
    parser.add_argument('--patience-psnr', type=int, default=20,
                        help='Early stopping patience for PSNR')
    parser.add_argument('--min-delta-loss', type=float, default=1e-4,
                        help='Minimum loss improvement to reset patience')
    parser.add_argument('--min-delta-psnr', type=float, default=0.1,
                        help='Minimum PSNR improvement in dB to reset patience')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("5-BAND UNIFORM 3m RESOLUTION SUPER-RESOLUTION TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Temporal frames: {args.num_frames}")
    print(f"  Augmentation: {not args.no_augment}")
    print(f"\nEarly Stopping:")
    if args.no_early_stop:
        print(f"  DISABLED - will run all {args.epochs} epochs")
    else:
        print(f"  Loss patience: {args.patience_loss} epochs (min_delta={args.min_delta_loss})")
        print(f"  PSNR patience: {args.patience_psnr} epochs (min_delta={args.min_delta_psnr} dB)")
    print()
    print("Resolution Configuration (UNIFORM 3m):")
    print("  10m bands (B2,B3,B4): 77×77 → 256×256 (3.33× upscaling)")
    print("  20m bands (B5,B8A):   38×38 → 256×256 (6.74× upscaling)")
    print("  Output: All 5 bands at 256×256 (3m) - pixel-aligned!")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    
    sys.path.insert(0, '/media/eamurciab/TOSHIBA_EXT1/PhD/SR/scripts/preprocessing')
    from convert_cache_to_5band import TrainingSample5Band
    
    import __main__
    __main__.TrainingSample5Band = TrainingSample5Band
    
    data_path = Path(args.data_dir) / 'dataset.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    samples = data['samples']
    metadata = data['metadata']
    
    print(f"  Loaded {len(samples)} samples")
    print(f"  Sensor type: {metadata['sensor_type']}")
    print(f"  HR bands: {metadata['hr_bands']}")
    print(f"  Band mapping:")
    for k, v in metadata['band_mapping'].items():
        print(f"    {k}: {v}")
    
    # Create dataset - uses full 256×256 HR for all bands
    dataset = FiveBandUniform3mDataset(
        samples, 
        sequence_length=args.num_frames,
        augment=not args.no_augment
    )
    
    # Split into train/val
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    print(f"\n  Train samples: {n_train}")
    print(f"  Val samples: {n_val}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Check shapes
    sample_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  LR 10m: {sample_batch['lr_10m'].shape}")
    print(f"  LR 20m: {sample_batch['lr_20m'].shape}")
    print(f"  HR 10m: {sample_batch['hr_10m'].shape}")
    print(f"  HR 20m: {sample_batch['hr_20m'].shape} ← NOW 256×256 (same as 10m!)")
    
    # Calculate scale factors
    lr_10m_size = sample_batch['lr_10m'].shape[-1]  # 77
    lr_20m_size = sample_batch['lr_20m'].shape[-1]  # 38
    hr_size = sample_batch['hr_10m'].shape[-1]       # 256
    
    scale_factor_10m = hr_size / lr_10m_size  # 256/77 = 3.32
    scale_factor_20m = hr_size / lr_20m_size  # 256/38 = 6.74
    
    print(f"\nScale factors:")
    print(f"  10m branch: {scale_factor_10m:.2f}× ({lr_10m_size}→{hr_size})")
    print(f"  20m branch: {scale_factor_20m:.2f}× ({lr_20m_size}→{hr_size})")
    
    # Create model with adjusted scale factor for 20m branch
    model = MultiResolutionSatelliteSR(
        in_channels_10m=3,
        in_channels_20m=2,
        base_channels=64,
        num_encoder_blocks=4,
        num_frames=args.num_frames,
        scale_factor_10m=scale_factor_10m,  # 3.32× (10m → 3m)
        scale_factor_20m=scale_factor_20m,  # 6.74× (20m → 3m) ← DIFFERENT!
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: MultiResolutionSatelliteSR (Uniform 3m)")
    print(f"  Parameters: {n_params:,}")
    print(f"  Temporal frames: {args.num_frames}")
    print(f"  Scale 10m: {scale_factor_10m:.2f}× (77→256)")
    print(f"  Scale 20m: {scale_factor_20m:.2f}× (38→256)")
    
    # Loss and optimizer
    criterion = MultiResSpectralLoss(
        weight_10m=1.0, weight_20m=1.0,
        alpha_pixel=1.0, alpha_spectral=0.1, alpha_gradient=0.01
    ).to(device)
    print(f"\nLoss: MultiResSpectralLoss")
    print(f"  Components: L1 + 0.1×MSE + 0.1×SAM + 0.01×Gradient")
    print(f"  Branch weights: 10m={criterion.weight_10m}, 20m={criterion.weight_20m}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize convergence tracker
    if not args.no_early_stop:
        convergence = ConvergenceTracker(
            patience_loss=args.patience_loss,
            patience_psnr=args.patience_psnr,
            min_delta_loss=args.min_delta_loss,
            min_delta_psnr=args.min_delta_psnr,
        )
    else:
        convergence = None
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_avg_psnr = float('-inf')
    stopped_early = False
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss, train_loss_10m, train_loss_20m = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        avg_psnr = (val_metrics['psnr_10m'] + val_metrics['psnr_20m']) / 2
        
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} (10m: {train_loss_10m:.4f}, 20m: {train_loss_20m:.4f})")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  PSNR 10m: {val_metrics['psnr_10m']:.2f} dB | PSNR 20m: {val_metrics['psnr_20m']:.2f} dB | Avg: {avg_psnr:.2f} dB")
        print(f"  Per-band PSNR (10m): Blue={val_metrics['band_psnr_10m']['Blue']:.2f}, "
              f"Green={val_metrics['band_psnr_10m']['Green']:.2f}, "
              f"Red={val_metrics['band_psnr_10m']['Red']:.2f}")
        print(f"  Per-band PSNR (20m @ 3m): RedEdge={val_metrics['band_psnr_20m']['RedEdge']:.2f}, "
              f"NIR={val_metrics['band_psnr_20m']['NIR']:.2f}")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_avg_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_psnr_10m': val_metrics['psnr_10m'],
                'val_psnr_20m': val_metrics['psnr_20m'],
                'avg_psnr': avg_psnr,
                'scale_factor_10m': scale_factor_10m,
                'scale_factor_20m': scale_factor_20m,
                'config': {
                    'uniform_3m': True,
                    'output_size': 256,
                    'scale_10m': scale_factor_10m,
                    'scale_20m': scale_factor_20m,
                }
            }, save_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (loss: {best_val_loss:.4f}, PSNR: {best_avg_psnr:.2f} dB)")
        
        # Check convergence
        if convergence is not None:
            conv_result = convergence.update(
                epoch, train_loss, val_metrics['loss'],
                val_metrics['psnr_10m'], val_metrics['psnr_20m']
            )
            
            if (epoch + 1) % 5 == 0 or conv_result['should_stop']:
                print(f"  Convergence: {conv_result['status']['loss']}")
                print(f"              {conv_result['status']['psnr']}")
            
            if conv_result['should_stop']:
                print(f"\n{'='*80}")
                print(f"EARLY STOPPING: {conv_result['stop_reason']}")
                print(f"{'='*80}")
                stopped_early = True
                break
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    if stopped_early:
        print(f"Stopped early at epoch {epoch+1}/{args.epochs}")
    else:
        print(f"Completed all {args.epochs} epochs")
    
    print(f"\nBest Results:")
    print(f"  Validation loss: {best_val_loss:.4f}")
    print(f"  Average PSNR: {best_avg_psnr:.2f} dB")
    print(f"  Model saved to: {save_dir / 'best_model.pt'}")
    
    if convergence is not None:
        print(convergence.get_summary())
    
    print()
    print("Output configuration:")
    print(f"  All 5 bands at 256×256 (3m resolution)")
    print(f"  Bands can be directly stacked for pixel-aligned analysis")


if __name__ == '__main__':
    main()
