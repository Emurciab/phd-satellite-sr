#!/usr/bin/env python3
"""
Training script for 5-band multi-resolution super-resolution V2.

V2 CHANGES:
1. Uses MultiResolutionSatelliteSRv2 with residual learning
2. Removes sigmoid constraint (was causing mode collapse)
3. Output = bilinear_upscale(LR) + learned_residual

This script trains a model that:
1. Super-resolves 10m bands (3 channels) from 78×78 → 260×260 (~3.33×)
2. Super-resolves 20m bands (2 channels) from 39×39 → 130×130 (~3.33×)

Both branches use REAL HR targets from Planet SuperDove.
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

# Import V2 model and loss
from models.satellite_sr_model_v2 import MultiResolutionSatelliteSRv2, MultiResSpectralLoss, create_model_v2


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping to halt training when validation metrics stop improving."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = 'min', metric_name: str = 'val_loss'):
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
    
    def __init__(self, patience_loss: int = 15, patience_psnr: int = 20,
                 min_delta_loss: float = 1e-4, min_delta_psnr: float = 0.1,
                 plateau_window: int = 10, plateau_threshold: float = 0.05):
        self.es_loss = EarlyStopping(patience_loss, min_delta_loss, 'min', 'val_loss')
        self.es_psnr = EarlyStopping(patience_psnr, min_delta_psnr, 'max', 'avg_psnr')
        
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        
        self.metrics_history = {
            'train_loss': [], 'val_loss': [],
            'psnr_10m': [], 'psnr_20m': [], 'avg_psnr': [],
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
            stop_reason = f"Metrics plateaued"
        
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
            f"  Total epochs: {len(self.metrics_history['val_loss'])}"
        )


# =============================================================================
# DATASET
# =============================================================================

class FiveBandMmapDataset(Dataset):
    """Fast memory-mapped dataset for 5-band SuperDove data."""
    
    def __init__(self, mmap_dir: str, sequence_length: int = 3, augment: bool = True):
        from pathlib import Path
        mmap_dir = Path(mmap_dir)
        
        # Load memory-mapped arrays
        self.lr = np.load(mmap_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(mmap_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr = np.load(mmap_dir / 'hr.npy', mmap_mode='r')
        self.hr_10m = np.load(mmap_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(mmap_dir / 'hr_20m.npy', mmap_mode='r')
        
        self.n_samples = self.lr.shape[0]
        self.sequence_length = sequence_length
        self.augment = augment
        
        print(f"FiveBandMmapDataset: {self.n_samples} samples from {mmap_dir}")
        print(f"  lr: {self.lr.shape}, hr_10m: {self.hr_10m.shape}, hr_20m: {self.hr_20m.shape}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        lr_10m = self.lr[idx].copy()
        lr_20m = self.lr_20m[idx].copy()
        hr_10m = self.hr_10m[idx].copy()
        hr_20m = self.hr_20m[idx].copy()
        
        if self.augment:
            lr_10m, lr_20m, hr_10m, hr_20m = self._augment(lr_10m, lr_20m, hr_10m, hr_20m)
        
        lr_10m = torch.from_numpy(lr_10m).permute(0, 3, 1, 2).float()
        lr_20m = torch.from_numpy(lr_20m).permute(0, 3, 1, 2).float()
        
        mid_idx = self.sequence_length // 2
        hr_10m = torch.from_numpy(hr_10m[mid_idx]).permute(2, 0, 1).float()
        hr_20m = torch.from_numpy(hr_20m[mid_idx]).permute(2, 0, 1).float()
        
        return {'lr_10m': lr_10m, 'lr_20m': lr_20m, 'hr_10m': hr_10m, 'hr_20m': hr_20m}
    
    def _augment(self, lr_10m, lr_20m, hr_10m, hr_20m):
        import random
        
        if random.random() > 0.5:
            lr_10m = np.flip(lr_10m, axis=2).copy()
            lr_20m = np.flip(lr_20m, axis=2).copy()
            hr_10m = np.flip(hr_10m, axis=2).copy()
            hr_20m = np.flip(hr_20m, axis=2).copy()
        
        if random.random() > 0.5:
            lr_10m = np.flip(lr_10m, axis=1).copy()
            lr_20m = np.flip(lr_20m, axis=1).copy()
            hr_10m = np.flip(hr_10m, axis=1).copy()
            hr_20m = np.flip(hr_20m, axis=1).copy()
        
        k = random.randint(0, 3)
        if k > 0:
            lr_10m = np.rot90(lr_10m, k, axes=(1, 2)).copy()
            lr_20m = np.rot90(lr_20m, k, axes=(1, 2)).copy()
            hr_10m = np.rot90(hr_10m, k, axes=(1, 2)).copy()
            hr_20m = np.rot90(hr_20m, k, axes=(1, 2)).copy()
        
        return lr_10m, lr_20m, hr_10m, hr_20m


# =============================================================================
# LOSS FUNCTION (compatible with V1 but also used for V2)
# =============================================================================

class CombinedLoss(nn.Module):
    """Combined loss for multi-resolution SR."""
    
    def __init__(self, weight_10m=1.0, weight_20m=1.0, alpha_pixel=1.0, 
                 alpha_spectral=0.1, alpha_gradient=0.01):
        super().__init__()
        self.weight_10m = weight_10m
        self.weight_20m = weight_20m
        self.alpha_pixel = alpha_pixel
        self.alpha_spectral = alpha_spectral
        self.alpha_gradient = alpha_gradient
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def spectral_angle_loss(self, pred, target):
        pred_norm = nn.functional.normalize(pred, dim=1, eps=1e-8)
        target_norm = nn.functional.normalize(target, dim=1, eps=1e-8)
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        return 1.0 - cos_sim.mean()
    
    def gradient_loss(self, pred, target):
        return self.l1_loss(self._compute_edges(pred), self._compute_edges(target))
    
    def _compute_edges(self, x):
        B, C, H, W = x.shape
        edges = []
        for c in range(C):
            channel = x[:, c:c+1, :, :]
            edge_x = nn.functional.conv2d(channel, self.sobel_x, padding=1)
            edge_y = nn.functional.conv2d(channel, self.sobel_y, padding=1)
            edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
            edges.append(edge_mag)
        return torch.cat(edges, dim=1)
    
    def _compute_branch_loss(self, pred, target):
        pixel_loss = self.l1_loss(pred, target) + 0.1 * self.mse_loss(pred, target)
        sam_loss = self.spectral_angle_loss(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        return self.alpha_pixel * pixel_loss + self.alpha_spectral * sam_loss + self.alpha_gradient * grad_loss
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m):
        loss_10m = self._compute_branch_loss(pred_10m, target_10m)
        loss_20m = self._compute_branch_loss(pred_20m, target_20m)
        total_loss = self.weight_10m * loss_10m + self.weight_20m * loss_20m
        return total_loss, loss_10m, loss_20m


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)).item()


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
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
        
        pred_10m, pred_20m = model(
            lr_10m, lr_20m,
            target_size_10m=(hr_10m.shape[2], hr_10m.shape[3]),
            target_size_20m=(hr_20m.shape[2], hr_20m.shape[3])
        )
        
        loss, loss_10m, loss_20m = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
            
            # PSNR
            total_psnr_10m += calculate_psnr(pred_10m, hr_10m)
            total_psnr_20m += calculate_psnr(pred_20m, hr_20m)
    
    n_batches = len(dataloader)
    return (
        total_loss / n_batches,
        total_psnr_10m / n_batches,
        total_psnr_20m / n_batches
    )


def main():
    parser = argparse.ArgumentParser(description='Train 5-band SR V2 (Residual Learning)')
    parser.add_argument('--data_dir', type=str, 
                        default='data/sr_training/5band_v3/combined/mmap',
                        help='Directory with mmap training data')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/5band_v2_residual',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--num_encoder_blocks', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training 5-Band SR V2 (Residual Learning)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading data from: {args.data_dir}")
    dataset = FiveBandMmapDataset(args.data_dir, sequence_length=3, augment=True)
    
    # Split train/val
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Disable augmentation for validation
    val_dataset.dataset.augment = False
    
    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # Create V2 model
    print(f"\nCreating V2 model...")
    model = create_model_v2(
        base_channels=args.base_channels,
        num_encoder_blocks=args.num_encoder_blocks,
        num_frames=3,
        in_channels_10m=3,
        in_channels_20m=2
    )
    model = model.to(device)
    
    # Loss function
    criterion = CombinedLoss().to(device)
    
    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Convergence tracker
    tracker = ConvergenceTracker(
        patience_loss=args.patience,
        patience_psnr=args.patience + 5,
        min_delta_loss=1e-4,
        min_delta_psnr=0.1
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed at epoch {start_epoch}")
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    
    best_psnr = 0
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_loss_10m, train_loss_20m = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_psnr_10m, val_psnr_20m = validate(
            model, val_loader, criterion, device
        )
        avg_psnr = (val_psnr_10m + val_psnr_20m) / 2
        
        # Update scheduler
        scheduler.step()
        
        # Update tracker
        convergence = tracker.update(epoch, train_loss, val_loss, val_psnr_10m, val_psnr_20m)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PSNR: 10m={val_psnr_10m:.2f} 20m={val_psnr_20m:.2f} avg={avg_psnr:.2f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model (by PSNR)
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_psnr_10m': val_psnr_10m,
                'val_psnr_20m': val_psnr_20m,
                'avg_psnr': avg_psnr,
                'model_version': 'v2_residual',
            }, checkpoint_dir / 'best_model.pt')
            print(f"  ✓ New best PSNR: {avg_psnr:.2f} dB - saved!")
        
        # Save best by loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_psnr_10m': val_psnr_10m,
                'val_psnr_20m': val_psnr_20m,
                'avg_psnr': avg_psnr,
                'model_version': 'v2_residual',
            }, checkpoint_dir / 'best_loss_model.pt')
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'avg_psnr': avg_psnr,
                'model_version': 'v2_residual',
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        # Check early stopping
        if convergence['should_stop']:
            print(f"\n{'='*60}")
            print(f"Early stopping: {convergence['stop_reason']}")
            print(tracker.get_summary())
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(tracker.get_summary())


if __name__ == '__main__':
    main()
