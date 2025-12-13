#!/usr/bin/env python3
"""
Training script for 5-band multi-resolution super-resolution V3 with TUNED hyperparameters.

Based on Optuna hyperparameter tuning results (Trial #12):
- base_channels: 48
- num_encoder_blocks: 3
- optimizer: AdamW
- lr: 3.06e-05
- scheduler: ReduceLROnPlateau
- Loss weights: L1=0.54, MSE=0.41, SAM=0.14, Grad=0.00
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from models.satellite_sr_model_v3 import MultiResolutionSatelliteSRv3


# =============================================================================
# TUNED HYPERPARAMETERS (from Optuna Trial #12)
# =============================================================================
TUNED_CONFIG = {
    # Model architecture
    'base_channels': 48,
    'num_encoder_blocks': 3,
    
    # Optimizer
    'optimizer': 'adamw',
    'lr': 3.064129636469023e-05,
    'weight_decay': 5.234919184787775e-06,
    
    # Scheduler
    'scheduler': 'plateau',
    'patience_scheduler': 4,
    
    # Loss weights
    'alpha_l1': 0.5383242422158145,
    'alpha_mse': 0.41304820332609976,
    'alpha_spectral': 0.1373445277283548,
    'alpha_gradient': 0.0023267899467518133,
    
    # Resolution weights
    'weight_10m': 0.7739601935394229,
    'weight_20m': 0.5200223358333815,
    
    # Training
    'batch_size': 2,
}


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, mode='min', metric_name='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric_name = metric_name
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
    
    def __call__(self, value, epoch):
        improved = value < (self.best_value - self.min_delta) if self.mode == 'min' else value > (self.best_value + self.min_delta)
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class FiveBandMmapDataset(Dataset):
    def __init__(self, mmap_dir, sequence_length=3, augment=True):
        mmap_dir = Path(mmap_dir)
        self.lr = np.load(mmap_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(mmap_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr_10m = np.load(mmap_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(mmap_dir / 'hr_20m.npy', mmap_mode='r')
        self.n_samples = self.lr.shape[0]
        self.sequence_length = sequence_length
        self.augment = augment
        print(f"Dataset: {self.n_samples} samples")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        lr_10m = self.lr[idx].copy()
        lr_20m = self.lr_20m[idx].copy()
        hr_10m = self.hr_10m[idx].copy()
        hr_20m = self.hr_20m[idx].copy()
        
        if self.augment:
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
        
        lr_10m = torch.from_numpy(lr_10m).permute(0, 3, 1, 2).float()
        lr_20m = torch.from_numpy(lr_20m).permute(0, 3, 1, 2).float()
        mid_idx = self.sequence_length // 2
        hr_10m = torch.from_numpy(hr_10m[mid_idx]).permute(2, 0, 1).float()
        hr_20m = torch.from_numpy(hr_20m[mid_idx]).permute(2, 0, 1).float()
        
        return {'lr_10m': lr_10m, 'lr_20m': lr_20m, 'hr_10m': hr_10m, 'hr_20m': hr_20m}


class TunedLoss(nn.Module):
    """Loss function with tuned weights from hyperparameter optimization."""
    
    def __init__(self, 
                 alpha_l1=0.54, 
                 alpha_mse=0.41, 
                 alpha_spectral=0.14, 
                 alpha_gradient=0.00,
                 weight_10m=0.77,
                 weight_20m=0.52):
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_mse = alpha_mse
        self.alpha_spectral = alpha_spectral
        self.alpha_gradient = alpha_gradient
        self.weight_10m = weight_10m
        self.weight_20m = weight_20m
        
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
        # Sobel filters for gradient loss
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m):
        # 10m band loss
        loss_10m = (
            self.alpha_l1 * self.l1(pred_10m, target_10m) +
            self.alpha_mse * self.mse(pred_10m, target_10m) +
            self.alpha_spectral * self._spectral_loss(pred_10m, target_10m) +
            self.alpha_gradient * self._gradient_loss(pred_10m, target_10m)
        )
        
        # 20m band loss
        loss_20m = (
            self.alpha_l1 * self.l1(pred_20m, target_20m) +
            self.alpha_mse * self.mse(pred_20m, target_20m) +
            self.alpha_spectral * self._spectral_loss(pred_20m, target_20m) +
            self.alpha_gradient * self._gradient_loss(pred_20m, target_20m)
        )
        
        # Weighted combination
        total_loss = self.weight_10m * loss_10m + self.weight_20m * loss_20m
        
        return total_loss, loss_10m, loss_20m
    
    def _spectral_loss(self, pred, target):
        """Spectral Angle Mapper (SAM) loss."""
        pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
        target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        cos_sim = torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)
        sam = torch.acos(cos_sim)
        return sam.mean()
    
    def _gradient_loss(self, pred, target):
        """Gradient loss using Sobel filters."""
        loss = 0
        for c in range(pred.shape[1]):
            pred_gx = nn.functional.conv2d(pred[:, c:c+1], self.sobel_x, padding=1)
            pred_gy = nn.functional.conv2d(pred[:, c:c+1], self.sobel_y, padding=1)
            target_gx = nn.functional.conv2d(target[:, c:c+1], self.sobel_x, padding=1)
            target_gy = nn.functional.conv2d(target[:, c:c+1], self.sobel_y, padding=1)
            loss += self.l1(pred_gx, target_gx) + self.l1(pred_gy, target_gy)
        return loss / pred.shape[1]


def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)).item()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        lr_10m = batch['lr_10m'].to(device)
        lr_20m = batch['lr_20m'].to(device)
        hr_10m = batch['hr_10m'].to(device)
        hr_20m = batch['hr_20m'].to(device)
        
        optimizer.zero_grad()
        pred_10m, pred_20m = model(lr_10m, lr_20m, 
                                    target_size_10m=(hr_10m.shape[2], hr_10m.shape[3]),
                                    target_size_20m=(hr_20m.shape[2], hr_20m.shape[3]))
        loss, _, _ = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr_10m = 0
    total_psnr_20m = 0
    
    with torch.no_grad():
        for batch in loader:
            lr_10m = batch['lr_10m'].to(device)
            lr_20m = batch['lr_20m'].to(device)
            hr_10m = batch['hr_10m'].to(device)
            hr_20m = batch['hr_20m'].to(device)
            
            pred_10m, pred_20m = model(lr_10m, lr_20m,
                                        target_size_10m=(hr_10m.shape[2], hr_10m.shape[3]),
                                        target_size_20m=(hr_20m.shape[2], hr_20m.shape[3]))
            loss, _, _ = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
            total_loss += loss.item()
            total_psnr_10m += calculate_psnr(pred_10m, hr_10m)
            total_psnr_20m += calculate_psnr(pred_20m, hr_20m)
    
    n = len(loader)
    return total_loss / n, total_psnr_10m / n, total_psnr_20m / n


def main():
    parser = argparse.ArgumentParser(description='Train 5-band SR V3 with TUNED hyperparameters')
    parser.add_argument('--data_dir', type=str, default='data/sr_training/5band_v3/combined/mmap')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/5band_v3_tuned')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()
    
    cfg = TUNED_CONFIG
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training 5-Band SR V3 with TUNED Hyperparameters")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nTuned Configuration (from Optuna Trial #12):")
    print(f"  Model: base_channels={cfg['base_channels']}, num_encoder_blocks={cfg['num_encoder_blocks']}")
    print(f"  Optimizer: {cfg['optimizer']}, lr={cfg['lr']:.2e}, weight_decay={cfg['weight_decay']:.2e}")
    print(f"  Scheduler: {cfg['scheduler']} (patience={cfg['patience_scheduler']})")
    print(f"  Loss weights: L1={cfg['alpha_l1']:.2f}, MSE={cfg['alpha_mse']:.2f}, "
          f"SAM={cfg['alpha_spectral']:.2f}, Grad={cfg['alpha_gradient']:.3f}")
    print(f"  Resolution weights: 10m={cfg['weight_10m']:.2f}, 20m={cfg['weight_20m']:.2f}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"{'='*70}")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    dataset = FiveBandMmapDataset(args.data_dir, sequence_length=3, augment=True)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\nData: Train={n_train}, Val={n_val}")
    
    # Model with tuned architecture
    scale_10m = 260 / 78  # ~3.33x
    scale_20m = 130 / 39  # ~3.33x
    
    model = MultiResolutionSatelliteSRv3(
        in_channels_10m=3,
        in_channels_20m=2,
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=3,
        scale_factor_10m=scale_10m,
        scale_factor_20m=scale_20m,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Loss with tuned weights
    criterion = TunedLoss(
        alpha_l1=cfg['alpha_l1'],
        alpha_mse=cfg['alpha_mse'],
        alpha_spectral=cfg['alpha_spectral'],
        alpha_gradient=cfg['alpha_gradient'],
        weight_10m=cfg['weight_10m'],
        weight_20m=cfg['weight_20m'],
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    # Scheduler: ReduceLROnPlateau as per tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=cfg['patience_scheduler'], factor=0.5, verbose=True
    )
    
    es = EarlyStopping(patience=args.patience, mode='max', metric_name='psnr')
    
    best_psnr = 0
    
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, psnr_10m, psnr_20m = validate(model, val_loader, criterion, device)
        avg_psnr = (psnr_10m + psnr_20m) / 2
        
        # ReduceLROnPlateau uses validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PSNR: 10m={psnr_10m:.2f} 20m={psnr_20m:.2f} avg={avg_psnr:.2f} | "
              f"LR: {current_lr:.2e} | Time: {time.time()-t0:.1f}s")
        
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_psnr_10m': psnr_10m,
                'val_psnr_20m': psnr_20m,
                'avg_psnr': avg_psnr,
                'model_version': 'v3_tuned',
                'config': cfg,
            }, checkpoint_dir / 'best_model.pt')
            print(f"  ✓ New best PSNR: {avg_psnr:.2f} dB")
        
        if es(avg_psnr, epoch):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoint: {checkpoint_dir / 'best_model.pt'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
