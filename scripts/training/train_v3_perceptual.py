#!/usr/bin/env python3
"""
Comprehensive training script for V3 model with perceptual loss.

Trains on both full and filtered (high-correlation) datasets using 
the best hyperparameters from Optuna tuning + new perceptual loss.

Best config from Trial #12:
- Optimizer: AdamW, LR: 3.06e-05, weight_decay: 5.23e-06
- Scheduler: ReduceLROnPlateau, patience: 4
- Loss: L1 + MSE + SAM + Gradient + Perceptual (NEW)
- Model: 48 base channels, 3 encoder blocks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import time
from pathlib import Path
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from models.satellite_sr_model_v3 import MultiResolutionSatelliteSRv3

# Best hyperparameters from Optuna Trial #12
BEST_CONFIG = {
    # Model architecture
    'base_channels': 48,
    'num_encoder_blocks': 3,
    
    # Optimizer
    'optimizer': 'adamw',
    'lr': 3.064e-05,
    'weight_decay': 5.235e-06,
    
    # Scheduler
    'scheduler': 'plateau',
    'patience_scheduler': 4,
    
    # Loss weights (from tuning)
    'alpha_l1': 0.538,
    'alpha_mse': 0.413,
    'alpha_sam': 0.137,
    'alpha_gradient': 0.002,
    'alpha_perceptual': 0.1,  # NEW: VGG perceptual loss
    
    # Resolution weights
    'weight_10m': 0.774,
    'weight_20m': 0.520,
    
    # Training
    'batch_size': 4,
    'epochs': 200,
    'patience': 20,
}


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
    
    def __call__(self, value, epoch):
        if value < (self.best_value - self.min_delta):
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


class CombinedLossWithPerceptual(nn.Module):
    """
    Combined loss with L1, MSE, SAM, gradient, and VGG perceptual terms.
    Uses best hyperparameters from Optuna tuning.
    """
    def __init__(self, cfg):
        super().__init__()
        self.alpha_l1 = cfg['alpha_l1']
        self.alpha_mse = cfg['alpha_mse']
        self.alpha_sam = cfg['alpha_sam']
        self.alpha_gradient = cfg['alpha_gradient']
        self.alpha_perceptual = cfg['alpha_perceptual']
        self.weight_10m = cfg['weight_10m']
        self.weight_20m = cfg['weight_20m']
        
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
        # VGG Perceptual loss
        from losses.perceptual_loss import VGGPerceptualLoss
        self.perceptual_loss = VGGPerceptualLoss()
        
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m):
        # 10m branch: L1 + MSE + SAM + gradient + perceptual
        loss_10m = (
            self.alpha_l1 * self.l1(pred_10m, target_10m) +
            self.alpha_mse * self.mse(pred_10m, target_10m) +
            self.alpha_sam * self._sam_loss(pred_10m, target_10m) +
            self.alpha_gradient * self._grad_loss(pred_10m, target_10m) +
            self.alpha_perceptual * self.perceptual_loss(pred_10m, target_10m)
        )
        
        # 20m branch: L1 + MSE + SAM + gradient (no perceptual - only 2 channels)
        loss_20m = (
            self.alpha_l1 * self.l1(pred_20m, target_20m) +
            self.alpha_mse * self.mse(pred_20m, target_20m) +
            self.alpha_sam * self._sam_loss(pred_20m, target_20m) +
            self.alpha_gradient * self._grad_loss(pred_20m, target_20m)
        )
        
        total = self.weight_10m * loss_10m + self.weight_20m * loss_20m
        return total, loss_10m, loss_20m
    
    def _sam_loss(self, pred, target):
        pred_norm = nn.functional.normalize(pred, dim=1, eps=1e-8)
        target_norm = nn.functional.normalize(target, dim=1, eps=1e-8)
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-8, 1.0 - 1e-8)
        return 1.0 - cos_sim.mean()
    
    def _grad_loss(self, pred, target):
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
    psnr_10m_sum, psnr_20m_sum = 0, 0
    
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
            psnr_10m_sum += calculate_psnr(pred_10m, hr_10m)
            psnr_20m_sum += calculate_psnr(pred_20m, hr_20m)
    
    n = len(loader)
    return total_loss / n, psnr_10m_sum / n, psnr_20m_sum / n


def train_model(data_dir, checkpoint_dir, experiment_name, cfg=BEST_CONFIG):
    """Train a single model configuration."""
    
    print(f"\n{'='*60}")
    print(f"Training: {experiment_name}")
    print(f"Data: {data_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = FiveBandMmapDataset(data_dir, augment=True)
    
    # Split: 90% train, 10% val
    n_val = max(int(len(dataset) * 0.1), 10)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.augment = False
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = MultiResolutionSatelliteSRv3(
        in_channels_10m=3,
        in_channels_20m=2,
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=3
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['base_channels']} channels, {cfg['num_encoder_blocks']} blocks, {n_params:,} params")
    
    # Loss function with perceptual
    criterion = CombinedLossWithPerceptual(cfg).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=cfg['patience_scheduler'], verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=cfg['patience'])
    
    # Training loop
    best_val_loss = float('inf')
    best_psnr = 0
    history = []
    
    print(f"\nLoss weights: L1={cfg['alpha_l1']:.3f}, MSE={cfg['alpha_mse']:.3f}, "
          f"SAM={cfg['alpha_sam']:.3f}, Grad={cfg['alpha_gradient']:.4f}, Perc={cfg['alpha_perceptual']:.3f}")
    print(f"Branch weights: 10m={cfg['weight_10m']:.3f}, 20m={cfg['weight_20m']:.3f}")
    print(f"\nStarting training for {cfg['epochs']} epochs...")
    
    for epoch in range(1, cfg['epochs'] + 1):
        start = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, psnr_10m, psnr_20m = validate(model, val_loader, criterion, device)
        avg_psnr = (psnr_10m + psnr_20m) / 2
        
        scheduler.step(val_loss)
        
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]['lr']
        
        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'psnr_10m': psnr_10m, 'psnr_20m': psnr_20m, 'avg_psnr': avg_psnr, 'lr': lr
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'psnr_10m': psnr_10m,
                'psnr_20m': psnr_20m,
                'config': cfg
            }, checkpoint_dir / 'best_model.pt')
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': cfg
        }, checkpoint_dir / 'latest_model.pt')
        
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PSNR: 10m={psnr_10m:.2f} 20m={psnr_20m:.2f} avg={avg_psnr:.2f} | "
              f"LR: {lr:.2e} | Time: {elapsed:.1f}s")
        
        # Early stopping
        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping at epoch {epoch}. Best: epoch {early_stopping.best_epoch}")
            break
    
    # Save history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete: {experiment_name}")
    print(f"Best val loss: {best_val_loss:.4f}, Best PSNR: {best_psnr:.2f} dB")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"{'='*60}\n")
    
    return best_val_loss, best_psnr


def main():
    parser = argparse.ArgumentParser(description='Train V3 with perceptual loss')
    parser.add_argument('--dataset', choices=['full', 'filtered', 'both'], default='both',
                        help='Which dataset(s) to train on')
    parser.add_argument('--epochs', type=int, default=200, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--alpha_perceptual', type=float, default=0.1, help='Perceptual loss weight')
    args = parser.parse_args()
    
    # Update config with CLI args
    cfg = BEST_CONFIG.copy()
    cfg['epochs'] = args.epochs
    cfg['batch_size'] = args.batch_size
    cfg['alpha_perceptual'] = args.alpha_perceptual
    
    # Dataset paths
    FULL_DATA = Path('data/sr_training/5band_v3/combined/mmap')
    FILTERED_DATA = Path('data/sr_training/5band_v3/combined/mmap_filtered')
    
    # Checkpoint base
    CHECKPOINT_BASE = Path('checkpoints')
    
    results = {}
    
    if args.dataset in ['full', 'both']:
        # Train on full dataset
        val_loss, best_psnr = train_model(
            data_dir=FULL_DATA,
            checkpoint_dir=CHECKPOINT_BASE / '5band_v3_perceptual_full',
            experiment_name='V3 + Perceptual (Full Dataset)',
            cfg=cfg
        )
        results['full'] = {'val_loss': val_loss, 'psnr': best_psnr}
    
    if args.dataset in ['filtered', 'both']:
        # Train on filtered dataset
        val_loss, best_psnr = train_model(
            data_dir=FILTERED_DATA,
            checkpoint_dir=CHECKPOINT_BASE / '5band_v3_perceptual_filtered',
            experiment_name='V3 + Perceptual (Filtered Dataset)',
            cfg=cfg
        )
        results['filtered'] = {'val_loss': val_loss, 'psnr': best_psnr}
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for name, res in results.items():
        print(f"{name:10s}: val_loss={res['val_loss']:.4f}, PSNR={res['psnr']:.2f} dB")
    print("="*60)


if __name__ == '__main__':
    main()
