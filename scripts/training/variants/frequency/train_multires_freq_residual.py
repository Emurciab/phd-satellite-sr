#!/usr/bin/env python3
"""
Training script for Frequency Domain SR - RESIDUAL VERSION.

FREQUENCY DOMAIN RESIDUAL LEARNING:
- Bilinear upsampling provides low-frequency baseline
- Model learns to add high-frequency details in frequency domain
- More stable than direct learning - starts from reasonable baseline
- Uses specialized frequency loss that emphasizes HF recovery

Key features:
- Temporal attention for multi-frame input
- Cross-resolution fusion between 10m and 20m branches
- Frequency band loss (2x weight on high frequencies)
- VGG perceptual loss for visual quality
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

# Import Frequency Residual model
from satellite_sr_model_freq_residual import FrequencyDomainSRResidual, FrequencyMultiResLoss

# Configuration - same as temporal models for fair comparison
BEST_CONFIG = {
    # Model architecture  
    'base_channels': 64,
    'num_encoder_blocks': 4,
    
    # Optimizer
    'optimizer': 'adamw',
    'lr': 3.064e-05,
    'weight_decay': 5.235e-06,
    
    # Scheduler
    'patience_scheduler': 4,
    
    # Loss weights
    'alpha_pixel': 1.0,
    'alpha_freq': 0.5,           # Frequency domain loss weight
    'alpha_spectral': 0.137,
    'alpha_gradient': 0.01,
    'alpha_perceptual': 0.1,
    
    # Resolution weights
    'weight_10m': 1.0,
    'weight_20m': 1.0,
    
    # Training
    'batch_size': 2,
    'accumulation_steps': 2,  # Effective batch size = 4
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


class TemporalFiveBandMmapDataset(Dataset):
    """
    Dataset that loads temporal sequences for multi-frame SR.
    Returns middle frame HR as primary target.
    """
    def __init__(self, mmap_dir, sequence_length=3, augment=True):
        mmap_dir = Path(mmap_dir)
        self.lr = np.load(mmap_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(mmap_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr_10m = np.load(mmap_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(mmap_dir / 'hr_20m.npy', mmap_mode='r')
        self.n_samples = self.lr.shape[0]
        self.sequence_length = sequence_length
        self.augment = augment
        print(f"Temporal Dataset: {self.n_samples} samples, T={sequence_length} frames")
        print(f"  LR 10m: {self.lr.shape}, LR 20m: {self.lr_20m.shape}")
        print(f"  HR 10m: {self.hr_10m.shape}, HR 20m: {self.hr_20m.shape}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        lr_10m = self.lr[idx].copy()
        lr_20m = self.lr_20m[idx].copy()
        hr_10m = self.hr_10m[idx].copy()
        hr_20m = self.hr_20m[idx].copy()
        
        if self.augment:
            import random
            # Apply SAME augmentation to all frames
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
        
        # Convert to tensors: (T, C, H, W)
        lr_10m = torch.from_numpy(lr_10m).permute(0, 3, 1, 2).float()
        lr_20m = torch.from_numpy(lr_20m).permute(0, 3, 1, 2).float()
        hr_10m = torch.from_numpy(hr_10m).permute(0, 3, 1, 2).float()
        hr_20m = torch.from_numpy(hr_20m).permute(0, 3, 1, 2).float()
        
        return {'lr_10m': lr_10m, 'lr_20m': lr_20m, 'hr_10m': hr_10m, 'hr_20m': hr_20m}


def compute_psnr(pred, target):
    """Compute PSNR in dB."""
    mse = torch.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * torch.log10(1.0 / mse).item()


def train_epoch(model, loader, criterion, optimizer, device, cfg):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    accumulation_steps = cfg.get('accumulation_steps', 1)
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        lr_10m = batch['lr_10m'].to(device)
        lr_20m = batch['lr_20m'].to(device)
        hr_10m = batch['hr_10m'].to(device)
        hr_20m = batch['hr_20m'].to(device)
        
        # Get middle frame targets
        T = hr_10m.shape[1]
        mid_idx = T // 2
        target_10m = hr_10m[:, mid_idx]
        target_20m = hr_20m[:, mid_idx]
        
        # Get target sizes from HR
        target_size_10m = (target_10m.shape[2], target_10m.shape[3])
        target_size_20m = (target_20m.shape[2], target_20m.shape[3])
        
        # Forward pass
        sr_10m, sr_20m = model(lr_10m, lr_20m, target_size_10m, target_size_20m)
        
        # Compute loss
        loss, loss_dict = criterion(sr_10m, sr_20m, target_10m, target_20m)
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    # Handle remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and compute PSNR metrics."""
    model.eval()
    total_loss = 0
    psnr_10m_sum = 0
    psnr_20m_sum = 0
    n_batches = 0
    
    for batch in loader:
        lr_10m = batch['lr_10m'].to(device)
        lr_20m = batch['lr_20m'].to(device)
        hr_10m = batch['hr_10m'].to(device)
        hr_20m = batch['hr_20m'].to(device)
        
        # Get middle frame targets
        T = hr_10m.shape[1]
        mid_idx = T // 2
        target_10m = hr_10m[:, mid_idx]
        target_20m = hr_20m[:, mid_idx]
        
        target_size_10m = (target_10m.shape[2], target_10m.shape[3])
        target_size_20m = (target_20m.shape[2], target_20m.shape[3])
        
        # Forward pass
        sr_10m, sr_20m = model(lr_10m, lr_20m, target_size_10m, target_size_20m)
        
        # Loss
        loss, _ = criterion(sr_10m, sr_20m, target_10m, target_20m)
        total_loss += loss.item()
        
        # PSNR
        psnr_10m_sum += compute_psnr(sr_10m, target_10m)
        psnr_20m_sum += compute_psnr(sr_20m, target_20m)
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    avg_psnr_10m = psnr_10m_sum / n_batches
    avg_psnr_20m = psnr_20m_sum / n_batches
    avg_psnr = (avg_psnr_10m + avg_psnr_20m) / 2
    
    return avg_loss, avg_psnr_10m, avg_psnr_20m, avg_psnr


def train_model(model, train_loader, val_loader, cfg, device, checkpoint_dir):
    """Main training loop."""
    # Loss function
    criterion = FrequencyMultiResLoss(cfg).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, 
        patience=cfg['patience_scheduler'], verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=cfg['patience'])
    
    # Training history
    history = []
    best_psnr = 0
    
    print(f"\nStarting training for {cfg['epochs']} epochs...")
    
    for epoch in range(1, cfg['epochs'] + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, cfg)
        
        # Validate
        val_loss, psnr_10m, psnr_20m, avg_psnr = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time
        
        # Log
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PSNR: 10m={psnr_10m:.2f} 20m={psnr_20m:.2f} avg={avg_psnr:.2f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'psnr_10m': psnr_10m,
            'psnr_20m': psnr_20m,
            'avg_psnr': avg_psnr,
            'lr': current_lr
        })
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'psnr_10m': psnr_10m,
                'psnr_20m': psnr_20m,
                'avg_psnr': avg_psnr,
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
        
        # Save history
        with open(checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Early stopping
        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best epoch was {early_stopping.best_epoch} with val_loss={early_stopping.best_value:.4f}")
            break
    
    return val_loss, best_psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['full', 'filtered', 'test'],
                        default='full', help='Dataset to use')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent.parent  # Go up to SR/
    
    if args.dataset == 'full':
        data_dir = base_dir / 'data/sr_training/5band_v3/combined/mmap'
        checkpoint_dir = base_dir / 'checkpoints/multires_freq_residual_full'
    elif args.dataset == 'filtered':
        data_dir = base_dir / 'data/sr_training/5band_v3/combined/mmap_filtered'
        checkpoint_dir = base_dir / 'checkpoints/multires_freq_residual_filtered'
    else:
        data_dir = base_dir / 'data/sr_training/5band_v3/combined/mmap_test'
        checkpoint_dir = base_dir / 'checkpoints/multires_freq_residual_test'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = BEST_CONFIG.copy()
    
    print("=" * 70)
    print("Training: Frequency Domain SR - RESIDUAL VERSION")
    print("  -> Bilinear baseline + learned HF residuals")
    print(f"Data: {data_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset
    dataset = TemporalFiveBandMmapDataset(data_dir, sequence_length=3, augment=True)
    
    # Split
    n_val = min(100, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.augment = False
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Model
    model = FrequencyDomainSRResidual(
        in_channels_10m=3,
        in_channels_20m=2,
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=3,
        scale_factor_10m=260/78,  # ~3.33
        scale_factor_20m=130/39   # ~3.33
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['base_channels']} channels, {cfg['num_encoder_blocks']} blocks, {total_params:,} params")
    print(f"Batch size: {cfg['batch_size']}, Accumulation steps: {cfg['accumulation_steps']}")
    print(f"Effective batch size: {cfg['batch_size'] * cfg['accumulation_steps']}")
    
    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nLoss weights: Pixel={cfg['alpha_pixel']:.3f}, Freq={cfg['alpha_freq']:.3f}, "
          f"SAM={cfg['alpha_spectral']:.4f}, Grad={cfg['alpha_gradient']:.4f}, Perc={cfg['alpha_perceptual']:.3f}")
    print(f"Branch weights: 10m={cfg['weight_10m']:.3f}, 20m={cfg['weight_20m']:.3f}")
    
    # Train
    val_loss, best_psnr = train_model(
        model, train_loader, val_loader, cfg, device, checkpoint_dir
    )
    
    print(f"\nTraining complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
