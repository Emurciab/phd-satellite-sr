#!/usr/bin/env python3
"""
Training script for 5-band multi-resolution super-resolution V3.

V3: Simplified direct prediction (no sigmoid, no residual learning)
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

from models.satellite_sr_model_v3 import MultiResolutionSatelliteSRv3, create_model_v3


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


class CombinedLoss(nn.Module):
    """
    Combined loss with pixel, SAM, gradient, and VGG perceptual terms.
    
    Args:
        alpha_pixel: Weight for L1 pixel loss (default: 1.0)
        alpha_sam: Weight for SAM spectral loss (default: 0.1)
        alpha_gradient: Weight for gradient/edge loss (default: 0.01)
        alpha_perceptual: Weight for VGG perceptual loss (default: 0.1)
        use_perceptual: Whether to use VGG perceptual loss (default: True)
    """
    def __init__(self, alpha_pixel=1.0, alpha_sam=0.1, alpha_gradient=0.01, 
                 alpha_perceptual=0.1, use_perceptual=True):
        super().__init__()
        self.alpha_pixel = alpha_pixel
        self.alpha_sam = alpha_sam
        self.alpha_gradient = alpha_gradient
        self.alpha_perceptual = alpha_perceptual
        self.use_perceptual = use_perceptual
        
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
        # VGG Perceptual loss (only for 10m branch which has 3 channels)
        if use_perceptual:
            from losses.perceptual_loss import VGGPerceptualLoss
            self.perceptual_loss = VGGPerceptualLoss()
        
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m):
        # 10m branch: L1 + SAM + gradient + perceptual
        l1_10m = self.l1(pred_10m, target_10m)
        sam_10m = self._sam_loss(pred_10m, target_10m)
        grad_10m = self._grad_loss(pred_10m, target_10m)
        
        loss_10m = self.alpha_pixel * l1_10m + self.alpha_sam * sam_10m + self.alpha_gradient * grad_10m
        
        # Add perceptual loss for 10m (3 channels, good for VGG)
        if self.use_perceptual and self.alpha_perceptual > 0:
            perc_10m = self.perceptual_loss(pred_10m, target_10m)
            loss_10m = loss_10m + self.alpha_perceptual * perc_10m
        
        # 20m branch: L1 + SAM + gradient (no perceptual - only 2 channels)
        l1_20m = self.l1(pred_20m, target_20m)
        sam_20m = self._sam_loss(pred_20m, target_20m)
        grad_20m = self._grad_loss(pred_20m, target_20m)
        
        loss_20m = self.alpha_pixel * l1_20m + self.alpha_sam * sam_20m + self.alpha_gradient * grad_20m
        
        return loss_10m + loss_20m, loss_10m, loss_20m
    
    def _sam_loss(self, pred, target):
        """Spectral Angle Mapper (SAM) loss - numerically stable."""
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
    parser = argparse.ArgumentParser(description='Train 5-band SR V3')
    parser.add_argument('--data_dir', type=str, default='data/sr_training/5band_v3/combined/mmap')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/5band_v3_direct')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training 5-Band SR V3 (Direct Prediction)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = FiveBandMmapDataset(args.data_dir, sequence_length=3, augment=True)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    model = create_model_v3()
    model = model.to(device)
    
    criterion = CombinedLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    es = EarlyStopping(patience=args.patience, mode='max', metric_name='psnr')
    
    best_psnr = 0
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, psnr_10m, psnr_20m = validate(model, val_loader, criterion, device)
        avg_psnr = (psnr_10m + psnr_20m) / 2
        scheduler.step()
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PSNR: 10m={psnr_10m:.2f} 20m={psnr_20m:.2f} avg={avg_psnr:.2f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {time.time()-t0:.1f}s")
        
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
                'model_version': 'v3_direct',
            }, checkpoint_dir / 'best_model.pt')
            print(f"  ✓ New best PSNR: {avg_psnr:.2f} dB")
        
        if es(avg_psnr, epoch):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best PSNR: {best_psnr:.2f} dB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
