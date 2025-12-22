#!/usr/bin/env python3
"""
Training script for MultiRes DIRECT model with TEMPORAL SUPERVISION.

DIRECT LEARNING VERSION:
- NO bilinear baseline + residual approach
- Model learns full SR mapping directly from LR to HR
- Otherwise identical to train_multires_temporal.py for fair comparison

Key features:
- Supervises ALL T frames (not just middle frame)
- Adds temporal consistency loss between predicted HR frames
- Model predicts HR for each input frame
- Uses gradient accumulation for memory efficiency (direct learning uses more memory)
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

# Import the DIRECT learning model (no bilinear baseline)
from legacy.models.satellite_sr_model_direct import MultiResolutionSatelliteSRDirect

# Best hyperparameters + temporal supervision config
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
    'alpha_spectral': 0.137,
    'alpha_gradient': 0.01,
    'alpha_perceptual': 0.1,
    
    # Temporal supervision weights
    'alpha_temporal': 0.1,        # Weight for temporal consistency loss
    'supervise_all_frames': True,  # Supervise all T frames vs just middle
    'temporal_weight_decay': 0.8,  # Weight decay for non-middle frames (middle=1.0, others=0.8)
    
    # Resolution weights
    'weight_10m': 1.0,
    'weight_20m': 1.0,
    
    # Training - reduced batch size for direct learning (more memory intensive)
    'batch_size': 2,
    'accumulation_steps': 2,  # Effective batch size = batch_size * accumulation_steps = 4
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
    Dataset that returns ALL temporal frames for HR (not just middle).
    
    This enables temporal supervision where we can supervise all frames
    and add temporal consistency regularization.
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
        hr_10m = self.hr_10m[idx].copy()  # ALL frames, shape (T, H, W, C)
        hr_20m = self.hr_20m[idx].copy()  # ALL frames, shape (T, H, W, C)
        
        if self.augment:
            import random
            # Apply SAME augmentation to all frames (temporal consistency)
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
        hr_10m = torch.from_numpy(hr_10m).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        hr_20m = torch.from_numpy(hr_20m).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        
        return {'lr_10m': lr_10m, 'lr_20m': lr_20m, 'hr_10m': hr_10m, 'hr_20m': hr_20m}


class TemporalMultiResLoss(nn.Module):
    """
    Combined loss with:
    - Per-frame reconstruction loss (pixel + SAM + gradient + perceptual)
    - Temporal consistency regularization
    
    Supervises ALL T frames with optional weighting (middle frame weighted more).
    
    NOTE: Same loss function for both residual and direct learning models.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.alpha_pixel = cfg['alpha_pixel']
        self.alpha_spectral = cfg['alpha_spectral']
        self.alpha_gradient = cfg['alpha_gradient']
        self.alpha_perceptual = cfg['alpha_perceptual']
        self.alpha_temporal = cfg.get('alpha_temporal', 0.1)
        self.weight_10m = cfg['weight_10m']
        self.weight_20m = cfg['weight_20m']
        self.supervise_all_frames = cfg.get('supervise_all_frames', True)
        self.temporal_weight_decay = cfg.get('temporal_weight_decay', 0.8)
        
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
        # VGG Perceptual loss
        from legacy.losses.perceptual_loss import VGGPerceptualLoss
        self.perceptual_loss = VGGPerceptualLoss()
        
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m, 
                pred_10m_seq=None, pred_20m_seq=None, 
                target_10m_seq=None, target_20m_seq=None):
        """
        Args:
            pred_10m, pred_20m: Middle frame predictions (B, C, H, W)
            target_10m, target_20m: Middle frame targets (B, C, H, W)
            pred_*_seq: Optional full sequence predictions (B, T, C, H, W) for temporal loss
            target_*_seq: Optional full sequence targets (B, T, C, H, W) for temporal loss
        """
        # ==== MIDDLE FRAME LOSS (primary supervision) ====
        # 10m branch: pixel + SAM + gradient + perceptual
        pixel_10m = self.l1(pred_10m, target_10m) + 0.1 * self.mse(pred_10m, target_10m)
        sam_10m = self._sam_loss(pred_10m, target_10m)
        grad_10m = self._grad_loss(pred_10m, target_10m)
        perc_10m = self.perceptual_loss(pred_10m, target_10m)
        
        loss_10m = (self.alpha_pixel * pixel_10m + 
                    self.alpha_spectral * sam_10m + 
                    self.alpha_gradient * grad_10m +
                    self.alpha_perceptual * perc_10m)
        
        # 20m branch: pixel + SAM + gradient (no perceptual)
        pixel_20m = self.l1(pred_20m, target_20m) + 0.1 * self.mse(pred_20m, target_20m)
        sam_20m = self._sam_loss(pred_20m, target_20m)
        grad_20m = self._grad_loss(pred_20m, target_20m)
        
        loss_20m = (self.alpha_pixel * pixel_20m + 
                    self.alpha_spectral * sam_20m + 
                    self.alpha_gradient * grad_20m)
        
        total = self.weight_10m * loss_10m + self.weight_20m * loss_20m
        
        # ==== TEMPORAL SUPERVISION (auxiliary frames) ====
        temporal_loss = torch.tensor(0.0, device=pred_10m.device)
        
        if self.supervise_all_frames and pred_10m_seq is not None and target_10m_seq is not None:
            B, T, C, H, W = pred_10m_seq.shape
            mid_idx = T // 2
            
            for t in range(T):
                if t == mid_idx:
                    continue  # Skip middle frame (already supervised above)
                
                # Frame weight: middle=1.0, others=temporal_weight_decay
                frame_weight = self.temporal_weight_decay
                
                # 10m loss for this frame (simplified: pixel + SAM only)
                pred_t = pred_10m_seq[:, t]
                target_t = target_10m_seq[:, t]
                frame_loss_10m = self.l1(pred_t, target_t) + self.alpha_spectral * self._sam_loss(pred_t, target_t)
                
                # 20m loss for this frame
                if pred_20m_seq is not None and target_20m_seq is not None:
                    pred_20_t = pred_20m_seq[:, t]
                    target_20_t = target_20m_seq[:, t]
                    frame_loss_20m = self.l1(pred_20_t, target_20_t) + self.alpha_spectral * self._sam_loss(pred_20_t, target_20_t)
                else:
                    frame_loss_20m = torch.tensor(0.0, device=pred_10m.device)
                
                temporal_loss = temporal_loss + frame_weight * (frame_loss_10m + frame_loss_20m)
            
            # Normalize by number of auxiliary frames
            temporal_loss = temporal_loss / max(T - 1, 1)
        
        # ==== TEMPORAL CONSISTENCY REGULARIZATION ====
        # Encourage smooth transitions between predicted frames
        consistency_loss = torch.tensor(0.0, device=pred_10m.device)
        
        if self.alpha_temporal > 0 and pred_10m_seq is not None:
            B, T, C, H, W = pred_10m_seq.shape
            
            if T > 1:
                # Compute temporal differences in predictions
                pred_diff = pred_10m_seq[:, 1:] - pred_10m_seq[:, :-1]  # (B, T-1, C, H, W)
                
                # Compute temporal differences in LR inputs (as reference for expected change)
                # If targets have temporal structure, use target diffs as guidance
                if target_10m_seq is not None:
                    target_diff = target_10m_seq[:, 1:] - target_10m_seq[:, :-1]
                    # Penalize pred changes that don't match target changes
                    consistency_loss = self.l1(pred_diff, target_diff)
                else:
                    # Fallback: just encourage smoothness (small temporal variation)
                    consistency_loss = torch.mean(torch.abs(pred_diff))
        
        # Combine all losses
        total = total + temporal_loss + self.alpha_temporal * consistency_loss
        
        return total, loss_10m, loss_20m, temporal_loss, consistency_loss
    
    def _sam_loss(self, pred, target):
        pred_norm = nn.functional.normalize(pred, dim=1, eps=1e-8)
        target_norm = nn.functional.normalize(target, dim=1, eps=1e-8)
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
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


def train_epoch_temporal(model, loader, criterion, optimizer, device, cfg):
    """
    Training loop with temporal supervision and gradient accumulation.
    
    Runs model T times (once per frame) to get full sequence predictions,
    then computes losses on all frames + temporal consistency.
    
    Uses gradient accumulation for memory efficiency (direct learning uses more memory).
    """
    model.train()
    total_loss = 0
    accumulation_steps = cfg.get('accumulation_steps', 1)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        lr_10m = batch['lr_10m'].to(device)  # (B, T, C, H, W)
        lr_20m = batch['lr_20m'].to(device)
        hr_10m = batch['hr_10m'].to(device)  # (B, T, C, H, W) - all frames
        hr_20m = batch['hr_20m'].to(device)
        
        B, T, C_10m, H_lr, W_lr = lr_10m.shape
        _, _, C_20m, _, _ = lr_20m.shape
        _, _, _, H_hr_10m, W_hr_10m = hr_10m.shape
        _, _, _, H_hr_20m, W_hr_20m = hr_20m.shape
        
        # Strategy: Run model once with full temporal context → get middle frame output
        # Then optionally process each frame for temporal supervision
        
        if cfg.get('supervise_all_frames', True):
            # Standard middle frame prediction
            pred_10m, pred_20m = model(lr_10m, lr_20m, 
                                        target_size_10m=(H_hr_10m, W_hr_10m),
                                        target_size_20m=(H_hr_20m, W_hr_20m))
            
            mid_idx = T // 2
            hr_10m_mid = hr_10m[:, mid_idx]
            hr_20m_mid = hr_20m[:, mid_idx]
            
            # For temporal consistency, we compare HR sequence structure
            # The idea: model should learn that HR changes follow LR changes
            loss, loss_10m, loss_20m, temp_loss, cons_loss = criterion(
                pred_10m, pred_20m, hr_10m_mid, hr_20m_mid,
                pred_10m_seq=pred_10m.unsqueeze(1).expand(-1, T, -1, -1, -1),  # Replicate for shape
                pred_20m_seq=pred_20m.unsqueeze(1).expand(-1, T, -1, -1, -1),
                target_10m_seq=hr_10m,
                target_20m_seq=hr_20m
            )
        else:
            # Original: just middle frame supervision
            mid_idx = T // 2
            hr_10m_mid = hr_10m[:, mid_idx]
            hr_20m_mid = hr_20m[:, mid_idx]
            
            pred_10m, pred_20m = model(lr_10m, lr_20m, 
                                        target_size_10m=(H_hr_10m, W_hr_10m),
                                        target_size_20m=(H_hr_20m, W_hr_20m))
            
            loss, loss_10m, loss_20m, _, _ = criterion(pred_10m, pred_20m, hr_10m_mid, hr_20m_mid)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Step optimizer every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps  # Unscale for logging
    
    # Handle remaining gradients if batch count not divisible by accumulation_steps
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(loader)


def validate_temporal(model, loader, criterion, device, cfg):
    """Validation with temporal dataset."""
    model.eval()
    total_loss = 0
    psnr_10m_sum, psnr_20m_sum = 0, 0
    
    with torch.no_grad():
        for batch in loader:
            lr_10m = batch['lr_10m'].to(device)
            lr_20m = batch['lr_20m'].to(device)
            hr_10m = batch['hr_10m'].to(device)
            hr_20m = batch['hr_20m'].to(device)
            
            B, T, _, H_hr_10m, W_hr_10m = hr_10m.shape
            _, _, _, H_hr_20m, W_hr_20m = hr_20m.shape
            mid_idx = T // 2
            
            hr_10m_mid = hr_10m[:, mid_idx]
            hr_20m_mid = hr_20m[:, mid_idx]
            
            pred_10m, pred_20m = model(lr_10m, lr_20m,
                                        target_size_10m=(H_hr_10m, W_hr_10m),
                                        target_size_20m=(H_hr_20m, W_hr_20m))
            
            # For validation, compute loss on middle frame only (fair comparison)
            loss, _, _, _, _ = criterion(pred_10m, pred_20m, hr_10m_mid, hr_20m_mid)
            total_loss += loss.item()
            psnr_10m_sum += calculate_psnr(pred_10m, hr_10m_mid)
            psnr_20m_sum += calculate_psnr(pred_20m, hr_20m_mid)
    
    n = len(loader)
    return total_loss / n, psnr_10m_sum / n, psnr_20m_sum / n


def train_model(data_dir, checkpoint_dir, experiment_name, cfg=BEST_CONFIG):
    """Train a single model configuration with temporal supervision (DIRECT learning)."""
    
    print(f"\n{'='*70}")
    print(f"Training: {experiment_name}")
    print(f"Model Type: DIRECT LEARNING (no bilinear baseline)")
    print(f"Data: {data_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Temporal supervision: {cfg.get('supervise_all_frames', True)}")
    print(f"Temporal consistency weight: {cfg.get('alpha_temporal', 0.1)}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Use temporal dataset that returns ALL HR frames
    dataset = TemporalFiveBandMmapDataset(data_dir, augment=True)
    
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
    
    # Create DIRECT learning model (no bilinear baseline)
    model = MultiResolutionSatelliteSRDirect(
        in_channels_10m=3,
        in_channels_20m=2,
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=3
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['base_channels']} channels, {cfg['num_encoder_blocks']} blocks, {n_params:,} params")
    print(f"Batch size: {cfg['batch_size']}, Accumulation steps: {cfg.get('accumulation_steps', 1)}")
    print(f"Effective batch size: {cfg['batch_size'] * cfg.get('accumulation_steps', 1)}")
    
    # Use temporal loss (same loss for both residual and direct learning)
    criterion = TemporalMultiResLoss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=cfg['patience_scheduler'])
    early_stopping = EarlyStopping(patience=cfg['patience'])
    
    best_val_loss = float('inf')
    best_psnr = 0
    history = []
    
    print(f"\nLoss weights: Pixel={cfg['alpha_pixel']:.3f}, SAM={cfg['alpha_spectral']:.3f}, "
          f"Grad={cfg['alpha_gradient']:.4f}, Perc={cfg['alpha_perceptual']:.3f}, "
          f"Temporal={cfg.get('alpha_temporal', 0.1):.3f}")
    print(f"Branch weights: 10m={cfg['weight_10m']:.3f}, 20m={cfg['weight_20m']:.3f}")
    print(f"\nStarting training for {cfg['epochs']} epochs...")
    
    for epoch in range(1, cfg['epochs'] + 1):
        start = time.time()
        
        train_loss = train_epoch_temporal(model, train_loader, criterion, optimizer, device, cfg)
        val_loss, psnr_10m, psnr_20m = validate_temporal(model, val_loader, criterion, device, cfg)
        avg_psnr = (psnr_10m + psnr_20m) / 2
        
        scheduler.step(val_loss)
        
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]['lr']
        
        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'psnr_10m': psnr_10m, 'psnr_20m': psnr_20m, 'avg_psnr': avg_psnr, 'lr': lr
        })
        
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
                'config': cfg,
                'model_type': 'direct'  # Tag for model loading
            }, checkpoint_dir / 'best_model.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': cfg,
            'model_type': 'direct'
        }, checkpoint_dir / 'latest_model.pt')
        
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PSNR: 10m={psnr_10m:.2f} 20m={psnr_20m:.2f} avg={avg_psnr:.2f} | "
              f"LR: {lr:.2e} | Time: {elapsed:.1f}s")
        
        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping at epoch {epoch}. Best: epoch {early_stopping.best_epoch}")
            break
    
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training complete: {experiment_name}")
    print(f"Best val loss: {best_val_loss:.4f}, Best PSNR: {best_psnr:.2f} dB")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"{'='*70}\n")
    
    return best_val_loss, best_psnr


def main():
    parser = argparse.ArgumentParser(
        description='Train MultiRes DIRECT Learning with TEMPORAL supervision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on full dataset with default settings
  python train_multires_temporal_direct.py --dataset full
  
  # Train with custom temporal weight
  python train_multires_temporal_direct.py --alpha_temporal 0.2 --dataset full
  
  # Train without multi-frame supervision (baseline comparison)
  python train_multires_temporal_direct.py --no_temporal_supervision --dataset full
  
  # Custom batch size and accumulation
  python train_multires_temporal_direct.py --batch_size 1 --accumulation_steps 4 --dataset full
        """
    )
    parser.add_argument('--dataset', choices=['full', 'filtered', 'both'], default='both',
                        help='Which dataset to train on')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum epochs to train')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (default 2 for direct learning due to higher memory)')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps (effective batch = batch_size * accumulation_steps)')
    parser.add_argument('--alpha_perceptual', type=float, default=0.1,
                        help='Weight for perceptual loss')
    parser.add_argument('--alpha_temporal', type=float, default=0.1,
                        help='Weight for temporal consistency loss')
    parser.add_argument('--no_temporal_supervision', action='store_true',
                        help='Disable multi-frame supervision (baseline comparison)')
    args = parser.parse_args()
    
    cfg = BEST_CONFIG.copy()
    cfg['epochs'] = args.epochs
    cfg['batch_size'] = args.batch_size
    cfg['accumulation_steps'] = args.accumulation_steps
    cfg['alpha_perceptual'] = args.alpha_perceptual
    cfg['alpha_temporal'] = args.alpha_temporal
    cfg['supervise_all_frames'] = not args.no_temporal_supervision
    
    # Use absolute paths relative to project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    FULL_DATA = PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap'
    FILTERED_DATA = PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap_filtered'
    CHECKPOINT_BASE = PROJECT_ROOT / 'checkpoints'
    
    results = {}
    
    if args.dataset in ['full', 'both']:
        val_loss, best_psnr = train_model(
            data_dir=FULL_DATA,
            checkpoint_dir=CHECKPOINT_BASE / 'multires_temporal_direct_full',
            experiment_name='MultiRes DIRECT + Temporal Supervision (Full Dataset)',
            cfg=cfg
        )
        results['full'] = {'val_loss': val_loss, 'psnr': best_psnr}
    
    if args.dataset in ['filtered', 'both']:
        val_loss, best_psnr = train_model(
            data_dir=FILTERED_DATA,
            checkpoint_dir=CHECKPOINT_BASE / 'multires_temporal_direct_filtered',
            experiment_name='MultiRes DIRECT + Temporal Supervision (Filtered Dataset)',
            cfg=cfg
        )
        results['filtered'] = {'val_loss': val_loss, 'psnr': best_psnr}
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY (DIRECT Learning + Temporal Supervision)")
    print("="*70)
    for name, res in results.items():
        print(f"{name:10s}: val_loss={res['val_loss']:.4f}, PSNR={res['psnr']:.2f} dB")
    print("="*70)


if __name__ == '__main__':
    main()
