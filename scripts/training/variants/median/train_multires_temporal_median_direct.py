#!/usr/bin/env python3
"""
Training script for MultiRes DIRECT LEARNING model with MEDIAN FRAME COMPOSITING.

KEY DIFFERENCE: Uses DIRECT LEARNING (no bilinear baseline).
Model learns full SR mapping: Output = learned_SR (not baseline + residual)

Key idea: Instead of using the middle frame or supervising all frames,
we create a CLEANER composite by taking the pixel-wise MEDIAN across
the temporal sequence. This is a robust method to reduce noise:

1. Noise Reduction: Median is robust to outliers (clouds, sensor noise, etc.)
2. Simplified Training: One clean target per sample (no multi-frame complexity)
3. Better Signal: The median preserves true signal while filtering noise

The model still receives all T frames as input (temporal context),
but learns to predict a single, clean median composite as output.

This is a COMPARISON variant to test if residual learning provides
benefits over direct learning.
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

# Import the DIRECT LEARNING model (no bilinear baseline)
from satellite_sr_model_direct import MultiResolutionSatelliteSRDirect as MultiResolutionSatelliteSR

# Configuration with median compositing
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
    
    # Median compositing options
    'composite_method': 'median',  # 'median', 'mean', or 'middle'
    'lr_composite': True,          # Also composite LR inputs (vs just HR target)
    
    # Resolution weights
    'weight_10m': 1.0,
    'weight_20m': 1.0,
    
    # Training
    'batch_size': 2,           # Reduced for direct learning (more memory intensive)
    'accumulation_steps': 2,   # Effective batch size = batch_size * accumulation_steps = 4
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


class MedianCompositeMmapDataset(Dataset):
    """
    Dataset that creates MEDIAN composite from temporal frames.
    
    For each sample:
    - LR inputs: Either full T-frame sequence (for temporal context) 
                 OR median-composited single frame
    - HR target: Pixel-wise MEDIAN across T frames (cleaner, denoised)
    
    The median operation is robust to outliers like:
    - Temporary clouds that weren't masked
    - Sensor noise spikes
    - Registration artifacts in individual frames
    """
    
    def __init__(self, mmap_dir, sequence_length=3, augment=True, 
                 composite_method='median', lr_composite=False):
        """
        Args:
            mmap_dir: Path to mmap data directory
            sequence_length: Number of temporal frames (T)
            augment: Apply data augmentation
            composite_method: 'median', 'mean', or 'middle'
            lr_composite: If True, also composite LR (otherwise keep full sequence)
        """
        mmap_dir = Path(mmap_dir)
        self.lr = np.load(mmap_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(mmap_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr_10m = np.load(mmap_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(mmap_dir / 'hr_20m.npy', mmap_mode='r')
        self.n_samples = self.lr.shape[0]
        self.sequence_length = sequence_length
        self.augment = augment
        self.composite_method = composite_method
        self.lr_composite = lr_composite
        
        print(f"MedianComposite Dataset: {self.n_samples} samples")
        print(f"  Composite method: {composite_method}")
        print(f"  LR composite: {lr_composite}")
        print(f"  LR 10m: {self.lr.shape}, LR 20m: {self.lr_20m.shape}")
        print(f"  HR 10m: {self.hr_10m.shape}, HR 20m: {self.hr_20m.shape}")
    
    def _compute_composite(self, frames, method='median'):
        """
        Compute composite from temporal frames.
        
        Args:
            frames: (T, H, W, C) array
            method: 'median', 'mean', or 'middle'
        
        Returns:
            (H, W, C) composite array
        """
        if method == 'median':
            return np.median(frames, axis=0)
        elif method == 'mean':
            return np.mean(frames, axis=0)
        elif method == 'middle':
            mid_idx = frames.shape[0] // 2
            return frames[mid_idx]
        else:
            raise ValueError(f"Unknown composite method: {method}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        lr_10m = self.lr[idx].copy()      # (T, H, W, C)
        lr_20m = self.lr_20m[idx].copy()  # (T, H, W, C)
        hr_10m = self.hr_10m[idx].copy()  # (T, H, W, C)
        hr_20m = self.hr_20m[idx].copy()  # (T, H, W, C)
        
        # Apply augmentation BEFORE compositing (to all frames consistently)
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
        
        # Compute HR target as MEDIAN composite (clean, denoised target)
        hr_10m_composite = self._compute_composite(hr_10m, self.composite_method)  # (H, W, C)
        hr_20m_composite = self._compute_composite(hr_20m, self.composite_method)  # (H, W, C)
        
        # LR: either keep full sequence (for temporal context) or composite
        if self.lr_composite:
            # Single-frame LR (composited)
            lr_10m_out = self._compute_composite(lr_10m, self.composite_method)  # (H, W, C)
            lr_20m_out = self._compute_composite(lr_20m, self.composite_method)  # (H, W, C)
            # Expand to (T=1, H, W, C) for model compatibility
            lr_10m_out = np.expand_dims(lr_10m_out, axis=0)
            lr_20m_out = np.expand_dims(lr_20m_out, axis=0)
        else:
            # Full temporal sequence for LR (model uses temporal attention)
            lr_10m_out = lr_10m  # (T, H, W, C)
            lr_20m_out = lr_20m
        
        # Convert to tensors
        # LR: (T, H, W, C) -> (T, C, H, W)
        lr_10m_t = torch.from_numpy(lr_10m_out).permute(0, 3, 1, 2).float()
        lr_20m_t = torch.from_numpy(lr_20m_out).permute(0, 3, 1, 2).float()
        
        # HR: (H, W, C) -> (C, H, W) - single composite frame
        hr_10m_t = torch.from_numpy(hr_10m_composite).permute(2, 0, 1).float()
        hr_20m_t = torch.from_numpy(hr_20m_composite).permute(2, 0, 1).float()
        
        return {'lr_10m': lr_10m_t, 'lr_20m': lr_20m_t, 
                'hr_10m': hr_10m_t, 'hr_20m': hr_20m_t}


class MultiResSpectralLoss(nn.Module):
    """
    Standard multi-resolution loss (same as train_best_multires_sr.py).
    
    No temporal complexity - just single frame supervision.
    The "denoising" happens in the TARGET (median composite),
    not in the loss function.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.alpha_pixel = cfg['alpha_pixel']
        self.alpha_spectral = cfg['alpha_spectral']
        self.alpha_gradient = cfg['alpha_gradient']
        self.alpha_perceptual = cfg['alpha_perceptual']
        self.weight_10m = cfg['weight_10m']
        self.weight_20m = cfg['weight_20m']
        
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
        # VGG Perceptual loss with input scaling for satellite data
        # Satellite data has low dynamic range (mean ~0.05) vs ImageNet (mean ~0.45)
        # Scale by 2.5 to shift data closer to expected VGG input distribution
        from perceptual_loss import VGGPerceptualLoss
        self.perceptual_loss = VGGPerceptualLoss(input_scale=2.5)
        
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m):
        """
        Args:
            pred_10m: (B, C, H, W) 10m branch prediction
            pred_20m: (B, C, H, W) 20m branch prediction
            target_10m: (B, C, H, W) 10m target (median composite)
            target_20m: (B, C, H, W) 20m target (median composite)
        """
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
        
        return total, loss_10m, loss_20m
    
    def _sam_loss(self, pred, target):
        """Spectral Angle Mapper loss."""
        pred_norm = nn.functional.normalize(pred, dim=1, eps=1e-8)
        target_norm = nn.functional.normalize(target, dim=1, eps=1e-8)
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        return 1.0 - cos_sim.mean()
    
    def _grad_loss(self, pred, target):
        """Gradient loss for edge preservation."""
        loss = 0
        for c in range(pred.shape[1]):
            pred_gx = nn.functional.conv2d(pred[:, c:c+1], self.sobel_x, padding=1)
            pred_gy = nn.functional.conv2d(pred[:, c:c+1], self.sobel_y, padding=1)
            target_gx = nn.functional.conv2d(target[:, c:c+1], self.sobel_x, padding=1)
            target_gy = nn.functional.conv2d(target[:, c:c+1], self.sobel_y, padding=1)
            loss += self.l1(pred_gx, target_gx) + self.l1(pred_gy, target_gy)
        return loss / pred.shape[1]


def calculate_psnr(pred, target):
    """Calculate PSNR between prediction and target."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)).item()


def train_epoch(model, loader, criterion, optimizer, device, cfg):
    """Training with gradient accumulation for memory efficiency."""
    model.train()
    total_loss = 0
    accumulation_steps = cfg.get('accumulation_steps', 1)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        lr_10m = batch['lr_10m'].to(device)  # (B, T, C, H, W) or (B, 1, C, H, W)
        lr_20m = batch['lr_20m'].to(device)
        hr_10m = batch['hr_10m'].to(device)  # (B, C, H, W) - single composite frame
        hr_20m = batch['hr_20m'].to(device)
        
        _, _, _, H_hr_10m, W_hr_10m = lr_10m.shape[0], lr_10m.shape[1], lr_10m.shape[2], hr_10m.shape[2], hr_10m.shape[3]
        _, _, _, H_hr_20m, W_hr_20m = lr_20m.shape[0], lr_20m.shape[1], lr_20m.shape[2], hr_20m.shape[2], hr_20m.shape[3]
        
        # Model forward pass
        pred_10m, pred_20m = model(
            lr_10m, lr_20m,
            target_size_10m=(H_hr_10m, W_hr_10m),
            target_size_20m=(H_hr_20m, W_hr_20m)
        )
        
        # Loss on median composite target
        loss, loss_10m, loss_20m = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
        
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


def validate(model, loader, criterion, device):
    """Validation with median composite targets."""
    model.eval()
    total_loss = 0
    psnr_10m_sum, psnr_20m_sum = 0, 0
    
    with torch.no_grad():
        for batch in loader:
            lr_10m = batch['lr_10m'].to(device)
            lr_20m = batch['lr_20m'].to(device)
            hr_10m = batch['hr_10m'].to(device)  # Median composite
            hr_20m = batch['hr_20m'].to(device)
            
            H_hr_10m, W_hr_10m = hr_10m.shape[2], hr_10m.shape[3]
            H_hr_20m, W_hr_20m = hr_20m.shape[2], hr_20m.shape[3]
            
            pred_10m, pred_20m = model(
                lr_10m, lr_20m,
                target_size_10m=(H_hr_10m, W_hr_10m),
                target_size_20m=(H_hr_20m, W_hr_20m)
            )
            
            loss, _, _ = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
            total_loss += loss.item()
            psnr_10m_sum += calculate_psnr(pred_10m, hr_10m)
            psnr_20m_sum += calculate_psnr(pred_20m, hr_20m)
    
    n = len(loader)
    return total_loss / n, psnr_10m_sum / n, psnr_20m_sum / n


def train_model(data_dir, checkpoint_dir, experiment_name, cfg=BEST_CONFIG):
    """Train model with median composite targets."""
    
    print(f"\n{'='*70}")
    print(f"Training: {experiment_name}")
    print(f"Data: {data_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Composite method: {cfg.get('composite_method', 'median')}")
    print(f"LR composite: {cfg.get('lr_composite', False)}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Use median composite dataset
    dataset = MedianCompositeMmapDataset(
        data_dir, 
        augment=True,
        composite_method=cfg.get('composite_method', 'median'),
        lr_composite=cfg.get('lr_composite', False)
    )
    
    n_val = max(int(len(dataset) * 0.1), 10)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.augment = False
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['batch_size'], 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['batch_size'], 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Determine num_frames for model based on whether we're compositing LR
    num_frames = 1 if cfg.get('lr_composite', False) else 3
    
    # Create model
    model = MultiResolutionSatelliteSR(
        in_channels_10m=3,
        in_channels_20m=2,
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=num_frames
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['base_channels']} channels, {cfg['num_encoder_blocks']} blocks, "
          f"T={num_frames}, {n_params:,} params")
    
    # Standard loss (no temporal complexity)
    criterion = MultiResSpectralLoss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=cfg['patience_scheduler']
    )
    early_stopping = EarlyStopping(patience=cfg['patience'])
    
    best_val_loss = float('inf')
    best_psnr = 0
    history = []
    
    print(f"\nLoss weights: Pixel={cfg['alpha_pixel']:.3f}, SAM={cfg['alpha_spectral']:.3f}, "
          f"Grad={cfg['alpha_gradient']:.4f}, Perc={cfg['alpha_perceptual']:.3f}")
    print(f"Branch weights: 10m={cfg['weight_10m']:.3f}, 20m={cfg['weight_20m']:.3f}")
    print(f"\nStarting training for {cfg['epochs']} epochs...")
    
    for epoch in range(1, cfg['epochs'] + 1):
        start = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, cfg)
        val_loss, psnr_10m, psnr_20m = validate(model, val_loader, criterion, device)
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
                'config': cfg
            }, checkpoint_dir / 'best_model.pt')
        
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
        description='Train MultiRes with MEDIAN FRAME COMPOSITING',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Median Frame Compositing Strategy:
  - HR Target: Pixel-wise MEDIAN across T frames (clean, denoised)
  - LR Input: Either full T-frame sequence (temporal context) 
              or median-composited single frame
  
  This approach is robust to outliers (clouds, noise) in individual frames.

Examples:
  # Default: median HR target, full LR sequence
  python train_multires_temporal_median.py --dataset full
  
  # Median both LR and HR (fully composited)
  python train_multires_temporal_median.py --lr_composite
  
  # Compare with mean compositing
  python train_multires_temporal_median.py --composite mean
        """
    )
    parser.add_argument('--dataset', choices=['full', 'filtered', 'both'], default='full',
                        help='Which dataset to train on')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum epochs to train')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (default 2 for direct learning due to higher memory)')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps (effective batch = batch_size * accumulation_steps)')
    parser.add_argument('--alpha_perceptual', type=float, default=0.1,
                        help='Weight for perceptual loss')
    parser.add_argument('--composite', choices=['median', 'mean', 'middle'], default='median',
                        help='Compositing method for temporal frames')
    parser.add_argument('--lr_composite', action='store_true',
                        help='Also composite LR inputs (vs keeping full sequence)')
    args = parser.parse_args()
    
    cfg = BEST_CONFIG.copy()
    cfg['epochs'] = args.epochs
    cfg['batch_size'] = args.batch_size
    cfg['accumulation_steps'] = args.accumulation_steps
    cfg['alpha_perceptual'] = args.alpha_perceptual
    cfg['composite_method'] = args.composite
    cfg['lr_composite'] = args.lr_composite
    
    # Use absolute paths relative to project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
    FULL_DATA = PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap'
    FILTERED_DATA = PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap_filtered'
    CHECKPOINT_BASE = PROJECT_ROOT / 'checkpoints'
    
    # Checkpoint name includes composite method
    suffix = f"_{args.composite}"
    if args.lr_composite:
        suffix += "_lrcomp"
    
    results = {}
    
    if args.dataset in ['full', 'both']:
        val_loss, best_psnr = train_model(
            data_dir=FULL_DATA,
            checkpoint_dir=CHECKPOINT_BASE / f'multires_median{suffix}_direct_full',
            experiment_name=f'MultiRes DIRECT + {args.composite.capitalize()} Composite (Full Dataset)',
            cfg=cfg
        )
        results['full'] = {'val_loss': val_loss, 'psnr': best_psnr}
    
    if args.dataset in ['filtered', 'both']:
        val_loss, best_psnr = train_model(
            data_dir=FILTERED_DATA,
            checkpoint_dir=CHECKPOINT_BASE / f'multires_median{suffix}_direct_filtered',
            experiment_name=f'MultiRes DIRECT + {args.composite.capitalize()} Composite (Filtered Dataset)',
            cfg=cfg
        )
        results['filtered'] = {'val_loss': val_loss, 'psnr': best_psnr}
    
    print("\n" + "="*70)
    print(f"TRAINING SUMMARY ({args.composite.upper()} Compositing)")
    print("="*70)
    for name, res in results.items():
        print(f"{name:10s}: val_loss={res['val_loss']:.4f}, PSNR={res['psnr']:.2f} dB")
    print("="*70)


if __name__ == '__main__':
    main()
