#!/usr/bin/env python3
"""
Training script for 5-band multi-resolution super-resolution.

This script trains a model that:
1. Super-resolves 10m bands (B2, B3, B4) from 77×77 → 256×256 (3.33×)
2. Super-resolves 20m bands (B5, B8A) from 38×38 → 128×128 (3.33×)

Both branches use REAL HR targets from Planet SuperDove:
- 10m HR: Planet B2, B4, B6 (Blue, Green, Red) at 3m
- 20m HR: Planet B7, B8 (RedEdge, NIR) at 6m

This is the key improvement: we now have REAL HR targets for RedEdge and NIR!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Import model and loss
from models.satellite_sr_model import MultiResolutionSatelliteSR, SpectralFidelityLoss


class FiveBandDataset(Dataset):
    """Dataset for 5-band SuperDove data with real HR targets.
    
    Supports full temporal sequences (T=3) for temporal attention,
    consistent with train_native_resolution.py.
    """
    
    def __init__(self, samples: List, sequence_length: int = 3, augment: bool = True):
        """
        Args:
            samples: List of TrainingSample5Band objects
            sequence_length: Number of temporal frames (default 3)
            augment: Whether to apply data augmentation
        """
        self.samples = samples
        self.sequence_length = sequence_length
        self.augment = augment
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get arrays - (T, H, W, C) format from cache
        lr_10m = sample.lr_sequence.copy()        # (T, 77, 77, 3)
        lr_20m = sample.lr_20m_sequence.copy()    # (T, 38, 38, 2)
        hr_10m = sample.hr_10m_sequence.copy()    # (T, 256, 256, 3)
        hr_20m = sample.hr_20m_sequence.copy()    # (T, 128, 128, 2)
        
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
        hr_20m = torch.from_numpy(hr_20m[mid_idx]).permute(2, 0, 1).float()  # (2, 128, 128)
        
        return {
            'lr_10m': lr_10m,      # (T, 3, 77, 77)
            'lr_20m': lr_20m,      # (T, 2, 38, 38)
            'hr_10m': hr_10m,      # (3, 256, 256)
            'hr_20m': hr_20m,      # (2, 128, 128)
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
        
        Args:
            pred: (B, C, H, W) predicted
            target: (B, C, H, W) target
        
        Returns:
            Scalar loss (1 - mean cosine similarity)
        """
        # Normalize along channel dimension
        pred_norm = nn.functional.normalize(pred, dim=1, eps=1e-8)
        target_norm = nn.functional.normalize(target, dim=1, eps=1e-8)
        
        # Cosine similarity per pixel
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)  # (B, H, W)
        
        # SAM loss: 1 - mean cosine similarity
        return 1.0 - cos_sim.mean()
    
    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Gradient loss for edge preservation using Sobel filters.
        """
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
        # Pixel loss: L1 + 0.1*MSE
        pixel_loss = self.l1_loss(pred, target) + 0.1 * self.mse_loss(pred, target)
        
        # Spectral angle loss (SAM)
        sam_loss = self.spectral_angle_loss(pred, target)
        
        # Gradient/edge loss
        grad_loss = self.gradient_loss(pred, target)
        
        # Combined: pixel + spectral + gradient
        total = self.alpha_pixel * pixel_loss + \
                self.alpha_spectral * sam_loss + \
                self.alpha_gradient * grad_loss
        
        return total
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m):
        # 10m branch loss (full spectral fidelity)
        loss_10m = self._compute_branch_loss(pred_10m, target_10m)
        
        # 20m branch loss (full spectral fidelity)
        loss_20m = self._compute_branch_loss(pred_20m, target_20m)
        
        # Combined loss with branch weights
        total_loss = self.weight_10m * loss_10m + self.weight_20m * loss_20m
        
        return total_loss, loss_10m, loss_20m


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate PSNR between prediction and target."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 1.0  # Normalized data
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
        
        # Forward pass with explicit target sizes
        pred_10m, pred_20m = model(
            lr_10m, lr_20m,
            target_size_10m=(hr_10m.shape[2], hr_10m.shape[3]),  # (256, 256)
            target_size_20m=(hr_20m.shape[2], hr_20m.shape[3])   # (128, 128)
        )
        
        # Calculate loss
        loss, loss_10m, loss_20m = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
        
        # Backward pass
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
    
    # Per-band PSNR accumulators
    band_psnr_10m = {'Blue': 0, 'Green': 0, 'Red': 0}
    band_psnr_20m = {'RedEdge': 0, 'NIR': 0}
    
    with torch.no_grad():
        for batch in dataloader:
            lr_10m = batch['lr_10m'].to(device)
            lr_20m = batch['lr_20m'].to(device)
            hr_10m = batch['hr_10m'].to(device)
            hr_20m = batch['hr_20m'].to(device)
            
            # Forward pass with explicit target sizes
            pred_10m, pred_20m = model(
                lr_10m, lr_20m,
                target_size_10m=(hr_10m.shape[2], hr_10m.shape[3]),
                target_size_20m=(hr_20m.shape[2], hr_20m.shape[3])
            )
            
            # Calculate loss
            loss, _, _ = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
            total_loss += loss.item()
            
            # Calculate PSNR
            total_psnr_10m += calculate_psnr(pred_10m, hr_10m)
            total_psnr_20m += calculate_psnr(pred_20m, hr_20m)
            
            # Per-band PSNR
            psnr_10m = calculate_psnr_per_band(pred_10m, hr_10m, ['Blue', 'Green', 'Red'])
            psnr_20m = calculate_psnr_per_band(pred_20m, hr_20m, ['RedEdge', 'NIR'])
            
            for k, v in psnr_10m.items():
                band_psnr_10m[k] += v
            for k, v in psnr_20m.items():
                band_psnr_20m[k] += v
    
    n_batches = len(dataloader)
    
    # Average per-band PSNR
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
    parser = argparse.ArgumentParser(description='Train 5-band multi-resolution SR model')
    parser.add_argument('--data-dir', type=str, 
                        default='./data/sr_training/5band_test/superdove_5band',
                        help='Path to 5-band dataset')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/5band_multires',
                        help='Directory to save checkpoints')
    parser.add_argument('--num-frames', type=int, default=3,
                        help='Number of temporal frames (default 3, same as train_native_resolution.py)')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("5-BAND MULTI-RESOLUTION SUPER-RESOLUTION TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Temporal frames: {args.num_frames}")
    print(f"  Augmentation: {not args.no_augment}")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    
    # Need to import the class for unpickling
    # The pickle was saved with __main__.TrainingSample5Band, so we need to make it available
    sys.path.insert(0, '/media/eamurciab/TOSHIBA_EXT1/PhD/SR/scripts/preprocessing')
    from convert_cache_to_5band import TrainingSample5Band
    
    # Make the class available under __main__ for pickle to find it
    import types
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
    
    # Create dataset with temporal sequences (same as train_native_resolution.py)
    dataset = FiveBandDataset(
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
    print(f"  HR 20m: {sample_batch['hr_20m'].shape}")
    
    # Create model - consistent with train_native_resolution.py
    model = MultiResolutionSatelliteSR(
        in_channels_10m=3,
        in_channels_20m=2,
        base_channels=64,
        num_encoder_blocks=4,  # Same as train_native_resolution.py
        num_frames=args.num_frames  # Full temporal sequence
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: MultiResolutionSatelliteSR")
    print(f"  Parameters: {n_params:,}")
    print(f"  Temporal frames: {args.num_frames}")
    
    # Loss and optimizer - consistent with train_native_resolution.py
    criterion = MultiResSpectralLoss(
        weight_10m=1.0, weight_20m=1.0,
        alpha_pixel=1.0, alpha_spectral=0.1, alpha_gradient=0.01
    ).to(device)  # Move to device for Sobel buffers
    print(f"\nLoss: MultiResSpectralLoss")
    print(f"  Components: L1 + 0.1×MSE + 0.1×SAM + 0.01×Gradient")
    print(f"  Branch weights: 10m={criterion.weight_10m}, 20m={criterion.weight_20m}")
    
    # Optimizer - AdamW same as train_native_resolution.py
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_loss_10m, train_loss_20m = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler (CosineAnnealingLR doesn't need val_loss)
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} (10m: {train_loss_10m:.4f}, 20m: {train_loss_20m:.4f})")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  PSNR 10m: {val_metrics['psnr_10m']:.2f} dB | PSNR 20m: {val_metrics['psnr_20m']:.2f} dB")
        print(f"  Per-band PSNR (10m): Blue={val_metrics['band_psnr_10m']['Blue']:.2f}, "
              f"Green={val_metrics['band_psnr_10m']['Green']:.2f}, "
              f"Red={val_metrics['band_psnr_10m']['Red']:.2f}")
        print(f"  Per-band PSNR (20m): RedEdge={val_metrics['band_psnr_20m']['RedEdge']:.2f}, "
              f"NIR={val_metrics['band_psnr_20m']['NIR']:.2f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_psnr_10m': val_metrics['psnr_10m'],
                'val_psnr_20m': val_metrics['psnr_20m'],
            }, save_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (loss: {best_val_loss:.4f})")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
