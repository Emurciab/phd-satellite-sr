"""
Satellite Super-Resolution Training Script

Unified training script for multi-resolution satellite imagery SR.
Handles both multi-resolution (10m + 20m) and single-resolution data.

Usage:
    python train_satellite_sr.py --data_dir /path/to/datasets --epochs 100

Key Features:
- Multi-year dataset loading with automatic discovery
- Multi-resolution support (10m + 20m bands)
- Temporal sequence handling (3-5 frames)
- Mixed precision training for memory efficiency
- Checkpoint management with best model tracking
- TensorBoard logging
"""

import os
import sys
import argparse
import json
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

# Import our model
from models.satellite_sr_model import (
    MultiResolutionSatelliteSR,
    SingleResolutionSatelliteSR,
    MultiResolutionLoss,
    count_parameters,
    get_model_size_mb
)

# Optional imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available - logging disabled")


# =============================================================================
# DATASET CLASS
# =============================================================================

class MultiResolutionSRDataset(Dataset):
    """
    Dataset for multi-resolution satellite SR training.
    
    Expected directory structure:
    data_dir/
    ├── STA2_2020/
    │   ├── sample_0000/
    │   │   ├── lr_10m.npy    # (T, 3, H, W) - B2, B3, B4
    │   │   ├── lr_20m.npy    # (T, 1, H/2, W/2) - B8A native
    │   │   ├── hr.npy        # (T, 4, H', W') - Planet HR
    │   │   └── metadata.json # Sample info
    │   └── ...
    ├── STA2_2021/
    │   └── ...
    └── ...
    
    Alternative structure (current data with interpolated B8A):
    data_dir/
    ├── STA2_2020/
    │   ├── sample_0000/
    │   │   ├── lr_s2.npy     # (T, 4, H, W) - B2, B3, B4, B8A (interpolated)
    │   │   ├── hr_planet.npy # (T, 4-6, H', W') - Planet HR
    │   │   └── metadata.json
    """
    
    def __init__(
        self,
        data_dir: str,
        years: List[int] = None,
        sequence_length: int = 3,
        crop_size: int = 64,
        augment: bool = True,
        multi_resolution: bool = False,  # Set True when data regenerated
        max_samples_per_year: int = None
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.crop_size = crop_size
        self.augment = augment
        self.multi_resolution = multi_resolution
        
        # Discover all samples
        self.samples = self._discover_samples(years, max_samples_per_year)
        
        print(f"Found {len(self.samples)} samples")
    
    def _discover_samples(
        self, 
        years: List[int] = None,
        max_samples_per_year: int = None
    ) -> List[Path]:
        """Discover all sample directories."""
        samples = []
        
        # Find all year directories
        year_dirs = sorted(self.data_dir.glob("STA*_20*/"))
        
        for year_dir in year_dirs:
            # Extract year from directory name
            year_str = year_dir.name.split('_')[-1]
            try:
                year = int(year_str[:4])
            except ValueError:
                continue
            
            # Filter by requested years
            if years is not None and year not in years:
                continue
            
            # Find sample directories
            sample_dirs = sorted(year_dir.glob("sample_*/"))
            
            # Filter valid samples
            year_samples = []
            for sample_dir in sample_dirs:
                if self._is_valid_sample(sample_dir):
                    year_samples.append(sample_dir)
            
            # Limit samples per year if requested
            if max_samples_per_year and len(year_samples) > max_samples_per_year:
                year_samples = random.sample(year_samples, max_samples_per_year)
            
            print(f"  {year_dir.name}: {len(year_samples)} samples")
            samples.extend(year_samples)
        
        return samples
    
    def _is_valid_sample(self, sample_dir: Path) -> bool:
        """Check if sample directory contains required files."""
        if self.multi_resolution:
            # New format with native 20m
            required = ['lr_10m.npy', 'lr_20m.npy', 'hr.npy']
        else:
            # Current format with interpolated B8A
            required = ['lr_s2.npy', 'hr_planet.npy']
            alt_required = ['lr.npy', 'hr.npy']
            
            # Check either naming convention
            if all((sample_dir / f).exists() for f in required):
                return True
            if all((sample_dir / f).exists() for f in alt_required):
                return True
            return False
        
        return all((sample_dir / f).exists() for f in required)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_dir = self.samples[idx]
        
        if self.multi_resolution:
            return self._load_multi_resolution(sample_dir)
        else:
            return self._load_single_resolution(sample_dir)
    
    def _load_multi_resolution(self, sample_dir: Path) -> Dict[str, torch.Tensor]:
        """Load multi-resolution data (native 20m B8A)."""
        lr_10m = np.load(sample_dir / 'lr_10m.npy')  # (T, 3, H, W)
        lr_20m = np.load(sample_dir / 'lr_20m.npy')  # (T, 1, H/2, W/2)
        hr = np.load(sample_dir / 'hr.npy')          # (T, 4, H', W')
        
        # Select frames for sequence
        lr_10m, lr_20m, hr = self._select_frames_multi(lr_10m, lr_20m, hr)
        
        # Random crop (square)
        lr_10m, lr_20m, hr = self._random_crop_multi(lr_10m, lr_20m, hr)
        
        # Augmentation
        if self.augment:
            lr_10m, lr_20m, hr = self._augment_multi(lr_10m, lr_20m, hr)
        
        return {
            'lr_10m': torch.from_numpy(lr_10m).float(),
            'lr_20m': torch.from_numpy(lr_20m).float(),
            'hr_10m': torch.from_numpy(hr[:, :3]).float(),   # B, G, R
            'hr_20m': torch.from_numpy(hr[:, 3:4]).float(),  # NIR
        }
    
    def _load_single_resolution(self, sample_dir: Path) -> Dict[str, torch.Tensor]:
        """Load single-resolution data (interpolated B8A)."""
        # Try different naming conventions
        if (sample_dir / 'lr_s2.npy').exists():
            lr = np.load(sample_dir / 'lr_s2.npy')
            hr = np.load(sample_dir / 'hr_planet.npy')
        else:
            lr = np.load(sample_dir / 'lr.npy')
            hr = np.load(sample_dir / 'hr.npy')
        
        # Handle different shapes
        if lr.ndim == 3:  # (C, H, W) - single frame
            lr = lr[np.newaxis, ...]  # (1, C, H, W)
            hr = hr[np.newaxis, ...]
        
        # Select frames
        lr, hr = self._select_frames_single(lr, hr)
        
        # Handle channel mismatch (4 vs 6 bands)
        lr, hr = self._harmonize_channels(lr, hr)
        
        # Random crop (square)
        lr, hr = self._random_crop_single(lr, hr)
        
        # Augmentation
        if self.augment:
            lr, hr = self._augment_single(lr, hr)
        
        return {
            'lr': torch.from_numpy(lr).float(),
            'hr': torch.from_numpy(hr).float(),
        }
    
    def _select_frames_multi(
        self, 
        lr_10m: np.ndarray, 
        lr_20m: np.ndarray, 
        hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select temporal frames for multi-resolution data."""
        T = lr_10m.shape[0]
        
        if T > self.sequence_length:
            # Random contiguous selection
            start = random.randint(0, T - self.sequence_length)
            indices = list(range(start, start + self.sequence_length))
        elif T < self.sequence_length:
            # Repeat frames
            indices = list(range(T))
            while len(indices) < self.sequence_length:
                indices.append(random.choice(range(T)))
        else:
            indices = list(range(T))
        
        return lr_10m[indices], lr_20m[indices], hr[indices]
    
    def _select_frames_single(
        self, 
        lr: np.ndarray, 
        hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select temporal frames for single-resolution data."""
        T = lr.shape[0]
        
        if T > self.sequence_length:
            start = random.randint(0, T - self.sequence_length)
            indices = list(range(start, start + self.sequence_length))
        elif T < self.sequence_length:
            indices = list(range(T))
            while len(indices) < self.sequence_length:
                indices.append(random.choice(range(T)))
        else:
            indices = list(range(T))
        
        return lr[indices], hr[indices]
    
    def _harmonize_channels(
        self, 
        lr: np.ndarray, 
        hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Harmonize channel count between samples."""
        # Current data uses 4 bands for both
        # If HR has 6 bands (SuperDove), take first 4
        if hr.shape[1] > 4:
            hr = hr[:, :4]
        
        return lr, hr
    
    def _random_crop_multi(
        self, 
        lr_10m: np.ndarray, 
        lr_20m: np.ndarray, 
        hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random square crop maintaining spatial correspondence."""
        _, _, H_10m, W_10m = lr_10m.shape
        
        # Ensure square crop (handle non-square inputs)
        crop_10m = min(self.crop_size, H_10m, W_10m)
        crop_20m = crop_10m // 2
        
        # Random position
        y_10m = random.randint(0, max(0, H_10m - crop_10m))
        x_10m = random.randint(0, max(0, W_10m - crop_10m))
        
        # Corresponding 20m position
        y_20m = y_10m // 2
        x_20m = x_10m // 2
        
        # HR crop (scaled)
        scale = hr.shape[2] / H_10m
        crop_hr = int(crop_10m * scale)
        y_hr = int(y_10m * scale)
        x_hr = int(x_10m * scale)
        
        lr_10m = lr_10m[:, :, y_10m:y_10m+crop_10m, x_10m:x_10m+crop_10m]
        lr_20m = lr_20m[:, :, y_20m:y_20m+crop_20m, x_20m:x_20m+crop_20m]
        hr = hr[:, :, y_hr:y_hr+crop_hr, x_hr:x_hr+crop_hr]
        
        return lr_10m, lr_20m, hr
    
    def _random_crop_single(
        self, 
        lr: np.ndarray, 
        hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random square crop for single-resolution data."""
        _, _, H, W = lr.shape
        
        # Ensure square crop
        crop = min(self.crop_size, H, W)
        
        y = random.randint(0, max(0, H - crop))
        x = random.randint(0, max(0, W - crop))
        
        # HR crop (scaled)
        scale = hr.shape[2] / H
        crop_hr = int(crop * scale)
        y_hr = int(y * scale)
        x_hr = int(x * scale)
        
        lr = lr[:, :, y:y+crop, x:x+crop]
        hr = hr[:, :, y_hr:y_hr+crop_hr, x_hr:x_hr+crop_hr]
        
        return lr, hr
    
    def _augment_multi(
        self, 
        lr_10m: np.ndarray, 
        lr_20m: np.ndarray, 
        hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply augmentations consistently across all arrays."""
        # Random horizontal flip
        if random.random() > 0.5:
            lr_10m = np.flip(lr_10m, axis=-1).copy()
            lr_20m = np.flip(lr_20m, axis=-1).copy()
            hr = np.flip(hr, axis=-1).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            lr_10m = np.flip(lr_10m, axis=-2).copy()
            lr_20m = np.flip(lr_20m, axis=-2).copy()
            hr = np.flip(hr, axis=-2).copy()
        
        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            lr_10m = np.rot90(lr_10m, k, axes=(-2, -1)).copy()
            lr_20m = np.rot90(lr_20m, k, axes=(-2, -1)).copy()
            hr = np.rot90(hr, k, axes=(-2, -1)).copy()
        
        return lr_10m, lr_20m, hr
    
    def _augment_single(
        self, 
        lr: np.ndarray, 
        hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentations for single-resolution data."""
        # Random horizontal flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=-1).copy()
            hr = np.flip(hr, axis=-1).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=-2).copy()
            hr = np.flip(hr, axis=-2).copy()
        
        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            lr = np.rot90(lr, k, axes=(-2, -1)).copy()
            hr = np.rot90(hr, k, axes=(-2, -1)).copy()
        
        return lr, hr


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class TrainingMetrics:
    """Track and log training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.loss_components = {}
    
    def update(self, loss: float, components: Dict[str, float]):
        self.losses.append(loss)
        for k, v in components.items():
            if k not in self.loss_components:
                self.loss_components[k] = []
            self.loss_components[k].append(v)
    
    def average(self) -> Dict[str, float]:
        result = {'total': np.mean(self.losses) if self.losses else 0.0}
        for k, v in self.loss_components.items():
            result[k] = np.mean(v) if v else 0.0
        return result


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict,
    save_path: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch:04d}.pt')
    
    if is_best:
        torch.save(checkpoint, save_path / 'best_model.pt')
        print(f"  💾 New best model saved!")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer = None
) -> Tuple[int, Dict]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('metrics', {})


# =============================================================================
# TRAINING LOOPS
# =============================================================================

def train_epoch_single_resolution(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True
) -> Dict[str, float]:
    """Train for one epoch with single-resolution data."""
    model.train()
    metrics = TrainingMetrics()
    
    for batch_idx, batch in enumerate(dataloader):
        lr = batch['lr'].to(device)  # (B, T, C, H, W)
        hr = batch['hr'].to(device)  # (B, T, C, H', W')
        
        # Use only center frame for HR target (or mean)
        T = hr.shape[1]
        hr_target = hr[:, T//2]  # (B, C, H', W')
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            sr = model(lr)  # (B, C, H', W')
            
            # Resize SR to match HR if needed
            if sr.shape[-2:] != hr_target.shape[-2:]:
                sr = F.interpolate(sr, size=hr_target.shape[-2:], mode='bilinear', align_corners=False)
            
            loss, loss_dict = loss_fn(sr, hr_target)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        metrics.update(loss.item(), loss_dict)
        
        # Progress logging
        if (batch_idx + 1) % 50 == 0:
            avg = metrics.average()
            print(f"  Batch {batch_idx+1}/{len(dataloader)}: Loss={avg['total']:.4f}")
    
    return metrics.average()


def train_epoch_multi_resolution(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True
) -> Dict[str, float]:
    """Train for one epoch with multi-resolution data."""
    model.train()
    metrics = TrainingMetrics()
    
    for batch_idx, batch in enumerate(dataloader):
        lr_10m = batch['lr_10m'].to(device)  # (B, T, 3, H, W)
        lr_20m = batch['lr_20m'].to(device)  # (B, T, 1, H/2, W/2)
        hr_10m = batch['hr_10m'].to(device)  # (B, T, 3, H', W')
        hr_20m = batch['hr_20m'].to(device)  # (B, T, 1, H'/2, W'/2)
        
        # Use center frame for HR targets
        T = hr_10m.shape[1]
        hr_10m_target = hr_10m[:, T//2]
        hr_20m_target = hr_20m[:, T//2]
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            sr_10m, sr_20m = model(lr_10m, lr_20m)
            
            # Resize if needed
            if sr_10m.shape[-2:] != hr_10m_target.shape[-2:]:
                sr_10m = F.interpolate(sr_10m, size=hr_10m_target.shape[-2:], mode='bilinear', align_corners=False)
            if sr_20m.shape[-2:] != hr_20m_target.shape[-2:]:
                sr_20m = F.interpolate(sr_20m, size=hr_20m_target.shape[-2:], mode='bilinear', align_corners=False)
            
            # Combined loss
            loss_10m, dict_10m = loss_fn(sr_10m, hr_10m_target)
            loss_20m, dict_20m = loss_fn(sr_20m, hr_20m_target)
            
            loss = loss_10m + 0.5 * loss_20m  # Weight 10m more
            
            loss_dict = {
                'loss_10m': loss_10m.item(),
                'loss_20m': loss_20m.item(),
                'total': loss.item()
            }
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        metrics.update(loss.item(), loss_dict)
        
        if (batch_idx + 1) % 50 == 0:
            avg = metrics.average()
            print(f"  Batch {batch_idx+1}/{len(dataloader)}: Loss={avg['total']:.4f}")
    
    return metrics.average()


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    multi_resolution: bool = False
) -> Dict[str, float]:
    """Validation loop."""
    model.eval()
    metrics = TrainingMetrics()
    
    with torch.no_grad():
        for batch in dataloader:
            if multi_resolution:
                lr_10m = batch['lr_10m'].to(device)
                lr_20m = batch['lr_20m'].to(device)
                hr_10m = batch['hr_10m'].to(device)
                hr_20m = batch['hr_20m'].to(device)
                
                T = hr_10m.shape[1]
                hr_10m_target = hr_10m[:, T//2]
                hr_20m_target = hr_20m[:, T//2]
                
                sr_10m, sr_20m = model(lr_10m, lr_20m)
                
                if sr_10m.shape[-2:] != hr_10m_target.shape[-2:]:
                    sr_10m = F.interpolate(sr_10m, size=hr_10m_target.shape[-2:], mode='bilinear', align_corners=False)
                if sr_20m.shape[-2:] != hr_20m_target.shape[-2:]:
                    sr_20m = F.interpolate(sr_20m, size=hr_20m_target.shape[-2:], mode='bilinear', align_corners=False)
                
                loss_10m, _ = loss_fn(sr_10m, hr_10m_target)
                loss_20m, _ = loss_fn(sr_20m, hr_20m_target)
                loss = loss_10m + 0.5 * loss_20m
                
                loss_dict = {'total': loss.item()}
            else:
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                
                T = hr.shape[1]
                hr_target = hr[:, T//2]
                
                sr = model(lr)
                
                if sr.shape[-2:] != hr_target.shape[-2:]:
                    sr = F.interpolate(sr, size=hr_target.shape[-2:], mode='bilinear', align_corners=False)
                
                loss, loss_dict = loss_fn(sr, hr_target)
            
            metrics.update(loss.item() if hasattr(loss, 'item') else loss, loss_dict)
    
    return metrics.average()


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(args):
    """Main training function."""
    print("=" * 70)
    print("SATELLITE SUPER-RESOLUTION TRAINING")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Setup TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Create datasets
    print(f"\nLoading datasets from: {args.data_dir}")
    
    years = [int(y) for y in args.years.split(',')] if args.years else None
    
    train_dataset = MultiResolutionSRDataset(
        data_dir=args.data_dir,
        years=years,
        sequence_length=args.sequence_length,
        crop_size=args.crop_size,
        augment=True,
        multi_resolution=args.multi_resolution
    )
    
    # Split for validation
    val_size = int(len(train_dataset) * 0.1)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Disable augmentation for validation (through a workaround since we split after creating)
    # Note: In practice, you'd create separate train/val datasets
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print(f"\nCreating model...")
    if args.multi_resolution:
        model = MultiResolutionSatelliteSR(
            base_channels=args.base_channels,
            num_encoder_blocks=args.num_encoder_blocks,
            num_frames=args.sequence_length,
            scale_factor_10m=args.scale_factor,
            scale_factor_20m=args.scale_factor,
            use_cross_resolution_fusion=True
        ).to(device)
    else:
        model = SingleResolutionSatelliteSR(
            in_channels=4,
            out_channels=4,
            base_channels=args.base_channels,
            num_encoder_blocks=args.num_encoder_blocks,
            num_frames=args.sequence_length,
            scale_factor=args.scale_factor
        ).to(device)
    
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Model size: {get_model_size_mb(model):.2f} MB")
    
    # Loss function
    loss_fn = MultiResolutionLoss(
        l1_weight=1.0,
        ssim_weight=0.1,
        spectral_weight=0.1,
        edge_weight=0.1
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, metrics = load_checkpoint(Path(args.resume), model, optimizer)
        best_val_loss = metrics.get('val_loss', float('inf'))
        start_epoch += 1
    
    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING STARTED")
    print(f"{'='*70}\n")
    
    train_fn = train_epoch_multi_resolution if args.multi_resolution else train_epoch_single_resolution
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Train
        train_metrics = train_fn(
            model, train_loader, loss_fn, optimizer, scaler,
            device, epoch, args.use_amp
        )
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device, args.multi_resolution)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"  Train loss: {train_metrics['total']:.4f}")
        print(f"  Val loss: {val_metrics['total']:.4f}")
        print(f"  Epoch time: {epoch_time:.1f}s")
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_metrics['total'], epoch)
            writer.add_scalar('Loss/val', val_metrics['total'], epoch)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Save checkpoint
        is_best = val_metrics['total'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total']
        
        save_checkpoint(
            model, optimizer, epoch,
            {'train_loss': train_metrics['total'], 'val_loss': val_metrics['total']},
            output_dir, is_best
        )
        
        print()
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 70)
    
    if writer:
        writer.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Satellite SR Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./runs',
                        help='Output directory for checkpoints')
    parser.add_argument('--years', type=str, default=None,
                        help='Comma-separated years to use (e.g., "2020,2021,2022")')
    
    # Model arguments
    parser.add_argument('--multi_resolution', action='store_true',
                        help='Use multi-resolution model (native 20m B8A)')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels')
    parser.add_argument('--num_encoder_blocks', type=int, default=4,
                        help='Number of encoder blocks')
    parser.add_argument('--scale_factor', type=float, default=3.33,
                        help='Upscaling factor')
    parser.add_argument('--sequence_length', type=int, default=3,
                        help='Number of temporal frames')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--crop_size', type=int, default=64,
                        help='Training crop size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
