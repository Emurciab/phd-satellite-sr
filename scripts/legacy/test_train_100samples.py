"""
Legacy Test: Train SR Model on 100 Fixed Samples

This is a test/proof-of-concept script using the 100 samples generated
with the stable cache key fix (Dec 4, 2025).

Dataset: data/sr_training/sta2_2020_100samples_fixed/dataset.pkl
- 100 LR-HR pairs from STA-2 region, 2020
- LR: Sentinel-2 (4 bands: B2, B3, B4, B8A at 10m)
- HR: Planet (4 bands at 3m)
- Temporal: 3-frame sequences

Purpose: Verify training pipeline works with correctly paired data.
"""

import os
import sys
import pickle
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

# Add parent directories to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / 'training'))
sys.path.insert(0, str(script_dir.parent / 'preprocessing'))


# Define TrainingSample for pickle loading (must match original definition)
@dataclass
class TrainingSample:
    """Training sample with temporal sequences and multi-resolution LR bands"""
    lr_sequence: np.ndarray  # (T, H, W, C) - 10m bands at native resolution
    hr_sequence: np.ndarray  # (T, H, W, C)
    temporal_dates: List[str]  # Sentinel-2 dates
    planet_dates: List[str]  # Planet dates (may differ by ±7 days)
    region_name: str
    sequence_id: int
    tile_id: int
    tile_bounds: Dict
    # Solar zenith angles for each frame
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None
    # Multi-resolution: 20m bands at native resolution
    lr_20m_sequence: Optional[np.ndarray] = None
    # Frame quality statistics
    frame_quality: Optional[List[Dict]] = None

from models.satellite_sr_model import (
    SingleResolutionSatelliteSR,
    count_parameters,
    get_model_size_mb
)


# =============================================================================
# DATASET
# =============================================================================

class PickleDataset(Dataset):
    """Dataset for loading samples from pickle file."""
    
    def __init__(
        self,
        pickle_path: str,
        crop_size: int = 64,
        augment: bool = True,
        sequence_length: int = 3
    ):
        self.crop_size = crop_size
        self.augment = augment
        self.sequence_length = sequence_length
        
        # Load pickle
        print(f"Loading dataset from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        self.samples = data['samples']
        self.metadata = data.get('config', {})  # Use 'config' key for dataset metadata
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Dataset config: normalized={self.metadata.get('normalized', 'unknown')}")
        
        # Samples are TrainingSample dataclass objects
        sample0 = self.samples[0]
        # TrainingSample has lr_sequence (T, H, W, C), need to convert to (T, C, H, W)
        print(f"Sample LR shape (T,H,W,C): {sample0.lr_sequence.shape}")
        print(f"Sample HR shape (T,H,W,C): {sample0.hr_sequence.shape}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # TrainingSample stores as (T, H, W, C) - convert to (T, C, H, W)
        lr = sample.lr_sequence.copy().transpose(0, 3, 1, 2)  # (T, C, H, W)
        hr = sample.hr_sequence.copy().transpose(0, 3, 1, 2)  # (T, C, H', W')
        
        # Check if data needs normalization (handle both old and new datasets)
        # New datasets have 'normalized' in config, old ones don't
        if not self.metadata.get('normalized', False):
            # Old dataset: normalize from 0-10000 to 0-1
            lr = np.clip(lr / 10000.0, 0, 1).astype(np.float32)
            hr = np.clip(hr / 10000.0, 0, 1).astype(np.float32)
        else:
            # Already normalized, just ensure float32
            lr = lr.astype(np.float32)
            hr = hr.astype(np.float32)
        
        # Select frames if needed
        T = lr.shape[0]
        if T > self.sequence_length:
            start = random.randint(0, T - self.sequence_length)
            lr = lr[start:start + self.sequence_length]
            hr = hr[start:start + self.sequence_length]
        elif T < self.sequence_length:
            # Repeat last frame
            while lr.shape[0] < self.sequence_length:
                lr = np.concatenate([lr, lr[-1:]], axis=0)
                hr = np.concatenate([hr, hr[-1:]], axis=0)
        
        # Random crop
        lr, hr = self._random_crop(lr, hr)
        
        # Augmentation
        if self.augment:
            lr, hr = self._augment(lr, hr)
        
        return {
            'lr': torch.from_numpy(lr).float(),
            'hr': torch.from_numpy(hr).float()
        }
    
    def _random_crop(self, lr, hr):
        """Random crop maintaining spatial correspondence."""
        _, _, H_lr, W_lr = lr.shape
        _, _, H_hr, W_hr = hr.shape
        
        # Scale factor (approximate)
        scale = H_hr / H_lr
        
        # LR crop size
        crop_lr = min(self.crop_size, H_lr, W_lr)
        crop_hr = int(crop_lr * scale)
        
        # Random position
        y_lr = random.randint(0, max(0, H_lr - crop_lr))
        x_lr = random.randint(0, max(0, W_lr - crop_lr))
        
        y_hr = int(y_lr * scale)
        x_hr = int(x_lr * scale)
        
        # Crop
        lr = lr[:, :, y_lr:y_lr+crop_lr, x_lr:x_lr+crop_lr]
        hr = hr[:, :, y_hr:y_hr+crop_hr, x_hr:x_hr+crop_hr]
        
        return lr, hr
    
    def _augment(self, lr, hr):
        """Random augmentation (flip, rotate)."""
        # Random horizontal flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=3).copy()
            hr = np.flip(hr, axis=3).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=2).copy()
            hr = np.flip(hr, axis=2).copy()
        
        # Random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            lr = np.rot90(lr, k, axes=(2, 3)).copy()
            hr = np.rot90(hr, k, axes=(2, 3)).copy()
        
        return lr, hr


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class CombinedLoss(nn.Module):
    """Combined L1 + Perceptual-like loss."""
    
    def __init__(self, l1_weight=1.0, edge_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.edge_weight = edge_weight
        self.l1 = nn.L1Loss()
        
        # Sobel filters for edge loss
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        # L1 loss
        l1_loss = self.l1(pred, target)
        
        # Edge loss (per channel)
        edge_loss = 0
        B, T, C, H, W = pred.shape
        pred_flat = pred.reshape(B * T * C, 1, H, W)
        target_flat = target.reshape(B * T * C, 1, H, W)
        
        pred_edge_x = F.conv2d(pred_flat, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_flat, self.sobel_y, padding=1)
        target_edge_x = F.conv2d(target_flat, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target_flat, self.sobel_y, padding=1)
        
        edge_loss = (
            self.l1(pred_edge_x, target_edge_x) + 
            self.l1(pred_edge_y, target_edge_y)
        )
        
        return self.l1_weight * l1_loss + self.edge_weight * edge_loss


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        lr = batch['lr'].to(device)  # (B, T, C, H, W)
        hr = batch['hr'].to(device)  # (B, T, C, H', W')
        
        optimizer.zero_grad()
        
        with autocast():
            # Forward - model returns (B, C, H', W') after temporal fusion
            sr = model(lr)
            
            # Model collapses temporal dimension, so compare to middle/first HR frame
            # or average of HR frames
            hr_target = hr[:, hr.shape[1]//2]  # Use middle frame as target (B, C, H, W)
            
            # Handle size mismatch
            if sr.shape[-2:] != hr_target.shape[-2:]:
                hr_target = F.interpolate(
                    hr_target,
                    size=sr.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Need (B, T, C, H, W) for loss - add back T dimension
            sr_for_loss = sr.unsqueeze(1)  # (B, 1, C, H, W)
            hr_for_loss = hr_target.unsqueeze(1)  # (B, 1, C, H, W)
            
            loss = criterion(sr_for_loss, hr_for_loss)
        
        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_psnr = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            lr = batch['lr'].to(device)  # (B, T, C, H, W)
            hr = batch['hr'].to(device)  # (B, T, C, H', W')
            
            with autocast():
                sr = model(lr)  # (B, C, H, W)
                
                # Use middle frame as target
                hr_target = hr[:, hr.shape[1]//2]  # (B, C, H, W)
                
                # Handle size mismatch
                if sr.shape[-2:] != hr_target.shape[-2:]:
                    hr_target = F.interpolate(
                        hr_target,
                        size=sr.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Add T dimension for loss
                sr_for_loss = sr.unsqueeze(1)
                hr_for_loss = hr_target.unsqueeze(1)
                
                loss = criterion(sr_for_loss, hr_for_loss)
            
            # PSNR (on 4D tensors without T)
            mse = F.mse_loss(sr, hr_target).item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-8))
            
            total_loss += loss.item()
            total_psnr += psnr
            num_batches += 1
    
    return total_loss / num_batches, total_psnr / num_batches


def main():
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / 'data' / 'sr_training' / 'sta2_2020_100samples_normalized' / 'dataset.pkl'
    output_dir = project_root / 'outputs' / 'legacy_test_100samples_normalized'
    
    # Training params
    config = {
        'epochs': 50,
        'batch_size': 4,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'crop_size': 64,
        'sequence_length': 3,
        'val_split': 0.2,
        'seed': 42,
        # Model
        'base_channels': 32,
        'num_encoder_blocks': 4,  # Match model parameter name
        'num_frames': 3,
        'scale_factor': 1.0,  # LR and HR are same spatial size (256x256)
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Seed
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # ==========================================================================
    # DATASET
    # ==========================================================================
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run generate_sta2_grid_parallel.py first to create the dataset.")
        return
    
    # Load full dataset
    full_dataset = PickleDataset(
        str(dataset_path),
        crop_size=config['crop_size'],
        augment=True,
        sequence_length=config['sequence_length']
    )
    
    # Split train/val
    n_samples = len(full_dataset)
    n_val = int(n_samples * config['val_split'])
    n_train = n_samples - n_val
    
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Disable augmentation for validation
    val_dataset.dataset.augment = False
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # ==========================================================================
    # MODEL
    # ==========================================================================
    
    # Detect input/output channels from data
    sample = full_dataset[0]
    in_channels = sample['lr'].shape[1]  # C dimension
    out_channels = sample['hr'].shape[1]
    
    print(f"Input channels: {in_channels}")
    print(f"Output channels: {out_channels}")
    
    model = SingleResolutionSatelliteSR(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=config['base_channels'],
        num_encoder_blocks=config['num_encoder_blocks'],
        num_frames=config['num_frames'],
        scale_factor=config['scale_factor']
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")
    
    # ==========================================================================
    # TRAINING SETUP
    # ==========================================================================
    
    criterion = CombinedLoss(l1_weight=1.0, edge_weight=0.1).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['lr'] * 0.01
    )
    scaler = GradScaler()
    
    # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    
    best_psnr = 0
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'lr': []}
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(config['epochs']):
        epoch_start = datetime.now()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validate
        val_loss, val_psnr = validate(model, val_loader, criterion, device)
        
        # LR step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)
        history['lr'].append(current_lr)
        
        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'config': config
            }, output_dir / 'best_model.pth')
        
        # Print progress
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PSNR: {val_psnr:.2f} dB | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'config': config
            }, output_dir / f'checkpoint_epoch{epoch+1}.pth')
    
    # ==========================================================================
    # SAVE FINAL
    # ==========================================================================
    
    # Save final model
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config,
        'best_psnr': best_psnr
    }, output_dir / 'final_model.pth')
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Output saved to: {output_dir}")
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # PSNR
        axes[1].plot(history['val_psnr'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].set_title('Validation PSNR')
        axes[1].grid(True)
        
        # LR
        axes[2].plot(history['lr'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=150)
        plt.close()
        
        print(f"Training curves saved to: {output_dir / 'training_curves.png'}")
        
    except ImportError:
        print("Matplotlib not available - skipping plot generation")


if __name__ == '__main__':
    import json
    main()
