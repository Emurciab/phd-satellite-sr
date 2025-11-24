"""
Train SR model using dataset created with chunked downloads
Compatible with numpy_efficient_pipeline output
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pickle
from typing import List

from create_chunked_dataset import load_chunked_dataset


class ChunkedSRDataset(Dataset):
    """PyTorch Dataset wrapper for chunked pipeline samples"""
    
    def __init__(self, samples, normalize=True):
        """
        Args:
            samples: List of TrainingSample objects from pipeline
            normalize: Whether to normalize to [0,1] range
        """
        self.samples = samples
        self.normalize = normalize
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract sequences
        lr = sample.lr_sequence  # Shape: (T, H, W, C)
        hr = sample.hr_sequence  # Shape: (T, H, W, C)
        
        # Convert to torch tensors and transpose to (T, C, H, W)
        lr_tensor = torch.from_numpy(lr).float().permute(0, 3, 1, 2)
        hr_tensor = torch.from_numpy(hr).float().permute(0, 3, 1, 2)
        
        # Normalize to [0, 1] range (reflectance values typically 0-10000)
        if self.normalize:
            lr_tensor = torch.clamp(lr_tensor / 10000.0, 0, 1)
            hr_tensor = torch.clamp(hr_tensor / 10000.0, 0, 1)
        
        return lr_tensor, hr_tensor


class SpectralFidelityLoss(nn.Module):
    """Multi-component loss for satellite SR"""
    
    def __init__(self, alpha_pixel=1.0, alpha_spectral=0.1, alpha_perceptual=0.01):
        super().__init__()
        self.alpha_pixel = alpha_pixel
        self.alpha_spectral = alpha_spectral
        self.alpha_perceptual = alpha_perceptual
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted HR (B, C, H, W)
            target: Ground truth HR (B, C, H, W)
        """
        # Pixel-level loss (L1 + MSE)
        pixel_loss = self.l1_loss(pred, target) + 0.1 * self.mse_loss(pred, target)
        
        # Spectral angle loss (band correlations)
        # Normalize along channel dimension
        pred_norm = torch.nn.functional.normalize(pred, dim=1)
        target_norm = torch.nn.functional.normalize(target, dim=1)
        spectral_loss = 1 - torch.mean(torch.sum(pred_norm * target_norm, dim=1))
        
        # Simple gradient loss (edge preservation)
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        perceptual_loss = self.l1_loss(pred_grad_x, target_grad_x) + \
                         self.l1_loss(pred_grad_y, target_grad_y)
        
        total_loss = (self.alpha_pixel * pixel_loss + 
                     self.alpha_spectral * spectral_loss + 
                     self.alpha_perceptual * perceptual_loss)
        
        return total_loss, {
            'pixel': pixel_loss.item(),
            'spectral': spectral_loss.item(),
            'perceptual': perceptual_loss.item()
        }


class ImprovedSatelliteSR(nn.Module):
    """
    Simplified SR model for satellite imagery
    Uses temporal attention if sequence_length > 1
    """
    
    def __init__(self, in_channels=4, out_channels=4, base_channels=64, 
                 num_blocks=8, upscale_factor=4, sequence_length=3):
        super().__init__()
        self.sequence_length = sequence_length
        self.upscale_factor = upscale_factor
        
        # Feature extraction from each temporal frame
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.extend([
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels)
            ])
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Temporal fusion (if using sequences)
        if sequence_length > 1:
            self.temporal_attention = nn.Sequential(
                nn.Conv2d(base_channels * sequence_length, base_channels, 1),
                nn.Sigmoid()
            )
        
        # Upsampling
        self.upscale = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * (upscale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input LR sequence (B, T, C, H, W)
        Returns:
            SR image (B, C, H*scale, W*scale)
        """
        B, T, C, H, W = x.shape
        
        # Extract features from each frame
        features = []
        for t in range(T):
            frame = x[:, t, :, :, :]  # (B, C, H, W)
            feat = self.feature_extract(frame)
            feat = self.res_blocks(feat) + feat  # Residual connection
            features.append(feat)
        
        # Temporal fusion
        if self.sequence_length > 1 and T > 1:
            # Concatenate temporal features
            combined = torch.cat(features, dim=1)  # (B, T*C, H, W)
            # Apply attention
            attention = self.temporal_attention(combined)
            # Weight the center frame
            center_idx = T // 2
            fused = features[center_idx] * attention
        else:
            fused = features[0]  # Single frame
        
        # Upscale
        sr = self.upscale(fused)
        
        return sr


def train_with_chunked_dataset(
    dataset_dir="./chunked_sr_dataset",
    output_dir="./sr_models",
    batch_size=4,
    num_epochs=50,
    learning_rate=1e-4,
    num_blocks=8,
    base_channels=64,
    device='cuda'
):
    """
    Train SR model using chunked dataset
    
    Args:
        dataset_dir: Path to dataset created with create_chunked_dataset.py
        output_dir: Where to save trained models
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: torch device (None = auto-detect)
    """
    
    print("="*70)
    print("🚀 TRAINING SR MODEL WITH CHUNKED DATASET")
    print("="*70)
    print(f"Dataset: {dataset_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("="*70 + "\n")
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load dataset
    print("📂 Loading dataset...")
    samples, metadata = load_chunked_dataset(dataset_dir)
    
    print(f"\n📊 Dataset info:")
    print(f"   Total samples: {len(samples)}")
    print(f"   LR shape: {metadata['lr_shape']}")
    print(f"   HR shape: {metadata['hr_shape']}")
    
    # Split train/val
    val_split = 0.1
    val_size = int(len(samples) * val_split)
    train_samples = samples[:-val_size] if val_size > 0 else samples
    val_samples = samples[-val_size:] if val_size > 0 else []
    
    print(f"\n📊 Train/Val split:")
    print(f"   Training: {len(train_samples)} samples")
    print(f"   Validation: {len(val_samples)} samples")
    
    # Create datasets
    train_dataset = ChunkedSRDataset(train_samples, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    if len(val_samples) > 0:
        val_dataset = ChunkedSRDataset(val_samples, normalize=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        val_loader = None
    
    # Initialize model
    print("\n🏗️  Building model...")
    sequence_length = metadata['lr_shape'][0]  # T from (T, H, W, C)
    in_channels = metadata['lr_shape'][-1]     # C
    
    # Calculate upscale factor from data shapes
    lr_size = metadata['lr_shape'][1]  # H from (T, H, W, C)
    hr_size = metadata['hr_shape'][1]  # H from (T, H, W, C)
    upscale_factor = hr_size // lr_size  # 256 // 64 = 4
    
    model = ImprovedSatelliteSR(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        num_blocks=num_blocks,
        upscale_factor=upscale_factor,
        sequence_length=sequence_length
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Architecture: {num_blocks} blocks, {base_channels} base channels")
    
    # Setup training
    criterion = SpectralFidelityLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_components = {'pixel': 0, 'spectral': 0, 'perceptual': 0}
        
        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            sr = model(lr)
            
            # Compute loss
            loss, components = criterion(sr, hr[:, hr.size(1)//2, :, :, :])  # Use center frame
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            for k, v in components.items():
                train_components[k] += v
        
        train_loss /= len(train_loader)
        for k in train_components:
            train_components[k] /= len(train_loader)
        
        # Validation phase
        val_loss = 0.0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for lr, hr in val_loader:
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)
                    loss, _ = criterion(sr, hr[:, hr.size(1)//2, :, :, :])
                    val_loss += loss.item()
            val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        val_loss_str = f"{val_loss:.6f}" if val_loader else "N/A"
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.6f} "
              f"(pixel: {train_components['pixel']:.4f}, "
              f"spectral: {train_components['spectral']:.4f}, "
              f"perceptual: {train_components['perceptual']:.4f}) - "
              f"Val Loss: {val_loss_str} - "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loader and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_path / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metadata': metadata
            }, best_model_path)
            print(f"   ✅ Saved best model (val_loss: {val_loss:.6f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_path / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss if val_loader else None,
                'metadata': metadata
            }, checkpoint_path)
            print(f"   💾 Saved checkpoint")
    
    # Save final model
    final_model_path = output_path / "final_model.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata
    }, final_model_path)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Models saved to: {output_dir}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SR model with chunked dataset')
    parser.add_argument('--dataset-dir', default='./chunked_sr_dataset',
                       help='Path to chunked dataset')
    parser.add_argument('--output-dir', default='./sr_models',
                       help='Output directory for models')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-blocks', type=int, default=8,
                       help='Number of residual blocks in the model')
    parser.add_argument('--base-channels', type=int, default=64,
                       help='Number of base channels in the model')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with 2 epochs')
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("\n🧪 QUICK TEST MODE: 2 epochs\n")
        train_with_chunked_dataset(
            dataset_dir="./quick_test_dataset",
            output_dir="./quick_test_models",
            batch_size=2,
            num_epochs=2,
            learning_rate=args.lr
        )
    else:
        train_with_chunked_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            num_blocks=args.num_blocks,
            base_channels=args.base_channels
        )
