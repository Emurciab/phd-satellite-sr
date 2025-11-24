"""
Multi-Resolution SR Training for Satellite Imagery
Handles native 10m and 20m bands separately without artificial interpolation

Approach:
- 10m bands (B2,B3,B4): 3m→10m SR (upscale factor 3.33x)
- 20m band (B8A): 3m→20m SR (upscale factor 6.67x)
- Separate models preserve native resolution characteristics
- Joint inference maintains spectral consistency

Technical Rationale:
Interpolating 20m B8A to 10m introduces:
1. Artificial smoothing → model learns upsampled artifacts
2. Spectral inconsistencies → corrupts NIR signals (critical for vegetation)
3. Biased generalization → fails on native 20m data
4. Inflated metrics → PSNR/SSIM misleading on interpolated test sets

Solution: Train separate SR models at native resolutions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../preprocessing"))

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
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MultiResolutionSample:
    """Training sample with separate 10m and 20m bands"""
    # Low resolution inputs
    lr_10m: np.ndarray  # (T, H, W, 3) - B2,B3,B4 at 10m
    lr_20m: np.ndarray  # (T, H/2, W/2, 1) - B8A at native 20m
    
    # High resolution targets
    hr_10m: np.ndarray  # (T, H*3.33, W*3.33, 3) - Planet 3m for RGB
    hr_20m: np.ndarray  # (T, H*3.33/2, W*3.33/2, 1) - Planet 3m downsampled to 6.67m (target for 20m band)
    
    # Metadata
    temporal_dates: List[str]
    planet_dates: List[str]
    region_name: str
    sample_id: int


def prepare_multi_resolution_dataset(dataset_path: str, output_path: str):
    """
    Convert grid-based dataset to multi-resolution format
    
    Args:
        dataset_path: Path to sta2_grid_parallel/dataset.pkl
        output_path: Where to save multi-resolution dataset
    """
    print("="*70)
    print("CONVERTING TO MULTI-RESOLUTION FORMAT")
    print("="*70)
    print(f"Input: {dataset_path}")
    print(f"Output: {output_path}\n")
    
    # Load original dataset
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    samples = data['samples']
    print(f"📊 Original dataset: {len(samples)} samples")
    
    # Convert each sample
    multi_res_samples = []
    
    for idx, sample in enumerate(samples):
        if idx % 50 == 0:
            print(f"   Processing sample {idx}/{len(samples)}...")
        
        # Original: lr_sequence (T, H, W, 4), hr_sequence (T, H', W', 4)
        lr_orig = sample.lr_sequence  # All bands at 10m (B8A was interpolated - WRONG!)
        hr_orig = sample.hr_sequence  # All bands at 3m
        
        T, H_lr, W_lr, _ = lr_orig.shape
        T, H_hr, W_hr, _ = hr_orig.shape
        
        # Split LR into 10m bands (B2,B3,B4) and 20m band (B8A)
        lr_10m = lr_orig[:, :, :, :3]  # (T, H, W, 3) - RGB bands
        lr_20m_interp = lr_orig[:, :, :, 3:4]  # (T, H, W, 1) - NIR (interpolated - need to downsample)
        
        # Downsample LR B8A to native 20m resolution (remove interpolation artifact)
        # Native 20m = half the resolution of 10m
        lr_20m = lr_20m_interp[:, ::2, ::2, :]  # (T, H/2, W/2, 1)
        
        # Split HR into RGB and NIR
        hr_10m = hr_orig[:, :, :, :3]  # (T, H', W', 3) - Target for 10m bands
        hr_nir_full = hr_orig[:, :, :, 3:4]  # (T, H', W', 1) - Full 3m NIR
        
        # Downsample HR NIR to 6.67m (2x downsampling) as target for 20m band
        # This maintains proper scale factor: 20m → 6.67m (3x upscaling)
        hr_20m = hr_nir_full[:, ::2, ::2, :]  # (T, H'/2, W'/2, 1)
        
        multi_res_sample = MultiResolutionSample(
            lr_10m=lr_10m,
            lr_20m=lr_20m,
            hr_10m=hr_10m,
            hr_20m=hr_20m,
            temporal_dates=sample.temporal_dates,
            planet_dates=sample.planet_dates,
            region_name=sample.region_name,
            sample_id=idx
        )
        
        multi_res_samples.append(multi_res_sample)
    
    # Save multi-resolution dataset
    output_data = {
        'samples': multi_res_samples,
        'metadata': {
            'num_samples': len(multi_res_samples),
            'lr_10m_shape': multi_res_samples[0].lr_10m.shape,
            'lr_20m_shape': multi_res_samples[0].lr_20m.shape,
            'hr_10m_shape': multi_res_samples[0].hr_10m.shape,
            'hr_20m_shape': multi_res_samples[0].hr_20m.shape,
            'bands_10m': ['B2', 'B3', 'B4'],
            'bands_20m': ['B8A'],
            'description': 'Multi-resolution dataset with native band resolutions',
            'original_config': data.get('config', {})
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Samples: {len(multi_res_samples)}")
    print(f"   LR 10m shape: {output_data['metadata']['lr_10m_shape']}")
    print(f"   LR 20m shape: {output_data['metadata']['lr_20m_shape']}")
    print(f"   HR 10m shape: {output_data['metadata']['hr_10m_shape']}")
    print(f"   HR 20m shape: {output_data['metadata']['hr_20m_shape']}")
    print(f"   Saved to: {output_path}")
    print("="*70 + "\n")
    
    return output_data


class MultiResolutionDataset(Dataset):
    """PyTorch Dataset for multi-resolution training"""
    
    def __init__(self, samples: List[MultiResolutionSample], band_group: str, normalize=True):
        """
        Args:
            samples: List of MultiResolutionSample objects
            band_group: '10m' or '20m' to select which resolution to train
            normalize: Whether to normalize to [0,1]
        """
        self.samples = samples
        self.band_group = band_group
        self.normalize = normalize
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.band_group == '10m':
            lr = sample.lr_10m  # (T, H, W, 3)
            hr = sample.hr_10m  # (T, H', W', 3)
        elif self.band_group == '20m':
            lr = sample.lr_20m  # (T, H/2, W/2, 1)
            hr = sample.hr_20m  # (T, H'/2, W'/2, 1)
        else:
            raise ValueError(f"Unknown band_group: {self.band_group}")
        
        # Convert to torch tensors and transpose to (T, C, H, W)
        lr_tensor = torch.from_numpy(lr).float().permute(0, 3, 1, 2)
        hr_tensor = torch.from_numpy(hr).float().permute(0, 3, 1, 2)
        
        # Normalize
        if self.normalize:
            lr_tensor = torch.clamp(lr_tensor / 10000.0, 0, 1)
            hr_tensor = torch.clamp(hr_tensor / 10000.0, 0, 1)
        
        return lr_tensor, hr_tensor


class SatelliteSRModel(nn.Module):
    """
    SR model for satellite imagery (can handle different channel counts)
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, 
                 num_blocks=8, upscale_factor=3, sequence_length=3):
        super().__init__()
        self.sequence_length = sequence_length
        self.upscale_factor = upscale_factor
        
        # Feature extraction
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
            frame = x[:, t, :, :, :]
            feat = self.feature_extract(frame)
            feat = self.res_blocks(feat) + feat
            features.append(feat)
        
        # Temporal fusion
        if self.sequence_length > 1 and T > 1:
            combined = torch.cat(features, dim=1)
            attention = self.temporal_attention(combined)
            center_idx = T // 2
            fused = features[center_idx] * attention
        else:
            fused = features[0]
        
        # Upscale
        sr = self.upscale(fused)
        
        return sr


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
        # Pixel-level loss
        pixel_loss = self.l1_loss(pred, target) + 0.1 * self.mse_loss(pred, target)
        
        # Spectral angle loss (for multi-channel)
        if pred.size(1) > 1:
            pred_norm = torch.nn.functional.normalize(pred, dim=1)
            target_norm = torch.nn.functional.normalize(target, dim=1)
            spectral_loss = 1 - torch.mean(torch.sum(pred_norm * target_norm, dim=1))
        else:
            spectral_loss = torch.tensor(0.0, device=pred.device)
        
        # Gradient loss (edge preservation)
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


def train_multi_resolution_models(
    dataset_path: str,
    output_dir: str = "./sr_models_multi_res",
    batch_size: int = 4,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    num_blocks: int = 8,
    base_channels: int = 64,
    device: str = 'cuda',
    train_both: bool = True
):
    """
    Train separate SR models for 10m and 20m bands
    
    Args:
        dataset_path: Path to multi-resolution dataset
        output_dir: Where to save models
        batch_size: Training batch size
        num_epochs: Number of epochs
        learning_rate: Initial LR
        num_blocks: Number of residual blocks
        base_channels: Base channel count
        device: torch device
        train_both: If True, train both models; if False, train only 10m
    """
    
    print("="*70)
    print("🚀 TRAINING MULTI-RESOLUTION SR MODELS")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Strategy: Separate models for 10m and 20m bands")
    print(f"Preserves: Native resolution characteristics")
    print("="*70 + "\n")
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load multi-resolution dataset
    print("📂 Loading multi-resolution dataset...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    samples = data['samples']
    metadata = data['metadata']
    
    print(f"\n📊 Dataset info:")
    print(f"   Total samples: {len(samples)}")
    print(f"   10m bands: {metadata['bands_10m']}")
    print(f"   20m bands: {metadata['bands_20m']}")
    print(f"   LR 10m shape: {metadata['lr_10m_shape']}")
    print(f"   LR 20m shape: {metadata['lr_20m_shape']}")
    print(f"   HR 10m shape: {metadata['hr_10m_shape']}")
    print(f"   HR 20m shape: {metadata['hr_20m_shape']}")
    
    # Train/val split
    val_split = 0.1
    val_size = int(len(samples) * val_split)
    train_samples = samples[:-val_size] if val_size > 0 else samples
    val_samples = samples[-val_size:] if val_size > 0 else []
    
    print(f"\n📊 Train/Val split:")
    print(f"   Training: {len(train_samples)} samples")
    print(f"   Validation: {len(val_samples)} samples")
    
    # Band groups to train
    band_groups = ['10m', '20m'] if train_both else ['10m']
    
    for band_group in band_groups:
        print(f"\n{'='*70}")
        print(f"TRAINING {band_group.upper()} MODEL")
        print(f"{'='*70}\n")
        
        # Create datasets
        train_dataset = MultiResolutionDataset(train_samples, band_group, normalize=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        if len(val_samples) > 0:
            val_dataset = MultiResolutionDataset(val_samples, band_group, normalize=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            val_loader = None
        
        # Get shapes for this band group
        sample_lr, sample_hr = train_dataset[0]
        _, C_in, _, _ = sample_lr.shape  # (T, C, H, W)
        _, C_out, _, _ = sample_hr.shape
        
        # Calculate upscale factor
        H_lr = sample_lr.shape[2]
        H_hr = sample_hr.shape[2]
        upscale_factor = int(np.round(H_hr / H_lr))
        
        print(f"📊 {band_group} Configuration:")
        print(f"   Input channels: {C_in}")
        print(f"   Output channels: {C_out}")
        print(f"   Upscale factor: {upscale_factor}x")
        print(f"   Input size: {sample_lr.shape}")
        print(f"   Output size: {sample_hr.shape}")
        
        # Initialize model
        sequence_length = sample_lr.shape[0]  # T
        model = SatelliteSRModel(
            in_channels=C_in,
            out_channels=C_out,
            base_channels=base_channels,
            num_blocks=num_blocks,
            upscale_factor=upscale_factor,
            sequence_length=sequence_length
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        # Setup training
        criterion = SpectralFidelityLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_components = {'pixel': 0, 'spectral': 0, 'perceptual': 0}
            
            for batch_idx, (lr, hr) in enumerate(train_loader):
                lr, hr = lr.to(device), hr.to(device)
                
                optimizer.zero_grad()
                sr = model(lr)
                
                # Use center frame as target
                center_idx = hr.size(1) // 2
                target = hr[:, center_idx, :, :, :]
                
                loss, components = criterion(sr, target)
                loss.backward()
                optimizer.step()
                
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
                        center_idx = hr.size(1) // 2
                        target = hr[:, center_idx, :, :, :]
                        loss, _ = criterion(sr, target)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
            
            scheduler.step()
            
            # Print progress
            val_str = f"{val_loss:.6f}" if val_loader else "N/A"
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Loss: {train_loss:.6f} "
                  f"(pix: {train_components['pixel']:.4f}, "
                  f"spec: {train_components['spectral']:.4f}, "
                  f"perc: {train_components['perceptual']:.4f}) - "
                  f"Val: {val_str} - "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_loader and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = output_path / f"best_model_{band_group}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'band_group': band_group,
                    'metadata': metadata
                }, best_model_path)
                print(f"   ✅ Saved best {band_group} model (val: {val_loss:.6f})")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = output_path / f"checkpoint_{band_group}_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss if val_loader else None,
                    'band_group': band_group,
                    'metadata': metadata
                }, checkpoint_path)
                print(f"   💾 Saved checkpoint")
        
        # Save final model
        final_model_path = output_path / f"final_model_{band_group}.pth"
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'band_group': band_group,
            'metadata': metadata
        }, final_model_path)
        
        print(f"\n✅ {band_group} model training complete!")
        print(f"   Best val loss: {best_val_loss:.6f}\n")
    
    print("="*70)
    print("✅ MULTI-RESOLUTION TRAINING COMPLETE")
    print("="*70)
    print(f"Models saved to: {output_dir}")
    print(f"  • 10m model: RGB bands (B2,B3,B4)")
    if train_both:
        print(f"  • 20m model: NIR band (B8A)")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multi-resolution SR models')
    parser.add_argument('--dataset-path', required=True,
                       help='Path to sta2_grid_parallel/dataset.pkl')
    parser.add_argument('--output-dir', default='./sr_models_multi_res',
                       help='Output directory for models')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-blocks', type=int, default=8,
                       help='Number of residual blocks')
    parser.add_argument('--base-channels', type=int, default=64,
                       help='Base channel count')
    parser.add_argument('--skip-conversion', action='store_true',
                       help='Skip dataset conversion (if already converted)')
    parser.add_argument('--train-10m-only', action='store_true',
                       help='Train only 10m model (for testing)')
    
    args = parser.parse_args()
    
    # Step 1: Convert dataset to multi-resolution format
    multi_res_path = args.dataset_path.replace('.pkl', '_multi_res.pkl')
    
    if not args.skip_conversion:
        print("STEP 1: Converting to multi-resolution format\n")
        prepare_multi_resolution_dataset(args.dataset_path, multi_res_path)
    else:
        print(f"STEP 1: Skipped (using existing {multi_res_path})\n")
    
    # Step 2: Train models
    print("STEP 2: Training SR models\n")
    train_multi_resolution_models(
        dataset_path=multi_res_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_blocks=args.num_blocks,
        base_channels=args.base_channels,
        train_both=not args.train_10m_only
    )
