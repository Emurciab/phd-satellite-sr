#!/usr/bin/env python3
"""
Quick Training Test for Multi-Resolution Satellite SR Model

Tests that the model correctly works with the new multi-resolution sample format:
- lr_sequence: (T, H, W, 3) - 10m bands (B2, B3, B4)
- lr_20m_sequence: (T, H/2, W/2, 1) - 20m band (B8A) at native resolution
- hr_sequence: (T, H_hr, W_hr, 3) - Planet 3m bands

Usage:
    python scripts/training/test_multi_res_training.py --dataset data/datasets/STA34_2020_multi_res/dataset.pkl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/preprocessing'))

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time

# Import TrainingSample class for pickle loading
from scripts.preprocessing.generate_sta2_grid_parallel import TrainingSample

# Import model
from scripts.training.models.satellite_sr_model import (
    MultiResolutionSatelliteSR,
    MultiResolutionLoss,
    count_parameters,
    get_model_size_mb
)


class MultiResolutionDataset(Dataset):
    """
    PyTorch Dataset for multi-resolution satellite SR samples.
    
    Converts from our TrainingSample format to PyTorch tensors.
    """
    def __init__(self, samples, normalize=True, max_value=10000.0):
        """
        Args:
            samples: List of TrainingSample objects
            normalize: Whether to normalize to [0, 1] range
            max_value: Max value for normalization (10000 for S2 surface reflectance)
        """
        self.samples = samples
        self.normalize = normalize
        self.max_value = max_value
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get sequences - shape: (T, H, W, C)
        lr_10m = sample.lr_sequence  # 10m bands
        lr_20m = sample.lr_20m_sequence  # 20m bands (may be None)
        hr = sample.hr_sequence  # Planet HR
        
        # Convert to (T, C, H, W) format for PyTorch
        lr_10m = np.transpose(lr_10m, (0, 3, 1, 2))  # (T, C, H, W)
        hr = np.transpose(hr, (0, 3, 1, 2))
        
        if lr_20m is not None:
            lr_20m = np.transpose(lr_20m, (0, 3, 1, 2))  # (T, C, H/2, W/2)
        else:
            # Create dummy 20m data if not available (for backward compatibility)
            T, _, H, W = lr_10m.shape
            lr_20m = np.zeros((T, 1, H // 2, W // 2), dtype=np.float32)
        
        # Normalize
        if self.normalize:
            lr_10m = lr_10m.astype(np.float32) / self.max_value
            lr_20m = lr_20m.astype(np.float32) / self.max_value
            hr = hr.astype(np.float32) / self.max_value
        
        # Clip to valid range
        lr_10m = np.clip(lr_10m, 0, 1)
        lr_20m = np.clip(lr_20m, 0, 1)
        hr = np.clip(hr, 0, 1)
        
        # Convert to tensors
        lr_10m_tensor = torch.from_numpy(lr_10m)
        lr_20m_tensor = torch.from_numpy(lr_20m)
        hr_tensor = torch.from_numpy(hr)
        
        # Get solar zenith angles if available
        s2_sza = sample.s2_solar_zenith if hasattr(sample, 's2_solar_zenith') else None
        
        return {
            'lr_10m': lr_10m_tensor,
            'lr_20m': lr_20m_tensor,
            'hr': hr_tensor,
            's2_sza': s2_sza,
            'dates': sample.temporal_dates if hasattr(sample, 'temporal_dates') else [],
        }


def quick_training_test(dataset_path: str, num_epochs: int = 3, batch_size: int = 1):
    """
    Quick test to verify model trains correctly with multi-resolution data.
    
    Args:
        dataset_path: Path to dataset.pkl
        num_epochs: Number of epochs to train
        batch_size: Batch size
    """
    print("=" * 70)
    print("MULTI-RESOLUTION SR TRAINING TEST")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    print(f"\n📂 Loading dataset: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    samples = data['samples']
    print(f"   Samples loaded: {len(samples)}")
    
    # Analyze sample shapes
    sample = samples[0]
    print(f"\n📊 Sample structure:")
    print(f"   lr_sequence (10m):   {sample.lr_sequence.shape}")
    print(f"   hr_sequence:         {sample.hr_sequence.shape}")
    if sample.lr_20m_sequence is not None:
        print(f"   lr_20m_sequence:     {sample.lr_20m_sequence.shape}")
    else:
        print(f"   lr_20m_sequence:     None")
    print(f"   dates:               {sample.temporal_dates}")
    
    # Create dataset and dataloader
    dataset = MultiResolutionDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Get a sample batch to determine input/output sizes
    sample_batch = next(iter(dataloader))
    lr_10m = sample_batch['lr_10m']  # (B, T, C, H, W)
    lr_20m = sample_batch['lr_20m']
    hr = sample_batch['hr']
    
    print(f"\n📦 Batch shapes:")
    print(f"   lr_10m: {lr_10m.shape}")
    print(f"   lr_20m: {lr_20m.shape}")
    print(f"   hr:     {hr.shape}")
    
    # Calculate scale factors
    T = lr_10m.shape[1]
    H_lr, W_lr = lr_10m.shape[3], lr_10m.shape[4]
    H_hr, W_hr = hr.shape[3], hr.shape[4]
    scale_factor = H_hr / H_lr
    
    print(f"\n📐 Scale factor: {scale_factor:.2f}x ({H_lr}x{W_lr} → {H_hr}x{W_hr})")
    
    # Create model
    print(f"\n🧠 Creating model...")
    model = MultiResolutionSatelliteSR(
        base_channels=32,  # Smaller for quick test
        num_encoder_blocks=2,  # Fewer blocks for speed
        num_frames=T,
        scale_factor_10m=scale_factor,
        scale_factor_20m=scale_factor,
        use_cross_resolution_fusion=True
    ).to(device)
    
    print(f"   Parameters: {count_parameters(model):,}")
    print(f"   Model size: {get_model_size_mb(model):.2f} MB")
    
    # Loss and optimizer
    loss_fn_10m = MultiResolutionLoss().to(device)
    loss_fn_20m = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    print(f"\n🚀 Starting training ({num_epochs} epochs)...")
    print("-" * 70)
    
    model.train()
    total_start = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            lr_10m = batch['lr_10m'].to(device)  # (B, T, 3, H, W)
            lr_20m = batch['lr_20m'].to(device)  # (B, T, 1, H/2, W/2)
            hr = batch['hr'].to(device)          # (B, T, 3, H_hr, W_hr)
            
            # Target sizes
            target_size_10m = (hr.shape[3], hr.shape[4])
            target_size_20m = (hr.shape[3] // 2, hr.shape[4] // 2)
            
            # Forward pass
            optimizer.zero_grad()
            sr_10m, sr_20m = model(
                lr_10m, lr_20m,
                target_size_10m=target_size_10m,
                target_size_20m=target_size_20m
            )
            
            # Compute loss for 10m bands (compare with HR)
            # Use middle frame as target for simplicity
            hr_target = hr[:, T // 2]  # (B, 3, H_hr, W_hr)
            loss_10m, loss_dict = loss_fn_10m(sr_10m, hr_target)
            
            # Total loss (can add 20m loss if we have 20m HR target)
            loss = loss_10m
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx == 0:
                print(f"   Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)}")
                print(f"      SR 10m output: {sr_10m.shape}")
                print(f"      SR 20m output: {sr_20m.shape}")
                print(f"      Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(dataloader)
        print(f"   Epoch {epoch+1}/{num_epochs} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
    
    total_time = time.time() - total_start
    
    print("-" * 70)
    print(f"\n✅ Training test PASSED!")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Final loss: {avg_loss:.4f}")
    
    # Test inference
    print(f"\n🔍 Testing inference...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(dataloader))
        lr_10m = test_batch['lr_10m'].to(device)
        lr_20m = test_batch['lr_20m'].to(device)
        
        # Target sizes based on HR
        target_size_10m = (hr.shape[3], hr.shape[4])
        target_size_20m = (hr.shape[3] // 2, hr.shape[4] // 2)
        
        sr_10m, sr_20m = model(
            lr_10m, lr_20m,
            target_size_10m=target_size_10m,
            target_size_20m=target_size_20m
        )
        
        print(f"   Input 10m:  {lr_10m.shape}")
        print(f"   Input 20m:  {lr_20m.shape}")
        print(f"   Output 10m: {sr_10m.shape}")
        print(f"   Output 20m: {sr_20m.shape}")
        print(f"   Output range 10m: [{sr_10m.min():.3f}, {sr_10m.max():.3f}]")
        print(f"   Output range 20m: [{sr_20m.min():.3f}, {sr_20m.max():.3f}]")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - Model works with multi-resolution samples!")
    print("=" * 70)
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test multi-resolution SR training')
    parser.add_argument('--dataset', type=str, 
                        default='./data/datasets/STA34_2020_multi_res/dataset.pkl',
                        help='Path to dataset pickle file')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    
    args = parser.parse_args()
    
    quick_training_test(args.dataset, args.epochs, args.batch_size)
