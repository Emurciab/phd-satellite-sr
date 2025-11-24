"""
Train using the fixed pre-downloaded dataset
MUCH FASTER - no GEE downloads during training!
"""

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import torch
import torch.optim as optim
from pathlib import Path
import time
import numpy as np

from create_fixed_dataset import load_fixed_dataset
from ImprovedModel import ImprovedSatelliteSR, SpectralFidelityLoss


def train_with_fixed_dataset(
    dataset_dir="./fixed_dataset",
    base_channels=32,
    batch_size=4,
    num_epochs=50,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Train using pre-downloaded fixed dataset"""
    
    print("="*70)
    print("🚀 TRAINING WITH FIXED DATASET")
    print("="*70)
    print(f"Device: {device}")
    print(f"Base channels: {base_channels}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print("="*70 + "\n")
    
    # Load dataset
    samples, metadata = load_fixed_dataset(dataset_dir)
    print(f"\n✅ Dataset loaded: {len(samples)} samples\n")
    
    # Create model
    model = ImprovedSatelliteSR(
        base_channels=base_channels,
        scale_factor=4,
        use_temporal=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model: {total_params:,} parameters ({total_params * 4 / 1024**2:.2f} MB)\n")
    
    # Loss and optimizer
    criterion = SpectralFidelityLoss(
        lambda_pixel=1.0,
        lambda_sam=0.5,
        lambda_corr=0.1,
        lambda_lr=0.3,
        lambda_edge=0.2
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Prepare batches
    def prepare_batch(sample_list):
        """Convert list of samples to batch tensors"""
        lr_batch = np.stack([s.lr_sequence for s in sample_list], axis=0)
        hr_batch = np.stack([s.hr_sequence for s in sample_list], axis=0)
        
        # Convert to PyTorch format (B, T, C, H, W)
        lr_batch = np.transpose(lr_batch, (0, 1, 4, 2, 3))
        hr_batch = np.transpose(hr_batch, (0, 1, 4, 2, 3))
        
        # Normalize
        lr_batch = np.clip(lr_batch / 10000.0, 0, 1).astype(np.float32)
        hr_batch = np.clip(hr_batch / 10000.0, 0, 1).astype(np.float32)
        
        return (
            torch.from_numpy(lr_batch).to(device),
            torch.from_numpy(hr_batch).to(device)
        )
    
    # Training loop
    checkpoint_dir = Path("./fixed_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_loss = float('inf')
    
    print("="*70)
    print("TRAINING")
    print("="*70 + "\n")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        
        # Shuffle samples
        np.random.shuffle(samples)
        
        epoch_loss = 0
        batch_count = 0
        
        # Create batches
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            
            # Skip incomplete batches
            if len(batch_samples) < batch_size:
                continue
            
            # Prepare batch
            lr_tensor, hr_tensor = prepare_batch(batch_samples)
            
            # Forward pass
            optimizer.zero_grad()
            sr_output = model(lr_tensor)
            
            # Loss
            loss_dict = criterion(sr_output, hr_tensor, lr_tensor)
            loss = loss_dict['total']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        
        # Epoch summary
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:3d}/{num_epochs}: "
              f"loss={avg_loss:.4f}, "
              f"lr={optimizer.param_groups[0]['lr']:.6f}, "
              f"time={epoch_time:.1f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'base_channels': base_channels,
                    'scale_factor': 4,
                    'use_temporal': True
                }
            }, checkpoint_dir / "best_model.pth")
            print(f"  💾 Saved best model (loss: {best_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best model: {checkpoint_dir / 'best_model.pth'}")
    print(f"{'='*70}\n")
    
    return model


if __name__ == "__main__":
    
    if not Path("./fixed_dataset/training_samples.pkl").exists():
        print("❌ Fixed dataset not found!")
        print("   Run: python create_fixed_dataset.py")
        exit(1)
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    # Train
    model = train_with_fixed_dataset(
        dataset_dir="./fixed_dataset",
        base_channels=32,      # Small model for speed
        batch_size=4,          # Adjust based on GPU memory
        num_epochs=20,         # Quick training
        learning_rate=1e-4
    )
    
    print("\n💡 Next steps:")
    print("   1. Check loss curve - is it decreasing?")
    print("   2. If working, create larger dataset: num_samples=200")
    print("   3. Then scale up: base_channels=64, num_epochs=50")
    print("   4. Finally train on full data with train_sr_model.py")
