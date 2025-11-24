"""
FAST Training Script - Uses only ONE region with minimal tiles
Perfect for rapid iteration and debugging
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
import torch.optim as optim
from pathlib import Path
import time

from numpy_efficient_pipeline import TensorFlowFreeDataPipeline, NumpyTrainingDataGenerator
from ImprovedModel import ImprovedSatelliteSR, SpectralFidelityLoss
from Auth.AuthGEE import regions

def train_fast():
    """Fast training with minimal configuration"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 FAST TRAINING MODE")
    print(f"   Device: {device}")
    
    # MINIMAL configuration
    base_channels = 32
    batch_size = 2
    tile_size = 1024  # Very large tiles = very few tiles!
    sequence_length = 3
    num_epochs = 5
    
    # Use ONLY the first region
    fast_regions = [regions[0]]  # Just STA-2 or whichever is first
    print(f"   Using only region: {fast_regions[0]['name']}")
    
    # Initialize pipeline with larger tiles
    pipeline = TensorFlowFreeDataPipeline(
        cache_dir="./sr_fast_cache",
        tile_size=tile_size,
        overlap=128,  # Larger overlap for large tiles
        max_cache_gb=5.0
    )
    
    data_generator = NumpyTrainingDataGenerator(
        pipeline=pipeline,
        batch_size=batch_size,
        normalize=True,
        augment=False  # Disable augmentation for speed
    )
    
    # Create smaller model
    model = ImprovedSatelliteSR(
        base_channels=base_channels,
        scale_factor=4,
        use_temporal=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,} ({total_params * 4 / 1024**2:.2f} MB)")
    
    # Loss and optimizer
    criterion = SpectralFidelityLoss(
        lambda_pixel=1.0,
        lambda_sam=0.5,
        lambda_corr=0.1,
        lambda_lr=0.3,
        lambda_edge=0.2
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        
        epoch_loss = 0
        batch_count = 0
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Generate batches from single region
        for lr_batch, hr_batch in data_generator.generate_batches(
            fast_regions, 
            sequence_length=sequence_length,
            pytorch_format=True
        ):
            batch_start = time.time()
            
            # Forward pass
            lr_tensor = torch.from_numpy(lr_batch).float().to(device)
            hr_tensor = torch.from_numpy(hr_batch).float().to(device)
            
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
            
            batch_time = time.time() - batch_start
            print(f"  Batch {batch_count}: loss={loss.item():.4f}, time={batch_time:.2f}s")
        
        # Epoch summary
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            epoch_time = time.time() - epoch_start
            print(f"  ✅ Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}, time={epoch_time:.1f}s\n")
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            Path("./fast_checkpoints").mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"./fast_checkpoints/epoch_{epoch+1}.pth")
            print(f"  💾 Saved checkpoint\n")
    
    print(f"\n{'='*60}")
    print(f"✅ FAST TRAINING COMPLETED")
    print(f"{'='*60}")
    
    # Print statistics
    pipeline.print_stats()
    
    return model


if __name__ == "__main__":
    import ee
    
    # Initialize Earth Engine (if not already done)
    try:
        ee.Initialize()
    except:
        from Auth.AuthGEE import ee
        # Should already be initialized
    
    print("\n" + "="*60)
    print("FAST TRAINING CONFIGURATION")
    print("="*60)
    print("✅ Single region only")
    print("✅ Large tiles (1024px) → fewer tiles")
    print("✅ Small model (32 channels)")
    print("✅ Small batches (2)")
    print("✅ No augmentation")
    print("✅ 5 epochs")
    print("="*60 + "\n")
    
    model = train_fast()
    
    print("\n💡 After fast training works, scale up in train_sr_model.py:")
    print("   - Increase base_channels to 64")
    print("   - Use more regions")
    print("   - Enable augmentation")
    print("   - Train for more epochs")
