"""
Complete Training Script for Satellite Super Resolution
Uses NumpyTrainingDataGenerator + ImprovedModel.py
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
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from datetime import datetime

# Import your components
from numpy_efficient_pipeline import TensorFlowFreeDataPipeline, NumpyTrainingDataGenerator
from ImprovedModel import ImprovedSatelliteSR, SpectralFidelityLoss
from Auth.AuthGEE import regions  # Your region configurations

def train_sr_model(
    # Model config
    base_channels: int = 64,
    scale_factor: int = 4,
    use_temporal: bool = True,
    
    # Data config
    sequence_length: int = 3,
    tile_size: int = 256,
    batch_size: int = 4,
    
    # Training config
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    
    # Loss weights
    loss_weights: dict = None,
    
    # Paths
    cache_dir: str = "./sr_training_cache",
    checkpoint_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Main training function for satellite super resolution.
    """
    
    print("="*80)
    print("🚀 SATELLITE SUPER RESOLUTION TRAINING")
    print("="*80)
    
    # Create directories
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(log_path / timestamp))
    
    print(f"\n📊 Configuration:")
    print(f"   Device: {device}")
    print(f"   Base channels: {base_channels}")
    print(f"   Scale factor: {scale_factor}")
    print(f"   Temporal: {use_temporal}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # Initialize data pipeline
    print(f"\n🔄 Initializing data pipeline...")
    pipeline = TensorFlowFreeDataPipeline(
        cache_dir=cache_dir,
        tile_size=tile_size,
        overlap=64,
        max_cache_gb=10.0
    )
    
    data_generator = NumpyTrainingDataGenerator(
        pipeline=pipeline,
        batch_size=batch_size,
        normalize=True,
        augment=True
    )
    
    # Initialize model
    print(f"\n🏗️ Creating model...")
    model = ImprovedSatelliteSR(
        base_channels=base_channels,
        scale_factor=scale_factor,
        use_temporal=use_temporal
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    
    # Initialize loss
    if loss_weights is None:
        loss_weights = {
            'lambda_pixel': 1.0,
            'lambda_sam': 0.5,
            'lambda_corr': 0.1,
            'lambda_lr': 0.3,
            'lambda_edge': 0.2
        }
    
    criterion = SpectralFidelityLoss(**loss_weights).to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart interval each time
        eta_min=1e-6
    )
    
    print(f"\n✅ Setup complete. Starting training...")
    print("="*80)
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        
        epoch_losses = {
            'total': 0,
            'pixel': 0,
            'sam': 0,
            'correlation': 0,
            'lr_consistency': 0,
            'edge': 0
        }
        batch_count = 0
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*80}")
        
        # Generate batches
        for lr_batch, hr_batch in data_generator.generate_batches(
            regions, 
            sequence_length=sequence_length,
            pytorch_format=True
        ):
            batch_start = time.time()
            
            # Convert to torch tensors
            lr_tensor = torch.from_numpy(lr_batch).float().to(device)
            hr_tensor = torch.from_numpy(hr_batch).float().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            sr_output = model(lr_tensor)
            
            # Compute loss
            loss_dict = criterion(sr_output, hr_tensor, lr_tensor)
            total_loss = loss_dict['total']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update statistics
            for key in epoch_losses.keys():
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()
                elif key == 'total':
                    epoch_losses[key] += total_loss.item()
            
            batch_count += 1
            global_step += 1
            
            # Log to tensorboard
            if global_step % 10 == 0:
                writer.add_scalar('Loss/total', total_loss.item(), global_step)
                for key, value in loss_dict.items():
                    if key != 'total':
                        writer.add_scalar(f'Loss/{key}', value.item(), global_step)
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)
            
            # Print progress
            batch_time = time.time() - batch_start
            if batch_count % 5 == 0:
                print(f"   Batch {batch_count}: loss={total_loss.item():.4f}, time={batch_time:.2f}s")
        
        # Epoch statistics
        if batch_count > 0:
            for key in epoch_losses.keys():
                epoch_losses[key] /= batch_count
            
            epoch_time = time.time() - epoch_start
            
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   Total loss: {epoch_losses['total']:.4f}")
            print(f"   - Pixel: {epoch_losses['pixel']:.4f}")
            print(f"   - SAM: {epoch_losses['sam']:.4f}")
            print(f"   - Correlation: {epoch_losses['correlation']:.4f}")
            print(f"   - LR consistency: {epoch_losses['lr_consistency']:.4f}")
            print(f"   - Edge: {epoch_losses['edge']:.4f}")
            print(f"   Time: {epoch_time:.1f}s ({batch_count} batches)")
            print(f"   Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if epoch_losses['total'] < best_loss:
                best_loss = epoch_losses['total']
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': best_loss,
                    'config': {
                        'base_channels': base_channels,
                        'scale_factor': scale_factor,
                        'use_temporal': use_temporal
                    }
                }
                best_path = checkpoint_path / "best_model.pth"
                torch.save(checkpoint, best_path)
                print(f"   💾 Saved best model (loss: {best_loss:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': epoch_losses['total'],
                    'config': {
                        'base_channels': base_channels,
                        'scale_factor': scale_factor,
                        'use_temporal': use_temporal
                    }
                }
                epoch_path = checkpoint_path / f"checkpoint_epoch_{epoch+1}.pth"
                torch.save(checkpoint, epoch_path)
                print(f"   💾 Saved checkpoint: {epoch_path}")
        
        # Step scheduler
        scheduler.step()
        
        # Print dataset statistics
        if (epoch + 1) % 5 == 0:
            print(f"\n📈 Dataset Statistics:")
            pipeline.print_stats()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED")
    print("="*80)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final checkpoint: {checkpoint_path / 'best_model.pth'}")
    
    writer.close()
    return model


if __name__ == "__main__":
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🎮 GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ No GPU available, training will be slow!")
    
    # Train model
    trained_model = train_sr_model(
        # Model
        base_channels=64,
        scale_factor=4,
        use_temporal=True,
        
        # Data
        sequence_length=3,
        tile_size=256,
        batch_size=4,
        
        # Training
        num_epochs=50,
        learning_rate=1e-4,
        
        # Paths
        cache_dir="./sr_training_cache",
        checkpoint_dir="./checkpoints",
        log_dir="./logs"
    )
    
    print("\n💡 Next steps:")
    print("   1. Monitor training: tensorboard --logdir=./logs")
    print("   2. Evaluate on test set: python evaluate_sr_model.py")
    print("   3. Generate predictions: python inference_sr_model.py")
