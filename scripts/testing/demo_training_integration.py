"""
Quick demonstration of training integration
Uses synthetic data to show the full pipeline works
"""

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass

print("="*70)
print("🧪 TRAINING INTEGRATION DEMO WITH SYNTHETIC DATA")
print("="*70)
print("This demonstrates the full pipeline without waiting for real data\n")

# Define TrainingSample (matches pipeline output)
@dataclass
class TrainingSample:
    """Training sample with LR and HR sequences"""
    lr_sequence: np.ndarray  # (T, H, W, C)
    hr_sequence: np.ndarray  # (T, H, W, C)
    metadata: dict

print("Step 1: Create synthetic training samples")
print("-" * 70)

# Create realistic synthetic data
# LR: 3 frames × 256×256 × 4 bands (Sentinel-2 @ 10m)
# HR: 3 frames × 768×768 × 4 bands (Planet @ 3m, 3x upscale)

num_samples = 5
sequence_length = 3
tile_size_lr = 64  # Small for quick test
tile_size_hr = tile_size_lr * 3  # 3x upscale

samples = []

for i in range(num_samples):
    # Generate realistic reflectance patterns
    # Bands: Blue, Green, Red, NIR
    
    # Create spatial patterns (simulates vegetation, water, urban)
    y, x = np.ogrid[:tile_size_hr, :tile_size_hr]
    
    # Temporal sequences (slight variations between frames)
    lr_seq = []
    hr_seq = []
    
    for t in range(sequence_length):
        # Add temporal variation
        temporal_factor = 0.9 + 0.2 * (t / sequence_length)
        
        # Generate HR bands with realistic values
        hr_blue = (1000 + 500 * np.sin(y/20) + 200 * np.random.rand(tile_size_hr, tile_size_hr)) * temporal_factor
        hr_green = (1500 + 700 * np.sin(y/20) + 300 * np.random.rand(tile_size_hr, tile_size_hr)) * temporal_factor
        hr_red = (1800 + 900 * np.sin(y/20) + 400 * np.random.rand(tile_size_hr, tile_size_hr)) * temporal_factor
        hr_nir = (3000 + 1500 * np.sin(y/20) + 600 * np.random.rand(tile_size_hr, tile_size_hr)) * temporal_factor
        
        hr_frame = np.stack([hr_blue, hr_green, hr_red, hr_nir], axis=-1).astype(np.float32)
        
        # Downsample for LR (simulates Sentinel-2) - simple averaging
        lr_frame = hr_frame[::3, ::3, :].astype(np.float32)
        
        # Add noise to LR
        lr_frame += np.random.normal(0, 50, lr_frame.shape).astype(np.float32)
        
        lr_seq.append(lr_frame)
        hr_seq.append(hr_frame)
    
    # Create sample
    sample = TrainingSample(
        lr_sequence=np.stack(lr_seq),  # (T, H, W, C)
        hr_sequence=np.stack(hr_seq),  # (T, H, W, C)
        metadata={
            'sample_id': i,
            'region': 'synthetic',
            'dates': [f'2024-01-{i+1:02d}' for i in range(sequence_length)]
        }
    )
    
    samples.append(sample)
    
    print(f"   ✅ Sample {i+1}/{num_samples}: LR {sample.lr_sequence.shape}, HR {sample.hr_sequence.shape}")

print(f"\n✅ Created {len(samples)} synthetic samples")

print("\nStep 2: Save as chunked dataset format")
print("-" * 70)

output_dir = Path("./demo_dataset")
output_dir.mkdir(exist_ok=True)

# Create metadata
metadata = {
    'regions': [{'name': 'synthetic', 'sample_count': num_samples}],
    'total_samples': num_samples,
    'tile_size': tile_size_lr,
    'sequence_length': sequence_length,
    'lr_shape': samples[0].lr_sequence.shape,
    'hr_shape': samples[0].hr_sequence.shape,
    'creation_date': '2024-demo'
}

# Save
samples_file = output_dir / "training_samples.pkl"
metadata_file = output_dir / "metadata.pkl"

with open(samples_file, 'wb') as f:
    pickle.dump(samples, f)
with open(metadata_file, 'wb') as f:
    pickle.dump(metadata, f)

print(f"   💾 Saved to {output_dir}")
print(f"   - training_samples.pkl")
print(f"   - metadata.pkl")

print("\nStep 3: Validate dataset")
print("-" * 70)

# Reload and validate
with open(samples_file, 'rb') as f:
    loaded_samples = pickle.load(f)
with open(metadata_file, 'rb') as f:
    loaded_metadata = pickle.load(f)

print(f"   ✅ Loaded {len(loaded_samples)} samples")
print(f"   LR shape: {loaded_metadata['lr_shape']}")
print(f"   HR shape: {loaded_metadata['hr_shape']}")

# Quick validation
sample = loaded_samples[0]
lr = sample.lr_sequence
hr = sample.hr_sequence

print(f"\n   Sample 0 statistics:")
print(f"   LR: min={lr.min():.1f}, max={lr.max():.1f}, mean={lr.mean():.1f}")
print(f"   HR: min={hr.min():.1f}, max={hr.max():.1f}, mean={hr.mean():.1f}")

print("\nStep 4: Test PyTorch dataset wrapper")
print("-" * 70)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    class ChunkedSRDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            lr = torch.from_numpy(sample.lr_sequence).float().permute(0, 3, 1, 2)
            hr = torch.from_numpy(sample.hr_sequence).float().permute(0, 3, 1, 2)
            # Normalize
            lr = torch.clamp(lr / 10000.0, 0, 1)
            hr = torch.clamp(hr / 10000.0, 0, 1)
            return lr, hr
    
    dataset = ChunkedSRDataset(loaded_samples)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Test batch
    lr_batch, hr_batch = next(iter(dataloader))
    
    print(f"   ✅ PyTorch Dataset works!")
    print(f"   Batch LR shape: {lr_batch.shape}")  # (B, T, C, H, W)
    print(f"   Batch HR shape: {hr_batch.shape}")  # (B, T, C, H, W)
    print(f"   LR range: [{lr_batch.min():.3f}, {lr_batch.max():.3f}]")
    print(f"   HR range: [{hr_batch.min():.3f}, {hr_batch.max():.3f}]")
    
    print("\nStep 5: Test model forward pass")
    print("-" * 70)
    
    # Simple test model
    class SimpleSR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(4, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 4*9, 3, padding=1),
                torch.nn.PixelShuffle(3)
            )
        
        def forward(self, x):
            B, T, C, H, W = x.shape
            # Use center frame
            x = x[:, T//2, :, :, :]
            return self.conv(x)
    
    model = SimpleSR()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"   🖥️  Device: {device}")
    
    # Forward pass
    lr_batch = lr_batch.to(device)
    hr_batch = hr_batch.to(device)
    
    with torch.no_grad():
        sr_batch = model(lr_batch)
    
    print(f"   ✅ Model forward pass works!")
    print(f"   Input LR: {lr_batch.shape}")
    print(f"   Output SR: {sr_batch.shape}")
    print(f"   Target HR: {hr_batch[:, hr_batch.size(1)//2].shape}")
    
    # Check upscale factor
    expected_size = lr_batch.shape[-1] * 3
    actual_size = sr_batch.shape[-1]
    if actual_size == expected_size:
        print(f"   ✅ Upscale factor correct: 3x")
    else:
        print(f"   ⚠️  Upscale mismatch: {actual_size} vs {expected_size}")
    
    print("\nStep 6: Test loss computation")
    print("-" * 70)
    
    # Simple loss
    hr_center = hr_batch[:, hr_batch.size(1)//2, :, :, :]
    loss_fn = torch.nn.L1Loss()
    loss = loss_fn(sr_batch, hr_center)
    
    print(f"   ✅ Loss computation works!")
    print(f"   L1 Loss: {loss.item():.6f}")
    
    print("\n" + "="*70)
    print("✅ TRAINING INTEGRATION DEMO COMPLETE")
    print("="*70)
    print("\n📋 Summary:")
    print("   ✅ Synthetic dataset created (5 samples)")
    print("   ✅ Saved in chunked dataset format")
    print("   ✅ PyTorch Dataset wrapper works")
    print("   ✅ Model forward pass successful")
    print("   ✅ Loss computation successful")
    print(f"   ✅ Using device: {device}")
    print("\n💡 Next steps:")
    print("   1. Wait for integration test to complete with REAL data")
    print("   2. Run: python create_chunked_dataset.py --quick-test")
    print("   3. Run: python train_with_chunked_dataset.py --quick-test")
    print("   4. Generate production dataset")
    print("\n📁 Demo dataset saved to: ./demo_dataset/")
    print("="*70 + "\n")

except ImportError as e:
    print(f"   ⚠️  PyTorch not available: {e}")
    print("   This is OK - the dataset format is validated")
    print("   Install PyTorch to test training: pip install torch")
    
    print("\n" + "="*70)
    print("✅ DATASET FORMAT VALIDATED")
    print("="*70)
    print("\n📋 Summary:")
    print("   ✅ Synthetic dataset created (5 samples)")
    print("   ✅ Saved in chunked dataset format")
    print("   ✅ Dataset can be loaded successfully")
    print("   ⚠️  PyTorch not installed (needed for training)")
    print("\n💡 Next steps:")
    print("   1. Install PyTorch: pip install torch")
    print("   2. Wait for integration test with REAL data")
    print("   3. Run training scripts")
    print("\n📁 Demo dataset saved to: ./demo_dataset/")
    print("="*70 + "\n")
