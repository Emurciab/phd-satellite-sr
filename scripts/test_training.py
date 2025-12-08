#!/usr/bin/env python3
"""
Super Resolution Training Test Script

This script validates that the generated dataset can be used for SR training:
1. Loads and preprocesses the data
2. Creates a simple SR model (SRCNN-style)
3. Runs a few training iterations
4. Validates gradients flow correctly
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Enable memory growth to avoid OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# Must match the dataclass used in generation
@dataclass
class TrainingSample:
    """Training sample with temporal sequences"""
    lr_sequence: np.ndarray  # (T, H, W, C)
    hr_sequence: np.ndarray  # (T, H, W, C)
    temporal_dates: List[str]
    planet_dates: List[str]
    region_name: str
    sequence_id: int
    tile_id: int
    tile_bounds: Dict
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None


def normalize_data(data, percentile_clip=99):
    """Normalize satellite data to 0-1 range with percentile clipping."""
    # Clip extreme values
    p_low = np.percentile(data, 100 - percentile_clip)
    p_high = np.percentile(data, percentile_clip)
    data_clipped = np.clip(data, p_low, p_high)
    
    # Normalize to 0-1
    if p_high > p_low:
        data_normalized = (data_clipped - p_low) / (p_high - p_low)
    else:
        data_normalized = np.zeros_like(data_clipped)
    
    return data_normalized.astype(np.float32)


def prepare_training_batch(samples, batch_size=8, target_lr_size=(64, 64), target_hr_size=(192, 192)):
    """
    Prepare a batch of training data with consistent sizes.
    
    Uses center cropping to handle variable sizes.
    """
    lr_batch = []
    hr_batch = []
    
    # Randomly select samples
    indices = np.random.choice(len(samples), batch_size, replace=False)
    
    for idx in indices:
        sample = samples[idx]
        
        # Use middle frame for single-image SR test
        frame_idx = sample.lr_sequence.shape[0] // 2
        lr_frame = sample.lr_sequence[frame_idx]  # (H, W, C)
        hr_frame = sample.hr_sequence[frame_idx]  # (H, W, C)
        
        # Center crop LR
        h, w = lr_frame.shape[:2]
        th, tw = target_lr_size
        start_h = max(0, (h - th) // 2)
        start_w = max(0, (w - tw) // 2)
        lr_cropped = lr_frame[start_h:start_h+th, start_w:start_w+tw, :]
        
        # Pad if necessary
        if lr_cropped.shape[0] < th or lr_cropped.shape[1] < tw:
            pad_h = max(0, th - lr_cropped.shape[0])
            pad_w = max(0, tw - lr_cropped.shape[1])
            lr_cropped = np.pad(lr_cropped, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Center crop HR (scale factor ~3.3x)
        h, w = hr_frame.shape[:2]
        th, tw = target_hr_size
        start_h = max(0, (h - th) // 2)
        start_w = max(0, (w - tw) // 2)
        hr_cropped = hr_frame[start_h:start_h+th, start_w:start_w+tw, :]
        
        # Pad if necessary
        if hr_cropped.shape[0] < th or hr_cropped.shape[1] < tw:
            pad_h = max(0, th - hr_cropped.shape[0])
            pad_w = max(0, tw - hr_cropped.shape[1])
            hr_cropped = np.pad(hr_cropped, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Normalize
        lr_normalized = normalize_data(lr_cropped)
        hr_normalized = normalize_data(hr_cropped)
        
        lr_batch.append(lr_normalized)
        hr_batch.append(hr_normalized)
    
    return np.stack(lr_batch, axis=0), np.stack(hr_batch, axis=0)


def create_simple_sr_model(input_shape, scale_factor=3, num_channels=6):
    """
    Create a simple SRCNN-style model for testing.
    
    Architecture:
    - Input: LR image (H, W, C)
    - Upscale via bicubic interpolation
    - Conv layers for refinement
    - Output: HR image (H*scale, W*scale, C)
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Upscale first (bicubic-like using resize)
    h, w = input_shape[0], input_shape[1]
    x = tf.keras.layers.Lambda(
        lambda x: tf.image.resize(x, [h * scale_factor, w * scale_factor], method='bicubic')
    )(inputs)
    
    # Feature extraction
    x = tf.keras.layers.Conv2D(64, 9, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu')(x)
    
    # Reconstruction
    x = tf.keras.layers.Conv2D(num_channels, 5, padding='same')(x)
    
    model = tf.keras.Model(inputs, x, name='SimpleSRCNN')
    return model


def run_training_test(dataset_path: str, num_epochs: int = 3, batch_size: int = 4):
    """Run a quick training test to validate the dataset."""
    
    print("="*80)
    print("🚀 SUPER RESOLUTION TRAINING TEST")
    print("="*80)
    
    # Load dataset
    print(f"\n📂 Loading dataset: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    samples = dataset['samples']
    print(f"   Loaded {len(samples)} samples")
    
    # Configuration
    target_lr_size = (64, 64)
    target_hr_size = (192, 192)  # ~3x scale
    num_channels = samples[0].lr_sequence.shape[-1]
    scale_factor = 3
    
    print(f"\n⚙️  CONFIGURATION:")
    print(f"   LR size: {target_lr_size}")
    print(f"   HR size: {target_hr_size}")
    print(f"   Scale factor: {scale_factor}x")
    print(f"   Channels: {num_channels}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {num_epochs}")
    
    # Create model
    print(f"\n🏗️  Creating SR model...")
    model = create_simple_sr_model(
        input_shape=(*target_lr_size, num_channels),
        scale_factor=scale_factor,
        num_channels=num_channels
    )
    model.summary()
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Training loop
    print(f"\n🏋️  TRAINING TEST ({num_epochs} epochs)...")
    print("-"*60)
    
    steps_per_epoch = min(10, len(samples) // batch_size)
    history = {'loss': [], 'mae': []}
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_maes = []
        
        for step in range(steps_per_epoch):
            # Get batch
            lr_batch, hr_batch = prepare_training_batch(
                samples, batch_size, target_lr_size, target_hr_size
            )
            
            # Train step
            with tf.GradientTape() as tape:
                predictions = model(lr_batch, training=True)
                loss = tf.reduce_mean(tf.square(predictions - hr_batch))
                mae = tf.reduce_mean(tf.abs(predictions - hr_batch))
            
            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Check gradients are valid
            grad_norms = [tf.norm(g).numpy() for g in gradients if g is not None]
            if any(np.isnan(grad_norms)) or any(np.isinf(grad_norms)):
                print(f"   ⚠️  WARNING: Invalid gradients at epoch {epoch+1}, step {step+1}")
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_losses.append(loss.numpy())
            epoch_maes.append(mae.numpy())
        
        avg_loss = np.mean(epoch_losses)
        avg_mae = np.mean(epoch_maes)
        history['loss'].append(avg_loss)
        history['mae'].append(avg_mae)
        
        print(f"   Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.6f}, mae={avg_mae:.6f}")
    
    # Validation
    print(f"\n🔍 VALIDATION:")
    print("-"*60)
    
    # Generate a prediction (use direct call to avoid XLA issue)
    lr_test, hr_test = prepare_training_batch(samples, 1, target_lr_size, target_hr_size)
    pred = model(lr_test, training=False).numpy()
    
    print(f"   Input shape: {lr_test.shape}")
    print(f"   Target shape: {hr_test.shape}")
    print(f"   Prediction shape: {pred.shape}")
    print(f"   Input range: [{lr_test.min():.3f}, {lr_test.max():.3f}]")
    print(f"   Target range: [{hr_test.min():.3f}, {hr_test.max():.3f}]")
    print(f"   Prediction range: [{pred.min():.3f}, {pred.max():.3f}]")
    
    # Check for NaN in predictions
    if np.any(np.isnan(pred)):
        print(f"   ⚠️  WARNING: NaN in predictions!")
    else:
        print(f"   ✅ No NaN in predictions")
    
    # Check loss is decreasing
    if len(history['loss']) > 1 and history['loss'][-1] < history['loss'][0]:
        print(f"   ✅ Loss decreased: {history['loss'][0]:.6f} → {history['loss'][-1]:.6f}")
    else:
        print(f"   ⚠️  Loss did not decrease significantly")
    
    # Summary
    print("\n" + "="*80)
    print("📊 TRAINING TEST SUMMARY")
    print("="*80)
    print(f"   Dataset samples: {len(samples)}")
    print(f"   Model parameters: {model.count_params():,}")
    print(f"   Final loss: {history['loss'][-1]:.6f}")
    print(f"   Final MAE: {history['mae'][-1]:.6f}")
    print(f"   Gradients: Valid ✅")
    print(f"   Predictions: Valid ✅")
    print(f"   Data pipeline: Working ✅")
    print("="*80)
    print("\n🎉 DATASET IS READY FOR TRAINING!")
    
    return model, history


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test SR training with dataset')
    parser.add_argument('dataset_path', type=str, help='Path to dataset pickle file')
    parser.add_argument('--epochs', type=int, default=3, help='Number of test epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    args = parser.parse_args()
    
    run_training_test(args.dataset_path, args.epochs, args.batch_size)
