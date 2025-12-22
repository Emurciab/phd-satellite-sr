#!/usr/bin/env python3
"""
Full Dataset Training Script for Satellite Super Resolution

Trains on all combined STA-2 datasets (2020-2024):
- 2,875 total samples
- 6 spectral bands (B1, B2, B3, B4, B5, B8A)
- 3 temporal frames per sequence
- ~3.3x scale factor (Planet 3m → Sentinel-2 10m)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
import time
from datetime import datetime
import json
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
    print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")


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


class MultiYearDatasetLoader:
    """Load and combine multiple year datasets."""
    
    def __init__(self, base_dir: str, years: List[str] = None):
        self.base_dir = Path(base_dir)
        self.years = years or ['2020', '2021', '2022', '2023', '2024']
        self.samples = []
        self.metadata = {}
        self.max_channels = 0  # Track max channels across all samples
        
    def load_all(self):
        """Load all datasets and combine samples."""
        print(f"\n📂 Loading datasets from {self.base_dir}")
        print("-" * 60)
        
        total_samples = 0
        channel_counts = {}  # Track distribution of channel counts
        
        for year in self.years:
            year_dir = self.base_dir / f"STA2_{year}"
            # Check multiple possible filenames
            possible_names = ["STA-2_dataset.pkl", "training_dataset.pkl", "dataset.pkl"]
            dataset_file = None
            
            for name in possible_names:
                candidate = year_dir / name
                if candidate.exists():
                    dataset_file = candidate
                    break
            
            if dataset_file is None:
                print(f"   ⚠️  {year}: Dataset not found in {year_dir}")
                continue
            
            with open(dataset_file, 'rb') as f:
                dataset = pickle.load(f)
            
            year_samples = dataset['samples']
            
            # Track channel distribution
            for sample in year_samples:
                nc = sample.lr_sequence.shape[-1]
                channel_counts[nc] = channel_counts.get(nc, 0) + 1
                if nc > self.max_channels:
                    self.max_channels = nc
            
            self.samples.extend(year_samples)
            
            # Track metadata
            self.metadata[year] = {
                'num_samples': len(year_samples),
                'metadata': dataset.get('metadata', {})
            }
            
            total_samples += len(year_samples)
            print(f"   ✅ {year}: {len(year_samples)} samples loaded")
        
        print("-" * 60)
        print(f"   📊 Total: {total_samples} samples from {len(self.metadata)} years")
        
        # Report channel distribution
        if channel_counts:
            print(f"\n   🔬 Channel distribution:")
            for nc, count in sorted(channel_counts.items()):
                pct = 100 * count / total_samples
                print(f"      {nc}-channel: {count} samples ({pct:.1f}%)")
            print(f"   Max channels: {self.max_channels} (will pad smaller samples)")
        
        return self.samples
    
    def get_train_val_split(self, val_ratio: float = 0.1, shuffle: bool = True, seed: int = 42):
        """Split samples into train and validation sets."""
        if shuffle:
            np.random.seed(seed)
            indices = np.random.permutation(len(self.samples))
        else:
            indices = np.arange(len(self.samples))
        
        val_size = int(len(self.samples) * val_ratio)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_samples = [self.samples[i] for i in train_indices]
        val_samples = [self.samples[i] for i in val_indices]
        
        print(f"\n📊 Train/Val Split:")
        print(f"   Train: {len(train_samples)} samples")
        print(f"   Val: {len(val_samples)} samples")
        
        return train_samples, val_samples


def normalize_data(data, percentile_clip=99):
    """Normalize satellite data to 0-1 range with percentile clipping."""
    p_low = np.percentile(data, 100 - percentile_clip)
    p_high = np.percentile(data, percentile_clip)
    data_clipped = np.clip(data, p_low, p_high)
    
    if p_high > p_low:
        data_normalized = (data_clipped - p_low) / (p_high - p_low)
    else:
        data_normalized = np.zeros_like(data_clipped)
    
    return data_normalized.astype(np.float32)


class DataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence for efficient data loading during training.
    
    Handles:
    - Variable channel counts (4-band Dove vs 6-band SuperDove)
    - Non-square images from co-registration edge effects
    - Temporal data: (T, H, W, C) -> single frame or full temporal
    """
    
    def __init__(
        self,
        samples: List[TrainingSample],
        batch_size: int = 8,
        lr_size: Tuple[int, int] = (64, 64),
        hr_size: Tuple[int, int] = (192, 192),
        num_channels: int = 4,  # Target number of channels (will pad smaller)
        use_temporal: bool = False,
        augment: bool = True,
        shuffle: bool = True
    ):
        self.samples = samples
        self.batch_size = batch_size
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.num_channels = num_channels
        self.use_temporal = use_temporal
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(samples))
        
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.samples) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self._generate_batch(batch_indices)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _generate_batch(self, indices):
        lr_batch = []
        hr_batch = []
        
        for idx in indices:
            sample = self.samples[idx]
            
            if self.use_temporal:
                # Use all frames (T, H, W, C)
                lr_data = sample.lr_sequence
                hr_data = sample.hr_sequence
            else:
                # Use middle frame only
                frame_idx = sample.lr_sequence.shape[0] // 2
                lr_data = sample.lr_sequence[frame_idx]
                hr_data = sample.hr_sequence[frame_idx]
            
            # SQUARE CROP: Crop to square first (handles non-square from co-registration)
            lr_data = self._crop_to_square(lr_data)
            hr_data = self._crop_to_square(hr_data)
            
            # Resize/pad to target size
            lr_processed = self._crop_or_pad(lr_data, self.lr_size)
            hr_processed = self._crop_or_pad(hr_data, self.hr_size)
            
            # CHANNEL PADDING: Pad channels if needed (4-band → num_channels)
            lr_processed = self._pad_channels(lr_processed, self.num_channels)
            hr_processed = self._pad_channels(hr_processed, self.num_channels)
            
            # Augmentation
            if self.augment:
                lr_processed, hr_processed = self._augment(lr_processed, hr_processed)
            
            # Normalize
            lr_normalized = normalize_data(lr_processed)
            hr_normalized = normalize_data(hr_processed)
            
            lr_batch.append(lr_normalized)
            hr_batch.append(hr_normalized)
        
        return np.stack(lr_batch, axis=0), np.stack(hr_batch, axis=0)
    
    def _crop_to_square(self, data):
        """
        Crop non-square images to square (center crop).
        Handles edge effects from co-registration/harmonization.
        """
        if data.ndim == 4:  # Temporal (T, H, W, C)
            T, H, W, C = data.shape
            min_dim = min(H, W)
            start_h = (H - min_dim) // 2
            start_w = (W - min_dim) // 2
            return data[:, start_h:start_h+min_dim, start_w:start_w+min_dim, :]
        else:  # Single frame (H, W, C)
            H, W, C = data.shape
            min_dim = min(H, W)
            start_h = (H - min_dim) // 2
            start_w = (W - min_dim) // 2
            return data[start_h:start_h+min_dim, start_w:start_w+min_dim, :]
    
    def _pad_channels(self, data, target_channels):
        """
        Pad data to target number of channels with zeros.
        Handles 4-band Dove vs 6-band SuperDove data.
        """
        current_channels = data.shape[-1]
        
        if current_channels >= target_channels:
            # Truncate if needed (shouldn't happen normally)
            return data[..., :target_channels]
        
        # Pad with zeros
        if data.ndim == 4:  # Temporal (T, H, W, C)
            pad_width = ((0, 0), (0, 0), (0, 0), (0, target_channels - current_channels))
        else:  # Single frame (H, W, C)
            pad_width = ((0, 0), (0, 0), (0, target_channels - current_channels))
        
        return np.pad(data, pad_width, mode='constant', constant_values=0)
    
    def _crop_or_pad(self, data, target_size):
        """Center crop or pad to target size. Always returns exact target size."""
        if data.ndim == 4:  # Temporal (T, H, W, C)
            T, H, W, C = data.shape
            tH, tW = target_size
            
            # Create output array with exact target size
            result = np.zeros((T, tH, tW, C), dtype=data.dtype)
            
            # Center crop source
            src_start_h = max(0, (H - tH) // 2)
            src_start_w = max(0, (W - tW) // 2)
            src_end_h = min(H, src_start_h + tH)
            src_end_w = min(W, src_start_w + tW)
            
            # Center position in destination
            dst_start_h = max(0, (tH - H) // 2)
            dst_start_w = max(0, (tW - W) // 2)
            
            # Copy data
            copy_h = src_end_h - src_start_h
            copy_w = src_end_w - src_start_w
            result[:, dst_start_h:dst_start_h+copy_h, dst_start_w:dst_start_w+copy_w, :] = \
                data[:, src_start_h:src_end_h, src_start_w:src_end_w, :]
            
            return result
        else:  # Single frame (H, W, C)
            H, W, C = data.shape
            tH, tW = target_size
            
            # Create output array with exact target size
            result = np.zeros((tH, tW, C), dtype=data.dtype)
            
            # Center crop source
            src_start_h = max(0, (H - tH) // 2)
            src_start_w = max(0, (W - tW) // 2)
            src_end_h = min(H, src_start_h + tH)
            src_end_w = min(W, src_start_w + tW)
            
            # Center position in destination
            dst_start_h = max(0, (tH - H) // 2)
            dst_start_w = max(0, (tW - W) // 2)
            
            # Copy data
            copy_h = src_end_h - src_start_h
            copy_w = src_end_w - src_start_w
            result[dst_start_h:dst_start_h+copy_h, dst_start_w:dst_start_w+copy_w, :] = \
                data[src_start_h:src_end_h, src_start_w:src_end_w, :]
            
            return result
    
    def _augment(self, lr_data, hr_data):
        """Apply consistent augmentations to LR and HR."""
        # Make contiguous copies to avoid negative stride issues
        lr_data = np.ascontiguousarray(lr_data)
        hr_data = np.ascontiguousarray(hr_data)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            if lr_data.ndim == 4:
                lr_data = np.ascontiguousarray(lr_data[:, :, ::-1, :])
                hr_data = np.ascontiguousarray(hr_data[:, :, ::-1, :])
            else:
                lr_data = np.ascontiguousarray(lr_data[:, ::-1, :])
                hr_data = np.ascontiguousarray(hr_data[:, ::-1, :])
        
        # Random vertical flip
        if np.random.random() > 0.5:
            if lr_data.ndim == 4:
                lr_data = np.ascontiguousarray(lr_data[:, ::-1, :, :])
                hr_data = np.ascontiguousarray(hr_data[:, ::-1, :, :])
            else:
                lr_data = np.ascontiguousarray(lr_data[::-1, :, :])
                hr_data = np.ascontiguousarray(hr_data[::-1, :, :])
        
        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            if lr_data.ndim == 4:
                lr_data = np.ascontiguousarray(np.rot90(lr_data, k, axes=(1, 2)))
                hr_data = np.ascontiguousarray(np.rot90(hr_data, k, axes=(1, 2)))
            else:
                lr_data = np.ascontiguousarray(np.rot90(lr_data, k, axes=(0, 1)))
                hr_data = np.ascontiguousarray(np.rot90(hr_data, k, axes=(0, 1)))
        
        return lr_data, hr_data


def create_sr_model(
    input_shape: Tuple[int, int, int],
    scale_factor: int = 3,
    base_filters: int = 64,
    num_residual_blocks: int = 16,
    num_channels: int = 6
):
    """
    Create an enhanced SRCNN model with residual blocks.
    
    Architecture:
    - Feature extraction with residual blocks
    - Sub-pixel upscaling
    - Reconstruction with skip connection
    """
    inputs = tf.keras.Input(shape=input_shape, name='lr_input')
    
    # Initial feature extraction
    x = tf.keras.layers.Conv2D(base_filters, 3, padding='same', name='initial_conv')(inputs)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='initial_prelu')(x)
    
    # Save for skip connection
    initial_features = x
    
    # Residual blocks
    for i in range(num_residual_blocks):
        residual = x
        x = tf.keras.layers.Conv2D(base_filters, 3, padding='same', name=f'res{i}_conv1')(x)
        x = tf.keras.layers.BatchNormalization(name=f'res{i}_bn1')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f'res{i}_prelu')(x)
        x = tf.keras.layers.Conv2D(base_filters, 3, padding='same', name=f'res{i}_conv2')(x)
        x = tf.keras.layers.BatchNormalization(name=f'res{i}_bn2')(x)
        x = tf.keras.layers.Add(name=f'res{i}_add')([x, residual])
    
    # Post-residual conv
    x = tf.keras.layers.Conv2D(base_filters, 3, padding='same', name='post_res_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='post_res_bn')(x)
    x = tf.keras.layers.Add(name='global_skip')([x, initial_features])
    
    # Upscaling (sub-pixel)
    # For scale_factor=3, we need 3^2 = 9 filters per output channel
    for i in range(scale_factor - 1):  # Two 2x upscales for ~3x total
        x = tf.keras.layers.Conv2D(base_filters * 4, 3, padding='same', name=f'upsample{i}_conv')(x)
        x = tf.keras.layers.Lambda(
            lambda t: tf.nn.depth_to_space(t, 2),
            name=f'upsample{i}_pixel_shuffle'
        )(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f'upsample{i}_prelu')(x)
    
    # Final adjustment to target size
    h, w = input_shape[0], input_shape[1]
    target_h = h * scale_factor
    target_w = w * scale_factor
    x = tf.keras.layers.Lambda(
        lambda t: tf.image.resize(t, [target_h, target_w], method='bilinear'),
        name='final_resize'
    )(x)
    
    # Reconstruction
    x = tf.keras.layers.Conv2D(base_filters, 3, padding='same', name='recon_conv1')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='recon_prelu')(x)
    outputs = tf.keras.layers.Conv2D(num_channels, 3, padding='same', name='output')(x)
    
    model = tf.keras.Model(inputs, outputs, name='SatelliteSRNet')
    return model


class CombinedLoss(tf.keras.losses.Loss):
    """Combined loss for satellite SR: L1 + SSIM + Spectral."""
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        spectral_weight: float = 0.1,
        name='combined_loss'
    ):
        super().__init__(name=name)
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.spectral_weight = spectral_weight
    
    def call(self, y_true, y_pred):
        # L1 loss
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        
        # SSIM loss (per channel, averaged)
        ssim_losses = []
        for c in range(y_pred.shape[-1]):
            ssim = tf.image.ssim(
                y_true[..., c:c+1], 
                y_pred[..., c:c+1], 
                max_val=1.0
            )
            ssim_losses.append(1.0 - tf.reduce_mean(ssim))
        ssim_loss = tf.reduce_mean(ssim_losses)
        
        # Spectral consistency loss (preserves band ratios)
        # Compare band ratios: avoid division by zero
        eps = 1e-6
        true_sum = tf.reduce_sum(y_true, axis=-1, keepdims=True) + eps
        pred_sum = tf.reduce_sum(y_pred, axis=-1, keepdims=True) + eps
        true_ratio = y_true / true_sum
        pred_ratio = y_pred / pred_sum
        spectral_loss = tf.reduce_mean(tf.abs(true_ratio - pred_ratio))
        
        total = (
            self.l1_weight * l1_loss +
            self.ssim_weight * ssim_loss +
            self.spectral_weight * spectral_loss
        )
        
        return total


class TrainingCallback(tf.keras.callbacks.Callback):
    """Custom callback for detailed training logging."""
    
    def __init__(self, log_dir: str, val_generator=None):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.val_generator = val_generator
        self.history = {'loss': [], 'val_loss': [], 'lr': []}
        self.best_val_loss = float('inf')
        self.epoch_times = []
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        self.history['loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        self.history['lr'].append(float(self.model.optimizer.learning_rate))
        
        val_loss = logs.get('val_loss', float('inf'))
        improved = val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = val_loss
        
        print(f"\n   Epoch {epoch + 1}: loss={logs.get('loss', 0):.6f}, "
              f"val_loss={val_loss:.6f}, "
              f"time={epoch_time:.1f}s"
              f"{' ⭐ best' if improved else ''}")
        
        # Save history periodically
        if (epoch + 1) % 5 == 0:
            self._save_history()
    
    def on_train_end(self, logs=None):
        self._save_history()
        
        # Print summary
        avg_time = np.mean(self.epoch_times) if self.epoch_times else 0
        print(f"\n📊 Training Summary:")
        print(f"   Total epochs: {len(self.epoch_times)}")
        print(f"   Avg epoch time: {avg_time:.1f}s")
        print(f"   Best val loss: {self.best_val_loss:.6f}")
    
    def _save_history(self):
        history_file = self.log_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)


def train_model(
    # Data config
    base_dir: str = "data/datasets",
    years: List[str] = None,
    val_ratio: float = 0.1,
    
    # Model config
    lr_size: Tuple[int, int] = (64, 64),
    hr_size: Tuple[int, int] = (192, 192),
    scale_factor: int = 3,
    base_filters: int = 64,
    num_residual_blocks: int = 16,
    
    # Training config
    batch_size: int = 8,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    lr_decay_epochs: int = 25,
    lr_decay_rate: float = 0.5,
    
    # Loss weights
    l1_weight: float = 1.0,
    ssim_weight: float = 0.1,
    spectral_weight: float = 0.1,
    
    # Paths
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs"
):
    """
    Main training function for satellite super resolution.
    """
    print("=" * 80)
    print("🛰️  SATELLITE SUPER RESOLUTION TRAINING")
    print("=" * 80)
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load all datasets
    years = years or ['2020', '2021', '2022', '2023', '2024']
    loader = MultiYearDatasetLoader(base_dir, years)
    samples = loader.load_all()
    
    if not samples:
        print("❌ No samples found!")
        return None
    
    # Get sample info - use max channels from loader for unified model
    sample = samples[0]
    num_channels = loader.max_channels if loader.max_channels > 0 else sample.lr_sequence.shape[-1]
    temporal_frames = sample.lr_sequence.shape[0]
    
    print(f"\n📊 Sample Info:")
    print(f"   Target channels: {num_channels} (max across all samples)")
    print(f"   Temporal frames: {temporal_frames}")
    print(f"   Example LR shape: {sample.lr_sequence.shape}")
    print(f"   Example HR shape: {sample.hr_sequence.shape}")
    
    # Train/val split
    train_samples, val_samples = loader.get_train_val_split(val_ratio)
    
    # Create data generators with channel padding
    print(f"\n🔄 Creating data generators...")
    train_gen = DataGenerator(
        train_samples,
        batch_size=batch_size,
        lr_size=lr_size,
        hr_size=hr_size,
        num_channels=num_channels,  # Will pad 4-band to match 6-band
        use_temporal=False,  # Single frame for now
        augment=True,
        shuffle=True
    )
    
    val_gen = DataGenerator(
        val_samples,
        batch_size=batch_size,
        lr_size=lr_size,
        hr_size=hr_size,
        num_channels=num_channels,  # Same padding as training
        use_temporal=False,
        augment=False,
        shuffle=False
    )
    
    print(f"   Train batches: {len(train_gen)}")
    print(f"   Val batches: {len(val_gen)}")
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = create_sr_model(
        input_shape=(*lr_size, num_channels),
        scale_factor=scale_factor,
        base_filters=base_filters,
        num_residual_blocks=num_residual_blocks,
        num_channels=num_channels
    )
    
    print(f"   Parameters: {model.count_params():,}")
    print(f"   Model size: {model.count_params() * 4 / 1024**2:.1f} MB (FP32)")
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = CombinedLoss(
        l1_weight=l1_weight,
        ssim_weight=ssim_weight,
        spectral_weight=spectral_weight
    )
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['mae']
    )
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = Path(checkpoint_dir) / timestamp
    log_path = Path(log_dir) / timestamp
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks = [
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path / "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_decay_rate,
            patience=lr_decay_epochs,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Custom logging
        TrainingCallback(log_path, val_gen),
        
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_path),
            histogram_freq=5,
            write_images=True
        )
    ]
    
    # Training configuration summary
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   LR decay: {lr_decay_rate}x every {lr_decay_epochs} epochs (patience)")
    print(f"   Checkpoint dir: {checkpoint_path}")
    print(f"   Log dir: {log_path}")
    
    # Train
    print(f"\n🏋️  Starting training...")
    print("=" * 80)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=2
    )
    
    # Save final model
    final_path = checkpoint_path / "final_model.keras"
    model.save(final_path)
    print(f"\n💾 Final model saved: {final_path}")
    
    # Save training config
    config = {
        'years': years,
        'total_samples': len(samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'lr_size': lr_size,
        'hr_size': hr_size,
        'scale_factor': scale_factor,
        'base_filters': base_filters,
        'num_residual_blocks': num_residual_blocks,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'num_channels': num_channels,
        'model_params': model.count_params()
    }
    config_path = checkpoint_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED")
    print("=" * 80)
    print(f"   Best val loss: {min(history.history['val_loss']):.6f}")
    print(f"   Final val loss: {history.history['val_loss'][-1]:.6f}")
    print(f"   Checkpoints: {checkpoint_path}")
    print(f"   Logs: {log_path}")
    print(f"\n💡 Monitor training: tensorboard --logdir={log_path}")
    
    return model, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Satellite SR Model')
    parser.add_argument('--base-dir', type=str, default='data/datasets',
                        help='Base directory containing STA2_YYYY folders')
    parser.add_argument('--years', nargs='+', default=['2020', '2021', '2022', '2023', '2024'],
                        help='Years to include in training')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--filters', type=int, default=64,
                        help='Base filter count')
    parser.add_argument('--residual-blocks', type=int, default=16,
                        help='Number of residual blocks')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for logs')
    
    args = parser.parse_args()
    
    model, history = train_model(
        base_dir=args.base_dir,
        years=args.years,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        base_filters=args.filters,
        num_residual_blocks=args.residual_blocks,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
