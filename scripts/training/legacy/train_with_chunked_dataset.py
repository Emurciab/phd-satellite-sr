"""
Train SR model using dataset created with chunked downloads
Compatible with:
- numpy_efficient_pipeline output (training_samples.pkl + metadata.pkl)
- generate_sta2_grid_parallel.py output (dataset.pkl)

Supports:
- Single-resolution training (all bands at 10m)
- Multi-resolution training (10m and 20m bands separately)
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
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass


# =============================================================================
# DATA STRUCTURES FOR UNPICKLING
# =============================================================================

@dataclass
class TrainingSample:
    """Training sample from generate_sta2_grid_parallel.py - needed for unpickling"""
    lr_sequence: np.ndarray
    hr_sequence: np.ndarray
    temporal_dates: List[str]
    planet_dates: List[str]
    region_name: str
    sequence_id: int = 0
    tile_id: int = 0
    tile_bounds: Dict[str, Any] = None
    # Solar zenith angles for each frame (optimal ~45° for aquatic vegetation)
    s2_solar_zenith: Optional[List[float]] = None   # Sentinel-2 MEAN_SOLAR_ZENITH_ANGLE per frame
    planet_solar_zenith: Optional[List[float]] = None  # PlanetScope zenith (90 - sun_elevation) per frame


# =============================================================================
# MULTI-RESOLUTION DATA STRUCTURES
# =============================================================================

@dataclass
class MultiResolutionSample:
    """Training sample with separate 10m and 20m bands"""
    # Low resolution inputs
    lr_10m: np.ndarray  # (T, H, W, 3) - B2,B3,B4 at 10m
    lr_20m: np.ndarray  # (T, H/2, W/2, 1) - B8A at native 20m
    
    # High resolution targets
    hr_10m: np.ndarray  # (T, H', W', 3) - Planet 3m for RGB
    hr_20m: np.ndarray  # (T, H'/2, W'/2, 1) - Planet 3m downsampled for NIR
    
    # Metadata
    temporal_dates: List[str]
    planet_dates: List[str]
    region_name: str
    sample_id: int
    band_mode: str  # '4-band' or '8-band'


def prepare_multi_resolution_samples(samples: List, verbose: bool = True) -> List[MultiResolutionSample]:
    """
    Convert standard samples to multi-resolution format.
    
    This preserves native 20m resolution for B8A instead of using
    interpolated 10m data, which would introduce artifacts.
    
    Args:
        samples: List of TrainingSample objects
        verbose: Print conversion progress
    
    Returns:
        List of MultiResolutionSample objects
    """
    if verbose:
        print("\n🔬 Converting to multi-resolution format...")
        print("   This preserves native 20m resolution for B8A band")
    
    multi_res_samples = []
    
    for idx, sample in enumerate(samples):
        if verbose and idx % 100 == 0:
            print(f"   Processing sample {idx}/{len(samples)}...")
        
        # Original: lr_sequence (T, H, W, C), hr_sequence (T, H', W', C)
        lr_orig = sample.lr_sequence
        hr_orig = sample.hr_sequence
        
        # Get number of channels
        num_channels = lr_orig.shape[-1]
        
        # Handle different band configurations
        if num_channels >= 4:
            # Standard 4-band or 6-band (8-band SuperDove)
            # Split LR into 10m bands and 20m band
            lr_10m = lr_orig[:, :, :, :3]  # (T, H, W, 3) - RGB bands
            lr_20m_interp = lr_orig[:, :, :, 3:4]  # (T, H, W, 1) - NIR (interpolated)
            
            # Downsample LR B8A to native 20m resolution (remove interpolation artifact)
            # Native 20m = half the resolution of 10m
            lr_20m = lr_20m_interp[:, ::2, ::2, :]  # (T, H/2, W/2, 1)
            
            # Split HR into RGB and NIR
            hr_10m = hr_orig[:, :, :, :3]  # (T, H', W', 3)
            hr_nir_full = hr_orig[:, :, :, 3:4]  # (T, H', W', 1)
            
            # Downsample HR NIR to match 20m→6.67m upscaling
            hr_20m = hr_nir_full[:, ::2, ::2, :]  # (T, H'/2, W'/2, 1)
        else:
            # Less than 4 bands - treat all as 10m
            lr_10m = lr_orig
            lr_20m = np.zeros((lr_orig.shape[0], lr_orig.shape[1]//2, 
                              lr_orig.shape[2]//2, 1), dtype=lr_orig.dtype)
            hr_10m = hr_orig
            hr_20m = np.zeros((hr_orig.shape[0], hr_orig.shape[1]//2,
                              hr_orig.shape[2]//2, 1), dtype=hr_orig.dtype)
        
        # Get band mode from sample metadata if available
        band_mode = '4-band'
        if hasattr(sample, 'tile_bounds') and isinstance(sample.tile_bounds, dict):
            band_mode = sample.tile_bounds.get('band_mode', '4-band')
        
        multi_res_sample = MultiResolutionSample(
            lr_10m=lr_10m,
            lr_20m=lr_20m,
            hr_10m=hr_10m,
            hr_20m=hr_20m,
            temporal_dates=sample.temporal_dates,
            planet_dates=sample.planet_dates,
            region_name=sample.region_name,
            sample_id=idx,
            band_mode=band_mode
        )
        
        multi_res_samples.append(multi_res_sample)
    
    if verbose:
        print(f"   ✅ Converted {len(multi_res_samples)} samples to multi-resolution format")
        if len(multi_res_samples) > 0:
            s = multi_res_samples[0]
            print(f"   LR 10m shape: {s.lr_10m.shape}")
            print(f"   LR 20m shape: {s.lr_20m.shape}")
            print(f"   HR 10m shape: {s.hr_10m.shape}")
            print(f"   HR 20m shape: {s.hr_20m.shape}")
    
    return multi_res_samples


def load_dataset(dataset_path: str) -> Tuple[List, Dict[str, Any]]:
    """
    Flexible dataset loader that handles multiple formats:
    
    1. Directory with training_samples.pkl + metadata.pkl (old format)
    2. Single dataset.pkl file (new grid-parallel format)
    3. Direct path to a .pkl file
    
    Args:
        dataset_path: Path to dataset directory or .pkl file
    
    Returns:
        Tuple of (samples_list, metadata_dict)
    """
    path = Path(dataset_path)
    
    # Case 1: Direct path to a .pkl file
    if path.is_file() and path.suffix == '.pkl':
        print(f"📂 Loading dataset from file: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Check format: new grid-parallel format has 'samples' key
        if isinstance(data, dict) and 'samples' in data:
            samples = data['samples']
            config = data.get('config', {})
            
            # Generate metadata from first sample
            if len(samples) > 0:
                first_sample = samples[0]
                metadata = {
                    'lr_shape': first_sample.lr_sequence.shape,  # (T, H, W, C)
                    'hr_shape': first_sample.hr_sequence.shape,  # (T, H, W, C)
                    'num_samples': len(samples),
                    'creation_date': datetime.now().isoformat(),
                    **config  # Include any config from the file
                }
            else:
                raise ValueError("Dataset contains no samples!")
            
            print(f"   ✅ Loaded {len(samples)} samples (grid-parallel format)")
            print(f"   LR shape: {metadata['lr_shape']}")
            print(f"   HR shape: {metadata['hr_shape']}")
            return samples, metadata
        
        # Old format: might be just a list of samples
        elif isinstance(data, list):
            samples = data
            if len(samples) > 0:
                first_sample = samples[0]
                metadata = {
                    'lr_shape': first_sample.lr_sequence.shape,
                    'hr_shape': first_sample.hr_sequence.shape,
                    'num_samples': len(samples),
                    'creation_date': datetime.now().isoformat()
                }
            else:
                raise ValueError("Dataset contains no samples!")
            
            print(f"   ✅ Loaded {len(samples)} samples (list format)")
            return samples, metadata
        
        else:
            raise ValueError(f"Unknown dataset format in {path}")
    
    # Case 2: Directory path
    elif path.is_dir():
        # Check for new format first (dataset.pkl in directory)
        dataset_file = path / 'dataset.pkl'
        if dataset_file.exists():
            return load_dataset(str(dataset_file))  # Recursive call with file path
        
        # Check for old format (training_samples.pkl + metadata.pkl)
        samples_file = path / "training_samples.pkl"
        metadata_file = path / "metadata.pkl"
        
        if samples_file.exists():
            print(f"📂 Loading dataset from directory: {path}")
            
            with open(samples_file, 'rb') as f:
                samples = pickle.load(f)
            
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
            else:
                # Generate metadata from samples
                if len(samples) > 0:
                    first_sample = samples[0]
                    metadata = {
                        'lr_shape': first_sample.lr_sequence.shape,
                        'hr_shape': first_sample.hr_sequence.shape,
                        'num_samples': len(samples),
                        'creation_date': datetime.now().isoformat()
                    }
                else:
                    raise ValueError("Dataset contains no samples!")
            
            print(f"   ✅ Loaded {len(samples)} samples (chunked format)")
            print(f"   LR shape: {metadata['lr_shape']}")
            print(f"   HR shape: {metadata['hr_shape']}")
            return samples, metadata
        
        else:
            raise FileNotFoundError(
                f"No dataset found in {path}\n"
                f"Expected either:\n"
                f"  - {path}/dataset.pkl (grid-parallel format)\n"
                f"  - {path}/training_samples.pkl (chunked format)"
            )
    
    else:
        raise FileNotFoundError(f"Dataset path not found: {path}")


# Keep backward compatibility with old import
def load_chunked_dataset(dataset_dir="./chunked_sr_dataset"):
    """Legacy function for backward compatibility"""
    return load_dataset(dataset_dir)


class ChunkedSRDataset(Dataset):
    """PyTorch Dataset wrapper for chunked pipeline samples"""
    
    def __init__(self, samples, normalize=True, crop_to_consistent=True):
        """
        Args:
            samples: List of TrainingSample objects from pipeline
            normalize: Whether to normalize to [0,1] range
            crop_to_consistent: Crop all samples to minimum consistent size
        """
        self.samples = samples
        self.normalize = normalize
        self.crop_to_consistent = crop_to_consistent
        
        # Calculate consistent sizes if needed
        if crop_to_consistent and len(samples) > 0:
            lr_shapes = [s.lr_sequence.shape for s in samples]
            hr_shapes = [s.hr_sequence.shape for s in samples]
            
            # Find minimum H and W for LR and HR
            self.lr_h = min(s[1] for s in lr_shapes)
            self.lr_w = min(s[2] for s in lr_shapes)
            self.hr_h = min(s[1] for s in hr_shapes)
            self.hr_w = min(s[2] for s in hr_shapes)
        else:
            self.lr_h = self.lr_w = self.hr_h = self.hr_w = None
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract sequences
        lr = sample.lr_sequence  # Shape: (T, H, W, C)
        hr = sample.hr_sequence  # Shape: (T, H, W, C)
        
        # Crop to consistent size if enabled
        if self.crop_to_consistent and self.lr_h is not None:
            lr = lr[:, :self.lr_h, :self.lr_w, :]
            hr = hr[:, :self.hr_h, :self.hr_w, :]
        
        # Convert to torch tensors and transpose to (T, C, H, W)
        lr_tensor = torch.from_numpy(lr).float().permute(0, 3, 1, 2)
        hr_tensor = torch.from_numpy(hr).float().permute(0, 3, 1, 2)
        
        # Normalize to [0, 1] range (reflectance values typically 0-10000)
        if self.normalize:
            lr_tensor = torch.clamp(lr_tensor / 10000.0, 0, 1)
            hr_tensor = torch.clamp(hr_tensor / 10000.0, 0, 1)
        
        # Extract solar zenith angles (use mean of S2 and Planet if both available)
        # For training, we use a single representative SZA per sample
        sza = None
        if hasattr(sample, 's2_solar_zenith') and sample.s2_solar_zenith:
            # Use mean across temporal frames
            valid_s2 = [z for z in sample.s2_solar_zenith if z is not None]
            if valid_s2:
                sza = np.mean(valid_s2)
        
        # Fallback to Planet SZA if S2 not available
        if sza is None and hasattr(sample, 'planet_solar_zenith') and sample.planet_solar_zenith:
            valid_planet = [z for z in sample.planet_solar_zenith if z is not None]
            if valid_planet:
                sza = np.mean(valid_planet)
        
        return lr_tensor, hr_tensor, sza


def collate_with_sza(batch):
    """
    Custom collate function that handles SZA values which may be None.
    
    Args:
        batch: List of (lr_tensor, hr_tensor, sza) tuples
    
    Returns:
        (lr_batch, hr_batch, sza_list) where sza_list contains floats or Nones
    """
    lr_list = []
    hr_list = []
    sza_list = []
    
    for lr, hr, sza in batch:
        lr_list.append(lr)
        hr_list.append(hr)
        sza_list.append(sza)  # Keep as-is (may be None or float)
    
    # Stack tensors
    lr_batch = torch.stack(lr_list, dim=0)
    hr_batch = torch.stack(hr_list, dim=0)
    
    # Return SZA as a plain list (not tensor) to preserve None values
    return lr_batch, hr_batch, sza_list


class MultiResolutionDataset(Dataset):
    """
    PyTorch Dataset for multi-resolution training.
    
    Handles 10m bands (B2,B3,B4) and 20m bands (B8A) separately,
    preserving native resolution characteristics.
    """
    
    def __init__(self, samples: List[MultiResolutionSample], band_group: str = '10m', 
                 normalize: bool = True):
        """
        Args:
            samples: List of MultiResolutionSample objects
            band_group: '10m' for RGB bands, '20m' for NIR band
            normalize: Whether to normalize to [0,1] range
        """
        assert band_group in ('10m', '20m'), f"band_group must be '10m' or '20m', got {band_group}"
        
        self.samples = samples
        self.band_group = band_group
        self.normalize = normalize
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Select data based on band group
        if self.band_group == '10m':
            lr = sample.lr_10m  # (T, H, W, 3)
            hr = sample.hr_10m  # (T, H', W', 3)
        else:  # 20m
            lr = sample.lr_20m  # (T, H/2, W/2, 1)
            hr = sample.hr_20m  # (T, H'/2, W'/2, 1)
        
        # Convert to torch tensors and transpose to (T, C, H, W)
        lr_tensor = torch.from_numpy(lr).float().permute(0, 3, 1, 2)
        hr_tensor = torch.from_numpy(hr).float().permute(0, 3, 1, 2)
        
        # Normalize to [0, 1] range (reflectance values typically 0-10000)
        if self.normalize:
            lr_tensor = torch.clamp(lr_tensor / 10000.0, 0, 1)
            hr_tensor = torch.clamp(hr_tensor / 10000.0, 0, 1)
        
        return lr_tensor, hr_tensor


class SpectralFidelityLoss(nn.Module):
    """
    Multi-component loss for satellite SR with solar zenith angle weighting.
    
    The loss is weighted by a Gaussian penalty based on deviation from optimal
    solar zenith angle (45°) for aquatic vegetation analysis:
        w(SZA) = exp(-(SZA - SZA_optimal)^2 / (2 * sigma^2))
    
    This penalizes samples with extreme sun angles where water-leaving radiance
    is less reliable for vegetation analysis.
    """
    
    def __init__(self, alpha_pixel=1.0, alpha_spectral=0.1, alpha_perceptual=0.01,
                 sza_optimal=45.0, sza_sigma=30.0, use_sza_weighting=True):
        """
        Args:
            alpha_pixel: Weight for pixel-level loss (L1 + MSE)
            alpha_spectral: Weight for spectral angle loss
            alpha_perceptual: Weight for gradient/edge loss
            sza_optimal: Optimal solar zenith angle in degrees (default: 45°)
            sza_sigma: Standard deviation for Gaussian penalty (default: 30°)
                       Larger values = gentler penalty for angle deviations
            use_sza_weighting: Whether to apply SZA weighting (default: True)
        """
        super().__init__()
        self.alpha_pixel = alpha_pixel
        self.alpha_spectral = alpha_spectral
        self.alpha_perceptual = alpha_perceptual
        self.sza_optimal = sza_optimal
        self.sza_sigma = sza_sigma
        self.use_sza_weighting = use_sza_weighting
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def compute_sza_weight(self, sza: Optional[float]) -> float:
        """
        Compute Gaussian weight based on solar zenith angle deviation.
        
        Args:
            sza: Solar zenith angle in degrees, or None if unavailable
        
        Returns:
            Weight in range (0, 1], where 1.0 = optimal angle
        """
        if sza is None or not self.use_sza_weighting:
            return 1.0  # No penalty if SZA unknown or weighting disabled
        
        deviation = sza - self.sza_optimal
        weight = np.exp(-(deviation ** 2) / (2 * self.sza_sigma ** 2))
        return float(weight)
        
    def forward(self, pred, target, sza=None):
        """
        Args:
            pred: Predicted HR (B, C, H, W)
            target: Ground truth HR (B, C, H, W)
            sza: Solar zenith angle in degrees (optional). Can be:
                 - float: Single angle for entire batch
                 - List[float]: Per-sample angles (length = batch size)
                 - None: No SZA weighting applied
        
        Returns:
            total_loss: Weighted loss scalar
            loss_dict: Dictionary with individual loss components
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
        
        # Compute base loss
        base_loss = (self.alpha_pixel * pixel_loss + 
                     self.alpha_spectral * spectral_loss + 
                     self.alpha_perceptual * perceptual_loss)
        
        # Apply SZA weighting
        if sza is not None and self.use_sza_weighting:
            if isinstance(sza, (list, tuple)):
                # Per-sample weights - compute mean weight for batch
                weights = [self.compute_sza_weight(s) for s in sza]
                sza_weight = np.mean(weights)
            else:
                # Single weight for entire batch
                sza_weight = self.compute_sza_weight(sza)
        else:
            sza_weight = 1.0
        
        total_loss = sza_weight * base_loss
        
        return total_loss, {
            'pixel': pixel_loss.item(),
            'spectral': spectral_loss.item(),
            'perceptual': perceptual_loss.item(),
            'sza_weight': sza_weight,
            'base_loss': base_loss.item()
        }


class ImprovedSatelliteSR(nn.Module):
    """
    Simplified SR model for satellite imagery
    Uses temporal attention if sequence_length > 1
    Supports non-integer upscale factors via interpolation
    """
    
    def __init__(self, in_channels=4, out_channels=4, base_channels=64, 
                 num_blocks=8, upscale_factor=4, sequence_length=3, target_size=None):
        super().__init__()
        self.sequence_length = sequence_length
        self.upscale_factor = upscale_factor
        self.target_size = target_size  # (H, W) for final output
        
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
        
        # Upsampling - use 4x as base, then interpolate to exact size if needed
        effective_scale = min(upscale_factor, 4)  # Cap at 4x for PixelShuffle
        self.upscale = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * (effective_scale ** 2), 3, padding=1),
            nn.PixelShuffle(effective_scale),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
    def forward(self, x, target_size=None):
        """
        Args:
            x: Input LR sequence (B, T, C, H, W)
            target_size: Optional (H, W) to resize output to exact dimensions
        Returns:
            SR image (B, C, H', W')
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
        
        # Resize to exact target size if specified
        final_size = target_size or self.target_size
        if final_size is not None and (sr.shape[2] != final_size[0] or sr.shape[3] != final_size[1]):
            sr = torch.nn.functional.interpolate(sr, size=final_size, mode='bilinear', align_corners=False)
        
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
    
    # Load dataset (supports both old and new formats)
    samples, metadata = load_dataset(dataset_dir)
    
    print(f"\n📊 Dataset info:")
    print(f"   Total samples: {len(samples)}")
    print(f"   LR shape (from metadata): {metadata['lr_shape']}")
    print(f"   HR shape (from metadata): {metadata['hr_shape']}")
    
    # Compute consistent crop sizes from ALL samples BEFORE splitting
    lr_shapes = [s.lr_sequence.shape for s in samples]
    hr_shapes = [s.hr_sequence.shape for s in samples]
    crop_lr_h = min(s[1] for s in lr_shapes)
    crop_lr_w = min(s[2] for s in lr_shapes)
    crop_hr_h = min(s[1] for s in hr_shapes)
    crop_hr_w = min(s[2] for s in hr_shapes)
    
    print(f"   Crop sizes (min across all): LR={crop_lr_h}x{crop_lr_w}, HR={crop_hr_h}x{crop_hr_w}")
    
    # Split train/val
    val_split = 0.1
    val_size = int(len(samples) * val_split)
    train_samples = samples[:-val_size] if val_size > 0 else samples
    val_samples = samples[-val_size:] if val_size > 0 else []
    
    print(f"\n📊 Train/Val split:")
    print(f"   Training: {len(train_samples)} samples")
    print(f"   Validation: {len(val_samples)} samples")
    
    # Create datasets with FIXED crop sizes (computed from all samples)
    train_dataset = ChunkedSRDataset(train_samples, normalize=True, crop_to_consistent=True)
    # Override crop sizes with global values
    train_dataset.lr_h, train_dataset.lr_w = crop_lr_h, crop_lr_w
    train_dataset.hr_h, train_dataset.hr_w = crop_hr_h, crop_hr_w
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, collate_fn=collate_with_sza)
    
    if len(val_samples) > 0:
        val_dataset = ChunkedSRDataset(val_samples, normalize=True, crop_to_consistent=True)
        # Use same crop sizes as training
        val_dataset.lr_h, val_dataset.lr_w = crop_lr_h, crop_lr_w
        val_dataset.hr_h, val_dataset.hr_w = crop_hr_h, crop_hr_w
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=2, collate_fn=collate_with_sza)
    else:
        val_loader = None
    
    # Initialize model using consistent crop sizes
    print("\n🏗️  Building model...")
    sequence_length = metadata['lr_shape'][0]  # T from (T, H, W, C)
    in_channels = metadata['lr_shape'][-1]     # C
    
    lr_h, lr_w = crop_lr_h, crop_lr_w
    hr_h, hr_w = crop_hr_h, crop_hr_w
    upscale_factor = max(hr_h // lr_h, 3)  # At least 3x
    target_size = (hr_h, hr_w)  # Exact HR dimensions
    
    print(f"   LR size (cropped): {lr_h}x{lr_w}")
    print(f"   HR size (cropped): {hr_h}x{hr_w}")
    print(f"   Upscale factor: {upscale_factor}x (final resize to {target_size})")
    
    model = ImprovedSatelliteSR(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        num_blocks=num_blocks,
        upscale_factor=upscale_factor,
        sequence_length=sequence_length,
        target_size=target_size
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
        train_components = {'pixel': 0, 'spectral': 0, 'perceptual': 0, 'sza_weight': 0}
        
        for batch_idx, (lr, hr, sza_list) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            
            # sza_list is already a plain Python list from collate_with_sza
            # Filter out any NaN values
            sza_clean = [s if s is not None and not (isinstance(s, float) and np.isnan(s)) else None for s in sza_list]
            
            # Forward pass
            optimizer.zero_grad()
            sr = model(lr)
            
            # Compute loss with SZA weighting
            loss, components = criterion(sr, hr[:, hr.size(1)//2, :, :, :], sza=sza_clean)  # Use center frame
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            for k, v in components.items():
                if k in train_components:
                    train_components[k] += v
        
        train_loss /= len(train_loader)
        for k in train_components:
            train_components[k] /= len(train_loader)
        
        # Validation phase
        val_loss = 0.0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for lr, hr, sza_list in val_loader:
                    lr, hr = lr.to(device), hr.to(device)
                    # sza_list is already a plain Python list from collate_with_sza
                    sza_clean = [s if s is not None and not (isinstance(s, float) and np.isnan(s)) else None for s in sza_list]
                    sr = model(lr)
                    loss, _ = criterion(sr, hr[:, hr.size(1)//2, :, :, :], sza=sza_clean)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress (include average SZA weight)
        val_loss_str = f"{val_loss:.6f}" if val_loader else "N/A"
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.6f} "
              f"(pixel: {train_components['pixel']:.4f}, "
              f"spectral: {train_components['spectral']:.4f}, "
              f"perceptual: {train_components['perceptual']:.4f}, "
              f"sza_w: {train_components['sza_weight']:.3f}) - "
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


def train_multi_resolution(
    dataset_dir="./chunked_sr_dataset",
    output_dir="./sr_models_multires",
    batch_size=4,
    num_epochs=50,
    learning_rate=1e-4,
    num_blocks=8,
    base_channels=64,
    device='cuda'
):
    """
    Train separate SR models for 10m and 20m bands.
    
    This preserves native resolution characteristics:
    - 10m bands (B2, B3, B4): 10m→3m upscaling (3.33x)
    - 20m band (B8A): 20m→6.67m upscaling (3x)
    
    Args:
        dataset_dir: Path to dataset
        output_dir: Where to save trained models
        batch_size: Training batch size
        num_epochs: Number of training epochs per band group
        learning_rate: Initial learning rate
        device: torch device
    """
    
    print("="*70)
    print("🚀 MULTI-RESOLUTION TRAINING")
    print("   Separate models for 10m and 20m bands")
    print("="*70)
    print(f"Dataset: {dataset_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs per band group: {num_epochs}")
    print("="*70 + "\n")
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load and convert to multi-resolution format
    samples, metadata = load_dataset(dataset_dir)
    multi_res_samples = prepare_multi_resolution_samples(samples, verbose=True)
    
    # Split train/val
    val_split = 0.1
    val_size = int(len(multi_res_samples) * val_split)
    train_samples = multi_res_samples[:-val_size] if val_size > 0 else multi_res_samples
    val_samples = multi_res_samples[-val_size:] if val_size > 0 else []
    
    print(f"\n📊 Train/Val split:")
    print(f"   Training: {len(train_samples)} samples")
    print(f"   Validation: {len(val_samples)} samples")
    
    # Train both band groups
    results = {}
    
    for band_group in ['10m', '20m']:
        print(f"\n{'='*70}")
        print(f"📡 TRAINING {band_group.upper()} BANDS MODEL")
        print(f"{'='*70}\n")
        
        # Create datasets
        train_dataset = MultiResolutionDataset(train_samples, band_group=band_group, normalize=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        if len(val_samples) > 0:
            val_dataset = MultiResolutionDataset(val_samples, band_group=band_group, normalize=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            val_loader = None
        
        # Get sample shapes for this band group
        sample_lr, sample_hr = train_dataset[0]
        in_channels = sample_lr.shape[1]  # C from (T, C, H, W)
        sequence_length = sample_lr.shape[0]  # T
        lr_size = sample_lr.shape[2]  # H
        hr_size = sample_hr.shape[2]  # H
        upscale_factor = hr_size // lr_size
        
        print(f"   LR shape: {sample_lr.shape} (T, C, H, W)")
        print(f"   HR shape: {sample_hr.shape} (T, C, H, W)")
        print(f"   Upscale factor: {upscale_factor}x")
        print(f"   Channels: {in_channels}")
        
        # Initialize model for this band group
        model = ImprovedSatelliteSR(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_blocks,
            upscale_factor=upscale_factor,
            sequence_length=sequence_length
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Model parameters: {total_params:,}")
        
        # Setup training (disable SZA weighting for multi-resolution mode)
        criterion = SpectralFidelityLoss(use_sza_weighting=False)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (lr, hr) in enumerate(train_loader):
                lr, hr = lr.to(device), hr.to(device)
                
                optimizer.zero_grad()
                sr = model(lr)
                
                # Use center frame for loss (no SZA in multi-resolution mode)
                loss, components = criterion(sr, hr[:, hr.size(1)//2, :, :, :])
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
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
            
            scheduler.step()
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_str = f"{val_loss:.6f}" if val_loader else "N/A"
                print(f"   [{band_group}] Epoch {epoch+1}/{num_epochs} - "
                      f"Train: {train_loss:.6f}, Val: {val_str}")
            
            # Save best model
            if val_loader and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = output_path / f"best_model_{band_group}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'band_group': band_group,
                    'upscale_factor': upscale_factor,
                    'in_channels': in_channels,
                    'metadata': metadata
                }, best_model_path)
        
        # Save final model for this band group
        final_model_path = output_path / f"final_model_{band_group}.pth"
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'band_group': band_group,
            'upscale_factor': upscale_factor,
            'in_channels': in_channels,
            'metadata': metadata
        }, final_model_path)
        
        results[band_group] = {
            'best_val_loss': best_val_loss,
            'upscale_factor': upscale_factor,
            'in_channels': in_channels
        }
        
        print(f"\n   ✅ {band_group} model trained. Best val loss: {best_val_loss:.6f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ MULTI-RESOLUTION TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Models saved to: {output_dir}")
    for bg, r in results.items():
        print(f"   {bg}: {r['in_channels']} channels, {r['upscale_factor']}x upscale, "
              f"best val loss: {r['best_val_loss']:.6f}")
    print(f"{'='*70}\n")
    
    return results


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
    parser.add_argument('--mode', choices=['standard', 'multi_resolution'], default='standard',
                       help='Training mode: standard (all bands together) or multi_resolution (separate 10m/20m)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with 2 epochs')
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("\n🧪 QUICK TEST MODE: 2 epochs\n")
        if args.mode == 'multi_resolution':
            train_multi_resolution(
                dataset_dir="./quick_test_dataset",
                output_dir="./quick_test_models_multires",
                batch_size=2,
                num_epochs=2,
                learning_rate=args.lr
            )
        else:
            train_with_chunked_dataset(
                dataset_dir="./quick_test_dataset",
                output_dir="./quick_test_models",
                batch_size=2,
                num_epochs=2,
                learning_rate=args.lr
            )
    else:
        if args.mode == 'multi_resolution':
            print("\n🔬 MULTI-RESOLUTION MODE: Training separate 10m and 20m models\n")
            train_multi_resolution(
                dataset_dir=args.dataset_dir,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                num_blocks=args.num_blocks,
                base_channels=args.base_channels
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
