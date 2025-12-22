#!/usr/bin/env python3
"""
Training script for MultiRes model with BEST FRAME SELECTION.

Key idea: Instead of combining frames or supervising all frames,
SELECT the highest-quality frame from the temporal sequence.

Quality scoring methods:
1. Variance-based: Higher variance = more detail = better quality
2. Sharpness-based: Laplacian variance measures edge sharpness
3. Entropy-based: Higher entropy = more information content
4. Consistency-based: Frame most similar to median (least outlier)

The model receives only the BEST LR frame and learns to predict
the corresponding BEST HR frame.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import time
from pathlib import Path
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.ndimage import laplace

# Import the multires model
from satellite_sr_model_v2 import MultiResolutionSatelliteSRv2 as MultiResolutionSatelliteSR

# Configuration
BEST_CONFIG = {
    # Model architecture  
    'base_channels': 64,
    'num_encoder_blocks': 4,
    
    # Optimizer
    'optimizer': 'adamw',
    'lr': 3.064e-05,
    'weight_decay': 5.235e-06,
    
    # Scheduler
    'patience_scheduler': 4,
    
    # Loss weights
    'alpha_pixel': 1.0,
    'alpha_spectral': 0.137,
    'alpha_gradient': 0.01,
    'alpha_perceptual': 0.1,
    
    # Best frame selection options
    # Available metrics: 'sharpness', 'variance', 'entropy', 'consistency',
    #                   'haze', 'bad_pixels', 'contrast', 'combined'
    'quality_metric': 'combined',
    'use_temporal_context': True,   # Still provide all 3 frames to model (vs just best)
    
    # Resolution weights
    'weight_10m': 1.0,
    'weight_20m': 1.0,
    
    # Training
    'batch_size': 4,
    'epochs': 200,
    'patience': 20,
}


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
    
    def __call__(self, value, epoch):
        if value < (self.best_value - self.min_delta):
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def compute_sharpness(frame):
    """
    Compute sharpness using Laplacian variance.
    Higher value = sharper image = better quality.
    
    Args:
        frame: (H, W, C) numpy array
    Returns:
        float: sharpness score
    """
    # Average across channels
    if frame.ndim == 3:
        gray = np.mean(frame, axis=-1)
    else:
        gray = frame
    
    # Laplacian variance (Pech-Pacheco et al.)
    lap = laplace(gray.astype(np.float64))
    return np.var(lap)


def compute_variance(frame):
    """
    Compute overall variance as quality metric.
    Higher variance often indicates more detail.
    
    Args:
        frame: (H, W, C) numpy array
    Returns:
        float: variance score
    """
    return np.var(frame)


def compute_entropy(frame):
    """
    Compute image entropy as quality metric.
    Higher entropy = more information content.
    
    Args:
        frame: (H, W, C) numpy array
    Returns:
        float: entropy score
    """
    # Quantize to 256 bins for histogram
    if frame.ndim == 3:
        gray = np.mean(frame, axis=-1)
    else:
        gray = frame
    
    # Normalize to 0-255 range
    gray_normalized = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-8) * 255).astype(np.uint8)
    
    # Compute histogram
    hist, _ = np.histogram(gray_normalized, bins=256, range=(0, 256))
    hist = hist[hist > 0]  # Remove zero bins
    
    # Compute entropy
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log2(prob + 1e-10))
    
    return entropy


def compute_consistency(frame, all_frames):
    """
    Compute consistency with median (inverse of outlier-ness).
    Frame closest to median is most "typical" and likely cleanest.
    
    Args:
        frame: (H, W, C) numpy array
        all_frames: (T, H, W, C) numpy array
    Returns:
        float: consistency score (higher = more consistent = better)
    """
    median_frame = np.median(all_frames, axis=0)
    # Use negative MAE so higher score = better (more consistent)
    mae = np.mean(np.abs(frame - median_frame))
    return -mae  # Negative because we want to maximize consistency


def compute_haze_score(frame):
    """
    Compute haze/clarity score using Dark Channel Prior.
    
    Hazy images have higher minimum values in local patches (no true dark pixels).
    Clear images have lower dark channel values.
    
    Based on: "Single Image Haze Removal Using Dark Channel Prior" (He et al., 2009)
    
    Args:
        frame: (H, W, C) numpy array
    Returns:
        float: clarity score (higher = less haze = better)
    """
    from scipy.ndimage import minimum_filter
    
    # Dark channel: minimum across channels, then local minimum
    if frame.ndim == 3:
        # Minimum across spectral channels
        min_channel = np.min(frame, axis=-1)
    else:
        min_channel = frame
    
    # Local minimum in 15x15 patch (dark channel)
    patch_size = 15
    dark_channel = minimum_filter(min_channel, size=patch_size)
    
    # Haze indicator: mean of dark channel (high = hazy)
    # We return negative so higher score = clearer image
    haze_level = np.mean(dark_channel)
    
    return -haze_level  # Negative: lower dark channel mean = clearer = better


def compute_bad_pixel_score(frame, threshold_low=0.001, threshold_high=0.95):
    """
    Compute bad pixel score based on extreme/anomalous values.
    
    Bad pixels include:
    - Dead pixels (very low values, near zero)
    - Saturated pixels (very high values)
    - Hot pixels (isolated bright spots)
    
    Args:
        frame: (H, W, C) numpy array
        threshold_low: Values below this are considered dead pixels
        threshold_high: Values above this are considered saturated
    Returns:
        float: quality score (higher = fewer bad pixels = better)
    """
    # Count extreme value pixels
    dead_pixels = np.sum(frame < threshold_low)
    saturated_pixels = np.sum(frame > threshold_high)
    total_pixels = frame.size
    
    # Fraction of bad pixels
    bad_pixel_fraction = (dead_pixels + saturated_pixels) / total_pixels
    
    # Return negative bad pixel fraction (fewer bad pixels = higher score)
    return -bad_pixel_fraction


def compute_local_contrast(frame):
    """
    Compute local contrast score.
    
    Hazy/foggy images have reduced local contrast.
    Clear images have higher local contrast.
    
    Args:
        frame: (H, W, C) numpy array
    Returns:
        float: local contrast score (higher = better contrast = clearer)
    """
    from scipy.ndimage import uniform_filter
    
    if frame.ndim == 3:
        gray = np.mean(frame, axis=-1)
    else:
        gray = frame
    
    # Local mean
    window_size = 11
    local_mean = uniform_filter(gray.astype(np.float64), size=window_size)
    
    # Local standard deviation (contrast)
    local_sq_mean = uniform_filter((gray.astype(np.float64))**2, size=window_size)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    
    # Mean local contrast
    return np.mean(local_std)


def compute_combined_quality(frame, all_frames=None):
    """
    Combined quality score focusing on haze and bad pixels.
    
    This metric is designed to reject:
    - Hazy frames (high dark channel minimum = atmospheric scattering)
    - Frames with bad pixels (extreme values that create artificial edges)
    
    The key insight is that bad pixels create "fake" sharpness/contrast,
    so we use a multiplicative penalty for bad pixels instead of additive.
    
    Args:
        frame: (H, W, C) numpy array
        all_frames: Optional (T, H, W, C) for consistency check
    Returns:
        float: combined quality score (higher = better)
    """
    # Compute individual scores
    haze = compute_haze_score(frame)  # Negative, closer to 0 = clearer
    bad_pixels = compute_bad_pixel_score(frame)  # Negative, closer to 0 = better
    contrast = compute_local_contrast(frame)  # Positive, higher = better
    sharpness = compute_sharpness(frame)  # Positive, higher = better
    
    # Compute consistency with temporal neighbors (if available)
    if all_frames is not None:
        consistency = compute_consistency(frame, all_frames)  # Positive, higher = better
    else:
        consistency = 0.5
    
    # === Normalization ===
    # Sharpness: typical range 0.0001 to 0.1, cap at 1
    sharpness_norm = min(sharpness * 10, 1.0)
    
    # Contrast: typical range 0 to 0.2
    contrast_norm = min(contrast * 5, 1.0)
    
    # Haze: -1 (very hazy) to 0 (clear), convert to 0-1
    haze_norm = max(0, 1.0 + haze)  # 1 = clear, 0 = very hazy
    
    # Bad pixels: -1 (all bad) to 0 (none), convert to 0-1
    bad_pixel_norm = max(0, 1.0 + bad_pixels * 5)  # Amplify penalty, cap at 0
    
    # Consistency: already 0-1 range
    consistency_norm = max(0, consistency)
    
    # === Combined Score ===
    # Use multiplicative penalty for bad pixels (frames with bad pixels 
    # should NOT be selected even if other metrics look good)
    
    # Base quality from haze and atmospheric clarity
    base_quality = (
        0.4 * haze_norm +           # Haze is primary concern
        0.25 * contrast_norm +       # Contrast (but real contrast, not from bad pixels)
        0.2 * sharpness_norm +       # Sharpness
        0.15 * consistency_norm      # Temporal consistency
    )
    
    # Multiplicative penalty for bad pixels
    # If 5% bad pixels: penalty = 0.75 (25% reduction)
    # If 10% bad pixels: penalty = 0.5 (50% reduction)
    bad_pixel_penalty = bad_pixel_norm
    
    combined = base_quality * bad_pixel_penalty
    
    return combined


def select_best_frame(frames, method='sharpness'):
    """
    Select the best frame from temporal sequence.
    
    Args:
        frames: (T, H, W, C) numpy array
        method: Quality metric to use
            - 'sharpness': Laplacian variance (edge sharpness)
            - 'variance': Overall pixel variance
            - 'entropy': Information content
            - 'consistency': Similarity to temporal median
            - 'haze': Dark channel prior (haze detection)
            - 'bad_pixels': Extreme value detection
            - 'contrast': Local contrast (haze-sensitive)
            - 'combined': Weighted combination of haze metrics
    
    Returns:
        int: Index of best frame
        float: Quality score of best frame
    """
    T = frames.shape[0]
    scores = []
    
    for t in range(T):
        frame = frames[t]
        
        if method == 'sharpness':
            score = compute_sharpness(frame)
        elif method == 'variance':
            score = compute_variance(frame)
        elif method == 'entropy':
            score = compute_entropy(frame)
        elif method == 'consistency':
            score = compute_consistency(frame, frames)
        elif method == 'haze':
            # Haze score: higher = clearer (less haze)
            score = compute_haze_score(frame)
        elif method == 'bad_pixels':
            # Bad pixel score: higher = fewer bad pixels (closer to 0)
            score = compute_bad_pixel_score(frame)
        elif method == 'contrast':
            # Local contrast: higher = better (less haze)
            score = compute_local_contrast(frame)
        elif method == 'combined':
            # Combined haze + bad pixel + contrast + sharpness
            score = compute_combined_quality(frame, frames)
        else:
            raise ValueError(f"Unknown quality metric: {method}")
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    return best_idx, scores[best_idx]


class BestFrameMmapDataset(Dataset):
    """
    Dataset that selects the BEST frame from temporal sequence.
    
    For each sample:
    - Quality score is computed for each of T frames
    - Best frame is selected based on chosen metric
    - HR target: The HR frame corresponding to best LR frame
    - LR input: Either all T frames (temporal context) or just best frame
    """
    
    def __init__(self, mmap_dir, sequence_length=3, augment=True, 
                 quality_metric='combined', use_temporal_context=True):
        """
        Args:
            mmap_dir: Path to mmap data directory
            sequence_length: Number of temporal frames (T)
            augment: Apply data augmentation
            quality_metric: Quality metric for frame selection
                - 'sharpness': Laplacian variance (edge sharpness)
                - 'variance': Overall pixel variance
                - 'entropy': Information content
                - 'consistency': Similarity to temporal median
                - 'haze': Dark channel prior (haze detection)
                - 'bad_pixels': Extreme value detection
                - 'contrast': Local contrast (haze-sensitive)
                - 'combined': Weighted combination of haze metrics (recommended)
            use_temporal_context: If True, provide all T frames to model
        """
        mmap_dir = Path(mmap_dir)
        self.lr = np.load(mmap_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(mmap_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr_10m = np.load(mmap_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(mmap_dir / 'hr_20m.npy', mmap_mode='r')
        self.n_samples = self.lr.shape[0]
        self.sequence_length = sequence_length
        self.augment = augment
        self.quality_metric = quality_metric
        self.use_temporal_context = use_temporal_context
        
        print(f"BestFrame Dataset: {self.n_samples} samples")
        print(f"  Quality metric: {quality_metric}")
        print(f"  Use temporal context: {use_temporal_context}")
        print(f"  LR 10m: {self.lr.shape}, LR 20m: {self.lr_20m.shape}")
        print(f"  HR 10m: {self.hr_10m.shape}, HR 20m: {self.hr_20m.shape}")
        
        # Pre-compute best frame indices for efficiency (optional)
        self._precomputed_indices = None
    
    def precompute_best_indices(self):
        """Pre-compute best frame indices for all samples (faster training)."""
        print(f"Pre-computing best frame indices using '{self.quality_metric}' metric...")
        self._precomputed_indices = []
        
        for idx in range(self.n_samples):
            lr_10m = np.array(self.lr[idx])
            best_idx, _ = select_best_frame(lr_10m, self.quality_metric)
            self._precomputed_indices.append(best_idx)
        
        # Print distribution of selected frames
        counts = np.bincount(self._precomputed_indices, minlength=self.sequence_length)
        print(f"  Frame selection distribution: {dict(enumerate(counts))}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        lr_10m = self.lr[idx].copy()      # (T, H, W, C)
        lr_20m = self.lr_20m[idx].copy()  # (T, H, W, C)
        hr_10m = self.hr_10m[idx].copy()  # (T, H, W, C)
        hr_20m = self.hr_20m[idx].copy()  # (T, H, W, C)
        
        # Select best frame BEFORE augmentation (quality metrics work on original data)
        if self._precomputed_indices is not None:
            best_idx = self._precomputed_indices[idx]
        else:
            best_idx, _ = select_best_frame(lr_10m, self.quality_metric)
        
        # Apply augmentation (to ALL frames consistently)
        if self.augment:
            import random
            if random.random() > 0.5:
                lr_10m = np.flip(lr_10m, axis=2).copy()
                lr_20m = np.flip(lr_20m, axis=2).copy()
                hr_10m = np.flip(hr_10m, axis=2).copy()
                hr_20m = np.flip(hr_20m, axis=2).copy()
            if random.random() > 0.5:
                lr_10m = np.flip(lr_10m, axis=1).copy()
                lr_20m = np.flip(lr_20m, axis=1).copy()
                hr_10m = np.flip(hr_10m, axis=1).copy()
                hr_20m = np.flip(hr_20m, axis=1).copy()
            k = random.randint(0, 3)
            if k > 0:
                lr_10m = np.rot90(lr_10m, k, axes=(1, 2)).copy()
                lr_20m = np.rot90(lr_20m, k, axes=(1, 2)).copy()
                hr_10m = np.rot90(hr_10m, k, axes=(1, 2)).copy()
                hr_20m = np.rot90(hr_20m, k, axes=(1, 2)).copy()
        
        # HR target: Use the BEST frame
        hr_10m_best = hr_10m[best_idx]  # (H, W, C)
        hr_20m_best = hr_20m[best_idx]  # (H, W, C)
        
        # LR input: Either all frames (temporal context) or just best frame
        if self.use_temporal_context:
            # Reorder so best frame is in middle (if not already)
            # This helps the model's temporal attention focus on best frame
            mid_idx = self.sequence_length // 2
            if best_idx != mid_idx:
                # Swap best frame to middle position
                lr_10m_reordered = lr_10m.copy()
                lr_20m_reordered = lr_20m.copy()
                lr_10m_reordered[mid_idx] = lr_10m[best_idx]
                lr_10m_reordered[best_idx] = lr_10m[mid_idx]
                lr_20m_reordered[mid_idx] = lr_20m[best_idx]
                lr_20m_reordered[best_idx] = lr_20m[mid_idx]
                lr_10m = lr_10m_reordered
                lr_20m = lr_20m_reordered
            
            lr_10m_out = lr_10m  # (T, H, W, C)
            lr_20m_out = lr_20m
        else:
            # Only provide best frame (no temporal context)
            lr_10m_out = np.expand_dims(lr_10m[best_idx], axis=0)  # (1, H, W, C)
            lr_20m_out = np.expand_dims(lr_20m[best_idx], axis=0)
        
        # Convert to tensors
        # LR: (T, H, W, C) -> (T, C, H, W)
        lr_10m_t = torch.from_numpy(lr_10m_out).permute(0, 3, 1, 2).float()
        lr_20m_t = torch.from_numpy(lr_20m_out).permute(0, 3, 1, 2).float()
        
        # HR: (H, W, C) -> (C, H, W)
        hr_10m_t = torch.from_numpy(hr_10m_best).permute(2, 0, 1).float()
        hr_20m_t = torch.from_numpy(hr_20m_best).permute(2, 0, 1).float()
        
        return {'lr_10m': lr_10m_t, 'lr_20m': lr_20m_t, 
                'hr_10m': hr_10m_t, 'hr_20m': hr_20m_t,
                'best_idx': best_idx}


class MultiResSpectralLoss(nn.Module):
    """Standard multi-resolution loss."""
    
    def __init__(self, cfg):
        super().__init__()
        self.alpha_pixel = cfg['alpha_pixel']
        self.alpha_spectral = cfg['alpha_spectral']
        self.alpha_gradient = cfg['alpha_gradient']
        self.alpha_perceptual = cfg['alpha_perceptual']
        self.weight_10m = cfg['weight_10m']
        self.weight_20m = cfg['weight_20m']
        
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
        # VGG Perceptual loss
        from perceptual_loss import VGGPerceptualLoss
        self.perceptual_loss = VGGPerceptualLoss()
        
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m):
        # 10m branch
        pixel_10m = self.l1(pred_10m, target_10m) + 0.1 * self.mse(pred_10m, target_10m)
        sam_10m = self._sam_loss(pred_10m, target_10m)
        grad_10m = self._grad_loss(pred_10m, target_10m)
        perc_10m = self.perceptual_loss(pred_10m, target_10m)
        
        loss_10m = (self.alpha_pixel * pixel_10m + 
                    self.alpha_spectral * sam_10m + 
                    self.alpha_gradient * grad_10m +
                    self.alpha_perceptual * perc_10m)
        
        # 20m branch (no perceptual)
        pixel_20m = self.l1(pred_20m, target_20m) + 0.1 * self.mse(pred_20m, target_20m)
        sam_20m = self._sam_loss(pred_20m, target_20m)
        grad_20m = self._grad_loss(pred_20m, target_20m)
        
        loss_20m = (self.alpha_pixel * pixel_20m + 
                    self.alpha_spectral * sam_20m + 
                    self.alpha_gradient * grad_20m)
        
        total = self.weight_10m * loss_10m + self.weight_20m * loss_20m
        return total, loss_10m, loss_20m
    
    def _sam_loss(self, pred, target):
        pred_norm = nn.functional.normalize(pred, dim=1, eps=1e-8)
        target_norm = nn.functional.normalize(target, dim=1, eps=1e-8)
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        return 1.0 - cos_sim.mean()
    
    def _grad_loss(self, pred, target):
        loss = 0
        for c in range(pred.shape[1]):
            pred_gx = nn.functional.conv2d(pred[:, c:c+1], self.sobel_x, padding=1)
            pred_gy = nn.functional.conv2d(pred[:, c:c+1], self.sobel_y, padding=1)
            target_gx = nn.functional.conv2d(target[:, c:c+1], self.sobel_x, padding=1)
            target_gy = nn.functional.conv2d(target[:, c:c+1], self.sobel_y, padding=1)
            loss += self.l1(pred_gx, target_gx) + self.l1(pred_gy, target_gy)
        return loss / pred.shape[1]


def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)).item()


def train_epoch(model, loader, criterion, optimizer, device, cfg):
    model.train()
    total_loss = 0
    
    for batch in loader:
        lr_10m = batch['lr_10m'].to(device)
        lr_20m = batch['lr_20m'].to(device)
        hr_10m = batch['hr_10m'].to(device)
        hr_20m = batch['hr_20m'].to(device)
        
        H_hr_10m, W_hr_10m = hr_10m.shape[2], hr_10m.shape[3]
        H_hr_20m, W_hr_20m = hr_20m.shape[2], hr_20m.shape[3]
        
        optimizer.zero_grad()
        
        pred_10m, pred_20m = model(
            lr_10m, lr_20m,
            target_size_10m=(H_hr_10m, W_hr_10m),
            target_size_20m=(H_hr_20m, W_hr_20m)
        )
        
        loss, loss_10m, loss_20m = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    psnr_10m_sum, psnr_20m_sum = 0, 0
    
    with torch.no_grad():
        for batch in loader:
            lr_10m = batch['lr_10m'].to(device)
            lr_20m = batch['lr_20m'].to(device)
            hr_10m = batch['hr_10m'].to(device)
            hr_20m = batch['hr_20m'].to(device)
            
            H_hr_10m, W_hr_10m = hr_10m.shape[2], hr_10m.shape[3]
            H_hr_20m, W_hr_20m = hr_20m.shape[2], hr_20m.shape[3]
            
            pred_10m, pred_20m = model(
                lr_10m, lr_20m,
                target_size_10m=(H_hr_10m, W_hr_10m),
                target_size_20m=(H_hr_20m, W_hr_20m)
            )
            
            loss, _, _ = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
            total_loss += loss.item()
            psnr_10m_sum += calculate_psnr(pred_10m, hr_10m)
            psnr_20m_sum += calculate_psnr(pred_20m, hr_20m)
    
    n = len(loader)
    return total_loss / n, psnr_10m_sum / n, psnr_20m_sum / n


def train_model(data_dir, checkpoint_dir, experiment_name, cfg=BEST_CONFIG):
    """Train model with best frame selection."""
    
    print(f"\n{'='*70}")
    print(f"Training: {experiment_name}")
    print(f"Data: {data_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Quality metric: {cfg.get('quality_metric', 'sharpness')}")
    print(f"Use temporal context: {cfg.get('use_temporal_context', True)}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dataset = BestFrameMmapDataset(
        data_dir, 
        augment=True,
        quality_metric=cfg.get('quality_metric', 'sharpness'),
        use_temporal_context=cfg.get('use_temporal_context', True)
    )
    
    # Pre-compute best indices for faster training
    dataset.precompute_best_indices()
    
    n_val = max(int(len(dataset) * 0.1), 10)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.augment = False
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['batch_size'], 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['batch_size'], 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Model frames: 3 if using temporal context, 1 otherwise
    num_frames = 3 if cfg.get('use_temporal_context', True) else 1
    
    model = MultiResolutionSatelliteSR(
        in_channels_10m=3,
        in_channels_20m=2,
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=num_frames
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['base_channels']} channels, {cfg['num_encoder_blocks']} blocks, "
          f"T={num_frames}, {n_params:,} params")
    
    criterion = MultiResSpectralLoss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=cfg['patience_scheduler']
    )
    early_stopping = EarlyStopping(patience=cfg['patience'])
    
    best_val_loss = float('inf')
    best_psnr = 0
    history = []
    
    print(f"\nLoss weights: Pixel={cfg['alpha_pixel']:.3f}, SAM={cfg['alpha_spectral']:.3f}, "
          f"Grad={cfg['alpha_gradient']:.4f}, Perc={cfg['alpha_perceptual']:.3f}")
    print(f"Branch weights: 10m={cfg['weight_10m']:.3f}, 20m={cfg['weight_20m']:.3f}")
    print(f"\nStarting training for {cfg['epochs']} epochs...")
    
    for epoch in range(1, cfg['epochs'] + 1):
        start = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, cfg)
        val_loss, psnr_10m, psnr_20m = validate(model, val_loader, criterion, device)
        avg_psnr = (psnr_10m + psnr_20m) / 2
        
        scheduler.step(val_loss)
        
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]['lr']
        
        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'psnr_10m': psnr_10m, 'psnr_20m': psnr_20m, 'avg_psnr': avg_psnr, 'lr': lr
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'psnr_10m': psnr_10m,
                'psnr_20m': psnr_20m,
                'config': cfg
            }, checkpoint_dir / 'best_model.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': cfg
        }, checkpoint_dir / 'latest_model.pt')
        
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PSNR: 10m={psnr_10m:.2f} 20m={psnr_20m:.2f} avg={avg_psnr:.2f} | "
              f"LR: {lr:.2e} | Time: {elapsed:.1f}s")
        
        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping at epoch {epoch}. Best: epoch {early_stopping.best_epoch}")
            break
    
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training complete: {experiment_name}")
    print(f"Best val loss: {best_val_loss:.4f}, Best PSNR: {best_psnr:.2f} dB")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"{'='*70}\n")
    
    return best_val_loss, best_psnr


def main():
    parser = argparse.ArgumentParser(
        description='Train MultiRes with BEST FRAME SELECTION',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Best Frame Selection Strategy:
  - For each sample, compute quality score for all T frames
  - Select the highest-quality LR/HR pair
  - Model learns to super-resolve the "best" data
  
Quality metrics:
  - sharpness: Laplacian variance (edge sharpness)
  - variance: Overall pixel variance (detail level)
  - entropy: Information content
  - consistency: Similarity to median (least outlier)

Examples:
  # Default: combined metric (haze + bad pixels + contrast + sharpness)
  python train_multires_temporal_bestFrame.py --dataset full
  
  # Use specific haze-focused metric
  python train_multires_temporal_bestFrame.py --metric haze
  
  # Use combined metric (recommended for satellite imagery)
  python train_multires_temporal_bestFrame.py --metric combined
  
  # Use classic sharpness metric
  python train_multires_temporal_bestFrame.py --metric sharpness
  
  # No temporal context (only best frame as input)
  python train_multires_temporal_bestFrame.py --no_temporal_context
        """
    )
    parser.add_argument('--dataset', choices=['full', 'filtered', 'both'], default='full',
                        help='Which dataset to train on')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum epochs to train')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--alpha_perceptual', type=float, default=0.1,
                        help='Weight for perceptual loss')
    parser.add_argument('--metric', 
                        choices=['sharpness', 'variance', 'entropy', 'consistency',
                                 'haze', 'bad_pixels', 'contrast', 'combined'], 
                        default='combined', 
                        help='Quality metric for frame selection. "combined" is recommended for satellite imagery')
    parser.add_argument('--no_temporal_context', action='store_true',
                        help='Only use best frame as input (no temporal context)')
    args = parser.parse_args()
    
    cfg = BEST_CONFIG.copy()
    cfg['epochs'] = args.epochs
    cfg['batch_size'] = args.batch_size
    cfg['alpha_perceptual'] = args.alpha_perceptual
    cfg['quality_metric'] = args.metric
    cfg['use_temporal_context'] = not args.no_temporal_context
    
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    FULL_DATA = PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap'
    FILTERED_DATA = PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap_filtered'
    CHECKPOINT_BASE = PROJECT_ROOT / 'checkpoints'
    
    # Checkpoint name includes metric
    suffix = f"_{args.metric}"
    if args.no_temporal_context:
        suffix += "_nocontext"
    
    results = {}
    
    if args.dataset in ['full', 'both']:
        val_loss, best_psnr = train_model(
            data_dir=FULL_DATA,
            checkpoint_dir=CHECKPOINT_BASE / f'multires_bestframe{suffix}_full',
            experiment_name=f'MultiRes + Best Frame ({args.metric}) (Full Dataset)',
            cfg=cfg
        )
        results['full'] = {'val_loss': val_loss, 'psnr': best_psnr}
    
    if args.dataset in ['filtered', 'both']:
        val_loss, best_psnr = train_model(
            data_dir=FILTERED_DATA,
            checkpoint_dir=CHECKPOINT_BASE / f'multires_bestframe{suffix}_filtered',
            experiment_name=f'MultiRes + Best Frame ({args.metric}) (Filtered Dataset)',
            cfg=cfg
        )
        results['filtered'] = {'val_loss': val_loss, 'psnr': best_psnr}
    
    print("\n" + "="*70)
    print(f"TRAINING SUMMARY (Best Frame Selection: {args.metric.upper()})")
    print("="*70)
    for name, res in results.items():
        print(f"{name:10s}: val_loss={res['val_loss']:.4f}, PSNR={res['psnr']:.2f} dB")
    print("="*70)


if __name__ == '__main__':
    main()
