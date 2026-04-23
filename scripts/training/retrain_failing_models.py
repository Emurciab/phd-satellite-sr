#!/usr/bin/env python3
"""
Unified Retraining Script for Failing Models with Enhanced Monitoring

Models to retrain:
1. BestFrame Direct
2. Median Direct  
3. MultiOutput Direct
4. Temporal Residual
5. GAN Direct

Key Fixes Applied:
- Scheduled perceptual loss (starts at 0.01, ramps to 0.10 over 30 epochs)
- Per-branch PSNR monitoring (early warning for dead branches)
- Gradient norm monitoring (detect vanishing gradients)
- Enhanced validation logging

Based on analysis from architecture_analysis.md
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
from pathlib import Path
import numpy as np
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================================
# CONFIGURATION - All models use these base settings
# ============================================================================

BASE_CONFIG = {
    # Model architecture  
    'base_channels': 64,
    'num_encoder_blocks': 4,
    'num_frames': 3,
    
    # Optimizer
    'optimizer': 'adamw',
    'lr': 1e-05,  # Reduced from 3.064e-05 for stability
    'weight_decay': 5.235e-06,
    
    # Scheduler
    'patience_scheduler': 4,
    
    # Loss weights (with scheduled perceptual - THE KEY FIX)
    'alpha_pixel': 1.0,
    'alpha_spectral': 0.137,
    'alpha_gradient': 0.01,
    'alpha_perceptual_final': 0.10,    # Final perceptual weight
    'alpha_perceptual_start': 0.01,    # Initial perceptual weight (10x lower!)
    'perceptual_warmup_epochs': 30,    # Epochs to ramp up perceptual loss
    'alpha_temporal': 0.5,             # Temporal consistency (for applicable models)
    
    # Resolution weights
    'weight_10m': 1.0,
    'weight_20m': 1.0,
    
    # Training
    'batch_size': 2,
    'accumulation_steps': 2,
    'epochs': 200,
    'patience': 20,
}

# ============================================================================
# SCHEDULED PERCEPTUAL LOSS - The Critical Fix
# ============================================================================

def get_scheduled_perceptual_weight(epoch: int, cfg: dict) -> float:
    """
    Compute perceptual loss weight with gradual warmup.
    
    This is THE KEY FIX that prevents dead 10m branches.
    
    Strategy:
    - Epoch 1-10: Low perceptual loss → pixel-level accuracy first
    - Epoch 10-30: Gradual increase → adds texture without disrupting gradients
    - Epoch 30+: Full perceptual → refines high-frequency details
    
    Args:
        epoch: Current epoch (1-indexed)
        cfg: Config dict with perceptual parameters
    
    Returns:
        Current perceptual loss weight
    """
    start = cfg.get('alpha_perceptual_start', 0.01)
    final = cfg.get('alpha_perceptual_final', 0.10)
    warmup = cfg.get('perceptual_warmup_epochs', 30)
    
    if epoch <= warmup:
        # Linear ramp from start to final
        progress = epoch / warmup
        return start + progress * (final - start)
    else:
        return final


# ============================================================================
# MONITORING UTILITIES - Early Warning System
# ============================================================================

def compute_psnr(pred, target, max_val=1.0):
    """Compute PSNR between prediction and target with NaN safety."""
    # Check for NaN or Inf in inputs
    if torch.isnan(pred).any() or torch.isnan(target).any():
        return float('nan')
    if torch.isinf(pred).any() or torch.isinf(target).any():
        return float('nan')
    
    mse = F.mse_loss(pred, target)
    
    # Check for NaN in MSE
    if torch.isnan(mse) or mse < 1e-10:
        return 100.0 if mse < 1e-10 else float('nan')
    
    psnr = 10 * torch.log10(torch.tensor(max_val**2, device=mse.device) / mse)
    return psnr.item()


def check_gradient_health(model, branch_name='all'):
    """
    Check gradient norms to detect vanishing/exploding gradients.
    
    Returns dict with gradient statistics.
    """
    grad_norms = []
    zero_grads = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            total_params += 1
            if grad_norm < 1e-8:
                zero_grads += 1
    
    if len(grad_norms) == 0:
        return {'mean': 0, 'max': 0, 'min': 0, 'zero_ratio': 1.0}
    
    return {
        'mean': np.mean(grad_norms),
        'max': np.max(grad_norms),
        'min': np.min(grad_norms),
        'zero_ratio': zero_grads / total_params if total_params > 0 else 0
    }


def check_output_health(output_10m, output_20m):
    """
    Check if outputs are healthy (not constant/dead).
    
    Returns warning message if issues detected.
    """
    warnings = []
    
    # Check 10m output
    var_10m = output_10m.var().item()
    mean_10m = output_10m.mean().item()
    if var_10m < 1e-6:
        warnings.append(f"⚠️ 10m output has near-zero variance ({var_10m:.2e}) - BRANCH MAY BE DEAD!")
    if mean_10m < 0.01 or mean_10m > 0.99:
        warnings.append(f"⚠️ 10m output mean is extreme ({mean_10m:.4f}) - possible saturation!")
    
    # Check 20m output
    var_20m = output_20m.var().item()
    mean_20m = output_20m.mean().item()
    if var_20m < 1e-6:
        warnings.append(f"⚠️ 20m output has near-zero variance ({var_20m:.2e}) - BRANCH MAY BE DEAD!")
    if mean_20m < 0.01 or mean_20m > 0.99:
        warnings.append(f"⚠️ 20m output mean is extreme ({mean_20m:.4f}) - possible saturation!")
    
    return warnings


class BranchHealthMonitor:
    """
    Monitor health of each output branch during training.
    
    Tracks:
    - Per-branch PSNR history
    - Variance of outputs
    - Early warning if a branch stops improving
    """
    def __init__(self, patience=5):
        self.history_10m = []
        self.history_20m = []
        self.patience = patience
        self.stuck_10m_count = 0
        self.stuck_20m_count = 0
    
    def update(self, psnr_10m, psnr_20m, epoch):
        """Update history and check for stuck branches."""
        self.history_10m.append(psnr_10m)
        self.history_20m.append(psnr_20m)
        
        warnings = []
        
        # Check if 10m branch is stuck (PSNR not improving)
        if len(self.history_10m) >= self.patience:
            recent = self.history_10m[-self.patience:]
            if max(recent) - min(recent) < 0.05:  # Less than 0.05 dB change
                self.stuck_10m_count += 1
                if self.stuck_10m_count >= 3:
                    warnings.append(f"🚨 10m PSNR stuck at {psnr_10m:.2f} for {self.stuck_10m_count * self.patience} epochs!")
            else:
                self.stuck_10m_count = 0
        
        # Check if 20m branch is stuck
        if len(self.history_20m) >= self.patience:
            recent = self.history_20m[-self.patience:]
            if max(recent) - min(recent) < 0.05:
                self.stuck_20m_count += 1
                if self.stuck_20m_count >= 3:
                    warnings.append(f"🚨 20m PSNR stuck at {psnr_20m:.2f} for {self.stuck_20m_count * self.patience} epochs!")
            else:
                self.stuck_20m_count = 0
        
        return warnings
    
    def get_improvement_rate(self, branch='10m', window=10):
        """Get PSNR improvement rate over last N epochs."""
        history = self.history_10m if branch == '10m' else self.history_20m
        if len(history) < window:
            return None
        return history[-1] - history[-window]


# ============================================================================
# EARLY STOPPING with Branch Monitoring
# ============================================================================

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


# ============================================================================
# DATASET - Universal for all models
# ============================================================================

class UniversalMmapDataset(Dataset):
    """
    Universal dataset that works with all model types.
    Returns full temporal sequence for HR (T frames).
    """
    def __init__(self, mmap_dir, sequence_length=3, augment=True):
        mmap_dir = Path(mmap_dir)
        self.lr = np.load(mmap_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(mmap_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr_10m = np.load(mmap_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(mmap_dir / 'hr_20m.npy', mmap_mode='r')
        self.n_samples = self.lr.shape[0]
        self.sequence_length = sequence_length
        self.augment = augment
        print(f"Dataset: {self.n_samples} samples, T={sequence_length} frames")
        print(f"  LR 10m: {self.lr.shape}, LR 20m: {self.lr_20m.shape}")
        print(f"  HR 10m: {self.hr_10m.shape}, HR 20m: {self.hr_20m.shape}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        lr_10m = self.lr[idx].copy()
        lr_20m = self.lr_20m[idx].copy()
        hr_10m = self.hr_10m[idx].copy()
        hr_20m = self.hr_20m[idx].copy()
        
        if self.augment:
            import random
            # Apply SAME augmentation to all frames
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
        
        # Convert to tensors: (T, C, H, W)
        lr_10m = torch.from_numpy(lr_10m).permute(0, 3, 1, 2).float()
        lr_20m = torch.from_numpy(lr_20m).permute(0, 3, 1, 2).float()
        hr_10m = torch.from_numpy(hr_10m).permute(0, 3, 1, 2).float()
        hr_20m = torch.from_numpy(hr_20m).permute(0, 3, 1, 2).float()
        
        return {'lr_10m': lr_10m, 'lr_20m': lr_20m, 'hr_10m': hr_10m, 'hr_20m': hr_20m}


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class SpectralAngleLoss(nn.Module):
    """Spectral Angle Mapper loss for spectral fidelity with enhanced stability."""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, eps=1e-8):
        # Use F.normalize for better numerical stability
        pred_norm = F.normalize(pred, dim=1, eps=eps)
        target_norm = F.normalize(target, dim=1, eps=eps)
        
        # Compute cosine similarity
        cos_sim = (pred_norm * target_norm).sum(dim=1)
        
        # Clamp to valid range for acos with margin
        cos_sim = torch.clamp(cos_sim, -1 + 1e-6, 1 - 1e-6)
        
        # Return mean spectral angle
        angle = torch.acos(cos_sim)
        
        # Check for NaN and return 0 if detected
        if torch.isnan(angle).any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return angle.mean()


class GradientLoss(nn.Module):
    """Edge-preserving gradient loss."""
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        device = pred.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        
        B, C, H, W = pred.shape
        loss = 0
        for c in range(C):
            pred_c = pred[:, c:c+1]
            target_c = target[:, c:c+1]
            
            pred_gx = F.conv2d(pred_c, sobel_x, padding=1)
            pred_gy = F.conv2d(pred_c, sobel_y, padding=1)
            target_gx = F.conv2d(target_c, sobel_x, padding=1)
            target_gy = F.conv2d(target_c, sobel_y, padding=1)
            
            loss += F.l1_loss(pred_gx, target_gx)
            loss += F.l1_loss(pred_gy, target_gy)
        
        return loss / C


class EnhancedGradientLoss(nn.Module):
    """
    Enhanced gradient-based loss for edge/texture preservation.
    
    This is used as the "perceptual" component - like the working training scripts.
    VGG-based perceptual loss caused NaN with satellite data due to range mismatch.
    
    This loss computes:
    1. Sobel gradients in X and Y
    2. Laplacian (2nd order derivatives)
    3. L1 loss between pred and target gradients
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        # Laplacian kernel
        self.laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                       dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        device = pred.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        laplacian = self.laplacian.to(device)
        
        B, C, H, W = pred.shape
        loss = 0
        
        for c in range(C):
            pred_c = pred[:, c:c+1]
            target_c = target[:, c:c+1]
            
            # First-order gradients (edges)
            pred_gx = F.conv2d(pred_c, sobel_x, padding=1)
            pred_gy = F.conv2d(pred_c, sobel_y, padding=1)
            target_gx = F.conv2d(target_c, sobel_x, padding=1)
            target_gy = F.conv2d(target_c, sobel_y, padding=1)
            
            # Second-order (texture/fine detail)
            pred_lap = F.conv2d(pred_c, laplacian, padding=1)
            target_lap = F.conv2d(target_c, laplacian, padding=1)
            
            # Combine losses
            loss += F.l1_loss(pred_gx, target_gx)
            loss += F.l1_loss(pred_gy, target_gy)
            loss += 0.5 * F.l1_loss(pred_lap, target_lap)
        
        return loss / C


# ============================================================================
# TRAINING FUNCTION - Universal with Enhanced Monitoring
# ============================================================================

def train_model(
    model_type: str,
    data_dir: str,
    checkpoint_dir: str,
    log_file: str,
    device: str = 'cuda',
    cfg: dict = None
):
    """
    Train a model with enhanced monitoring.
    
    Args:
        model_type: One of 'bestframe_direct', 'median_direct', 'multioutput_direct',
                    'temporal_residual', 'gan_direct'
        data_dir: Path to mmap data directory
        checkpoint_dir: Path to save checkpoints
        log_file: Path to log file
        device: 'cuda' or 'cpu'
        cfg: Config dict (uses BASE_CONFIG if None)
    """
    if cfg is None:
        cfg = BASE_CONFIG.copy()
    
    # Create directories
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Import model based on type
    if model_type == 'bestframe_direct':
        sys.path.insert(0, str(Path(__file__).parent / 'variants' / 'bestframe'))
        from satellite_sr_model_direct import MultiResolutionSatelliteSRDirect
        model = MultiResolutionSatelliteSRDirect(
            base_channels=cfg['base_channels'],
            num_encoder_blocks=cfg['num_encoder_blocks']
        )
    elif model_type == 'median_direct':
        sys.path.insert(0, str(Path(__file__).parent / 'variants' / 'median'))
        from satellite_sr_model_direct import MultiResolutionSatelliteSRDirect
        model = MultiResolutionSatelliteSRDirect(
            base_channels=cfg['base_channels'],
            num_encoder_blocks=cfg['num_encoder_blocks']
        )
    elif model_type == 'multioutput_direct':
        sys.path.insert(0, str(Path(__file__).parent / 'variants' / 'multioutput'))
        from satellite_sr_model_direct import MultiOutputTemporalSRDirect
        model = MultiOutputTemporalSRDirect(
            base_channels=cfg['base_channels'],
            num_encoder_blocks=cfg['num_encoder_blocks'],
            num_frames=cfg['num_frames']
        )
    elif model_type == 'temporal_residual':
        sys.path.insert(0, str(Path(__file__).parent / 'variants' / 'temporal'))
        from satellite_sr_model_v2 import MultiResolutionSatelliteSRv2
        model = MultiResolutionSatelliteSRv2(
            base_channels=cfg['base_channels'],
            num_encoder_blocks=cfg['num_encoder_blocks']
        )
    elif model_type == 'gan_direct':
        sys.path.insert(0, str(Path(__file__).parent / 'variants' / 'GAN'))
        from satellite_sr_model_direct import MultiResolutionSatelliteSRDirect
        model = MultiResolutionSatelliteSRDirect(
            base_channels=cfg['base_channels'],
            num_encoder_blocks=cfg['num_encoder_blocks']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Setup logging with immediate flushing
    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
            f.flush()
    
    log("=" * 70)
    log(f"RETRAINING: {model_type.upper()} with Scheduled Perceptual Loss")
    log(f"Data: {data_dir}")
    log(f"Checkpoints: {checkpoint_dir}")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    log(f"Device: {device}")
    
    # Dataset
    dataset = UniversalMmapDataset(data_dir, sequence_length=cfg['num_frames'])
    n_samples = len(dataset)
    
    # Split: 90% train, 10% val
    n_val = min(500, int(0.1 * n_samples))
    n_train = n_samples - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], 
                              shuffle=True, num_workers=0, pin_memory=True)  # num_workers=0 for debugging
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'],
                            shuffle=False, num_workers=0, pin_memory=True)
    
    log(f"Train: {n_train}, Val: {n_val}")
    log(f"Model: {model_type}, {n_params:,} params")
    log("")
    log(f"Loss weights: Pixel={cfg['alpha_pixel']:.3f}, SAM={cfg['alpha_spectral']:.3f}, "
        f"Grad={cfg['alpha_gradient']:.4f}")
    log(f"Perceptual: START={cfg['alpha_perceptual_start']:.3f} → FINAL={cfg['alpha_perceptual_final']:.3f} "
        f"(warmup over {cfg['perceptual_warmup_epochs']} epochs)")
    if cfg.get('alpha_temporal', 0) > 0:
        log(f"Temporal consistency: {cfg['alpha_temporal']:.3f}")
    log(f"Branch weights: 10m={cfg['weight_10m']:.3f}, 20m={cfg['weight_20m']:.3f}")
    log("")
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], 
                           weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, 
        patience=cfg['patience_scheduler'], verbose=True
    )
    
    # Loss functions
    sam_loss = SpectralAngleLoss()
    grad_loss = GradientLoss()
    perceptual_loss = EnhancedGradientLoss().to(device)  # Gradient-based, not VGG
    
    # Monitoring
    early_stopping = EarlyStopping(patience=cfg['patience'])
    branch_monitor = BranchHealthMonitor(patience=5)
    
    best_psnr = 0
    nan_batch_count = 0  # Track consecutive NaN batches
    
    log(f"\nStarting training for {cfg['epochs']} epochs...")
    
    for epoch in range(1, cfg['epochs'] + 1):
        epoch_start = time.time()
        
        # Get scheduled perceptual weight
        alpha_perceptual = get_scheduled_perceptual_weight(epoch, cfg)
        
        # ====================== TRAINING ======================
        model.train()
        train_loss = 0
        train_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            lr_10m = batch['lr_10m'].to(device)
            lr_20m = batch['lr_20m'].to(device)
            hr_10m = batch['hr_10m'].to(device)
            hr_20m = batch['hr_20m'].to(device)
            
            # Debug first batch
            if epoch == 1 and batch_idx == 0:
                log(f"  [DEBUG] Input shapes: lr_10m={lr_10m.shape}, lr_20m={lr_20m.shape}")
                log(f"  [DEBUG] lr_10m range: [{lr_10m.min():.4f}, {lr_10m.max():.4f}], mean={lr_10m.mean():.4f}")
                log(f"  [DEBUG] lr_20m range: [{lr_20m.min():.4f}, {lr_20m.max():.4f}], mean={lr_20m.mean():.4f}")
                log(f"  [DEBUG] hr_10m range: [{hr_10m.min():.4f}, {hr_10m.max():.4f}], mean={hr_10m.mean():.4f}")
            
            # Skip batch if input has NaN
            if torch.isnan(lr_10m).any() or torch.isnan(lr_20m).any() or \
               torch.isnan(hr_10m).any() or torch.isnan(hr_20m).any():
                log(f"  [WARN] Batch {batch_idx}: NaN in input data, skipping")
                continue
            
            # For single-frame models, use middle frame
            if model_type in ['bestframe_direct', 'median_direct', 'temporal_residual', 'gan_direct']:
                hr_10m_target = hr_10m[:, 1]  # Middle frame
                hr_20m_target = hr_20m[:, 1]
            else:
                hr_10m_target = hr_10m  # All frames
                hr_20m_target = hr_20m
            
            # Forward pass
            optimizer.zero_grad()
            
            if model_type == 'multioutput_direct':
                pred_10m, pred_20m = model(lr_10m, lr_20m)
                # For multi-output, compute loss over all frames
                loss_10m = F.l1_loss(pred_10m, hr_10m_target)
                loss_20m = F.l1_loss(pred_20m, hr_20m_target)
                
                sam_10m = sam_loss(pred_10m.view(-1, *pred_10m.shape[2:]), 
                                   hr_10m_target.view(-1, *hr_10m_target.shape[2:]))
                sam_20m = sam_loss(pred_20m.view(-1, *pred_20m.shape[2:]),
                                   hr_20m_target.view(-1, *hr_20m_target.shape[2:]))
            else:
                pred_10m, pred_20m = model(lr_10m, lr_20m)
                
                # Resize predictions to match target size if needed
                if pred_10m.shape[-2:] != hr_10m_target.shape[-2:]:
                    pred_10m = F.interpolate(pred_10m, size=hr_10m_target.shape[-2:], 
                                             mode='bilinear', align_corners=False)
                if pred_20m.shape[-2:] != hr_20m_target.shape[-2:]:
                    pred_20m = F.interpolate(pred_20m, size=hr_20m_target.shape[-2:], 
                                             mode='bilinear', align_corners=False)
                
                loss_10m = F.l1_loss(pred_10m, hr_10m_target)
                loss_20m = F.l1_loss(pred_20m, hr_20m_target)
                
                sam_10m = sam_loss(pred_10m, hr_10m_target)
                sam_20m = sam_loss(pred_20m, hr_20m_target)
            
            # Gradient loss
            if model_type == 'multioutput_direct':
                grad_10m = grad_loss(pred_10m[:, 1], hr_10m_target[:, 1])
                grad_20m = grad_loss(pred_20m[:, 1], hr_20m_target[:, 1])
            else:
                grad_10m = grad_loss(pred_10m, hr_10m_target)
                grad_20m = grad_loss(pred_20m, hr_20m_target)
            
            # Perceptual loss (with scheduled weight)
            if model_type == 'multioutput_direct':
                perc_10m = perceptual_loss(pred_10m[:, 1], hr_10m_target[:, 1])
            else:
                perc_10m = perceptual_loss(pred_10m, hr_10m_target)
            
            # Debug: Log loss components for first few batches
            if epoch == 1 and batch_idx < 5:
                log(f"  [DEBUG] Batch {batch_idx}: L1_10m={loss_10m.item():.4f}, SAM_10m={sam_10m.item():.4f}, "
                    f"Grad_10m={grad_10m.item():.4f}, Perc_10m={perc_10m.item():.4f}")
                log(f"  [DEBUG] Batch {batch_idx}: L1_20m={loss_20m.item():.4f}, SAM_20m={sam_20m.item():.4f}, "
                    f"Grad_20m={grad_20m.item():.4f}")
            
            # Combine losses with NaN check
            total_10m = (cfg['alpha_pixel'] * loss_10m + 
                        cfg['alpha_spectral'] * sam_10m +
                        cfg['alpha_gradient'] * grad_10m +
                        alpha_perceptual * perc_10m)
            
            total_20m = (cfg['alpha_pixel'] * loss_20m +
                        cfg['alpha_spectral'] * sam_20m +
                        cfg['alpha_gradient'] * grad_20m)
            
            total_loss = (cfg['weight_10m'] * total_10m + 
                         cfg['weight_20m'] * total_20m)
            
            # Skip batch if NaN - with detailed debug
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_batch_count += 1
                if nan_batch_count <= 5:  # Only log first 5 NaN batches in detail
                    log(f"  ⚠️ Batch {batch_idx}: NaN/Inf loss detected!")
                    log(f"      L1_10m={loss_10m.item() if not torch.isnan(loss_10m) else 'NaN'}, "
                        f"SAM_10m={sam_10m.item() if not torch.isnan(sam_10m) else 'NaN'}, "
                        f"Grad_10m={grad_10m.item() if not torch.isnan(grad_10m) else 'NaN'}, "
                        f"Perc_10m={perc_10m.item() if not torch.isnan(perc_10m) else 'NaN'}")
                    log(f"      pred_10m range: [{pred_10m.min():.4f}, {pred_10m.max():.4f}]")
                    log(f"      pred_20m range: [{pred_20m.min():.4f}, {pred_20m.max():.4f}]")
                optimizer.zero_grad()
                continue
            
            # Temporal consistency for multi-output
            if model_type == 'multioutput_direct' and cfg.get('alpha_temporal', 0) > 0:
                temp_10m = F.mse_loss(pred_10m[:, 1:] - pred_10m[:, :-1],
                                      hr_10m_target[:, 1:] - hr_10m_target[:, :-1])
                temp_20m = F.mse_loss(pred_20m[:, 1:] - pred_20m[:, :-1],
                                      hr_20m_target[:, 1:] - hr_20m_target[:, :-1])
                total_loss += cfg['alpha_temporal'] * (temp_10m + temp_20m) / 2
            
            # Backward
            total_loss.backward()
            
            # Debug: Check gradient norms before clipping (first epoch only)
            if epoch == 1 and batch_idx < 3:
                total_grad_norm = 0.0
                max_grad_norm = 0.0
                nan_params = []
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        if np.isnan(grad_norm) or np.isinf(grad_norm):
                            nan_params.append(name)
                        else:
                            total_grad_norm += grad_norm ** 2
                            max_grad_norm = max(max_grad_norm, grad_norm)
                total_grad_norm = total_grad_norm ** 0.5
                if nan_params:
                    log(f"  [DEBUG] Batch {batch_idx}: NaN/Inf gradients in: {nan_params[:5]}...")
                log(f"  [DEBUG] Batch {batch_idx} BEFORE clip: total_grad_norm={total_grad_norm:.4f}, max={max_grad_norm:.4f}")
            
            # Gradient clipping (aggressive)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            # Progress logging every 50 batches
            if batch_idx % 50 == 0:
                log(f"  Batch {batch_idx}/{len(train_loader)}: loss={total_loss.item():.4f}")
            
            # Check for NaN in weights after step - early warning
            if batch_idx % 50 == 0:
                has_nan_weights = False
                for p in model.parameters():
                    if torch.isnan(p).any():
                        has_nan_weights = True
                        break
                if has_nan_weights:
                    log(f"  [CRITICAL] Batch {batch_idx}: NaN in weights detected! Training unstable.")
            
            # Only count valid losses
            loss_val = total_loss.item()
            if not np.isnan(loss_val):
                train_loss += loss_val * lr_10m.size(0)
                train_samples += lr_10m.size(0)
            
            # Check output health every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                warnings = check_output_health(pred_10m.detach(), pred_20m.detach())
                for w in warnings:
                    log(f"  Batch {batch_idx}: {w}")
        
        if train_samples > 0:
            train_loss /= train_samples
        else:
            train_loss = float('nan')
        
        # ====================== VALIDATION ======================
        model.eval()
        val_loss = 0
        val_psnr_10m = 0
        val_psnr_20m = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                lr_10m = batch['lr_10m'].to(device)
                lr_20m = batch['lr_20m'].to(device)
                hr_10m = batch['hr_10m'].to(device)
                hr_20m = batch['hr_20m'].to(device)
                
                if model_type in ['bestframe_direct', 'median_direct', 'temporal_residual', 'gan_direct']:
                    hr_10m_target = hr_10m[:, 1]
                    hr_20m_target = hr_20m[:, 1]
                else:
                    hr_10m_target = hr_10m
                    hr_20m_target = hr_20m
                
                if model_type == 'multioutput_direct':
                    pred_10m, pred_20m = model(lr_10m, lr_20m)
                    # Use middle frame for PSNR
                    psnr_10m = compute_psnr(pred_10m[:, 1], hr_10m_target[:, 1])
                    psnr_20m = compute_psnr(pred_20m[:, 1], hr_20m_target[:, 1])
                else:
                    pred_10m, pred_20m = model(lr_10m, lr_20m)
                    
                    # Resize predictions to match target size if needed
                    if pred_10m.shape[-2:] != hr_10m_target.shape[-2:]:
                        pred_10m = F.interpolate(pred_10m, size=hr_10m_target.shape[-2:], 
                                                 mode='bilinear', align_corners=False)
                    if pred_20m.shape[-2:] != hr_20m_target.shape[-2:]:
                        pred_20m = F.interpolate(pred_20m, size=hr_20m_target.shape[-2:], 
                                                 mode='bilinear', align_corners=False)
                    
                    psnr_10m = compute_psnr(pred_10m, hr_10m_target)
                    psnr_20m = compute_psnr(pred_20m, hr_20m_target)
                
                # Skip if NaN PSNR
                if not (np.isnan(psnr_10m) or np.isnan(psnr_20m)):
                    val_psnr_10m += psnr_10m * lr_10m.size(0)
                    val_psnr_20m += psnr_20m * lr_10m.size(0)
                    val_samples += lr_10m.size(0)
        
        if val_samples > 0:
            val_psnr_10m /= val_samples
            val_psnr_20m /= val_samples
        else:
            val_psnr_10m = 0
            val_psnr_20m = 0
        avg_psnr = (val_psnr_10m + val_psnr_20m) / 2
        
        # Update monitoring
        branch_warnings = branch_monitor.update(val_psnr_10m, val_psnr_20m, epoch)
        
        # Scheduler step
        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Log
        log(f"Epoch {epoch:3d}/{cfg['epochs']} | Train: {train_loss:.4f} | "
            f"PSNR 10m: {val_psnr_10m:.2f} | PSNR 20m: {val_psnr_20m:.2f} | "
            f"avg: {avg_psnr:.2f} | α_perc: {alpha_perceptual:.4f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Log warnings
        for w in branch_warnings:
            log(f"  {w}")
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'psnr_10m': val_psnr_10m,
                'psnr_20m': val_psnr_20m,
                'config': cfg,
            }, checkpoint_dir / 'best_model.pth')
            log(f"  ★ New best model saved! PSNR: {avg_psnr:.2f}")
        
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr_10m': val_psnr_10m,
                'psnr_20m': val_psnr_20m,
                'config': cfg,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Early stopping
        if early_stopping(train_loss, epoch):
            log(f"\nEarly stopping at epoch {epoch}. Best at epoch {early_stopping.best_epoch}")
            break
    
    log(f"\n{'='*70}")
    log(f"Training completed. Best PSNR: {best_psnr:.2f}")
    log(f"Final 10m PSNR: {val_psnr_10m:.2f}, 20m PSNR: {val_psnr_20m:.2f}")
    log(f"{'='*70}")
    
    return {
        'best_psnr': best_psnr,
        'final_psnr_10m': val_psnr_10m,
        'final_psnr_20m': val_psnr_20m,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain failing models with fixes')
    parser.add_argument('--model', type=str, required=True,
                        choices=['bestframe_direct', 'median_direct', 'multioutput_direct',
                                 'temporal_residual', 'gan_direct', 'all'],
                        help='Model type to retrain')
    parser.add_argument('--data', type=str, 
                        default='/media/eamurciab/TOSHIBA_EXT1/PhD/SR/data/sr_training/5band_v3/combined/mmap',
                        help='Path to mmap data directory')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='/media/eamurciab/TOSHIBA_EXT1/PhD/SR/checkpoints/retrain',
                        help='Base checkpoint directory')
    parser.add_argument('--log-dir', type=str,
                        default='/media/eamurciab/TOSHIBA_EXT1/PhD/SR/logs',
                        help='Log directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    
    args = parser.parse_args()
    
    # Models to retrain
    if args.model == 'all':
        models = ['bestframe_direct', 'median_direct', 'multioutput_direct',
                  'temporal_residual', 'gan_direct']
    else:
        models = [args.model]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {}
    for model_type in models:
        print(f"\n{'='*70}")
        print(f"RETRAINING: {model_type}")
        print(f"{'='*70}\n")
        
        cfg = BASE_CONFIG.copy()
        cfg['epochs'] = args.epochs
        
        checkpoint_dir = f"{args.checkpoint_dir}/{model_type}_v2_{timestamp}"
        log_file = f"{args.log_dir}/retrain_{model_type}_{timestamp}.log"
        
        result = train_model(
            model_type=model_type,
            data_dir=args.data,
            checkpoint_dir=checkpoint_dir,
            log_file=log_file,
            device=args.device,
            cfg=cfg
        )
        results[model_type] = result
    
    # Summary
    print("\n" + "="*70)
    print("RETRAINING SUMMARY")
    print("="*70)
    for model_type, result in results.items():
        print(f"{model_type:25s} | Best PSNR: {result['best_psnr']:.2f} | "
              f"10m: {result['final_psnr_10m']:.2f} | 20m: {result['final_psnr_20m']:.2f}")
