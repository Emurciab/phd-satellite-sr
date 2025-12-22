#!/usr/bin/env python3
"""
Training script for MultiRes SR with GAN for Perceptual Quality.

GAN-BASED SUPER RESOLUTION:
- Uses adversarial training to encourage perceptually realistic outputs
- Generator: MultiResolutionSatelliteSRv2 (residual learning - bilinear + learned residual)
  NOTE: Using V2 instead of Direct because Direct model has gradient issues with 10m branch
- Discriminator: PatchGAN for both 10m and 20m branches
- Hybrid loss: Pixel/Spectral fidelity + Adversarial "sharpness regularizer"

Key Strategy:
- Adversarial loss acts as a REGULARIZER, not the main objective
- Pixel and spectral fidelity losses remain DOMINANT
- Small alpha_adversarial (0.001) prevents artifacts while adding sharpness
- Gradual adversarial warmup to prevent generator collapse

Based on: train_multires_temporal.py (residual learning)
Reference: SRGAN, ESRGAN papers
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import time
from pathlib import Path
import numpy as np
import json
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Generator: Use RESIDUAL learning model (V2) which works properly
# The Direct model has vanishing gradient issues with 10m branch
from satellite_sr_model_v2 import MultiResolutionSatelliteSRv2

# Discriminator: PatchGAN for adversarial training
from discriminator import DualBranchDiscriminator, GANLoss


# =============================================================================
# Configuration with GAN-specific hyperparameters
# =============================================================================

BEST_CONFIG = {
    # Generator architecture  
    'base_channels': 64,
    'num_encoder_blocks': 4,
    
    # Generator optimizer
    'optimizer_G': 'adamw',
    'lr_G': 3.064e-05,
    'weight_decay_G': 5.235e-06,
    
    # Discriminator architecture
    'disc_base_channels': 64,
    'disc_num_layers': 3,
    'use_multiscale_disc': False,  # Single-scale is usually sufficient
    
    # Discriminator optimizer - LOWER LR than before to prevent D from getting too strong
    'optimizer_D': 'adam',
    'lr_D': 1e-05,  # Reduced from 1e-4 - slower D training prevents collapse
    'weight_decay_D': 0,
    'beta1_D': 0.0,   # TTUR: use 0.0 for discriminator (more stable)
    'beta2_D': 0.9,   # TTUR: use 0.9 for discriminator
    
    # Scheduler
    'patience_scheduler': 4,
    
    # === LOSS WEIGHTS ===
    # Fidelity losses (DOMINANT - these ensure spectral accuracy)
    'alpha_pixel': 1.0,
    'alpha_spectral': 0.5,    # INCREASED from 0.137 - stronger spectral fidelity
    'alpha_gradient': 0.01,   # Edge preservation
    'alpha_perceptual': 0.1,  # VGG perceptual
    
    # Adversarial loss (REGULARIZER - VERY small weight to prevent collapse)
    'alpha_adversarial': 0.001,  # REDUCED from 0.005 - gentler adversarial pressure
    
    # GAN configuration
    'gan_mode': 'lsgan',  # 'vanilla', 'lsgan', 'hinge' - LSGAN is most stable
    'n_critic': 1,        # Train D every n_critic iterations (1 = every iteration)
    
    # Adversarial loss warmup (gradual ramp-up instead of sudden on/off)
    'adv_warmup_epochs': 20,  # Gradually increase adversarial weight over 20 epochs
    
    # Temporal supervision weights (from direct.py)
    'alpha_temporal': 0.1,
    'supervise_all_frames': True,
    'temporal_weight_decay': 0.8,
    
    # Resolution weights
    'weight_10m': 1.0,
    'weight_20m': 1.0,
    
    # Training - reduced batch for GAN (needs more memory)
    'batch_size': 2,
    'accumulation_steps': 2,  # Effective batch = 4
    'epochs': 200,
    'patience': 25,  # Slightly more patience for GAN
    
    # GAN training schedule - GRADUAL warmup to prevent collapse
    'warmup_epochs': 10,  # Train G only for first N epochs (stabilize before adding D)
}


def get_adversarial_weight(epoch: int, cfg: dict) -> float:
    """
    Compute adversarial weight with gradual warmup.
    
    Instead of sudden on/off at warmup_epochs, gradually ramp up
    the adversarial loss over adv_warmup_epochs after warmup.
    This prevents generator collapse from sudden adversarial pressure.
    """
    warmup_epochs = cfg.get('warmup_epochs', 10)
    adv_warmup_epochs = cfg.get('adv_warmup_epochs', 20)
    base_alpha = cfg['alpha_adversarial']
    
    if epoch <= warmup_epochs:
        # Pure generator training, no adversarial
        return 0.0
    
    # Gradual ramp-up after warmup
    progress = min((epoch - warmup_epochs) / adv_warmup_epochs, 1.0)
    
    # Use cosine warmup for smoother transition
    return base_alpha * (1 - math.cos(progress * math.pi)) / 2


# =============================================================================
# Early Stopping
# =============================================================================

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


# =============================================================================
# Dataset (same as direct.py)
# =============================================================================

class TemporalFiveBandMmapDataset(Dataset):
    """Dataset that returns ALL temporal frames for HR."""
    def __init__(self, mmap_dir, sequence_length=3, augment=True):
        mmap_dir = Path(mmap_dir)
        self.lr = np.load(mmap_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(mmap_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr_10m = np.load(mmap_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(mmap_dir / 'hr_20m.npy', mmap_mode='r')
        self.n_samples = self.lr.shape[0]
        self.sequence_length = sequence_length
        self.augment = augment
        print(f"Temporal Dataset: {self.n_samples} samples, T={sequence_length} frames")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        lr_10m = self.lr[idx].copy()
        lr_20m = self.lr_20m[idx].copy()
        hr_10m = self.hr_10m[idx].copy()
        hr_20m = self.hr_20m[idx].copy()
        
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
        
        lr_10m = torch.from_numpy(lr_10m).permute(0, 3, 1, 2).float()
        lr_20m = torch.from_numpy(lr_20m).permute(0, 3, 1, 2).float()
        hr_10m = torch.from_numpy(hr_10m).permute(0, 3, 1, 2).float()
        hr_20m = torch.from_numpy(hr_20m).permute(0, 3, 1, 2).float()
        
        return {'lr_10m': lr_10m, 'lr_20m': lr_20m, 'hr_10m': hr_10m, 'hr_20m': hr_20m}


# =============================================================================
# Generator Loss (Fidelity + Adversarial)
# =============================================================================

class GANMultiResLoss(nn.Module):
    """
    Combined loss for GAN-based SR:
    - Fidelity losses: Pixel, SAM, Gradient, Perceptual (DOMINANT)
    - Adversarial loss: GAN loss from discriminator (REGULARIZER)
    
    The adversarial loss encourages the generator to produce sharper,
    more realistic textures without sacrificing spectral fidelity.
    """
    
    def __init__(self, cfg, discriminator, device):
        super().__init__()
        self.alpha_pixel = cfg['alpha_pixel']
        self.alpha_spectral = cfg['alpha_spectral']
        self.alpha_gradient = cfg['alpha_gradient']
        self.alpha_perceptual = cfg['alpha_perceptual']
        self.alpha_adversarial = cfg['alpha_adversarial']
        self.alpha_temporal = cfg.get('alpha_temporal', 0.1)
        self.weight_10m = cfg['weight_10m']
        self.weight_20m = cfg['weight_20m']
        
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
        # VGG Perceptual loss with input scaling for satellite data
        # Satellite data has mean ~0.05, ImageNet expects ~0.45
        # Scale by 2.5 to bring satellite range closer to ImageNet distribution
        from perceptual_loss import VGGPerceptualLoss
        self.perceptual_loss = VGGPerceptualLoss(input_scale=2.5)
        
        # GAN loss
        self.gan_loss = GANLoss(gan_mode=cfg['gan_mode']).to(device)
        self.discriminator = discriminator
        
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m, 
                compute_adversarial=True, adv_weight_override=None):
        """
        Compute generator loss.
        
        Args:
            pred_10m, pred_20m: SR predictions
            target_10m, target_20m: HR targets
            compute_adversarial: Whether to compute adversarial loss
            adv_weight_override: Override adversarial weight (for gradual warmup)
            
        Returns:
            total_loss, loss_dict with individual components
        """
        loss_dict = {}
        
        # Use override weight if provided, otherwise use config
        adv_weight = adv_weight_override if adv_weight_override is not None else self.alpha_adversarial
        
        # ==== 10m Branch Fidelity Losses ====
        pixel_10m = self.l1(pred_10m, target_10m) + 0.1 * self.mse(pred_10m, target_10m)
        sam_10m = self._sam_loss(pred_10m, target_10m)
        grad_10m = self._grad_loss(pred_10m, target_10m)
        perc_10m = self.perceptual_loss(pred_10m, target_10m)
        
        loss_10m = (self.alpha_pixel * pixel_10m + 
                    self.alpha_spectral * sam_10m + 
                    self.alpha_gradient * grad_10m +
                    self.alpha_perceptual * perc_10m)
        
        # ==== 20m Branch Fidelity Losses ====
        pixel_20m = self.l1(pred_20m, target_20m) + 0.1 * self.mse(pred_20m, target_20m)
        sam_20m = self._sam_loss(pred_20m, target_20m)
        grad_20m = self._grad_loss(pred_20m, target_20m)
        
        loss_20m = (self.alpha_pixel * pixel_20m + 
                    self.alpha_spectral * sam_20m + 
                    self.alpha_gradient * grad_20m)
        
        # ==== Fidelity Total ====
        fidelity_loss = self.weight_10m * loss_10m + self.weight_20m * loss_20m
        
        loss_dict['pixel_10m'] = pixel_10m.item()
        loss_dict['pixel_20m'] = pixel_20m.item()
        loss_dict['sam_10m'] = sam_10m.item()
        loss_dict['sam_20m'] = sam_20m.item()
        loss_dict['perc_10m'] = perc_10m.item()
        loss_dict['fidelity'] = fidelity_loss.item()
        
        # ==== Adversarial Loss (Generator wants D to say "real") ====
        adv_loss = torch.tensor(0.0, device=pred_10m.device)
        
        if compute_adversarial and adv_weight > 0:
            # Get discriminator predictions for fake (SR) images
            # Don't need gradients through D for generator update
            disc_fake_10m, disc_fake_20m = self.discriminator(pred_10m, pred_20m)
            
            # Generator wants discriminator to classify fakes as real
            adv_loss_10m = self.gan_loss(disc_fake_10m, target_is_real=True, for_discriminator=False)
            adv_loss_20m = self.gan_loss(disc_fake_20m, target_is_real=True, for_discriminator=False)
            
            adv_loss = self.weight_10m * adv_loss_10m + self.weight_20m * adv_loss_20m
            
            loss_dict['adv_10m'] = adv_loss_10m.item()
            loss_dict['adv_20m'] = adv_loss_20m.item()
            loss_dict['adv_total'] = adv_loss.item()
        
        # ==== Total Generator Loss (use current adv_weight, not config) ====
        total_loss = fidelity_loss + adv_weight * adv_loss
        loss_dict['total'] = total_loss.item()
        loss_dict['adv_weight_used'] = adv_weight  # Log the weight used
        
        return total_loss, loss_dict
    
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


# =============================================================================
# Metrics
# =============================================================================

def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)).item()


# =============================================================================
# GAN Training Loop
# =============================================================================

def train_epoch_gan(generator, discriminator, loader, criterion_G, gan_loss_fn,
                    optimizer_G, optimizer_D, device, cfg, epoch):
    """
    GAN training loop with alternating Generator/Discriminator updates.
    
    Training Strategy:
    1. Warmup: First N epochs train G only (no adversarial loss)
    2. Gradual Ramp-up: Slowly increase adversarial weight over adv_warmup_epochs
    3. Main: Alternate between D and G updates
       - D update: Real vs Fake classification
       - G update: Fidelity + Adversarial loss (gradually increasing)
    
    Uses gradient accumulation for memory efficiency.
    """
    generator.train()
    discriminator.train()
    
    total_loss_G = 0
    total_loss_D = 0
    accumulation_steps = cfg.get('accumulation_steps', 1)
    n_critic = cfg.get('n_critic', 1)
    warmup_epochs = cfg.get('warmup_epochs', 10)
    
    # Get current adversarial weight (gradual warmup)
    current_adv_weight = get_adversarial_weight(epoch, cfg)
    
    # During warmup, don't train discriminator
    in_warmup = epoch <= warmup_epochs
    
    optimizer_G.zero_grad()
    optimizer_D.zero_grad()
    
    loss_dict_accum = {}
    loss_dict_accum['adv_weight'] = current_adv_weight  # Track for logging
    
    for batch_idx, batch in enumerate(loader):
        lr_10m = batch['lr_10m'].to(device)  # (B, T, C, H, W)
        lr_20m = batch['lr_20m'].to(device)
        hr_10m = batch['hr_10m'].to(device)
        hr_20m = batch['hr_20m'].to(device)
        
        B, T, C_10m, H_lr, W_lr = lr_10m.shape
        _, _, _, H_hr_10m, W_hr_10m = hr_10m.shape
        _, _, _, H_hr_20m, W_hr_20m = hr_20m.shape
        mid_idx = T // 2
        
        hr_10m_mid = hr_10m[:, mid_idx]
        hr_20m_mid = hr_20m[:, mid_idx]
        
        # ================================================================
        # DISCRIMINATOR UPDATE (if not in warmup)
        # ================================================================
        loss_D = torch.tensor(0.0, device=device)
        
        if not in_warmup and (batch_idx % n_critic == 0):
            # Generate fake images (no grad for G during D update)
            with torch.no_grad():
                pred_10m, pred_20m = generator(lr_10m, lr_20m,
                                               target_size_10m=(H_hr_10m, W_hr_10m),
                                               target_size_20m=(H_hr_20m, W_hr_20m))
            
            # Discriminator on real images
            disc_real_10m, disc_real_20m = discriminator(hr_10m_mid, hr_20m_mid)
            loss_D_real_10m = gan_loss_fn(disc_real_10m, target_is_real=True)
            loss_D_real_20m = gan_loss_fn(disc_real_20m, target_is_real=True)
            loss_D_real = (cfg['weight_10m'] * loss_D_real_10m + 
                          cfg['weight_20m'] * loss_D_real_20m)
            
            # Discriminator on fake images
            disc_fake_10m, disc_fake_20m = discriminator(pred_10m.detach(), pred_20m.detach())
            loss_D_fake_10m = gan_loss_fn(disc_fake_10m, target_is_real=False)
            loss_D_fake_20m = gan_loss_fn(disc_fake_20m, target_is_real=False)
            loss_D_fake = (cfg['weight_10m'] * loss_D_fake_10m + 
                          cfg['weight_20m'] * loss_D_fake_20m)
            
            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D_scaled = loss_D / accumulation_steps
            loss_D_scaled.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_D.step()
                optimizer_D.zero_grad()
            
            total_loss_D += loss_D.item()
        
        # ================================================================
        # GENERATOR UPDATE
        # ================================================================
        
        # Forward pass through generator
        pred_10m, pred_20m = generator(lr_10m, lr_20m,
                                       target_size_10m=(H_hr_10m, W_hr_10m),
                                       target_size_20m=(H_hr_20m, W_hr_20m))
        
        # Compute generator loss (fidelity + adversarial with current weight)
        loss_G, loss_dict = criterion_G(pred_10m, pred_20m, hr_10m_mid, hr_20m_mid,
                                        compute_adversarial=not in_warmup,
                                        adv_weight_override=current_adv_weight)
        
        loss_G_scaled = loss_G / accumulation_steps
        loss_G_scaled.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer_G.step()
            optimizer_G.zero_grad()
        
        total_loss_G += loss_G.item()
        
        # Accumulate loss dict
        for k, v in loss_dict.items():
            if k != 'adv_weight':  # Don't accumulate the weight
                loss_dict_accum[k] = loss_dict_accum.get(k, 0) + v
    
    # Handle remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer_G.step()
        optimizer_G.zero_grad()
        if not in_warmup:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer_D.step()
            optimizer_D.zero_grad()
    
    n_batches = len(loader)
    loss_dict_avg = {k: v / n_batches if k != 'adv_weight' else v for k, v in loss_dict_accum.items()}
    
    return total_loss_G / n_batches, total_loss_D / n_batches, loss_dict_avg


def validate_gan(generator, loader, device, cfg):
    """
    Validation using ONLY fidelity metrics (PSNR, etc.).
    
    Discriminator is NOT used during validation - it's only for training.
    We evaluate the generator's reconstruction quality, not perceptual realism.
    """
    generator.eval()
    total_loss = 0
    psnr_10m_sum, psnr_20m_sum = 0, 0
    
    l1 = nn.L1Loss()
    
    with torch.no_grad():
        for batch in loader:
            lr_10m = batch['lr_10m'].to(device)
            lr_20m = batch['lr_20m'].to(device)
            hr_10m = batch['hr_10m'].to(device)
            hr_20m = batch['hr_20m'].to(device)
            
            B, T, _, H_hr_10m, W_hr_10m = hr_10m.shape
            _, _, _, H_hr_20m, W_hr_20m = hr_20m.shape
            mid_idx = T // 2
            
            hr_10m_mid = hr_10m[:, mid_idx]
            hr_20m_mid = hr_20m[:, mid_idx]
            
            pred_10m, pred_20m = generator(lr_10m, lr_20m,
                                           target_size_10m=(H_hr_10m, W_hr_10m),
                                           target_size_20m=(H_hr_20m, W_hr_20m))
            
            # Simple L1 loss for validation
            loss = l1(pred_10m, hr_10m_mid) + l1(pred_20m, hr_20m_mid)
            total_loss += loss.item()
            
            psnr_10m_sum += calculate_psnr(pred_10m, hr_10m_mid)
            psnr_20m_sum += calculate_psnr(pred_20m, hr_20m_mid)
    
    n = len(loader)
    return total_loss / n, psnr_10m_sum / n, psnr_20m_sum / n


# =============================================================================
# Main Training Function
# =============================================================================

def train_model(data_dir, checkpoint_dir, experiment_name, cfg=BEST_CONFIG):
    """Train GAN-based SR model."""
    
    print(f"\n{'='*70}")
    print(f"Training: {experiment_name}")
    print(f"Model Type: GAN-based SR (Direct Generator + PatchGAN Discriminator)")
    print(f"Data: {data_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    dataset = TemporalFiveBandMmapDataset(data_dir, augment=True)
    
    n_val = max(int(len(dataset) * 0.1), 10)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.augment = False
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # ================================================================
    # Create Generator (Residual learning SR model - V2)
    # Using V2 instead of Direct because Direct has gradient issues
    # ================================================================
    generator = MultiResolutionSatelliteSRv2(
        in_channels_10m=3,
        in_channels_20m=2,
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=3
    ).to(device)
    
    n_params_G = sum(p.numel() for p in generator.parameters())
    print(f"Generator (V2 Residual): {cfg['base_channels']} channels, {cfg['num_encoder_blocks']} blocks, {n_params_G:,} params")
    
    # ================================================================
    # Create Discriminator (PatchGAN for both branches)
    # ================================================================
    discriminator = DualBranchDiscriminator(
        channels_10m=3,
        channels_20m=2,
        base_channels=cfg['disc_base_channels'],
        num_layers=cfg['disc_num_layers'],
        use_multiscale=cfg['use_multiscale_disc']
    ).to(device)
    
    n_params_D = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator: {cfg['disc_base_channels']} channels, {cfg['disc_num_layers']} layers, {n_params_D:,} params")
    print(f"Total parameters: {n_params_G + n_params_D:,}")
    
    # ================================================================
    # Loss and Optimizers
    # ================================================================
    
    # Generator loss (fidelity + adversarial)
    criterion_G = GANMultiResLoss(cfg, discriminator, device).to(device)
    
    # GAN loss function (for discriminator)
    gan_loss_fn = GANLoss(gan_mode=cfg['gan_mode']).to(device)
    
    # Optimizers
    optimizer_G = optim.AdamW(generator.parameters(), lr=cfg['lr_G'], 
                              weight_decay=cfg['weight_decay_G'])
    optimizer_D = optim.Adam(discriminator.parameters(), lr=cfg['lr_D'],
                             betas=(cfg['beta1_D'], cfg['beta2_D']),
                             weight_decay=cfg['weight_decay_D'])
    
    # Schedulers (reduce LR on plateau)
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode='min', factor=0.5, patience=cfg['patience_scheduler'])
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.5, patience=cfg['patience_scheduler'])
    
    early_stopping = EarlyStopping(patience=cfg['patience'])
    
    best_val_loss = float('inf')
    best_psnr = 0
    history = []
    
    print(f"\nLoss weights:")
    print(f"  Fidelity: Pixel={cfg['alpha_pixel']:.3f}, SAM={cfg['alpha_spectral']:.3f}, "
          f"Grad={cfg['alpha_gradient']:.4f}, Perc={cfg['alpha_perceptual']:.3f}")
    print(f"  Adversarial: {cfg['alpha_adversarial']:.4f} (GAN mode: {cfg['gan_mode']})")
    print(f"  Warmup epochs: {cfg['warmup_epochs']} (G-only training)")
    print(f"Batch size: {cfg['batch_size']}, Accumulation: {cfg['accumulation_steps']}")
    print(f"\nStarting training for {cfg['epochs']} epochs...")
    
    for epoch in range(1, cfg['epochs'] + 1):
        start = time.time()
        
        # Training
        train_loss_G, train_loss_D, loss_dict = train_epoch_gan(
            generator, discriminator, train_loader, criterion_G, gan_loss_fn,
            optimizer_G, optimizer_D, device, cfg, epoch
        )
        
        # Validation (G only, no adversarial)
        val_loss, psnr_10m, psnr_20m = validate_gan(generator, val_loader, device, cfg)
        avg_psnr = (psnr_10m + psnr_20m) / 2
        
        # Step schedulers
        scheduler_G.step(val_loss)
        scheduler_D.step(val_loss)
        
        elapsed = time.time() - start
        lr_G = optimizer_G.param_groups[0]['lr']
        lr_D = optimizer_D.param_groups[0]['lr']
        
        history.append({
            'epoch': epoch, 
            'train_loss_G': train_loss_G, 
            'train_loss_D': train_loss_D,
            'val_loss': val_loss,
            'psnr_10m': psnr_10m, 
            'psnr_20m': psnr_20m, 
            'avg_psnr': avg_psnr, 
            'lr_G': lr_G,
            'lr_D': lr_D,
            **{f'loss_{k}': v for k, v in loss_dict.items()}
        })
        
        # Save best model (based on validation fidelity, not adversarial)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'val_loss': val_loss,
                'psnr_10m': psnr_10m,
                'psnr_20m': psnr_20m,
                'config': cfg,
                'model_type': 'gan'
            }, checkpoint_dir / 'best_model.pt')
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'val_loss': val_loss,
            'config': cfg,
            'model_type': 'gan'
        }, checkpoint_dir / 'latest_model.pt')
        
        # Logging with SAM and adversarial weight info
        warmup_tag = " [WARMUP]" if epoch <= cfg['warmup_epochs'] else ""
        adv_weight_used = loss_dict.get('adv_weight', 0)
        adv_str = f"Adv={loss_dict.get('adv_total', 0):.4f}(w={adv_weight_used:.4f})" if 'adv_total' in loss_dict else f"Adv=N/A(w={adv_weight_used:.4f})"
        sam_str = f"SAM:10m={loss_dict.get('sam_10m', 0):.3f},20m={loss_dict.get('sam_20m', 0):.3f}"
        
        print(f"Epoch {epoch:3d}/{cfg['epochs']}{warmup_tag} | "
              f"G: {train_loss_G:.4f} D: {train_loss_D:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"PSNR: 10m={psnr_10m:.2f} 20m={psnr_20m:.2f} | {sam_str} | {adv_str} | "
              f"LR: G={lr_G:.2e} | {elapsed:.1f}s")
        
        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping at epoch {epoch}. Best: epoch {early_stopping.best_epoch}")
            break
    
    # Save history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training complete: {experiment_name}")
    print(f"Best val loss: {best_val_loss:.4f}, Best PSNR: {best_psnr:.2f} dB")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"{'='*70}\n")
    
    return best_val_loss, best_psnr


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train GAN-based MultiRes SR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train_multires_temporal_GAN.py --dataset full
  
  # Train with stronger adversarial loss (more sharpness, risk of artifacts)
  python train_multires_temporal_GAN.py --alpha_adversarial 0.01 --dataset full
  
  # Train with weaker adversarial loss (safer, less sharpness)
  python train_multires_temporal_GAN.py --alpha_adversarial 0.001 --dataset full
  
  # Longer warmup (more stable G before adding D)
  python train_multires_temporal_GAN.py --warmup_epochs 10 --dataset full
        """
    )
    parser.add_argument('--dataset', choices=['full', 'filtered', 'both'], default='both',
                        help='Which dataset to train on')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum epochs to train')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--alpha_adversarial', type=float, default=0.005,
                        help='Weight for adversarial loss (0.001-0.01 recommended)')
    parser.add_argument('--alpha_perceptual', type=float, default=0.1,
                        help='Weight for perceptual loss')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Train G only for this many epochs before adding D')
    parser.add_argument('--gan_mode', choices=['vanilla', 'lsgan', 'hinge'], default='lsgan',
                        help='GAN loss type (lsgan most stable)')
    args = parser.parse_args()
    
    cfg = BEST_CONFIG.copy()
    cfg['epochs'] = args.epochs
    cfg['batch_size'] = args.batch_size
    cfg['accumulation_steps'] = args.accumulation_steps
    cfg['alpha_adversarial'] = args.alpha_adversarial
    cfg['alpha_perceptual'] = args.alpha_perceptual
    cfg['warmup_epochs'] = args.warmup_epochs
    cfg['gan_mode'] = args.gan_mode
    
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    FULL_DATA = PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap'
    FILTERED_DATA = PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap_filtered'
    CHECKPOINT_BASE = PROJECT_ROOT / 'checkpoints'
    
    results = {}
    
    if args.dataset in ['full', 'both']:
        val_loss, best_psnr = train_model(
            data_dir=FULL_DATA,
            checkpoint_dir=CHECKPOINT_BASE / 'multires_GAN_full',
            experiment_name='MultiRes GAN SR (Full Dataset)',
            cfg=cfg
        )
        results['full'] = {'val_loss': val_loss, 'psnr': best_psnr}
    
    if args.dataset in ['filtered', 'both']:
        val_loss, best_psnr = train_model(
            data_dir=FILTERED_DATA,
            checkpoint_dir=CHECKPOINT_BASE / 'multires_GAN_filtered',
            experiment_name='MultiRes GAN SR (Filtered Dataset)',
            cfg=cfg
        )
        results['filtered'] = {'val_loss': val_loss, 'psnr': best_psnr}
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY (GAN-based SR)")
    print("="*70)
    for name, res in results.items():
        print(f"{name:10s}: val_loss={res['val_loss']:.4f}, PSNR={res['psnr']:.2f} dB")
    print("="*70)


if __name__ == '__main__':
    main()
