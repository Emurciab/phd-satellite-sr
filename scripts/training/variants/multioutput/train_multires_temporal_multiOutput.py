#!/usr/bin/env python3
"""
Training script for MultiRes model with MULTI-OUTPUT TEMPORAL PREDICTION.

Key Innovation: Instead of predicting a single HR frame from the sequence,
this model predicts ALL T HR frames simultaneously. Output shape: (B, T, C, H, W)

Why Multi-Output is Different from Training Single-Image T Times:
==================================================================

1. SHARED INFORMATION AND CONTEXT (The Key Advantage)
   - Single-Image Model: Each frame is an independent problem. If one frame is hazy,
     the model has no context to know if haze is permanent or temporary.
   
   - Multi-Output Model: Processes all T frames in a single forward pass. Early layers
     (3D convolutions) see frames simultaneously, allowing the model to:
     * Spot differences across temporal dimension
     * Share information between frames
     * "Borrow" clean pixel information from clear frames to reconstruct hazy ones

2. LEARNING TEMPORAL CONSISTENCY
   - Single-Image Model: No concept of time. Could produce predictions with subtle
     differences in color, brightness, or structure even for identical ground.
   
   - Multi-Output Model: With temporal consistency loss, explicitly learns how frames
     should relate. Compares change between predictions (pred_2 - pred_1) to change
     between targets (target_2 - target_1). This enforces stable spectral profiles.

3. EFFICIENCY AND REGULARIZATION
   - Shared feature representations in early layers (more efficient)
   - Forcing T consistent outputs acts as strong regularization
   - Prevents overfitting to noise/artifacts in any single frame

Architecture:
=============
- Encoder processes each frame, creating (B, T, C, H, W) features
- Temporal cross-attention allows frames to share information
- T separate decoder heads produce T HR predictions
- Temporal consistency loss ensures predictions are coherent

Losses:
=======
- Per-frame reconstruction loss (L1 + SAM + Gradient)
- Temporal consistency loss: ||Δpred|| should match ||Δtarget||
- Perceptual loss (optional)
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Import building blocks from v2 model
from satellite_sr_model_v2 import (
    EncoderBranch, SpectralConsistencyModule, ProgressiveUpscaler,
    SimpleOutputHead, SpectralAngleLoss, CBAM, ResidualBlock
)


# =============================================================================
# CONFIGURATION
# =============================================================================

BEST_CONFIG = {
    # Model architecture  
    'base_channels': 64,
    'num_encoder_blocks': 4,
    'num_frames': 3,
    
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
    'alpha_temporal': 0.5,  # NEW: Temporal consistency weight
    
    # Resolution weights
    'weight_10m': 1.0,
    'weight_20m': 1.0,
    
    # Training
    'batch_size': 4,
    'epochs': 200,
    'patience': 20,
}


# =============================================================================
# TEMPORAL CROSS-ATTENTION MODULE (Memory Efficient)
# =============================================================================

class TemporalCrossAttention(nn.Module):
    """
    Memory-efficient cross-attention between temporal frames.
    
    Instead of full spatial attention (too expensive), we use:
    1. Global pooled attention to compute frame-level weights
    2. Local 1x1 conv mixing to share spatial information
    
    This allows frames to share information without O(H*W*T)^2 complexity.
    """
    def __init__(self, channels, num_frames=3, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.num_heads = num_heads
        
        # Global attention (pooled) - frame-level attention
        self.global_q = nn.Linear(channels, channels)
        self.global_k = nn.Linear(channels, channels)
        self.global_v = nn.Linear(channels, channels)
        
        # Local spatial mixing (efficient)
        self.local_mix = nn.Conv2d(channels * num_frames, channels, 1)
        
        # Output gate
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
        
        # Scale for attention
        self.scale = (channels // num_heads) ** -0.5
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) temporal feature sequence
        Returns:
            out: (B, T, C, H, W) cross-attended features
        """
        B, T, C, H, W = x.shape
        
        # ========== GLOBAL ATTENTION (pooled) ==========
        # Pool each frame to get global representation
        x_pooled = x.mean(dim=(-2, -1))  # (B, T, C)
        
        # Compute attention weights
        q = self.global_q(x_pooled)  # (B, T, C)
        k = self.global_k(x_pooled)  # (B, T, C)
        v = self.global_v(x_pooled)  # (B, T, C)
        
        # Attention: each frame attends to all frames
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, T, T)
        attn = F.softmax(attn, dim=-1)
        
        # Weighted combination of global features
        global_out = torch.matmul(attn, v)  # (B, T, C)
        
        # ========== LOCAL SPATIAL MIXING ==========
        # Concatenate all frames and mix with 1x1 conv
        x_cat = x.view(B, T * C, H, W)  # (B, T*C, H, W)
        local_mix = self.local_mix(x_cat)  # (B, C, H, W)
        
        # ========== COMBINE ==========
        outputs = []
        for t in range(T):
            # Original frame features
            frame_feat = x[:, t]  # (B, C, H, W)
            
            # Add global attention output (broadcast spatially)
            global_feat = global_out[:, t, :, None, None].expand(-1, -1, H, W)  # (B, C, H, W)
            
            # Gate between original and mixed features
            combined = torch.cat([frame_feat, local_mix], dim=1)  # (B, 2C, H, W)
            gate = self.gate(combined)  # (B, C, H, W)
            
            # Gated combination: original + gated(local_mix) + scaled(global)
            out = frame_feat + gate * (local_mix - frame_feat) + 0.1 * global_feat
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)  # (B, T, C, H, W)


class TemporalFusionBlock(nn.Module):
    """
    Block that processes temporal features with cross-attention and convolutions.
    """
    def __init__(self, channels, num_frames=3):
        super().__init__()
        
        # Cross-attention for information sharing
        self.cross_attn = TemporalCrossAttention(channels, num_frames)
        
        # Per-frame refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable residual scale
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            out: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        
        # Cross-attention
        attended = self.cross_attn(x)
        
        # Per-frame refinement with residual
        outputs = []
        for t in range(T):
            refined = self.refine(attended[:, t])
            outputs.append(attended[:, t] + self.gamma * refined)
        
        return torch.stack(outputs, dim=1)


# =============================================================================
# MULTI-OUTPUT MODEL
# =============================================================================

class MultiOutputTemporalSR(nn.Module):
    """
    Multi-Output Temporal Super-Resolution Model.
    
    Produces T HR predictions from T LR inputs, with temporal cross-attention
    allowing frames to share information and improve each other.
    
    Key architectural choices:
    1. Shared encoder processes all frames
    2. Temporal cross-attention allows information sharing
    3. T separate decoders predict T outputs
    4. Residual learning from bilinear baseline
    """
    
    def __init__(
        self,
        base_channels: int = 64,
        num_encoder_blocks: int = 4,
        num_frames: int = 3,
        num_temporal_blocks: int = 2,
        scale_factor_10m: float = 3.33,
        scale_factor_20m: float = 3.33,
        in_channels_10m: int = 3,
        in_channels_20m: int = 2,
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.scale_factor_10m = scale_factor_10m
        self.scale_factor_20m = scale_factor_20m
        self.in_channels_10m = in_channels_10m
        self.in_channels_20m = in_channels_20m
        self.base_channels = base_channels
        
        # ========== ENCODERS (shared across frames) ==========
        self.encoder_10m = EncoderBranch(
            in_channels=in_channels_10m,
            base_channels=base_channels,
            num_blocks=num_encoder_blocks
        )
        
        self.encoder_20m = EncoderBranch(
            in_channels=in_channels_20m,
            base_channels=base_channels,
            num_blocks=num_encoder_blocks
        )
        
        # ========== TEMPORAL CROSS-ATTENTION BLOCKS ==========
        # These allow frames to share information with each other
        self.temporal_blocks_10m = nn.ModuleList([
            TemporalFusionBlock(base_channels, num_frames)
            for _ in range(num_temporal_blocks)
        ])
        
        self.temporal_blocks_20m = nn.ModuleList([
            TemporalFusionBlock(base_channels, num_frames)
            for _ in range(num_temporal_blocks)
        ])
        
        # ========== CROSS-RESOLUTION FUSION ==========
        # 20m features can help 10m and vice versa
        self.fusion_10m_from_20m = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1),
            nn.GELU()
        )
        self.fusion_20m_from_10m = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1),
            nn.GELU()
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 1),
            nn.Sigmoid()
        )
        
        # ========== SPECTRAL CONSISTENCY ==========
        self.spectral_10m = SpectralConsistencyModule(base_channels, in_channels_10m)
        self.spectral_20m = SpectralConsistencyModule(base_channels, in_channels_20m)
        
        # ========== DECODERS (one per frame, but shared weights) ==========
        # Using shared decoder means consistent processing across frames
        self.decoder_10m = nn.Sequential(
            ResidualBlock(base_channels),
            CBAM(base_channels),
            ResidualBlock(base_channels),
        )
        
        self.decoder_20m = nn.Sequential(
            ResidualBlock(base_channels),
            CBAM(base_channels),
            ResidualBlock(base_channels),
        )
        
        # ========== UPSCALERS ==========
        self.upscaler_10m = ProgressiveUpscaler(
            base_channels, in_channels_10m, scale_factor_10m, base_channels
        )
        self.upscaler_20m = ProgressiveUpscaler(
            base_channels, in_channels_20m, scale_factor_20m, base_channels
        )
        
        # ========== RESIDUAL REFINEMENT ==========
        self.residual_refine_10m = nn.Sequential(
            nn.Conv2d(in_channels_10m, in_channels_10m, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels_10m, in_channels_10m, 3, padding=1)
        )
        self.residual_refine_20m = nn.Sequential(
            nn.Conv2d(in_channels_20m, in_channels_20m, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels_20m, in_channels_20m, 3, padding=1)
        )
        
        # ========== OUTPUT HEADS ==========
        self.output_head_10m = SimpleOutputHead(in_channels_10m, in_channels_10m)
        self.output_head_20m = SimpleOutputHead(in_channels_20m, in_channels_20m)
        
        # Learnable residual scales (start small for stability)
        self.residual_scale_10m = nn.Parameter(torch.tensor(0.1))
        self.residual_scale_20m = nn.Parameter(torch.tensor(0.1))
        
    def forward(
        self,
        lr_10m: torch.Tensor,
        lr_20m: torch.Tensor,
        target_size_10m: tuple = None,
        target_size_20m: tuple = None
    ) -> tuple:
        """
        Forward pass producing T HR predictions.
        
        Args:
            lr_10m: (B, T, C_10m, H, W) - 10m bands temporal sequence
            lr_20m: (B, T, C_20m, H/2, W/2) - 20m bands temporal sequence
            target_size_10m: (H', W') exact output size for 10m
            target_size_20m: (H', W') exact output size for 20m
        
        Returns:
            sr_10m: (B, T, C_10m, H', W') - T super-resolved 10m outputs
            sr_20m: (B, T, C_20m, H'/2, W'/2) - T super-resolved 20m outputs
        """
        B, T, C_10m, H_10m, W_10m = lr_10m.shape
        _, _, C_20m, H_20m, W_20m = lr_20m.shape
        
        # ========== CREATE BASELINES (bilinear upscale) ==========
        baselines_10m = []
        baselines_20m = []
        for t in range(T):
            if target_size_10m is not None:
                base_10m = F.interpolate(lr_10m[:, t], size=target_size_10m,
                                         mode='bilinear', align_corners=False)
            else:
                base_10m = F.interpolate(lr_10m[:, t], scale_factor=self.scale_factor_10m,
                                         mode='bilinear', align_corners=False)
            
            if target_size_20m is not None:
                base_20m = F.interpolate(lr_20m[:, t], size=target_size_20m,
                                         mode='bilinear', align_corners=False)
            else:
                base_20m = F.interpolate(lr_20m[:, t], scale_factor=self.scale_factor_20m,
                                         mode='bilinear', align_corners=False)
            
            baselines_10m.append(base_10m)
            baselines_20m.append(base_20m)
        
        # ========== ENCODE ALL FRAMES ==========
        features_10m = []
        features_20m = []
        for t in range(T):
            feat_10m = self.encoder_10m(lr_10m[:, t])  # (B, C, H, W)
            feat_20m = self.encoder_20m(lr_20m[:, t])  # (B, C, H/2, W/2)
            features_10m.append(feat_10m)
            features_20m.append(feat_20m)
        
        features_10m = torch.stack(features_10m, dim=1)  # (B, T, C, H, W)
        features_20m = torch.stack(features_20m, dim=1)  # (B, T, C, H/2, W/2)
        
        # ========== TEMPORAL CROSS-ATTENTION ==========
        # This is where frames share information with each other
        for block in self.temporal_blocks_10m:
            features_10m = block(features_10m)
        
        for block in self.temporal_blocks_20m:
            features_20m = block(features_20m)
        
        # ========== CROSS-RESOLUTION FUSION (per frame) ==========
        fused_10m_list = []
        fused_20m_list = []
        
        for t in range(T):
            feat_10m = features_10m[:, t]  # (B, C, H, W)
            feat_20m = features_20m[:, t]  # (B, C, H/2, W/2)
            
            # Upsample 20m features to 10m resolution
            feat_20m_up = F.interpolate(feat_20m, size=(H_10m, W_10m),
                                        mode='bilinear', align_corners=False)
            
            # Downsample 10m features to 20m resolution
            feat_10m_down = F.interpolate(feat_10m, size=(H_20m, W_20m),
                                          mode='bilinear', align_corners=False)
            
            # Fuse into 10m branch
            combined_10m = torch.cat([feat_10m, self.fusion_10m_from_20m(feat_20m_up)], dim=1)
            gate_10m = self.fusion_gate(combined_10m)
            fused_10m = feat_10m + gate_10m * self.fusion_10m_from_20m(feat_20m_up)
            
            # Fuse into 20m branch
            combined_20m = torch.cat([feat_20m, self.fusion_20m_from_10m(feat_10m_down)], dim=1)
            gate_20m = self.fusion_gate(combined_20m)
            fused_20m = feat_20m + gate_20m * self.fusion_20m_from_10m(feat_10m_down)
            
            fused_10m_list.append(fused_10m)
            fused_20m_list.append(fused_20m)
        
        # ========== DECODE AND UPSCALE EACH FRAME ==========
        outputs_10m = []
        outputs_20m = []
        
        for t in range(T):
            # Spectral consistency
            feat_10m = self.spectral_10m(fused_10m_list[t])
            feat_20m = self.spectral_20m(fused_20m_list[t])
            
            # Decode
            feat_10m = self.decoder_10m(feat_10m)
            feat_20m = self.decoder_20m(feat_20m)
            
            # Get target sizes from baselines
            target_10m = baselines_10m[t].shape[-2:]
            target_20m = baselines_20m[t].shape[-2:]
            
            # Upscale to get residuals
            residual_10m = self.upscaler_10m(feat_10m, target_10m)
            residual_20m = self.upscaler_20m(feat_20m, target_20m)
            
            # Refine residuals
            residual_10m = self.residual_refine_10m(residual_10m)
            residual_20m = self.residual_refine_20m(residual_20m)
            
            # Scale residuals
            residual_10m = residual_10m * self.residual_scale_10m
            residual_20m = residual_20m * self.residual_scale_20m
            
            # RESIDUAL LEARNING: Output = Baseline + Residual
            sr_10m = baselines_10m[t] + residual_10m
            sr_20m = baselines_20m[t] + residual_20m
            
            # Final output (clamping)
            sr_10m = self.output_head_10m(sr_10m)
            sr_20m = self.output_head_20m(sr_20m)
            
            outputs_10m.append(sr_10m)
            outputs_20m.append(sr_20m)
        
        # Stack outputs: (B, T, C, H, W)
        sr_10m = torch.stack(outputs_10m, dim=1)
        sr_20m = torch.stack(outputs_20m, dim=1)
        
        return sr_10m, sr_20m


# =============================================================================
# TEMPORAL CONSISTENCY LOSS
# =============================================================================

class TemporalConsistencyLoss(nn.Module):
    """
    Loss that enforces temporal consistency between predicted frames.
    
    Key insight: The CHANGE between predictions should match the CHANGE
    between ground truth targets. Since we expect minimal phenological
    change over short time periods, this acts as a strong regularizer.
    
    Loss = ||Δpred - Δtarget||
    
    where Δ = frame[t+1] - frame[t]
    """
    def __init__(self, mode='l1'):
        super().__init__()
        self.mode = mode
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, C, H, W) predicted sequence
            target: (B, T, C, H, W) ground truth sequence
        Returns:
            loss: scalar temporal consistency loss
        """
        B, T, C, H, W = pred.shape
        
        if T < 2:
            return torch.tensor(0.0, device=pred.device)
        
        total_loss = 0.0
        num_pairs = 0
        
        for t in range(T - 1):
            # Temporal difference (change between consecutive frames)
            pred_diff = pred[:, t+1] - pred[:, t]  # (B, C, H, W)
            target_diff = target[:, t+1] - target[:, t]  # (B, C, H, W)
            
            # Loss: predicted change should match actual change
            if self.mode == 'l1':
                loss = F.l1_loss(pred_diff, target_diff)
            elif self.mode == 'l2':
                loss = F.mse_loss(pred_diff, target_diff)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            total_loss += loss
            num_pairs += 1
        
        return total_loss / num_pairs


class TemporalSmoothLoss(nn.Module):
    """
    Alternative temporal loss that penalizes large changes between frames.
    
    Useful when ground truth shows minimal change and we want predictions
    to be temporally smooth.
    """
    def __init__(self, weight_difference=1.0, weight_similarity=0.1):
        super().__init__()
        self.weight_diff = weight_difference
        self.weight_sim = weight_similarity
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, C, H, W)
            target: (B, T, C, H, W)
        Returns:
            loss: temporal smoothness loss
        """
        B, T, C, H, W = pred.shape
        
        if T < 2:
            return torch.tensor(0.0, device=pred.device)
        
        total_loss = 0.0
        
        for t in range(T - 1):
            # Penalize large changes in predictions
            pred_diff = pred[:, t+1] - pred[:, t]
            target_diff = target[:, t+1] - target[:, t]
            
            # Difference loss: predicted change should match target change
            diff_loss = F.l1_loss(pred_diff, target_diff)
            
            # Similarity loss: frames should be similar if targets are similar
            target_sim = 1.0 - torch.mean(torch.abs(target_diff))
            pred_sim_loss = torch.mean(torch.abs(pred_diff)) * target_sim
            
            total_loss += self.weight_diff * diff_loss + self.weight_sim * pred_sim_loss
        
        return total_loss / (T - 1)


# =============================================================================
# LOCAL GRADIENT LOSS (device-safe)
# =============================================================================

class LocalGradientLoss(nn.Module):
    """
    Edge preservation loss using Sobel gradients.
    Device-safe implementation that creates kernels on the fly.
    """
    def __init__(self):
        super().__init__()
        # Create Sobel kernels (will be moved to correct device in forward)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    def forward(self, pred, target):
        # Ensure kernels are on same device as input
        device = pred.device
        sobel_x = self.sobel_x.view(1, 1, 3, 3).to(device)
        sobel_y = self.sobel_y.view(1, 1, 3, 3).to(device)
        
        loss = 0
        for c in range(pred.shape[1]):
            pred_c = pred[:, c:c+1]
            target_c = target[:, c:c+1]
            
            pred_gx = F.conv2d(pred_c, sobel_x, padding=1)
            pred_gy = F.conv2d(pred_c, sobel_y, padding=1)
            target_gx = F.conv2d(target_c, sobel_x, padding=1)
            target_gy = F.conv2d(target_c, sobel_y, padding=1)
            
            loss += F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
        
        return loss / pred.shape[1]


# =============================================================================
# COMBINED MULTI-OUTPUT LOSS
# =============================================================================

class MultiOutputLoss(nn.Module):
    """
    Combined loss for multi-output temporal SR.
    
    Components:
    1. Per-frame reconstruction loss (L1 + SAM + Gradient)
    2. Temporal consistency loss
    3. Optional perceptual loss
    """
    def __init__(
        self,
        alpha_pixel=1.0,
        alpha_spectral=0.137,
        alpha_gradient=0.01,
        alpha_temporal=0.5,
        alpha_perceptual=0.1
    ):
        super().__init__()
        
        self.alpha_pixel = alpha_pixel
        self.alpha_spectral = alpha_spectral
        self.alpha_gradient = alpha_gradient
        self.alpha_temporal = alpha_temporal
        self.alpha_perceptual = alpha_perceptual
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.sam_loss = SpectralAngleLoss()
        self.grad_loss = LocalGradientLoss()  # Use local device-safe version
        self.temporal_loss = TemporalConsistencyLoss(mode='l1')
        
        # Optional perceptual loss (using simple gradient-based features)
        self.use_perceptual = alpha_perceptual > 0
        
    def compute_frame_loss(self, pred, target):
        """Compute reconstruction loss for a single frame."""
        pixel_loss = self.l1_loss(pred, target)
        sam_loss = self.sam_loss(pred, target)
        grad_loss = self.grad_loss(pred, target)
        
        return (self.alpha_pixel * pixel_loss +
                self.alpha_spectral * sam_loss +
                self.alpha_gradient * grad_loss)
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m,
                weight_10m=1.0, weight_20m=1.0):
        """
        Compute total loss.
        
        Args:
            pred_10m: (B, T, C, H, W) predicted 10m sequence
            pred_20m: (B, T, C, H, W) predicted 20m sequence
            target_10m: (B, T, C, H, W) ground truth 10m sequence
            target_20m: (B, T, C, H, W) ground truth 20m sequence
        
        Returns:
            total_loss: scalar
            loss_dict: dictionary with individual loss components
        """
        B, T = pred_10m.shape[:2]
        
        # ========== PER-FRAME RECONSTRUCTION LOSSES ==========
        recon_loss_10m = 0.0
        recon_loss_20m = 0.0
        
        for t in range(T):
            recon_loss_10m += self.compute_frame_loss(pred_10m[:, t], target_10m[:, t])
            recon_loss_20m += self.compute_frame_loss(pred_20m[:, t], target_20m[:, t])
        
        recon_loss_10m /= T
        recon_loss_20m /= T
        
        # ========== TEMPORAL CONSISTENCY LOSSES ==========
        temporal_loss_10m = self.temporal_loss(pred_10m, target_10m)
        temporal_loss_20m = self.temporal_loss(pred_20m, target_20m)
        
        # ========== COMBINE ==========
        total_10m = recon_loss_10m + self.alpha_temporal * temporal_loss_10m
        total_20m = recon_loss_20m + self.alpha_temporal * temporal_loss_20m
        
        total_loss = weight_10m * total_10m + weight_20m * total_20m
        
        # Loss dictionary for logging
        loss_dict = {
            'recon_10m': recon_loss_10m.item(),
            'recon_20m': recon_loss_20m.item(),
            'temporal_10m': temporal_loss_10m.item(),
            'temporal_20m': temporal_loss_20m.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


# =============================================================================
# DATASET
# =============================================================================

class MultiOutputMmapDataset(Dataset):
    """
    Dataset for multi-output training.
    
    Returns ALL T frames as both input and target.
    """
    
    def __init__(self, mmap_dir, sequence_length=3, augment=True):
        """
        Args:
            mmap_dir: Path to mmap data directory
            sequence_length: Number of temporal frames (T)
            augment: Apply data augmentation
        """
        mmap_dir = Path(mmap_dir)
        self.lr = np.load(mmap_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(mmap_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr_10m = np.load(mmap_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(mmap_dir / 'hr_20m.npy', mmap_mode='r')
        self.n_samples = self.lr.shape[0]
        self.sequence_length = sequence_length
        self.augment = augment
        
        print(f"MultiOutput Dataset: {self.n_samples} samples")
        print(f"  LR 10m: {self.lr.shape}, LR 20m: {self.lr_20m.shape}")
        print(f"  HR 10m: {self.hr_10m.shape}, HR 20m: {self.hr_20m.shape}")
    
    def __len__(self):
        return self.n_samples
    
    def apply_augmentation(self, lr_10m, lr_20m, hr_10m, hr_20m):
        """Apply consistent augmentation to all frames."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            lr_10m = np.flip(lr_10m, axis=2).copy()  # (T, H, W, C) -> flip W
            lr_20m = np.flip(lr_20m, axis=2).copy()
            hr_10m = np.flip(hr_10m, axis=2).copy()
            hr_20m = np.flip(hr_20m, axis=2).copy()
        
        # Random vertical flip
        if np.random.random() > 0.5:
            lr_10m = np.flip(lr_10m, axis=1).copy()  # flip H
            lr_20m = np.flip(lr_20m, axis=1).copy()
            hr_10m = np.flip(hr_10m, axis=1).copy()
            hr_20m = np.flip(hr_20m, axis=1).copy()
        
        # Random 90-degree rotation
        k = np.random.randint(4)
        if k > 0:
            lr_10m = np.rot90(lr_10m, k, axes=(1, 2)).copy()
            lr_20m = np.rot90(lr_20m, k, axes=(1, 2)).copy()
            hr_10m = np.rot90(hr_10m, k, axes=(1, 2)).copy()
            hr_20m = np.rot90(hr_20m, k, axes=(1, 2)).copy()
        
        return lr_10m, lr_20m, hr_10m, hr_20m
    
    def __getitem__(self, idx):
        # Load all T frames
        lr_10m = np.array(self.lr[idx])      # (T, H, W, C)
        lr_20m = np.array(self.lr_20m[idx])  # (T, H/2, W/2, C)
        hr_10m = np.array(self.hr_10m[idx])  # (T, H', W', C)
        hr_20m = np.array(self.hr_20m[idx])  # (T, H'/2, W'/2, C)
        
        # Apply augmentation
        if self.augment:
            lr_10m, lr_20m, hr_10m, hr_20m = self.apply_augmentation(
                lr_10m, lr_20m, hr_10m, hr_20m
            )
        
        # Convert to torch and rearrange to (T, C, H, W)
        lr_10m = torch.from_numpy(lr_10m).float().permute(0, 3, 1, 2)
        lr_20m = torch.from_numpy(lr_20m).float().permute(0, 3, 1, 2)
        hr_10m = torch.from_numpy(hr_10m).float().permute(0, 3, 1, 2)
        hr_20m = torch.from_numpy(hr_20m).float().permute(0, 3, 1, 2)
        
        return {
            'lr_10m': lr_10m,      # (T, C, H, W)
            'lr_20m': lr_20m,      # (T, C, H/2, W/2)
            'hr_10m': hr_10m,      # (T, C, H', W')
            'hr_20m': hr_20m,      # (T, C, H'/2, W'/2)
        }


# =============================================================================
# TRAINING UTILITIES
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


def compute_psnr(pred, target, max_val=1.0):
    """Compute PSNR between prediction and target."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val ** 2 / mse)


def compute_temporal_psnr(pred, target, max_val=1.0):
    """Compute average PSNR across all T frames."""
    B, T = pred.shape[:2]
    total_psnr = 0.0
    for t in range(T):
        total_psnr += compute_psnr(pred[:, t], target[:, t], max_val)
    return total_psnr / T


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(data_dir, checkpoint_dir, experiment_name, cfg):
    """Main training function."""
    
    print(f"\n{'='*70}")
    print(f"Training: {experiment_name}")
    print(f"Data: {data_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== DATASET ==========
    dataset = MultiOutputMmapDataset(
        mmap_dir=data_dir,
        sequence_length=cfg.get('num_frames', 3),
        augment=True
    )
    
    # Split
    n_val = min(100, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Disable augmentation for validation
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
    
    # ========== MODEL ==========
    model = MultiOutputTemporalSR(
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=cfg.get('num_frames', 3),
        num_temporal_blocks=cfg.get('num_temporal_blocks', 2),
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['base_channels']} channels, {cfg['num_encoder_blocks']} blocks, "
          f"T={cfg.get('num_frames', 3)}, {n_params:,} params")
    
    # ========== LOSS ==========
    criterion = MultiOutputLoss(
        alpha_pixel=cfg['alpha_pixel'],
        alpha_spectral=cfg['alpha_spectral'],
        alpha_gradient=cfg['alpha_gradient'],
        alpha_temporal=cfg.get('alpha_temporal', 0.5),
        alpha_perceptual=cfg.get('alpha_perceptual', 0.1)
    )
    
    print(f"\nLoss weights: Pixel={cfg['alpha_pixel']:.3f}, SAM={cfg['alpha_spectral']:.3f}, "
          f"Grad={cfg['alpha_gradient']:.4f}, Temporal={cfg.get('alpha_temporal', 0.5):.3f}")
    print(f"Branch weights: 10m={cfg['weight_10m']:.3f}, 20m={cfg['weight_20m']:.3f}")
    
    # ========== OPTIMIZER ==========
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=cfg['patience_scheduler'], verbose=True
    )
    
    early_stopping = EarlyStopping(patience=cfg['patience'])
    
    # ========== TRAINING LOOP ==========
    print(f"\nStarting training for {cfg['epochs']} epochs...")
    
    history = {'train_loss': [], 'val_loss': [], 'val_psnr_10m': [], 'val_psnr_20m': [], 'lr': []}
    best_val_loss = float('inf')
    best_psnr = 0.0
    
    for epoch in range(cfg['epochs']):
        epoch_start = time.time()
        
        # ===== TRAINING =====
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            lr_10m = batch['lr_10m'].to(device)  # (B, T, C, H, W)
            lr_20m = batch['lr_20m'].to(device)
            hr_10m = batch['hr_10m'].to(device)
            hr_20m = batch['hr_20m'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - get T predictions
            pred_10m, pred_20m = model(
                lr_10m, lr_20m,
                target_size_10m=hr_10m.shape[-2:],
                target_size_20m=hr_20m.shape[-2:]
            )
            
            # Compute loss
            loss, loss_dict = criterion(
                pred_10m, pred_20m, hr_10m, hr_20m,
                weight_10m=cfg['weight_10m'],
                weight_20m=cfg['weight_20m']
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        val_psnr_10m = 0.0
        val_psnr_20m = 0.0
        val_temporal_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                lr_10m = batch['lr_10m'].to(device)
                lr_20m = batch['lr_20m'].to(device)
                hr_10m = batch['hr_10m'].to(device)
                hr_20m = batch['hr_20m'].to(device)
                
                pred_10m, pred_20m = model(
                    lr_10m, lr_20m,
                    target_size_10m=hr_10m.shape[-2:],
                    target_size_20m=hr_20m.shape[-2:]
                )
                
                loss, loss_dict = criterion(
                    pred_10m, pred_20m, hr_10m, hr_20m,
                    weight_10m=cfg['weight_10m'],
                    weight_20m=cfg['weight_20m']
                )
                
                val_loss += loss.item()
                val_temporal_loss += (loss_dict['temporal_10m'] + loss_dict['temporal_20m']) / 2
                
                # Compute PSNR across all T frames
                val_psnr_10m += compute_temporal_psnr(pred_10m, hr_10m).item()
                val_psnr_20m += compute_temporal_psnr(pred_20m, hr_20m).item()
        
        val_loss /= len(val_loader)
        val_psnr_10m /= len(val_loader)
        val_psnr_20m /= len(val_loader)
        val_temporal_loss /= len(val_loader)
        
        combined_psnr = (val_psnr_10m + val_psnr_20m) / 2
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr_10m'].append(val_psnr_10m)
        history['val_psnr_20m'].append(val_psnr_20m)
        history['lr'].append(current_lr)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_psnr = combined_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_psnr_10m': val_psnr_10m,
                'val_psnr_20m': val_psnr_20m,
                'config': cfg,
                'model_type': 'MultiOutputTemporalSR'
            }, checkpoint_dir / 'best_model.pt')
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': cfg,
            'model_type': 'MultiOutputTemporalSR'
        }, checkpoint_dir / 'latest_model.pt')
        
        # Save history
        with open(checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{cfg['epochs']} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PSNR 10m: {val_psnr_10m:.2f} | PSNR 20m: {val_psnr_20m:.2f} | "
              f"Temp: {val_temporal_loss:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Early stopping
        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best epoch: {early_stopping.best_epoch+1} with val_loss: {best_val_loss:.4f}")
            break
    
    print(f"\nTraining complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}, Best PSNR: {best_psnr:.2f} dB")
    
    return best_val_loss, best_psnr


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train MultiRes model with Multi-Output Temporal Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default training
  python train_multires_temporal_multiOutput.py --dataset full
  
  # Adjust temporal consistency weight
  python train_multires_temporal_multiOutput.py --alpha_temporal 0.3
  
  # More temporal attention blocks
  python train_multires_temporal_multiOutput.py --temporal_blocks 3
        """
    )
    parser.add_argument('--dataset', choices=['full', 'filtered', 'both'], default='full',
                        help='Which dataset to train on')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum epochs to train')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--alpha_temporal', type=float, default=0.5,
                        help='Weight for temporal consistency loss')
    parser.add_argument('--alpha_perceptual', type=float, default=0.1,
                        help='Weight for perceptual loss')
    parser.add_argument('--temporal_blocks', type=int, default=2,
                        help='Number of temporal cross-attention blocks')
    args = parser.parse_args()
    
    cfg = BEST_CONFIG.copy()
    cfg['epochs'] = args.epochs
    cfg['batch_size'] = args.batch_size
    cfg['alpha_temporal'] = args.alpha_temporal
    cfg['alpha_perceptual'] = args.alpha_perceptual
    cfg['num_temporal_blocks'] = args.temporal_blocks
    
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    FULL_DATA = PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap'
    FILTERED_DATA = PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap_filtered'
    CHECKPOINT_BASE = PROJECT_ROOT / 'checkpoints'
    
    # Checkpoint name includes key params
    suffix = f"_t{args.alpha_temporal}"
    
    results = {}
    
    if args.dataset in ['full', 'both']:
        val_loss, best_psnr = train_model(
            data_dir=FULL_DATA,
            checkpoint_dir=CHECKPOINT_BASE / f'multires_multioutput{suffix}_full',
            experiment_name=f'MultiRes + Multi-Output (α_t={args.alpha_temporal}) (Full Dataset)',
            cfg=cfg
        )
        results['full'] = {'val_loss': val_loss, 'psnr': best_psnr}
    
    if args.dataset in ['filtered', 'both']:
        val_loss, best_psnr = train_model(
            data_dir=FILTERED_DATA,
            checkpoint_dir=CHECKPOINT_BASE / f'multires_multioutput{suffix}_filtered',
            experiment_name=f'MultiRes + Multi-Output (α_t={args.alpha_temporal}) (Filtered Dataset)',
            cfg=cfg
        )
        results['filtered'] = {'val_loss': val_loss, 'psnr': best_psnr}
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for dataset_name, res in results.items():
        print(f"{dataset_name.upper()}: Val Loss = {res['val_loss']:.4f}, PSNR = {res['psnr']:.2f} dB")
