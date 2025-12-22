"""
Multi-Resolution Satellite Super-Resolution Model - DIRECT LEARNING (OLD ARCHITECTURE)

This is the OLD version with QKV+gamma TemporalAttention for loading old checkpoints.
The new version uses V2-style weighted average TemporalAttention.

Used by: evaluate_models.py to load old checkpoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


# =============================================================================
# BUILDING BLOCKS (same as V2 for fair comparison)
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    """Efficient convolution: 8-9x fewer parameters than standard conv."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        # GroupNorm with safe number of groups
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class ResidualBlock(nn.Module):
    """Residual block with depthwise separable convolutions."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels)
        )
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention using max and avg pooling."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# =============================================================================
# ENCODER
# =============================================================================

class MultiScaleEncoder(nn.Module):
    """Hierarchical encoder that extracts multi-scale features."""
    def __init__(self, in_channels, base_channels, num_blocks=4):
        super().__init__()
        
        # Initial conv
        self.input_conv = DepthwiseSeparableConv(in_channels, base_channels)
        
        # Residual blocks with attention
        blocks = []
        for i in range(num_blocks):
            blocks.extend([
                ResidualBlock(base_channels),
                CBAM(base_channels) if i % 2 == 1 else nn.Identity()
            ])
        self.encoder_blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = self.input_conv(x)
        x = self.encoder_blocks(x)
        return x


# =============================================================================
# TEMPORAL ATTENTION (OLD VERSION - QKV + GAMMA)
# =============================================================================

class TemporalAttention(nn.Module):
    """
    OLD VERSION: Temporal attention using Query-Key-Value mechanism with gamma residual.
    
    This is the OLD architecture used in early training runs.
    Kept for backward compatibility with old checkpoints.
    """
    def __init__(self, channels, num_frames=3):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
        # Learnable blend parameter (starts at 0 = only use middle frame)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) temporal features
        Returns:
            (B, C, H, W) attention-weighted features
        """
        B, T, C, H, W = x.shape
        
        # Use middle frame as query reference
        middle_idx = T // 2
        middle_frame = x[:, middle_idx]  # (B, C, H, W)
        
        # Compute Q, K, V
        Q = self.query(middle_frame).view(B, C, -1)  # (B, C, H*W)
        
        # Gather keys and values from all frames
        K_list = []
        V_list = []
        for t in range(T):
            K_list.append(self.key(x[:, t]).view(B, C, -1))
            V_list.append(self.value(x[:, t]).view(B, C, -1))
        
        K = torch.stack(K_list, dim=1)  # (B, T, C, H*W)
        V = torch.stack(V_list, dim=1)  # (B, T, C, H*W)
        
        # Attention scores (across temporal dimension)
        Q = Q.unsqueeze(1)  # (B, 1, C, H*W)
        attention = torch.einsum('bqcn,btcn->bqt', Q, K)  # (B, 1, T)
        attention = attention / math.sqrt(C)
        attention = F.softmax(attention, dim=-1)  # (B, 1, T)
        
        # Weighted value sum
        attention = attention.unsqueeze(-1)  # (B, 1, T, 1)
        V = V.permute(0, 1, 3, 2)  # (B, T, H*W, C)
        attended = torch.einsum('bqtk,btkc->bqkc', attention, V)  # (B, 1, H*W, C)
        attended = attended.squeeze(1).permute(0, 2, 1)  # (B, C, H*W)
        attended = attended.view(B, C, H, W)
        
        # Blend with middle frame (gamma controls how much attention contributes)
        return self.gamma * attended + middle_frame


# =============================================================================
# FEATURE FUSION (same as V2)
# =============================================================================

class CrossBranchFusion(nn.Module):
    """Fuse features between 10m and 20m branches."""
    def __init__(self, channels):
        super().__init__()
        self.attention = CBAM(channels * 2)
        self.conv = nn.Conv2d(channels * 2, channels, 1)
        
    def forward(self, feat_10m, feat_20m):
        # Align spatial dimensions
        if feat_10m.shape[-2:] != feat_20m.shape[-2:]:
            feat_20m = F.interpolate(feat_20m, size=feat_10m.shape[-2:], 
                                      mode='bilinear', align_corners=False)
        
        # Concatenate and attend
        combined = torch.cat([feat_10m, feat_20m], dim=1)
        combined = self.attention(combined)
        fused = self.conv(combined)
        
        return fused


# =============================================================================
# DECODER (DIRECT LEARNING - NO BILINEAR BASELINE)
# =============================================================================

class DirectDecoder(nn.Module):
    """
    DIRECT LEARNING DECODER
    - Directly outputs full SR image (no residual/baseline)
    - Same architecture as V2 decoder but different output interpretation
    """
    def __init__(self, in_channels, out_channels, scale_factor, base_channels):
        super().__init__()
        
        # Upsampling blocks
        upsample_layers = []
        current_channels = in_channels
        
        # Progressive upsampling
        total_scale = int(round(scale_factor))
        remaining_scale = total_scale
        
        while remaining_scale > 1:
            if remaining_scale >= 2:
                step_scale = 2
                remaining_scale = remaining_scale // 2
            else:
                step_scale = remaining_scale
                remaining_scale = 1
                
            upsample_layers.extend([
                nn.Upsample(scale_factor=step_scale, mode='bilinear', align_corners=False),
                DepthwiseSeparableConv(current_channels, current_channels),
                ResidualBlock(current_channels)
            ])
        
        self.upsample = nn.Sequential(*upsample_layers)
        
        # Final refinement and output - DIRECT SR (not residual)
        self.refine = nn.Sequential(
            ResidualBlock(current_channels),
            CBAM(current_channels),
            ResidualBlock(current_channels)
        )
        
        # Output conv - produces full SR directly
        self.output_conv = nn.Sequential(
            nn.Conv2d(current_channels, current_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(current_channels // 2, out_channels, 3, padding=1),
            nn.Sigmoid()  # Normalize to [0, 1] for direct output
        )
        
        self.scale_factor = scale_factor
        
    def forward(self, features, lr_input=None):
        """
        DIRECT LEARNING: Output full SR without bilinear baseline
        
        Args:
            features: Encoded features
            lr_input: Original LR (unused, kept for API compatibility)
        Returns:
            Full SR image directly learned (not baseline + residual)
        """
        x = self.upsample(features)
        x = self.refine(x)
        sr_output = self.output_conv(x)
        
        return sr_output


# =============================================================================
# MAIN MODEL (OLD ARCHITECTURE)
# =============================================================================

class MultiResolutionSatelliteSRDirect(nn.Module):
    """
    Multi-Resolution SR Model with DIRECT LEARNING (OLD ARCHITECTURE)
    
    Uses OLD QKV+gamma TemporalAttention for backward compatibility.
    """
    def __init__(
        self,
        in_channels_10m: int = 3,
        in_channels_20m: int = 2,
        base_channels: int = 64,
        num_encoder_blocks: int = 4,
        num_frames: int = 3,
        scale_factor_10m: float = 3.33,
        scale_factor_20m: float = 3.33,
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.scale_factor_10m = scale_factor_10m
        self.scale_factor_20m = scale_factor_20m
        
        # Encoders
        self.encoder_10m = MultiScaleEncoder(
            in_channels_10m, base_channels, num_encoder_blocks
        )
        self.encoder_20m = MultiScaleEncoder(
            in_channels_20m, base_channels, num_encoder_blocks
        )
        
        # Temporal attention (OLD VERSION with QKV + gamma)
        self.temporal_attn_10m = TemporalAttention(base_channels, num_frames)
        self.temporal_attn_20m = TemporalAttention(base_channels, num_frames)
        
        # Cross-branch fusion
        self.fusion_10m = CrossBranchFusion(base_channels)
        self.fusion_20m = CrossBranchFusion(base_channels)
        
        # Decoders (DIRECT - no bilinear baseline)
        self.decoder_10m = DirectDecoder(
            base_channels, in_channels_10m, scale_factor_10m, base_channels
        )
        self.decoder_20m = DirectDecoder(
            base_channels, in_channels_20m, scale_factor_20m, base_channels
        )
        
    def forward(
        self, 
        lr_10m: torch.Tensor,
        lr_20m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lr_10m: (B, T, C, H, W) or (B, C, H, W)
            lr_20m: (B, T, C, H, W) or (B, C, H, W)
        Returns:
            sr_10m: Super-resolved 10m bands
            sr_20m: Super-resolved 20m bands
        """
        # Handle both temporal and non-temporal inputs
        if lr_10m.dim() == 4:
            lr_10m = lr_10m.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)
            lr_20m = lr_20m.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)
        
        B, T, C_10m, H_10m, W_10m = lr_10m.shape
        _, _, C_20m, H_20m, W_20m = lr_20m.shape
        
        # Encode each frame
        enc_10m_frames = []
        enc_20m_frames = []
        for t in range(T):
            enc_10m_frames.append(self.encoder_10m(lr_10m[:, t]))
            enc_20m_frames.append(self.encoder_20m(lr_20m[:, t]))
        
        # Stack for temporal attention
        enc_10m_stack = torch.stack(enc_10m_frames, dim=1)  # (B, T, C, H, W)
        enc_20m_stack = torch.stack(enc_20m_frames, dim=1)
        
        # Apply temporal attention
        feat_10m = self.temporal_attn_10m(enc_10m_stack)
        feat_20m = self.temporal_attn_20m(enc_20m_stack)
        
        # Cross-branch fusion
        fused_10m = self.fusion_10m(feat_10m, feat_20m)
        fused_20m = self.fusion_20m(feat_20m, feat_10m)
        
        # Use middle frame as reference for LR input
        middle_idx = T // 2
        lr_10m_ref = lr_10m[:, middle_idx]
        lr_20m_ref = lr_20m[:, middle_idx]
        
        # Decode (DIRECT - no bilinear baseline)
        sr_10m = self.decoder_10m(fused_10m, lr_10m_ref)
        sr_20m = self.decoder_20m(fused_20m, lr_20m_ref)
        
        return sr_10m, sr_20m
