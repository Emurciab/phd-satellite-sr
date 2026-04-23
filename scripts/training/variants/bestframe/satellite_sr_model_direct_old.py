"""
Multi-Resolution Satellite Super-Resolution Model - DIRECT LEARNING (OLD ARCHITECTURE)

This version uses the OLD TemporalAttention (gamma/QKV style) for loading checkpoints
trained before the TemporalAttention refactor.

Checkpoint signature:
- Has upscaler_10m, upscaler_20m (upscaler-style)
- Has spectral_consistency_* modules
- Has temporal_attn_*.gamma, *.query, *.key, *.value (old QKV style)
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
# TEMPORAL ATTENTION - OLD VERSION (gamma/QKV style)
# =============================================================================

class TemporalAttention(nn.Module):
    """
    OLD-STYLE temporal attention using QKV with learnable gamma.
    
    This matches checkpoints trained before the TemporalAttention refactor.
    Checkpoint keys: gamma, query.weight, query.bias, key.weight, key.bias, value.weight, value.bias
    
    Key shapes from checkpoint:
    - query.weight: [8, 64, 1, 1] (num_heads=8, channels=64)
    - key.weight: [8, 64, 1, 1]
    - value.weight: [64, 64, 1, 1] (full channels for value)
    """
    def __init__(self, channels, num_frames=3, num_heads=8):
        super().__init__()
        self.num_frames = num_frames
        self.channels = channels
        self.num_heads = num_heads
        
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # QKV projections - num_heads for Q/K, full channels for V
        self.query = nn.Conv2d(channels, num_heads, 1)  # (B*T, heads, H, W)
        self.key = nn.Conv2d(channels, num_heads, 1)    # (B*T, heads, H, W)
        self.value = nn.Conv2d(channels, channels, 1)   # Full channels for value
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) temporal features
        Returns:
            fused: (B, C, H, W) attention-weighted fusion
        """
        B, T, C, H, W = x.shape
        
        # Use middle frame as base
        middle_idx = T // 2
        base = x[:, middle_idx]  # (B, C, H, W)
        
        # Stack all frames for batch processing
        x_flat = x.view(B * T, C, H, W)
        
        # Compute Q, K, V
        Q = self.query(x_flat)  # (B*T, heads, H, W)
        K = self.key(x_flat)    # (B*T, heads, H, W)
        V = self.value(x_flat)  # (B*T, C, H, W)
        
        # Reshape for attention computation
        Q = Q.view(B, T, self.num_heads, H * W)  # (B, T, heads, H*W)
        K = K.view(B, T, self.num_heads, H * W)  # (B, T, heads, H*W)
        V = V.view(B, T, C, H, W)                # (B, T, C, H, W)
        
        # Compute attention scores: middle frame query against all frame keys
        Q_mid = Q[:, middle_idx]  # (B, heads, H*W)
        
        # Attention: Q_mid (B, heads, H*W) @ K (B, T, heads, H*W) -> (B, heads, T)
        # Using einsum: for each head, compute dot product over spatial dimension
        attn_scores = torch.einsum('bhw,bthw->bht', Q_mid, K)  # (B, heads, T)
        attn_scores = attn_scores / math.sqrt(H * W)
        attn = F.softmax(attn_scores, dim=-1)  # (B, heads, T)
        
        # Average attention across heads
        attn_avg = attn.mean(dim=1)  # (B, T)
        attn_avg = attn_avg.view(B, T, 1, 1, 1)  # (B, T, 1, 1, 1)
        
        # Weighted sum of values
        attended = (V * attn_avg).sum(dim=1)  # (B, C, H, W)
        
        # Learnable residual connection (gamma starts at 0)
        fused = base + self.gamma * (attended - base)
        
        return fused


# =============================================================================
# SPECTRAL CONSISTENCY
# =============================================================================

class SpectralConsistencyModule(nn.Module):
    """Ensures spectral relationships are preserved during SR."""
    def __init__(self, channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels)
        )
    
    def forward(self, x):
        return x + self.conv(x)


# =============================================================================
# UPSAMPLER
# =============================================================================

class ProgressiveUpscaler(nn.Module):
    """
    Progressive upsampling with learned refinement.
    NO BILINEAR BASELINE - learns full upsampling.
    """
    def __init__(self, in_channels, out_channels, scale_factor, num_features=64):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Calculate number of 2x upsampling steps needed
        num_upsample = int(math.log2(scale_factor))
        remainder = scale_factor / (2 ** num_upsample)
        
        layers = []
        current_channels = in_channels
        
        # Progressive 2x upsampling
        for i in range(num_upsample):
            layers.extend([
                nn.Conv2d(current_channels, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),  # 2x upsampling
                nn.GELU(),
                ResidualBlock(num_features)
            ])
            current_channels = num_features
        
        # Handle non-power-of-2 scaling
        if remainder > 1.0:
            layers.append(nn.Upsample(scale_factor=remainder, mode='bilinear', align_corners=False))
            layers.extend([
                nn.Conv2d(current_channels, num_features, 3, padding=1),
                nn.GELU(),
                ResidualBlock(num_features)
            ])
        
        # Final refinement
        layers.extend([
            ResidualBlock(num_features),
            nn.Conv2d(num_features, out_channels, 3, padding=1)
        ])
        
        self.upsampler = nn.Sequential(*layers)
    
    def forward(self, x, target_size=None):
        """
        Args:
            x: (B, C, H, W) feature map
            target_size: (H', W') optional exact output size
        Returns:
            upsampled: (B, out_channels, H', W')
        """
        x = self.upsampler(x)
        
        # Ensure exact output size if specified
        if target_size is not None and x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


# =============================================================================
# OUTPUT HEAD
# =============================================================================

class SimpleOutputHead(nn.Module):
    """
    Simple output head with soft clamping.
    NO SIGMOID (causes mode collapse).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.conv(x)
        # Soft clamping to [0, 1] range
        return torch.clamp(x, 0.0, 1.0)


# =============================================================================
# MAIN MODEL: DIRECT LEARNING (OLD ARCHITECTURE)
# =============================================================================

class MultiResolutionSatelliteSRDirect(nn.Module):
    """
    Direct learning super-resolution model (OLD ARCHITECTURE).
    
    This version uses the old TemporalAttention (gamma/QKV) style
    for loading checkpoints trained before the refactor.
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
        use_cross_resolution_fusion: bool = True
    ):
        super().__init__()
        
        self.in_channels_10m = in_channels_10m
        self.in_channels_20m = in_channels_20m
        self.scale_factor_10m = scale_factor_10m
        self.scale_factor_20m = scale_factor_20m
        self.num_frames = num_frames
        self.use_cross_resolution_fusion = use_cross_resolution_fusion
        
        # ========== ENCODERS ==========
        self.encoder_10m = MultiScaleEncoder(in_channels_10m, base_channels, num_encoder_blocks)
        self.encoder_20m = MultiScaleEncoder(in_channels_20m, base_channels, num_encoder_blocks)
        
        # ========== TEMPORAL ATTENTION (OLD STYLE) ==========
        self.temporal_attn_10m = TemporalAttention(base_channels, num_frames)
        self.temporal_attn_20m = TemporalAttention(base_channels, num_frames)
        
        # ========== CROSS-RESOLUTION FUSION ==========
        if use_cross_resolution_fusion:
            self.fusion_10m_from_20m = nn.Conv2d(base_channels, base_channels, 1)
            self.fusion_20m_from_10m = nn.Conv2d(base_channels, base_channels, 1)
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels, 1),
                nn.Sigmoid()
            )
        
        # ========== SPECTRAL CONSISTENCY ==========
        self.spectral_consistency_10m = SpectralConsistencyModule(base_channels, in_channels_10m)
        self.spectral_consistency_20m = SpectralConsistencyModule(base_channels, in_channels_20m)
        
        # ========== DIRECT UPSAMPLING (NO RESIDUAL) ==========
        self.upscaler_10m = ProgressiveUpscaler(
            base_channels, in_channels_10m, scale_factor_10m, base_channels
        )
        self.upscaler_20m = ProgressiveUpscaler(
            base_channels, in_channels_20m, scale_factor_20m, base_channels
        )
        
        # ========== FINAL REFINEMENT ==========
        self.final_refine_10m = nn.Sequential(
            nn.Conv2d(in_channels_10m, in_channels_10m, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels_10m, in_channels_10m, 3, padding=1)
        )
        self.final_refine_20m = nn.Sequential(
            nn.Conv2d(in_channels_20m, in_channels_20m, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels_20m, in_channels_20m, 3, padding=1)
        )
        
        # ========== OUTPUT HEADS ==========
        self.output_head_10m = SimpleOutputHead(in_channels_10m, in_channels_10m)
        self.output_head_20m = SimpleOutputHead(in_channels_20m, in_channels_20m)
    
    def forward(
        self,
        lr_10m: torch.Tensor,
        lr_20m: torch.Tensor,
        target_size_10m: Tuple[int, int] = None,
        target_size_20m: Tuple[int, int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lr_10m: (B, T, 3, H, W) - 10m bands temporal sequence
            lr_20m: (B, T, 2, H/2, W/2) - 20m bands temporal sequence
            target_size_10m: (H', W') exact output size for 10m bands
            target_size_20m: (H', W') exact output size for 20m bands
        
        Returns:
            sr_10m: (B, 3, H', W') - Super-resolved 10m bands
            sr_20m: (B, 2, H'/2, W'/2) - Super-resolved 20m bands
        """
        B, T, _, H_10m, W_10m = lr_10m.shape
        _, _, _, H_20m, W_20m = lr_20m.shape
        
        # ========== ENCODE ALL FRAMES ==========
        features_10m = []
        for t in range(T):
            feat = self.encoder_10m(lr_10m[:, t])
            features_10m.append(feat)
        features_10m = torch.stack(features_10m, dim=1)  # (B, T, C, H, W)
        
        features_20m = []
        for t in range(T):
            feat = self.encoder_20m(lr_20m[:, t])
            features_20m.append(feat)
        features_20m = torch.stack(features_20m, dim=1)  # (B, T, C, H/2, W/2)
        
        # ========== TEMPORAL ATTENTION ==========
        fused_10m = self.temporal_attn_10m(features_10m)  # (B, C, H, W)
        fused_20m = self.temporal_attn_20m(features_20m)  # (B, C, H/2, W/2)
        
        # ========== CROSS-RESOLUTION FUSION ==========
        if self.use_cross_resolution_fusion:
            # Upsample 20m features to 10m resolution for fusion
            fused_20m_up = F.interpolate(fused_20m, size=(H_10m, W_10m), 
                                         mode='bilinear', align_corners=False)
            cross_20m_to_10m = self.fusion_10m_from_20m(fused_20m_up)
            
            # Downsample 10m features to 20m resolution for fusion
            fused_10m_down = F.interpolate(fused_10m, size=(H_20m, W_20m),
                                           mode='bilinear', align_corners=False)
            cross_10m_to_20m = self.fusion_20m_from_10m(fused_10m_down)
            
            # Gated fusion
            gate_10m = self.fusion_gate(torch.cat([fused_10m, cross_20m_to_10m], dim=1))
            fused_10m = fused_10m + gate_10m * cross_20m_to_10m
            
            gate_20m = self.fusion_gate(torch.cat([fused_20m, cross_10m_to_20m], dim=1))
            fused_20m = fused_20m + gate_20m * cross_10m_to_20m
        
        # ========== SPECTRAL CONSISTENCY ==========
        fused_10m = self.spectral_consistency_10m(fused_10m)
        fused_20m = self.spectral_consistency_20m(fused_20m)
        
        # ========== DIRECT UPSAMPLING ==========
        sr_10m = self.upscaler_10m(fused_10m, target_size_10m)
        sr_20m = self.upscaler_20m(fused_20m, target_size_20m)
        
        # ========== FINAL REFINEMENT ==========
        sr_10m = sr_10m + self.final_refine_10m(sr_10m)
        sr_20m = sr_20m + self.final_refine_20m(sr_20m)
        
        # ========== OUTPUT HEADS ==========
        sr_10m = self.output_head_10m(sr_10m)
        sr_20m = self.output_head_20m(sr_20m)
        
        return sr_10m, sr_20m
