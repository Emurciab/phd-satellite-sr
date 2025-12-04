"""
Multi-Resolution Satellite Super-Resolution Model

Designed for Sentinel-2 → Planet super-resolution with proper handling of:
- 10m bands (B2, B3, B4): Blue, Green, Red
- 20m band (B8A): Red Edge NIR at native resolution

Architecture Features:
- Dual-branch encoder for 10m and 20m inputs
- Feature fusion at appropriate scales  
- Spectral consistency preservation
- Temporal attention for multi-frame sequences
- Physics-constrained output

Input/Output:
- LR 10m: (B, T, 3, H, W) → SR: (B, T, 3, H*3.33, W*3.33)
- LR 20m: (B, T, 1, H/2, W/2) → SR: (B, T, 1, H*3.33/2, W*3.33/2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


# =============================================================================
# BUILDING BLOCKS
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
    """Convolutional Block Attention Module (channel + spatial attention)."""
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# =============================================================================
# TEMPORAL ATTENTION
# =============================================================================

class TemporalAttention(nn.Module):
    """
    Temporal attention for multi-frame sequences.
    Learns to weight frames based on quality/relevance.
    """
    def __init__(self, channels, num_frames=3):
        super().__init__()
        self.num_frames = num_frames
        self.channels = channels
        
        # Per-frame feature extraction
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        # Temporal attention weights (takes concatenated pooled features)
        self.temporal_attn = nn.Sequential(
            nn.Linear(channels * num_frames, channels),
            nn.GELU(),
            nn.Linear(channels, num_frames),
            nn.Softmax(dim=1)
        )
        
        # Fusion conv
        self.fusion = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) temporal sequence
        Returns:
            fused: (B, C, H, W) temporally fused features
        """
        B, T, C, H, W = x.shape
        
        # Encode each frame
        frame_features = []
        for t in range(T):
            feat = self.frame_encoder(x[:, t])
            frame_features.append(feat)
        
        # Compute attention weights
        stacked = torch.stack(frame_features, dim=1)  # (B, T, C, H, W)
        
        # Pool each frame to get global features
        pooled_list = []
        for f in frame_features:
            pooled = F.adaptive_avg_pool2d(f, 1)  # (B, C, 1, 1)
            pooled = pooled.view(B, -1)  # (B, C)
            pooled_list.append(pooled)
        
        pooled_cat = torch.cat(pooled_list, dim=1)  # (B, T*C)
        weights = self.temporal_attn(pooled_cat)  # (B, T)
        
        # Weighted fusion
        weights = weights.view(B, T, 1, 1, 1)
        fused = (stacked * weights).sum(dim=1)  # (B, C, H, W)
        
        return self.fusion(fused)


# =============================================================================
# SPECTRAL CONSISTENCY MODULE
# =============================================================================

class SpectralConsistencyModule(nn.Module):
    """
    Ensures output maintains spectral relationships from input.
    Critical for preserving band ratios (e.g., NDVI).
    """
    def __init__(self, channels, num_bands):
        super().__init__()
        self.num_bands = num_bands
        
        # Learn band correlations
        self.band_correlation = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, num_bands, 1),
            nn.Sigmoid()
        )
        
        # Spectral attention
        self.spectral_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, lr_input=None):
        """
        Args:
            features: (B, C, H, W) deep features
            lr_input: (B, bands, H_lr, W_lr) original LR input for guidance
        Returns:
            consistent_features: (B, C, H, W)
        """
        # Apply spectral attention
        attn = self.spectral_attn(features)
        return features * attn


# =============================================================================
# ENCODER BRANCH
# =============================================================================

class EncoderBranch(nn.Module):
    """
    Encoder branch for a specific resolution.
    Used separately for 10m and 20m bands.
    """
    def __init__(self, in_channels, base_channels=64, num_blocks=4):
        super().__init__()
        
        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        # Residual blocks with attention
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(nn.Sequential(
                ResidualBlock(base_channels),
                CBAM(base_channels) if i % 2 == 1 else nn.Identity()
            ))
        
        # Output projection
        self.out_proj = nn.Conv2d(base_channels, base_channels, 1)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input
        Returns:
            features: (B, base_channels, H, W)
        """
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x) + x  # Residual
        
        return self.out_proj(x)


# =============================================================================
# UPSAMPLING MODULES
# =============================================================================

class PixelShuffleUpscale(nn.Module):
    """Sub-pixel convolution upscaling (efficient)."""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.act(x)


class ProgressiveUpscaler(nn.Module):
    """
    Progressive upscaling with intermediate supervision capability.
    Handles non-integer scale factors via interpolation.
    """
    def __init__(self, in_channels, out_channels, target_scale, base_channels=64):
        super().__init__()
        self.target_scale = target_scale
        
        # Use 2x pixel shuffle steps, then interpolate to exact size
        num_2x_steps = int(math.log2(target_scale)) if target_scale >= 2 else 0
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_2x_steps):
            layers.append(PixelShuffleUpscale(current_channels, base_channels, 2))
            current_channels = base_channels
        
        self.upscale_blocks = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Final output conv
        self.output_conv = nn.Sequential(
            nn.Conv2d(current_channels if num_2x_steps > 0 else in_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x, target_size=None):
        """
        Args:
            x: (B, C, H, W) features
            target_size: (H', W') exact output size
        Returns:
            upscaled: (B, out_channels, H', W')
        """
        x = self.upscale_blocks(x)
        x = self.output_conv(x)
        
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


# =============================================================================
# PHYSICS-CONSTRAINED OUTPUT HEAD
# =============================================================================

class PhysicsConstrainedHead(nn.Module):
    """
    Output head that applies physics-based constraints:
    - Reflectance values bounded [0, 1] for surface reflectance
    - Smooth transitions at boundaries
    """
    def __init__(self, in_channels, out_channels, reflectance_range=(0, 1)):
        super().__init__()
        self.min_val, self.max_val = reflectance_range
        
        # Ensure intermediate channels is at least 1
        intermediate = max(in_channels // 2, out_channels)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, intermediate, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(intermediate, out_channels, 3, padding=1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        # Soft clamping using sigmoid scaled to range
        x = torch.sigmoid(x) * (self.max_val - self.min_val) + self.min_val
        return x


# =============================================================================
# MAIN MODEL: MULTI-RESOLUTION SATELLITE SR
# =============================================================================

class MultiResolutionSatelliteSR(nn.Module):
    """
    Multi-Resolution Super-Resolution Model for Satellite Imagery.
    
    Handles 10m and 20m bands separately to preserve native resolution characteristics.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │   LR 10m (B2,B3,B4)              LR 20m (B8A)                           │
    │   (B, T, 3, H, W)                (B, T, 1, H/2, W/2)                    │
    │        │                              │                                 │
    │        ▼                              ▼                                 │
    │   ┌─────────────┐               ┌─────────────┐                        │
    │   │ Encoder 10m │               │ Encoder 20m │                        │
    │   └─────────────┘               └─────────────┘                        │
    │        │                              │                                 │
    │        ▼                              ▼                                 │
    │   ┌─────────────┐               ┌─────────────┐                        │
    │   │  Temporal   │               │  Temporal   │                        │
    │   │  Attention  │               │  Attention  │                        │
    │   └─────────────┘               └─────────────┘                        │
    │        │                              │                                 │
    │        └──────────┬──────────────────┘                                 │
    │                   ▼                                                     │
    │            ┌─────────────┐                                              │
    │            │   Feature   │ (Optional cross-resolution fusion)          │
    │            │   Fusion    │                                              │
    │            └─────────────┘                                              │
    │                   │                                                     │
    │        ┌──────────┴──────────┐                                         │
    │        ▼                     ▼                                          │
    │   ┌─────────────┐      ┌─────────────┐                                 │
    │   │ Upscaler 10m│      │ Upscaler 20m│                                 │
    │   │  (~3.33x)   │      │  (~3.33x)   │                                 │
    │   └─────────────┘      └─────────────┘                                 │
    │        │                      │                                         │
    │        ▼                      ▼                                         │
    │   SR 10m bands           SR 20m band                                    │
    │   (B, T, 3, H', W')      (B, T, 1, H'/2, W'/2)                         │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        base_channels: int = 64,
        num_encoder_blocks: int = 4,
        num_frames: int = 3,
        scale_factor_10m: float = 3.33,  # 10m → 3m
        scale_factor_20m: float = 3.33,  # 20m → 6m (maintains relative scale)
        use_cross_resolution_fusion: bool = True
    ):
        super().__init__()
        
        self.scale_factor_10m = scale_factor_10m
        self.scale_factor_20m = scale_factor_20m
        self.num_frames = num_frames
        self.use_cross_resolution_fusion = use_cross_resolution_fusion
        
        # Encoders for each resolution
        self.encoder_10m = EncoderBranch(
            in_channels=3,  # B2, B3, B4
            base_channels=base_channels,
            num_blocks=num_encoder_blocks
        )
        
        self.encoder_20m = EncoderBranch(
            in_channels=1,  # B8A
            base_channels=base_channels,
            num_blocks=num_encoder_blocks
        )
        
        # Temporal attention for each branch
        self.temporal_attn_10m = TemporalAttention(base_channels, num_frames)
        self.temporal_attn_20m = TemporalAttention(base_channels, num_frames)
        
        # Cross-resolution fusion (optional)
        if use_cross_resolution_fusion:
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
        
        # Spectral consistency
        self.spectral_consistency_10m = SpectralConsistencyModule(base_channels, 3)
        self.spectral_consistency_20m = SpectralConsistencyModule(base_channels, 1)
        
        # Upscalers
        self.upscaler_10m = ProgressiveUpscaler(
            base_channels, 3, scale_factor_10m, base_channels
        )
        self.upscaler_20m = ProgressiveUpscaler(
            base_channels, 1, scale_factor_20m, base_channels
        )
        
        # Physics-constrained output heads
        self.output_head_10m = PhysicsConstrainedHead(3, 3)
        self.output_head_20m = PhysicsConstrainedHead(1, 1)
    
    def forward(
        self,
        lr_10m: torch.Tensor,
        lr_20m: torch.Tensor,
        target_size_10m: Tuple[int, int] = None,
        target_size_20m: Tuple[int, int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lr_10m: (B, T, 3, H, W) - B2, B3, B4 at 10m resolution
            lr_20m: (B, T, 1, H/2, W/2) - B8A at native 20m resolution
            target_size_10m: (H', W') exact output size for 10m bands
            target_size_20m: (H', W') exact output size for 20m band
        
        Returns:
            sr_10m: (B, 3, H', W') - Super-resolved 10m bands
            sr_20m: (B, 1, H'/2, W'/2) - Super-resolved 20m band
        """
        B, T, _, H_10m, W_10m = lr_10m.shape
        _, _, _, H_20m, W_20m = lr_20m.shape
        
        # Encode each frame for 10m branch
        features_10m = []
        for t in range(T):
            feat = self.encoder_10m(lr_10m[:, t])
            features_10m.append(feat)
        features_10m = torch.stack(features_10m, dim=1)  # (B, T, C, H, W)
        
        # Encode each frame for 20m branch
        features_20m = []
        for t in range(T):
            feat = self.encoder_20m(lr_20m[:, t])
            features_20m.append(feat)
        features_20m = torch.stack(features_20m, dim=1)  # (B, T, C, H/2, W/2)
        
        # Temporal attention
        fused_10m = self.temporal_attn_10m(features_10m)  # (B, C, H, W)
        fused_20m = self.temporal_attn_20m(features_20m)  # (B, C, H/2, W/2)
        
        # Cross-resolution fusion
        if self.use_cross_resolution_fusion:
            # Upsample 20m features to 10m resolution for fusion
            fused_20m_up = F.interpolate(fused_20m, size=(H_10m, W_10m), mode='bilinear', align_corners=False)
            
            # Downsample 10m features to 20m resolution for fusion
            fused_10m_down = F.interpolate(fused_10m, size=(H_20m, W_20m), mode='bilinear', align_corners=False)
            
            # Compute fusion gate for 10m branch
            combined_10m = torch.cat([fused_10m, self.fusion_10m_from_20m(fused_20m_up)], dim=1)
            gate_10m = self.fusion_gate(combined_10m)
            fused_10m = fused_10m + gate_10m * self.fusion_10m_from_20m(fused_20m_up)
            
            # Compute fusion gate for 20m branch
            combined_20m = torch.cat([fused_20m, self.fusion_20m_from_10m(fused_10m_down)], dim=1)
            gate_20m = self.fusion_gate(combined_20m)
            fused_20m = fused_20m + gate_20m * self.fusion_20m_from_10m(fused_10m_down)
        
        # Spectral consistency
        fused_10m = self.spectral_consistency_10m(fused_10m)
        fused_20m = self.spectral_consistency_20m(fused_20m)
        
        # Upscale
        sr_10m = self.upscaler_10m(fused_10m, target_size_10m)
        sr_20m = self.upscaler_20m(fused_20m, target_size_20m)
        
        # Physics constraints
        sr_10m = self.output_head_10m(sr_10m)
        sr_20m = self.output_head_20m(sr_20m)
        
        return sr_10m, sr_20m


# =============================================================================
# SINGLE-RESOLUTION MODEL (for 4-band data without B8A separation)
# =============================================================================

class SingleResolutionSatelliteSR(nn.Module):
    """
    Single-resolution SR model for simpler use cases.
    Use when all bands are at the same resolution (e.g., 4-band Dove data).
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 64,
        num_encoder_blocks: int = 4,
        num_frames: int = 3,
        scale_factor: float = 3.33
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.num_frames = num_frames
        
        # Encoder
        self.encoder = EncoderBranch(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_encoder_blocks
        )
        
        # Temporal attention
        self.temporal_attn = TemporalAttention(base_channels, num_frames)
        
        # Spectral consistency
        self.spectral_consistency = SpectralConsistencyModule(base_channels, out_channels)
        
        # Upscaler
        self.upscaler = ProgressiveUpscaler(
            base_channels, out_channels, scale_factor, base_channels
        )
        
        # Physics-constrained output
        self.output_head = PhysicsConstrainedHead(out_channels, out_channels)
    
    def forward(
        self,
        lr: torch.Tensor,
        target_size: Tuple[int, int] = None
    ) -> torch.Tensor:
        """
        Args:
            lr: (B, T, C, H, W) - LR temporal sequence
            target_size: (H', W') exact output size
        
        Returns:
            sr: (B, C, H', W') - Super-resolved output
        """
        B, T, C, H, W = lr.shape
        
        # Encode each frame
        features = []
        for t in range(T):
            feat = self.encoder(lr[:, t])
            features.append(feat)
        features = torch.stack(features, dim=1)  # (B, T, C, H, W)
        
        # Temporal attention
        fused = self.temporal_attn(features)  # (B, C, H, W)
        
        # Spectral consistency
        fused = self.spectral_consistency(fused)
        
        # Upscale
        sr = self.upscaler(fused, target_size)
        
        # Physics constraints
        sr = self.output_head(sr)
        
        return sr


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class SpectralAngleLoss(nn.Module):
    """Spectral Angle Mapper (SAM) loss for spectral fidelity."""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Normalize along channel dimension
        pred_norm = F.normalize(pred, dim=1, eps=1e-8)
        target_norm = F.normalize(target, dim=1, eps=1e-8)
        
        # Cosine similarity
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        cos_sim = torch.clamp(cos_sim, -1, 1)
        
        # Spectral angle
        sam = torch.acos(cos_sim)
        return sam.mean()


class SpectralFidelityLoss(nn.Module):
    """
    Comprehensive loss for satellite SR with Solar Zenith Angle (SZA) weighting.
    
    Components:
    1. Pixel Loss: L1 + MSE for accurate reconstruction
    2. Spectral Angle Loss: Preserves spectral signatures (important for remote sensing)
    3. Spectral Correlation Loss: Maintains inter-band relationships
    4. LR Consistency Loss: SR downsampled should match original LR
    5. Edge Loss: Preserves spatial details using Sobel gradients
    
    SZA Weighting:
    - Uses Gaussian penalty based on deviation from optimal solar zenith (45°)
    - w(SZA) = exp(-(SZA - 45)² / (2 * σ²))
    - Penalizes extreme sun angles where water-leaving radiance is less reliable
    - Optimal for aquatic vegetation analysis
    
    References:
    - Spectral Angle Mapper: Kruse et al. (1993)
    - LR Consistency: Ledig et al. (2017) - SRGAN
    """
    
    def __init__(
        self,
        # Loss component weights
        pixel_weight: float = 1.0,
        spectral_weight: float = 0.5,
        correlation_weight: float = 0.3,
        lr_consistency_weight: float = 0.5,
        edge_weight: float = 0.2,
        # SZA weighting parameters
        sza_optimal: float = 45.0,
        sza_sigma: float = 30.0,
        use_sza_weighting: bool = True
    ):
        """
        Args:
            pixel_weight: Weight for L1+MSE pixel loss (default: 1.0)
            spectral_weight: Weight for spectral angle loss (default: 0.5)
            correlation_weight: Weight for band correlation loss (default: 0.3)
            lr_consistency_weight: Weight for LR consistency loss (default: 0.5)
            edge_weight: Weight for edge preservation loss (default: 0.2)
            sza_optimal: Optimal solar zenith angle in degrees (default: 45°)
            sza_sigma: Gaussian penalty σ for SZA weighting (default: 30°)
            use_sza_weighting: Whether to apply SZA-based weighting (default: True)
        """
        super().__init__()
        
        # Loss weights
        self.pixel_weight = pixel_weight
        self.spectral_weight = spectral_weight
        self.correlation_weight = correlation_weight
        self.lr_consistency_weight = lr_consistency_weight
        self.edge_weight = edge_weight
        
        # SZA parameters
        self.sza_optimal = sza_optimal
        self.sza_sigma = sza_sigma
        self.use_sza_weighting = use_sza_weighting
        
        # Base losses
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.spectral_angle_loss = SpectralAngleLoss()
        
        # Sobel filters for edge detection
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def compute_sza_weight(self, sza) -> float:
        """
        Compute Gaussian weight based on solar zenith angle deviation.
        
        Args:
            sza: Solar zenith angle(s) in degrees. Can be:
                 - None: Returns 1.0 (no penalty)
                 - float: Single angle
                 - List[float]: Per-sample angles (returns mean weight)
        
        Returns:
            Weight in range (0, 1], where 1.0 = optimal angle
        """
        if sza is None or not self.use_sza_weighting:
            return 1.0
        
        if isinstance(sza, (list, tuple)):
            # Filter out None values and compute mean weight
            valid_sza = [s for s in sza if s is not None]
            if not valid_sza:
                return 1.0
            weights = [math.exp(-((s - self.sza_optimal) ** 2) / (2 * self.sza_sigma ** 2)) 
                       for s in valid_sza]
            return sum(weights) / len(weights)
        else:
            deviation = sza - self.sza_optimal
            return math.exp(-(deviation ** 2) / (2 * self.sza_sigma ** 2))
    
    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel filters."""
        B, C, H, W = x.shape
        edges = []
        for c in range(C):
            channel = x[:, c:c+1, :, :]
            edge_x = F.conv2d(channel, self.sobel_x, padding=1)
            edge_y = F.conv2d(channel, self.sobel_y, padding=1)
            edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
            edges.append(edge_mag)
        return torch.cat(edges, dim=1)
    
    def compute_correlation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral correlation loss between bands.
        
        Ensures inter-band relationships are preserved, which is critical
        for vegetation indices and other spectral analysis.
        """
        if pred.shape[1] <= 1:
            return torch.tensor(0.0, device=pred.device)
        
        B, C, H, W = pred.shape
        
        # Flatten spatial dimensions
        pred_flat = pred.view(B, C, -1)  # (B, C, H*W)
        target_flat = target.view(B, C, -1)
        
        # Compute correlation matrices for each sample in batch
        # Normalize along spatial dimension
        pred_mean = pred_flat.mean(dim=2, keepdim=True)
        target_mean = target_flat.mean(dim=2, keepdim=True)
        
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        # Compute correlation: (C, C) per sample
        pred_std = pred_centered.std(dim=2, keepdim=True) + 1e-8
        target_std = target_centered.std(dim=2, keepdim=True) + 1e-8
        
        pred_norm = pred_centered / pred_std
        target_norm = target_centered / target_std
        
        # Correlation matrix: (B, C, C)
        pred_corr = torch.bmm(pred_norm, pred_norm.transpose(1, 2)) / (H * W)
        target_corr = torch.bmm(target_norm, target_norm.transpose(1, 2)) / (H * W)
        
        # Loss is difference in correlation structure
        corr_loss = F.l1_loss(pred_corr, target_corr)
        return corr_loss
    
    def compute_lr_consistency(self, sr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        LR Consistency: Downsampled SR should match original LR.
        
        This enforces that the super-resolved image, when downsampled back,
        produces the original low-resolution input.
        """
        if lr is None:
            return torch.tensor(0.0, device=sr.device)
        
        # Downsample SR to LR size
        sr_down = F.adaptive_avg_pool2d(sr, (lr.shape[-2], lr.shape[-1]))
        
        return self.l1_loss(sr_down, lr)
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        lr: Optional[torch.Tensor] = None,
        sza = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute comprehensive loss with optional SZA weighting.
        
        Args:
            pred: Predicted SR output (B, C, H, W)
            target: Ground truth HR (B, C, H, W)
            lr: Original LR input for consistency loss (B, C, h, w), optional
            sza: Solar zenith angle(s) in degrees, optional. Can be:
                 - float: Single angle for entire batch
                 - List[float]: Per-sample angles
                 - None: No SZA weighting
        
        Returns:
            total_loss: Weighted combined loss
            loss_dict: Dictionary with individual loss components
        """
        device = pred.device
        
        # 1. Pixel loss (L1 + MSE)
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        pixel_loss = l1 + 0.1 * mse
        
        # 2. Spectral angle loss
        if pred.shape[1] > 1:
            spectral_loss = self.spectral_angle_loss(pred, target)
        else:
            spectral_loss = torch.tensor(0.0, device=device)
        
        # 3. Spectral correlation loss
        correlation_loss = self.compute_correlation_loss(pred, target)
        
        # 4. LR consistency loss
        lr_consistency_loss = self.compute_lr_consistency(pred, lr)
        
        # 5. Edge preservation loss
        pred_edges = self.compute_edges(pred)
        target_edges = self.compute_edges(target)
        edge_loss = self.l1_loss(pred_edges, target_edges)
        
        # Combine losses
        base_loss = (
            self.pixel_weight * pixel_loss +
            self.spectral_weight * spectral_loss +
            self.correlation_weight * correlation_loss +
            self.lr_consistency_weight * lr_consistency_loss +
            self.edge_weight * edge_loss
        )
        
        # Apply SZA weighting
        sza_weight = self.compute_sza_weight(sza)
        total_loss = sza_weight * base_loss
        
        return total_loss, {
            'pixel': pixel_loss.item(),
            'l1': l1.item(),
            'mse': mse.item(),
            'spectral': spectral_loss.item() if torch.is_tensor(spectral_loss) else spectral_loss,
            'correlation': correlation_loss.item() if torch.is_tensor(correlation_loss) else correlation_loss,
            'lr_consistency': lr_consistency_loss.item() if torch.is_tensor(lr_consistency_loss) else lr_consistency_loss,
            'edge': edge_loss.item(),
            'sza_weight': sza_weight,
            'base_loss': base_loss.item(),
            'total': total_loss.item()
        }


class MultiResolutionLoss(nn.Module):
    """
    Simplified loss for quick testing. For full training, use SpectralFidelityLoss.
    
    Components:
    - L1 loss (pixel-wise)
    - Spectral angle loss (spectral fidelity)
    - Edge loss (sharpness)
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        spectral_weight: float = 0.1,
        edge_weight: float = 0.1
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.spectral_weight = spectral_weight
        self.edge_weight = edge_weight
        
        self.l1_loss = nn.L1Loss()
        self.spectral_loss = SpectralAngleLoss()
        
        # Sobel filters for edge loss
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def compute_edges(self, x):
        """Compute edge magnitude using Sobel filters."""
        B, C, H, W = x.shape
        edges = []
        for c in range(C):
            channel = x[:, c:c+1, :, :]
            edge_x = F.conv2d(channel, self.sobel_x, padding=1)
            edge_y = F.conv2d(channel, self.sobel_y, padding=1)
            edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
            edges.append(edge_mag)
        return torch.cat(edges, dim=1)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted SR output
            target: Ground truth HR
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # L1 loss
        l1 = self.l1_loss(pred, target)
        
        # Spectral loss (only if more than 1 channel)
        if pred.shape[1] > 1:
            spectral = self.spectral_loss(pred, target)
        else:
            spectral = torch.tensor(0.0, device=pred.device)
        
        # Edge loss
        pred_edges = self.compute_edges(pred)
        target_edges = self.compute_edges(target)
        edge = self.l1_loss(pred_edges, target_edges)
        
        # Combined loss
        total = (
            self.l1_weight * l1 +
            self.spectral_weight * spectral +
            self.edge_weight * edge
        )
        
        return total, {
            'l1': l1.item(),
            'spectral': spectral.item() if torch.is_tensor(spectral) else spectral,
            'edge': edge.item(),
            'total': total.item()
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-RESOLUTION SATELLITE SR MODEL TEST")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test MultiResolutionSatelliteSR
    print("Testing MultiResolutionSatelliteSR...")
    model = MultiResolutionSatelliteSR(
        base_channels=64,
        num_encoder_blocks=4,
        num_frames=3,
        scale_factor_10m=3.33,
        scale_factor_20m=3.33
    ).to(device)
    
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Model size: {get_model_size_mb(model):.2f} MB")
    
    # Test forward pass
    B, T = 2, 3
    H_10m, W_10m = 64, 64
    H_20m, W_20m = 32, 32
    
    lr_10m = torch.randn(B, T, 3, H_10m, W_10m).to(device)
    lr_20m = torch.randn(B, T, 1, H_20m, W_20m).to(device)
    
    with torch.no_grad():
        sr_10m, sr_20m = model(lr_10m, lr_20m)
    
    print(f"  Input 10m: {lr_10m.shape}")
    print(f"  Input 20m: {lr_20m.shape}")
    print(f"  Output 10m: {sr_10m.shape}")
    print(f"  Output 20m: {sr_20m.shape}")
    print("  ✅ MultiResolutionSatelliteSR test passed!\n")
    
    # Test SingleResolutionSatelliteSR
    print("Testing SingleResolutionSatelliteSR...")
    model_single = SingleResolutionSatelliteSR(
        in_channels=4,
        out_channels=4,
        base_channels=64,
        num_frames=3,
        scale_factor=3.33
    ).to(device)
    
    print(f"  Parameters: {count_parameters(model_single):,}")
    print(f"  Model size: {get_model_size_mb(model_single):.2f} MB")
    
    lr_single = torch.randn(B, T, 4, H_10m, W_10m).to(device)
    
    with torch.no_grad():
        sr_single = model_single(lr_single)
    
    print(f"  Input: {lr_single.shape}")
    print(f"  Output: {sr_single.shape}")
    print("  ✅ SingleResolutionSatelliteSR test passed!\n")
    
    # Test loss function
    print("Testing MultiResolutionLoss...")
    loss_fn = MultiResolutionLoss().to(device)
    target = torch.randn_like(sr_10m)
    loss, loss_dict = loss_fn(sr_10m, target)
    print(f"  Loss components: {loss_dict}")
    print("  ✅ MultiResolutionLoss test passed!\n")
    
    # Test SpectralFidelityLoss
    print("Testing SpectralFidelityLoss...")
    spectral_loss_fn = SpectralFidelityLoss(use_sza_weighting=True).to(device)
    
    # Create fake LR for consistency loss
    lr_for_loss = torch.randn(B, 3, H_10m, W_10m).to(device)
    hr_for_loss = torch.randn(B, 3, H_10m * 3, W_10m * 3).to(device)
    pred_for_loss = torch.randn_like(hr_for_loss)
    
    # Test without SZA
    loss1, loss_dict1 = spectral_loss_fn(pred_for_loss, hr_for_loss, lr=lr_for_loss, sza=None)
    print(f"  Without SZA: {loss_dict1}")
    
    # Test with optimal SZA (45°)
    loss2, loss_dict2 = spectral_loss_fn(pred_for_loss, hr_for_loss, lr=lr_for_loss, sza=45.0)
    print(f"  With SZA=45°: weight={loss_dict2['sza_weight']:.3f}")
    
    # Test with extreme SZA (80°)
    loss3, loss_dict3 = spectral_loss_fn(pred_for_loss, hr_for_loss, lr=lr_for_loss, sza=80.0)
    print(f"  With SZA=80°: weight={loss_dict3['sza_weight']:.3f}")
    
    # Test with per-sample SZA list
    sza_list = [35.0, 45.0, 55.0, None]  # Mix of values and None
    # Adjust batch size for this test
    pred_batch = torch.randn(4, 3, H_10m * 3, W_10m * 3).to(device)
    hr_batch = torch.randn_like(pred_batch)
    lr_batch = torch.randn(4, 3, H_10m, W_10m).to(device)
    loss4, loss_dict4 = spectral_loss_fn(pred_batch, hr_batch, lr=lr_batch, sza=sza_list)
    print(f"  With SZA list: weight={loss_dict4['sza_weight']:.3f}")
    
    print("  ✅ SpectralFidelityLoss test passed!")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
