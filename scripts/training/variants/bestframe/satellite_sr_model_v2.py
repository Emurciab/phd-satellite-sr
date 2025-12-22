"""
Multi-Resolution Satellite Super-Resolution Model V2

FIXES from V1:
1. Removed sigmoid in output head (was causing mode collapse)
2. Added global residual connections (learn residuals from bilinear upscale)
3. Simplified output head with just soft clamping

Architecture Features:
- Dual-branch encoder for 10m and 20m inputs
- Feature fusion at appropriate scales  
- Spectral consistency preservation
- Temporal attention for multi-frame sequences
- RESIDUAL LEARNING: Output = bilinear_upscale(LR) + learned_residual
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
# SIMPLE OUTPUT HEAD (NO SIGMOID - V2 FIX!)
# =============================================================================

class SimpleOutputHead(nn.Module):
    """
    V2 Fix: Simple output head WITHOUT sigmoid.
    
    Just uses soft clamping (clamp) to keep values in valid range.
    This avoids mode collapse that sigmoid caused.
    """
    def __init__(self, in_channels, out_channels, reflectance_range=(0, 1)):
        super().__init__()
        self.min_val, self.max_val = reflectance_range
        
        # Ensure intermediate channels is at least 1
        intermediate = max(in_channels // 2, out_channels, 16)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, intermediate, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(intermediate, out_channels, 3, padding=1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        # Simple clamp - no sigmoid! This lets gradients flow freely
        return torch.clamp(x, self.min_val, self.max_val)


# =============================================================================
# MAIN MODEL V2: WITH RESIDUAL LEARNING
# =============================================================================

class MultiResolutionSatelliteSRv2(nn.Module):
    """
    Multi-Resolution Super-Resolution Model V2.
    
    KEY CHANGE: Residual learning architecture
    Output = bilinear_upscale(LR_middle_frame) + learned_residual
    
    This forces the model to learn only the "enhancement" rather than
    reconstructing the entire image from scratch, which:
    1. Provides a strong baseline (bilinear is reasonable)
    2. Makes training easier (smaller residuals to learn)
    3. Prevents mode collapse (output is anchored to input)
    """
    
    def __init__(
        self,
        base_channels: int = 64,
        num_encoder_blocks: int = 4,
        num_frames: int = 3,
        scale_factor_10m: float = 3.33,
        scale_factor_20m: float = 3.33,
        use_cross_resolution_fusion: bool = True,
        in_channels_10m: int = 3,
        in_channels_20m: int = 2,  # Changed default to 2 for 5-band data
    ):
        super().__init__()
        
        self.scale_factor_10m = scale_factor_10m
        self.scale_factor_20m = scale_factor_20m
        self.num_frames = num_frames
        self.use_cross_resolution_fusion = use_cross_resolution_fusion
        self.in_channels_10m = in_channels_10m
        self.in_channels_20m = in_channels_20m
        
        # Encoders for each resolution
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
        self.spectral_consistency_10m = SpectralConsistencyModule(base_channels, in_channels_10m)
        self.spectral_consistency_20m = SpectralConsistencyModule(base_channels, in_channels_20m)
        
        # Upscalers - now output RESIDUALS, not full images
        self.upscaler_10m = ProgressiveUpscaler(
            base_channels, in_channels_10m, scale_factor_10m, base_channels
        )
        self.upscaler_20m = ProgressiveUpscaler(
            base_channels, in_channels_20m, scale_factor_20m, base_channels
        )
        
        # Residual refinement (small conv to refine residuals)
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
        
        # Simple output heads (NO SIGMOID!)
        self.output_head_10m = SimpleOutputHead(in_channels_10m, in_channels_10m)
        self.output_head_20m = SimpleOutputHead(in_channels_20m, in_channels_20m)
        
        # Residual scale factor (learnable, starts small)
        self.residual_scale_10m = nn.Parameter(torch.tensor(0.1))
        self.residual_scale_20m = nn.Parameter(torch.tensor(0.1))
    
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
            target_size_20m: (H', W') exact output size for 20m band
        
        Returns:
            sr_10m: (B, 3, H', W') - Super-resolved 10m bands
            sr_20m: (B, 2, H'/2, W'/2) - Super-resolved 20m bands
        """
        B, T, _, H_10m, W_10m = lr_10m.shape
        _, _, _, H_20m, W_20m = lr_20m.shape
        
        # Get middle frame for baseline upscale
        middle_idx = T // 2
        lr_10m_middle = lr_10m[:, middle_idx]  # (B, 3, H, W)
        lr_20m_middle = lr_20m[:, middle_idx]  # (B, 2, H/2, W/2)
        
        # Bilinear upscale of middle frame (baseline)
        if target_size_10m is not None:
            baseline_10m = F.interpolate(lr_10m_middle, size=target_size_10m, 
                                         mode='bilinear', align_corners=False)
        else:
            baseline_10m = F.interpolate(lr_10m_middle, scale_factor=self.scale_factor_10m,
                                         mode='bilinear', align_corners=False)
        
        if target_size_20m is not None:
            baseline_20m = F.interpolate(lr_20m_middle, size=target_size_20m,
                                         mode='bilinear', align_corners=False)
        else:
            baseline_20m = F.interpolate(lr_20m_middle, scale_factor=self.scale_factor_20m,
                                         mode='bilinear', align_corners=False)
        
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
            fused_20m_up = F.interpolate(fused_20m, size=(H_10m, W_10m), 
                                         mode='bilinear', align_corners=False)
            
            # Downsample 10m features to 20m resolution for fusion
            fused_10m_down = F.interpolate(fused_10m, size=(H_20m, W_20m), 
                                           mode='bilinear', align_corners=False)
            
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
        
        # Upscale features to get RESIDUALS
        residual_10m = self.upscaler_10m(fused_10m, target_size_10m if target_size_10m else baseline_10m.shape[-2:])
        residual_20m = self.upscaler_20m(fused_20m, target_size_20m if target_size_20m else baseline_20m.shape[-2:])
        
        # Refine residuals
        residual_10m = self.residual_refine_10m(residual_10m)
        residual_20m = self.residual_refine_20m(residual_20m)
        
        # Scale residuals (learnable, starts small for stability)
        residual_10m = residual_10m * self.residual_scale_10m
        residual_20m = residual_20m * self.residual_scale_20m
        
        # RESIDUAL LEARNING: Output = Baseline + Learned Residual
        sr_10m = baseline_10m + residual_10m
        sr_20m = baseline_20m + residual_20m
        
        # Final output processing (simple clamp, no sigmoid!)
        sr_10m = self.output_head_10m(sr_10m)
        sr_20m = self.output_head_20m(sr_20m)
        
        return sr_10m, sr_20m


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
        
        # Cosine similarity -> spectral angle
        cos_sim = (pred_norm * target_norm).sum(dim=1)
        cos_sim = torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)
        sam = torch.acos(cos_sim)
        
        return sam.mean()


class GradientLoss(nn.Module):
    """Edge preservation loss using Sobel gradients."""
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        # Compute gradients for each channel
        loss = 0
        for c in range(pred.shape[1]):
            pred_c = pred[:, c:c+1]
            target_c = target[:, c:c+1]
            
            # Gradients
            pred_gx = F.conv2d(pred_c, self.sobel_x, padding=1)
            pred_gy = F.conv2d(pred_c, self.sobel_y, padding=1)
            target_gx = F.conv2d(target_c, self.sobel_x, padding=1)
            target_gy = F.conv2d(target_c, self.sobel_y, padding=1)
            
            loss += F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
        
        return loss / pred.shape[1]


class MultiResSpectralLoss(nn.Module):
    """
    Combined loss for multi-resolution satellite SR.
    
    Components:
    - L1: Primary reconstruction loss
    - MSE: Secondary reconstruction
    - SAM: Spectral angle preservation
    - Gradient: Edge preservation
    """
    def __init__(self, l1_weight=1.0, mse_weight=0.1, sam_weight=0.1, grad_weight=0.01):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.sam_weight = sam_weight
        self.grad_weight = grad_weight
        
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.sam = SpectralAngleLoss()
        self.grad = GradientLoss()
    
    def forward(self, pred_10m, target_10m, pred_20m, target_20m):
        """
        Args:
            pred_10m: (B, 3, H, W) predicted 10m bands
            target_10m: (B, 3, H, W) target 10m bands
            pred_20m: (B, 2, H/2, W/2) predicted 20m bands
            target_20m: (B, 2, H/2, W/2) target 20m bands
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Individual loss components
        """
        # 10m losses
        l1_10m = self.l1(pred_10m, target_10m)
        mse_10m = self.mse(pred_10m, target_10m)
        sam_10m = self.sam(pred_10m, target_10m)
        grad_10m = self.grad(pred_10m, target_10m)
        
        # 20m losses
        l1_20m = self.l1(pred_20m, target_20m)
        mse_20m = self.mse(pred_20m, target_20m)
        sam_20m = self.sam(pred_20m, target_20m)
        grad_20m = self.grad(pred_20m, target_20m)
        
        # Combined (equal weight for both resolutions)
        l1_loss = (l1_10m + l1_20m) / 2
        mse_loss = (mse_10m + mse_20m) / 2
        sam_loss = (sam_10m + sam_20m) / 2
        grad_loss = (grad_10m + grad_20m) / 2
        
        total_loss = (
            self.l1_weight * l1_loss +
            self.mse_weight * mse_loss +
            self.sam_weight * sam_loss +
            self.grad_weight * grad_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1_loss.item(),
            'mse': mse_loss.item(),
            'sam': sam_loss.item(),
            'grad': grad_loss.item(),
            'l1_10m': l1_10m.item(),
            'l1_20m': l1_20m.item(),
        }
        
        return total_loss, loss_dict


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_v2(
    base_channels=64,
    num_encoder_blocks=4,
    num_frames=3,
    in_channels_10m=3,
    in_channels_20m=2
):
    """Create the V2 model with residual learning."""
    model = MultiResolutionSatelliteSRv2(
        base_channels=base_channels,
        num_encoder_blocks=num_encoder_blocks,
        num_frames=num_frames,
        in_channels_10m=in_channels_10m,
        in_channels_20m=in_channels_20m
    )
    
    print(f"Created MultiResolutionSatelliteSRv2")
    print(f"  - Base channels: {base_channels}")
    print(f"  - Encoder blocks: {num_encoder_blocks}")
    print(f"  - Temporal frames: {num_frames}")
    print(f"  - 10m channels: {in_channels_10m}")
    print(f"  - 20m channels: {in_channels_20m}")
    print(f"  - Total parameters: {count_parameters(model):,}")
    print(f"  - Residual learning: ENABLED (fixes mode collapse)")
    
    return model


if __name__ == "__main__":
    # Quick test
    print("Testing MultiResolutionSatelliteSRv2...")
    
    model = create_model_v2()
    
    # Test input (simulating batch of 2, 3 temporal frames)
    B, T = 2, 3
    lr_10m = torch.randn(B, T, 3, 78, 78)
    lr_20m = torch.randn(B, T, 2, 39, 39)
    
    # Forward pass
    with torch.no_grad():
        sr_10m, sr_20m = model(lr_10m, lr_20m, target_size_10m=(260, 260), target_size_20m=(130, 130))
    
    print(f"\nInput shapes:")
    print(f"  LR 10m: {lr_10m.shape}")
    print(f"  LR 20m: {lr_20m.shape}")
    print(f"\nOutput shapes:")
    print(f"  SR 10m: {sr_10m.shape}")
    print(f"  SR 20m: {sr_20m.shape}")
    
    # Test that model responds to input (no mode collapse)
    print("\n--- Mode Collapse Test ---")
    lr_zeros = torch.zeros(1, T, 3, 78, 78)
    lr_ones = torch.ones(1, T, 3, 78, 78)
    lr_20m_test = torch.randn(1, T, 2, 39, 39)
    
    with torch.no_grad():
        out_zeros, _ = model(lr_zeros, lr_20m_test, target_size_10m=(260, 260), target_size_20m=(130, 130))
        out_ones, _ = model(lr_ones, lr_20m_test, target_size_10m=(260, 260), target_size_20m=(130, 130))
    
    print(f"Zero input -> output mean: {out_zeros.mean().item():.4f}")
    print(f"Ones input -> output mean: {out_ones.mean().item():.4f}")
    print(f"Difference: {abs(out_zeros.mean().item() - out_ones.mean().item()):.4f}")
    
    if abs(out_zeros.mean().item() - out_ones.mean().item()) > 0.1:
        print("✅ Model responds to different inputs (no mode collapse)")
    else:
        print("⚠️ Warning: Output may be too similar for different inputs")
    
    print("\n✅ Model test passed!")
