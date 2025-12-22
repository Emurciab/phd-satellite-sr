"""
Multi-Resolution Satellite Super-Resolution Model V3

Simplified approach: NO residual learning, NO sigmoid.
Just standard direct prediction with clamp.

This is the simplest fix to avoid mode collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


# =============================================================================
# BUILDING BLOCKS (same as V2)
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
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
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels)
        )
        
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ChannelAttention(nn.Module):
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


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
    
    def forward(self, x):
        return self.channel_attn(x)


class TemporalAttention(nn.Module):
    def __init__(self, channels, num_frames=3):
        super().__init__()
        self.num_frames = num_frames
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.temporal_attn = nn.Sequential(
            nn.Linear(channels * num_frames, channels),
            nn.GELU(),
            nn.Linear(channels, num_frames),
            nn.Softmax(dim=1)
        )
        self.fusion = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        frame_features = [self.frame_encoder(x[:, t]) for t in range(T)]
        stacked = torch.stack(frame_features, dim=1)
        
        pooled_list = [F.adaptive_avg_pool2d(f, 1).view(B, -1) for f in frame_features]
        pooled_cat = torch.cat(pooled_list, dim=1)
        weights = self.temporal_attn(pooled_cat).view(B, T, 1, 1, 1)
        
        fused = (stacked * weights).sum(dim=1)
        return self.fusion(fused)


class EncoderBranch(nn.Module):
    def __init__(self, in_channels, base_channels=64, num_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(nn.Sequential(
                ResidualBlock(base_channels),
                CBAM(base_channels) if i % 2 == 1 else nn.Identity()
            ))
        self.out_proj = nn.Conv2d(base_channels, base_channels, 1)
        
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x) + x
        return self.out_proj(x)


class PixelShuffleUpscale(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(self.pixel_shuffle(self.conv(x)))


class ProgressiveUpscaler(nn.Module):
    def __init__(self, in_channels, out_channels, target_scale, base_channels=64):
        super().__init__()
        num_2x_steps = int(math.log2(target_scale)) if target_scale >= 2 else 0
        
        layers = []
        current_channels = in_channels
        for _ in range(num_2x_steps):
            layers.append(PixelShuffleUpscale(current_channels, base_channels, 2))
            current_channels = base_channels
        
        self.upscale_blocks = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_conv = nn.Sequential(
            nn.Conv2d(current_channels if num_2x_steps > 0 else in_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x, target_size=None):
        x = self.upscale_blocks(x)
        x = self.output_conv(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


# =============================================================================
# MAIN MODEL V3: Direct prediction with clamp (no sigmoid, no residual)
# =============================================================================

class MultiResolutionSatelliteSRv3(nn.Module):
    """
    V3: Direct prediction without sigmoid constraint.
    
    Key changes from V1:
    - No sigmoid in output head (was causing mode collapse)
    - Simple clamp to keep values in valid range
    
    Key changes from V2:
    - No residual learning (simpler, let the model learn directly)
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
        in_channels_20m: int = 2,
    ):
        super().__init__()
        
        self.scale_factor_10m = scale_factor_10m
        self.scale_factor_20m = scale_factor_20m
        self.num_frames = num_frames
        self.use_cross_resolution_fusion = use_cross_resolution_fusion
        self.in_channels_10m = in_channels_10m
        self.in_channels_20m = in_channels_20m
        
        # Encoders
        self.encoder_10m = EncoderBranch(in_channels_10m, base_channels, num_encoder_blocks)
        self.encoder_20m = EncoderBranch(in_channels_20m, base_channels, num_encoder_blocks)
        
        # Temporal attention
        self.temporal_attn_10m = TemporalAttention(base_channels, num_frames)
        self.temporal_attn_20m = TemporalAttention(base_channels, num_frames)
        
        # Cross-resolution fusion
        if use_cross_resolution_fusion:
            self.fusion_10m_from_20m = nn.Sequential(nn.Conv2d(base_channels, base_channels, 1), nn.GELU())
            self.fusion_20m_from_10m = nn.Sequential(nn.Conv2d(base_channels, base_channels, 1), nn.GELU())
            self.fusion_gate = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels, 1), nn.Sigmoid())
        
        # Upscalers
        self.upscaler_10m = ProgressiveUpscaler(base_channels, in_channels_10m, scale_factor_10m, base_channels)
        self.upscaler_20m = ProgressiveUpscaler(base_channels, in_channels_20m, scale_factor_20m, base_channels)
        
        # Simple output refinement (no sigmoid!)
        self.output_refine_10m = nn.Sequential(
            nn.Conv2d(in_channels_10m, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels_10m, 3, padding=1)
        )
        self.output_refine_20m = nn.Sequential(
            nn.Conv2d(in_channels_20m, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels_20m, 3, padding=1)
        )
    
    def forward(self, lr_10m, lr_20m, target_size_10m=None, target_size_20m=None):
        B, T, _, H_10m, W_10m = lr_10m.shape
        _, _, _, H_20m, W_20m = lr_20m.shape
        
        # Encode each frame
        features_10m = torch.stack([self.encoder_10m(lr_10m[:, t]) for t in range(T)], dim=1)
        features_20m = torch.stack([self.encoder_20m(lr_20m[:, t]) for t in range(T)], dim=1)
        
        # Temporal attention
        fused_10m = self.temporal_attn_10m(features_10m)
        fused_20m = self.temporal_attn_20m(features_20m)
        
        # Cross-resolution fusion
        if self.use_cross_resolution_fusion:
            fused_20m_up = F.interpolate(fused_20m, size=(H_10m, W_10m), mode='bilinear', align_corners=False)
            fused_10m_down = F.interpolate(fused_10m, size=(H_20m, W_20m), mode='bilinear', align_corners=False)
            
            combined_10m = torch.cat([fused_10m, self.fusion_10m_from_20m(fused_20m_up)], dim=1)
            gate_10m = self.fusion_gate(combined_10m)
            fused_10m = fused_10m + gate_10m * self.fusion_10m_from_20m(fused_20m_up)
            
            combined_20m = torch.cat([fused_20m, self.fusion_20m_from_10m(fused_10m_down)], dim=1)
            gate_20m = self.fusion_gate(combined_20m)
            fused_20m = fused_20m + gate_20m * self.fusion_20m_from_10m(fused_10m_down)
        
        # Upscale
        sr_10m = self.upscaler_10m(fused_10m, target_size_10m)
        sr_20m = self.upscaler_20m(fused_20m, target_size_20m)
        
        # Refine
        sr_10m = sr_10m + self.output_refine_10m(sr_10m)
        sr_20m = sr_20m + self.output_refine_20m(sr_20m)
        
        # Simple clamp - NO SIGMOID!
        sr_10m = torch.clamp(sr_10m, 0, 1)
        sr_20m = torch.clamp(sr_20m, 0, 1)
        
        return sr_10m, sr_20m


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_v3(base_channels=64, num_encoder_blocks=4, num_frames=3, in_channels_10m=3, in_channels_20m=2):
    model = MultiResolutionSatelliteSRv3(
        base_channels=base_channels,
        num_encoder_blocks=num_encoder_blocks,
        num_frames=num_frames,
        in_channels_10m=in_channels_10m,
        in_channels_20m=in_channels_20m
    )
    
    print(f"Created MultiResolutionSatelliteSRv3")
    print(f"  - Base channels: {base_channels}")
    print(f"  - Encoder blocks: {num_encoder_blocks}")
    print(f"  - Temporal frames: {num_frames}")
    print(f"  - 10m channels: {in_channels_10m}")
    print(f"  - 20m channels: {in_channels_20m}")
    print(f"  - Total parameters: {count_parameters(model):,}")
    print(f"  - Direct prediction (no sigmoid, no residual)")
    
    return model


if __name__ == "__main__":
    print("Testing MultiResolutionSatelliteSRv3...")
    
    model = create_model_v3()
    
    B, T = 2, 3
    lr_10m = torch.randn(B, T, 3, 78, 78)
    lr_20m = torch.randn(B, T, 2, 39, 39)
    
    with torch.no_grad():
        sr_10m, sr_20m = model(lr_10m, lr_20m, target_size_10m=(260, 260), target_size_20m=(130, 130))
    
    print(f"\nInput: LR 10m {lr_10m.shape}, LR 20m {lr_20m.shape}")
    print(f"Output: SR 10m {sr_10m.shape}, SR 20m {sr_20m.shape}")
    
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
        print("✅ Model responds to different inputs")
    
    print("\n✅ V3 Model test passed!")
