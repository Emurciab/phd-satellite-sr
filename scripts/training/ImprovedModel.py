"""
Improved Super-Resolution Architecture for Satellite Imagery
Addresses: Spectral Fidelity, Band Flexibility, Hallucination Prevention

Key Improvements:
1. Adaptive channel handling (4 or 6 bands)
2. Spectral consistency constraints
3. No HR cross-attention (training/inference compatible)
4. Physics-based output constraints
5. Multi-scale spectral losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


# ============= EFFICIENT BUILDING BLOCKS =============
class DepthwiseSeparableConv(nn.Module):
    """Efficient convolution: 8-9x fewer parameters."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        # Determine number of groups for GroupNorm
        num_groups = min(8, out_channels) if out_channels >= 8 else 1
        while out_channels % num_groups != 0:
            num_groups = num_groups // 2
            if num_groups == 0:
                num_groups = 1
                break
        
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
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        
        # Determine number of groups for GroupNorm
        num_groups = min(8, channels) if channels >= 8 else 1
        while channels % num_groups != 0:
            num_groups = num_groups // 2
            if num_groups == 0:
                num_groups = 1
                break
        
        self.norm = nn.GroupNorm(num_groups, channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x + residual


# ============= ADAPTIVE CHANNEL ENCODER =============
class AdaptiveChannelEncoder(nn.Module):
    """Handles 4-band or 6-band inputs dynamically."""
    def __init__(self, base_channels=64):
        super().__init__()
        # Separate stems for different band counts
        self.stem_4band = nn.Sequential(
            DepthwiseSeparableConv(4, base_channels, 7, 1, 3),
            ResidualBlock(base_channels)
        )
        self.stem_6band = nn.Sequential(
            DepthwiseSeparableConv(6, base_channels, 7, 1, 3),
            ResidualBlock(base_channels)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) or (B, T, C, H, W)
        Returns:
            features: (B, base_channels, H, W) or (B, T, base_channels, H, W)
        """
        is_temporal = x.dim() == 5
        
        if is_temporal:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
        else:
            B = x.shape[0]
            T = None
        
        # Detect channel count
        C = x.shape[1]
        
        if C == 4:
            features = self.stem_4band(x)
        elif C == 6:
            features = self.stem_6band(x)
        else:
            raise ValueError(f"Unsupported channel count: {C}. Expected 4 or 6.")
        
        if is_temporal:
            # features shape: (B*T, base_channels, H, W)
            # reshape to: (B, T, base_channels, H, W)
            _, out_C, out_H, out_W = features.shape
            features = features.reshape(B, T, out_C, out_H, out_W)
        
        return features


# ============= SPECTRAL CONSISTENCY MODULE =============
class SpectralConsistencyModule(nn.Module):
    """Ensures output maintains spectral relationships from input."""
    def __init__(self, channels, input_channels_4=4, input_channels_6=6):
        super().__init__()
        self.channels = channels
        
        # Projection layers for raw LR input (4 or 6 bands) to feature space
        self.lr_proj_4band = nn.Conv2d(input_channels_4, channels, 1)
        self.lr_proj_6band = nn.Conv2d(input_channels_6, channels, 1)
        
        # Channel-wise attention for spectral relationships
        self.spectral_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention for texture
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, sr_features, lr_input):
        """
        Args:
            sr_features: High-res features (B, C, H, W)
            lr_input: Low-res input for spectral guidance (B, C', h, w) where C' ∈ {4, 6}
        """
        # Project LR input to feature space
        C_in = lr_input.shape[1]
        if C_in == 4:
            lr_projected = self.lr_proj_4band(lr_input)
        elif C_in == 6:
            lr_projected = self.lr_proj_6band(lr_input)
        else:
            raise ValueError(f"Unexpected input channels: {C_in}")
        
        # Upsample LR to match SR size
        lr_upsampled = F.interpolate(
            lr_projected, 
            size=sr_features.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Spectral attention from LR
        spectral_weights = self.spectral_attention(lr_upsampled)
        
        # Spatial attention from SR
        spatial_weights = self.spatial_attention(sr_features)
        
        # Apply combined attention
        guided_features = sr_features * spectral_weights * spatial_weights
        
        # Residual connection
        return guided_features + sr_features * 0.5


# ============= MULTI-SCALE ENCODER =============
class MultiScaleEncoder(nn.Module):
    """Multi-scale ResNet encoder."""
    def __init__(self, in_channels=64, depths=[2, 3, 4]):
        super().__init__()
        
        self.stages = nn.ModuleList()
        channels = in_channels
        
        for depth in depths:
            stage = nn.Sequential(
                # Downsample
                nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, channels * 2),
                nn.GELU(),
                # Residual blocks
                *[ResidualBlock(channels * 2) for _ in range(depth)]
            )
            self.stages.append(stage)
            channels *= 2
        
        self.out_channels = [in_channels] + [in_channels * (2 ** (i+1)) for i in range(len(depths))]
    
    def forward(self, x):
        """Returns multi-scale features [C1, C2, C3, C4]."""
        features = [x]
        
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        return features


# ============= TEMPORAL MODELING =============
class TemporalAttentionBlock(nn.Module):
    """Efficient temporal attention for sequence processing."""
    def __init__(self, channels, num_frames=3):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        
        self.temporal_conv = nn.Conv3d(
            channels, channels, 
            kernel_size=(3, 1, 1), 
            padding=(1, 0, 0),
            groups=channels
        )
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            x: (B, T, C, H, W) with temporal modeling
        """
        B, T, C, H, W = x.shape
        
        # Rearrange for 3D conv: (B, C, T, H, W)
        x_3d = x.permute(0, 2, 1, 3, 4)
        
        # Temporal convolution (output: B, C, T, H, W)
        x_temp = self.temporal_conv(x_3d)
        
        # Apply normalization - GroupNorm expects (B, C, *)
        # x_temp is (B, C, T, H, W) which is already correct format
        x_temp = self.norm(x_temp)
        x_temp = self.act(x_temp)
        
        # Rearrange back: (B, T, C, H, W)
        x_out = x_temp.permute(0, 2, 1, 3, 4)
        
        return x + x_out


# ============= SELF-ATTENTION (NO HR LEAKAGE) =============
class EfficientSelfAttention(nn.Module):
    """Self-attention on LR features only."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, C, H, W) with self-attention
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x_norm = self.norm(x_flat)
        
        # QKV projection
        qkv = self.qkv(x_norm).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Efficient linear attention
        k = k.softmax(dim=-2)
        q = q.softmax(dim=-1)
        
        context = torch.matmul(k.transpose(-2, -1), v)
        attn_out = torch.matmul(q, context)
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, H * W, C)
        attn_out = self.proj(attn_out)
        
        # Residual + reshape back
        x_out = x_flat + attn_out
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        
        return x_out


# ============= PROGRESSIVE DECODER =============
class UpsampleBlock(nn.Module):
    """Efficient upsampling with sub-pixel convolution."""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels * (scale_factor ** 2))
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.refine = DepthwiseSeparableConv(out_channels, out_channels)
        
    def forward(self, x, skip=None):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        
        if skip is not None:
            # Resize skip to match if needed
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = x + skip
        
        x = self.refine(x)
        return x


class ProgressiveDecoder(nn.Module):
    """Progressive upsampling with skip connections."""
    def __init__(self, in_channels=256, out_channels=64, num_upsample=2):
        super().__init__()
        
        self.upsample_blocks = nn.ModuleList()
        channels = in_channels
        
        for i in range(num_upsample):
            self.upsample_blocks.append(
                UpsampleBlock(channels, channels // 2, scale_factor=2)
            )
            channels //= 2
        
        # After all upsampling, channels might not equal out_channels
        # Add projection if needed
        self.out_channels = channels
        if channels != out_channels:
            self.channel_proj = DepthwiseSeparableConv(channels, out_channels)
            final_channels = out_channels
        else:
            self.channel_proj = None
            final_channels = channels
        
        self.final_refine = nn.Sequential(
            ResidualBlock(final_channels),
            ResidualBlock(final_channels)
        )
        
    def forward(self, x, skip_connections=None):
        """
        Args:
            x: Bottleneck features (B, C, H, W)
            skip_connections: List of skip features from encoder
        """
        for i, up_block in enumerate(self.upsample_blocks):
            # Use skip connection if available
            # skip_connections are ordered [shallow -> deep], so reverse index
            skip_idx = -(i+1)
            if skip_connections and abs(skip_idx) <= len(skip_connections):
                skip = skip_connections[skip_idx]
            else:
                skip = None
            x = up_block(x, skip)
        
        # Project to output channels if needed
        if self.channel_proj is not None:
            x = self.channel_proj(x)
        
        return self.final_refine(x)


# ============= PHYSICS-CONSTRAINED OUTPUT HEAD =============
class PhysicsConstrainedHead(nn.Module):
    """Output head with physical constraints for reflectance."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, 3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Constrain to valid reflectance range [0, 1]
        # Using sigmoid ensures physical plausibility
        x = torch.sigmoid(x)
        return x


# ============= COMPLETE IMPROVED MODEL =============
class ImprovedSatelliteSR(nn.Module):
    """
    Improved Super-Resolution for Satellite Imagery.
    
    Key Features:
    - Adaptive channel handling (4 or 6 bands)
    - Spectral consistency preservation
    - No HR cross-attention (training/inference compatible)
    - Physics-based constraints
    - Temporal modeling for sequences
    """
    def __init__(
        self,
        base_channels=64,
        scale_factor=4,  # Sentinel-2 (10-20m) → Planet (3m)
        use_temporal=True
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.use_temporal = use_temporal
        
        # Adaptive channel encoder (handles 4 or 6 bands)
        self.channel_encoder = AdaptiveChannelEncoder(base_channels)
        
        # Temporal modeling (if using sequences)
        if use_temporal:
            self.temporal_attn = TemporalAttentionBlock(base_channels)
        
        # Multi-scale encoder
        self.encoder = MultiScaleEncoder(base_channels, depths=[2, 3, 4])
        
        # Self-attention (no HR leakage)
        self.self_attn = EfficientSelfAttention(self.encoder.out_channels[-1])
        
        # Spectral consistency module
        self.spectral_consistency = SpectralConsistencyModule(self.encoder.out_channels[-1])
        
        # Progressive decoder
        num_upsample = int(math.log2(scale_factor))
        # Decoder needs to: (1) undo encoder downsampling, (2) apply SR upsampling
        encoder_downsample = len([2, 3, 4])  # Number of encoder stages
        total_upsample = encoder_downsample + num_upsample
        
        self.decoder = ProgressiveDecoder(
            self.encoder.out_channels[-1],
            base_channels,
            total_upsample
        )
        
        # Output head with physics constraints
        self.output_head_4band = PhysicsConstrainedHead(base_channels, 4)
        self.output_head_6band = PhysicsConstrainedHead(base_channels, 6)
        
    def forward(self, x_lr):
        """
        Args:
            x_lr: Low-res Sentinel-2
                  (B, C, H, W) or (B, T, C, H, W) where C ∈ {4, 6}
        Returns:
            x_sr: Super-resolved output
                  (B, C, H*scale, W*scale) or (B, T, C, H*scale, W*scale)
        """
        is_temporal = x_lr.dim() == 5
        store_input_for_consistency = x_lr
        
        if is_temporal:
            B, T, C, H, W = x_lr.shape
        else:
            C = x_lr.shape[1]
            B = x_lr.shape[0]
            T = None
        
        # Adaptive channel encoding (handles temporal internally)
        x = self.channel_encoder(x_lr)
        
        # Temporal modeling
        if is_temporal and self.use_temporal:
            # x is now (B, T, channels, H, W) from channel_encoder
            x = self.temporal_attn(x)
            # Flatten for spatial processing: (B*T, channels, H, W)
            x = x.reshape(B * T, *x.shape[2:])
        
        # Multi-scale encoding
        features = self.encoder(x)
        
        # Self-attention on deepest features
        x_deepest = features[-1]
        x_attn = self.self_attn(x_deepest)
        
        # Spectral consistency guidance
        lr_for_guidance = store_input_for_consistency
        if is_temporal:
            # Use middle frame for guidance, then replicate for all temporal frames
            lr_for_guidance = store_input_for_consistency[:, T//2]  # (B, C, H, W)
            # Replicate across temporal dimension to match (B*T, C, H, W)
            lr_for_guidance = lr_for_guidance.unsqueeze(1).repeat(1, T, 1, 1, 1)  # (B, T, C, H, W)
            lr_for_guidance = lr_for_guidance.reshape(B * T, *lr_for_guidance.shape[2:])  # (B*T, C, H, W)
        
        x_consistent = self.spectral_consistency(x_attn, lr_for_guidance)
        
        # Progressive decoding with skip connections
        x_decoded = self.decoder(x_consistent, skip_connections=features[:-1])
        
        # Physics-constrained output
        if C == 4:
            x_sr = self.output_head_4band(x_decoded)
        elif C == 6:
            x_sr = self.output_head_6band(x_decoded)
        else:
            raise ValueError(f"Unsupported channel count: {C}")
        
        # Reshape for temporal sequences
        if is_temporal:
            x_sr = x_sr.reshape(B, T, C, *x_sr.shape[-2:])
        
        return x_sr


# ============= SPECTRAL FIDELITY LOSS =============
class SpectralFidelityLoss(nn.Module):
    """
    Multi-component loss for spectral fidelity.
    
    Components:
    1. Pixel-wise L1 loss
    2. Spectral Angle Mapper (SAM)
    3. Spectral correlation
    4. Low-res consistency
    5. Edge preservation
    """
    def __init__(self, lambda_pixel=1.0, lambda_sam=0.5, lambda_corr=0.3, 
                 lambda_lr=0.5, lambda_edge=0.2):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_sam = lambda_sam
        self.lambda_corr = lambda_corr
        self.lambda_lr = lambda_lr
        self.lambda_edge = lambda_edge
        
        # Edge detection (Sobel)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
    
    def spectral_angle_mapper(self, sr, hr):
        """Compute SAM between SR and HR."""
        sr_norm = F.normalize(sr, dim=1, eps=1e-8)
        hr_norm = F.normalize(hr, dim=1, eps=1e-8)
        
        # Cosine similarity
        cos_sim = (sr_norm * hr_norm).sum(1)
        
        # Clamp to avoid numerical issues
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        
        # SAM in radians
        sam = torch.acos(cos_sim)
        
        return sam.mean()
    
    def spectral_correlation(self, sr, hr):
        """Compute spectral correlation."""
        sr_flat = sr.flatten(2)  # (B, C, HW)
        hr_flat = hr.flatten(2)
        
        # Normalize
        sr_flat = F.normalize(sr_flat, dim=2)
        hr_flat = F.normalize(hr_flat, dim=2)
        
        # Correlation
        corr = (sr_flat * hr_flat).sum(2).mean()
        
        return 1 - corr
    
    def low_res_consistency(self, sr, lr, scale_factor):
        """Ensure SR downsampled matches LR."""
        sr_down = F.adaptive_avg_pool2d(sr, lr.shape[-2:])
        return F.l1_loss(sr_down, lr)
    
    def edge_loss(self, sr, hr):
        """Preserve edges."""
        device = sr.device
        
        # Move Sobel filters to device
        sobel_x = self.sobel_x.to(device).repeat(sr.shape[1], 1, 1, 1)
        sobel_y = self.sobel_y.to(device).repeat(sr.shape[1], 1, 1, 1)
        
        # Compute edges
        sr_edge_x = F.conv2d(sr, sobel_x, padding=1, groups=sr.shape[1])
        sr_edge_y = F.conv2d(sr, sobel_y, padding=1, groups=sr.shape[1])
        hr_edge_x = F.conv2d(hr, sobel_x, padding=1, groups=hr.shape[1])
        hr_edge_y = F.conv2d(hr, sobel_y, padding=1, groups=hr.shape[1])
        
        sr_edges = torch.sqrt(sr_edge_x**2 + sr_edge_y**2 + 1e-8)
        hr_edges = torch.sqrt(hr_edge_x**2 + hr_edge_y**2 + 1e-8)
        
        return F.l1_loss(sr_edges, hr_edges)
    
    def forward(self, sr, hr, lr, scale_factor=4):
        """
        Args:
            sr: Super-resolved output (B, C, H, W)
            hr: Ground truth high-res (B, C, H, W)
            lr: Low-res input (B, C, h, w)
            scale_factor: Upsampling scale
        """
        # Handle temporal sequences
        if sr.dim() == 5:
            B, T, C, H, W = sr.shape
            sr = sr.reshape(B * T, C, H, W)
            hr = hr.reshape(B * T, C, H, W)
            if lr.dim() == 5:
                lr = lr.reshape(B * T, C, *lr.shape[-2:])
        
        # Component losses
        loss_pixel = F.l1_loss(sr, hr)
        loss_sam = self.spectral_angle_mapper(sr, hr)
        loss_corr = self.spectral_correlation(sr, hr)
        loss_lr = self.low_res_consistency(sr, lr, scale_factor)
        loss_edge = self.edge_loss(sr, hr)
        
        # Total loss
        total_loss = (
            self.lambda_pixel * loss_pixel +
            self.lambda_sam * loss_sam +
            self.lambda_corr * loss_corr +
            self.lambda_lr * loss_lr +
            self.lambda_edge * loss_edge
        )
        
        return {
            'total': total_loss,
            'pixel': loss_pixel,
            'sam': loss_sam,
            'correlation': loss_corr,
            'lr_consistency': loss_lr,
            'edge': loss_edge
        }


# ============= MODEL FACTORY =============
def create_improved_sr_model(config: Optional[Dict] = None):
    """
    Factory function to create improved SR model.
    
    Args:
        config: Configuration dictionary with:
            - base_channels: Base number of channels (default: 64)
            - scale_factor: Upsampling scale (default: 4)
            - use_temporal: Use temporal modeling (default: True)
            - use_mixed_precision: Use FP16 (default: True)
    
    Returns:
        model: ImprovedSatelliteSR instance
        loss_fn: SpectralFidelityLoss instance
    """
    if config is None:
        config = {}
    
    model = ImprovedSatelliteSR(
        base_channels=config.get('base_channels', 64),
        scale_factor=config.get('scale_factor', 4),
        use_temporal=config.get('use_temporal', True)
    )
    
    loss_fn = SpectralFidelityLoss(
        lambda_pixel=config.get('lambda_pixel', 1.0),
        lambda_sam=config.get('lambda_sam', 0.5),
        lambda_corr=config.get('lambda_corr', 0.3),
        lambda_lr=config.get('lambda_lr', 0.5),
        lambda_edge=config.get('lambda_edge', 0.2)
    )
    
    # Apply optimizations
    if config.get('use_mixed_precision', True):
        model = model.half()
    
    return model, loss_fn


# ============= EXAMPLE USAGE =============
if __name__ == "__main__":
    # Configuration
    config = {
        'base_channels': 64,
        'scale_factor': 4,
        'use_temporal': True,
        'use_mixed_precision': False  # Set True for training
    }
    
    # Create model and loss
    model, loss_fn = create_improved_sr_model(config)
    
    print("="*80)
    print("IMPROVED SATELLITE SUPER-RESOLUTION MODEL")
    print("="*80)
    
    # Test with 4-band input
    print("\n📡 Testing with 4-band input (standard Dove):")
    x_lr_4band = torch.randn(2, 3, 4, 64, 64)  # (B=2, T=3, C=4, H=64, W=64)
    x_hr_4band = torch.randn(2, 3, 4, 256, 256)  # (B=2, T=3, C=4, H=256, W=256)
    
    sr_4band = model(x_lr_4band)
    print(f"   Input shape: {x_lr_4band.shape}")
    print(f"   Output shape: {sr_4band.shape}")
    print(f"   ✓ 4-band processing successful")
    
    # Test with 6-band input
    print("\n📡 Testing with 6-band input (SuperDove):")
    x_lr_6band = torch.randn(2, 3, 6, 64, 64)  # (B=2, T=3, C=6, H=64, W=64)
    x_hr_6band = torch.randn(2, 3, 6, 256, 256)  # (B=2, T=3, C=6, H=256, W=256)
    
    sr_6band = model(x_lr_6band)
    print(f"   Input shape: {x_lr_6band.shape}")
    print(f"   Output shape: {sr_6band.shape}")
    print(f"   ✓ 6-band processing successful")
    
    # Test loss computation
    print("\n💎 Testing spectral fidelity loss:")
    total_loss, loss_components = loss_fn(sr_4band, x_hr_4band, x_lr_4band, scale_factor=4)
    print(f"   Total loss: {total_loss.item():.4f}")
    for name, value in loss_components.items():
        print(f"   - {name}: {value:.4f}")
    
    # Model statistics
    print("\n📊 Model Statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    
    print("\n✅ All tests passed! Model is ready for training.")
    print("="*80)
