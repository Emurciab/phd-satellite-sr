"""
Multi-Resolution Satellite Super-Resolution Model - DIRECT LEARNING

Key Difference from V2:
- NO BILINEAR BASELINE: Model directly learns full SR mapping
- Output = learned_full_SR (not baseline + residual)
- More challenging but may learn better representations

This is a comparison variant to test whether residual learning
provides benefits or if direct learning is equally/more effective.

Architecture Features:
- Same encoder/decoder as V2 for fair comparison
- Dual-branch for 10m and 20m inputs
- Feature fusion at appropriate scales  
- Spectral consistency preservation
- Temporal attention for multi-frame sequences
- DIRECT LEARNING: Output = full_learned_SR (no shortcuts)
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
# TEMPORAL ATTENTION
# =============================================================================

class TemporalAttention(nn.Module):
    """
    Temporal attention for multi-frame sequences.
    Uses V2-style weighted average (proven stable) instead of residual+gamma.
    
    This approach:
    - Learns per-frame weights via global pooling + MLP
    - Weights always sum to 1 (Softmax) - stable training
    - No middle-frame bias - works with Median/Temporal targets
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
            x: (B, T, C, H, W) temporal features
        Returns:
            fused: (B, C, H, W) attention-weighted fusion
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
    Simple output head with proper initialization for direct learning.
    
    Key improvements:
    1. Initialize as near-identity to preserve signal initially
    2. Bias initialized to expected satellite data mean (~0.05)
    3. Soft clamping to [0, 1] range
    """
    def __init__(self, in_channels, out_channels, init_bias=0.05):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        
        # Initialize for direct learning stability
        self._init_weights(init_bias)
    
    def _init_weights(self, init_bias):
        """Initialize weights to approximate identity, bias to data mean."""
        with torch.no_grad():
            # Small weights - don't amplify or suppress too much initially
            nn.init.xavier_uniform_(self.conv.weight, gain=0.1)
            
            # Bias initialized to expected output mean (satellite data ~0.05)
            if self.conv.bias is not None:
                nn.init.constant_(self.conv.bias, init_bias)
    
    def forward(self, x):
        x = self.conv(x)
        # Soft clamping to [0, 1] range
        return torch.clamp(x, 0.0, 1.0)


# =============================================================================
# MAIN MODEL: DIRECT LEARNING (NO BILINEAR BASELINE)
# =============================================================================

class MultiResolutionSatelliteSRDirect(nn.Module):
    """
    Direct learning super-resolution model.
    
    KEY DIFFERENCE: Learns full SR mapping without bilinear baseline.
    Output = fully_learned_SR (not baseline + residual)
    
    This variant:
    1. May learn better representations (no shortcuts)
    2. More challenging to train (larger gradients needed)
    3. Tests if residual learning is actually beneficial
    
    Architecture otherwise identical to V2 for fair comparison.
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
        
        # ========== TEMPORAL ATTENTION ==========
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
        self.output_head_10m = SimpleOutputHead(in_channels_10m, in_channels_10m, init_bias=0.05)
        self.output_head_20m = SimpleOutputHead(in_channels_20m, in_channels_20m, init_bias=0.05)
        
        # Initialize weights for stable direct learning
        self._init_direct_learning_weights()
    
    def _init_direct_learning_weights(self):
        """
        Initialize weights specifically for direct learning stability.
        
        Key insight: Direct learning needs the output to start close to 
        the expected data range (~0.05 mean for satellite data).
        """
        # Initialize final refinement layers to near-zero (start with identity)
        for refine_block in [self.final_refine_10m, self.final_refine_20m]:
            for module in refine_block.modules():
                if isinstance(module, nn.Conv2d):
                    # Very small weights so refinement starts near zero
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # Initialize upscaler final conv layers with controlled scale
        for upscaler in [self.upscaler_10m, self.upscaler_20m]:
            for name, module in upscaler.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Use smaller gain for output-producing layers
                    if module.out_channels <= 3:  # Output channels (RGB or 2-band)
                        nn.init.xavier_uniform_(module.weight, gain=0.1)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0.05)  # Start at data mean

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
        
        # NOTE: NO BILINEAR BASELINE COMPUTATION HERE
        # Model learns full mapping from scratch
        
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
        
        # ========== SPECTRAL CONSISTENCY ==========
        fused_10m = self.spectral_consistency_10m(fused_10m)
        fused_20m = self.spectral_consistency_20m(fused_20m)
        
        # ========== DIRECT UPSAMPLING (NO BASELINE) ==========
        # Calculate target sizes if not provided
        if target_size_10m is None:
            target_size_10m = (int(H_10m * self.scale_factor_10m), 
                             int(W_10m * self.scale_factor_10m))
        if target_size_20m is None:
            target_size_20m = (int(H_20m * self.scale_factor_20m), 
                             int(W_20m * self.scale_factor_20m))
        
        # Upscale features to get FULL SR output (not residuals)
        sr_10m = self.upscaler_10m(fused_10m, target_size_10m)
        sr_20m = self.upscaler_20m(fused_20m, target_size_20m)
        
        # ========== FINAL REFINEMENT ==========
        sr_10m = sr_10m + self.final_refine_10m(sr_10m)
        sr_20m = sr_20m + self.final_refine_20m(sr_20m)
        
        # ========== OUTPUT PROCESSING ==========
        sr_10m = self.output_head_10m(sr_10m)
        sr_20m = self.output_head_20m(sr_20m)
        
        return sr_10m, sr_20m


# =============================================================================
# LOSS FUNCTIONS (same as V2)
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
        pred_grad_x = F.conv2d(pred, self.sobel_x.repeat(pred.shape[1], 1, 1, 1), 
                               padding=1, groups=pred.shape[1])
        pred_grad_y = F.conv2d(pred, self.sobel_y.repeat(pred.shape[1], 1, 1, 1), 
                               padding=1, groups=pred.shape[1])
        
        target_grad_x = F.conv2d(target, self.sobel_x.repeat(target.shape[1], 1, 1, 1), 
                                 padding=1, groups=target.shape[1])
        target_grad_y = F.conv2d(target, self.sobel_y.repeat(target.shape[1], 1, 1, 1), 
                                 padding=1, groups=target.shape[1])
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return (loss_x + loss_y) / 2


# =============================================================================
# TEMPORAL CROSS-ATTENTION MODULE (for MultiOutput)
# =============================================================================

class TemporalCrossAttentionDirect(nn.Module):
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
        x_pooled = x.mean(dim=(-2, -1))  # (B, T, C)
        
        q = self.global_q(x_pooled)  # (B, T, C)
        k = self.global_k(x_pooled)  # (B, T, C)
        v = self.global_v(x_pooled)  # (B, T, C)
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, T, T)
        attn = F.softmax(attn, dim=-1)
        
        global_out = torch.matmul(attn, v)  # (B, T, C)
        
        # ========== LOCAL SPATIAL MIXING ==========
        x_cat = x.view(B, T * C, H, W)  # (B, T*C, H, W)
        local_mix = self.local_mix(x_cat)  # (B, C, H, W)
        
        # ========== COMBINE ==========
        outputs = []
        for t in range(T):
            frame_feat = x[:, t]  # (B, C, H, W)
            global_feat = global_out[:, t, :, None, None].expand(-1, -1, H, W)
            
            combined = torch.cat([frame_feat, local_mix], dim=1)
            gate = self.gate(combined)
            
            out = frame_feat + gate * (local_mix - frame_feat) + 0.1 * global_feat
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)


class TemporalFusionBlockDirect(nn.Module):
    """
    Block that processes temporal features with cross-attention and convolutions.
    """
    def __init__(self, channels, num_frames=3):
        super().__init__()
        
        self.cross_attn = TemporalCrossAttentionDirect(channels, num_frames)
        
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            out: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        
        attended = self.cross_attn(x)
        
        outputs = []
        for t in range(T):
            refined = self.refine(attended[:, t])
            outputs.append(attended[:, t] + self.gamma * refined)
        
        return torch.stack(outputs, dim=1)


# =============================================================================
# MULTI-OUTPUT DIRECT LEARNING MODEL (NO BILINEAR BASELINE)
# =============================================================================

class MultiOutputTemporalSRDirect(nn.Module):
    """
    Multi-Output Temporal Super-Resolution Model - DIRECT LEARNING.
    
    KEY DIFFERENCE: No bilinear baseline - learns full SR mapping directly.
    
    Produces T HR predictions from T LR inputs, with temporal cross-attention
    allowing frames to share information and improve each other.
    
    Architecture:
    1. Shared encoder processes all frames
    2. Temporal cross-attention allows information sharing
    3. T separate decoders predict T outputs
    4. DIRECT LEARNING: Output = fully_learned_SR (no baseline + residual)
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
        self.encoder_10m = MultiScaleEncoder(
            in_channels=in_channels_10m,
            base_channels=base_channels,
            num_blocks=num_encoder_blocks
        )
        
        self.encoder_20m = MultiScaleEncoder(
            in_channels=in_channels_20m,
            base_channels=base_channels,
            num_blocks=num_encoder_blocks
        )
        
        # ========== TEMPORAL CROSS-ATTENTION BLOCKS ==========
        self.temporal_blocks_10m = nn.ModuleList([
            TemporalFusionBlockDirect(base_channels, num_frames)
            for _ in range(num_temporal_blocks)
        ])
        
        self.temporal_blocks_20m = nn.ModuleList([
            TemporalFusionBlockDirect(base_channels, num_frames)
            for _ in range(num_temporal_blocks)
        ])
        
        # ========== CROSS-RESOLUTION FUSION ==========
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
        
        # ========== DECODERS (shared weights across frames) ==========
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
        
        # ========== DIRECT UPSAMPLING (learns full SR, not residuals) ==========
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
        self.output_head_10m = SimpleOutputHead(in_channels_10m, in_channels_10m, init_bias=0.05)
        self.output_head_20m = SimpleOutputHead(in_channels_20m, in_channels_20m, init_bias=0.05)
        
        # Initialize weights for stable direct learning
        self._init_direct_learning_weights()
    
    def _init_direct_learning_weights(self):
        """Initialize weights specifically for direct learning stability."""
        # Initialize final refinement layers to near-zero (start with identity)
        for refine_block in [self.final_refine_10m, self.final_refine_20m]:
            for module in refine_block.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # Initialize upscaler final conv layers with controlled scale
        for upscaler in [self.upscaler_10m, self.upscaler_20m]:
            for name, module in upscaler.named_modules():
                if isinstance(module, nn.Conv2d):
                    if module.out_channels <= 3:
                        nn.init.xavier_uniform_(module.weight, gain=0.1)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0.05)
        
    def forward(
        self,
        lr_10m: torch.Tensor,
        lr_20m: torch.Tensor,
        target_size_10m: tuple = None,
        target_size_20m: tuple = None
    ) -> tuple:
        """
        Forward pass producing T HR predictions via DIRECT LEARNING.
        
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
        
        # NOTE: NO BILINEAR BASELINE - Direct learning!
        
        # Calculate target sizes if not provided
        if target_size_10m is None:
            target_size_10m = (int(H_10m * self.scale_factor_10m), 
                             int(W_10m * self.scale_factor_10m))
        if target_size_20m is None:
            target_size_20m = (int(H_20m * self.scale_factor_20m), 
                             int(W_20m * self.scale_factor_20m))
        
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
        for block in self.temporal_blocks_10m:
            features_10m = block(features_10m)
        
        for block in self.temporal_blocks_20m:
            features_20m = block(features_20m)
        
        # ========== CROSS-RESOLUTION FUSION (per frame) ==========
        fused_10m_list = []
        fused_20m_list = []
        
        for t in range(T):
            feat_10m = features_10m[:, t]
            feat_20m = features_20m[:, t]
            
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
        
        # ========== DECODE AND UPSCALE EACH FRAME (DIRECT LEARNING) ==========
        outputs_10m = []
        outputs_20m = []
        
        for t in range(T):
            # Spectral consistency
            feat_10m = self.spectral_10m(fused_10m_list[t])
            feat_20m = self.spectral_20m(fused_20m_list[t])
            
            # Decode
            feat_10m = self.decoder_10m(feat_10m)
            feat_20m = self.decoder_20m(feat_20m)
            
            # DIRECT UPSCALING: Learn full SR (not residuals)
            sr_10m = self.upscaler_10m(feat_10m, target_size_10m)
            sr_20m = self.upscaler_20m(feat_20m, target_size_20m)
            
            # Final refinement with residual connection
            sr_10m = sr_10m + self.final_refine_10m(sr_10m)
            sr_20m = sr_20m + self.final_refine_20m(sr_20m)
            
            # NOTE: NO BASELINE ADDITION - this IS the full SR output
            
            # Final output (clamping)
            sr_10m = self.output_head_10m(sr_10m)
            sr_20m = self.output_head_20m(sr_20m)
            
            outputs_10m.append(sr_10m)
            outputs_20m.append(sr_20m)
        
        # Stack outputs: (B, T, C, H, W)
        sr_10m = torch.stack(outputs_10m, dim=1)
        sr_20m = torch.stack(outputs_20m, dim=1)
        
        return sr_10m, sr_20m
