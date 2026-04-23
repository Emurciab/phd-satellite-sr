"""
Multi-Resolution Satellite Super-Resolution Model V2 - LEARNED UPSCALING

This variant replaces the bilinear interpolation (154→256) with learned upscaling methods.

Two options are provided:
1. MetaUpscale (LIIF-inspired): Learns position-dependent interpolation using implicit functions
2. AdaptiveInterpolation: Learns content-aware blending weights for nearby pixels

Key insight: The PixelShuffle already does 77→154 (2×). We only need to learn 154→256 (1.66×).

Architecture:
- Same encoder/decoder as V2
- Same PixelShuffle for 2× upscaling
- NEW: Learned resize module for the final 154→256 step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal
import math

# Import shared building blocks from the original v2 model
from satellite_sr_model_v2 import (
    DepthwiseSeparableConv,
    ResidualBlock,
    ChannelAttention,
    SpatialAttention,
    CBAM,
    TemporalAttention,
    SpectralConsistencyModule,
    EncoderBranch,
    PixelShuffleUpscale,
    SimpleOutputHead,
    SpectralAngleLoss,
    GradientLoss,
    MultiResSpectralLoss,
    count_parameters
)


# =============================================================================
# OPTION A: META-UPSCALE (LIIF-INSPIRED)
# =============================================================================

class MetaUpscale(nn.Module):
    """
    Meta-learning based upscaling inspired by LIIF (Local Implicit Image Function).
    
    Instead of using fixed bilinear weights, this module learns to predict
    the output pixel value based on:
    1. Local features around the query position
    2. The relative position within the cell
    
    For 154→256 upscaling (scale ~1.66×), each output pixel queries
    nearby input pixels and uses learned weights based on position.
    
    Key idea: The network learns a continuous image representation,
    allowing arbitrary scale upsampling.
    """
    
    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Feature encoder - extracts rich local features
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        
        # Position encoder - encodes relative position (dx, dy) within cell
        # Input: 2 (dx, dy normalized to [-1, 1])
        # Also includes cell size info for scale-awareness
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # dx, dy, cell_h, cell_w
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Implicit function - predicts RGB from features + position
        # This is the core of LIIF: f(z, x) -> RGB where z=features, x=position
        self.implicit_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # features + position
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_channels),
        )
        
        # Learnable cell decoding for better gradient flow
        self.cell_decode = nn.Linear(hidden_dim, in_channels)
        
    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: (B, C, H_in, W_in) input features (e.g., 154×154)
            target_size: (H_out, W_out) desired output size (e.g., 256×256)
        
        Returns:
            output: (B, C, H_out, W_out) upscaled features
        """
        B, C, H_in, W_in = x.shape
        H_out, W_out = target_size
        
        # Extract local features
        features = self.feature_encoder(x)  # (B, hidden_dim, H_in, W_in)
        
        # Create output coordinate grid (normalized to [-1, 1])
        # These are the query positions in the output space
        y_coords = torch.linspace(-1, 1, H_out, device=x.device)
        x_coords = torch.linspace(-1, 1, W_out, device=x.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack into grid for grid_sample: (1, H_out, W_out, 2)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        grid = grid.expand(B, -1, -1, -1)
        
        # Sample features at query positions using bilinear (for feature lookup)
        sampled_features = F.grid_sample(
            features, grid, mode='bilinear', padding_mode='border', align_corners=True
        )  # (B, hidden_dim, H_out, W_out)
        
        # Compute relative positions within each cell
        # Scale factors
        scale_h = H_in / H_out
        scale_w = W_in / W_out
        
        # For each output pixel, compute its position relative to input cell center
        # This is the "continuous coordinate" that LIIF uses
        input_y = (grid_y + 1) / 2 * (H_in - 1)  # Map to [0, H_in-1]
        input_x = (grid_x + 1) / 2 * (W_in - 1)  # Map to [0, W_in-1]
        
        # Relative position within cell (normalized)
        rel_y = (input_y - input_y.floor()) * 2 - 1  # [-1, 1]
        rel_x = (input_x - input_x.floor()) * 2 - 1  # [-1, 1]
        
        # Cell size (for scale awareness)
        cell_h = torch.full_like(rel_y, 2.0 / H_in)
        cell_w = torch.full_like(rel_x, 2.0 / W_in)
        
        # Position encoding: (H_out, W_out, 4)
        pos_input = torch.stack([rel_x, rel_y, cell_h, cell_w], dim=-1)
        pos_input = pos_input.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H_out, W_out, 4)
        
        # Encode positions
        pos_encoded = self.pos_encoder(pos_input)  # (B, H_out, W_out, hidden_dim)
        
        # Reshape sampled features for concatenation
        sampled_features = sampled_features.permute(0, 2, 3, 1)  # (B, H_out, W_out, hidden_dim)
        
        # Concatenate features and position
        combined = torch.cat([sampled_features, pos_encoded], dim=-1)  # (B, H_out, W_out, hidden_dim*2)
        
        # Apply implicit function
        output = self.implicit_fn(combined)  # (B, H_out, W_out, C)
        
        # Reshape to (B, C, H_out, W_out)
        output = output.permute(0, 3, 1, 2)
        
        # Residual connection from bilinear upscale (helps training stability)
        bilinear_baseline = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        output = output + bilinear_baseline
        
        return output


# =============================================================================
# OPTION D: ADAPTIVE INTERPOLATION NETWORK
# =============================================================================

class AdaptiveInterpolation(nn.Module):
    """
    Content-aware adaptive interpolation network.
    
    Instead of fixed bilinear weights, this network learns to predict
    optimal blending weights based on local image content.
    
    For each output pixel, it:
    1. Looks at a local neighborhood in the input
    2. Predicts custom interpolation weights
    3. Applies weighted combination of nearby pixels
    
    This is simpler than MetaUpscale but still content-aware.
    Similar to "Learned Image Downscaling" but for upscaling.
    """
    
    def __init__(self, in_channels: int, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Number of weights to predict per output pixel
        # We predict weights for a kernel_size × kernel_size neighborhood
        num_weights = kernel_size * kernel_size
        
        # Weight predictor network
        # Takes input features and predicts per-pixel interpolation weights
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            # Output: num_weights channels (one weight per kernel position)
            nn.Conv2d(64, num_weights, 3, padding=1),
        )
        
        # Optional: Offset predictor for deformable-like behavior
        # Predicts small offsets for sampling positions
        self.use_offsets = True
        if self.use_offsets:
            self.offset_predictor = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 2 * num_weights, 3, padding=1),  # 2 (dx, dy) per weight
                nn.Tanh(),  # Limit offsets to [-1, 1] then scale
            )
            self.offset_scale = 2.0  # Maximum offset in pixels
        
        # Residual refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
        
    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: (B, C, H_in, W_in) input features
            target_size: (H_out, W_out) desired output size
        
        Returns:
            output: (B, C, H_out, W_out) upscaled features
        """
        B, C, H_in, W_in = x.shape
        H_out, W_out = target_size
        
        # First, bilinear upscale input to target size (for weight prediction)
        x_up = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Predict interpolation weights at target resolution
        weights = self.weight_predictor(x_up)  # (B, K*K, H_out, W_out)
        weights = F.softmax(weights, dim=1)  # Normalize weights to sum to 1
        
        # Create base sampling grid
        # For each output pixel, we need to sample from K×K input neighbors
        K = self.kernel_size
        
        # Generate offset grid for K×K kernel
        offsets_y = torch.arange(-(K//2), K//2 + 1, device=x.device, dtype=torch.float32)
        offsets_x = torch.arange(-(K//2), K//2 + 1, device=x.device, dtype=torch.float32)
        offset_grid_y, offset_grid_x = torch.meshgrid(offsets_y, offsets_x, indexing='ij')
        offset_grid = torch.stack([offset_grid_x.flatten(), offset_grid_y.flatten()], dim=-1)  # (K*K, 2)
        
        # Create output coordinate grid
        y_coords = torch.linspace(0, H_in - 1, H_out, device=x.device)
        x_coords = torch.linspace(0, W_in - 1, W_out, device=x.device)
        base_grid_y, base_grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        base_grid = torch.stack([base_grid_x, base_grid_y], dim=-1)  # (H_out, W_out, 2)
        
        # Predict offsets if enabled
        if self.use_offsets:
            pred_offsets = self.offset_predictor(x_up)  # (B, 2*K*K, H_out, W_out)
            pred_offsets = pred_offsets * self.offset_scale
            pred_offsets = pred_offsets.view(B, K*K, 2, H_out, W_out)
            pred_offsets = pred_offsets.permute(0, 1, 3, 4, 2)  # (B, K*K, H_out, W_out, 2)
        
        # Sample from input at all K*K positions for each output pixel
        output = torch.zeros(B, C, H_out, W_out, device=x.device)
        
        for k in range(K * K):
            # Base offset for this kernel position
            base_offset = offset_grid[k]  # (2,)
            
            # Sampling grid for this kernel position
            sample_grid = base_grid.clone()  # (H_out, W_out, 2)
            sample_grid[..., 0] += base_offset[0]
            sample_grid[..., 1] += base_offset[1]
            
            # Add predicted offsets if enabled
            if self.use_offsets:
                sample_grid = sample_grid.unsqueeze(0).expand(B, -1, -1, -1)
                sample_grid = sample_grid + pred_offsets[:, k]
            else:
                sample_grid = sample_grid.unsqueeze(0).expand(B, -1, -1, -1)
            
            # Normalize to [-1, 1] for grid_sample
            sample_grid_norm = sample_grid.clone()
            sample_grid_norm[..., 0] = sample_grid[..., 0] / (W_in - 1) * 2 - 1
            sample_grid_norm[..., 1] = sample_grid[..., 1] / (H_in - 1) * 2 - 1
            
            # Sample from input
            sampled = F.grid_sample(
                x, sample_grid_norm, mode='bilinear', padding_mode='border', align_corners=True
            )  # (B, C, H_out, W_out)
            
            # Weight and accumulate
            w = weights[:, k:k+1]  # (B, 1, H_out, W_out)
            output = output + sampled * w
        
        # Residual refinement
        output = output + self.refine(output)
        
        return output


# =============================================================================
# PROGRESSIVE UPSCALER WITH LEARNED RESIZE
# =============================================================================

class ProgressiveUpscalerLearned(nn.Module):
    """
    Progressive upscaling with LEARNED final resize (instead of bilinear).
    
    Pipeline:
    1. PixelShuffle 2× steps (77 → 154) - same as original
    2. Output convolution - same as original
    3. LEARNED resize (154 → 256) - NEW! Replaces bilinear
    
    Supports two resize methods:
    - 'meta': MetaUpscale (LIIF-inspired implicit function)
    - 'adaptive': AdaptiveInterpolation (content-aware weights)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        target_scale: float,
        base_channels: int = 64,
        resize_method: Literal['meta', 'adaptive', 'bilinear'] = 'adaptive'
    ):
        super().__init__()
        self.target_scale = target_scale
        self.resize_method = resize_method
        
        # Use 2x pixel shuffle steps (same as original)
        num_2x_steps = int(math.log2(target_scale)) if target_scale >= 2 else 0
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_2x_steps):
            layers.append(PixelShuffleUpscale(current_channels, base_channels, 2))
            current_channels = base_channels
        
        self.upscale_blocks = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Output conv (same as original)
        self.output_conv = nn.Sequential(
            nn.Conv2d(current_channels if num_2x_steps > 0 else in_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
        # NEW: Learned resize module for the final step
        if resize_method == 'meta':
            self.learned_resize = MetaUpscale(out_channels, hidden_dim=64)
        elif resize_method == 'adaptive':
            self.learned_resize = AdaptiveInterpolation(out_channels, kernel_size=5)
        else:
            self.learned_resize = None  # Fall back to bilinear
    
    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            target_size: (H', W') exact output size
        
        Returns:
            upscaled: (B, out_channels, H', W')
        """
        # PixelShuffle upscaling (e.g., 77 → 154)
        x = self.upscale_blocks(x)
        
        # Output conv
        x = self.output_conv(x)
        
        # Final resize to target size
        if target_size is not None and x.shape[-2:] != target_size:
            if self.learned_resize is not None:
                # Use learned resize (MetaUpscale or AdaptiveInterpolation)
                x = self.learned_resize(x, target_size)
            else:
                # Fall back to bilinear
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


# =============================================================================
# MAIN MODEL WITH LEARNED UPSCALING
# =============================================================================

class MultiResolutionSatelliteSRv2Learned(nn.Module):
    """
    Multi-Resolution Super-Resolution Model V2 with Learned Upscaling.
    
    Same architecture as V2 but with learned resize instead of bilinear
    for the final 154→256 upscaling step.
    
    Residual learning architecture:
    Output = bilinear_upscale(LR_middle_frame) + learned_residual
    
    The learned upscaling is used in the residual path (ProgressiveUpscaler).
    The baseline can optionally also use learned upscaling.
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
        resize_method: Literal['meta', 'adaptive', 'bilinear'] = 'adaptive',
        learned_baseline: bool = False,  # Whether to use learned upscale for baseline too
    ):
        super().__init__()
        
        self.scale_factor_10m = scale_factor_10m
        self.scale_factor_20m = scale_factor_20m
        self.num_frames = num_frames
        self.use_cross_resolution_fusion = use_cross_resolution_fusion
        self.in_channels_10m = in_channels_10m
        self.in_channels_20m = in_channels_20m
        self.resize_method = resize_method
        self.learned_baseline = learned_baseline
        
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
        
        # Upscalers with LEARNED resize
        self.upscaler_10m = ProgressiveUpscalerLearned(
            base_channels, in_channels_10m, scale_factor_10m, base_channels,
            resize_method=resize_method
        )
        self.upscaler_20m = ProgressiveUpscalerLearned(
            base_channels, in_channels_20m, scale_factor_20m, base_channels,
            resize_method=resize_method
        )
        
        # Optional: Learned upscale for baseline
        if learned_baseline:
            if resize_method == 'meta':
                self.baseline_resize_10m = MetaUpscale(in_channels_10m, hidden_dim=32)
                self.baseline_resize_20m = MetaUpscale(in_channels_20m, hidden_dim=32)
            elif resize_method == 'adaptive':
                self.baseline_resize_10m = AdaptiveInterpolation(in_channels_10m, kernel_size=5)
                self.baseline_resize_20m = AdaptiveInterpolation(in_channels_20m, kernel_size=5)
        
        # Residual refinement
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
        
        # Output heads
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
            target_size_20m: (H', W') exact output size for 20m bands
        
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
        
        # Compute target sizes
        if target_size_10m is None:
            target_size_10m = (int(H_10m * self.scale_factor_10m), int(W_10m * self.scale_factor_10m))
        if target_size_20m is None:
            target_size_20m = (int(H_20m * self.scale_factor_20m), int(W_20m * self.scale_factor_20m))
        
        # Create baselines (bilinear or learned)
        if self.learned_baseline:
            baseline_10m = self.baseline_resize_10m(lr_10m_middle, target_size_10m)
            baseline_20m = self.baseline_resize_20m(lr_20m_middle, target_size_20m)
        else:
            baseline_10m = F.interpolate(lr_10m_middle, size=target_size_10m, 
                                         mode='bilinear', align_corners=False)
            baseline_20m = F.interpolate(lr_20m_middle, size=target_size_20m,
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
        
        # Upscale features with LEARNED resize (this is the key change!)
        residual_10m = self.upscaler_10m(fused_10m, target_size_10m)
        residual_20m = self.upscaler_20m(fused_20m, target_size_20m)
        
        # Refine residuals
        residual_10m = self.residual_refine_10m(residual_10m)
        residual_20m = self.residual_refine_20m(residual_20m)
        
        # Scale residuals
        residual_10m = residual_10m * self.residual_scale_10m
        residual_20m = residual_20m * self.residual_scale_20m
        
        # RESIDUAL LEARNING: Output = Baseline + Learned Residual
        sr_10m = baseline_10m + residual_10m
        sr_20m = baseline_20m + residual_20m
        
        # Final output processing
        sr_10m = self.output_head_10m(sr_10m)
        sr_20m = self.output_head_20m(sr_20m)
        
        return sr_10m, sr_20m


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_model_v2_learned(
    base_channels: int = 64,
    num_encoder_blocks: int = 4,
    num_frames: int = 3,
    in_channels_10m: int = 3,
    in_channels_20m: int = 2,
    resize_method: Literal['meta', 'adaptive', 'bilinear'] = 'adaptive',
    learned_baseline: bool = False
):
    """Create the V2 model with learned upscaling."""
    model = MultiResolutionSatelliteSRv2Learned(
        base_channels=base_channels,
        num_encoder_blocks=num_encoder_blocks,
        num_frames=num_frames,
        in_channels_10m=in_channels_10m,
        in_channels_20m=in_channels_20m,
        resize_method=resize_method,
        learned_baseline=learned_baseline
    )
    
    print(f"Created MultiResolutionSatelliteSRv2Learned")
    print(f"  - Base channels: {base_channels}")
    print(f"  - Encoder blocks: {num_encoder_blocks}")
    print(f"  - Temporal frames: {num_frames}")
    print(f"  - 10m channels: {in_channels_10m}")
    print(f"  - 20m channels: {in_channels_20m}")
    print(f"  - Resize method: {resize_method}")
    print(f"  - Learned baseline: {learned_baseline}")
    print(f"  - Total parameters: {count_parameters(model):,}")
    
    return model


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Learned Upscaling Modules")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test dimensions (typical for our SR task)
    B = 2
    C = 3
    H_in, W_in = 154, 154
    H_out, W_out = 256, 256
    
    x = torch.randn(B, C, H_in, W_in).to(device)
    
    # Test MetaUpscale
    print("\n--- Testing MetaUpscale ---")
    meta_upscale = MetaUpscale(C, hidden_dim=64).to(device)
    print(f"MetaUpscale parameters: {count_parameters(meta_upscale):,}")
    
    with torch.no_grad():
        out_meta = meta_upscale(x, (H_out, W_out))
    print(f"Input:  {x.shape}")
    print(f"Output: {out_meta.shape}")
    
    # Test AdaptiveInterpolation
    print("\n--- Testing AdaptiveInterpolation ---")
    adaptive_interp = AdaptiveInterpolation(C, kernel_size=5).to(device)
    print(f"AdaptiveInterpolation parameters: {count_parameters(adaptive_interp):,}")
    
    with torch.no_grad():
        out_adaptive = adaptive_interp(x, (H_out, W_out))
    print(f"Input:  {x.shape}")
    print(f"Output: {out_adaptive.shape}")
    
    # Test full model with both methods
    print("\n--- Testing Full Model (adaptive) ---")
    model_adaptive = create_model_v2_learned(resize_method='adaptive')
    model_adaptive = model_adaptive.to(device)
    
    T = 3
    lr_10m = torch.randn(B, T, 3, 77, 77).to(device)
    lr_20m = torch.randn(B, T, 2, 38, 38).to(device)
    
    with torch.no_grad():
        sr_10m, sr_20m = model_adaptive(lr_10m, lr_20m, (256, 256), (128, 128))
    
    print(f"LR 10m: {lr_10m.shape} -> SR 10m: {sr_10m.shape}")
    print(f"LR 20m: {lr_20m.shape} -> SR 20m: {sr_20m.shape}")
    
    print("\n--- Testing Full Model (meta) ---")
    model_meta = create_model_v2_learned(resize_method='meta')
    model_meta = model_meta.to(device)
    print(f"Meta model parameters: {count_parameters(model_meta):,}")
    
    with torch.no_grad():
        sr_10m, sr_20m = model_meta(lr_10m, lr_20m, (256, 256), (128, 128))
    
    print(f"LR 10m: {lr_10m.shape} -> SR 10m: {sr_10m.shape}")
    print(f"LR 20m: {lr_20m.shape} -> SR 20m: {sr_20m.shape}")
    
    # Compare with bilinear baseline
    print("\n--- Testing Full Model (bilinear baseline) ---")
    model_bilinear = create_model_v2_learned(resize_method='bilinear')
    model_bilinear = model_bilinear.to(device)
    
    print(f"\nParameter comparison:")
    print(f"  Bilinear:  {count_parameters(model_bilinear):,}")
    print(f"  Adaptive:  {count_parameters(model_adaptive):,}")
    print(f"  Meta:      {count_parameters(model_meta):,}")
    
    print("\n✅ All tests passed!")
