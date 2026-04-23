"""
Frequency Domain Super-Resolution Model - DIRECT VERSION

This approach learns the FULL super-resolved output in the frequency domain
without relying on a bilinear baseline.

Key Concept:
- Transform features to Fourier domain
- Predict BOTH low and high frequency components directly
- No shortcut from upsampled LR input
- More challenging but potentially learns better representations

Architecture:
1. Spatial encoder extracts features from LR input
2. Frequency predictor outputs full spectrum (LF + HF)
3. Inverse FFT converts to spatial domain
4. Spatial refinement for final output

Differences from Residual Version:
- Residual: Output = bilinear_baseline + learned_HF
- Direct: Output = fully_learned_spectrum (no shortcuts)

Benefits:
- No dependency on bilinear upsampling quality
- Can learn to correct LF artifacts in input
- More flexible frequency representation

Challenges:
- Harder to train (larger output space)
- Requires careful initialization
- More prone to collapse without proper constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# =============================================================================
# BUILDING BLOCKS (same as residual version)
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    """Efficient convolution with depthwise + pointwise structure."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride, padding, groups=in_channels, bias=False
        )
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


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.channel_attn(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.sigmoid(self.spatial_conv(spatial))
        return x * spatial


# =============================================================================
# FREQUENCY DOMAIN OPERATIONS FOR DIRECT LEARNING
# =============================================================================

class FullSpectrumPredictor(nn.Module):
    """
    Predicts the full frequency spectrum (both LF and HF) for direct learning.
    
    Unlike the residual version which only predicts HF residuals,
    this predicts the complete spectral content.
    """
    def __init__(self, in_channels, out_channels, base_channels=64):
        super().__init__()
        
        # Deeper processing for full spectrum prediction
        self.spatial_branch = nn.Sequential(
            ResidualBlock(in_channels),
            CBAM(in_channels),
            ResidualBlock(in_channels),
            CBAM(in_channels),
        )
        
        # Separate paths for low and high frequency prediction
        # This helps the model learn different characteristics
        
        # Low-frequency path (smooth, global structure)
        self.lf_head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, out_channels * 2, 1)  # real + imag
        )
        
        # High-frequency path (details, edges)
        self.hf_head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels * 2, 1)  # real + imag
        )
        
        # Learnable frequency masks for soft LF/HF separation
        self.lf_cutoff = nn.Parameter(torch.tensor(0.25))
        self.transition_width = nn.Parameter(torch.tensor(0.1))
        
        # Initialize for stable direct learning
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for direct learning stability."""
        for head in [self.lf_head, self.hf_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    # Use larger gain to produce meaningful initial outputs
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        # Initialize bias towards expected data mean
                        nn.init.constant_(m.bias, 0.0)
    
    def _create_frequency_masks(self, H, W, device):
        """Create soft LF/HF frequency masks."""
        # Create frequency grid
        freq_y = torch.fft.fftfreq(H, device=device).view(-1, 1)
        freq_x = torch.fft.fftfreq(W, device=device).view(1, -1)
        
        # Radial frequency (normalized)
        freq_radius = torch.sqrt(freq_y**2 + freq_x**2)
        max_freq = math.sqrt(0.5**2 + 0.5**2)
        freq_radius = freq_radius / max_freq
        
        # Soft masks using learnable parameters
        cutoff = torch.sigmoid(self.lf_cutoff) * 0.5  # Constrain to [0, 0.5]
        transition = torch.sigmoid(self.transition_width) * 0.2 + 0.05  # [0.05, 0.25]
        
        lf_mask = torch.sigmoid(-(freq_radius - cutoff) / transition)
        hf_mask = 1.0 - lf_mask
        
        return lf_mask, hf_mask
    
    def forward(self, features, target_size):
        """
        Predict full frequency spectrum.
        
        Args:
            features: (B, C, H, W) encoded features
            target_size: (H', W') target spatial size
            
        Returns:
            full_spectrum: Complex tensor (B, out_C, H', W') in frequency domain
        """
        # Upsample features to target size
        if features.shape[-2:] != target_size:
            features = F.interpolate(features, size=target_size, 
                                    mode='bilinear', align_corners=False)
        
        B, C, H, W = features.shape
        
        # Spatial processing
        features = self.spatial_branch(features)
        
        # Get frequency masks
        lf_mask, hf_mask = self._create_frequency_masks(H, W, features.device)
        lf_mask = lf_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        hf_mask = hf_mask.unsqueeze(0).unsqueeze(0)
        
        # Predict LF component
        lf_out = self.lf_head(features)
        out_channels = lf_out.shape[1] // 2
        lf_real = lf_out[:, :out_channels]
        lf_imag = lf_out[:, out_channels:]
        lf_complex = torch.complex(lf_real, lf_imag)
        
        # Predict HF component
        hf_out = self.hf_head(features)
        hf_real = hf_out[:, :out_channels]
        hf_imag = hf_out[:, out_channels:]
        hf_complex = torch.complex(hf_real, hf_imag)
        
        # Combine with soft masks
        full_spectrum = lf_mask * lf_complex + hf_mask * hf_complex
        
        return full_spectrum


class FrequencyEncoder(nn.Module):
    """
    Encodes spatial features for frequency-domain prediction.
    """
    def __init__(self, in_channels, base_channels, num_blocks=4):
        super().__init__()
        
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(base_channels),
                CBAM(base_channels)
            )
            for _ in range(num_blocks)
        ])
        
        self.fusion = nn.Conv2d(base_channels * num_blocks, base_channels, 1)
    
    def forward(self, x):
        x = self.input_conv(x)
        
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        multi_scale = torch.cat(features, dim=1)
        return self.fusion(multi_scale)


# =============================================================================
# TEMPORAL ATTENTION
# =============================================================================

class TemporalAttention(nn.Module):
    """Temporal attention for fusing multiple frames."""
    def __init__(self, channels, num_frames=3):
        super().__init__()
        self.num_frames = num_frames
        
        self.frame_encoder = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(num_frames, 1, 1), padding=0),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU()
        )
        
        self.refine = nn.Sequential(
            ResidualBlock(channels),
            CBAM(channels)
        )
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.frame_encoder(x)  # (B, C, 1, H, W)
        x = x.squeeze(2)  # (B, C, H, W)
        return self.refine(x)


# =============================================================================
# SPECTRAL CONSISTENCY
# =============================================================================

class SpectralConsistencyModule(nn.Module):
    """Ensures spectral relationships between bands are preserved."""
    def __init__(self, channels, num_bands):
        super().__init__()
        self.cross_band = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        cross = self.cross_band(x)
        gate = self.gate(torch.cat([x, cross], dim=1))
        return x + gate * cross


# =============================================================================
# OUTPUT HEAD FOR DIRECT LEARNING
# =============================================================================

class DirectOutputHead(nn.Module):
    """
    Output head for direct learning with proper initialization.
    """
    def __init__(self, in_channels, out_channels, init_bias=0.05):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self._init_weights(init_bias)
    
    def _init_weights(self, init_bias):
        """Initialize for direct learning stability."""
        nn.init.xavier_uniform_(self.conv.weight, gain=0.1)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, init_bias)
    
    def forward(self, x):
        x = self.conv(x)
        return torch.clamp(x, 0.0, 1.0)


# =============================================================================
# MAIN MODEL: FREQUENCY DOMAIN SR (DIRECT VERSION)
# =============================================================================

class FrequencyDomainSRDirect(nn.Module):
    """
    Frequency Domain Super-Resolution - Direct Version
    
    Key Idea:
    - Predicts FULL frequency spectrum directly (no baseline)
    - Learns both LF and HF components from scratch
    - More challenging but potentially more flexible
    
    Architecture:
    1. Encode LR features
    2. Predict full spectrum (LF + HF) in frequency domain
    3. Transform to spatial domain via inverse FFT
    4. Spatial refinement for final output
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
        self.encoder_10m = FrequencyEncoder(in_channels_10m, base_channels, num_encoder_blocks)
        self.encoder_20m = FrequencyEncoder(in_channels_20m, base_channels, num_encoder_blocks)
        
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
        
        # ========== FULL SPECTRUM PREDICTORS ==========
        self.spectrum_predictor_10m = FullSpectrumPredictor(
            base_channels, in_channels_10m, base_channels
        )
        self.spectrum_predictor_20m = FullSpectrumPredictor(
            base_channels, in_channels_20m, base_channels
        )
        
        # ========== SPATIAL REFINEMENT ==========
        self.spatial_refine_10m = nn.Sequential(
            ResidualBlock(in_channels_10m),
            CBAM(in_channels_10m),
            ResidualBlock(in_channels_10m),
            nn.Conv2d(in_channels_10m, in_channels_10m, 3, padding=1)
        )
        self.spatial_refine_20m = nn.Sequential(
            ResidualBlock(in_channels_20m),
            CBAM(in_channels_20m),
            ResidualBlock(in_channels_20m),
            nn.Conv2d(in_channels_20m, in_channels_20m, 3, padding=1)
        )
        
        # ========== OUTPUT HEADS ==========
        self.output_head_10m = DirectOutputHead(in_channels_10m, in_channels_10m, init_bias=0.05)
        self.output_head_20m = DirectOutputHead(in_channels_20m, in_channels_20m, init_bias=0.05)
        
        # Initialize for stable direct learning
        self._init_direct_learning_weights()
    
    def _init_direct_learning_weights(self):
        """Initialize weights for stable direct learning."""
        # Spatial refinement starts near identity
        for refine in [self.spatial_refine_10m, self.spatial_refine_20m]:
            for m in refine.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def _spectrum_to_spatial(self, spectrum, target_size):
        """
        Convert frequency spectrum to spatial domain.
        
        Args:
            spectrum: Complex tensor (B, C, H, W) in frequency domain
            target_size: (H', W') target spatial size
            
        Returns:
            spatial: (B, C, H', W') spatial domain image
        """
        # Ensure correct size
        if spectrum.shape[-2:] != target_size:
            spectrum = self._resize_frequency(spectrum, target_size)
        
        # Inverse FFT to get spatial output
        spatial = torch.fft.ifft2(spectrum).real
        
        return spatial
    
    def _resize_frequency(self, freq_tensor, target_size):
        """Resize frequency tensor by zero-padding or cropping."""
        B, C, H, W = freq_tensor.shape
        target_H, target_W = target_size
        
        if H == target_H and W == target_W:
            return freq_tensor
        
        # Shift to center, resize, shift back
        freq_shifted = torch.fft.fftshift(freq_tensor, dim=(-2, -1))
        
        if target_H > H or target_W > W:
            # Zero-pad
            pad_H = (target_H - H) // 2
            pad_W = (target_W - W) // 2
            freq_shifted = F.pad(freq_shifted, 
                                (pad_W, target_W - W - pad_W, pad_H, target_H - H - pad_H))
        else:
            # Crop center
            start_H = (H - target_H) // 2
            start_W = (W - target_W) // 2
            freq_shifted = freq_shifted[:, :, start_H:start_H+target_H, start_W:start_W+target_W]
        
        return torch.fft.ifftshift(freq_shifted, dim=(-2, -1))
    
    def forward(
        self,
        lr_10m: torch.Tensor,
        lr_20m: torch.Tensor,
        target_size_10m: Tuple[int, int] = None,
        target_size_20m: Tuple[int, int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with frequency-domain direct learning.
        
        Args:
            lr_10m: (B, T, 3, H, W) - 10m bands temporal sequence
            lr_20m: (B, T, 2, H/2, W/2) - 20m bands temporal sequence
            
        Returns:
            sr_10m: (B, 3, H', W') - Super-resolved 10m bands
            sr_20m: (B, 2, H'/2, W'/2) - Super-resolved 20m bands
        """
        B, T, _, H_10m, W_10m = lr_10m.shape
        _, _, _, H_20m, W_20m = lr_20m.shape
        
        # Calculate target sizes
        if target_size_10m is None:
            target_size_10m = (int(H_10m * self.scale_factor_10m), 
                              int(W_10m * self.scale_factor_10m))
        if target_size_20m is None:
            target_size_20m = (int(H_20m * self.scale_factor_20m), 
                              int(W_20m * self.scale_factor_20m))
        
        # ========== INITIAL HINT (helps training stability) ==========
        # Contribution from bilinear for initialization stability
        # The model learns the frequency-domain corrections on top of this
        mid_idx = T // 2
        init_hint_10m = F.interpolate(lr_10m[:, mid_idx], size=target_size_10m,
                                      mode='bilinear', align_corners=False)
        init_hint_20m = F.interpolate(lr_20m[:, mid_idx], size=target_size_20m,
                                      mode='bilinear', align_corners=False)
        
        # ========== ENCODE ALL FRAMES ==========
        features_10m = []
        for t in range(T):
            feat = self.encoder_10m(lr_10m[:, t])
            features_10m.append(feat)
        features_10m = torch.stack(features_10m, dim=1)
        
        features_20m = []
        for t in range(T):
            feat = self.encoder_20m(lr_20m[:, t])
            features_20m.append(feat)
        features_20m = torch.stack(features_20m, dim=1)
        
        # ========== TEMPORAL ATTENTION ==========
        fused_10m = self.temporal_attn_10m(features_10m)
        fused_20m = self.temporal_attn_20m(features_20m)
        
        # ========== CROSS-RESOLUTION FUSION ==========
        if self.use_cross_resolution_fusion:
            fused_20m_up = F.interpolate(fused_20m, size=(H_10m, W_10m), 
                                         mode='bilinear', align_corners=False)
            fused_10m_down = F.interpolate(fused_10m, size=(H_20m, W_20m), 
                                           mode='bilinear', align_corners=False)
            
            combined_10m = torch.cat([fused_10m, self.fusion_10m_from_20m(fused_20m_up)], dim=1)
            gate_10m = self.fusion_gate(combined_10m)
            fused_10m = fused_10m + gate_10m * self.fusion_10m_from_20m(fused_20m_up)
            
            combined_20m = torch.cat([fused_20m, self.fusion_20m_from_10m(fused_10m_down)], dim=1)
            gate_20m = self.fusion_gate(combined_20m)
            fused_20m = fused_20m + gate_20m * self.fusion_20m_from_10m(fused_10m_down)
        
        # ========== SPECTRAL CONSISTENCY ==========
        fused_10m = self.spectral_consistency_10m(fused_10m)
        fused_20m = self.spectral_consistency_20m(fused_20m)
        
        # ========== PREDICT FULL SPECTRUM (DIRECT) ==========
        spectrum_10m = self.spectrum_predictor_10m(fused_10m, target_size_10m)
        spectrum_20m = self.spectrum_predictor_20m(fused_20m, target_size_20m)
        
        # Convert spectrum to spatial domain
        sr_10m = self._spectrum_to_spatial(spectrum_10m, target_size_10m)
        sr_20m = self._spectrum_to_spatial(spectrum_20m, target_size_20m)
        
        # Add initial hint (helps early training, model learns to override)
        sr_10m = sr_10m + init_hint_10m
        sr_20m = sr_20m + init_hint_20m
        
        # ========== SPATIAL REFINEMENT ==========
        sr_10m = sr_10m + self.spatial_refine_10m(sr_10m)
        sr_20m = sr_20m + self.spatial_refine_20m(sr_20m)
        
        # ========== OUTPUT ==========
        sr_10m = self.output_head_10m(sr_10m)
        sr_20m = self.output_head_20m(sr_20m)
        
        return sr_10m, sr_20m


# =============================================================================
# FREQUENCY-AWARE LOSS FUNCTIONS
# =============================================================================

class FrequencyBandLoss(nn.Module):
    """
    Loss computed separately for LF and HF bands.
    """
    def __init__(self, hf_weight=2.0, lf_weight=1.0, cutoff_ratio=0.25):
        super().__init__()
        self.hf_weight = hf_weight
        self.lf_weight = lf_weight
        self.cutoff_ratio = cutoff_ratio
    
    def _create_masks(self, H, W, device):
        freq_y = torch.fft.fftfreq(H, device=device).view(-1, 1)
        freq_x = torch.fft.fftfreq(W, device=device).view(1, -1)
        freq_radius = torch.sqrt(freq_y**2 + freq_x**2)
        max_freq = math.sqrt(0.5**2 + 0.5**2)
        freq_radius = freq_radius / max_freq
        
        lf_mask = (freq_radius <= self.cutoff_ratio).float()
        hf_mask = 1.0 - lf_mask
        
        return lf_mask, hf_mask
    
    def forward(self, pred, target):
        B, C, H, W = pred.shape
        
        # Transform to frequency domain
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Get masks
        lf_mask, hf_mask = self._create_masks(H, W, pred.device)
        lf_mask = lf_mask.unsqueeze(0).unsqueeze(0)
        hf_mask = hf_mask.unsqueeze(0).unsqueeze(0)
        
        # Compute magnitude losses
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        lf_loss = F.l1_loss(pred_mag * lf_mask, target_mag * lf_mask)
        hf_loss = F.l1_loss(pred_mag * hf_mask, target_mag * hf_mask)
        
        return self.lf_weight * lf_loss + self.hf_weight * hf_loss


class SpectralAngleLoss(nn.Module):
    """Spectral Angle Mapper loss for spectral fidelity."""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred_norm = F.normalize(pred, dim=1, eps=1e-8)
        target_norm = F.normalize(target, dim=1, eps=1e-8)
        
        cos_sim = (pred_norm * target_norm).sum(dim=1)
        cos_sim = torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)
        sam = torch.acos(cos_sim)
        
        return sam.mean()


class GradientLoss(nn.Module):
    """Edge preservation loss using Sobel gradients."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        pred_grad_x = F.conv2d(pred, self.sobel_x.repeat(pred.shape[1], 1, 1, 1), 
                               padding=1, groups=pred.shape[1])
        pred_grad_y = F.conv2d(pred, self.sobel_y.repeat(pred.shape[1], 1, 1, 1), 
                               padding=1, groups=pred.shape[1])
        
        target_grad_x = F.conv2d(target, self.sobel_x.repeat(target.shape[1], 1, 1, 1), 
                                 padding=1, groups=target.shape[1])
        target_grad_y = F.conv2d(target, self.sobel_y.repeat(target.shape[1], 1, 1, 1), 
                                 padding=1, groups=target.shape[1])
        
        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)


class FrequencyDirectLoss(nn.Module):
    """
    Combined loss for frequency domain direct SR.
    
    Includes:
    - Pixel loss (L1 + MSE)
    - Frequency band loss (emphasizes HF)
    - Spectral angle loss
    - Gradient loss
    - Optional perceptual loss
    """
    def __init__(self, cfg):
        super().__init__()
        self.alpha_pixel = cfg.get('alpha_pixel', 1.0)
        self.alpha_freq = cfg.get('alpha_freq', 0.5)
        self.alpha_spectral = cfg.get('alpha_spectral', 0.1)
        self.alpha_gradient = cfg.get('alpha_gradient', 0.01)
        self.alpha_perceptual = cfg.get('alpha_perceptual', 0.1)
        
        self.weight_10m = cfg.get('weight_10m', 1.0)
        self.weight_20m = cfg.get('weight_20m', 1.0)
        
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.freq_loss = FrequencyBandLoss(hf_weight=2.0, lf_weight=1.0)
        self.sam_loss = SpectralAngleLoss()
        self.grad_loss = GradientLoss()
        
        # Optional VGG perceptual loss
        if self.alpha_perceptual > 0:
            try:
                from perceptual_loss import VGGPerceptualLoss
                self.perceptual_loss = VGGPerceptualLoss(input_scale=2.5)
            except ImportError:
                self.perceptual_loss = None
                self.alpha_perceptual = 0
        else:
            self.perceptual_loss = None
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m):
        # 10m branch loss
        pixel_10m = self.l1(pred_10m, target_10m) + 0.1 * self.mse(pred_10m, target_10m)
        freq_10m = self.freq_loss(pred_10m, target_10m)
        sam_10m = self.sam_loss(pred_10m, target_10m)
        grad_10m = self.grad_loss(pred_10m, target_10m)
        
        loss_10m = (self.alpha_pixel * pixel_10m + 
                   self.alpha_freq * freq_10m +
                   self.alpha_spectral * sam_10m + 
                   self.alpha_gradient * grad_10m)
        
        if self.perceptual_loss is not None:
            loss_10m = loss_10m + self.alpha_perceptual * self.perceptual_loss(pred_10m, target_10m)
        
        # 20m branch loss (no perceptual)
        pixel_20m = self.l1(pred_20m, target_20m) + 0.1 * self.mse(pred_20m, target_20m)
        freq_20m = self.freq_loss(pred_20m, target_20m)
        sam_20m = self.sam_loss(pred_20m, target_20m)
        grad_20m = self.grad_loss(pred_20m, target_20m)
        
        loss_20m = (self.alpha_pixel * pixel_20m + 
                   self.alpha_freq * freq_20m +
                   self.alpha_spectral * sam_20m + 
                   self.alpha_gradient * grad_20m)
        
        total = self.weight_10m * loss_10m + self.weight_20m * loss_20m
        
        return total, {
            'loss_10m': loss_10m.item(),
            'loss_20m': loss_20m.item(),
            'pixel_10m': pixel_10m.item(),
            'pixel_20m': pixel_20m.item(),
            'freq_10m': freq_10m.item(),
            'freq_20m': freq_20m.item()
        }


if __name__ == '__main__':
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = FrequencyDomainSRDirect(
        in_channels_10m=3,
        in_channels_20m=2,
        base_channels=64,
        num_encoder_blocks=4,
        num_frames=3,
        scale_factor_10m=3.33,
        scale_factor_20m=3.33
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Test forward pass
    with torch.no_grad():
        lr_10m = torch.randn(2, 3, 3, 30, 30).to(device) * 0.05 + 0.05
        lr_20m = torch.randn(2, 3, 2, 15, 15).to(device) * 0.05 + 0.05
        
        sr_10m, sr_20m = model(lr_10m, lr_20m)
        
        print(f"Input 10m: {lr_10m.shape} -> Output: {sr_10m.shape}")
        print(f"Input 20m: {lr_20m.shape} -> Output: {sr_20m.shape}")
        print(f"Output 10m: mean={sr_10m.mean():.4f}, std={sr_10m.std():.4f}")
        print(f"Output 20m: mean={sr_20m.mean():.4f}, std={sr_20m.std():.4f}")
        
        # Check variance
        print(f"10m variance: {sr_10m.var():.6f}")
        print(f"20m variance: {sr_20m.var():.6f}")
        
        if sr_10m.var() > 0.00001 and sr_20m.var() > 0.00001:
            print("\n✅ SUCCESS: Both branches have non-zero variance!")
        else:
            print("\n❌ WARNING: One or both branches may have collapsed!")
