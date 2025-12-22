#!/usr/bin/env python3
"""
Discriminator Networks for GAN-based Super Resolution.

This module defines PatchGAN discriminators for the adversarial training
of satellite image super-resolution models. The discriminator acts as a
"sharpness regularizer" to encourage perceptually realistic outputs.

Key Design Choices:
- PatchGAN architecture: Classifies NxN patches instead of full images
  - More parameter efficient
  - Works well with varying image sizes
  - Captures local texture/sharpness better
  
- Spectral Normalization: Stabilizes GAN training
- LeakyReLU: Standard for discriminators (negative slope allows gradients)
- No BatchNorm in discriminator: More stable with spectral norm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for satellite image super-resolution.
    
    Classifies 70x70 overlapping patches as real or fake.
    Uses spectral normalization for stable training.
    
    Input: (B, C, H, W) image tensor
    Output: (B, 1, H', W') patch-wise predictions
    
    Architecture:
        Conv(C->64) -> LReLU ->
        Conv(64->128, stride=2) -> LReLU ->
        Conv(128->256, stride=2) -> LReLU ->
        Conv(256->512, stride=1) -> LReLU ->
        Conv(512->1, stride=1) -> output
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64, 
                 num_layers: int = 3, use_spectral_norm: bool = True):
        """
        Args:
            in_channels: Number of input channels (3 for 10m RGB, 2 for 20m)
            base_channels: Base channel count (doubled at each layer)
            num_layers: Number of strided conv layers (default 3 -> 70x70 receptive field)
            use_spectral_norm: Whether to apply spectral normalization
        """
        super().__init__()
        
        self.in_channels = in_channels
        norm_layer = spectral_norm if use_spectral_norm else (lambda x: x)
        
        # First layer: no stride, no normalization
        sequence = [
            norm_layer(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Middle layers: strided convolutions
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 8x base channels
            sequence += [
                norm_layer(nn.Conv2d(
                    base_channels * nf_mult_prev, 
                    base_channels * nf_mult,
                    kernel_size=4, stride=2, padding=1
                )),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Second to last layer: stride 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        sequence += [
            norm_layer(nn.Conv2d(
                base_channels * nf_mult_prev,
                base_channels * nf_mult,
                kernel_size=4, stride=1, padding=1
            )),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Final layer: output single channel (real/fake score)
        sequence += [
            norm_layer(nn.Conv2d(base_channels * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        ]
        
        self.model = nn.Sequential(*sequence)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights with small values for stable training."""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Patch-wise real/fake scores (B, 1, H', W')
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN Discriminator.
    
    Operates at multiple scales to capture both fine textures and 
    global structure. Useful for satellite imagery where details 
    matter at different scales.
    
    Uses 3 discriminators at scales: 1x, 0.5x, 0.25x
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_layers: int = 3, num_scales: int = 3):
        """
        Args:
            in_channels: Number of input channels
            base_channels: Base channel count for each discriminator
            num_layers: Number of layers per discriminator
            num_scales: Number of scales (default 3)
        """
        super().__init__()
        
        self.num_scales = num_scales
        
        # Create discriminators for each scale
        self.discriminators = nn.ModuleList()
        for _ in range(num_scales):
            self.discriminators.append(
                PatchDiscriminator(in_channels, base_channels, num_layers)
            )
        
        # Downsampling for multi-scale input
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass at multiple scales.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            List of outputs from each scale discriminator
        """
        outputs = []
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            if i < self.num_scales - 1:
                x = self.downsample(x)
        return outputs


class DualBranchDiscriminator(nn.Module):
    """
    Discriminator for multi-resolution SR (10m and 20m branches).
    
    Contains two separate discriminators, one for each resolution branch.
    This allows the GAN to optimize both branches simultaneously.
    """
    
    def __init__(self, channels_10m: int = 3, channels_20m: int = 2,
                 base_channels: int = 64, num_layers: int = 3,
                 use_multiscale: bool = False):
        """
        Args:
            channels_10m: Channels for 10m branch (default 3: RGB)
            channels_20m: Channels for 20m branch (default 2: RE, SWIR)
            base_channels: Base channel count
            num_layers: Number of conv layers
            use_multiscale: Whether to use multi-scale discriminators
        """
        super().__init__()
        
        if use_multiscale:
            self.disc_10m = MultiScaleDiscriminator(channels_10m, base_channels, num_layers)
            self.disc_20m = MultiScaleDiscriminator(channels_20m, base_channels, num_layers)
        else:
            self.disc_10m = PatchDiscriminator(channels_10m, base_channels, num_layers)
            self.disc_20m = PatchDiscriminator(channels_20m, base_channels, num_layers)
        
        self.use_multiscale = use_multiscale
    
    def forward(self, x_10m: torch.Tensor, x_20m: torch.Tensor):
        """
        Forward pass for both branches.
        
        Args:
            x_10m: 10m resolution input (B, 3, H, W)
            x_20m: 20m resolution input (B, 2, H, W)
            
        Returns:
            Tuple of (disc_10m_output, disc_20m_output)
        """
        out_10m = self.disc_10m(x_10m)
        out_20m = self.disc_20m(x_20m)
        return out_10m, out_20m


# =============================================================================
# GAN Loss Functions
# =============================================================================

class GANLoss(nn.Module):
    """
    GAN loss with support for different loss types.
    
    Supports:
    - 'vanilla': Original GAN loss (BCE)
    - 'lsgan': Least Squares GAN (MSE) - more stable, less mode collapse
    - 'wgan': Wasserstein GAN (raw scores) - requires gradient penalty
    - 'hinge': Hinge loss - good for high-resolution generation
    """
    
    def __init__(self, gan_mode: str = 'lsgan', target_real_label: float = 1.0,
                 target_fake_label: float = 0.0):
        """
        Args:
            gan_mode: Type of GAN loss ('vanilla', 'lsgan', 'wgan', 'hinge')
            target_real_label: Label for real images
            target_fake_label: Label for fake images
        """
        super().__init__()
        
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode in ['wgan', 'hinge']:
            self.loss = None  # Computed directly
        else:
            raise ValueError(f"Unknown GAN mode: {gan_mode}")
    
    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create target tensor with same shape as prediction."""
        target_label = self.real_label if target_is_real else self.fake_label
        return target_label.expand_as(prediction)
    
    def forward(self, prediction: torch.Tensor, target_is_real: bool, 
                for_discriminator: bool = True) -> torch.Tensor:
        """
        Calculate GAN loss.
        
        Args:
            prediction: Discriminator output
            target_is_real: Whether target is real or fake
            for_discriminator: Whether computing loss for D or G
            
        Returns:
            Computed loss
        """
        if self.gan_mode in ['vanilla', 'lsgan']:
            target = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target)
            
        elif self.gan_mode == 'wgan':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
                
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    return F.relu(1.0 - prediction).mean()
                else:
                    return F.relu(1.0 + prediction).mean()
            else:
                # Generator loss
                return -prediction.mean()


def compute_gradient_penalty(discriminator: nn.Module, real: torch.Tensor, 
                            fake: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    Args:
        discriminator: Discriminator network
        real: Real images
        fake: Fake/generated images
        device: Computation device
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real)
    
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    
    disc_interpolated = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    # Test discriminators
    print("Testing Discriminator Networks...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test PatchDiscriminator
    disc = PatchDiscriminator(in_channels=3, base_channels=64).to(device)
    x = torch.randn(2, 3, 256, 256, device=device)
    out = disc(x)
    print(f"PatchDiscriminator: input {x.shape} -> output {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in disc.parameters()):,}")
    
    # Test MultiScaleDiscriminator
    ms_disc = MultiScaleDiscriminator(in_channels=3, base_channels=64).to(device)
    outputs = ms_disc(x)
    print(f"MultiScaleDiscriminator: input {x.shape} -> outputs {[o.shape for o in outputs]}")
    print(f"  Parameters: {sum(p.numel() for p in ms_disc.parameters()):,}")
    
    # Test DualBranchDiscriminator
    dual_disc = DualBranchDiscriminator(channels_10m=3, channels_20m=2).to(device)
    x_10m = torch.randn(2, 3, 256, 256, device=device)
    x_20m = torch.randn(2, 2, 128, 128, device=device)
    out_10m, out_20m = dual_disc(x_10m, x_20m)
    print(f"DualBranchDiscriminator: 10m {x_10m.shape} -> {out_10m.shape}, 20m {x_20m.shape} -> {out_20m.shape}")
    print(f"  Parameters: {sum(p.numel() for p in dual_disc.parameters()):,}")
    
    # Test GAN losses
    print("\nTesting GAN Losses...")
    for mode in ['vanilla', 'lsgan', 'hinge']:
        gan_loss = GANLoss(gan_mode=mode).to(device)
        pred = torch.randn(2, 1, 16, 16, device=device)
        loss_real = gan_loss(pred, target_is_real=True)
        loss_fake = gan_loss(pred, target_is_real=False)
        print(f"  {mode}: real_loss={loss_real.item():.4f}, fake_loss={loss_fake.item():.4f}")
    
    print("\n✓ All discriminator tests passed!")
