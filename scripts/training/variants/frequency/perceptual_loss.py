"""
Perceptual Loss Module for Satellite Super-Resolution

Contains:
- VGGPerceptualLoss: VGG-based feature matching loss for perceptual quality
- SpectralAngleLoss: Numerically stable SAM loss for spectral fidelity

Notes:
- VGG expects 3-channel RGB input normalized to ImageNet stats
- For satellite bands, we adapt by:
  1. Using only 10m bands (3 channels) for perceptual loss
  2. Or using first 3 channels of any input
- SAM loss works on any number of channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional


class VGGPerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss using feature matching.
    
    Extracts features from intermediate VGG layers and computes L1 distance
    between predicted and target features.
    
    Args:
        feature_layers: Which VGG layers to use (default: relu1_2, relu2_2, relu3_3)
        weights: Weights for each layer (default: equal)
        use_input_norm: Whether to normalize inputs to ImageNet stats
        resize_input: Resize input to this size before VGG (None = no resize)
        input_scale: Scale factor to apply to input before normalization.
                     Useful for satellite data with low dynamic range (mean ~0.05)
                     to shift it closer to ImageNet distribution (mean ~0.45).
                     Recommended: 2.0-3.0 for satellite imagery.
    """
    
    def __init__(
        self,
        feature_layers: List[int] = [4, 9, 16],  # relu1_2, relu2_2, relu3_3
        weights: Optional[List[float]] = None,
        use_input_norm: bool = True,
        resize_input: Optional[int] = None,
        input_scale: float = 1.0
    ):
        super().__init__()
        
        self.feature_layers = feature_layers
        self.weights = weights if weights else [1.0] * len(feature_layers)
        self.use_input_norm = use_input_norm
        self.resize_input = resize_input
        self.input_scale = input_scale
        
        # Load pretrained VGG16 (only need features up to max layer)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        max_layer = max(feature_layers) + 1
        self.vgg_layers = nn.Sequential(*list(vgg.features.children())[:max_layer])
        
        # Freeze VGG weights
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
        # ImageNet normalization constants
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to ImageNet stats."""
        return (x - self.mean) / self.std
    
    def _adapt_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapt input to 3 channels for VGG.
        - If 3 channels: use as-is
        - If >3 channels: use first 3
        - If <3 channels: repeat to 3
        """
        c = x.shape[1]
        if c == 3:
            return x
        elif c > 3:
            return x[:, :3, :, :]
        else:
            # Repeat channels to get 3
            repeats = (3 + c - 1) // c
            return x.repeat(1, repeats, 1, 1)[:, :3, :, :]
    
    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from specified VGG layers."""
        features = []
        current = x
        
        for i, layer in enumerate(self.vgg_layers):
            current = layer(current)
            if i in self.feature_layers:
                features.append(current)
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between prediction and target.
        
        Args:
            pred: (B, C, H, W) predicted image [0, 1]
            target: (B, C, H, W) target image [0, 1]
        
        Returns:
            Scalar perceptual loss
        """
        # Adapt to 3 channels
        pred_3ch = self._adapt_channels(pred)
        target_3ch = self._adapt_channels(target)
        
        # Apply input scaling (for satellite data with low dynamic range)
        # This shifts data closer to ImageNet distribution before normalization
        if self.input_scale != 1.0:
            pred_3ch = torch.clamp(pred_3ch * self.input_scale, 0.0, 1.0)
            target_3ch = torch.clamp(target_3ch * self.input_scale, 0.0, 1.0)
        
        # Optional resize
        if self.resize_input is not None:
            pred_3ch = F.interpolate(pred_3ch, size=self.resize_input, mode='bilinear', align_corners=False)
            target_3ch = F.interpolate(target_3ch, size=self.resize_input, mode='bilinear', align_corners=False)
        
        # Normalize to ImageNet stats
        if self.use_input_norm:
            pred_3ch = self._normalize(pred_3ch)
            target_3ch = self._normalize(target_3ch)
        
        # Extract features
        pred_features = self._extract_features(pred_3ch)
        target_features = self._extract_features(target_3ch)
        
        # Compute weighted L1 loss across layers
        loss = 0.0
        for pred_feat, target_feat, weight in zip(pred_features, target_features, self.weights):
            loss += weight * F.l1_loss(pred_feat, target_feat)
        
        return loss


class SpectralAngleLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss.
    
    Measures spectral similarity by computing the angle between spectral vectors.
    More robust than pixel-wise losses for preserving spectral characteristics.
    
    Returns: 1 - mean(cos_similarity), so lower is better (range [0, 2]).
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SAM loss.
        
        Args:
            pred: (B, C, H, W) predicted
            target: (B, C, H, W) target
        
        Returns:
            Scalar SAM loss (1 - mean cosine similarity)
        """
        # Normalize along channel dimension
        pred_norm = F.normalize(pred, dim=1, eps=self.eps)
        target_norm = F.normalize(target, dim=1, eps=self.eps)
        
        # Cosine similarity per pixel: dot product of normalized vectors
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)  # (B, H, W)
        
        # Clamp to avoid numerical issues with acos
        cos_sim = torch.clamp(cos_sim, -1.0 + self.eps, 1.0 - self.eps)
        
        # SAM loss: 1 - mean cosine similarity
        # Alternative: mean(acos(cos_sim)) for actual angle in radians
        return 1.0 - cos_sim.mean()


class CombinedPerceptualLoss(nn.Module):
    """
    Combined loss with pixel, spectral (SAM), gradient, and perceptual terms.
    
    Args:
        alpha_pixel: Weight for L1 pixel loss (default: 1.0)
        alpha_sam: Weight for SAM spectral loss (default: 0.1)
        alpha_gradient: Weight for gradient/edge loss (default: 0.01)
        alpha_perceptual: Weight for VGG perceptual loss (default: 0.1)
        use_perceptual: Whether to use VGG perceptual loss (default: True)
    """
    
    def __init__(
        self,
        alpha_pixel: float = 1.0,
        alpha_sam: float = 0.1,
        alpha_gradient: float = 0.01,
        alpha_perceptual: float = 0.1,
        use_perceptual: bool = True
    ):
        super().__init__()
        
        self.alpha_pixel = alpha_pixel
        self.alpha_sam = alpha_sam
        self.alpha_gradient = alpha_gradient
        self.alpha_perceptual = alpha_perceptual
        self.use_perceptual = use_perceptual
        
        # Component losses
        self.l1_loss = nn.L1Loss()
        self.sam_loss = SpectralAngleLoss()
        
        if use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss()
        
        # Sobel filters for gradient loss
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient/edge loss using Sobel filters."""
        loss = 0.0
        for c in range(pred.shape[1]):
            pred_gx = F.conv2d(pred[:, c:c+1], self.sobel_x, padding=1)
            pred_gy = F.conv2d(pred[:, c:c+1], self.sobel_y, padding=1)
            target_gx = F.conv2d(target[:, c:c+1], self.sobel_x, padding=1)
            target_gy = F.conv2d(target[:, c:c+1], self.sobel_y, padding=1)
            loss += self.l1_loss(pred_gx, target_gx) + self.l1_loss(pred_gy, target_gy)
        return loss / pred.shape[1]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: (B, C, H, W) predicted
            target: (B, C, H, W) target
        
        Returns:
            Scalar combined loss
        """
        # Pixel loss (L1)
        loss = self.alpha_pixel * self.l1_loss(pred, target)
        
        # SAM loss
        if self.alpha_sam > 0:
            loss = loss + self.alpha_sam * self.sam_loss(pred, target)
        
        # Gradient loss
        if self.alpha_gradient > 0:
            loss = loss + self.alpha_gradient * self.gradient_loss(pred, target)
        
        # Perceptual loss (VGG)
        if self.use_perceptual and self.alpha_perceptual > 0:
            loss = loss + self.alpha_perceptual * self.perceptual_loss(pred, target)
        
        return loss
