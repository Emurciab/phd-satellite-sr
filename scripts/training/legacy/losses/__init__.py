# Losses module for satellite super-resolution training
from .perceptual_loss import (
    VGGPerceptualLoss,
    SpectralAngleLoss,
    CombinedPerceptualLoss
)

__all__ = [
    'VGGPerceptualLoss',
    'SpectralAngleLoss', 
    'CombinedPerceptualLoss'
]
