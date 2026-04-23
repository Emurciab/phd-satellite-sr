#!/usr/bin/env python3
"""
Heteroscedastic Deep Ensemble Training — Direct LearnedUps Variant
==================================================================

Implements a heteroscedastic deep ensemble for uncertainty quantification:
- Same architecture as MultiOutputTemporalSRDirectLearnedUps, but with
  additional variance heads that predict per-pixel log-variance.
- Trained with Gaussian negative log-likelihood (NLL) loss so each member
  learns BOTH the predictive mean AND the aleatoric (data) noise.
- Multiple members (typically M=5) are trained from different random seeds
  to capture epistemic (model) uncertainty through ensemble disagreement.

Decomposition at inference:
    Aleatoric  = (1/M) Σ_m σ²_m(x)          (mean predicted variance)
    Epistemic  = (1/M) Σ_m (μ_m(x) - μ̄(x))²  (variance of means)
    Predictive = Aleatoric + Epistemic

Usage:
    # Train member 0 with seed 1000
    python train_het_ensemble_direct_learnedUps.py --seed 1000 --ensemble_id 0

    # Train member 1 with seed 2000
    python train_het_ensemble_direct_learnedUps.py --seed 2000 --ensemble_id 1

Reference:
    Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty
    Estimation using Deep Ensembles", NeurIPS 2017.
"""

import sys
import os
import argparse
import json
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ---------------------------------------------------------------------------
# deterministic settings (overridden per-seed later, but set defaults now)
# ---------------------------------------------------------------------------
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------------------
# imports from the parent package (unchanged building-block files)
# ---------------------------------------------------------------------------
PARENT_DIR = str(Path(__file__).resolve().parent.parent)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from satellite_sr_model_v2 import (
    EncoderBranch,
    SpectralConsistencyModule,
    SimpleOutputHead,
    SpectralAngleLoss,
    CBAM,
    ResidualBlock,
)
from satellite_sr_model_v2_learned_upscale import (
    ProgressiveUpscalerLearned,
    MetaUpscale,
    AdaptiveInterpolation,
)


# =============================================================================
# CONFIG — same hyper-params as the original winning Direct checkpoint
# =============================================================================
BEST_CONFIG = {
    'base_channels': 64,
    'num_encoder_blocks': 4,
    'lr': 3.064e-05,
    'weight_decay': 5.235e-06,
    'batch_size': 2,
    'accumulation_steps': 2,
    'epochs': 200,
    'patience': 20,
    'patience_scheduler': 7,
    'alpha_pixel': 1.0,
    'alpha_spectral': 0.137,
    'alpha_gradient': 0.01,
    'alpha_temporal': 0.5,
    'weight_10m': 1.0,
    'weight_20m': 1.0,
    'num_frames': 3,
    'num_temporal_blocks': 2,
    # NLL-specific
    'nll_warmup_epochs': 5,           # epochs before full NLL kicks in
    'logvar_init': -3.0,              # initial log-variance (≈ σ² = 0.05)
    'logvar_clamp': (-10.0, 10.0),    # safety clamp for numerical stability
}


# =============================================================================
# SEED UTILITY
# =============================================================================

def set_all_seeds(seed: int):
    """Set all random seeds for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SEED] All seeds set to {seed}")


# =============================================================================
# TEMPORAL MODULES  (identical to original — copied here so the file is
#                    self-contained and the original is never touched)
# =============================================================================

class TemporalCrossAttention(nn.Module):
    """Memory-efficient cross-attention between temporal frames."""

    def __init__(self, channels, num_frames=3, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.num_heads = num_heads

        self.global_q = nn.Linear(channels, channels)
        self.global_k = nn.Linear(channels, channels)
        self.global_v = nn.Linear(channels, channels)

        self.local_mix = nn.Conv2d(channels * num_frames, channels, 1)

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x):
        B, T, C, H, W = x.shape

        x_pooled = x.mean(dim=(-2, -1))

        q = self.global_q(x_pooled)
        k = self.global_k(x_pooled)
        v = self.global_v(x_pooled)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        global_out = torch.matmul(attn, v)

        x_cat = x.view(B, T * C, H, W)
        local_mix = self.local_mix(x_cat)

        outputs = []
        for t in range(T):
            frame_feat = x[:, t]
            global_feat = global_out[:, t, :, None, None].expand(-1, -1, H, W)

            combined = torch.cat([frame_feat, local_mix], dim=1)
            gate = self.gate(combined)

            out = frame_feat + gate * (local_mix - frame_feat) + 0.1 * global_feat
            outputs.append(out)

        return torch.stack(outputs, dim=1)


class TemporalFusionBlock(nn.Module):
    """Block that processes temporal features with cross-attention."""

    def __init__(self, channels, num_frames=3):
        super().__init__()

        self.cross_attn = TemporalCrossAttention(channels, num_frames)

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels)
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, T, C, H, W = x.shape

        attended = self.cross_attn(x)

        outputs = []
        for t in range(T):
            refined = self.refine(attended[:, t])
            outputs.append(attended[:, t] + self.gamma * refined)

        return torch.stack(outputs, dim=1)


# =============================================================================
# VARIANCE HEAD
# =============================================================================

class VarianceHead(nn.Module):
    """
    Predicts per-pixel log-variance from the same decoder features.

    Architecture mirrors SimpleOutputHead but outputs *unclamped* log-var
    (clamped only for numerical safety, not to [0,1]).
    """

    def __init__(self, in_channels, out_channels, logvar_init=-3.0,
                 logvar_clamp=(-10.0, 10.0)):
        super().__init__()
        self.logvar_clamp = logvar_clamp
        intermediate = max(in_channels // 2, out_channels, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, intermediate, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(intermediate, out_channels, 3, padding=1),
        )
        # Initialise the final bias so that the initial predicted σ² is
        # reasonable (≈ exp(logvar_init)), avoiding giant initial losses.
        nn.init.constant_(self.conv[-1].bias, logvar_init)

    def forward(self, x):
        logvar = self.conv(x)
        return torch.clamp(logvar, *self.logvar_clamp)


# =============================================================================
# HETEROSCEDASTIC MODEL — Direct + LearnedUps
# =============================================================================

class HetMultiOutputTemporalSRDirectLearnedUps(nn.Module):
    """
    Heteroscedastic variant of MultiOutputTemporalSRDirectLearnedUps.

    Identical encoder/decoder/temporal/fusion/upscaler architecture, but each
    resolution output has TWO heads:
        • mean head  → μ(x)      — same SimpleOutputHead as original
        • var  head  → log σ²(x) — new VarianceHead

    forward() returns (mean_10m, logvar_10m, mean_20m, logvar_20m).
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
        resize_method: str = 'adaptive',
        logvar_init: float = -3.0,
        logvar_clamp: tuple = (-10.0, 10.0),
    ):
        super().__init__()
        self.num_frames = num_frames
        self.scale_factor_10m = scale_factor_10m
        self.scale_factor_20m = scale_factor_20m
        self.in_channels_10m = in_channels_10m
        self.in_channels_20m = in_channels_20m
        self.base_channels = base_channels
        self.resize_method = resize_method

        # ========== ENCODERS ==========
        self.encoder_10m = EncoderBranch(
            in_channels=in_channels_10m,
            base_channels=base_channels,
            num_blocks=num_encoder_blocks,
        )
        self.encoder_20m = EncoderBranch(
            in_channels=in_channels_20m,
            base_channels=base_channels,
            num_blocks=num_encoder_blocks,
        )

        # ========== TEMPORAL CROSS-ATTENTION ==========
        self.temporal_blocks_10m = nn.ModuleList([
            TemporalFusionBlock(base_channels, num_frames)
            for _ in range(num_temporal_blocks)
        ])
        self.temporal_blocks_20m = nn.ModuleList([
            TemporalFusionBlock(base_channels, num_frames)
            for _ in range(num_temporal_blocks)
        ])

        # ========== CROSS-RESOLUTION FUSION (gated) ==========
        self.fusion_10m_from_20m = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1),
            nn.GELU(),
        )
        self.fusion_20m_from_10m = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 1),
            nn.Sigmoid(),
        )

        # ========== SPECTRAL CONSISTENCY ==========
        self.spectral_10m = SpectralConsistencyModule(base_channels, in_channels_10m)
        self.spectral_20m = SpectralConsistencyModule(base_channels, in_channels_20m)

        # ========== DECODERS ==========
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

        # ========== UPSCALERS WITH LEARNED RESIZE ==========
        self.upscaler_10m = ProgressiveUpscalerLearned(
            base_channels, in_channels_10m, scale_factor_10m, base_channels,
            resize_method=resize_method,
        )
        self.upscaler_20m = ProgressiveUpscalerLearned(
            base_channels, in_channels_20m, scale_factor_20m, base_channels,
            resize_method=resize_method,
        )

        # ========== OUTPUT REFINEMENT (in output-channel space) ==========
        self.residual_refine_10m = nn.Sequential(
            nn.Conv2d(in_channels_10m, in_channels_10m, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels_10m, in_channels_10m, 3, padding=1),
        )
        self.residual_refine_20m = nn.Sequential(
            nn.Conv2d(in_channels_20m, in_channels_20m, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels_20m, in_channels_20m, 3, padding=1),
        )

        # ========== MEAN HEADS (same as original) ==========
        self.output_head_10m = SimpleOutputHead(in_channels_10m, in_channels_10m)
        self.output_head_20m = SimpleOutputHead(in_channels_20m, in_channels_20m)

        # ========== VARIANCE HEADS (NEW — heteroscedastic) ==========
        self.var_head_10m = VarianceHead(
            in_channels_10m, in_channels_10m,
            logvar_init=logvar_init, logvar_clamp=logvar_clamp,
        )
        self.var_head_20m = VarianceHead(
            in_channels_20m, in_channels_20m,
            logvar_init=logvar_init, logvar_clamp=logvar_clamp,
        )

    # --------------------------------------------------------------------- #
    def forward(self, lr_10m, lr_20m, target_size_10m=None, target_size_20m=None):
        """
        Forward pass — identical to original, but returns variance too.

        Args:
            lr_10m: (B, T, C_10m, H, W)
            lr_20m: (B, T, C_20m, H/2, W/2)
            target_size_10m: (H', W') exact output size for 10 m
            target_size_20m: (H', W') exact output size for 20 m

        Returns:
            mean_10m  : (B, T, C_10m, H', W')
            logvar_10m: (B, T, C_10m, H', W')
            mean_20m  : (B, T, C_20m, H'', W'')
            logvar_20m: (B, T, C_20m, H'', W'')
        """
        B, T, C_10m, H_10m, W_10m = lr_10m.shape
        _, _, C_20m, H_20m, W_20m = lr_20m.shape

        # Compute target sizes
        if target_size_10m is not None:
            tgt_size_10m = target_size_10m
        else:
            tgt_size_10m = (int(H_10m * self.scale_factor_10m),
                            int(W_10m * self.scale_factor_10m))
        if target_size_20m is not None:
            tgt_size_20m = target_size_20m
        else:
            tgt_size_20m = (int(H_20m * self.scale_factor_20m),
                            int(W_20m * self.scale_factor_20m))

        # ========== ENCODE ALL FRAMES ==========
        features_10m = []
        features_20m = []
        for t in range(T):
            features_10m.append(self.encoder_10m(lr_10m[:, t]))
            features_20m.append(self.encoder_20m(lr_20m[:, t]))

        features_10m = torch.stack(features_10m, dim=1)  # (B,T,C,H,W)
        features_20m = torch.stack(features_20m, dim=1)

        # ========== TEMPORAL CROSS-ATTENTION ==========
        for block in self.temporal_blocks_10m:
            features_10m = block(features_10m)
        for block in self.temporal_blocks_20m:
            features_20m = block(features_20m)

        # ========== CROSS-RESOLUTION FUSION (gated) ==========
        fused_10m_list = []
        fused_20m_list = []

        for t in range(T):
            feat_10m = features_10m[:, t]
            feat_20m = features_20m[:, t]

            feat_20m_up = F.interpolate(
                feat_20m, size=(H_10m, W_10m),
                mode='bilinear', align_corners=False,
            )
            feat_10m_down = F.interpolate(
                feat_10m, size=(H_20m, W_20m),
                mode='bilinear', align_corners=False,
            )

            combined_10m = torch.cat(
                [feat_10m, self.fusion_10m_from_20m(feat_20m_up)], dim=1
            )
            gate_10m = self.fusion_gate(combined_10m)
            fused_10m = feat_10m + gate_10m * self.fusion_10m_from_20m(feat_20m_up)

            combined_20m = torch.cat(
                [feat_20m, self.fusion_20m_from_10m(feat_10m_down)], dim=1
            )
            gate_20m = self.fusion_gate(combined_20m)
            fused_20m = feat_20m + gate_20m * self.fusion_20m_from_10m(feat_10m_down)

            fused_10m_list.append(fused_10m)
            fused_20m_list.append(fused_20m)

        # ========== DECODE, UPSCALE, REFINE, OUTPUT ==========
        means_10m, logvars_10m = [], []
        means_20m, logvars_20m = [], []

        for t in range(T):
            # spectral consistency
            feat_10m = self.spectral_10m(fused_10m_list[t])
            feat_20m = self.spectral_20m(fused_20m_list[t])

            # decode
            feat_10m = self.decoder_10m(feat_10m)
            feat_20m = self.decoder_20m(feat_20m)

            # upscale with LEARNED resize
            sr_10m = self.upscaler_10m(feat_10m, tgt_size_10m)
            sr_20m = self.upscaler_20m(feat_20m, tgt_size_20m)

            # refine (in output-channel space)
            ref_10m = self.residual_refine_10m(sr_10m)
            ref_20m = self.residual_refine_20m(sr_20m)

            # mean heads
            means_10m.append(self.output_head_10m(ref_10m))
            means_20m.append(self.output_head_20m(ref_20m))

            # variance heads (operate on same refined features)
            logvars_10m.append(self.var_head_10m(ref_10m))
            logvars_20m.append(self.var_head_20m(ref_20m))

        mean_10m   = torch.stack(means_10m, dim=1)      # (B,T,C,H',W')
        logvar_10m = torch.stack(logvars_10m, dim=1)
        mean_20m   = torch.stack(means_20m, dim=1)
        logvar_20m = torch.stack(logvars_20m, dim=1)

        return mean_10m, logvar_10m, mean_20m, logvar_20m


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class TemporalConsistencyLoss(nn.Module):
    """Penalises temporal jitter between consecutive frames."""

    def __init__(self, mode='l1'):
        super().__init__()
        self.mode = mode

    def forward(self, pred, target):
        B, T = pred.shape[:2]
        if T < 2:
            return torch.tensor(0.0, device=pred.device)

        total_loss, num_pairs = 0.0, 0
        for t in range(T - 1):
            pred_diff   = pred[:, t + 1] - pred[:, t]
            target_diff = target[:, t + 1] - target[:, t]
            if self.mode == 'l1':
                total_loss += F.l1_loss(pred_diff, target_diff)
            else:
                total_loss += F.mse_loss(pred_diff, target_diff)
            num_pairs += 1
        return total_loss / num_pairs


class LocalGradientLoss(nn.Module):
    """Edge-preservation loss via Sobel gradients."""

    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )

    def forward(self, pred, target):
        device = pred.device
        sx = self.sobel_x.view(1, 1, 3, 3).to(device)
        sy = self.sobel_y.view(1, 1, 3, 3).to(device)

        loss = 0
        for c in range(pred.shape[1]):
            pc, tc = pred[:, c:c + 1], target[:, c:c + 1]
            loss += (F.l1_loss(F.conv2d(pc, sx, padding=1),
                               F.conv2d(tc, sx, padding=1))
                     + F.l1_loss(F.conv2d(pc, sy, padding=1),
                                 F.conv2d(tc, sy, padding=1)))
        return loss / pred.shape[1]


class HeteroscedasticMultiOutputLoss(nn.Module):
    """
    Gaussian NLL loss for a heteroscedastic multi-output temporal SR model.

    For each pixel i with predicted mean μ_i and log-variance s_i = log σ²_i:

        NLL_i = (1/2) · [ exp(-s_i) · (y_i - μ_i)² + s_i ]

    This is the proper scoring rule from Lakshminarayanan et al. (2017).

    Auxiliary losses (spectral, gradient, temporal) operate ONLY on the mean
    predictions so they don't distort variance learning.

    Warm-up
    -------
    During the first ``nll_warmup_epochs`` we linearly blend from pure L1
    to pure NLL so the mean heads can converge before variance learning
    kicks in (avoids early divergence).
    """

    def __init__(
        self,
        alpha_spectral=0.137,
        alpha_gradient=0.01,
        alpha_temporal=0.5,
        nll_warmup_epochs=5,
    ):
        super().__init__()
        self.alpha_spectral = alpha_spectral
        self.alpha_gradient = alpha_gradient
        self.alpha_temporal = alpha_temporal
        self.nll_warmup_epochs = nll_warmup_epochs

        self.sam_loss = SpectralAngleLoss()
        self.grad_loss = LocalGradientLoss()
        self.temporal_loss = TemporalConsistencyLoss(mode='l1')

        # blend factor: 0 → pure L1, 1 → pure NLL
        self._nll_weight = 0.0

    def update_nll_weight(self, epoch: int) -> float:
        """Linear warm-up of NLL contribution."""
        if self.nll_warmup_epochs <= 0:
            self._nll_weight = 1.0
        elif epoch >= self.nll_warmup_epochs:
            self._nll_weight = 1.0
        else:
            self._nll_weight = epoch / self.nll_warmup_epochs
        return self._nll_weight

    # ------------------------------------------------------------------ #
    @staticmethod
    def _gaussian_nll(mean, logvar, target):
        """
        Pixel-wise Gaussian NLL, averaged over all dims.

            L = (1/2) [ exp(-logvar) * (target - mean)² + logvar ]

        The constant log(2π)/2 is dropped (doesn't affect gradients).
        """
        precision = torch.exp(-logvar)
        return 0.5 * torch.mean(precision * (target - mean) ** 2 + logvar)

    # ------------------------------------------------------------------ #
    def _frame_nll(self, mean, logvar, target):
        """NLL + auxiliary losses for a single spatial frame."""
        nll = self._gaussian_nll(mean, logvar, target)
        l1  = F.l1_loss(mean, target)

        # blend L1 → NLL during warm-up
        recon = (1.0 - self._nll_weight) * l1 + self._nll_weight * nll

        sam  = self.sam_loss(mean, target)
        grad = self.grad_loss(mean, target)

        return recon + self.alpha_spectral * sam + self.alpha_gradient * grad

    # ------------------------------------------------------------------ #
    def forward(
        self,
        mean_10m, logvar_10m, target_10m,
        mean_20m, logvar_20m, target_20m,
        weight_10m=1.0, weight_20m=1.0,
    ):
        B, T = mean_10m.shape[:2]

        loss_10m, loss_20m = 0.0, 0.0
        for t in range(T):
            loss_10m += self._frame_nll(
                mean_10m[:, t], logvar_10m[:, t], target_10m[:, t]
            )
            loss_20m += self._frame_nll(
                mean_20m[:, t], logvar_20m[:, t], target_20m[:, t]
            )
        loss_10m /= T
        loss_20m /= T

        # temporal consistency (mean only)
        temp_10m = self.temporal_loss(mean_10m, target_10m)
        temp_20m = self.temporal_loss(mean_20m, target_20m)

        total_10m = loss_10m + self.alpha_temporal * temp_10m
        total_20m = loss_20m + self.alpha_temporal * temp_20m

        total = weight_10m * total_10m + weight_20m * total_20m

        # diagnostics
        with torch.no_grad():
            avg_var_10m = torch.exp(logvar_10m).mean().item()
            avg_var_20m = torch.exp(logvar_20m).mean().item()

        loss_dict = {
            'recon_10m': loss_10m.item() if torch.is_tensor(loss_10m) else loss_10m,
            'recon_20m': loss_20m.item() if torch.is_tensor(loss_20m) else loss_20m,
            'temporal_10m': temp_10m.item(),
            'temporal_20m': temp_20m.item(),
            'avg_var_10m': avg_var_10m,
            'avg_var_20m': avg_var_20m,
            'nll_weight': self._nll_weight,
            'total': total.item(),
        }
        return total, loss_dict


# =============================================================================
# DATASET  (identical to original — copied for self-containment)
# =============================================================================

class MultiOutputMmapDataset(Dataset):
    """Memory-mapped multi-output dataset."""

    def __init__(self, mmap_dir, sequence_length=3, augment=True):
        mmap_dir = Path(mmap_dir)
        self.lr     = np.load(mmap_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(mmap_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr_10m = np.load(mmap_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(mmap_dir / 'hr_20m.npy', mmap_mode='r')
        self.n_samples = self.lr.shape[0]
        self.sequence_length = sequence_length
        self.augment = augment

        print(f"MultiOutput Dataset: {self.n_samples} samples")
        print(f"  LR 10m: {self.lr.shape}, LR 20m: {self.lr_20m.shape}")
        print(f"  HR 10m: {self.hr_10m.shape}, HR 20m: {self.hr_20m.shape}")

    def __len__(self):
        return self.n_samples

    def apply_augmentation(self, lr_10m, lr_20m, hr_10m, hr_20m):
        if np.random.random() > 0.5:
            lr_10m = np.flip(lr_10m, axis=2).copy()
            lr_20m = np.flip(lr_20m, axis=2).copy()
            hr_10m = np.flip(hr_10m, axis=2).copy()
            hr_20m = np.flip(hr_20m, axis=2).copy()
        if np.random.random() > 0.5:
            lr_10m = np.flip(lr_10m, axis=1).copy()
            lr_20m = np.flip(lr_20m, axis=1).copy()
            hr_10m = np.flip(hr_10m, axis=1).copy()
            hr_20m = np.flip(hr_20m, axis=1).copy()
        k = np.random.randint(4)
        if k > 0:
            lr_10m = np.rot90(lr_10m, k, axes=(1, 2)).copy()
            lr_20m = np.rot90(lr_20m, k, axes=(1, 2)).copy()
            hr_10m = np.rot90(hr_10m, k, axes=(1, 2)).copy()
            hr_20m = np.rot90(hr_20m, k, axes=(1, 2)).copy()
        return lr_10m, lr_20m, hr_10m, hr_20m

    def __getitem__(self, idx):
        lr_10m = np.array(self.lr[idx])
        lr_20m = np.array(self.lr_20m[idx])
        hr_10m = np.array(self.hr_10m[idx])
        hr_20m = np.array(self.hr_20m[idx])

        if self.augment:
            lr_10m, lr_20m, hr_10m, hr_20m = self.apply_augmentation(
                lr_10m, lr_20m, hr_10m, hr_20m
            )

        lr_10m = torch.from_numpy(lr_10m).float().permute(0, 3, 1, 2)
        lr_20m = torch.from_numpy(lr_20m).float().permute(0, 3, 1, 2)
        hr_10m = torch.from_numpy(hr_10m).float().permute(0, 3, 1, 2)
        hr_20m = torch.from_numpy(hr_20m).float().permute(0, 3, 1, 2)

        return {
            'lr_10m': lr_10m,
            'lr_20m': lr_20m,
            'hr_10m': hr_10m,
            'hr_20m': hr_20m,
        }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, value, epoch):
        if value < (self.best_value - self.min_delta):
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val ** 2 / mse)


def compute_temporal_psnr(pred, target, max_val=1.0):
    B, T = pred.shape[:2]
    total_psnr = 0.0
    for t in range(T):
        total_psnr += compute_psnr(pred[:, t], target[:, t], max_val)
    return total_psnr / T


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_model(data_dir, checkpoint_dir, experiment_name, cfg, seed):
    """Train a single heteroscedastic ensemble member."""

    print(f"\n{'=' * 70}")
    print(f"Training: {experiment_name}")
    print(f"Data: {data_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Learned Upscaling: {cfg['resize_method'].upper()}")
    print(f"Seed: {seed}")
    print(f"{'=' * 70}")

    set_all_seeds(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ========== DATASET ==========
    dataset = MultiOutputMmapDataset(
        mmap_dir=data_dir,
        sequence_length=cfg.get('num_frames', 3),
        augment=True,
    )

    n_val = min(100, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),  # same split for all members
    )
    val_dataset.dataset.augment = False

    print(f"Train: {n_train}, Val: {n_val}")

    # NOTE: workers seeded differently per member via the global seed
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True,
        worker_init_fn=lambda wid: np.random.seed(seed + wid),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # ========== MODEL ==========
    model = HetMultiOutputTemporalSRDirectLearnedUps(
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=cfg.get('num_frames', 3),
        num_temporal_blocks=cfg.get('num_temporal_blocks', 2),
        resize_method=cfg['resize_method'],
        logvar_init=cfg.get('logvar_init', -3.0),
        logvar_clamp=cfg.get('logvar_clamp', (-10.0, 10.0)),
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model: {cfg['base_channels']}ch, {cfg['num_encoder_blocks']}blk, "
          f"T={cfg.get('num_frames', 3)}, resize={cfg['resize_method']}, "
          f"{n_params:,} params (incl. variance heads)")

    # ========== LOSS ==========
    criterion = HeteroscedasticMultiOutputLoss(
        alpha_spectral=cfg['alpha_spectral'],
        alpha_gradient=cfg['alpha_gradient'],
        alpha_temporal=cfg.get('alpha_temporal', 0.5),
        nll_warmup_epochs=cfg.get('nll_warmup_epochs', 5),
    )

    print(f"\nLoss: Gaussian NLL (warmup {cfg.get('nll_warmup_epochs', 5)} epochs)")
    print(f"  Aux weights: SAM={cfg['alpha_spectral']:.3f}, "
          f"Grad={cfg['alpha_gradient']:.4f}, "
          f"Temporal={cfg.get('alpha_temporal', 0.5):.3f}")

    # ========== OPTIMIZER ==========
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=cfg['patience_scheduler'],
    )
    early_stopping = EarlyStopping(patience=cfg['patience'])

    # ========== RESUME ==========
    start_epoch = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_psnr_10m': [], 'val_psnr_20m': [],
        'lr': [], 'avg_var_10m': [], 'avg_var_20m': [],
        'nll_weight': [],
    }
    best_val_loss = float('inf')
    best_psnr = 0.0

    latest_ckpt = checkpoint_dir / 'latest_model.pt'
    if latest_ckpt.exists():
        print(f"\nResuming from: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        best_psnr = (ckpt.get('val_psnr_10m', 0) + ckpt.get('val_psnr_20m', 0)) / 2

        history_file = checkpoint_dir / 'training_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)

        print(f"Resuming from epoch {start_epoch} (best_val_loss: {best_val_loss:.4f})")
    else:
        print("\nNo checkpoint found — starting fresh.")

    # ========== TRAINING LOOP ==========
    print(f"\nStarting training for {cfg['epochs']} epochs…")
    accumulation_steps = cfg.get('accumulation_steps', 1)

    for epoch in range(start_epoch, cfg['epochs']):
        epoch_start = time.time()

        # update NLL warmup blend
        nll_w = criterion.update_nll_weight(epoch)

        # ----- TRAIN -----
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            lr_10m = batch['lr_10m'].to(device)
            lr_20m = batch['lr_20m'].to(device)
            hr_10m = batch['hr_10m'].to(device)
            hr_20m = batch['hr_20m'].to(device)

            mean_10m, logvar_10m, mean_20m, logvar_20m = model(
                lr_10m, lr_20m,
                target_size_10m=hr_10m.shape[-2:],
                target_size_20m=hr_20m.shape[-2:],
            )

            loss, loss_dict = criterion(
                mean_10m, logvar_10m, hr_10m,
                mean_20m, logvar_20m, hr_20m,
                weight_10m=cfg['weight_10m'],
                weight_20m=cfg['weight_20m'],
            )

            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

        # flush remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= len(train_loader)

        # ----- VALIDATE -----
        model.eval()
        val_loss = 0.0
        val_psnr_10m = 0.0
        val_psnr_20m = 0.0
        val_avg_var_10m = 0.0
        val_avg_var_20m = 0.0

        with torch.no_grad():
            for batch in val_loader:
                lr_10m = batch['lr_10m'].to(device)
                lr_20m = batch['lr_20m'].to(device)
                hr_10m = batch['hr_10m'].to(device)
                hr_20m = batch['hr_20m'].to(device)

                mean_10m, logvar_10m, mean_20m, logvar_20m = model(
                    lr_10m, lr_20m,
                    target_size_10m=hr_10m.shape[-2:],
                    target_size_20m=hr_20m.shape[-2:],
                )

                loss, loss_dict = criterion(
                    mean_10m, logvar_10m, hr_10m,
                    mean_20m, logvar_20m, hr_20m,
                    weight_10m=cfg['weight_10m'],
                    weight_20m=cfg['weight_20m'],
                )

                val_loss += loss.item()
                val_psnr_10m += compute_temporal_psnr(mean_10m, hr_10m).item()
                val_psnr_20m += compute_temporal_psnr(mean_20m, hr_20m).item()
                val_avg_var_10m += loss_dict['avg_var_10m']
                val_avg_var_20m += loss_dict['avg_var_20m']

        n_val_batches = len(val_loader)
        val_loss /= n_val_batches
        val_psnr_10m /= n_val_batches
        val_psnr_20m /= n_val_batches
        val_avg_var_10m /= n_val_batches
        val_avg_var_20m /= n_val_batches
        combined_psnr = (val_psnr_10m + val_psnr_20m) / 2

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # ----- history -----
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr_10m'].append(val_psnr_10m)
        history['val_psnr_20m'].append(val_psnr_20m)
        history['lr'].append(current_lr)
        history['avg_var_10m'].append(val_avg_var_10m)
        history['avg_var_20m'].append(val_avg_var_20m)
        history['nll_weight'].append(nll_w)

        # ----- checkpoints -----
        ckpt_payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_psnr_10m': val_psnr_10m,
            'val_psnr_20m': val_psnr_20m,
            'config': cfg,
            'seed': seed,
            'model_type': 'HetMultiOutputTemporalSRDirectLearnedUps',
            'resize_method': cfg['resize_method'],
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_psnr = combined_psnr
            torch.save(ckpt_payload, checkpoint_dir / 'best_model.pt')

        torch.save(ckpt_payload, checkpoint_dir / 'latest_model.pt')

        with open(checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        epoch_time = time.time() - epoch_start
        print(
            f"E {epoch + 1:3d}/{cfg['epochs']} | "
            f"Tr: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"PSNR 10m: {val_psnr_10m:.2f} 20m: {val_psnr_20m:.2f} | "
            f"σ²_10m: {val_avg_var_10m:.4f} σ²_20m: {val_avg_var_20m:.4f} | "
            f"NLL_w: {nll_w:.2f} | LR: {current_lr:.2e} | {epoch_time:.1f}s"
        )

        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            print(f"Best epoch: {early_stopping.best_epoch + 1}, "
                  f"val_loss: {best_val_loss:.4f}")
            break

    print(f"\nTraining complete!  Best Val Loss: {best_val_loss:.4f}, "
          f"Best PSNR: {best_psnr:.2f} dB")

    # Write completion marker so the shell script can skip already-finished members
    (checkpoint_dir / 'COMPLETED').write_text(
        f"best_epoch={early_stopping.best_epoch + 1}\n"
        f"best_val_loss={best_val_loss:.6f}\n"
        f"best_psnr={best_psnr:.4f}\n"
    )

    return best_val_loss, best_psnr


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train heteroscedastic ensemble member — Direct + LearnedUps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ensemble member 0 (seed 1000) with adaptive upscaling
  python train_het_ensemble_direct_learnedUps.py --seed 1000 --ensemble_id 0

  # Train ensemble member 1 (seed 2000)
  python train_het_ensemble_direct_learnedUps.py --seed 2000 --ensemble_id 1

  # Train 5 members sequentially using the companion script:
  bash run_ensemble.sh
        """,
    )
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for this ensemble member')
    parser.add_argument('--ensemble_id', type=int, required=True,
                        help='Ensemble member index (0, 1, 2, …)')
    parser.add_argument('--dataset', choices=['full', 'filtered', 'both'],
                        default='full', help='Which dataset split to train on')
    parser.add_argument('--resize_method', choices=['meta', 'adaptive', 'bilinear'],
                        default='adaptive',
                        help='Learned upscaling method')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--alpha_temporal', type=float, default=0.5)
    parser.add_argument('--nll_warmup', type=int, default=5,
                        help='Epochs of L1→NLL warm-up')
    parser.add_argument('--temporal_blocks', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory path')
    parser.add_argument('--checkpoint_base', type=str, default=None,
                        help='Override base checkpoint directory')
    args = parser.parse_args()

    # ---- build config ----
    cfg = BEST_CONFIG.copy()
    cfg['epochs'] = args.epochs
    cfg['batch_size'] = args.batch_size
    cfg['accumulation_steps'] = args.accumulation_steps
    cfg['alpha_temporal'] = args.alpha_temporal
    cfg['nll_warmup_epochs'] = args.nll_warmup
    cfg['num_temporal_blocks'] = args.temporal_blocks
    cfg['resize_method'] = args.resize_method

    # ---- paths ----
    # Uncert_Quant -> multioutput_schedper -> variants -> training -> scripts -> SR
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    FULL_DATA = Path(args.data_dir) if args.data_dir else (
        PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap'
    )
    FILTERED_DATA = PROJECT_ROOT / 'data/sr_training/5band_v3/combined/mmap_filtered'
    CHECKPOINT_BASE = Path(args.checkpoint_base) if args.checkpoint_base else (
        PROJECT_ROOT / 'checkpoints'
    )

    suffix = (
        f"het_ensemble_direct_learnedUps_{args.resize_method}"
        f"_seed{args.seed}_member{args.ensemble_id}"
    )

    results = {}

    if args.dataset in ['full', 'both']:
        val_loss, best_psnr = train_model(
            data_dir=FULL_DATA,
            checkpoint_dir=CHECKPOINT_BASE / f'{suffix}_full',
            experiment_name=(
                f'Het-Ensemble Direct+LearnedUps({args.resize_method}) '
                f'member={args.ensemble_id} seed={args.seed} [Full]'
            ),
            cfg=cfg,
            seed=args.seed,
        )
        results['full'] = {'val_loss': val_loss, 'psnr': best_psnr}

    if args.dataset in ['filtered', 'both']:
        val_loss, best_psnr = train_model(
            data_dir=FILTERED_DATA,
            checkpoint_dir=CHECKPOINT_BASE / f'{suffix}_filtered',
            experiment_name=(
                f'Het-Ensemble Direct+LearnedUps({args.resize_method}) '
                f'member={args.ensemble_id} seed={args.seed} [Filtered]'
            ),
            cfg=cfg,
            seed=args.seed,
        )
        results['filtered'] = {'val_loss': val_loss, 'psnr': best_psnr}

    print(f"\n{'=' * 70}")
    print(f"ENSEMBLE MEMBER {args.ensemble_id} (seed={args.seed}) SUMMARY")
    print(f"{'=' * 70}")
    for dataset_name, res in results.items():
        print(f"  {dataset_name.upper()}: Val Loss = {res['val_loss']:.4f}, "
              f"PSNR = {res['psnr']:.2f} dB")
