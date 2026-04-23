#!/usr/bin/env python3
"""
SOTA Super-Resolution Benchmark Script

Compares our proposed model against state-of-the-art SR models using their
original pretrained weights (no retraining), following the framework-level
comparison methodology established by Aybar et al. (2024) / OpenSR-Test.

SOTA models evaluated:
  1. HAN (Holistic Attention Network) - General-purpose pixel-wise regression
  2. LDM (Latent Diffusion Model) - General-purpose generative SR
  3. SatlasSR (ESRGAN-based) - Satellite cross-sensor SR
  4. Swin2-MoSE - Satellite Transformer-based SR
  5. SR4RS - Satellite SR (TensorFlow-based, evaluated separately)
  6. OpenSR-Model - Conservative satellite SR baseline
  7. SEN2SR (SEN2SRLite) - Lightweight CNN SR for Sentinel-2

Usage:
    # Step 1: Install dependencies
    pip install super-image diffusers opensr-test huggingface_hub timm

    # Step 2: Run benchmark
    python benchmark_sota.py --test_dir data/sr_training/all_stas_combined/mmap_test

    # Step 3: Run with subset for quick test
    python benchmark_sota.py --test_dir data/sr_training/all_stas_combined/mmap_test --max_samples 20

    # Step 4: Specify which models to include
    python benchmark_sota.py --models proposed_direct proposed_residual bilinear han ldm
"""

import sys
import os
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training' / 'models'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training' / 'variants' / 'multioutput_schedper'))

# Suppress excessive warnings from pretrained model downloads
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')


# =============================================================================
# METRICS (reuse from evaluate_models.py)
# =============================================================================

from scripts.evaluation.evaluate_models import (
    compute_psnr, compute_ssim, compute_scc, compute_rmse,
    compute_ergas, compute_sid, compute_all_metrics,
    compute_per_band_metrics
)


# =============================================================================
# DATA LOADING
# =============================================================================

class BenchmarkDataLoader:
    """Load test data and prepare it for both proposed and SOTA models."""

    def __init__(self, test_dir: Path, skip_indices: Optional[List[int]] = None):
        self.test_dir = Path(test_dir)
        self.skip_indices = set(skip_indices) if skip_indices else set()

        # Load mmap arrays: (N, T=3, H, W, C)
        self.lr_10m = np.load(self.test_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(self.test_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr_10m = np.load(self.test_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(self.test_dir / 'hr_20m.npy', mmap_mode='r')

        n_total = self.lr_10m.shape[0]
        self.valid_indices = [i for i in range(n_total) if i not in self.skip_indices]
        self.n_samples = len(self.valid_indices)

        print(f"  Test data: {n_total} total, {self.n_samples} valid samples")
        print(f"  LR 10m: {self.lr_10m.shape}, HR 10m: {self.hr_10m.shape}")
        print(f"  LR 20m: {self.lr_20m.shape}, HR 20m: {self.hr_20m.shape}")

    def __len__(self):
        return self.n_samples

    def get_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """Get sample by index. Returns dict with all arrays as (T, H, W, C)."""
        actual_idx = self.valid_indices[idx]
        mid = self.lr_10m.shape[1] // 2  # middle temporal frame

        lr_10m_seq = np.array(self.lr_10m[actual_idx])   # (T, 77, 77, 3)
        lr_20m_seq = np.array(self.lr_20m[actual_idx])    # (T, 38, 38, 2)
        hr_10m_mid = np.array(self.hr_10m[actual_idx, mid])  # (256, 256, 3)
        hr_20m_mid = np.array(self.hr_20m[actual_idx, mid])  # (128, 128, 2)

        return {
            'lr_10m_seq': lr_10m_seq,       # (T, 77, 77, 3) BGR, [0,1]
            'lr_20m_seq': lr_20m_seq,       # (T, 38, 38, 2)
            'lr_10m_mid': lr_10m_seq[mid],  # (77, 77, 3) - center frame
            'lr_20m_mid': lr_20m_seq[mid],  # (38, 38, 2)
            'hr_10m': hr_10m_mid,           # (256, 256, 3) - ground truth
            'hr_20m': hr_20m_mid,           # (128, 128, 2)
        }

    def prepare_for_proposed_model(self, sample: Dict) -> Dict[str, Any]:
        """Prepare tensors for our proposed model: (B, T, C, H, W)."""
        lr_10m = torch.from_numpy(sample['lr_10m_seq']).permute(0, 3, 1, 2).unsqueeze(0).float()
        lr_20m = torch.from_numpy(sample['lr_20m_seq']).permute(0, 3, 1, 2).unsqueeze(0).float()
        return {
            'lr_10m': lr_10m,      # (1, T, 3, 77, 77)
            'lr_20m': lr_20m,      # (1, T, 2, 38, 38)
            'target_size_10m': sample['hr_10m'].shape[:2],  # (256, 256)
            'target_size_20m': sample['hr_20m'].shape[:2],  # (128, 128)
        }

    def prepare_for_sisr_model(self, sample: Dict) -> torch.Tensor:
        """
        Prepare center frame for single-image SR models.
        Returns: (1, 3, 77, 77) RGB, [0,1] — native resolution, no resizing.

        SOTA models are fully convolutional and accept arbitrary input sizes.
        A 77×77 input with ×4 upscaling produces 308×308 output, which each
        wrapper then resizes to 256×256 to match the ground truth.
        This preserves the full input information for a fair comparison.
        """
        # Center frame: (77, 77, 3) BGR
        lr_mid = sample['lr_10m_mid']
        # BGR -> RGB
        lr_rgb = lr_mid[:, :, ::-1].copy()
        # (H, W, C) -> (1, C, H, W)
        lr_t = torch.from_numpy(lr_rgb).permute(2, 0, 1).unsqueeze(0).float()
        return lr_t


# =============================================================================
# SOTA MODEL WRAPPERS
# =============================================================================

class SOTAModelWrapper(ABC):
    """Abstract base class for SOTA model wrappers."""

    def __init__(self, name: str, device: torch.device):
        self.name = name
        self.device = device
        self.model = None

    @abstractmethod
    def load(self):
        """Load model and weights."""
        pass

    @abstractmethod
    def predict_10m(self, sample: Dict, data_loader: BenchmarkDataLoader) -> np.ndarray:
        """
        Produce SR prediction for 10m branch.
        Returns: (256, 256, 3) BGR, [0,1]
        """
        pass

    def predict_20m(self, sample: Dict, data_loader: BenchmarkDataLoader) -> np.ndarray:
        """
        Produce SR prediction for 20m branch.
        Default: bilinear upscale (SISR models have no 20m branch).
        Returns: (128, 128, 2) [0,1]
        """
        lr_20m = sample['lr_20m_mid']  # (38, 38, 2)
        target_h, target_w = sample['hr_20m'].shape[:2]
        lr_t = torch.from_numpy(lr_20m).permute(2, 0, 1).unsqueeze(0).float()
        sr_t = F.interpolate(lr_t, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return sr_t[0].permute(1, 2, 0).numpy()


class BilinearWrapper(SOTAModelWrapper):
    """Bilinear interpolation baseline."""

    def __init__(self, device: torch.device):
        super().__init__('Bilinear', device)

    def load(self):
        print(f"  [{self.name}] No model to load (analytical baseline)")

    def predict_10m(self, sample, data_loader):
        lr = sample['lr_10m_mid']  # (77, 77, 3)
        target_h, target_w = sample['hr_10m'].shape[:2]
        lr_t = torch.from_numpy(lr).permute(2, 0, 1).unsqueeze(0).float()
        sr_t = F.interpolate(lr_t, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return sr_t[0].permute(1, 2, 0).numpy()

    def predict_20m(self, sample, data_loader):
        lr = sample['lr_20m_mid']  # (38, 38, 2)
        target_h, target_w = sample['hr_20m'].shape[:2]
        lr_t = torch.from_numpy(lr).permute(2, 0, 1).unsqueeze(0).float()
        sr_t = F.interpolate(lr_t, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return sr_t[0].permute(1, 2, 0).numpy()


class BicubicWrapper(SOTAModelWrapper):
    """Bicubic interpolation baseline."""

    def __init__(self, device: torch.device):
        super().__init__('Bicubic', device)

    def load(self):
        print(f"  [{self.name}] No model to load (analytical baseline)")

    def predict_10m(self, sample, data_loader):
        lr = sample['lr_10m_mid']
        target_h, target_w = sample['hr_10m'].shape[:2]
        lr_t = torch.from_numpy(lr).permute(2, 0, 1).unsqueeze(0).float()
        sr_t = F.interpolate(lr_t, size=(target_h, target_w), mode='bicubic', align_corners=False)
        return torch.clamp(sr_t, 0, 1)[0].permute(1, 2, 0).numpy()

    def predict_20m(self, sample, data_loader):
        lr = sample['lr_20m_mid']
        target_h, target_w = sample['hr_20m'].shape[:2]
        lr_t = torch.from_numpy(lr).permute(2, 0, 1).unsqueeze(0).float()
        sr_t = F.interpolate(lr_t, size=(target_h, target_w), mode='bicubic', align_corners=False)
        return torch.clamp(sr_t, 0, 1)[0].permute(1, 2, 0).numpy()


class ProposedModelWrapper(SOTAModelWrapper):
    """Wrapper for our proposed model (direct or residual variant)."""

    def __init__(self, name: str, checkpoint_path: Path, device: torch.device):
        super().__init__(name, device)
        self.checkpoint_path = checkpoint_path

    def load(self):
        from scripts.evaluation.evaluate_models import load_model
        self.model, self.checkpoint = load_model(self.checkpoint_path, self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        epoch = self.checkpoint.get('epoch', '?')
        print(f"  [{self.name}] Loaded from {self.checkpoint_path.name}, epoch={epoch}, params={n_params:,}")

    def predict_10m(self, sample, data_loader):
        inputs = data_loader.prepare_for_proposed_model(sample)
        with torch.no_grad():
            lr_10m = inputs['lr_10m'].to(self.device)
            lr_20m = inputs['lr_20m'].to(self.device)
            try:
                sr_10m, sr_20m = self.model(
                    lr_10m, lr_20m,
                    target_size_10m=inputs['target_size_10m'],
                    target_size_20m=inputs['target_size_20m']
                )
            except TypeError:
                sr_10m, sr_20m = self.model(lr_10m, lr_20m)

            # Handle multi-temporal output (B, T, C, H, W)
            if sr_10m.dim() == 5:
                mid = sr_10m.shape[1] // 2
                sr_10m = sr_10m[:, mid]

            # Store 20m prediction for reuse
            if sr_20m.dim() == 5:
                mid = sr_20m.shape[1] // 2
                sr_20m = sr_20m[:, mid]
            self._last_sr_20m = sr_20m.cpu().numpy()[0].transpose(1, 2, 0)

        return sr_10m.cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, C)

    def predict_20m(self, sample, data_loader):
        # Return cached 20m prediction from predict_10m call
        if hasattr(self, '_last_sr_20m'):
            return self._last_sr_20m
        # Fallback to bilinear if called independently
        return super().predict_20m(sample, data_loader)


class HANWrapper(SOTAModelWrapper):
    """
    HAN (Holistic Attention Network) - Niu et al., 2020
    General-purpose SISR model, pretrained on DIV2K.
    Install: pip install super-image
    """

    def __init__(self, device: torch.device):
        super().__init__('HAN (Niu et al. 2020)', device)

    def load(self):
        from super_image import HanModel
        self.model = HanModel.from_pretrained('eugenesiow/han', scale=4)
        self.model = self.model.to(self.device)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  [{self.name}] Loaded from HuggingFace, params={n_params:,}")

    def predict_10m(self, sample, data_loader):
        lr_input = data_loader.prepare_for_sisr_model(sample).to(self.device)  # (1,3,77,77) RGB
        with torch.no_grad():
            sr_rgb = self.model(lr_input)  # (1, 3, 308, 308) RGB (77×4)
        sr_rgb = torch.clamp(sr_rgb, 0, 1)
        # Resize ×4 output (308×308) to match GT resolution (256×256)
        target_h, target_w = sample['hr_10m'].shape[:2]
        if sr_rgb.shape[2] != target_h or sr_rgb.shape[3] != target_w:
            sr_rgb = F.interpolate(sr_rgb, size=(target_h, target_w),
                                   mode='bilinear', align_corners=False)
        sr_np = sr_rgb[0].cpu().numpy().transpose(1, 2, 0)  # (256,256,3) RGB
        # RGB -> BGR to match our data format
        return sr_np[:, :, ::-1].copy()


class LDMWrapper(SOTAModelWrapper):
    """
    LDM (Latent Diffusion Model) - Rombach et al., 2022
    General-purpose generative SR, pretrained on OpenImages.
    Install: pip install diffusers transformers accelerate
    """

    def __init__(self, device: torch.device, num_inference_steps: int = 50):
        super().__init__('LDM (Rombach et al. 2022)', device)
        self.num_inference_steps = num_inference_steps

    def load(self):
        from diffusers import LDMSuperResolutionPipeline
        self.pipe = LDMSuperResolutionPipeline.from_pretrained(
            "CompVis/ldm-super-resolution-4x-openimages"
        )
        self.pipe = self.pipe.to(self.device)
        print(f"  [{self.name}] Loaded from HuggingFace (diffusers), steps={self.num_inference_steps}")

    def predict_10m(self, sample, data_loader):
        from PIL import Image

        # Get center frame as RGB PIL image at native 77×77 resolution
        lr_mid = sample['lr_10m_mid']  # (77, 77, 3) BGR [0,1]
        lr_rgb = lr_mid[:, :, ::-1].copy()  # BGR -> RGB
        lr_np = (lr_rgb * 255).astype(np.uint8)
        lr_pil = Image.fromarray(lr_np)  # 77×77 PIL

        # Run diffusion (output will be 308×308 = 77×4)
        with torch.no_grad():
            sr_pil = self.pipe(lr_pil, num_inference_steps=self.num_inference_steps,
                               eta=1.0).images[0]

        sr_np = np.array(sr_pil).astype(np.float32) / 255.0  # (308,308,3) RGB
        # Resize to match GT resolution (256×256)
        target_h, target_w = sample['hr_10m'].shape[:2]
        if sr_np.shape[0] != target_h or sr_np.shape[1] != target_w:
            sr_t = torch.from_numpy(sr_np).permute(2, 0, 1).unsqueeze(0)
            sr_t = F.interpolate(sr_t, size=(target_h, target_w),
                                 mode='bilinear', align_corners=False)
            sr_np = sr_t[0].permute(1, 2, 0).numpy()
        # RGB -> BGR
        return sr_np[:, :, ::-1].copy()


class OpenSRModelWrapper(SOTAModelWrapper):
    """
    OpenSR-Model (LDSR-S2) - Donike et al., 2025
    Latent diffusion SR for Sentinel-2 RGB-NIR bands.
    Install: pip install opensr-model omegaconf

    NOTE: The model expects 4-band input (R, G, B, NIR=B8 at 842nm).
    Our 20m branch band 2 is B8A (NIR narrow, 865nm), which falls within
    B8's spectral range (785-900nm) and is highly correlated (r>0.98).
    B8A at 38×38 is bilinearly upsampled to 77×77 to match the RGB bands.
    RGB output channels are evaluated against 10m GT; the super-resolved NIR
    channel is evaluated against B8A in the 20m GT (B5 falls back to bilinear).
    """

    def __init__(self, device: torch.device):
        super().__init__('OpenSR-Model (Donike et al. 2025)', device)

    def load(self):
        try:
            import opensr_model
            from io import StringIO
            import requests
            from omegaconf import OmegaConf

            config_url = ("https://raw.githubusercontent.com/ESAOpenSR/opensr-model/"
                          "refs/heads/main/opensr_model/configs/config_10m.yaml")
            response = requests.get(config_url)
            response.raise_for_status()
            config = OmegaConf.load(StringIO(response.text))

            self.model = opensr_model.SRLatentDiffusion(config, device=str(self.device))
            self.model.load_pretrained(config.ckpt_version)
            self.sampling_steps = config.denoiser_settings.sampling_steps
            print(f"  [{self.name}] Loaded via opensr-model package "
                  f"(ckpt={config.ckpt_version}, steps={self.sampling_steps})")
        except ImportError:
            raise ImportError(
                "opensr-model not installed. Install with: pip install opensr-model omegaconf"
            )

    def predict_10m(self, sample, data_loader):
        # OpenSR-Model expects (B, 4, H, W) Sentinel-2 reflectance [0, 1] (R,G,B,NIR)
        lr_mid = sample['lr_10m_mid']  # (77, 77, 3) BGR [0,1]
        lr_20m = sample['lr_20m_mid']  # (38, 38, 2) [B5, B8A]
        # BGR -> RGB
        lr_rgb = lr_mid[:, :, ::-1].copy()
        lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).float()  # (3, 77, 77)
        # Use real B8A (NIR narrow, 865nm) from 20m branch as NIR channel
        # B8A falls within B8's spectral range (785-900nm), highly correlated
        nir_20m = torch.from_numpy(lr_20m[:, :, 1:2].copy()).permute(2, 0, 1).float()  # (1, 38, 38)
        nir_77 = F.interpolate(nir_20m.unsqueeze(0), size=(77, 77),
                               mode='bilinear', align_corners=False)[0]  # (1, 77, 77)
        lr_4band = torch.cat([lr_tensor, nir_77], dim=0)  # (4, 77, 77)
        lr_4band = lr_4band.to(self.device)

        with torch.no_grad():
            sr = self.model(lr_4band, sampling_steps=self.sampling_steps)  # (1, 4, H, W)

        sr_clamped = torch.clamp(sr, 0, 1)[0]  # (4, H, W)

        # Cache SR NIR (band 4) for predict_20m
        sr_nir = sr_clamped[3:4].cpu()  # (1, H, W)
        target_20m_h, target_20m_w = sample['hr_20m'].shape[:2]
        sr_nir_resized = F.interpolate(sr_nir.unsqueeze(0), size=(target_20m_h, target_20m_w),
                                       mode='bilinear', align_corners=False)
        self._last_sr_nir = sr_nir_resized[0, 0].numpy()  # (128, 128)

        # RGB channels
        sr_rgb = sr_clamped[:3].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3) RGB

        # Resize to GT resolution (256×256) if needed
        target_h, target_w = sample['hr_10m'].shape[:2]
        if sr_rgb.shape[0] != target_h or sr_rgb.shape[1] != target_w:
            sr_t = torch.from_numpy(sr_rgb).permute(2, 0, 1).unsqueeze(0)
            sr_t = F.interpolate(sr_t, size=(target_h, target_w),
                                 mode='bilinear', align_corners=False)
            sr_rgb = sr_t[0].permute(1, 2, 0).numpy()

        # RGB -> BGR
        return sr_rgb[:, :, ::-1].copy()

    def predict_20m(self, sample, data_loader):
        """
        20m prediction: B5 (Red Edge) via bilinear, B8A (NIR) from SR output.
        The model produces a super-resolved NIR band which we use for B8A.
        Returns: (128, 128, 2) [B5, B8A]
        """
        # B5: bilinear upscale (model doesn't produce Red Edge)
        lr_20m = sample['lr_20m_mid']  # (38, 38, 2)
        target_h, target_w = sample['hr_20m'].shape[:2]
        b5_lr = torch.from_numpy(lr_20m[:, :, 0:1].copy()).permute(2, 0, 1).unsqueeze(0).float()
        b5_sr = F.interpolate(b5_lr, size=(target_h, target_w),
                              mode='bilinear', align_corners=False)
        b5_np = b5_sr[0, 0].numpy()  # (128, 128)

        # B8A: from cached SR NIR output
        if hasattr(self, '_last_sr_nir'):
            b8a_np = self._last_sr_nir  # (128, 128)
        else:
            # Fallback to bilinear if predict_10m wasn't called first
            b8a_lr = torch.from_numpy(lr_20m[:, :, 1:2].copy()).permute(2, 0, 1).unsqueeze(0).float()
            b8a_sr = F.interpolate(b8a_lr, size=(target_h, target_w),
                                   mode='bilinear', align_corners=False)
            b8a_np = b8a_sr[0, 0].numpy()

        return np.stack([b5_np, b8a_np], axis=-1)  # (128, 128, 2)


class SEN2SRWrapper(SOTAModelWrapper):
    """
    SEN2SRLite (NonReference_RGBN_x4) - Aybar & Donike, 2025
    Lightweight CNN SR for Sentinel-2 RGB+NIR bands.
    Install: pip install sen2sr mlstac

    NOTE: The model expects fixed 128×128 input (4-band: R,G,B,NIR) and produces
    512×512 output (×4). Our 77×77 LR is bilinearly resized to 128×128 before
    inference, then the 512×512 output is resized to 256×256 to match GT.
    NIR channel uses real B8A from the 20m branch (bilinearly upsampled to match).
    RGB output channels are evaluated against 10m GT; the super-resolved NIR
    channel is evaluated against B8A in the 20m GT (B5 falls back to bilinear).
    """

    def __init__(self, device: torch.device,
                 model_dir: Optional[Path] = None):
        super().__init__('SEN2SR (Aybar & Donike 2025)', device)
        self.model_dir = model_dir or PROJECT_ROOT / 'external' / 'sen2sr_model' / 'SEN2SRLite_RGBN'

    def load(self):
        try:
            import mlstac
        except ImportError:
            raise ImportError(
                "mlstac not installed. Install with: pip install sen2sr mlstac"
            )

        # Download model weights if not already present
        if not self.model_dir.exists():
            print(f"  [{self.name}] Downloading SEN2SRLite NonReference_RGBN_x4...")
            mlstac.download(
                file='https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json',
                output_dir=str(self.model_dir),
            )

        self.model = mlstac.load(str(self.model_dir)).compiled_model(device=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  [{self.name}] Loaded SEN2SRLite NonReference_RGBN_x4, params={n_params:,}")

    def predict_10m(self, sample, data_loader):
        # SEN2SRLite expects (B, 4, 128, 128) S2 reflectance [0,1] (R,G,B,NIR)
        lr_mid = sample['lr_10m_mid']  # (77, 77, 3) BGR [0,1]
        lr_20m = sample['lr_20m_mid']  # (38, 38, 2) [B5, B8A]
        # BGR -> RGB
        lr_rgb = lr_mid[:, :, ::-1].copy()
        lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).float()  # (3, 77, 77)
        # Use real B8A (NIR narrow, 865nm) from 20m branch
        nir_20m = torch.from_numpy(lr_20m[:, :, 1:2].copy()).permute(2, 0, 1).float()  # (1, 38, 38)
        nir_77 = F.interpolate(nir_20m.unsqueeze(0), size=(77, 77),
                               mode='bilinear', align_corners=False)[0]  # (1, 77, 77)
        lr_4band = torch.cat([lr_tensor, nir_77], dim=0).unsqueeze(0)  # (1, 4, 77, 77)

        # Resize to model's expected 128×128 input
        lr_128 = F.interpolate(lr_4band, size=(128, 128),
                               mode='bilinear', align_corners=False)
        lr_128 = lr_128.to(self.device)

        with torch.no_grad():
            sr = self.model(lr_128)  # (1, 4, 512, 512)

        sr_clamped = torch.clamp(sr, 0, 1)

        # Cache SR NIR (band 4) for predict_20m
        sr_nir = sr_clamped[:, 3:4]  # (1, 1, 512, 512)
        target_20m_h, target_20m_w = sample['hr_20m'].shape[:2]
        sr_nir_resized = F.interpolate(sr_nir, size=(target_20m_h, target_20m_w),
                                       mode='bilinear', align_corners=False)
        self._last_sr_nir = sr_nir_resized[0, 0].cpu().numpy()  # (128, 128)

        # RGB channels
        sr_rgb = sr_clamped[:, :3]

        # Resize 512×512 → 256×256 to match GT
        target_h, target_w = sample['hr_10m'].shape[:2]
        sr_rgb = F.interpolate(sr_rgb, size=(target_h, target_w),
                               mode='bilinear', align_corners=False)
        sr_np = sr_rgb[0].cpu().numpy().transpose(1, 2, 0)  # (256,256,3) RGB

        # RGB -> BGR
        return sr_np[:, :, ::-1].copy()

    def predict_20m(self, sample, data_loader):
        """
        20m prediction: B5 (Red Edge) via bilinear, B8A (NIR) from SR output.
        Returns: (128, 128, 2) [B5, B8A]
        """
        lr_20m = sample['lr_20m_mid']  # (38, 38, 2)
        target_h, target_w = sample['hr_20m'].shape[:2]

        # B5: bilinear upscale (model doesn't produce Red Edge)
        b5_lr = torch.from_numpy(lr_20m[:, :, 0:1].copy()).permute(2, 0, 1).unsqueeze(0).float()
        b5_sr = F.interpolate(b5_lr, size=(target_h, target_w),
                              mode='bilinear', align_corners=False)
        b5_np = b5_sr[0, 0].numpy()  # (128, 128)

        # B8A: from cached SR NIR output
        if hasattr(self, '_last_sr_nir'):
            b8a_np = self._last_sr_nir  # (128, 128)
        else:
            b8a_lr = torch.from_numpy(lr_20m[:, :, 1:2].copy()).permute(2, 0, 1).unsqueeze(0).float()
            b8a_sr = F.interpolate(b8a_lr, size=(target_h, target_w),
                                   mode='bilinear', align_corners=False)
            b8a_np = b8a_sr[0, 0].numpy()

        return np.stack([b5_np, b8a_np], axis=-1)  # (128, 128, 2)


class Swin2MoSEWrapper(SOTAModelWrapper):
    """
    Swin2-MoSE - Rossi et al., 2025
    Transformer-based SR for remote sensing, pretrained on Sen2Venus.
    Requires cloning the repo: git clone https://github.com/IMPLabUniPr/swin2-mose

    NOTE: This requires manual setup. See SETUP section below.
    """

    def __init__(self, device: torch.device, repo_path: Optional[Path] = None,
                 checkpoint_path: Optional[Path] = None):
        super().__init__('Swin2-MoSE (Rossi et al. 2025)', device)
        self.repo_path = repo_path or PROJECT_ROOT / 'external' / 'swin2-mose'
        self.checkpoint_path = checkpoint_path

    def load(self):
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"Swin2-MoSE repo not found at {self.repo_path}. "
                "Clone it with: git clone https://github.com/IMPLabUniPr/swin2-mose external/swin2-mose\n"
                "Then download pretrained weights from GitHub Releases v1.0"
            )

        # Add repo to path
        sys.path.insert(0, str(self.repo_path))
        from swin2_mose_model.model import Swin2MoSE

        # Load model (4-band: B5,B6,B7,B8A from Sen2Venus by default)
        # For RGB we need a 3-channel configuration
        self.model = Swin2MoSE(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
        )

        if self.checkpoint_path and self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            if 'model' in ckpt:
                self.model.load_state_dict(ckpt['model'], strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  [{self.name}] Loaded, params={n_params:,}")

    def predict_10m(self, sample, data_loader):
        lr_input = data_loader.prepare_for_sisr_model(sample).to(self.device)  # (1,3,77,77)
        with torch.no_grad():
            sr = self.model(lr_input)  # (1, 3, 308, 308)
        sr = torch.clamp(sr, 0, 1)
        # Resize to GT resolution
        target_h, target_w = sample['hr_10m'].shape[:2]
        if sr.shape[2] != target_h or sr.shape[3] != target_w:
            sr = F.interpolate(sr, size=(target_h, target_w),
                               mode='bilinear', align_corners=False)
        sr_np = sr[0].cpu().numpy().transpose(1, 2, 0)
        return sr_np[:, :, ::-1].copy()  # RGB -> BGR


class SatlasSRWrapper(SOTAModelWrapper):
    """
    SatlasSR (ESRGAN-based) - Wolters et al., 2023
    Satellite multi-temporal SR, pretrained on S2-NAIP.
    Requires cloning: git clone https://github.com/allenai/satlas-super-resolution

    NOTE: This wrapper uses only the center frame (single-image mode) for fair
    comparison. SatlasSR supports multi-temporal input but expects TCI format.
    """

    def __init__(self, device: torch.device, repo_path: Optional[Path] = None,
                 checkpoint_path: Optional[Path] = None):
        super().__init__('SatlasSR (Wolters et al. 2023)', device)
        self.repo_path = repo_path or PROJECT_ROOT / 'external' / 'satlas-super-resolution'
        self.checkpoint_path = checkpoint_path

    def load(self):
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"SatlasSR repo not found at {self.repo_path}. "
                "Clone it with: git clone https://github.com/allenai/satlas-super-resolution external/satlas-super-resolution\n"
                "Then download model weights from the repo README."
            )

        sys.path.insert(0, str(self.repo_path))
        from ssr.archs.rrdbnet_arch import RRDBNet

        # Standard ESRGAN config for S2-NAIP (3ch RGB, 1 input image)
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
        )

        if self.checkpoint_path and self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            if 'params_ema' in ckpt:
                self.model.load_state_dict(ckpt['params_ema'])
            elif 'params' in ckpt:
                self.model.load_state_dict(ckpt['params'])
            else:
                self.model.load_state_dict(ckpt)

        self.model = self.model.to(self.device)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  [{self.name}] Loaded, params={n_params:,}")

    def predict_10m(self, sample, data_loader):
        # SatlasSR processes TCI (8-bit RGB normalized to [0,1] via /255)
        # Our data is reflectance [0,1]. We use it directly since
        # we evaluate the model as-is (pretrained weights paradigm).
        lr_input = data_loader.prepare_for_sisr_model(sample).to(self.device)  # (1,3,77,77)
        with torch.no_grad():
            sr = self.model(lr_input)  # (1, 3, 308, 308)
        sr = torch.clamp(sr, 0, 1)
        # Resize to GT resolution
        target_h, target_w = sample['hr_10m'].shape[:2]
        if sr.shape[2] != target_h or sr.shape[3] != target_w:
            sr = F.interpolate(sr, size=(target_h, target_w),
                               mode='bilinear', align_corners=False)
        sr_np = sr[0].cpu().numpy().transpose(1, 2, 0)
        return sr_np[:, :, ::-1].copy()


# =============================================================================
# BENCHMARK ENGINE
# =============================================================================

class SOTABenchmark:
    """Main benchmark engine for SOTA comparison."""

    METRIC_NAMES = ['psnr', 'ssim', 'scc', 'rmse', 'ergas', 'sid']
    HIGHER_BETTER = {'psnr', 'ssim', 'scc'}

    def __init__(self, test_dir: Path, output_dir: Path,
                 device: Optional[torch.device] = None,
                 skip_indices: Optional[List[int]] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load bad indices if available
        if skip_indices is None:
            bad_path = Path(test_dir).parent / 'bad_indices.npy'
            if bad_path.exists():
                skip_indices = np.load(bad_path).tolist()
                print(f"Loaded {len(skip_indices)} bad indices from {bad_path}")
            else:
                skip_indices = []

        # Also skip samples with 100% NaN in lr_20m (from evaluation config)
        nan_indices = [1, 8, 223, 272, 1088, 1103, 1175]
        skip_indices = sorted(set(skip_indices) | set(nan_indices))
        print(f"  Total skip indices: {len(skip_indices)} (including {len(nan_indices)} NaN lr_20m samples)")

        self.data = BenchmarkDataLoader(test_dir, skip_indices=skip_indices)
        self.models: Dict[str, SOTAModelWrapper] = {}
        self.results: Dict[str, Dict] = {}

    def add_model(self, wrapper: SOTAModelWrapper):
        """Add a model wrapper to the benchmark."""
        self.models[wrapper.name] = wrapper

    def load_all_models(self):
        """Verify which models can be loaded. Does NOT keep them in memory."""
        print("\n" + "=" * 70)
        print("CHECKING MODEL AVAILABILITY")
        print("=" * 70)

        self._loadable: List[str] = []
        failed = []
        for name, wrapper in list(self.models.items()):
            try:
                wrapper.load()
                self._loadable.append(name)
                self._unload_model(wrapper)
            except Exception as e:
                print(f"  [SKIP] {name}: {e}")
                failed.append(name)

        for name in failed:
            del self.models[name]

        print(f"\nAvailable models: {len(self.models)}")
        if failed:
            print(f"  Skipped {len(failed)}: {', '.join(failed)}")

    @staticmethod
    def _unload_model(wrapper: SOTAModelWrapper):
        """Free GPU memory occupied by a model."""
        # Move model/pipe to CPU first, then delete
        for attr in ('model', 'pipe'):
            obj = getattr(wrapper, attr, None)
            if obj is not None:
                try:
                    if hasattr(obj, 'to'):
                        obj.to('cpu')
                except Exception:
                    pass
                setattr(wrapper, attr, None)
        for cached in ('_last_sr_20m', '_last_sr_nir'):
            if hasattr(wrapper, cached):
                delattr(wrapper, cached)
        import gc; gc.collect()
        torch.cuda.empty_cache()

    def evaluate(self, max_samples: Optional[int] = None):
        """Run evaluation on all models, one at a time to conserve GPU memory."""
        n = min(max_samples or len(self.data), len(self.data))

        print(f"\n{'=' * 70}")
        print(f"EVALUATING {len(self.models)} models on {n} samples")
        print(f"{'=' * 70}")

        for name, wrapper in self.models.items():
            # Initialize results for this model
            self.results[name] = {
                'per_sample_10m': [],
                'per_sample_20m': [],
                'per_sample_band': {},
            }

            # Load model
            print(f"\n--- Loading {name} ---")
            try:
                wrapper.load()
            except Exception as e:
                print(f"  [SKIP] {name}: {e}")
                # Fill with NaN so downstream code can handle it
                nan_metrics = {m: float('nan') for m in self.METRIC_NAMES}
                for _ in range(n):
                    self.results[name]['per_sample_10m'].append(nan_metrics)
                    self.results[name]['per_sample_20m'].append(nan_metrics)
                continue

            # Evaluate all samples
            for idx in tqdm(range(n), desc=name):
                sample = self.data.get_sample(idx)
                gt_10m = sample['hr_10m']  # (256, 256, 3)
                gt_20m = sample['hr_20m']  # (128, 128, 2)

                try:
                    # 10m prediction
                    sr_10m = wrapper.predict_10m(sample, self.data)
                    sr_10m = np.clip(sr_10m, 0, 1)

                    # 20m prediction
                    sr_20m = wrapper.predict_20m(sample, self.data)
                    sr_20m = np.clip(sr_20m, 0, 1)

                    # Compute metrics
                    metrics_10m = compute_all_metrics(sr_10m, gt_10m, scale_ratio=3.33)
                    metrics_20m = compute_all_metrics(sr_20m, gt_20m, scale_ratio=3.33)

                    per_band_10m = compute_per_band_metrics(sr_10m, gt_10m)
                    per_band_20m = compute_per_band_metrics(sr_20m, gt_20m)

                    self.results[name]['per_sample_10m'].append(metrics_10m)
                    self.results[name]['per_sample_20m'].append(metrics_20m)

                    # Per-band
                    if not self.results[name]['per_sample_band']:
                        self.results[name]['per_sample_band'] = {
                            b: [] for b in list(per_band_10m.keys()) + list(per_band_20m.keys())
                        }
                    for band, metrics in {**per_band_10m, **per_band_20m}.items():
                        self.results[name]['per_sample_band'][band].append(metrics)

                except Exception as e:
                    if idx == 0:
                        print(f"\n  [WARNING] {name} failed on sample {idx}: {e}")
                    # Append NaN metrics to keep alignment
                    nan_metrics = {m: float('nan') for m in self.METRIC_NAMES}
                    self.results[name]['per_sample_10m'].append(nan_metrics)
                    self.results[name]['per_sample_20m'].append(nan_metrics)

            # Unload model to free GPU memory for next model
            self._unload_model(wrapper)
            print(f"  [{name}] Done — model unloaded")

        self._compute_aggregate_stats()

    def _compute_aggregate_stats(self):
        """Compute mean, std, CI for all metrics."""
        for name, result in self.results.items():
            for branch in ['10m', '20m']:
                key = f'stats_{branch}'
                result[key] = {}
                for metric in self.METRIC_NAMES:
                    values = np.array([s[metric] for s in result[f'per_sample_{branch}']])
                    valid = values[~np.isnan(values)]
                    n_valid = len(valid)
                    result[key][metric] = {
                        'mean': float(np.mean(valid)) if n_valid > 0 else 0.0,
                        'std': float(np.std(valid)) if n_valid > 0 else 0.0,
                        'ci95': float(1.96 * np.std(valid) / np.sqrt(n_valid)) if n_valid > 1 else 0.0,
                        'n_valid': n_valid,
                    }

            # Combined
            result['stats_combined'] = {}
            for metric in self.METRIC_NAMES:
                m10 = result['stats_10m'][metric]['mean']
                m20 = result['stats_20m'][metric]['mean']
                s10 = result['stats_10m'][metric]['std']
                s20 = result['stats_20m'][metric]['std']
                result['stats_combined'][metric] = {
                    'mean': (m10 + m20) / 2,
                    'std': np.sqrt((s10 ** 2 + s20 ** 2) / 2),
                }

    def print_results(self):
        """Print formatted results."""
        print(f"\n{'=' * 120}")
        print("SOTA BENCHMARK RESULTS")
        print(f"{'=' * 120}")

        # 10m branch
        self._print_branch_table('10m', 'RGB (10m) Branch — Blue, Green, Red')
        # 20m branch
        self._print_branch_table('20m', '20m Branch — Red Edge (B5), NIR (B8A)')
        # Combined
        self._print_combined_table()
        # Improvements over bilinear
        if 'Bilinear' in self.results:
            self._print_improvement_table()

    def _print_branch_table(self, branch: str, title: str):
        """Print table for a specific branch."""
        stats_key = f'stats_{branch}'
        print(f"\n{'─' * 120}")
        print(f"  {title}")
        print(f"{'─' * 120}")

        header = f"{'Model':<40} {'PSNR↑':>12} {'SSIM↑':>12} {'SCC↑':>12} {'RMSE↓':>12} {'ERGAS↓':>12} {'SID↓':>12}"
        print(header)
        print("─" * 120)

        # Find best values
        best = {}
        for metric in self.METRIC_NAMES:
            vals = [(n, r[stats_key][metric]['mean']) for n, r in self.results.items()]
            if metric in self.HIGHER_BETTER:
                best[metric] = max(vals, key=lambda x: x[1])[0]
            else:
                best[metric] = min(vals, key=lambda x: x[1])[0]

        for name, result in self.results.items():
            stats = result[stats_key]
            cells = []
            for metric in self.METRIC_NAMES:
                m = stats[metric]['mean']
                s = stats[metric]['std']
                is_best = best[metric] == name
                if metric == 'psnr':
                    txt = f"{m:.2f}±{s:.2f}"
                elif metric in ['ssim', 'scc']:
                    txt = f"{m:.4f}"
                else:
                    txt = f"{m:.4f}"
                if is_best:
                    txt = f"*{txt}"
                cells.append(txt)
            row = f"{name:<40} {cells[0]:>12} {cells[1]:>12} {cells[2]:>12} {cells[3]:>12} {cells[4]:>12} {cells[5]:>12}"
            print(row)

        print("  * = Best in column")

    def _print_combined_table(self):
        """Print combined metrics."""
        print(f"\n{'=' * 120}")
        print("  COMBINED METRICS (Average of 10m + 20m)")
        print(f"{'=' * 120}")

        header = f"{'Model':<40} {'PSNR↑':>12} {'SSIM↑':>12} {'SCC↑':>12} {'RMSE↓':>12} {'ERGAS↓':>12} {'SID↓':>12}"
        print(header)
        print("─" * 120)

        best = {}
        for metric in self.METRIC_NAMES:
            vals = [(n, r['stats_combined'][metric]['mean']) for n, r in self.results.items()]
            if metric in self.HIGHER_BETTER:
                best[metric] = max(vals, key=lambda x: x[1])[0]
            else:
                best[metric] = min(vals, key=lambda x: x[1])[0]

        for name, result in self.results.items():
            stats = result['stats_combined']
            cells = []
            for metric in self.METRIC_NAMES:
                m = stats[metric]['mean']
                is_best = best[metric] == name
                if metric == 'psnr':
                    txt = f"{m:.2f}"
                elif metric in ['ssim', 'scc']:
                    txt = f"{m:.4f}"
                else:
                    txt = f"{m:.4f}"
                if is_best:
                    txt = f"*{txt}"
                cells.append(txt)
            row = f"{name:<40} {cells[0]:>12} {cells[1]:>12} {cells[2]:>12} {cells[3]:>12} {cells[4]:>12} {cells[5]:>12}"
            print(row)

    def _print_improvement_table(self):
        """Print % improvement over bilinear baseline."""
        bilinear = self.results['Bilinear']
        print(f"\n{'=' * 120}")
        print("  IMPROVEMENT OVER BILINEAR (%) — Combined")
        print(f"{'=' * 120}")

        header = f"{'Model':<40} {'PSNR':>10} {'SSIM':>10} {'SCC':>10} {'RMSE':>10} {'ERGAS':>10} {'SID':>10}"
        print(header)
        print("─" * 120)

        for name, result in self.results.items():
            if name == 'Bilinear':
                continue
            cells = []
            for metric in self.METRIC_NAMES:
                base = bilinear['stats_combined'][metric]['mean']
                val = result['stats_combined'][metric]['mean']
                if abs(base) < 1e-10:
                    pct = 0.0
                elif metric in self.HIGHER_BETTER:
                    pct = ((val - base) / abs(base)) * 100
                else:
                    pct = ((base - val) / abs(base)) * 100  # Lower is better
                sign = "+" if pct > 0 else ""
                cells.append(f"{sign}{pct:.1f}%")
            row = f"{name:<40} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10} {cells[3]:>10} {cells[4]:>10} {cells[5]:>10}"
            print(row)

    def save_results(self, prefix: str = 'sota_benchmark'):
        """Save results to JSON and CSV."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON
        json_results = {}
        for name, result in self.results.items():
            json_results[name] = {
                'stats_10m': result['stats_10m'],
                'stats_20m': result['stats_20m'],
                'stats_combined': result['stats_combined'],
            }

        json_path = self.output_dir / f'{prefix}_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        # CSV
        rows = []
        for name, result in self.results.items():
            row = {'Model': name}
            for metric in self.METRIC_NAMES:
                row[f'{metric.upper()}_10m_mean'] = result['stats_10m'][metric]['mean']
                row[f'{metric.upper()}_10m_std'] = result['stats_10m'][metric]['std']
                row[f'{metric.upper()}_20m_mean'] = result['stats_20m'][metric]['mean']
                row[f'{metric.upper()}_20m_std'] = result['stats_20m'][metric]['std']
                row[f'{metric.upper()}_combined'] = result['stats_combined'][metric]['mean']
            rows.append(row)

        csv_path = self.output_dir / f'{prefix}_{timestamp}.csv'
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
        return json_path, csv_path

    def plot_comparison(self, prefix: str = 'sota_benchmark'):
        """Generate comparison bar charts."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_names = list(self.results.keys())

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('SOTA Benchmark — Combined Metrics', fontsize=16, fontweight='bold')

        for i, metric in enumerate(self.METRIC_NAMES):
            ax = axes[i // 3][i % 3]
            vals = [self.results[n]['stats_combined'][metric]['mean'] for n in model_names]
            colors = []
            for n in model_names:
                if 'Proposed' in n or 'Direct' in n or 'Residual' in n:
                    colors.append('#2196F3')  # Blue for ours
                elif n in ('Bilinear', 'Bicubic'):
                    colors.append('#9E9E9E')  # Gray for baseline
                else:
                    colors.append('#FF9800')  # Orange for SOTA

            bars = ax.barh(range(len(model_names)), vals, color=colors)
            ax.set_yticks(range(len(model_names)))
            ax.set_yticklabels(model_names, fontsize=8)
            direction = '↑' if metric in self.HIGHER_BETTER else '↓'
            ax.set_xlabel(f'{metric.upper()} {direction}')
            ax.set_title(f'{metric.upper()}')

        plt.tight_layout()
        plot_path = self.output_dir / f'{prefix}_{timestamp}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot: {plot_path}")

    @staticmethod
    def _normalize_for_display(img: np.ndarray, percentile: int = 98) -> np.ndarray:
        """Percentile-stretch satellite reflectance for visualization.
        Raw reflectance is typically 0.0–0.2, which looks black with default
        [0,1] mapping. Stretching to the 2nd–98th percentile range fixes this."""
        vmin = np.percentile(img, 100 - percentile)
        vmax = np.percentile(img, percentile)
        return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)

    def save_visual_samples(self, n_samples: int = 5, prefix: str = 'sota_visual'):
        """Save visual comparison for a few samples."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        vis_dir = self.output_dir / f'{prefix}_{timestamp}'
        vis_dir.mkdir(exist_ok=True)

        model_names = list(self.models.keys())
        n = min(n_samples, len(self.data))

        # Pre-compute predictions for each model (load one at a time)
        model_preds = {name: [] for name in model_names}  # name -> list of sr_10m per sample
        for name, wrapper in self.models.items():
            print(f"  Generating visual predictions: {name}")
            try:
                wrapper.load()
                for idx in range(n):
                    sample = self.data.get_sample(idx)
                    sr_10m = wrapper.predict_10m(sample, self.data)
                    model_preds[name].append(np.clip(sr_10m, 0, 1))
                self._unload_model(wrapper)
            except Exception as e:
                print(f"  [SKIP vis] {name}: {e}")
                for _ in range(n):
                    model_preds[name].append(None)

        # Now create figures
        for idx in range(n):
            sample = self.data.get_sample(idx)
            gt_rgb = sample['hr_10m'][:, :, ::-1]  # BGR -> RGB for display

            # Compute shared stretch from GT so all panels use the same scale
            vmin = np.percentile(gt_rgb, 2)
            vmax = np.percentile(gt_rgb, 98)

            def stretch(img_rgb):
                return np.clip((img_rgb - vmin) / (vmax - vmin + 1e-8), 0, 1)

            fig, axes = plt.subplots(1, len(model_names) + 2, figsize=(4 * (len(model_names) + 2), 4))

            # LR (upscaled for display)
            lr_rgb = sample['lr_10m_mid'][:, :, ::-1]
            axes[0].imshow(stretch(lr_rgb))
            axes[0].set_title('LR Input', fontsize=9)
            axes[0].axis('off')

            # GT
            axes[1].imshow(stretch(gt_rgb))
            axes[1].set_title('Ground Truth', fontsize=9)
            axes[1].axis('off')

            # Each model
            for j, name in enumerate(model_names):
                sr_10m = model_preds[name][idx]
                if sr_10m is not None:
                    sr_rgb = sr_10m[:, :, ::-1]  # BGR -> RGB
                    axes[j + 2].imshow(stretch(sr_rgb))
                    psnr = compute_psnr(sr_10m, sample['hr_10m'])
                    axes[j + 2].set_title(f'{name}\nPSNR={psnr:.1f}', fontsize=7)
                else:
                    axes[j + 2].text(0.5, 0.5, 'Failed', ha='center', va='center', transform=axes[j + 2].transAxes)
                    axes[j + 2].set_title(name, fontsize=7)
                axes[j + 2].axis('off')

            plt.suptitle(f'Sample {idx}', fontsize=12)
            plt.tight_layout()
            plt.savefig(vis_dir / f'sample_{idx:03d}.png', dpi=150, bbox_inches='tight')
            plt.close()

        print(f"  Visual samples saved to: {vis_dir}")


# =============================================================================
# MODEL REGISTRY
# =============================================================================

def get_available_models(device: torch.device) -> Dict[str, SOTAModelWrapper]:
    """Return all available model wrappers."""
    ckpt_dir = PROJECT_ROOT / 'checkpoints'

    models = {
        # --- Baselines ---
        'bilinear': BilinearWrapper(device),
        'bicubic': BicubicWrapper(device),

        # --- Proposed Models ---
        'proposed_direct': ProposedModelWrapper(
            'Proposed (Direct)',
            ckpt_dir / 'multires_multioutput_schedPer_t0.5_adaptive_direct_learnedUps_full' / 'best_model.pt',
            device
        ),
        'proposed_residual': ProposedModelWrapper(
            'Proposed (Residual)',
            ckpt_dir / 'multires_multioutput_schedPer_t0.5_adaptive_learnedUps_full' / 'best_model.pt',
            device
        ),

        # --- SOTA Models (pretrained weights, no retraining) ---
        'han': HANWrapper(device),
        'ldm': LDMWrapper(device, num_inference_steps=50),
        'opensr': OpenSRModelWrapper(device),
        'sen2sr': SEN2SRWrapper(device),
        'swin2mose': Swin2MoSEWrapper(device),
        'satlas': SatlasSRWrapper(device),
    }
    return models


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SOTA SR Benchmark')
    parser.add_argument('--test_dir', type=str,
                        default='data/sr_training/all_stas_combined/mmap_test',
                        help='Path to test mmap directory')
    parser.add_argument('--output_dir', type=str,
                        default='results/sota_benchmark',
                        help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit evaluation to N samples (for testing)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to evaluate (default: all available)')
    parser.add_argument('--no_viz', action='store_true',
                        help='Skip visual sample generation')
    parser.add_argument('--n_viz', type=int, default=5,
                        help='Number of visual samples to save')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, auto-detect if omitted)')

    args = parser.parse_args()

    # Resolve paths
    test_dir = PROJECT_ROOT / args.test_dir
    output_dir = PROJECT_ROOT / args.output_dir

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create benchmark
    benchmark = SOTABenchmark(test_dir, output_dir, device)

    # Register models
    available = get_available_models(device)

    if args.models:
        # Use only specified models
        for key in args.models:
            if key in available:
                benchmark.add_model(available[key])
            else:
                print(f"  [WARNING] Unknown model key: {key}")
                print(f"  Available: {', '.join(available.keys())}")
    else:
        # Use all available models
        for wrapper in available.values():
            benchmark.add_model(wrapper)

    # Load models (will skip those that fail)
    benchmark.load_all_models()

    # Run evaluation
    benchmark.evaluate(max_samples=args.max_samples)

    # Print & save results
    benchmark.print_results()
    benchmark.save_results()
    benchmark.plot_comparison()

    if not args.no_viz:
        benchmark.save_visual_samples(n_samples=args.n_viz)

    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()
