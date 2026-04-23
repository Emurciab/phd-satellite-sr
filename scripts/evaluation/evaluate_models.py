#!/usr/bin/env python3
"""
Comprehensive Multi-Model Super-Resolution Evaluation Script

This script provides:
1. Fair comparison across multiple models
2. Full metrics suite (PSNR, SSIM, SCC, RMSE, ERGAS, SID)
3. Per-branch (10m, 20m) and combined metrics
4. Statistical analysis with confidence intervals
5. Visualization of metrics and sample outputs
6. Exportable results (JSON, CSV)

Usage:
    python evaluate_models.py --models MODEL1 MODEL2 --test_dir PATH
    python evaluate_models.py --config config.json
"""

import sys
import os

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from scipy.ndimage import uniform_filter, laplace, zoom
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training' / 'models'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training' / 'variants' / 'frequency'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training' / 'variants' / 'GAN'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training' / 'variants' / 'multioutput_schedper'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training' / 'variants' / 'ablationstudy'))


# =============================================================================
# MODEL LOADING
# =============================================================================

def detect_architecture_version(state_dict: dict) -> str:
    """
    Detect checkpoint architecture version based on key patterns.
    
    Architecture versions:
    - 'freq_residual': Frequency domain SR with residual learning (hf_predictor)
    - 'freq_direct': Frequency domain SR with direct learning (spectrum_predictor)
    - 'upscaler_old': Has upscaler_* + gamma (old QKV temporal attn) - variants/bestframe/satellite_sr_model_direct_old.py
    - 'upscaler': Has upscaler_*, spectral_consistency_*, no gamma (new temporal attn) - variants/*/satellite_sr_model_direct.py
    - 'decoder': Has decoder_*, fusion_*.attention (legacy/models/satellite_sr_model_direct_old.py)
    - 'new': Has frame_encoder (legacy/models/satellite_sr_model_direct.py)
    - 'v2': Standard V2 residual model
    """
    keys = list(state_dict.keys())
    
    has_upscaler = any('upscaler_10m' in k or 'upscaler_20m' in k for k in keys)
    has_spectral_consistency = any('spectral_consistency' in k for k in keys)
    has_decoder = any('decoder_10m' in k or 'decoder_20m' in k for k in keys)
    has_frame_encoder = any('frame_encoder' in k for k in keys)
    has_fusion_from = any('fusion_10m_from_20m' in k or 'fusion_20m_from_10m' in k for k in keys)
    has_gamma = any('gamma' in k for k in keys)
    
    # Frequency domain models
    has_hf_predictor = any('hf_predictor' in k for k in keys)
    has_spectrum_predictor = any('spectrum_predictor' in k for k in keys)
    
    # Frequency Residual model (baseline + HF residuals)
    if has_hf_predictor:
        return 'freq_residual'
    
    # Frequency Direct model (full spectrum prediction)
    if has_spectrum_predictor:
        return 'freq_direct'
    
    # Upscaler-style Direct with OLD temporal attention (gamma/QKV)
    if has_upscaler and has_spectral_consistency and has_gamma:
        return 'upscaler_old'
    
    # Upscaler-style Direct with NEW temporal attention (frame_encoder style)
    if has_upscaler and has_spectral_consistency and has_fusion_from:
        return 'upscaler'
    
    # Decoder-style with fusion.attention (legacy/models/satellite_sr_model_direct_old.py)
    if has_decoder and has_gamma:
        return 'decoder'
    
    # New style with frame_encoder (legacy/models/satellite_sr_model_direct.py)
    if has_frame_encoder:
        return 'new'
    
    # Default fallback
    return 'v2'


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    """
    Load a trained model from checkpoint.
    Automatically detects model architecture from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try to detect model type from checkpoint or path
    model_name = checkpoint_path.parent.name.lower()
    
    # Detect architecture version from state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('generator_state_dict', checkpoint))
    arch_version = detect_architecture_version(state_dict)
    
    # Default architecture parameters
    model_kwargs = {
        'in_channels_10m': 3,
        'in_channels_20m': 2,
        'base_channels': 64,
        'num_encoder_blocks': 4,
        'num_frames': 3,
        'scale_factor_10m': 3.33,
        'scale_factor_20m': 3.33,
    }
    
    # Override with config from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        if 'base_channels' in config:
            model_kwargs['base_channels'] = config['base_channels']
        if 'num_encoder_blocks' in config:
            model_kwargs['num_encoder_blocks'] = config['num_encoder_blocks']
    
    # Load appropriate model class based on architecture type
    is_direct = 'direct' in model_name
    is_schedper = 'schedper' in model_name
    is_temporal = 'temporal' in model_name and 'multioutput' not in model_name
    
    is_learnedups = 'learnedups' in model_name
    
    if 'ablation' in model_name:
        # Ablation study models — all based on LearnedUps architecture
        config = checkpoint.get('config', {})
        resize_method = checkpoint.get('resize_method', config.get('resize_method', 'adaptive'))
        model_kwargs['num_temporal_blocks'] = 2
        model_kwargs['resize_method'] = resize_method

        if 'no_temporal_no_fusion' in model_name:
            # Custom subclass with TemporalIdentity + skipped fusion
            if is_direct:
                from train_ablation_no_temporal_no_fusion_direct import AblatedNoTemporalNoFusionDirect as ModelClass
            else:
                from train_ablation_no_temporal_no_fusion_residual import AblatedNoTemporalNoFusionResidual as ModelClass
            model = ModelClass(**model_kwargs)
        elif 'no_se_module' in model_name or 'no_spectral' in model_name:
            # Full model with SE modules replaced by Identity
            import torch.nn as nn
            if is_direct:
                from train_multires_temporal_multiOutput_schedPer_direct_learnedUps import MultiOutputTemporalSRDirectLearnedUps as ModelClass
            else:
                from train_multires_temporal_multiOutput_schedPer_learnedUps import MultiOutputTemporalSRLearnedUps as ModelClass
            model = ModelClass(**model_kwargs)
            model.spectral_10m = nn.Identity()
            model.spectral_20m = nn.Identity()
        else:
            # No SAM Loss and other loss-only ablations — same architecture as full model
            if is_direct:
                from train_multires_temporal_multiOutput_schedPer_direct_learnedUps import MultiOutputTemporalSRDirectLearnedUps as ModelClass
            else:
                from train_multires_temporal_multiOutput_schedPer_learnedUps import MultiOutputTemporalSRLearnedUps as ModelClass
            model = ModelClass(**model_kwargs)
    elif 'multioutput' in model_name:
        # Multi-output temporal model
        if is_learnedups:
            # Learned upscaling variants (adaptive/meta resize methods)
            config = checkpoint.get('config', {})
            resize_method = checkpoint.get('resize_method', config.get('resize_method', 'adaptive'))
            model_kwargs['num_temporal_blocks'] = 2
            model_kwargs['resize_method'] = resize_method
            if is_direct:
                from train_multires_temporal_multiOutput_schedPer_direct_learnedUps import MultiOutputTemporalSRDirectLearnedUps as ModelClass
            else:
                from train_multires_temporal_multiOutput_schedPer_learnedUps import MultiOutputTemporalSRLearnedUps as ModelClass
            model = ModelClass(**model_kwargs)
        elif is_direct and is_schedper:
            # SchedPer direct uses inline model from training script
            from legacy.train_multires_temporal_multiOutput_schedPer_direct import MultiOutputTemporalSRDirect as ModelClass
        elif is_direct:
            from legacy.models.satellite_sr_model_direct import MultiOutputTemporalSRDirect as ModelClass
        elif is_schedper:
            from legacy.train_multires_temporal_multiOutput_schedPer import MultiOutputTemporalSR as ModelClass
        else:
            from legacy.train_multires_temporal_multiOutput import MultiOutputTemporalSR as ModelClass
        # Add num_temporal_blocks for multioutput (if not already set by learnedups)
        if 'num_temporal_blocks' not in model_kwargs:
            model_kwargs['num_temporal_blocks'] = 2
        if not is_learnedups:
            model = ModelClass(**model_kwargs)
    elif is_temporal:
        # Temporal supervision model (supervises all frames, not multioutput)
        if is_direct:
            # Check architecture version for Direct models
            if arch_version == 'upscaler':
                # Uses variants/temporal/satellite_sr_model_direct.py style
                from variants.temporal.satellite_sr_model_direct import MultiResolutionSatelliteSRDirect as ModelClass
            elif arch_version == 'decoder':
                from legacy.models.satellite_sr_model_direct_old import MultiResolutionSatelliteSRDirect as ModelClass
            else:
                from legacy.models.satellite_sr_model_direct import MultiResolutionSatelliteSRDirect as ModelClass
        else:
            from legacy.models.satellite_sr_model_v2 import MultiResolutionSatelliteSRv2 as ModelClass
        model = ModelClass(**model_kwargs)
    elif 'freq' in model_name:
        # Frequency domain models
        if arch_version == 'freq_residual' or 'residual' in model_name:
            # Frequency Residual model (bilinear baseline + learned HF)
            from variants.frequency.satellite_sr_model_freq_residual import FrequencyDomainSRResidual as ModelClass
        else:
            # Frequency Direct model (full spectrum prediction)
            from variants.frequency.satellite_sr_model_freq_direct import FrequencyDomainSRDirect as ModelClass
        model = ModelClass(**model_kwargs)
    elif 'median' in model_name or 'bestframe' in model_name:
        # Median and BestFrame models
        if is_direct:
            # Check architecture version for Direct models
            if arch_version == 'upscaler_old':
                # OLD: upscaler + gamma/QKV temporal attention
                from variants.bestframe.satellite_sr_model_direct_old import MultiResolutionSatelliteSRDirect as ModelClass
            elif arch_version == 'upscaler':
                # NEW: upscaler + frame_encoder temporal attention
                from variants.bestframe.satellite_sr_model_direct import MultiResolutionSatelliteSRDirect as ModelClass
            elif arch_version == 'decoder':
                from legacy.models.satellite_sr_model_direct_old import MultiResolutionSatelliteSRDirect as ModelClass
            else:
                from legacy.models.satellite_sr_model_direct import MultiResolutionSatelliteSRDirect as ModelClass
        else:
            from legacy.models.satellite_sr_model_v2 import MultiResolutionSatelliteSRv2 as ModelClass
        model = ModelClass(**model_kwargs)
    elif 'gan' in model_name:
        # GAN models - both use models from variants/GAN folder
        # Check model_type in checkpoint to determine architecture
        model_type = checkpoint.get('model_type', '')
        is_direct = 'direct' in model_name or model_type == 'gan_direct'
        if is_direct:
            # GAN Direct uses satellite_sr_model_direct from GAN variants
            from variants.GAN.satellite_sr_model_direct import MultiResolutionSatelliteSRDirect as ModelClass
        else:
            # GAN Residual uses satellite_sr_model_v2 from GAN variants (has stem encoder)
            from variants.GAN.satellite_sr_model_v2 import MultiResolutionSatelliteSRv2 as ModelClass
        model = ModelClass(**model_kwargs)
    elif 'v3' in model_name or 'goodonly' in model_name:
        from legacy.models.satellite_sr_model_v3 import MultiResolutionSatelliteSRv3 as ModelClass
        model = ModelClass(**model_kwargs)
    else:
        # Default to MultiRes V2
        from legacy.models.satellite_sr_model_v2 import MultiResolutionSatelliteSRv2 as ModelClass
        model = ModelClass(**model_kwargs)
    
    # Load weights - handle GAN checkpoint format (generator_state_dict vs model_state_dict)
    if 'generator_state_dict' in checkpoint:
        # GAN model checkpoint
        model.load_state_dict(checkpoint['generator_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


# =============================================================================
# QUALITY METRICS (same as reference)
# =============================================================================

def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    max_val = max(pred.max(), target.max(), 1.0)
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Root Mean Square Error."""
    return np.sqrt(np.mean((pred - target) ** 2))


def compute_ssim(pred: np.ndarray, target: np.ndarray, win_size: int = 7) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    Multi-channel images are handled by averaging per-channel SSIM.
    """
    if pred.ndim == 3:
        ssim_values = []
        for c in range(pred.shape[-1]):
            ssim_values.append(compute_ssim(pred[:, :, c], target[:, :, c], win_size))
        return np.mean(ssim_values)
    
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    mu_pred = uniform_filter(pred.astype(np.float64), size=win_size, mode='reflect')
    mu_target = uniform_filter(target.astype(np.float64), size=win_size, mode='reflect')
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = uniform_filter(pred.astype(np.float64) ** 2, size=win_size, mode='reflect') - mu_pred_sq
    sigma_target_sq = uniform_filter(target.astype(np.float64) ** 2, size=win_size, mode='reflect') - mu_target_sq
    sigma_pred_target = uniform_filter(pred.astype(np.float64) * target.astype(np.float64), size=win_size, mode='reflect') - mu_pred_target
    
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    
    ssim_map = numerator / (denominator + 1e-10)
    return np.mean(ssim_map)


def compute_scc(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Spatial Correlation Coefficient (SCC).
    Measures correlation of high-frequency spatial details.
    """
    if pred.ndim == 3:
        scc_values = []
        for c in range(pred.shape[-1]):
            scc_values.append(compute_scc(pred[:, :, c], target[:, :, c]))
        return np.mean(scc_values)
    
    pred_hf = laplace(pred.astype(np.float64))
    target_hf = laplace(target.astype(np.float64))
    
    pred_flat = pred_hf.flatten()
    target_flat = target_hf.flatten()
    
    pred_centered = pred_flat - np.mean(pred_flat)
    target_centered = target_flat - np.mean(target_flat)
    
    numerator = np.sum(pred_centered * target_centered)
    denominator = np.sqrt(np.sum(pred_centered ** 2) * np.sum(target_centered ** 2))
    
    if denominator < 1e-10:
        return 1.0 if numerator < 1e-10 else 0.0
    
    return numerator / denominator


def compute_ergas(pred: np.ndarray, target: np.ndarray, scale_ratio: float) -> float:
    """
    Compute ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse).
    Lower is better (0 = perfect reconstruction).
    """
    if pred.ndim == 2:
        pred = pred[:, :, np.newaxis]
        target = target[:, :, np.newaxis]
    
    num_bands = pred.shape[-1]
    sum_term = 0.0
    
    for b in range(num_bands):
        pred_band = pred[:, :, b]
        target_band = target[:, :, b]
        
        rmse_band = np.sqrt(np.mean((pred_band - target_band) ** 2))
        mean_target = np.mean(target_band)
        
        if mean_target > 1e-10:
            sum_term += (rmse_band / mean_target) ** 2
    
    ergas = 100 * (1 / scale_ratio) * np.sqrt(sum_term / num_bands)
    return ergas


def compute_sid(pred: np.ndarray, target: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Spectral Information Divergence (SID).
    Lower is better (0 = identical spectral signatures).
    """
    if pred.ndim == 2:
        return 0.0
    
    pred = np.clip(pred, epsilon, None)
    target = np.clip(target, epsilon, None)
    
    pred_sum = np.sum(pred, axis=-1, keepdims=True)
    target_sum = np.sum(target, axis=-1, keepdims=True)
    
    p = pred / (pred_sum + epsilon)
    q = target / (target_sum + epsilon)
    
    kl_pq = np.sum(p * np.log((p + epsilon) / (q + epsilon)), axis=-1)
    kl_qp = np.sum(q * np.log((q + epsilon) / (p + epsilon)), axis=-1)
    
    sid = kl_pq + kl_qp
    return np.mean(sid)


def compute_all_metrics(pred: np.ndarray, target: np.ndarray, scale_ratio: float = 3.33) -> Dict[str, float]:
    """Compute all quality metrics."""
    return {
        'psnr': compute_psnr(pred, target),
        'ssim': compute_ssim(pred, target),
        'scc': compute_scc(pred, target),
        'rmse': compute_rmse(pred, target),
        'ergas': compute_ergas(pred, target, scale_ratio),
        'sid': compute_sid(pred, target)
    }


def compute_per_band_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each spectral band individually.
    
    Args:
        pred: (H, W, C) predicted image
        target: (H, W, C) ground truth image
    
    Returns:
        Dictionary with band names as keys and metrics dict as values
    """
    if pred.ndim != 3:
        return {}
    
    num_bands = pred.shape[-1]
    results = {}
    
    # Band names based on number of channels
    if num_bands == 3:
        band_names = ['Blue', 'Green', 'Red']
    elif num_bands == 2:
        band_names = ['RedEdge', 'NIR']
    else:
        band_names = [f'Band_{i}' for i in range(num_bands)]
    
    for i, band_name in enumerate(band_names):
        pred_band = pred[:, :, i]
        target_band = target[:, :, i]
        
        results[band_name] = {
            'psnr': compute_psnr(pred_band, target_band),
            'ssim': compute_ssim(pred_band, target_band),
            'scc': compute_scc(pred_band, target_band),
            'rmse': compute_rmse(pred_band, target_band),
        }
    
    return results


# =============================================================================
# DATA LOADING
# =============================================================================

class TestDataLoader:
    """Load test data from mmap format."""
    
    def __init__(self, test_dir: Path, sequence_length: int = 3, skip_indices: List[int] = None):
        self.test_dir = Path(test_dir)
        self.sequence_length = sequence_length
        self.skip_indices = set(skip_indices) if skip_indices else set()
        
        # Load mmap arrays
        self.lr_10m = np.load(self.test_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(self.test_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr_10m = np.load(self.test_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(self.test_dir / 'hr_20m.npy', mmap_mode='r')
        
        # Create valid indices list (excluding bad samples)
        total_samples = self.lr_10m.shape[0]
        self.valid_indices = [i for i in range(total_samples) if i not in self.skip_indices]
        self.n_samples = len(self.valid_indices)
        
        if self.skip_indices:
            print(f"  Skipping {len(self.skip_indices)} bad samples: {sorted(self.skip_indices)}")
            print(f"  Using {self.n_samples} valid samples out of {total_samples}")
    
    def __len__(self):
        return self.n_samples
    
    def get_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample by index (maps to valid indices)."""
        actual_idx = self.valid_indices[idx]
        
        lr_10m = np.array(self.lr_10m[actual_idx])
        lr_20m = np.array(self.lr_20m[actual_idx])
        hr_10m = np.array(self.hr_10m[actual_idx])
        hr_20m = np.array(self.hr_20m[actual_idx])
        
        mid_idx = self.sequence_length // 2
        
        return {
            'lr_10m_seq': lr_10m,  # (T, H, W, C)
            'lr_20m_seq': lr_20m,  # (T, H, W, C)
            'lr_10m': lr_10m[mid_idx],  # Middle frame for display
            'lr_20m': lr_20m[mid_idx],
            'hr_10m': hr_10m[mid_idx],  # Ground truth (middle frame)
            'hr_20m': hr_20m[mid_idx],
        }
    
    def prepare_for_model(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Convert sample to model input tensors."""
        # (T, H, W, C) -> (1, T, C, H, W)
        lr_10m_t = torch.from_numpy(sample['lr_10m_seq']).permute(0, 3, 1, 2).unsqueeze(0).float()
        lr_20m_t = torch.from_numpy(sample['lr_20m_seq']).permute(0, 3, 1, 2).unsqueeze(0).float()
        
        return {
            'lr_10m': lr_10m_t,
            'lr_20m': lr_20m_t,
            'target_size_10m': sample['hr_10m'].shape[:2],
            'target_size_20m': sample['hr_20m'].shape[:2],
        }


# =============================================================================
# MODEL INFERENCE
# =============================================================================

def run_inference(model: torch.nn.Module, 
                  model_inputs: Dict[str, torch.Tensor],
                  device: torch.device) -> Dict[str, np.ndarray]:
    """Run model inference and return predictions."""
    with torch.no_grad():
        lr_10m = model_inputs['lr_10m'].to(device)
        lr_20m = model_inputs['lr_20m'].to(device)
        
        # Check if model supports target_size parameters
        try:
            sr_10m, sr_20m = model(
                lr_10m, lr_20m,
                target_size_10m=model_inputs['target_size_10m'],
                target_size_20m=model_inputs['target_size_20m']
            )
        except TypeError:
            # Model doesn't support target_size, run without
            sr_10m, sr_20m = model(lr_10m, lr_20m)
        
        # Check if output is multi-temporal (B, T, C, H, W) or single frame (B, C, H, W)
        if sr_10m.dim() == 5:
            # Multi-output temporal model: (B, T, C, H, W) -> use middle frame
            middle_idx = sr_10m.shape[1] // 2
            sr_10m = sr_10m[:, middle_idx]  # (B, C, H, W)
            sr_20m = sr_20m[:, middle_idx]  # (B, C, H, W)
        
        # Convert to numpy: (1, C, H, W) -> (H, W, C)
        sr_10m = sr_10m.cpu().numpy()[0].transpose(1, 2, 0)
        sr_20m = sr_20m.cpu().numpy()[0].transpose(1, 2, 0)
    
    return {'sr_10m': sr_10m, 'sr_20m': sr_20m}


def compute_bilinear_baseline(sample: Dict) -> Dict[str, np.ndarray]:
    """Compute bilinear interpolation baseline."""
    target_h_10m, target_w_10m = sample['hr_10m'].shape[:2]
    target_h_20m, target_w_20m = sample['hr_20m'].shape[:2]
    
    bilinear_10m = F.interpolate(
        torch.tensor(sample['lr_10m']).permute(2, 0, 1).unsqueeze(0).float(),
        size=(target_h_10m, target_w_10m),
        mode='bilinear', align_corners=False
    )[0].permute(1, 2, 0).numpy()
    
    bilinear_20m = F.interpolate(
        torch.tensor(sample['lr_20m']).permute(2, 0, 1).unsqueeze(0).float(),
        size=(target_h_20m, target_w_20m),
        mode='bilinear', align_corners=False
    )[0].permute(1, 2, 0).numpy()
    
    return {'sr_10m': bilinear_10m, 'sr_20m': bilinear_20m}


# =============================================================================
# EVALUATION ENGINE
# =============================================================================

class ModelEvaluator:
    """Evaluates multiple models on the same test data."""
    
    METRIC_NAMES = ['psnr', 'ssim', 'scc', 'rmse', 'ergas', 'sid']
    HIGHER_BETTER = ['psnr', 'ssim', 'scc']  # For these, higher is better
    
    def __init__(self, test_dir: Path, output_dir: Path, device: Optional[torch.device] = None,
                 skip_indices: List[int] = None):
        self.test_data = TestDataLoader(test_dir, skip_indices=skip_indices)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store results
        self.results: Dict[str, Dict] = {}
        
    def add_model(self, name: str, checkpoint_path: Path):
        """Add a model to evaluate."""
        print(f"Loading model: {name}")
        model, checkpoint = load_model(checkpoint_path, self.device)
        
        self.results[name] = {
            'model': model,
            'checkpoint': checkpoint,
            'checkpoint_path': str(checkpoint_path),
            'per_sample_10m': [],
            'per_sample_20m': [],
        }
        
        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'Unknown')
        n_params = sum(p.numel() for p in model.parameters())
        val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
        print(f"  → Epoch: {epoch}, Val Loss: {val_loss_str}, Params: {n_params:,}")
    
    def add_bilinear_baseline(self):
        """Add bilinear interpolation baseline."""
        self.results['Bilinear'] = {
            'model': None,  # Special case
            'checkpoint': None,
            'checkpoint_path': None,
            'per_sample_10m': [],
            'per_sample_20m': [],
        }
    
    def evaluate_all(self, max_samples: Optional[int] = None):
        """Evaluate all models on test data."""
        n_samples = len(self.test_data) if max_samples is None else min(max_samples, len(self.test_data))
        
        print(f"\nEvaluating {len(self.results)} models on {n_samples} test samples...")
        print("=" * 70)
        
        for idx in tqdm(range(n_samples), desc="Evaluating"):
            sample = self.test_data.get_sample(idx)
            model_inputs = self.test_data.prepare_for_model(sample)
            
            gt_10m = sample['hr_10m']
            gt_20m = sample['hr_20m']
            
            for name, result in self.results.items():
                if name == 'Bilinear':
                    preds = compute_bilinear_baseline(sample)
                else:
                    preds = run_inference(result['model'], model_inputs, self.device)
                
                # Compute metrics
                metrics_10m = compute_all_metrics(preds['sr_10m'], gt_10m, scale_ratio=3.33)
                metrics_20m = compute_all_metrics(preds['sr_20m'], gt_20m, scale_ratio=3.33)
                
                # Compute per-band metrics
                per_band_10m = compute_per_band_metrics(preds['sr_10m'], gt_10m)
                per_band_20m = compute_per_band_metrics(preds['sr_20m'], gt_20m)
                
                result['per_sample_10m'].append(metrics_10m)
                result['per_sample_20m'].append(metrics_20m)
                
                # Store per-band metrics
                if 'per_sample_band' not in result:
                    result['per_sample_band'] = {band: [] for band in list(per_band_10m.keys()) + list(per_band_20m.keys())}
                for band, metrics in per_band_10m.items():
                    result['per_sample_band'][band].append(metrics)
                for band, metrics in per_band_20m.items():
                    result['per_sample_band'][band].append(metrics)
        
        # Compute aggregate statistics
        self._compute_aggregate_stats()
    
    def _compute_aggregate_stats(self):
        """Compute mean, std, and confidence intervals for all metrics."""
        for name, result in self.results.items():
            # Aggregate 10m metrics
            result['stats_10m'] = {}
            for metric in self.METRIC_NAMES:
                values = np.array([s[metric] for s in result['per_sample_10m']])
                valid_count = np.sum(~np.isnan(values))
                result['stats_10m'][metric] = {
                    'mean': float(np.nanmean(values)),
                    'std': float(np.nanstd(values)),
                    'min': float(np.nanmin(values)) if valid_count > 0 else 0.0,
                    'max': float(np.nanmax(values)) if valid_count > 0 else 0.0,
                    'ci95': float(1.96 * np.nanstd(values) / np.sqrt(valid_count)) if valid_count > 0 else 0.0
                }
            
            # Aggregate 20m metrics
            result['stats_20m'] = {}
            for metric in self.METRIC_NAMES:
                values = np.array([s[metric] for s in result['per_sample_20m']])
                valid_count = np.sum(~np.isnan(values))
                result['stats_20m'][metric] = {
                    'mean': float(np.nanmean(values)),
                    'std': float(np.nanstd(values)),
                    'min': float(np.nanmin(values)) if valid_count > 0 else 0.0,
                    'max': float(np.nanmax(values)) if valid_count > 0 else 0.0,
                    'ci95': float(1.96 * np.nanstd(values) / np.sqrt(valid_count)) if valid_count > 0 else 0.0
                }
            
            # Combined metrics (average of 10m and 20m)
            result['stats_combined'] = {}
            for metric in self.METRIC_NAMES:
                mean_10m = result['stats_10m'][metric]['mean']
                mean_20m = result['stats_20m'][metric]['mean']
                std_10m = result['stats_10m'][metric]['std']
                std_20m = result['stats_20m'][metric]['std']
                
                # Handle NaN values in combined metrics
                if np.isnan(mean_10m) or np.isnan(mean_20m):
                    combined_mean = np.nanmean([mean_10m, mean_20m])
                else:
                    combined_mean = (mean_10m + mean_20m) / 2
                    
                if np.isnan(std_10m) or np.isnan(std_20m):
                    combined_std = np.nanmean([std_10m, std_20m])
                else:
                    combined_std = np.sqrt((std_10m**2 + std_20m**2) / 2)
                
                result['stats_combined'][metric] = {
                    'mean': float(combined_mean) if not np.isnan(combined_mean) else 0.0,
                    'std': float(combined_std) if not np.isnan(combined_std) else 0.0,
                }
            
            # Aggregate per-band metrics
            result['stats_per_band'] = {}
            if 'per_sample_band' in result:
                for band_name, samples in result['per_sample_band'].items():
                    result['stats_per_band'][band_name] = {}
                    for metric in ['psnr', 'ssim', 'scc', 'rmse']:
                        values = np.array([s[metric] for s in samples])
                        result['stats_per_band'][band_name][metric] = {
                            'mean': float(np.nanmean(values)),
                            'std': float(np.nanstd(values)),
                        }
    
    def get_summary_table(self) -> pd.DataFrame:
        """Generate summary table of all models and metrics."""
        rows = []
        for name, result in self.results.items():
            row = {'Model': name}
            for metric in self.METRIC_NAMES:
                row[f'{metric.upper()}_10m'] = result['stats_10m'][metric]['mean']
                row[f'{metric.upper()}_20m'] = result['stats_20m'][metric]['mean']
                row[f'{metric.upper()}_combined'] = result['stats_combined'][metric]['mean']
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def print_results(self):
        """Print formatted results to console."""
        print("\n" + "=" * 100)
        print("EVALUATION RESULTS")
        print("=" * 100)
        
        # Combined metrics table
        print("\n┌" + "─" * 98 + "┐")
        print("│" + " COMBINED METRICS (Average of 10m + 20m branches)".center(98) + "│")
        print("├" + "─" * 30 + "┬" + "─" * 11 + "┬" + "─" * 11 + "┬" + "─" * 11 + "┬" + "─" * 11 + "┬" + "─" * 11 + "┬" + "─" * 11 + "┤")
        header = "│" + " Model".ljust(29) + " │"
        for metric in self.METRIC_NAMES:
            direction = "↑" if metric in self.HIGHER_BETTER else "↓"
            header += f" {metric.upper()}{direction} ".center(10) + "│"
        print(header)
        print("├" + "─" * 30 + "┼" + "─" * 11 + "┼" + "─" * 11 + "┼" + "─" * 11 + "┼" + "─" * 11 + "┼" + "─" * 11 + "┼" + "─" * 11 + "┤")
        
        # Find best for each metric
        best_values = {}
        for metric in self.METRIC_NAMES:
            values = [(name, result['stats_combined'][metric]['mean']) 
                      for name, result in self.results.items()]
            if metric in self.HIGHER_BETTER:
                best_values[metric] = max(values, key=lambda x: x[1])[0]
            else:
                best_values[metric] = min(values, key=lambda x: x[1])[0]
        
        for name, result in self.results.items():
            row = "│" + f" {name}".ljust(29) + " │"
            for metric in self.METRIC_NAMES:
                val = result['stats_combined'][metric]['mean']
                is_best = best_values[metric] == name
                
                if metric == 'psnr':
                    cell = f"{val:.2f}"
                elif metric in ['ssim', 'scc']:
                    cell = f"{val:.4f}"
                elif metric in ['rmse', 'ergas', 'sid']:
                    cell = f"{val:.4f}"
                
                if is_best:
                    cell = f"*{cell}*"
                row += cell.center(10) + "│"
            print(row)
        
        print("└" + "─" * 30 + "┴" + "─" * 11 + "┴" + "─" * 11 + "┴" + "─" * 11 + "┴" + "─" * 11 + "┴" + "─" * 11 + "┴" + "─" * 11 + "┘")
        print("* = Best in column")
        
        # Per-branch details
        print("\n" + "─" * 100)
        print("10m BRANCH (Blue, Green, Red)")
        print("─" * 100)
        self._print_branch_table('10m')
        
        print("\n" + "─" * 100)
        print("20m BRANCH (RedEdge, NIR)")
        print("─" * 100)
        self._print_branch_table('20m')
        
        # Per-band metrics
        print("\n" + "=" * 100)
        print("PER-BAND METRICS (Individual Spectral Bands)")
        print("=" * 100)
        self._print_per_band_table()
        
        # Improvement over bilinear
        if 'Bilinear' in self.results:
            print("\n" + "=" * 100)
            print("IMPROVEMENT OVER BILINEAR BASELINE")
            print("=" * 100)
            self._print_improvements()
    
    def _print_branch_table(self, branch: str):
        """Print table for a specific branch."""
        stats_key = f'stats_{branch}'
        
        print(f"{'Model':<30} | {'PSNR':>10} | {'SSIM':>10} | {'SCC':>10} | {'ERGAS':>10} | {'SID':>10}")
        print("-" * 100)
        
        for name, result in self.results.items():
            stats = result[stats_key]
            psnr = f"{stats['psnr']['mean']:.2f}±{stats['psnr']['std']:.2f}"
            ssim = f"{stats['ssim']['mean']:.4f}"
            scc = f"{stats['scc']['mean']:.4f}"
            ergas = f"{stats['ergas']['mean']:.2f}"
            sid = f"{stats['sid']['mean']:.4f}"
            print(f"{name:<30} | {psnr:>10} | {ssim:>10} | {scc:>10} | {ergas:>10} | {sid:>10}")
    
    def _print_per_band_table(self):
        """Print per-band metrics for all models."""
        band_names = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR']
        metrics = ['psnr', 'ssim', 'scc', 'rmse']
        
        # Print PSNR per band
        print("\n📊 PSNR (dB) per Band - Higher is Better")
        print("-" * 110)
        header = f"{'Model':<30} |"
        for band in band_names:
            header += f" {band:>12} |"
        print(header)
        print("-" * 110)
        
        for name, result in self.results.items():
            row = f"{name:<30} |"
            if 'stats_per_band' in result:
                for band in band_names:
                    if band in result['stats_per_band']:
                        val = result['stats_per_band'][band]['psnr']['mean']
                        std = result['stats_per_band'][band]['psnr']['std']
                        row += f" {val:>5.2f}±{std:<4.2f} |"
                    else:
                        row += f" {'N/A':>12} |"
            else:
                row += " N/A" * len(band_names)
            print(row)
        
        # Print SSIM per band
        print("\n📊 SSIM per Band - Higher is Better")
        print("-" * 110)
        header = f"{'Model':<30} |"
        for band in band_names:
            header += f" {band:>12} |"
        print(header)
        print("-" * 110)
        
        for name, result in self.results.items():
            row = f"{name:<30} |"
            if 'stats_per_band' in result:
                for band in band_names:
                    if band in result['stats_per_band']:
                        val = result['stats_per_band'][band]['ssim']['mean']
                        row += f" {val:>12.4f} |"
                    else:
                        row += f" {'N/A':>12} |"
            else:
                row += " N/A" * len(band_names)
            print(row)
        
        # Print SCC per band
        print("\n📊 SCC (Spatial Correlation) per Band - Higher is Better")
        print("-" * 110)
        header = f"{'Model':<30} |"
        for band in band_names:
            header += f" {band:>12} |"
        print(header)
        print("-" * 110)
        
        for name, result in self.results.items():
            row = f"{name:<30} |"
            if 'stats_per_band' in result:
                for band in band_names:
                    if band in result['stats_per_band']:
                        val = result['stats_per_band'][band]['scc']['mean']
                        row += f" {val:>12.4f} |"
                    else:
                        row += f" {'N/A':>12} |"
            else:
                row += " N/A" * len(band_names)
            print(row)
        
        # Print RMSE per band
        print("\n📊 RMSE per Band - Lower is Better")
        print("-" * 110)
        header = f"{'Model':<30} |"
        for band in band_names:
            header += f" {band:>12} |"
        print(header)
        print("-" * 110)
        
        for name, result in self.results.items():
            row = f"{name:<30} |"
            if 'stats_per_band' in result:
                for band in band_names:
                    if band in result['stats_per_band']:
                        val = result['stats_per_band'][band]['rmse']['mean']
                        row += f" {val:>12.4f} |"
                    else:
                        row += f" {'N/A':>12} |"
            else:
                row += " N/A" * len(band_names)
            print(row)

    def _print_improvements(self):
        """Print improvements over bilinear baseline."""
        bilinear = self.results['Bilinear']
        
        print(f"\n{'Model':<30} | {'ΔPSNR':>10} | {'ΔSSIM':>10} | {'ΔERGAS':>10}")
        print("-" * 70)
        
        for name, result in self.results.items():
            if name == 'Bilinear':
                continue
            
            stats = result['stats_combined']
            bil_stats = bilinear['stats_combined']
            
            delta_psnr = stats['psnr']['mean'] - bil_stats['psnr']['mean']
            delta_ssim = stats['ssim']['mean'] - bil_stats['ssim']['mean']
            delta_ergas = bil_stats['ergas']['mean'] - stats['ergas']['mean']  # Lower is better
            
            print(f"{name:<30} | {delta_psnr:>+10.2f} | {delta_ssim:>+10.4f} | {delta_ergas:>+10.2f}")
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def save_results(self, filename: str = "evaluation_results"):
        """Save results to JSON and CSV."""
        # Prepare serializable results
        serializable = {}
        for name, result in self.results.items():
            serializable[name] = {
                'checkpoint_path': result['checkpoint_path'],
                'stats_10m': self._convert_to_serializable(result['stats_10m']),
                'stats_20m': self._convert_to_serializable(result['stats_20m']),
                'stats_combined': self._convert_to_serializable(result['stats_combined']),
            }
        
        # Save JSON (summary stats)
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSaved JSON results to: {json_path}")
        
        # Save per-sample metrics for violin plots etc.
        per_sample = {}
        for name, result in self.results.items():
            per_sample[name] = {
                'per_sample_10m': self._convert_to_serializable(result.get('per_sample_10m', [])),
                'per_sample_20m': self._convert_to_serializable(result.get('per_sample_20m', [])),
            }
        per_sample_path = self.output_dir / f"{filename}_per_sample.json"
        with open(per_sample_path, 'w') as f:
            json.dump(per_sample, f)
        print(f"Saved per-sample JSON to: {per_sample_path}")
        
        # Save CSV
        df = self.get_summary_table()
        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV results to: {csv_path}")
        
        return json_path, csv_path


# =============================================================================
# VISUALIZATION
# =============================================================================

def normalize_for_display(img: np.ndarray, percentile: int = 98) -> np.ndarray:
    """Normalize image for display using percentile clipping."""
    img = img.copy()
    vmin = np.percentile(img, 100 - percentile)
    vmax = np.percentile(img, percentile)
    img = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return img


def create_rgb_composite(bands_10m: np.ndarray) -> np.ndarray:
    """Create RGB composite from 10m bands (Blue, Green, Red order)."""
    rgb = bands_10m[:, :, [2, 1, 0]]  # Reorder to RGB
    return normalize_for_display(rgb)


def create_false_color(bands_10m: np.ndarray, bands_20m: np.ndarray) -> np.ndarray:
    """Create false color composite (NIR, Red, Green)."""
    nir = bands_20m[:, :, 1]
    red = bands_10m[:, :, 2]
    green = bands_10m[:, :, 1]
    
    if nir.shape[0] != red.shape[0]:
        scale = red.shape[0] / nir.shape[0]
        nir = zoom(nir, scale, order=1)
    
    false_color = np.stack([nir, red, green], axis=-1)
    return normalize_for_display(false_color)


class ResultVisualizer:
    """Generate visualizations for evaluation results."""
    
    def __init__(self, evaluator: ModelEvaluator, viz_subdir: str = 'visualizations'):
        self.evaluator = evaluator
        self.output_dir = evaluator.output_dir / viz_subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_metrics_comparison(self, save: bool = True) -> plt.Figure:
        """Create bar chart comparing all models across metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Comparison - Combined Metrics', fontsize=14, fontweight='bold')
        
        model_names = list(self.evaluator.results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        for ax, metric in zip(axes.flat, self.evaluator.METRIC_NAMES):
            values = [self.evaluator.results[name]['stats_combined'][metric]['mean'] 
                      for name in model_names]
            errors = [self.evaluator.results[name]['stats_combined'][metric].get('ci95', 0) 
                      for name in model_names]
            
            x = np.arange(len(model_names))
            bars = ax.bar(x, values, color=colors, yerr=errors, capsize=3)
            
            ax.set_xlabel('Model')
            direction = "↑" if metric in self.evaluator.HIGHER_BETTER else "↓"
            ax.set_ylabel(f'{metric.upper()} {direction}')
            ax.set_title(f'{metric.upper()}')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Highlight best
            if metric in self.evaluator.HIGHER_BETTER:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'metrics_comparison.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig
    
    def plot_branch_comparison(self, save: bool = True) -> plt.Figure:
        """Compare metrics across 10m and 20m branches."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Branch-wise Performance Comparison', fontsize=14, fontweight='bold')
        
        model_names = list(self.evaluator.results.keys())
        x = np.arange(len(model_names))
        width = 0.35
        
        metrics_to_plot = ['psnr', 'ssim', 'ergas', 'sid']
        
        for ax, metric in zip(axes.flat, metrics_to_plot):
            values_10m = [self.evaluator.results[name]['stats_10m'][metric]['mean'] 
                          for name in model_names]
            values_20m = [self.evaluator.results[name]['stats_20m'][metric]['mean'] 
                          for name in model_names]
            
            bars1 = ax.bar(x - width/2, values_10m, width, label='10m Branch', color='steelblue')
            bars2 = ax.bar(x + width/2, values_20m, width, label='20m Branch', color='coral')
            
            direction = "↑" if metric in self.evaluator.HIGHER_BETTER else "↓"
            ax.set_ylabel(f'{metric.upper()} {direction}')
            ax.set_title(f'{metric.upper()} by Branch')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'branch_comparison.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig
    
    def plot_improvement_heatmap(self, save: bool = True) -> Optional[plt.Figure]:
        """Heatmap of improvements over bilinear baseline."""
        if 'Bilinear' not in self.evaluator.results:
            return None
        
        bilinear = self.evaluator.results['Bilinear']
        model_names = [n for n in self.evaluator.results.keys() if n != 'Bilinear']
        
        if not model_names:
            return None
        
        # Compute improvements
        improvements = np.zeros((len(model_names), len(self.evaluator.METRIC_NAMES)))
        pct_improvements = np.zeros((len(model_names), len(self.evaluator.METRIC_NAMES)))
        
        for i, name in enumerate(model_names):
            result = self.evaluator.results[name]
            for j, metric in enumerate(self.evaluator.METRIC_NAMES):
                val = result['stats_combined'][metric]['mean']
                bil_val = bilinear['stats_combined'][metric]['mean']
                
                if metric in self.evaluator.HIGHER_BETTER:
                    improvements[i, j] = val - bil_val
                    pct_improvements[i, j] = ((val - bil_val) / (abs(bil_val) + 1e-10)) * 100
                else:
                    improvements[i, j] = bil_val - val  # Flip sign for lower-is-better
                    pct_improvements[i, j] = ((bil_val - val) / (abs(bil_val) + 1e-10)) * 100
        
        # Rank models by mean improvement across all metrics (best first)
        mean_improvement = improvements.mean(axis=1)
        rank_order = np.argsort(mean_improvement)[::-1]  # descending
        improvements = improvements[rank_order]
        pct_improvements = pct_improvements[rank_order]
        model_names = [model_names[i] for i in rank_order]
        
        fig, ax = plt.subplots(figsize=(14, max(4, len(model_names) * 0.9)))
        
        im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto')
        
        ax.set_xticks(np.arange(len(self.evaluator.METRIC_NAMES)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels([m.upper() for m in self.evaluator.METRIC_NAMES], fontsize=12)
        ax.set_yticklabels([f'#{i+1}  {name}' for i, name in enumerate(model_names)], fontsize=11)
        
        # Add text annotations with absolute value and percentage
        for i in range(len(model_names)):
            for j in range(len(self.evaluator.METRIC_NAMES)):
                val = improvements[i, j]
                pct = pct_improvements[i, j]
                text = ax.text(j, i, f'{val:+.3f}\n({pct:+.1f}%)',
                              ha='center', va='center', color='black', fontsize=11, fontweight='bold')
        
        ax.set_title('Improvement over Bilinear Baseline — Ranked by Mean Improvement (Best → Worst)\n(Green = Better, Red = Worse)', fontsize=13)
        fig.colorbar(im, ax=ax, label='Improvement')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'improvement_heatmap.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig
    
    def plot_sample_comparison(self, sample_indices: List[int], save: bool = True):
        """Generate visual comparison of sample outputs across models."""
        test_data = self.evaluator.test_data
        
        for idx in sample_indices:
            sample = test_data.get_sample(idx)
            model_inputs = test_data.prepare_for_model(sample)
            
            gt_10m = sample['hr_10m']
            gt_20m = sample['hr_20m']
            lr_10m = sample['lr_10m']
            
            n_models = len(self.evaluator.results)
            # Total items per row: LR, GT, + all models
            total_items = n_models + 2
            n_cols = min(total_items, 6)  # Max 6 columns
            n_rows_rgb = int(np.ceil(total_items / n_cols))
            n_rows_error = int(np.ceil(n_models / n_cols))  # Error maps only for models
            n_rows = n_rows_rgb + n_rows_error
            
            fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
            
            # Row 1+: RGB composites
            ax_lr = fig.add_subplot(n_rows, n_cols, 1)
            ax_lr.imshow(create_rgb_composite(lr_10m))
            ax_lr.set_title(f'LR Input\n({lr_10m.shape[0]}×{lr_10m.shape[1]})')
            ax_lr.axis('off')
            
            ax_gt = fig.add_subplot(n_rows, n_cols, 2)
            ax_gt.imshow(create_rgb_composite(gt_10m))
            ax_gt.set_title(f'Ground Truth\n({gt_10m.shape[0]}×{gt_10m.shape[1]})')
            ax_gt.axis('off')
            
            model_outputs = {}
            for i, (name, result) in enumerate(self.evaluator.results.items()):
                if name == 'Bilinear':
                    preds = compute_bilinear_baseline(sample)
                else:
                    preds = run_inference(result['model'], model_inputs, self.evaluator.device)
                
                model_outputs[name] = preds
                
                # Position in RGB section (starting after LR and GT)
                rgb_pos = 3 + i
                ax = fig.add_subplot(n_rows, n_cols, rgb_pos)
                ax.imshow(create_rgb_composite(preds['sr_10m']))
                
                # Compute metrics for this sample
                metrics = compute_all_metrics(preds['sr_10m'], gt_10m, scale_ratio=3.33)
                ax.set_title(f'{name}\nPSNR: {metrics["psnr"]:.2f} dB')
                ax.axis('off')
            
            # Error maps section (starts after RGB section)
            error_start_pos = n_rows_rgb * n_cols + 1
            
            for i, (name, preds) in enumerate(model_outputs.items()):
                # Position in error map section
                error_pos = error_start_pos + i
                ax = fig.add_subplot(n_rows, n_cols, error_pos)
                error = np.abs(preds['sr_10m'] - gt_10m).mean(axis=-1)
                im = ax.imshow(error, cmap='hot', vmin=0, vmax=0.1)
                ax.set_title(f'{name} MAE\n{error.mean():.4f}')
                ax.axis('off')
            
            plt.suptitle(f'Sample {idx} - Multi-Model Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save:
                path = self.output_dir / f'sample_{idx}_comparison.png'
                plt.savefig(path, dpi=150, bbox_inches='tight')
                print(f"Saved: {path}")
            
            plt.close(fig)
    
    def generate_all_visualizations(self, n_samples: int = 20,
                                     filter_best_samples: bool = False,
                                     filter_psnr_threshold: float = 2.0,
                                     filter_ref_models: list = None):
        """Generate all visualizations.
        
        Args:
            n_samples: Number of sample comparisons to plot.
            filter_best_samples: If True, only plot samples where reference models
                                 beat Bilinear by >= filter_psnr_threshold dB PSNR.
            filter_psnr_threshold: PSNR threshold (dB) above Bilinear.
            filter_ref_models: Model names to check. Default: SchedPer Adaptive variants.
        """
        print("\nGenerating visualizations...")
        
        # Metric plots
        self.plot_metrics_comparison()
        self.plot_branch_comparison()
        self.plot_improvement_heatmap()
        
        # Sample comparisons
        if filter_best_samples:
            sample_indices = self._select_best_samples(
                filter_psnr_threshold, filter_ref_models)
            self.plot_sample_comparison(sample_indices)
        elif n_samples > 0:
            n_test = len(self.evaluator.test_data)
            np.random.seed(42)
            sample_indices = np.random.choice(
                n_test, min(n_samples, n_test), replace=False).tolist()
            self.plot_sample_comparison(sample_indices)
        
        plt.close('all')
        print(f"\nAll visualizations saved to: {self.output_dir}")
    
    def _select_best_samples(self, psnr_threshold: float = 2.0,
                             ref_models: list = None) -> List[int]:
        """Select ALL sample indices where reference models beat Bilinear by >= threshold dB."""
        if 'Bilinear' not in self.evaluator.results:
            print("WARNING: No Bilinear baseline, cannot filter.")
            return []
        
        if ref_models is None:
            ref_models = ['SchedPer Adaptive (Residual)', 'SchedPer Adaptive (Direct)']
        
        available_refs = [m for m in ref_models if m in self.evaluator.results]
        if not available_refs:
            print(f"WARNING: None of {ref_models} found in results.")
            return []
        
        bilinear_psnr = np.array(
            [s['psnr'] for s in self.evaluator.results['Bilinear']['per_sample_10m']])
        n_total = len(bilinear_psnr)
        
        # Keep samples where ANY reference model beats Bilinear by >= threshold
        keep_mask = np.zeros(n_total, dtype=bool)
        for ref_name in available_refs:
            ref_psnr = np.array(
                [s['psnr'] for s in self.evaluator.results[ref_name]['per_sample_10m']])
            keep_mask |= (ref_psnr - bilinear_psnr) >= psnr_threshold
        
        eligible = np.where(keep_mask)[0].tolist()
        print(f"Best-sample filter: plotting {len(eligible)}/{n_total} samples where "
              f"{available_refs} >= {psnr_threshold} dB above Bilinear (10m PSNR)")
        return eligible


# =============================================================================
# CLI & MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare multiple SR models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate specific models
  python evaluate_models.py --models "MultiRes Full" --checkpoints path/to/model.pt
  
  # Use config file
  python evaluate_models.py --config evaluation_config.json
  
  # Quick test with limited samples
  python evaluate_models.py --max_samples 10
        """
    )
    
    parser.add_argument('--test_dir', type=str, 
                        default=str(PROJECT_ROOT / 'data' / 'sr_training' / '5band_v3' / 'combined' / 'mmap_test'),
                        help='Path to test data directory (mmap format)')
    
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT / 'results' / 'model_evaluation'),
                        help='Output directory for results and visualizations')
    
    parser.add_argument('--models', nargs='+', type=str,
                        help='Model names (space-separated)')
    
    parser.add_argument('--checkpoints', nargs='+', type=str,
                        help='Checkpoint paths (space-separated, same order as --models)')
    
    parser.add_argument('--config', type=str,
                        help='Path to JSON config file with model definitions')
    
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of test samples to evaluate')
    
    parser.add_argument('--n_vis_samples', type=int, default=5,
                        help='Number of samples to visualize')
    
    parser.add_argument('--no_bilinear', action='store_true',
                        help='Skip bilinear baseline comparison')
    
    parser.add_argument('--no_viz', action='store_true',
                        help='Skip visualization generation')
    
    parser.add_argument('--viz_subdir', type=str, default='visualizations',
                        help='Subdirectory name for visualizations (default: visualizations)')
    
    parser.add_argument('--filter_best_samples', action='store_true',
                        help='Filter to samples where SchedPer Adaptive models beat Bilinear by --filter_psnr_threshold dB')
    
    parser.add_argument('--filter_psnr_threshold', type=float, default=2.0,
                        help='PSNR threshold (dB) above Bilinear for sample filtering (default: 2.0)')
    
    parser.add_argument('--filter_ref_models', nargs='+', type=str, default=None,
                        help='Reference model names for filtering (default: SchedPer Adaptive Residual/Direct)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # If config file provided, override test_dir and output_dir if specified in config
    test_dir = args.test_dir
    output_dir = args.output_dir
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        if 'test_dir' in config:
            config_test_dir = Path(config['test_dir'])
            if not config_test_dir.is_absolute():
                config_test_dir = PROJECT_ROOT / config_test_dir
            test_dir = str(config_test_dir)
        if 'output_dir' in config:
            config_output_dir = Path(config['output_dir'])
            if not config_output_dir.is_absolute():
                config_output_dir = PROJECT_ROOT / config_output_dir
            output_dir = str(config_output_dir)
    
    # Get skip_indices from config if available
    skip_indices = config.get('skip_indices', []) if args.config else []
    
    print("=" * 70)
    print("MULTI-MODEL SR EVALUATION")
    print("=" * 70)
    print(f"Test data: {test_dir}")
    print(f"Output dir: {output_dir}")
    if skip_indices:
        print(f"Skipping {len(skip_indices)} bad samples: {skip_indices}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        test_dir=Path(test_dir),
        output_dir=Path(output_dir),
        skip_indices=skip_indices
    )
    
    # Add bilinear baseline
    if not args.no_bilinear:
        evaluator.add_bilinear_baseline()
    
    # Load models from config or CLI args
    if args.config:
        # Config already loaded above
        for model_def in config['models']:
            # Skip entries with null checkpoint (e.g., bilinear baseline)
            if model_def['checkpoint'] is None:
                continue
            # Convert relative paths to absolute paths from PROJECT_ROOT
            checkpoint_path = Path(model_def['checkpoint'])
            if not checkpoint_path.is_absolute():
                checkpoint_path = PROJECT_ROOT / checkpoint_path
            evaluator.add_model(model_def['name'], checkpoint_path)
    elif args.models and args.checkpoints:
        if len(args.models) != len(args.checkpoints):
            raise ValueError("Number of models must match number of checkpoints")
        for name, ckpt in zip(args.models, args.checkpoints):
            evaluator.add_model(name, Path(ckpt))
    else:
        # Default: load common models from checkpoints directory
        checkpoints_dir = PROJECT_ROOT / 'checkpoints'
        default_models = [
            # Residual learning models
            ('MultiRes (Full)', checkpoints_dir / 'multires_perceptual_full' / 'best_model.pt'),
            ('MultiRes (Filtered)', checkpoints_dir / 'multires_perceptual_filtered' / 'best_model.pt'),
            ('V3 (Full)', checkpoints_dir / '5band_v3_perceptual_full' / 'best_model.pt'),
            ('Temporal (Full)', checkpoints_dir / 'multires_temporal_full' / 'best_model.pt'),
            ('BestFrame (Full)', checkpoints_dir / 'multires_bestframe_combined_full' / 'best_model.pt'),
            ('Median (Full)', checkpoints_dir / 'multires_median_median_full' / 'best_model.pt'),
            ('MultiOutput (Full)', checkpoints_dir / 'multires_multioutput_t0.5_full' / 'best_model.pt'),
            ('MultiOutput SchedPer (Full)', checkpoints_dir / 'multires_multioutput_schedPer_t0.5_full' / 'best_model.pt'),
            # GAN model
            ('GAN (Full)', checkpoints_dir / 'multires_GAN_full' / 'best_model.pt'),
            # Direct learning models (no bilinear baseline)
            ('Temporal Direct (Full)', checkpoints_dir / 'multires_temporal_direct_full' / 'best_model.pt'),
            ('BestFrame Direct (Full)', checkpoints_dir / 'multires_bestframe_combined_direct_full' / 'best_model.pt'),
            ('Median Direct (Full)', checkpoints_dir / 'multires_median_median_direct_full' / 'best_model.pt'),
            ('MultiOutput Direct (Full)', checkpoints_dir / 'multires_multioutput_t0.5_direct_full' / 'best_model.pt'),
            ('MultiOutput SchedPer Direct (Full)', checkpoints_dir / 'multires_multioutput_schedPer_t0.5_direct_full' / 'best_model.pt'),
        ]
        
        for name, path in default_models:
            if path.exists():
                evaluator.add_model(name, path)
            else:
                print(f"  Skipping {name}: checkpoint not found at {path}")
    
    if len(evaluator.results) == (0 if args.no_bilinear else 1):
        print("\nERROR: No models loaded! Specify models with --models/--checkpoints or --config")
        return 1
    
    # Run evaluation
    evaluator.evaluate_all(max_samples=args.max_samples)
    
    # Print results
    evaluator.print_results()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    evaluator.save_results(f"evaluation_{timestamp}")
    
    # Generate visualizations
    if not args.no_viz:
        visualizer = ResultVisualizer(evaluator, viz_subdir=args.viz_subdir)
        visualizer.generate_all_visualizations(
            n_samples=args.n_vis_samples,
            filter_best_samples=args.filter_best_samples,
            filter_psnr_threshold=args.filter_psnr_threshold,
            filter_ref_models=args.filter_ref_models,
        )
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
