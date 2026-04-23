"""
Visualize 5 Individual Samples: LR, SR (model output), and HR

Shows the super-resolution results from the trained model compared to 
low-resolution input and high-resolution ground truth.
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add parent directories to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / 'training'))
sys.path.insert(0, str(script_dir.parent / 'preprocessing'))


# Define TrainingSample for pickle loading
@dataclass
class TrainingSample:
    """Training sample with temporal sequences and multi-resolution LR bands"""
    lr_sequence: np.ndarray
    hr_sequence: np.ndarray
    temporal_dates: List[str]
    planet_dates: List[str]
    region_name: str
    sequence_id: int
    tile_id: int
    tile_bounds: Dict
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None
    lr_20m_sequence: Optional[np.ndarray] = None
    frame_quality: Optional[List[Dict]] = None


from models.satellite_sr_model import SingleResolutionSatelliteSR


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = SingleResolutionSatelliteSR(
        in_channels=3,
        out_channels=3,
        base_channels=config['base_channels'],
        num_encoder_blocks=config['num_encoder_blocks'],
        num_frames=config['num_frames'],
        scale_factor=config['scale_factor']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def prepare_sample(sample, frame_idx=1):
    """Prepare a single sample for visualization.
    
    Args:
        sample: TrainingSample object
        frame_idx: Which temporal frame to visualize (default: middle frame)
    
    Returns:
        lr_tensor: (1, T, C, H, W) tensor for model input
        hr_frame: (H, W, C) numpy array for display
        lr_frame: (H, W, C) numpy array for display
    """
    # Get LR and HR sequences (T, H, W, C) -> (T, C, H, W)
    lr = sample.lr_sequence.transpose(0, 3, 1, 2).astype(np.float32)
    hr = sample.hr_sequence.transpose(0, 3, 1, 2).astype(np.float32)
    
    # Create tensor with batch dimension
    lr_tensor = torch.from_numpy(lr).unsqueeze(0)  # (1, T, C, H, W)
    
    # Get single frames for display (C, H, W) -> (H, W, C)
    lr_frame = lr[frame_idx].transpose(1, 2, 0)  # (H, W, C)
    hr_frame = hr[frame_idx].transpose(1, 2, 0)  # (H, W, C)
    
    return lr_tensor, lr_frame, hr_frame


def run_inference(model, lr_tensor, device):
    """Run model inference.
    
    Args:
        model: Trained SR model
        lr_tensor: (1, T, C, H, W) input tensor
        device: torch device
    
    Returns:
        sr_frame: (H, W, C) super-resolved numpy array
    """
    with torch.no_grad():
        lr_tensor = lr_tensor.to(device)
        sr_output = model(lr_tensor)  # (1, C, H', W')
        
        # Convert to numpy (H, W, C)
        sr_frame = sr_output[0].cpu().numpy().transpose(1, 2, 0)
    
    return sr_frame


def visualize_sample(lr_frame, sr_frame, hr_frame, sample_idx, dates, output_path):
    """Create visualization for a single sample.
    
    Shows:
    - LR (Sentinel-2 input)
    - SR (Model output)
    - HR (Planet ground truth)
    - Difference maps
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Normalize for display (clip to 0-1 and apply gamma for better visualization)
    def normalize_for_display(img, gamma=0.5):
        img = np.clip(img, 0, 1)
        # Apply gamma correction for better visualization of dark areas
        img = np.power(img, gamma)
        return img
    
    # Resize HR to match SR for comparison if needed
    if sr_frame.shape[:2] != hr_frame.shape[:2]:
        # Resize HR to SR size for fair comparison
        hr_resized = np.array([
            np.interp(
                np.linspace(0, 1, sr_frame.shape[0]),
                np.linspace(0, 1, hr_frame.shape[0]),
                hr_frame[:, :, c].flatten()
            ).reshape(sr_frame.shape[0], sr_frame.shape[1])
            for c in range(3)
        ]).transpose(1, 2, 0)
    else:
        hr_resized = hr_frame
    
    # Upscale LR for visual comparison
    lr_upscaled = np.array([
        np.interp(
            np.linspace(0, 1, sr_frame.shape[0] * sr_frame.shape[1]),
            np.linspace(0, 1, lr_frame.shape[0] * lr_frame.shape[1]),
            lr_frame[:, :, c].flatten()
        ).reshape(sr_frame.shape[0], sr_frame.shape[1])
        for c in range(3)
    ]).transpose(1, 2, 0)
    
    # Row 1: LR, SR, HR
    # RGB display (bands are B, G, R so reverse for display)
    axes[0, 0].imshow(normalize_for_display(lr_frame[:, :, ::-1]))
    axes[0, 0].set_title(f'LR Input (Sentinel-2)\n{lr_frame.shape[0]}×{lr_frame.shape[1]}', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(normalize_for_display(sr_frame[:, :, ::-1]))
    axes[0, 1].set_title(f'SR Output (Model)\n{sr_frame.shape[0]}×{sr_frame.shape[1]}', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(normalize_for_display(hr_frame[:, :, ::-1]))
    axes[0, 2].set_title(f'HR Ground Truth (Planet)\n{hr_frame.shape[0]}×{hr_frame.shape[1]}', fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: LR upscaled, Difference SR-HR, Metrics
    axes[1, 0].imshow(normalize_for_display(lr_upscaled[:, :, ::-1]))
    axes[1, 0].set_title('LR Upscaled (Bicubic)\nFor comparison', fontsize=12)
    axes[1, 0].axis('off')
    
    # Difference map (SR - HR)
    diff = np.abs(sr_frame - hr_resized)
    diff_display = np.clip(diff * 5, 0, 1)  # Amplify differences for visibility
    axes[1, 1].imshow(diff_display)
    axes[1, 1].set_title(f'|SR - HR| (×5 for visibility)\nMean diff: {diff.mean():.4f}', fontsize=12)
    axes[1, 1].axis('off')
    
    # Metrics text
    mse = np.mean((sr_frame - hr_resized) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))
    
    # Structural similarity (simplified)
    mean_sr = sr_frame.mean()
    mean_hr = hr_resized.mean()
    std_sr = sr_frame.std()
    std_hr = hr_resized.std()
    cov = np.mean((sr_frame - mean_sr) * (hr_resized - mean_hr))
    ssim_approx = (2 * mean_sr * mean_hr + 0.01) * (2 * cov + 0.03) / \
                  ((mean_sr**2 + mean_hr**2 + 0.01) * (std_sr**2 + std_hr**2 + 0.03))
    
    metrics_text = f"""Sample {sample_idx}
    
Dates:
  S2: {dates['s2']}
  Planet: {dates['planet']}

Metrics:
  MSE: {mse:.6f}
  PSNR: {psnr:.2f} dB
  SSIM (approx): {ssim_approx:.4f}

Value Ranges:
  LR: [{lr_frame.min():.3f}, {lr_frame.max():.3f}]
  SR: [{sr_frame.min():.3f}, {sr_frame.max():.3f}]
  HR: [{hr_frame.min():.3f}, {hr_frame.max():.3f}]
"""
    axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Metrics', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return psnr, ssim_approx


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / 'data' / 'sr_training' / 'sta2_2020_100samples_normalized' / 'dataset.pkl'
    model_path = project_root / 'outputs' / 'legacy_test_100samples_normalized' / 'best_model.pth'
    output_dir = project_root / 'visualizations' / 'sr_results_5samples'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, config = load_model(model_path, device)
    print(f"Model loaded: {config}")
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    samples = data['samples']
    print(f"Loaded {len(samples)} samples")
    
    # Select 5 samples (evenly spaced)
    indices = [0, 20, 40, 60, 80]
    
    print(f"\nVisualizing samples: {indices}")
    print("=" * 60)
    
    results = []
    
    for i, idx in enumerate(indices):
        sample = samples[idx]
        
        # Prepare sample
        lr_tensor, lr_frame, hr_frame = prepare_sample(sample, frame_idx=1)
        
        # Run inference
        sr_frame = run_inference(model, lr_tensor, device)
        
        # Get dates
        dates = {
            's2': sample.temporal_dates[1] if len(sample.temporal_dates) > 1 else sample.temporal_dates[0],
            'planet': sample.planet_dates[1] if len(sample.planet_dates) > 1 else sample.planet_dates[0]
        }
        
        # Visualize
        output_path = output_dir / f'sample_{idx:03d}.png'
        psnr, ssim = visualize_sample(lr_frame, sr_frame, hr_frame, idx, dates, output_path)
        
        results.append({'idx': idx, 'psnr': psnr, 'ssim': ssim})
        print(f"Sample {idx}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f} -> {output_path.name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
