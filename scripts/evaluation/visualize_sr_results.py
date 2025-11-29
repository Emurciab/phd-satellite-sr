"""
Visualize Super Resolution Results

Load a trained model and visualize predictions on validation samples.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

# Use non-interactive backend for batch processing
import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any

# Import model and dataset classes
from train_with_chunked_dataset import (
    ImprovedSatelliteSR, 
    ChunkedSRDataset,
    TrainingSample,
    load_dataset
)


def load_model(model_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    metadata = checkpoint.get('metadata', {})
    
    # Get model parameters from metadata
    lr_shape = metadata.get('lr_shape', (3, 78, 79, 4))
    hr_shape = metadata.get('hr_shape', (3, 260, 260, 4))
    
    sequence_length = lr_shape[0]
    in_channels = lr_shape[-1]
    lr_h, lr_w = lr_shape[1], lr_shape[2]
    hr_h, hr_w = hr_shape[1], hr_shape[2]
    
    # Create model
    model = ImprovedSatelliteSR(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=64,
        num_blocks=8,
        upscale_factor=3,
        sequence_length=sequence_length,
        target_size=(hr_h, hr_w)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded model from {model_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}" if checkpoint.get('val_loss') else "")
    
    return model, metadata


def visualize_sample(model, sample, device='cuda', save_path=None, sample_idx=0):
    """
    Visualize LR input, SR prediction, and HR ground truth for a sample.
    
    Shows RGB composite (bands 0,1,2) for visual comparison.
    """
    model.eval()
    
    # Get data
    lr = sample.lr_sequence  # (T, H, W, C)
    hr = sample.hr_sequence  # (T, H, W, C)
    
    # Crop LR to consistent size (same as training)
    lr = lr[:, :78, :79, :]
    
    # Prepare input tensor
    lr_tensor = torch.from_numpy(lr).float().permute(0, 3, 1, 2)  # (T, C, H, W)
    lr_tensor = torch.clamp(lr_tensor / 10000.0, 0, 1)
    lr_tensor = lr_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)
    
    # Generate SR
    with torch.no_grad():
        sr_tensor = model(lr_tensor)  # (1, C, H', W')
    
    sr = sr_tensor.squeeze(0).cpu().numpy()  # (C, H', W')
    sr = np.transpose(sr, (1, 2, 0))  # (H', W', C)
    
    # Prepare HR ground truth (center frame, normalized)
    center_idx = lr.shape[0] // 2
    hr_frame = hr[center_idx] / 10000.0
    hr_frame = np.clip(hr_frame, 0, 1)
    
    # Crop both SR and HR to same size (minimum of both)
    min_h = min(sr.shape[0], hr_frame.shape[0])
    min_w = min(sr.shape[1], hr_frame.shape[1])
    sr = sr[:min_h, :min_w, :]
    hr_frame = hr_frame[:min_h, :min_w, :]
    
    # Prepare LR (center frame, normalized, upscaled for display)
    lr_frame = lr[center_idx] / 10000.0
    lr_frame = np.clip(lr_frame, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # RGB visualization (bands 2,1,0 for RGB or 0,1,2 depending on band order)
    # Assuming bands are B2, B3, B4, B8A -> RGB would be indices 2,1,0 or just 0,1,2
    rgb_idx = [2, 1, 0] if lr_frame.shape[-1] >= 3 else [0, 1, 2]
    
    # Row 1: RGB composites
    # LR RGB
    lr_rgb = lr_frame[:, :, :3]
    axes[0, 0].imshow(np.clip(lr_rgb * 3, 0, 1))  # Brightness boost
    axes[0, 0].set_title(f'LR Input (Sentinel-2)\n{lr_frame.shape[0]}x{lr_frame.shape[1]}')
    axes[0, 0].axis('off')
    
    # SR RGB
    sr_rgb = sr[:, :, :3]
    axes[0, 1].imshow(np.clip(sr_rgb * 3, 0, 1))
    axes[0, 1].set_title(f'SR Prediction\n{sr.shape[0]}x{sr.shape[1]}')
    axes[0, 1].axis('off')
    
    # HR RGB
    hr_rgb = hr_frame[:, :, :3]
    axes[0, 2].imshow(np.clip(hr_rgb * 3, 0, 1))
    axes[0, 2].set_title(f'HR Ground Truth (Planet)\n{hr_frame.shape[0]}x{hr_frame.shape[1]}')
    axes[0, 2].axis('off')
    
    # Row 2: NIR band (band 3) and difference maps
    # NIR comparison
    if lr_frame.shape[-1] >= 4:
        lr_nir = lr_frame[:, :, 3]
        sr_nir = sr[:, :, 3]
        hr_nir = hr_frame[:, :, 3]
        
        axes[1, 0].imshow(lr_nir, cmap='RdYlGn', vmin=0, vmax=0.5)
        axes[1, 0].set_title('LR NIR Band')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(sr_nir, cmap='RdYlGn', vmin=0, vmax=0.5)
        axes[1, 1].set_title('SR NIR Band')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(hr_nir, cmap='RdYlGn', vmin=0, vmax=0.5)
        axes[1, 2].set_title('HR NIR Band')
        axes[1, 2].axis('off')
    else:
        # Show difference map instead
        diff = np.abs(sr_rgb - hr_rgb).mean(axis=-1)
        axes[1, 1].imshow(diff, cmap='hot', vmin=0, vmax=0.1)
        axes[1, 1].set_title('Absolute Difference (RGB)')
        axes[1, 1].axis('off')
        axes[1, 0].axis('off')
        axes[1, 2].axis('off')
    
    plt.suptitle(f'Sample {sample_idx} - Region: {sample.region_name}\nDates: {sample.temporal_dates}', 
                 fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close(fig)  # Close figure to free memory (don't use plt.show() in batch mode)
    
    # Calculate metrics
    mse = np.mean((sr - hr_frame) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    print(f"\n   Metrics for sample {sample_idx}:")
    print(f"   MSE: {mse:.6f}")
    print(f"   PSNR: {psnr:.2f} dB")
    
    return sr, hr_frame


def visualize_multiple_samples(model, samples, device='cuda', num_samples=5, output_dir=None):
    """Visualize multiple samples"""
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Select random samples if more than requested
    if len(samples) > num_samples:
        indices = np.random.choice(len(samples), num_samples, replace=False)
    else:
        indices = range(len(samples))
    
    for i, idx in enumerate(indices):
        sample = samples[idx]
        save_path = output_path / f"sr_result_{i}.png" if output_dir else None
        
        print(f"\n{'='*50}")
        print(f"Sample {i+1}/{num_samples} (index {idx})")
        print(f"{'='*50}")
        
        visualize_sample(model, sample, device, save_path, sample_idx=idx)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize SR results')
    parser.add_argument('--model', default='data/models/training_400samples/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--dataset', default='data/datasets/sta2_400samples_parallel',
                       help='Path to dataset')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--output-dir', default='data/visualizations/sr_results',
                       help='Output directory for saved figures')
    parser.add_argument('--use-val', action='store_true',
                       help='Use validation split (last 10%)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, metadata = load_model(args.model, device)
    
    # Load dataset
    samples, _ = load_dataset(args.dataset)
    print(f"\nLoaded {len(samples)} samples")
    
    # Use validation split if requested
    if args.use_val:
        val_size = int(len(samples) * 0.1)
        samples = samples[-val_size:]
        print(f"Using validation split: {len(samples)} samples")
    
    # Visualize
    visualize_multiple_samples(
        model, 
        samples, 
        device, 
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    print(f"\n✅ Visualization complete!")
    print(f"   Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
