"""
Evaluate the 600-sample SR model on real satellite test images
Single model evaluation with proper visualizations
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from typing import Dict

# Import model architecture
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from train_with_chunked_dataset import ImprovedSatelliteSR


def load_model(model_path: str, device: str = 'cuda', base_channels: int = 64, num_blocks: int = 8):
    """Load the trained SR model"""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = ImprovedSatelliteSR(
        in_channels=4,
        out_channels=4,
        base_channels=base_channels,
        num_blocks=num_blocks,
        upscale_factor=4,
        sequence_length=3
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded successfully")
    print(f"   Architecture: {num_blocks} blocks, {base_channels} base channels")
    print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'Unknown'):.6f}")
    
    return model, checkpoint


def load_test_samples(dataset_dir: str, num_samples: int = 10):
    """Load test samples from dataset"""
    dataset_path = Path(dataset_dir)
    samples_file = dataset_path / "training_samples.pkl"
    
    print(f"\nLoading test samples from {dataset_dir}...")
    with open(samples_file, 'rb') as f:
        all_samples = pickle.load(f)
    
    # Use samples from different parts of dataset
    step = len(all_samples) // num_samples
    test_samples = [all_samples[i * step] for i in range(num_samples)]
    
    print(f"   ✅ Loaded {len(test_samples)} test samples")
    print(f"   Sampling from dataset every {step} samples")
    
    return test_samples


def calculate_metrics(sr: np.ndarray, hr: np.ndarray) -> Dict[str, float]:
    """Calculate quality metrics"""
    # PSNR
    mse = np.mean((sr - hr) ** 2)
    if mse == 0:
        psnr = 100
    else:
        max_val = np.max(hr)
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if max_val > 0 else 0
    
    # SSIM (simplified)
    def ssim_channel(x, y):
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        sigma_x = np.var(x)
        sigma_y = np.var(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))
        
        max_val = np.max(hr)
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
        return ssim
    
    ssim_vals = []
    for b in range(sr.shape[-1]):
        ssim_vals.append(ssim_channel(sr[..., b], hr[..., b]))
    ssim = np.mean(ssim_vals)
    
    # Spectral Angle Mapper (SAM)
    dot_product = np.sum(sr * hr, axis=-1)
    norm_sr = np.linalg.norm(sr, axis=-1)
    norm_hr = np.linalg.norm(hr, axis=-1)
    
    mask = (norm_sr > 0) & (norm_hr > 0)
    cos_angle = np.zeros_like(norm_sr)
    cos_angle[mask] = np.clip(dot_product[mask] / (norm_sr[mask] * norm_hr[mask]), -1, 1)
    
    sam = np.mean(np.arccos(cos_angle)) * 180 / np.pi
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(sr - hr))
    
    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'SAM': sam,
        'MAE': mae
    }


def run_inference(model, lr_sequence: np.ndarray, device: str = 'cuda') -> np.ndarray:
    """Run SR inference"""
    lr_tensor = torch.from_numpy(lr_sequence).float()
    lr_tensor = lr_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)
    
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    sr_array = sr_tensor[0].permute(1, 2, 0).cpu().numpy()
    return sr_array


def normalize_for_display(img, percentile_clip=True):
    """Normalize image for display"""
    # Use first 3 bands as RGB
    if img.shape[-1] >= 3:
        rgb = img[..., :3]
    else:
        rgb = np.stack([img[..., 0]] * 3, axis=-1)
    
    if percentile_clip:
        rgb_min = np.percentile(rgb, 2)
        rgb_max = np.percentile(rgb, 98)
    else:
        rgb_min = np.min(rgb)
        rgb_max = np.max(rgb)
    
    rgb = np.clip((rgb - rgb_min) / (rgb_max - rgb_min + 1e-8), 0, 1)
    return rgb


def visualize_sample(lr_seq, sr, hr, metrics, output_path, sample_info):
    """Create visualization for one sample"""
    # Use middle frame from temporal sequence
    t_idx = lr_seq.shape[0] // 2
    lr_frame = lr_seq[t_idx]
    
    # Normalize for display
    lr_rgb = normalize_for_display(lr_frame)
    sr_rgb = normalize_for_display(sr)
    hr_rgb = normalize_for_display(hr)
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)
    
    # LR Input
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(lr_rgb)
    ax1.set_title(f'LR Input\n{lr_frame.shape[0]}×{lr_frame.shape[1]}×{lr_frame.shape[2]}', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # SR Output
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.imshow(sr_rgb)
    ax2.set_title(f'SR Output (4× upscaling)\n{sr.shape[0]}×{sr.shape[1]}×{sr.shape[2]}', 
                  fontsize=12, fontweight='bold', color='blue')
    ax2.axis('off')
    
    # HR Target
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.imshow(hr_rgb)
    ax3.set_title(f'HR Target\n{hr.shape[0]}×{hr.shape[1]}×{hr.shape[2]}', 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Add metrics text
    metrics_text = (
        f"Sample: {sample_info['index']}\n"
        f"Region: {sample_info['region']}\n"
        f"Date: {sample_info['date']}\n\n"
        f"Quality Metrics:\n"
        f"PSNR: {metrics['PSNR']:.2f} dB\n"
        f"SSIM: {metrics['SSIM']:.3f}\n"
        f"SAM: {metrics['SAM']:.2f}°\n"
        f"MAE: {metrics['MAE']:.2f}"
    )
    
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(model_path: str = "./sr_models/best_model.pth",
         dataset_dir: str = "./data/datasets/sr_dataset_600",
         output_dir: str = "./evaluation_results_single",
         num_samples: int = 10,
         base_channels: int = 64,
         num_blocks: int = 8):
    print("="*70)
    print("🔬 SR MODEL EVALUATION")
    print("="*70)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Architecture: {num_blocks} blocks, {base_channels} base channels")
    
    # Paths
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    model, checkpoint = load_model(model_path, device, base_channels, num_blocks)
    
    # Load test samples
    print("\n" + "="*70)
    print("LOADING TEST SAMPLES")
    print("="*70)
    test_samples = load_test_samples(dataset_dir, num_samples=num_samples)
    
    # Run evaluation
    print("\n" + "="*70)
    print("RUNNING EVALUATION")
    print("="*70)
    
    all_metrics = []
    
    for i, sample in enumerate(test_samples):
        print(f"\nProcessing sample {i+1}/{len(test_samples)}...")
        
        lr_seq = sample.lr_sequence
        hr_seq = sample.hr_sequence
        
        # Use middle frame
        t_idx = hr_seq.shape[0] // 2
        hr_frame = hr_seq[t_idx]
        
        # Run inference
        sr = run_inference(model, lr_seq, device)
        
        # Calculate metrics
        metrics = calculate_metrics(sr, hr_frame)
        all_metrics.append(metrics)
        
        print(f"   PSNR: {metrics['PSNR']:.2f} dB")
        print(f"   SSIM: {metrics['SSIM']:.3f}")
        print(f"   SAM: {metrics['SAM']:.2f}°")
        print(f"   MAE: {metrics['MAE']:.2f}")
        
        # Create visualization
        sample_info = {
            'index': i + 1,
            'region': sample.region_name,
            'date': ', '.join(sample.temporal_dates)
        }
        
        vis_output_path = output_path / f"sr_evaluation_sample_{i+1:02d}.png"
        visualize_sample(lr_seq, sr, hr_frame, metrics, str(vis_output_path), sample_info)
        print(f"   ✅ Saved: {vis_output_path.name}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    print()
    print(f"Model: 600-sample trained model (Epoch {checkpoint.get('epoch')})")
    print(f"Validation loss: {checkpoint.get('val_loss', 0):.6f}")
    print(f"Test samples: {len(test_samples)}")
    print()
    print("Average Quality Metrics:")
    print(f"   PSNR:  {np.mean([m['PSNR'] for m in all_metrics]):.2f} ± {np.std([m['PSNR'] for m in all_metrics]):.2f} dB")
    print(f"   SSIM:  {np.mean([m['SSIM'] for m in all_metrics]):.3f} ± {np.std([m['SSIM'] for m in all_metrics]):.3f}")
    print(f"   SAM:   {np.mean([m['SAM'] for m in all_metrics]):.2f} ± {np.std([m['SAM'] for m in all_metrics]):.2f}°")
    print(f"   MAE:   {np.mean([m['MAE'] for m in all_metrics]):.2f} ± {np.std([m['MAE'] for m in all_metrics]):.2f}")
    print()
    
    # Quality assessment
    avg_psnr = np.mean([m['PSNR'] for m in all_metrics])
    avg_ssim = np.mean([m['SSIM'] for m in all_metrics])
    
    print("💡 QUALITY ASSESSMENT:")
    if avg_psnr > 25 and avg_ssim > 0.7:
        print("   ✅ EXCELLENT: Model produces high-quality super-resolution!")
    elif avg_psnr > 20 and avg_ssim > 0.5:
        print("   ✓ GOOD: Model performs well, minor improvements possible")
    elif avg_psnr > 15 and avg_ssim > 0.3:
        print("   ⚠️  MODERATE: Model works but has room for improvement")
    else:
        print("   ❌ NEEDS IMPROVEMENT: Consider retraining with different hyperparameters")
    
    print(f"\n📁 Visualizations saved to: {output_path}/")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate SR model')
    parser.add_argument('--model-path', type=str, default='./sr_models/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset-dir', type=str, default='./data/datasets/sr_dataset_600',
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results_single',
                       help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of test samples to evaluate')
    parser.add_argument('--num-blocks', type=int, default=8,
                       help='Number of residual blocks in model')
    parser.add_argument('--base-channels', type=int, default=64,
                       help='Number of base channels in model')
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks
    )
