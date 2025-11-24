"""
Test and compare SR models on real satellite images
Evaluates both 100-sample and 600-sample trained models
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from typing import Tuple, Dict

# Import model architecture
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from train_with_chunked_dataset import ImprovedSatelliteSR


def load_model(model_path: str, device: str = 'cuda') -> torch.nn.Module:
    """Load a trained SR model"""
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model
    model = ImprovedSatelliteSR(
        in_channels=4,  # 4-band satellite imagery
        out_channels=4,
        base_channels=64,
        num_blocks=8,
        upscale_factor=4,
        sequence_length=3
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded (Epoch {checkpoint.get('epoch', 'Unknown')})")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'Unknown'):.6f}")
    
    return model, checkpoint


def load_test_samples(dataset_dir: str, num_samples: int = 5):
    """Load test samples from dataset"""
    dataset_path = Path(dataset_dir)
    samples_file = dataset_path / "training_samples.pkl"
    
    print(f"\nLoading test samples from {dataset_dir}...")
    with open(samples_file, 'rb') as f:
        all_samples = pickle.load(f)
    
    # Use last samples as test (not used in training validation)
    test_samples = all_samples[-num_samples:]
    print(f"   ✅ Loaded {len(test_samples)} test samples")
    
    return test_samples


def calculate_metrics(sr: np.ndarray, hr: np.ndarray) -> Dict[str, float]:
    """Calculate quality metrics"""
    # Ensure same shape
    assert sr.shape == hr.shape, f"Shape mismatch: {sr.shape} vs {hr.shape}"
    
    # PSNR
    mse = np.mean((sr - hr) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(np.max(hr) / np.sqrt(mse))
    
    # SSIM (simplified version)
    def ssim_channel(x, y):
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        sigma_x = np.var(x)
        sigma_y = np.var(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))
        
        c1 = (0.01 * np.max(hr)) ** 2
        c2 = (0.03 * np.max(hr)) ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
        return ssim
    
    # Average SSIM across all bands and pixels
    ssim_vals = []
    for b in range(sr.shape[-1]):
        ssim_vals.append(ssim_channel(sr[..., b], hr[..., b]))
    ssim = np.mean(ssim_vals)
    
    # Spectral Angle (SAM)
    dot_product = np.sum(sr * hr, axis=-1)
    norm_sr = np.linalg.norm(sr, axis=-1)
    norm_hr = np.linalg.norm(hr, axis=-1)
    
    # Avoid division by zero
    mask = (norm_sr > 0) & (norm_hr > 0)
    cos_angle = np.zeros_like(norm_sr)
    cos_angle[mask] = dot_product[mask] / (norm_sr[mask] * norm_hr[mask])
    cos_angle = np.clip(cos_angle, -1, 1)
    
    sam = np.mean(np.arccos(cos_angle)) * 180 / np.pi  # Convert to degrees
    
    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'SAM': sam
    }


def run_inference(model: torch.nn.Module, lr_sequence: np.ndarray, device: str = 'cuda') -> np.ndarray:
    """Run SR inference on a sample"""
    # Convert to torch tensor
    lr_tensor = torch.from_numpy(lr_sequence).float()
    
    # Add batch dimension and permute to [B, T, C, H, W]
    lr_tensor = lr_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)
    lr_tensor = lr_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        sr_tensor = model(lr_tensor)  # Output: (B, C, H, W)
    
    # Convert back to numpy [H, W, C] and add temporal dimension
    sr_array = sr_tensor[0].permute(1, 2, 0).cpu().numpy()
    
    # Expand to temporal sequence (replicate for middle frame)
    T = lr_sequence.shape[0]
    sr_sequence = np.stack([sr_array] * T, axis=0)  # (T, H, W, C)
    
    return sr_sequence


def visualize_comparison(lr: np.ndarray, sr_100: np.ndarray, sr_600: np.ndarray, 
                        hr: np.ndarray, metrics_100: Dict, metrics_600: Dict,
                        output_path: str, sample_idx: int):
    """Create comparison visualization"""
    # Use middle frame from temporal sequence
    t_idx = lr.shape[0] // 2
    
    lr_frame = lr[t_idx]
    sr_100_frame = sr_100[t_idx]
    sr_600_frame = sr_600[t_idx]
    hr_frame = hr[t_idx]
    
    # Normalize for visualization (use RGB bands: R=0, G=1, B=2)
    def normalize_for_display(img):
        # Use first 3 bands as RGB
        rgb = img[..., :3]
        # Normalize to 0-1
        rgb_min = np.percentile(rgb, 2)
        rgb_max = np.percentile(rgb, 98)
        rgb = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)
        return rgb
    
    lr_rgb = normalize_for_display(lr_frame)
    sr_100_rgb = normalize_for_display(sr_100_frame)
    sr_600_rgb = normalize_for_display(sr_600_frame)
    hr_rgb = normalize_for_display(hr_frame)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # LR Input
    axes[0, 0].imshow(lr_rgb)
    axes[0, 0].set_title(f'LR Input\n{lr_frame.shape[0]}×{lr_frame.shape[1]}', fontsize=14)
    axes[0, 0].axis('off')
    
    # SR 100-sample model
    axes[0, 1].imshow(sr_100_rgb)
    axes[0, 1].set_title(
        f'SR (100-sample model)\n'
        f'PSNR: {metrics_100["PSNR"]:.2f} dB | SSIM: {metrics_100["SSIM"]:.3f}\n'
        f'SAM: {metrics_100["SAM"]:.2f}°',
        fontsize=14
    )
    axes[0, 1].axis('off')
    
    # SR 600-sample model
    axes[1, 0].imshow(sr_600_rgb)
    axes[1, 0].set_title(
        f'SR (600-sample model)\n'
        f'PSNR: {metrics_600["PSNR"]:.2f} dB | SSIM: {metrics_600["SSIM"]:.3f}\n'
        f'SAM: {metrics_600["SAM"]:.2f}°',
        fontsize=14
    )
    axes[1, 0].axis('off')
    
    # HR Target
    axes[1, 1].imshow(hr_rgb)
    axes[1, 1].set_title(f'HR Target\n{hr_frame.shape[0]}×{hr_frame.shape[1]}', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved visualization: {output_path}")


def main():
    print("="*70)
    print("🔬 SR MODEL EVALUATION")
    print("="*70)
    print()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Paths
    model_100_path = "sr_models/model_100samples_best.pth"  # We'll need to identify this
    model_600_path = "sr_models/best_model.pth"  # Current best (600-sample)
    dataset_dir = "./data/datasets/sr_dataset_600"
    output_dir = Path("./evaluation_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check which model is which
    print("\n" + "="*70)
    print("IDENTIFYING MODELS")
    print("="*70)
    
    # Find 100-sample model (earlier timestamp, lower epoch)
    models_dir = Path("sr_models")
    checkpoints = list(models_dir.glob("*.pth"))
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime)
    
    # The model from 100-sample training should be older
    print("\nSearching for 100-sample model...")
    model_100_path = None
    for cp in checkpoints:
        try:
            ckpt = torch.load(cp, map_location='cpu', weights_only=False)
            epoch = ckpt.get('epoch', 0)
            val_loss = ckpt.get('val_loss', 0)
            # 100-sample best was at Epoch 6 with val_loss 0.010578
            if abs(val_loss - 0.010578) < 0.0001:
                model_100_path = str(cp)
                print(f"   ✅ Found 100-sample model: {cp.name}")
                print(f"      Epoch {epoch}, val_loss: {val_loss:.6f}")
                break
        except:
            continue
    
    if model_100_path is None:
        print("   ⚠️  Could not find 100-sample model, will use older checkpoint")
        # Use oldest best_model backup or checkpoint
        model_100_path = "sr_models/best_model.pth"  # Will be overwritten, but try
    
    print(f"\n600-sample model: {model_600_path}")
    
    # Load models
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)
    
    model_100, ckpt_100 = load_model(model_100_path, device)
    model_600, ckpt_600 = load_model(model_600_path, device)
    
    # Load test samples
    print("\n" + "="*70)
    print("LOADING TEST SAMPLES")
    print("="*70)
    
    test_samples = load_test_samples(dataset_dir, num_samples=5)
    
    # Run evaluation
    print("\n" + "="*70)
    print("RUNNING EVALUATION")
    print("="*70)
    
    all_metrics_100 = []
    all_metrics_600 = []
    
    for i, sample in enumerate(test_samples):
        print(f"\nSample {i+1}/{len(test_samples)}:")
        
        lr_seq = sample.lr_sequence
        hr_seq = sample.hr_sequence
        
        # Run inference
        print("   Running inference...")
        sr_100 = run_inference(model_100, lr_seq, device)
        sr_600 = run_inference(model_600, lr_seq, device)
        
        # Calculate metrics (use middle frame)
        t_idx = hr_seq.shape[0] // 2
        metrics_100 = calculate_metrics(sr_100[t_idx], hr_seq[t_idx])
        metrics_600 = calculate_metrics(sr_600[t_idx], hr_seq[t_idx])
        
        all_metrics_100.append(metrics_100)
        all_metrics_600.append(metrics_600)
        
        print(f"   100-sample: PSNR={metrics_100['PSNR']:.2f}, SSIM={metrics_100['SSIM']:.3f}, SAM={metrics_100['SAM']:.2f}°")
        print(f"   600-sample: PSNR={metrics_600['PSNR']:.2f}, SSIM={metrics_600['SSIM']:.3f}, SAM={metrics_600['SAM']:.2f}°")
        
        # Visualize
        output_path = output_dir / f"comparison_sample_{i+1}.png"
        visualize_comparison(lr_seq, sr_100, sr_600, hr_seq, 
                           metrics_100, metrics_600, str(output_path), i+1)
    
    # Summary statistics
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    
    print("\n100-SAMPLE MODEL (Epoch 6, val_loss: 0.010578):")
    print(f"   PSNR:  {np.mean([m['PSNR'] for m in all_metrics_100]):.2f} ± {np.std([m['PSNR'] for m in all_metrics_100]):.2f} dB")
    print(f"   SSIM:  {np.mean([m['SSIM'] for m in all_metrics_100]):.3f} ± {np.std([m['SSIM'] for m in all_metrics_100]):.3f}")
    print(f"   SAM:   {np.mean([m['SAM'] for m in all_metrics_100]):.2f} ± {np.std([m['SAM'] for m in all_metrics_100]):.2f}°")
    
    print("\n600-SAMPLE MODEL (Epoch 93, val_loss: 0.013872):")
    print(f"   PSNR:  {np.mean([m['PSNR'] for m in all_metrics_600]):.2f} ± {np.std([m['PSNR'] for m in all_metrics_600]):.2f} dB")
    print(f"   SSIM:  {np.mean([m['SSIM'] for m in all_metrics_600]):.3f} ± {np.std([m['SSIM'] for m in all_metrics_600]):.3f}")
    print(f"   SAM:   {np.mean([m['SAM'] for m in all_metrics_600]):.2f} ± {np.std([m['SAM'] for m in all_metrics_600]):.2f}°")
    
    # Winner determination
    print("\n🏆 WINNER:")
    psnr_100 = np.mean([m['PSNR'] for m in all_metrics_100])
    psnr_600 = np.mean([m['PSNR'] for m in all_metrics_600])
    ssim_100 = np.mean([m['SSIM'] for m in all_metrics_100])
    ssim_600 = np.mean([m['SSIM'] for m in all_metrics_600])
    
    if psnr_600 > psnr_100 and ssim_600 > ssim_100:
        print("   600-sample model performs better! 🎉")
        print("   Despite higher validation loss, it produces better SR quality.")
    elif psnr_100 > psnr_600 and ssim_100 > ssim_600:
        print("   100-sample model performs better!")
        print("   Lower validation loss correlates with better SR quality.")
    else:
        print("   Mixed results - models excel in different metrics.")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
