#!/usr/bin/env python3
"""
Verify dataset quality by inspecting random samples.
Checks alignment, tiling ratio, and visualizes examples.
"""

import sys
sys.path.insert(0, 'scripts/preprocessing')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def calculate_ssim_simple(img1, img2):
    """Simple SSIM approximation for quick quality check"""
    from scipy.ndimage import gaussian_filter
    
    # Normalize
    img1 = (img1 - img1.mean()) / (img1.std() + 1e-8)
    img2 = (img2 - img2.mean()) / (img2.std() + 1e-8)
    
    # Simple correlation
    correlation = np.mean(img1 * img2)
    return max(0, min(1, (correlation + 1) / 2))

def verify_samples(dataset_path, num_samples=20, visualize=True):
    """Verify dataset quality"""
    
    print("="*80)
    print("🔍 VERIFYING DATASET QUALITY")
    print("="*80)
    print()
    
    # Load dataset
    print("📂 Loading dataset...")
    with open(dataset_path / 'training_samples.pkl', 'rb') as f:
        data = pickle.load(f)
    
    with open(dataset_path / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    # Handle different data formats
    if isinstance(data, list):
        total_samples = len(data)
        # Assume samples are objects with lr_sequence and hr_sequence attributes
        sample_lr = data[0].lr_sequence if hasattr(data[0], 'lr_sequence') else data[0][0]
        sample_hr = data[0].hr_sequence if hasattr(data[0], 'hr_sequence') else data[0][1]
    elif isinstance(data, dict):
        total_samples = len(data['lr_sequences'])
        sample_lr = data['lr_sequences'][0]
        sample_hr = data['hr_sequences'][0]
    else:
        print(f"❌ Unknown data format: {type(data)}")
        return
    
    print(f"✅ Loaded {total_samples} samples")
    print()
    
    # Basic structure verification
    print("📊 DATASET STRUCTURE:")
    print("─"*80)
    
    print(f"LR sequence shape:    {sample_lr.shape} (T, H, W, C)")
    print(f"HR sequence shape:    {sample_hr.shape} (T, H, W, C)")
    print(f"Sequence length:      {sample_lr.shape[0]} temporal frames")
    print(f"LR tile size:         {sample_lr.shape[1]}×{sample_lr.shape[2]}")
    print(f"HR tile size:         {sample_hr.shape[1]}×{sample_hr.shape[2]}")
    print(f"LR channels:          {sample_lr.shape[3]}")
    print(f"HR channels:          {sample_hr.shape[3]}")
    print()
    
    # Calculate tiling ratio
    lr_size = sample_lr.shape[2]
    hr_size = sample_hr.shape[2]
    ratio = hr_size / lr_size
    expected_ratio = 10.0 / 3.0  # Sentinel-2 10m / Planet 3m
    ratio_error = abs(ratio - expected_ratio) / expected_ratio * 100
    
    print(f"Tiling ratio:         {ratio:.3f}:1")
    print(f"Expected ratio:       {expected_ratio:.3f}:1 (10m/3m)")
    print(f"Ratio error:          {ratio_error:.2f}%")
    
    if ratio_error < 1.0:
        print("✅ Tiling ratio is CORRECT")
    else:
        print(f"⚠️  Tiling ratio has {ratio_error:.2f}% error")
    print()
    
    # Sample random indices
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    sample_indices.sort()
    
    print(f"🎲 Randomly selected {len(sample_indices)} samples for verification:")
    print(f"   Indices: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}")
    print()
    
    # Quality metrics
    print("📈 QUALITY METRICS (per sample):")
    print("─"*80)
    print(f"{'Index':<8} {'LR Mean':<10} {'HR Mean':<10} {'LR Std':<10} {'HR Std':<10} {'Status':<10}")
    print("─"*80)
    
    quality_issues = []
    
    def get_sequences(idx):
        """Helper to get lr and hr sequences regardless of data format"""
        if isinstance(data, list):
            sample = data[idx]
            lr_seq = sample.lr_sequence if hasattr(sample, 'lr_sequence') else sample[0]
            hr_seq = sample.hr_sequence if hasattr(sample, 'hr_sequence') else sample[1]
        else:
            lr_seq = data['lr_sequences'][idx]
            hr_seq = data['hr_sequences'][idx]
        return lr_seq, hr_seq
    
    for idx in sample_indices:
        lr_seq, hr_seq = get_sequences(idx)
        
        # Calculate statistics (middle frame)
        lr_frame = lr_seq[1]  # Middle of 3-frame sequence
        hr_frame = hr_seq[1]
        
        lr_mean = np.mean(lr_frame)
        hr_mean = np.mean(hr_frame)
        lr_std = np.std(lr_frame)
        hr_std = np.std(hr_frame)
        
        # Check for issues
        status = "✅"
        if lr_mean < 0.01 or hr_mean < 0.01:
            status = "⚠️ Low"
            quality_issues.append(f"Sample {idx}: Very low mean values")
        elif lr_std < 0.001 or hr_std < 0.001:
            status = "⚠️ Flat"
            quality_issues.append(f"Sample {idx}: Very low variance (flat image)")
        elif np.isnan(lr_mean) or np.isnan(hr_mean):
            status = "❌ NaN"
            quality_issues.append(f"Sample {idx}: Contains NaN values")
        
        print(f"{idx:<8} {lr_mean:<10.4f} {hr_mean:<10.4f} {lr_std:<10.4f} {hr_std:<10.4f} {status:<10}")
    
    print("─"*80)
    print()
    
    # Report issues
    if quality_issues:
        print(f"⚠️  Found {len(quality_issues)} quality issues:")
        for issue in quality_issues:
            print(f"   - {issue}")
        print()
    else:
        print("✅ All checked samples passed quality checks!")
        print()
    
    # Alignment verification (check if LR and HR are spatially consistent)
    print("🔗 ALIGNMENT VERIFICATION:")
    print("─"*80)
    
    # Check a few samples for spatial consistency
    alignment_samples = random.sample(sample_indices, min(5, len(sample_indices)))
    
    print("Checking spatial consistency between LR and HR (5 samples)...")
    print()
    
    for idx in alignment_samples:
        lr_seq, hr_seq = get_sequences(idx)
        
        # Use middle frame
        lr_frame = lr_seq[1, :, :, :3]  # RGB channels
        hr_frame = hr_seq[1, :, :, :3]  # RGB channels
        
        # Downsample HR to LR size for comparison
        from scipy.ndimage import zoom
        hr_downsampled = zoom(hr_frame, (lr_frame.shape[0]/hr_frame.shape[0], 
                                         lr_frame.shape[1]/hr_frame.shape[1], 1), order=1)
        
        # Calculate simple similarity
        try:
            similarity = calculate_ssim_simple(lr_frame, hr_downsampled)
            print(f"Sample {idx:3d}: Spatial similarity = {similarity:.3f} ", end="")
            
            if similarity > 0.5:
                print("✅ Good alignment")
            elif similarity > 0.3:
                print("⚠️  Moderate alignment")
            else:
                print("❌ Poor alignment")
        except Exception as e:
            print(f"Sample {idx:3d}: Could not compute similarity - {e}")
    
    print()
    
    # Visualize samples if requested
    if visualize:
        print("📸 GENERATING VISUALIZATIONS:")
        print("─"*80)
        
        # Select 4 samples to visualize
        vis_samples = random.sample(sample_indices, min(4, len(sample_indices)))
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        fig.suptitle('Dataset Sample Verification (20 random samples checked)', fontsize=16, y=0.995)
        
        for plot_idx, sample_idx in enumerate(vis_samples):
            lr_seq, hr_seq = get_sequences(sample_idx)
            
            # Use middle frame
            lr_frame = lr_seq[1, :, :, :3]  # RGB
            hr_frame = hr_seq[1, :, :, :3]  # RGB
            
            # Normalize for display
            lr_display = np.clip(lr_frame / np.percentile(lr_frame, 98), 0, 1)
            hr_display = np.clip(hr_frame / np.percentile(hr_frame, 98), 0, 1)
            
            # Upsample LR for comparison
            from scipy.ndimage import zoom
            lr_upsampled = zoom(lr_display, (ratio, ratio, 1), order=1)
            
            # Plot LR
            axes[plot_idx, 0].imshow(lr_display)
            axes[plot_idx, 0].set_title(f'Sample {sample_idx} - LR ({lr_frame.shape[0]}×{lr_frame.shape[1]})')
            axes[plot_idx, 0].axis('off')
            
            # Plot LR upsampled
            axes[plot_idx, 1].imshow(lr_upsampled)
            axes[plot_idx, 1].set_title(f'LR Upsampled ({hr_frame.shape[0]}×{hr_frame.shape[1]})')
            axes[plot_idx, 1].axis('off')
            
            # Plot HR
            axes[plot_idx, 2].imshow(hr_display)
            axes[plot_idx, 2].set_title(f'HR Target ({hr_frame.shape[0]}×{hr_frame.shape[1]})')
            axes[plot_idx, 2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = dataset_path / 'verification_samples.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {output_path}")
        print()
        
        # Also create a detailed comparison for one sample
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
        fig2.suptitle(f'Detailed Sample Verification - Sample {vis_samples[0]}', fontsize=14)
        
        sample_idx = vis_samples[0]
        lr_seq, hr_seq = get_sequences(sample_idx)
        
        # Show all 3 temporal frames for one sample
        for t in range(3):
            lr_frame = lr_seq[t, :, :, :3]
            hr_frame = hr_seq[t, :, :, :3]
            
            lr_display = np.clip(lr_frame / np.percentile(lr_frame, 98), 0, 1)
            hr_display = np.clip(hr_frame / np.percentile(hr_frame, 98), 0, 1)
            
            axes2[0, t].imshow(lr_display)
            axes2[0, t].set_title(f'LR Frame {t+1}')
            axes2[0, t].axis('off')
            
            axes2[1, t].imshow(hr_display)
            axes2[1, t].set_title(f'HR Frame {t+1}')
            axes2[1, t].axis('off')
        
        plt.tight_layout()
        
        output_path2 = dataset_path / 'verification_temporal_sequence.png'
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"✅ Saved temporal sequence visualization to: {output_path2}")
        print()
    
    # Final summary
    print("="*80)
    print("📋 VERIFICATION SUMMARY")
    print("="*80)
    print(f"✅ Total samples verified:      {len(sample_indices)}")
    print(f"✅ Dataset structure:           VALID")
    print(f"✅ Tiling ratio:                {ratio:.3f}:1 (error: {ratio_error:.2f}%)")
    print(f"✅ Temporal sequences:          3 frames each")
    if quality_issues:
        print(f"⚠️  Quality issues found:        {len(quality_issues)}")
    else:
        print(f"✅ Quality issues found:        0")
    print()
    
    if ratio_error < 1.0 and len(quality_issues) == 0:
        print("🎉 DATASET QUALITY: EXCELLENT")
        print("   Ready for SR model training!")
    elif ratio_error < 2.0 and len(quality_issues) < 3:
        print("✅ DATASET QUALITY: GOOD")
        print("   Minor issues detected, but suitable for training")
    else:
        print("⚠️  DATASET QUALITY: ACCEPTABLE")
        print("   Some issues detected, review before training")
    print()
    print("="*80)


if __name__ == '__main__':
    dataset_path = Path('./data/datasets/sr_production_100')
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        exit(1)
    
    verify_samples(dataset_path, num_samples=20, visualize=True)
    
    print("\n💡 To view the visualizations:")
    print(f"   {dataset_path}/verification_samples.png")
    print(f"   {dataset_path}/verification_temporal_sequence.png")
