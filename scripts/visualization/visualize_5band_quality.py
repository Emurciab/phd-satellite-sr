#!/usr/bin/env python3
"""
Visualize 5-band formatted samples for quality assessment.

For each sample, creates a figure showing:
- Row 1: HR frames (Planet 5-band as RGB composite) - 3 temporal frames
- Row 2: LR_10m frames (Sentinel-2 10m bands as RGB) - 3 temporal frames  
- Row 3: LR_20m frames (Sentinel-2 20m bands) - 3 temporal frames (upscaled for display)

This allows visual inspection of:
1. Temporal consistency between frames
2. LR-HR alignment/registration quality
3. Cloud/artifact detection
4. Overall sample quality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TrainingSample5Band:
    """5-band training sample structure"""
    HR: np.ndarray = None           # (T, 256, 256, 5) - Planet B2,B4,B6,B7,B8
    HR_10m: np.ndarray = None       # (T, 256, 256, 3) - S2 10m bands at HR resolution
    HR_20m: np.ndarray = None       # (T, 128, 128, 2) - S2 20m bands at native res
    LR_10m: np.ndarray = None       # (T, 77, 77, 3) - S2 10m at LR scale
    LR_20m: np.ndarray = None       # (T, 38, 38, 2) - S2 20m at LR scale
    dates: list = None
    tile_id: str = ''
    sequence_idx: int = 0
    strategy: str = '8band'


def setup_unpickle_modules():
    """Register modules needed for unpickling"""
    import types
    
    # Create fake module with the dataclass
    fake_mod = types.ModuleType('convert_cache_to_5band')
    fake_mod.TrainingSample5Band = TrainingSample5Band
    sys.modules['convert_cache_to_5band'] = fake_mod
    
    # Also register under scripts.preprocessing path
    fake_mod2 = types.ModuleType('scripts.preprocessing.convert_cache_to_5band')
    fake_mod2.TrainingSample5Band = TrainingSample5Band
    sys.modules['scripts.preprocessing.convert_cache_to_5band'] = fake_mod2


def load_chunked_dataset(dataset_dir: Path, max_samples: int = None) -> List:
    """Load samples from chunked 5-band dataset"""
    setup_unpickle_modules()
    
    samples = []
    
    # Check for chunked format (metadata.pkl + chunk_*.pkl)
    metadata_file = dataset_dir / 'metadata.pkl'
    if metadata_file.exists():
        # Chunked format
        chunk_files = sorted(dataset_dir.glob('chunk_*.pkl'))
        print(f"Found {len(chunk_files)} chunk files")
        
        for chunk_file in chunk_files:
            with open(chunk_file, 'rb') as f:
                chunk_samples = pickle.load(f)
            samples.extend(chunk_samples)
            
            if max_samples and len(samples) >= max_samples:
                samples = samples[:max_samples]
                break
    else:
        # Single file format
        pkl_files = list(dataset_dir.glob('*.pkl'))
        if pkl_files:
            with open(pkl_files[0], 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                samples = data.get('samples', [])
            elif isinstance(data, list):
                samples = data
            
            if max_samples:
                samples = samples[:max_samples]
    
    return samples


def normalize_for_display(arr: np.ndarray, percentile_clip: float = 2.0) -> np.ndarray:
    """Normalize array for display with percentile clipping"""
    if arr is None or arr.size == 0:
        return arr
    
    arr = arr.astype(np.float32)
    
    # Clip outliers
    vmin = np.percentile(arr, percentile_clip)
    vmax = np.percentile(arr, 100 - percentile_clip)
    
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
    
    return np.clip(arr, 0, 1)


def create_rgb_from_5band(hr_5band: np.ndarray) -> np.ndarray:
    """
    Create RGB composite from 5-band Planet data.
    5-band order: B2(Blue), B4(Green), B6(Red), B7(RedEdge), B8(NIR)
    RGB = B6, B4, B2 (indices 2, 1, 0)
    """
    if hr_5band.ndim == 3:
        # Single frame (H, W, 5)
        rgb = hr_5band[:, :, [2, 1, 0]]  # R, G, B
    else:
        # Temporal sequence (T, H, W, 5)
        rgb = hr_5band[:, :, :, [2, 1, 0]]
    return rgb


def create_rgb_from_lr10m(lr_10m: np.ndarray) -> np.ndarray:
    """
    Create RGB from LR_10m Sentinel-2 bands.
    LR_10m order: B2(Blue), B3(Green), B4(Red)
    RGB = B4, B3, B2 (indices 2, 1, 0)
    """
    if lr_10m.ndim == 3:
        rgb = lr_10m[:, :, [2, 1, 0]]
    else:
        rgb = lr_10m[:, :, :, [2, 1, 0]]
    return rgb


def create_nir_rededge_from_5band(hr_5band: np.ndarray) -> np.ndarray:
    """
    Create NIR/RedEdge visualization from 5-band Planet data.
    5-band order: B2(Blue), B4(Green), B6(Red), B7(RedEdge), B8(NIR)
    Show as: NIR(B8), RedEdge(B7), Red(B6) -> indices 4, 3, 2
    """
    if hr_5band.ndim == 3:
        # Single frame (H, W, 5)
        rgb = hr_5band[:, :, [4, 3, 2]]  # NIR, RedEdge, Red
    else:
        # Temporal sequence (T, H, W, 5)
        rgb = hr_5band[:, :, :, [4, 3, 2]]
    return rgb


def visualize_sample(sample, sample_idx: int, output_dir: Path, 
                     show_stats: bool = True) -> Path:
    """
    Create visualization for a single sample showing all LR and HR components.
    
    Layout:
    - Row 1: HR (Planet) RGB (B6,B4,B2) - 3 frames
    - Row 2: HR (Planet) NIR/RE (B8,B7,B6) - 3 frames  
    - Row 3: LR_10m (S2) RGB - 3 frames  
    - Row 4: LR_20m (S2 B5,B8A) - 3 frames
    """
    # Extract arrays - handle both naming conventions
    hr = getattr(sample, 'HR', None) or getattr(sample, 'hr_sequence', None)
    lr_10m = getattr(sample, 'LR_10m', None) or getattr(sample, 'lr_sequence', None)
    lr_20m = getattr(sample, 'LR_20m', None) or getattr(sample, 'lr_20m_sequence', None)
    hr_10m = getattr(sample, 'HR_10m', None) or getattr(sample, 'hr_10m_sequence', None)
    hr_20m = getattr(sample, 'HR_20m', None) or getattr(sample, 'hr_20m_sequence', None)
    dates = getattr(sample, 'dates', None) or getattr(sample, 'temporal_dates', None) or getattr(sample, 'planet_dates', None)
    tile_id = getattr(sample, 'tile_id', 'unknown')
    strategy = getattr(sample, 'strategy', None) or getattr(sample, 'sensor_type', 'unknown')
    
    if hr is None:
        print(f"  Sample {sample_idx}: No HR data, skipping")
        return None
    
    n_frames = hr.shape[0]
    
    # Create figure - now 4 rows for: HR RGB, HR NIR/RE, LR_10m, LR_20m
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(4, n_frames + 1, figure=fig, width_ratios=[1]*n_frames + [0.3],
                  hspace=0.15, wspace=0.05)
    
    # Title
    date_str = ', '.join(dates) if dates else 'N/A'
    fig.suptitle(f'Sample {sample_idx} | Tile: {tile_id} | Strategy: {strategy}\nDates: {date_str}', 
                 fontsize=12, y=0.98)
    
    # Row labels - updated to show Planet B7/B8
    row_labels = [
        'HR Planet RGB\n(B6,B4,B2)', 
        'HR Planet NIR/RE\n(B8,B7,B6)', 
        'LR S2 10m RGB\n(B4,B3,B2)', 
        'LR S2 20m\n(B5,B8A)'
    ]
    
    # Process each row - now showing Planet NIR/RedEdge instead of S2 HR_10m
    for row_idx, (data, label, rgb_func) in enumerate([
        (hr, 'HR_RGB', create_rgb_from_5band),
        (hr, 'HR_NIR_RE', create_nir_rededge_from_5band),  # Planet B7, B8
        (lr_10m, 'LR_10m', create_rgb_from_lr10m),
        (lr_20m, 'LR_20m', None),  # Special handling for 2-band
    ]):
        if data is None:
            continue
            
        for t in range(min(n_frames, data.shape[0])):
            ax = fig.add_subplot(gs[row_idx, t])
            
            if rgb_func is not None:
                # RGB visualization
                frame = data[t]
                rgb = rgb_func(frame)
                rgb_norm = normalize_for_display(rgb)
                ax.imshow(rgb_norm)
            else:
                # LR_20m: show as 2-channel composite (B5=red, B8A=green)
                frame = data[t]
                # Create pseudo-RGB: B5, B8A, average
                if frame.shape[-1] >= 2:
                    pseudo_rgb = np.stack([
                        frame[:, :, 0],  # B5 -> Red
                        frame[:, :, 1],  # B8A -> Green
                        (frame[:, :, 0] + frame[:, :, 1]) / 2  # Average -> Blue
                    ], axis=-1)
                    pseudo_rgb = normalize_for_display(pseudo_rgb)
                    ax.imshow(pseudo_rgb)
                else:
                    ax.imshow(normalize_for_display(frame[:, :, 0]), cmap='gray')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add frame label on first row
            if row_idx == 0:
                ax.set_title(f'T{t+1}', fontsize=10)
            
            # Add row label on first column
            if t == 0:
                ax.set_ylabel(f'{row_labels[row_idx]}\n{data.shape[1]}×{data.shape[2]}', 
                             fontsize=9)
        
        # Add stats panel
        if show_stats and data is not None:
            ax_stats = fig.add_subplot(gs[row_idx, n_frames])
            ax_stats.axis('off')
            
            stats_text = f"Shape: {data.shape}\n"
            stats_text += f"Min: {data.min():.3f}\n"
            stats_text += f"Max: {data.max():.3f}\n"
            stats_text += f"Mean: {data.mean():.3f}\n"
            stats_text += f"Std: {data.std():.3f}"
            
            # Check for potential issues
            zero_frac = (data == 0).sum() / data.size
            if zero_frac > 0.1:
                stats_text += f"\n⚠️ {zero_frac*100:.1f}% zeros"
            
            nan_frac = np.isnan(data).sum() / data.size
            if nan_frac > 0:
                stats_text += f"\n⚠️ {nan_frac*100:.1f}% NaN"
            
            ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                         fontsize=8, verticalalignment='center', fontfamily='monospace')
    
    # Save figure
    output_path = output_dir / f'sample_{sample_idx:04d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Visualize 5-band samples for quality check')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to 5-band dataset directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to visualize (default: 100)')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Starting sample index (default: 0)')
    parser.add_argument('--random', action='store_true',
                        help='Randomly select samples instead of sequential')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("5-BAND SAMPLE QUALITY VISUALIZATION")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Samples: {args.num_samples} (starting at {args.start_idx})")
    print("=" * 70)
    
    # Load samples
    print("\n📂 Loading dataset...")
    samples = load_chunked_dataset(input_dir, max_samples=args.start_idx + args.num_samples + 100)
    print(f"   Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        print("❌ No samples found!")
        return
    
    # Select samples to visualize
    if args.random:
        np.random.seed(args.seed)
        indices = np.random.choice(len(samples), min(args.num_samples, len(samples)), replace=False)
        indices = sorted(indices)
    else:
        indices = range(args.start_idx, min(args.start_idx + args.num_samples, len(samples)))
    
    print(f"\n🎨 Generating {len(list(indices))} visualizations...")
    
    success_count = 0
    error_count = 0
    
    for i, idx in enumerate(indices):
        try:
            sample = samples[idx]
            output_path = visualize_sample(sample, idx, output_dir)
            
            if output_path:
                success_count += 1
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i+1}/{len(list(indices))} samples")
        except Exception as e:
            error_count += 1
            print(f"   ⚠️ Error on sample {idx}: {e}")
    
    print(f"\n✅ Completed: {success_count} visualizations saved")
    if error_count > 0:
        print(f"⚠️  Errors: {error_count}")
    print(f"📁 Output: {output_dir}")


if __name__ == '__main__':
    main()
