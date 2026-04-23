#!/usr/bin/env python3
"""
Option A — Normalized scatter: Planet vs S2 reflectance using ONLY curated
training/test data (post-harmonization, post-quality-filtering).

Uses the mmap arrays from data/sr_training/all_stas_combined/.
Data is already normalized [0,1], harmonized, and coregistered.

Layout: 5 bands × 1 column (all curated data pooled).
Also produces per-tile normalized variant.

Output:
  scatter_curated.png          — raw pooled scatter (harmonized)
  scatter_curated_normalized.png — per-tile mean-centred
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.ndimage import zoom
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────
PROJECT = Path("/media/eamurciab/TOSHIBA_EXT1/PhD/SR")
DATA    = PROJECT / "data" / "sr_training" / "all_stas_combined"
OUT_DIR = PROJECT / "results" / "model_evaluation" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAND_LABELS = ["Blue", "Green", "Red", "Red Edge", "NIR"]

np.random.seed(42)


def load_curated_data(split='train'):
    """Load mmap arrays for a split."""
    d = DATA / f"mmap_{split}"
    hr_10m = np.load(d / "hr_10m.npy", mmap_mode='r')  # (N, 3, 256, 256, 3)
    lr_10m = np.load(d / "lr.npy",     mmap_mode='r')   # (N, 3, 77, 77, 3)
    hr_20m = np.load(d / "hr_20m.npy", mmap_mode='r')   # (N, 3, 128, 128, 2)
    lr_20m = np.load(d / "lr_20m.npy", mmap_mode='r')   # (N, 3, 38, 38, 2)
    return hr_10m, lr_10m, hr_20m, lr_20m


def collect_band_pixels(hr_10m, lr_10m, hr_20m, lr_20m,
                        n_samples=800, pixels_per_sample=300,
                        normalize_per_tile=False):
    """
    Collect pixel pairs for each of the 5 bands.
    
    Returns dict: band_idx -> (planet_pixels, s2_pixels)
    """
    N = hr_10m.shape[0]
    idxs = np.random.choice(N, min(n_samples, N), replace=False)
    
    # 5 bands: 3 from 10m + 2 from 20m
    all_hr = {b: [] for b in range(5)}
    all_s2 = {b: [] for b in range(5)}
    per_tile_r2 = {b: [] for b in range(5)}
    
    for i in idxs:
        # Use temporal frame 0
        # --- 10m bands (Blue, Green, Red) ---
        for band in range(3):
            h = np.array(hr_10m[i, 0, :, :, band], dtype=np.float64)
            l = np.array(lr_10m[i, 0, :, :, band], dtype=np.float64)
            
            # Upsample LR (77→256)
            l_up = zoom(l, (256/77, 256/77), order=0)[:256, :256]
            
            valid = (h > 0.001) & (l_up > 0.001) & ~np.isnan(h) & ~np.isnan(l_up)
            n_valid = valid.sum()
            if n_valid < 50:
                continue
            
            hv, sv = h[valid], l_up[valid]
            
            # Per-tile R²
            if hv.std() > 1e-6 and sv.std() > 1e-6:
                r, _ = pearsonr(sv, hv)
                per_tile_r2[band].append(r**2)
            
            if normalize_per_tile:
                hv = hv - hv.mean()
                sv = sv - sv.mean()
            
            idx_px = np.random.choice(len(hv), min(pixels_per_sample, len(hv)), replace=False)
            all_hr[band].append(hv[idx_px])
            all_s2[band].append(sv[idx_px])
        
        # --- 20m bands (Red Edge, NIR) ---
        for band_20 in range(2):
            band_global = 3 + band_20  # indices 3, 4
            h = np.array(hr_20m[i, 0, :, :, band_20], dtype=np.float64)
            l = np.array(lr_20m[i, 0, :, :, band_20], dtype=np.float64)
            
            # Upsample LR (38→128)
            l_up = zoom(l, (128/38, 128/38), order=0)[:128, :128]
            
            valid = (h > 0.001) & (l_up > 0.001) & ~np.isnan(h) & ~np.isnan(l_up)
            n_valid = valid.sum()
            if n_valid < 50:
                continue
            
            hv, sv = h[valid], l_up[valid]
            
            if hv.std() > 1e-6 and sv.std() > 1e-6:
                r, _ = pearsonr(sv, hv)
                per_tile_r2[band_global].append(r**2)
            
            if normalize_per_tile:
                hv = hv - hv.mean()
                sv = sv - sv.mean()
            
            idx_px = np.random.choice(len(hv), min(pixels_per_sample, len(hv)), replace=False)
            all_hr[band_global].append(hv[idx_px])
            all_s2[band_global].append(sv[idx_px])
    
    # Concatenate
    result = {}
    for b in range(5):
        if all_hr[b]:
            result[b] = {
                'hr': np.concatenate(all_hr[b]),
                's2': np.concatenate(all_s2[b]),
                'per_tile_r2': np.array(per_tile_r2[b]) if per_tile_r2[b] else np.array([]),
            }
        else:
            result[b] = None
    return result


def make_scatter_figure(data, title, out_name, normalized=False):
    """Create a 5-band scatter figure."""
    cmap = plt.cm.inferno.copy()
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    
    for bi in range(5):
        ax = axes[bi]
        bd = data[bi]
        
        if bd is None or len(bd['hr']) < 100:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='gray')
            ax.set_title(BAND_LABELS[bi], fontsize=12, fontweight='bold')
            continue
        
        x = bd['s2']
        y = bd['hr']
        
        if len(x) > 300_000:
            idx = np.random.choice(len(x), 300_000, replace=False)
            x, y = x[idx], y[idx]
        
        if normalized:
            absmax = max(np.percentile(np.abs(x), 99.5),
                         np.percentile(np.abs(y), 99.5)) * 1.1
            ax.hexbin(x, y, gridsize=120, cmap=cmap, mincnt=1,
                      linewidths=0.15, norm=mcolors.LogNorm())
            ax.plot([-absmax, absmax], [-absmax, absmax], 'w--', lw=1.0, alpha=0.8)
            ax.axhline(0, color='w', lw=0.3, alpha=0.4)
            ax.axvline(0, color='w', lw=0.3, alpha=0.4)
            ax.set_xlim(-absmax, absmax)
            ax.set_ylim(-absmax, absmax)
        else:
            lim = min(max(np.percentile(x, 99.5), np.percentile(y, 99.5)) * 1.1, 1.0)
            ax.hexbin(x, y, gridsize=120, cmap=cmap, mincnt=1,
                      linewidths=0.15, norm=mcolors.LogNorm())
            ax.plot([0, lim], [0, lim], 'w--', lw=1.0, alpha=0.8)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
        
        ax.set_aspect('equal')
        
        # Stats
        r, _ = pearsonr(x, y)
        rmse = np.sqrt(np.mean((y - x)**2))
        bias = np.mean(y - x)
        n_tiles = len(bd['per_tile_r2'])
        med_r2 = np.median(bd['per_tile_r2']) if n_tiles > 0 else float('nan')
        
        stats_text = f"R²={r**2:.3f}\nRMSE={rmse:.4f}\nBias={bias:+.4f}"
        if n_tiles > 0:
            stats_text += f"\nMed. tile R²={med_r2:.3f}"
        
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes, fontsize=8, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5))
        ax.text(0.05, 0.05, f"n={len(x):,} ({n_tiles} tiles)",
                transform=ax.transAxes, fontsize=7, va='bottom', ha='left', color='white')
        
        ax.set_title(BAND_LABELS[bi], fontsize=12, fontweight='bold')
        
        prefix = "Δ " if normalized else ""
        ax.set_xlabel(f"{prefix}Sentinel-2 refl.", fontsize=9)
        if bi == 0:
            ax.set_ylabel(f"{prefix}Planet refl. (harmonized)", fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ {out_path.name}  ({out_path.stat().st_size/1024:.0f} KB)")


def main():
    print("Loading curated training + test data...")
    
    # Load both splits
    hr_10m_tr, lr_10m_tr, hr_20m_tr, lr_20m_tr = load_curated_data('train')
    hr_10m_te, lr_10m_te, hr_20m_te, lr_20m_te = load_curated_data('test')
    
    # Also load clean training data
    clean_dir = DATA / "mmap_train_clean"
    if clean_dir.exists():
        print("  Using mmap_train_clean (NaN-filtered)")
        hr_10m_tr = np.load(clean_dir / "hr_10m.npy", mmap_mode='r')
        lr_10m_tr = np.load(clean_dir / "lr.npy", mmap_mode='r')
        hr_20m_tr = np.load(clean_dir / "hr_20m.npy", mmap_mode='r')
        lr_20m_tr = np.load(clean_dir / "lr_20m.npy", mmap_mode='r')
    
    n_train = hr_10m_tr.shape[0]
    n_test  = hr_10m_te.shape[0]
    print(f"  Train: {n_train} samples, Test: {n_test} samples")
    print(f"  HR 10m: {hr_10m_tr.shape}, LR 10m: {lr_10m_tr.shape}")
    print(f"  HR 20m: {hr_20m_tr.shape}, LR 20m: {lr_20m_tr.shape}")
    
    # --- Figure 1: Raw scatter (curated, harmonized) ---
    print("\nCollecting pixels for raw scatter...")
    data_raw = collect_band_pixels(hr_10m_tr, lr_10m_tr, hr_20m_tr, lr_20m_tr,
                                    n_samples=800, pixels_per_sample=300,
                                    normalize_per_tile=False)
    make_scatter_figure(data_raw,
                        f"Curated training data — Planet (harmonized) vs Sentinel-2  "
                        f"[{n_train} samples]",
                        "scatter_curated.png",
                        normalized=False)
    
    # --- Figure 2: Normalized scatter (per-tile mean-centred) ---
    print("\nCollecting pixels for normalized scatter...")
    data_norm = collect_band_pixels(hr_10m_tr, lr_10m_tr, hr_20m_tr, lr_20m_tr,
                                     n_samples=800, pixels_per_sample=300,
                                     normalize_per_tile=True)
    make_scatter_figure(data_norm,
                        f"Curated training data — Per-tile normalized  "
                        f"[{n_train} samples]",
                        "scatter_curated_normalized.png",
                        normalized=True)
    
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
