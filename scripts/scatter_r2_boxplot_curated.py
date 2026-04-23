#!/usr/bin/env python3
"""
Option B — Per-tile R² box plots using ONLY curated training/test data.

Uses the mmap arrays from data/sr_training/all_stas_combined/.
Data is already harmonized + quality-filtered.

Shows the distribution of within-tile R² for each of the 5 bands.

Output: r2_boxplot_curated.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
COL_MAIN = '#2196F3'

np.random.seed(42)


def compute_per_tile_r2(split='train', n_samples=1000):
    """Compute per-tile R² for each band."""
    d = DATA / f"mmap_{split}"
    
    # Try clean version first
    clean_dir = DATA / f"mmap_{split}_clean"
    if clean_dir.exists():
        d = clean_dir
        print(f"  Using {d.name}")
    
    hr_10m = np.load(d / "hr_10m.npy", mmap_mode='r')
    lr_10m = np.load(d / "lr.npy",     mmap_mode='r')
    hr_20m = np.load(d / "hr_20m.npy", mmap_mode='r')
    lr_20m = np.load(d / "lr_20m.npy", mmap_mode='r')
    
    N = hr_10m.shape[0]
    idxs = np.random.choice(N, min(n_samples, N), replace=False)
    
    r2_per_band = {b: [] for b in range(5)}
    
    for i in idxs:
        # 10m bands
        for band in range(3):
            h = np.array(hr_10m[i, 0, :, :, band], dtype=np.float64)
            l = np.array(lr_10m[i, 0, :, :, band], dtype=np.float64)
            l_up = zoom(l, (256/77, 256/77), order=0)[:256, :256]
            
            valid = (h > 0.001) & (l_up > 0.001) & ~np.isnan(h) & ~np.isnan(l_up)
            if valid.sum() < 100:
                continue
            hv, sv = h[valid], l_up[valid]
            if hv.std() < 1e-6 or sv.std() < 1e-6:
                continue
            r, _ = pearsonr(sv, hv)
            r2_per_band[band].append(r**2)
        
        # 20m bands
        for band_20 in range(2):
            h = np.array(hr_20m[i, 0, :, :, band_20], dtype=np.float64)
            l = np.array(lr_20m[i, 0, :, :, band_20], dtype=np.float64)
            l_up = zoom(l, (128/38, 128/38), order=0)[:128, :128]
            
            valid = (h > 0.001) & (l_up > 0.001) & ~np.isnan(h) & ~np.isnan(l_up)
            if valid.sum() < 50:
                continue
            hv, sv = h[valid], l_up[valid]
            if hv.std() < 1e-6 or sv.std() < 1e-6:
                continue
            r, _ = pearsonr(sv, hv)
            r2_per_band[3 + band_20].append(r**2)
    
    return r2_per_band


def main():
    print("Computing per-tile R² from curated training data...")
    r2_data = compute_per_tile_r2('train', n_samples=1000)
    
    # Print summary
    print(f"\n{'Band':<12s} {'n_tiles':>8s} {'Median R²':>10s} {'Mean R²':>10s} {'Std':>8s}")
    for b in range(5):
        arr = np.array(r2_data[b])
        if len(arr) > 0:
            print(f"{BAND_LABELS[b]:<12s} {len(arr):8d} {np.median(arr):10.3f} {np.mean(arr):10.3f} {np.std(arr):8.3f}")
        else:
            print(f"{BAND_LABELS[b]:<12s}        0         -          -        -")
    
    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    
    positions = list(range(5))
    box_data = [np.array(r2_data[b]) if r2_data[b] else np.array([np.nan]) for b in range(5)]
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=2),
                    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
    
    for patch in bp['boxes']:
        patch.set_facecolor(COL_MAIN)
        patch.set_alpha(0.6)
    
    # Strip dots
    for b in range(5):
        arr = np.array(r2_data[b])
        if len(arr) > 0:
            jitter = np.random.uniform(-0.18, 0.18, len(arr))
            ax.scatter(positions[b] + jitter, arr, c=COL_MAIN, s=8,
                       alpha=0.25, edgecolors='none', zorder=3)
            
            med = np.median(arr)
            ax.text(positions[b], med + 0.04, f"{med:.2f}",
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#0D47A1')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(BAND_LABELS, fontsize=11)
    ax.set_ylabel("Within-tile R²  (Planet harmonized vs S2)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Curated training data — Per-tile spatial R²  (harmonized Planet vs Sentinel-2)",
                 fontsize=13, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add count annotation
    for b in range(5):
        arr = np.array(r2_data[b])
        ax.text(positions[b], -0.02, f"n={len(arr)}",
                ha='center', va='top', fontsize=7, color='grey')
    
    plt.tight_layout()
    out_path = OUT_DIR / "r2_boxplot_curated.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n✅ {out_path.name}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
