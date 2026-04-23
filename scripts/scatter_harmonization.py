#!/usr/bin/env python3
"""
Scatter plot: Planet vs Sentinel-2 reflectance per band, before/after harmonization.

For each spectral band shared between Planet SuperDove and Sentinel-2,
we compare pixel reflectance values:
  - Left column:  BEFORE harmonization (raw Planet HR vs S2 LR)
  - Right column: AFTER  harmonization (histogram-matched Planet HR vs S2 LR)

The S2 LR pixels are upsampled (nearest neighbour) to the Planet HR grid
so that every HR pixel has a paired LR value.

Band mapping (5-band SuperDove mode):
  Planet B2  ↔ S2 B2  (Blue)      — LR 10 m, idx 0
  Planet B4  ↔ S2 B3  (Green)     — LR 10 m, idx 1
  Planet B6  ↔ S2 B4  (Red)       — LR 10 m, idx 2
  Planet B7  ↔ S2 B5  (Red Edge)  — LR 20 m, idx 0
  Planet B8  ↔ S2 B8A (NIR)       — LR 20 m, idx 1

Output: results/model_evaluation/visualizations/scatter_harmonization.png
"""

import os, sys, hashlib, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT = Path("/media/eamurciab/TOSHIBA_EXT1/PhD/SR")
CACHE   = PROJECT / "cache"
OUT_DIR = PROJECT / "results" / "model_evaluation" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Inline histogram matching (avoid heavy imports from numpy_efficient_pipeline)
def histogram_match_band(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match histogram of source band to reference band distribution."""
    try:
        from skimage.exposure import match_histograms as skimage_match
        return skimage_match(source, reference).astype(source.dtype)
    except ImportError:
        pass
    # Manual fallback
    src_flat = source.flatten()
    ref_flat = reference.flatten()
    src_valid = src_flat[src_flat > 0]
    ref_valid = ref_flat[ref_flat > 0]
    if len(src_valid) == 0 or len(ref_valid) == 0:
        return source
    src_vals, src_inv = np.unique(src_valid, return_inverse=True)
    ref_vals = np.unique(ref_valid)
    src_cdf = np.cumsum(np.bincount(src_inv)).astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(np.bincount(np.searchsorted(ref_vals, ref_valid))).astype(np.float64)
    ref_cdf /= ref_cdf[-1]
    interp_vals = np.interp(src_cdf, ref_cdf, ref_vals)
    matched = src_flat.copy()
    valid_mask = src_flat > 0
    idx = np.clip(np.searchsorted(src_vals, src_flat[valid_mask]), 0, len(interp_vals) - 1)
    matched[valid_mask] = interp_vals[idx]
    return matched.reshape(source.shape)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Which caches to use (5-band SuperDove, grid_parallel → largest tile counts)
STA_CACHES = {
    "STA-1E": "sta1e_grid_parallel",
    "STA-1W": "sta1w_grid_parallel",
    "STA-2":  "sta2_grid_parallel",       # 4-band — will skip or handle
    "STA-34": "sta34_grid_parallel",
    "STA-56": "sta56_grid_parallel",
}

# Band labels for the 5-band case
BAND_LABELS_5 = ["Blue", "Green", "Red", "Red Edge", "NIR"]
# Corresponding HR band index, LR source ('10m' or '20m'), LR band index
BAND_MAP_5 = [
    (0, "10m", 0),   # Blue   → HR band 0, LR_10m band 0
    (1, "10m", 1),   # Green  → HR band 1, LR_10m band 1
    (2, "10m", 2),   # Red    → HR band 2, LR_10m band 2
    (3, "20m", 0),   # RedEdge→ HR band 3, LR_20m band 0
    (4, "20m", 1),   # NIR    → HR band 4, LR_20m band 1
]

# Band labels for the 4-band case (STA-2, older data)
BAND_LABELS_4 = ["Blue", "Green", "Red", "NIR"]
BAND_MAP_4 = [
    (0, "10m", 0),   # Blue   → HR band 0, LR_10m band 0
    (1, "10m", 1),   # Green  → HR band 1, LR_10m band 1
    (2, "10m", 2),   # Red    → HR band 2, LR_10m band 2
    (3, "20m", 0),   # NIR    → HR band 3, LR_20m band 0
]

TILES_PER_STA = 50         # tiles to sample per STA
PIXELS_PER_TILE = 500      # random pixels to sample per tile
MAX_REFLECTANCE = 5000     # clip for display (surface reflectance ×10000 scale)

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Helper: upsample LR to HR grid (nearest neighbour)
# ---------------------------------------------------------------------------
def upsample_to_hr(lr_band: np.ndarray, hr_shape: tuple) -> np.ndarray:
    """Upsample a single LR band to HR spatial dimensions using nearest."""
    zoom_h = hr_shape[0] / lr_band.shape[0]
    zoom_w = hr_shape[1] / lr_band.shape[1]
    return zoom(lr_band, (zoom_h, zoom_w), order=0)


# ---------------------------------------------------------------------------
# Collect pixel pairs
# ---------------------------------------------------------------------------
def collect_pixels(sta_name: str, cache_name: str):
    """
    For a given STA cache, sample tiles and return per-band pixel pairs
    (planet_raw, planet_harm, s2) for each matching band.
    """
    raw_dir  = CACHE / cache_name / "raw_downloads"
    if not raw_dir.exists():
        print(f"  ⚠️  {sta_name}: raw_downloads not found — skipping")
        return None

    raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.npz')])
    if not raw_files:
        return None

    # Determine band layout from first file
    sample = np.load(raw_dir / raw_files[0], allow_pickle=True)
    n_hr_bands = sample['hr_array'].shape[2]
    is_5band = (n_hr_bands == 5)
    band_map = BAND_MAP_5 if is_5band else BAND_MAP_4
    band_labels = BAND_LABELS_5 if is_5band else BAND_LABELS_4
    n_bands = len(band_map)

    print(f"  {sta_name}: {len(raw_files)} tiles, {n_hr_bands}-band, using {n_bands} matched bands")

    # Randomly sample tiles
    sampled_files = random.sample(raw_files, min(TILES_PER_STA, len(raw_files)))

    # Accumulators: per-band lists of (planet_raw, planet_harm, s2) pixel values
    raw_pixels  = [[] for _ in range(n_bands)]
    harm_pixels = [[] for _ in range(n_bands)]
    s2_pixels   = [[] for _ in range(n_bands)]

    for fname in sampled_files:
        try:
            data = np.load(raw_dir / fname, allow_pickle=True)
            hr_raw = data['hr_array']       # (H, W, C)
            lr_10m = data['lr_10m']          # (h, w, 3)
            lr_20m = data['lr_20m']          # (h2, w2, 1 or 2)
            hr_h, hr_w = hr_raw.shape[:2]
        except Exception:
            continue

        # --- Harmonize per-band (only first min(C_hr, C_lr_10m) bands, like pipeline) ---
        num_harm = min(hr_raw.shape[2], lr_10m.shape[2])
        hr_harm = hr_raw.copy()
        for bi in range(num_harm):
            hr_harm[:, :, bi] = histogram_match_band(hr_raw[:, :, bi], lr_10m[:, :, bi])

        # --- Sample pixels for each matched band ---
        for band_idx, (hr_bi, lr_src, lr_bi) in enumerate(band_map):
            # Get LR band at original resolution
            if lr_src == "10m":
                lr_band = lr_10m[:, :, lr_bi]
            else:
                if lr_20m.ndim == 2:
                    lr_band = lr_20m
                else:
                    if lr_bi >= lr_20m.shape[2]:
                        continue  # band not available
                    lr_band = lr_20m[:, :, lr_bi]

            # Upsample LR to HR grid
            s2_up = upsample_to_hr(lr_band, (hr_h, hr_w))

            # Planet bands (raw and harmonized)
            p_raw  = hr_raw[:, :, hr_bi]
            p_harm = hr_harm[:, :, hr_bi] if hr_bi < num_harm else hr_raw[:, :, hr_bi]

            # Random pixel indices (avoid zero/nodata)
            valid = (p_raw > 0) & (s2_up > 0) & (p_raw < MAX_REFLECTANCE * 1.5) & (s2_up < MAX_REFLECTANCE * 1.5)
            valid_idx = np.argwhere(valid)
            if len(valid_idx) < 10:
                continue
            chosen = valid_idx[np.random.choice(len(valid_idx), min(PIXELS_PER_TILE, len(valid_idx)), replace=False)]
            rows, cols = chosen[:, 0], chosen[:, 1]

            raw_pixels[band_idx].append(p_raw[rows, cols])
            harm_pixels[band_idx].append(p_harm[rows, cols])
            s2_pixels[band_idx].append(s2_up[rows, cols])

    # Concatenate
    result = []
    for bi in range(n_bands):
        if raw_pixels[bi]:
            result.append({
                'raw':  np.concatenate(raw_pixels[bi]),
                'harm': np.concatenate(harm_pixels[bi]),
                's2':   np.concatenate(s2_pixels[bi]),
            })
        else:
            result.append(None)

    return band_labels, result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_scatter_plot():
    """Create the full scatter figure."""

    # Collect data per STA
    all_data = {}  # {sta_name: (band_labels, [per_band_dict])}
    for sta_name, cache_name in STA_CACHES.items():
        print(f"Processing {sta_name}...")
        out = collect_pixels(sta_name, cache_name)
        if out is not None:
            all_data[sta_name] = out

    if not all_data:
        print("No data collected!")
        return

    # Use the 5-band layout if any STA has 5 bands; otherwise 4-band
    max_bands = max(len(v[0]) for v in all_data.values())
    band_labels_use = BAND_LABELS_5 if max_bands == 5 else BAND_LABELS_4
    n_bands = len(band_labels_use)

    # Merge all STAs into combined per-band arrays
    raw_all  = [[] for _ in range(n_bands)]
    harm_all = [[] for _ in range(n_bands)]
    s2_all   = [[] for _ in range(n_bands)]
    sta_labels_all = [[] for _ in range(n_bands)]

    for sta_name, (band_labels, band_data) in all_data.items():
        for bi, lbl in enumerate(band_labels):
            # Map this STA's band to the global band index
            if lbl in band_labels_use:
                global_bi = band_labels_use.index(lbl)
                if band_data[bi] is not None:
                    n_pix = len(band_data[bi]['raw'])
                    raw_all[global_bi].append(band_data[bi]['raw'])
                    harm_all[global_bi].append(band_data[bi]['harm'])
                    s2_all[global_bi].append(band_data[bi]['s2'])
                    sta_labels_all[global_bi].append(np.full(n_pix, sta_name))

    # Concatenate
    for bi in range(n_bands):
        if raw_all[bi]:
            raw_all[bi]  = np.concatenate(raw_all[bi])
            harm_all[bi] = np.concatenate(harm_all[bi])
            s2_all[bi]   = np.concatenate(s2_all[bi])
            sta_labels_all[bi] = np.concatenate(sta_labels_all[bi])
        else:
            raw_all[bi] = harm_all[bi] = s2_all[bi] = sta_labels_all[bi] = None

    # -----------------------------------------------------------------------
    # Figure: n_bands rows × 2 columns (Before | After)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(n_bands, 2, figsize=(10, 4 * n_bands),
                             sharex=False, sharey=False)
    if n_bands == 1:
        axes = axes.reshape(1, -1)

    # Color map for density
    cmap = plt.cm.inferno.copy()

    for bi in range(n_bands):
        for col, (planet_data, title_prefix) in enumerate([
            (raw_all[bi], "Before"),
            (harm_all[bi], "After"),
        ]):
            ax = axes[bi, col]

            if planet_data is None or s2_all[bi] is None:
                ax.text(0.5, 0.5, "No data", ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_xlim(0, MAX_REFLECTANCE)
                ax.set_ylim(0, MAX_REFLECTANCE)
            else:
                x = s2_all[bi]
                y = planet_data

                # Clip for plotting
                mask = (x > 0) & (y > 0) & (x < MAX_REFLECTANCE) & (y < MAX_REFLECTANCE)
                x, y = x[mask], y[mask]

                # Subsample if too many points for hexbin
                if len(x) > 200_000:
                    idx = np.random.choice(len(x), 200_000, replace=False)
                    x, y = x[idx], y[idx]

                # 2D histogram / density scatter
                hb = ax.hexbin(x, y, gridsize=120, cmap=cmap,
                               mincnt=1, linewidths=0.2,
                               norm=mcolors.LogNorm())

                # 1:1 line
                lim = max(x.max(), y.max()) * 1.05
                lim = min(lim, MAX_REFLECTANCE)
                ax.plot([0, lim], [0, lim], 'w--', lw=1.2, alpha=0.8, label='1:1')

                # R² and RMSE
                r, _ = pearsonr(x, y)
                rmse = np.sqrt(np.mean((y - x) ** 2))
                bias = np.mean(y - x)
                ax.text(0.04, 0.96,
                        f"R² = {r**2:.3f}\nRMSE = {rmse:.0f}\nBias = {bias:+.0f}",
                        transform=ax.transAxes, fontsize=9,
                        va='top', ha='left',
                        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2))
                ax.text(0.04, 0.04,
                        f"n = {len(x):,}",
                        transform=ax.transAxes, fontsize=8,
                        va='bottom', ha='left', color='white')

                ax.set_xlim(0, lim)
                ax.set_ylim(0, lim)
                ax.set_aspect('equal')

            # Labels
            if col == 0:
                ax.set_ylabel(f"{band_labels_use[bi]}\nPlanet reflectance", fontsize=11)
            if bi == n_bands - 1:
                ax.set_xlabel("Sentinel-2 reflectance", fontsize=11)
            if bi == 0:
                ax.set_title(f"{title_prefix} harmonization", fontsize=13, fontweight='bold')

    plt.tight_layout(h_pad=1.5, w_pad=1.0)
    out_path = OUT_DIR / "scatter_harmonization.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n✅ Saved to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    make_scatter_plot()
