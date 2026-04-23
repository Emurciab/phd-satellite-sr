#!/usr/bin/env python3
"""
Scatter plots: Planet vs Sentinel-2 reflectance — one figure per STA, per year.

For each STA, produces a figure with:
  - Rows  = spectral bands (Blue, Green, Red, [Red Edge, NIR])
  - Columns = 2 per year: Before | After harmonization

Band mapping (5-band SuperDove):
  Planet B2  ↔ S2 B2  (Blue)      — LR 10 m
  Planet B4  ↔ S2 B3  (Green)     — LR 10 m
  Planet B6  ↔ S2 B4  (Red)       — LR 10 m
  Planet B7  ↔ S2 B5  (Red Edge)  — LR 20 m
  Planet B8  ↔ S2 B8A (NIR)       — LR 20 m

Band mapping (4-band Dove):
  Planet B1  ↔ S2 B2  (Blue)      — LR 10 m
  Planet B2  ↔ S2 B3  (Green)     — LR 10 m
  Planet B3  ↔ S2 B4  (Red)       — LR 10 m
  Planet B4  ↔ S2 B8A (NIR)       — LR 20 m

Output: results/model_evaluation/visualizations/scatter_harmonization_{sta}.png
"""

import os, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom
from scipy.stats import pearsonr

# ── Project paths ──────────────────────────────────────────────────
PROJECT = Path("/media/eamurciab/TOSHIBA_EXT1/PhD/SR")
CACHE   = PROJECT / "cache"
OUT_DIR = PROJECT / "results" / "model_evaluation" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Histogram matching (self-contained) ───────────────────────────
def histogram_match_band(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match histogram of source band to reference band distribution."""
    try:
        from skimage.exposure import match_histograms as skimage_match
        return skimage_match(source, reference).astype(source.dtype)
    except ImportError:
        pass
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
    ref_cdf = np.cumsum(np.bincount(
        np.searchsorted(ref_vals, ref_valid))).astype(np.float64)
    ref_cdf /= ref_cdf[-1]
    interp_vals = np.interp(src_cdf, ref_cdf, ref_vals)
    matched = src_flat.copy()
    valid_mask = src_flat > 0
    idx = np.clip(np.searchsorted(src_vals, src_flat[valid_mask]),
                  0, len(interp_vals) - 1)
    matched[valid_mask] = interp_vals[idx]
    return matched.reshape(source.shape)

# ── Band configuration ────────────────────────────────────────────
BAND_LABELS_5 = ["Blue", "Green", "Red", "Red Edge", "NIR"]
BAND_MAP_5 = [
    (0, "10m", 0), (1, "10m", 1), (2, "10m", 2),
    (3, "20m", 0), (4, "20m", 1),
]
BAND_LABELS_4 = ["Blue", "Green", "Red", "NIR"]
BAND_MAP_4 = [
    (0, "10m", 0), (1, "10m", 1), (2, "10m", 2),
    (3, "20m", 0),
]

# ── STA → list of (year_label, cache_directory) ──────────────────
STA_YEARS = {
    "STA-1E": [
        ("2022",      "sta1e_2022"),           # 4-band, 533 tiles
        ("2023–2024", "sta1e_grid_parallel"),   # 5-band, 1257 tiles
    ],
    "STA-1W": [
        ("2023–2024", "sta1w_grid_parallel"),   # 5-band, 579 tiles
    ],
    "STA-2": [
        ("2020–2024", "sta2_grid_parallel"),    # 4-band, 5584 tiles
    ],
    "STA-34": [
        ("2023",      "sta34_2023"),            # 5-band, 1037 tiles
        ("2023–2024", "sta34_grid_parallel"),    # 5-band, 1336 tiles
    ],
    "STA-56": [
        ("2023",      "sta56_2023"),            # 5-band, 2454 tiles
        ("2024",      "sta56_2024"),            # 5-band, 1749 tiles
    ],
}

TILES_PER_YEAR  = 50
PIXELS_PER_TILE = 500
MAX_REFL        = 5000

random.seed(42)
np.random.seed(42)

# ── Helpers ────────────────────────────────────────────────────────
def upsample_to_hr(lr_band: np.ndarray, hr_shape: tuple) -> np.ndarray:
    zoom_h = hr_shape[0] / lr_band.shape[0]
    zoom_w = hr_shape[1] / lr_band.shape[1]
    return zoom(lr_band, (zoom_h, zoom_w), order=0)


def collect_pixels_for_cache(cache_name: str):
    """
    Sample tiles from a cache and return per-band pixel triplets
    (planet_raw, planet_harmonized, s2).
    Returns (band_labels, [dict_per_band]) or None.
    """
    raw_dir = CACHE / cache_name / "raw_downloads"
    if not raw_dir.is_dir():
        return None
    raw_files = sorted(f for f in os.listdir(raw_dir) if f.endswith('.npz'))
    if not raw_files:
        return None

    sample = np.load(raw_dir / raw_files[0], allow_pickle=True)
    n_hr = sample['hr_array'].shape[2]
    is5 = (n_hr == 5)
    band_map    = BAND_MAP_5 if is5 else BAND_MAP_4
    band_labels = BAND_LABELS_5 if is5 else BAND_LABELS_4
    n_bands = len(band_map)

    sampled = random.sample(raw_files, min(TILES_PER_YEAR, len(raw_files)))

    raw_px  = [[] for _ in range(n_bands)]
    harm_px = [[] for _ in range(n_bands)]
    s2_px   = [[] for _ in range(n_bands)]

    for fname in sampled:
        try:
            d = np.load(raw_dir / fname, allow_pickle=True)
            hr_raw = d['hr_array']
            lr_10m = d['lr_10m']
            lr_20m = d['lr_20m']
            hr_h, hr_w = hr_raw.shape[:2]
        except Exception:
            continue

        # ── Quality gate (mirror pipeline: valid_fraction >= 0.90) ──
        hr_valid = np.sum((hr_raw > 0) & ~np.isnan(hr_raw)) / hr_raw.size
        lr_valid = np.sum((lr_10m > 0) & ~np.isnan(lr_10m)) / lr_10m.size
        if hr_valid < 0.90 or lr_valid < 0.90:
            continue

        # Harmonize first min(C_hr, C_lr10m) bands
        num_harm = min(hr_raw.shape[2], lr_10m.shape[2])
        hr_harm = hr_raw.copy()
        for bi in range(num_harm):
            hr_harm[:, :, bi] = histogram_match_band(
                hr_raw[:, :, bi], lr_10m[:, :, bi])

        for band_idx, (hr_bi, lr_src, lr_bi) in enumerate(band_map):
            if lr_src == "10m":
                lr_band = lr_10m[:, :, lr_bi]
            else:
                if lr_20m.ndim == 2:
                    lr_band = lr_20m
                else:
                    if lr_bi >= lr_20m.shape[2]:
                        continue
                    lr_band = lr_20m[:, :, lr_bi]

            s2_up  = upsample_to_hr(lr_band, (hr_h, hr_w))
            p_raw  = hr_raw[:, :, hr_bi]
            p_harm = hr_harm[:, :, hr_bi] if hr_bi < num_harm else hr_raw[:, :, hr_bi]

            valid = ((p_raw > 0) & (s2_up > 0) &
                     (p_raw < MAX_REFL * 1.5) & (s2_up < MAX_REFL * 1.5))
            vi = np.argwhere(valid)
            if len(vi) < 10:
                continue
            chosen = vi[np.random.choice(len(vi),
                        min(PIXELS_PER_TILE, len(vi)), replace=False)]
            r, c = chosen[:, 0], chosen[:, 1]
            raw_px[band_idx].append(p_raw[r, c])
            harm_px[band_idx].append(p_harm[r, c])
            s2_px[band_idx].append(s2_up[r, c])

    result = []
    for bi in range(n_bands):
        if raw_px[bi]:
            result.append({
                'raw':  np.concatenate(raw_px[bi]),
                'harm': np.concatenate(harm_px[bi]),
                's2':   np.concatenate(s2_px[bi]),
            })
        else:
            result.append(None)
    return band_labels, result


# ── Single-panel helper ───────────────────────────────────────────
def plot_panel(ax, x, y, cmap, show_stats=True):
    """Density hexbin scatter on *ax*, returns True if data was plotted."""
    mask = (x > 0) & (y > 0) & (x < MAX_REFL) & (y < MAX_REFL)
    x, y = x[mask], y[mask]
    if len(x) < 20:
        ax.text(0.5, 0.5, "Insufficient\ndata", ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')
        return False

    if len(x) > 200_000:
        idx = np.random.choice(len(x), 200_000, replace=False)
        x, y = x[idx], y[idx]

    ax.hexbin(x, y, gridsize=100, cmap=cmap, mincnt=1,
              linewidths=0.15, norm=mcolors.LogNorm())

    lim = min(max(x.max(), y.max()) * 1.05, MAX_REFL)
    ax.plot([0, lim], [0, lim], 'w--', lw=1.0, alpha=0.8)

    if show_stats:
        r, _ = pearsonr(x, y)
        rmse = np.sqrt(np.mean((y - x) ** 2))
        bias = np.mean(y - x)
        ax.text(0.05, 0.95,
                f"R²={r**2:.3f}\nRMSE={rmse:.0f}\nBias={bias:+.0f}",
                transform=ax.transAxes, fontsize=8, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5))
        ax.text(0.05, 0.05, f"n={len(x):,}", transform=ax.transAxes,
                fontsize=7, va='bottom', ha='left', color='white')

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    return True


# ── Main: one figure per STA ──────────────────────────────────────
def main():
    cmap = plt.cm.inferno.copy()

    for sta_name, year_caches in STA_YEARS.items():
        print(f"\n{'='*60}")
        print(f"  {sta_name}")
        print(f"{'='*60}")

        # Collect data for each year
        year_data = {}  # {year_label: (band_labels, [band_dict])}
        for year_label, cache_name in year_caches:
            print(f"  Collecting {year_label} ({cache_name})...")
            out = collect_pixels_for_cache(cache_name)
            if out is not None:
                year_data[year_label] = out
                print(f"    ✓ {out[0]}")

        if not year_data:
            print(f"  ⚠️  No data for {sta_name}")
            continue

        n_years = len(year_data)
        year_labels = list(year_data.keys())

        # Use the maximum band set across years
        max_bands = max(len(v[0]) for v in year_data.values())
        band_labels_global = BAND_LABELS_5 if max_bands == 5 else BAND_LABELS_4
        n_bands = len(band_labels_global)

        # Layout: rows=bands, cols=2*n_years (Before|After per year)
        n_cols = 2 * n_years
        fig, axes = plt.subplots(n_bands, n_cols,
                                 figsize=(4.2 * n_cols, 3.8 * n_bands),
                                 squeeze=False)

        for yi, year_lbl in enumerate(year_labels):
            b_labels, b_data = year_data[year_lbl]

            for bi_global, global_lbl in enumerate(band_labels_global):
                ax_before = axes[bi_global, 2 * yi]
                ax_after  = axes[bi_global, 2 * yi + 1]

                # Find matching local band
                if global_lbl in b_labels:
                    bi_local = b_labels.index(global_lbl)
                    bd = b_data[bi_local]
                else:
                    bd = None

                if bd is None:
                    for ax in [ax_before, ax_after]:
                        ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                                transform=ax.transAxes, fontsize=11, color='gray')
                        ax.set_xlim(0, MAX_REFL)
                        ax.set_ylim(0, MAX_REFL)
                        ax.set_aspect('equal')
                else:
                    plot_panel(ax_before, bd['s2'], bd['raw'], cmap)
                    plot_panel(ax_after,  bd['s2'], bd['harm'], cmap)

                # --- Axis labels ---
                # Left-most column: band name as y-label
                if yi == 0:
                    ax_before.set_ylabel(f"{global_lbl}\nPlanet refl.",
                                         fontsize=10)
                # Bottom row: x-label
                if bi_global == n_bands - 1:
                    ax_before.set_xlabel("S2 refl.", fontsize=9)
                    ax_after.set_xlabel("S2 refl.", fontsize=9)

                # Top row: column titles
                if bi_global == 0:
                    ax_before.set_title(f"{year_lbl}\nBefore", fontsize=11,
                                        fontweight='bold')
                    ax_after.set_title(f"{year_lbl}\nAfter", fontsize=11,
                                       fontweight='bold')

        fig.suptitle(f"{sta_name} — Planet vs Sentinel-2 reflectance",
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout(h_pad=1.0, w_pad=0.6)

        out_path = OUT_DIR / f"scatter_harmonization_{sta_name.replace('-','').lower()}.png"
        fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        sz = out_path.stat().st_size / 1024
        print(f"  ✅ {out_path.name}  ({sz:.0f} KB)")

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
