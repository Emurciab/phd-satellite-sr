#!/usr/bin/env python3
"""
Option A: Per-tile NORMALIZED scatter plot.

Before pooling pixels across tiles, each tile's Planet and S2 values
are mean-centred:  x_norm = x - mean(x),  y_norm = y - mean(y).
This removes inter-tile atmospheric / calibration offsets and reveals
the true spatial correlation that exists within every tile.

Output: results/model_evaluation/visualizations/scatter_normalized_{sta}.png
"""

import os, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom
from scipy.stats import pearsonr

# ── Paths ──────────────────────────────────────────────────────────
PROJECT = Path("/media/eamurciab/TOSHIBA_EXT1/PhD/SR")
CACHE   = PROJECT / "cache"
OUT_DIR = PROJECT / "results" / "model_evaluation" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Histogram matching ────────────────────────────────────────────
def histogram_match_band(source, reference):
    try:
        from skimage.exposure import match_histograms as _hm
        return _hm(source, reference).astype(source.dtype)
    except ImportError:
        pass
    sf = source.flatten(); rf = reference.flatten()
    sv = sf[sf > 0]; rv = rf[rf > 0]
    if len(sv) == 0 or len(rv) == 0:
        return source
    svals, sinv = np.unique(sv, return_inverse=True)
    rvals = np.unique(rv)
    scdf = np.cumsum(np.bincount(sinv)).astype(np.float64); scdf /= scdf[-1]
    rcdf = np.cumsum(np.bincount(np.searchsorted(rvals, rv))).astype(np.float64); rcdf /= rcdf[-1]
    iv = np.interp(scdf, rcdf, rvals)
    m = sf.copy(); vm = sf > 0
    m[vm] = iv[np.clip(np.searchsorted(svals, sf[vm]), 0, len(iv)-1)]
    return m.reshape(source.shape)

# ── Band config ───────────────────────────────────────────────────
BAND_LABELS_5 = ["Blue", "Green", "Red", "Red Edge", "NIR"]
BAND_MAP_5 = [(0,"10m",0),(1,"10m",1),(2,"10m",2),(3,"20m",0),(4,"20m",1)]
BAND_LABELS_4 = ["Blue", "Green", "Red", "NIR"]
BAND_MAP_4 = [(0,"10m",0),(1,"10m",1),(2,"10m",2),(3,"20m",0)]

STA_YEARS = {
    "STA-1E": [("2022","sta1e_2022"),("2023–2024","sta1e_grid_parallel")],
    "STA-1W": [("2023–2024","sta1w_grid_parallel")],
    "STA-2":  [("2020–2024","sta2_grid_parallel")],
    "STA-34": [("2023","sta34_2023"),("2023–2024","sta34_grid_parallel")],
    "STA-56": [("2023","sta56_2023"),("2024","sta56_2024")],
}

TILES_PER_YEAR  = 50
PIXELS_PER_TILE = 500
MAX_REFL        = 5000

random.seed(42); np.random.seed(42)

def upsample(lr, shape):
    return zoom(lr, (shape[0]/lr.shape[0], shape[1]/lr.shape[1]), order=0)


# ── Data collection (keeps per-tile arrays separate) ──────────────
def collect_per_tile(cache_name):
    """Return (band_labels, per_band_list_of_tile_dicts)."""
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

    # per_band[bi] = list of dicts {'raw':..., 'harm':..., 's2':...}  per tile
    per_band = [[] for _ in range(n_bands)]

    for fname in sampled:
        try:
            d = np.load(raw_dir / fname, allow_pickle=True)
            hr_raw = d['hr_array']; lr_10m = d['lr_10m']; lr_20m = d['lr_20m']
            hh, hw = hr_raw.shape[:2]
        except Exception:
            continue

        # ── Quality gate (mirror pipeline: valid_fraction >= 0.90) ──
        hr_valid = np.sum((hr_raw > 0) & ~np.isnan(hr_raw)) / hr_raw.size
        lr_valid = np.sum((lr_10m > 0) & ~np.isnan(lr_10m)) / lr_10m.size
        if hr_valid < 0.90 or lr_valid < 0.90:
            continue

        num_harm = min(hr_raw.shape[2], lr_10m.shape[2])
        hr_harm = hr_raw.copy()
        for bi in range(num_harm):
            hr_harm[:,:,bi] = histogram_match_band(hr_raw[:,:,bi], lr_10m[:,:,bi])

        for band_idx, (hr_bi, lr_src, lr_bi) in enumerate(band_map):
            if lr_src == "10m":
                lr_band = lr_10m[:,:,lr_bi]
            else:
                if lr_20m.ndim == 2: lr_band = lr_20m
                elif lr_bi >= lr_20m.shape[2]: continue
                else: lr_band = lr_20m[:,:,lr_bi]

            s2_up  = upsample(lr_band, (hh, hw))
            p_raw  = hr_raw[:,:,hr_bi]
            p_harm = hr_harm[:,:,hr_bi] if hr_bi < num_harm else hr_raw[:,:,hr_bi]

            valid = (p_raw > 0) & (s2_up > 0) & (p_raw < MAX_REFL*1.5) & (s2_up < MAX_REFL*1.5)
            vi = np.argwhere(valid)
            if len(vi) < 30:
                continue
            ch = vi[np.random.choice(len(vi), min(PIXELS_PER_TILE, len(vi)), replace=False)]
            r, c = ch[:,0], ch[:,1]
            per_band[band_idx].append({
                'raw':  p_raw[r, c],
                'harm': p_harm[r, c],
                's2':   s2_up[r, c],
            })

    return band_labels, per_band


# ── Panel plotter for normalized data ─────────────────────────────
def plot_normalized_panel(ax, tile_list, key, cmap):
    """
    For each tile, subtract tile mean from both Planet(key) and S2,
    then pool and plot density scatter.
    """
    all_x, all_y = [], []
    for td in tile_list:
        s2 = td['s2'].astype(np.float64)
        pl = td[key].astype(np.float64)
        s2_n = s2 - s2.mean()
        pl_n = pl - pl.mean()
        all_x.append(s2_n)
        all_y.append(pl_n)

    if not all_x:
        ax.text(0.5, 0.5, "No data", ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')
        return

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)

    if len(x) > 200_000:
        idx = np.random.choice(len(x), 200_000, replace=False)
        x, y = x[idx], y[idx]

    # Symmetric limits
    absmax = max(np.abs(x).max(), np.abs(y).max()) * 1.05
    absmax = min(absmax, 2000)

    ax.hexbin(x, y, gridsize=100, cmap=cmap, mincnt=1,
              linewidths=0.15, norm=mcolors.LogNorm())
    ax.plot([-absmax, absmax], [-absmax, absmax], 'w--', lw=1.0, alpha=0.8)
    ax.axhline(0, color='w', lw=0.3, alpha=0.4)
    ax.axvline(0, color='w', lw=0.3, alpha=0.4)

    r, _ = pearsonr(x, y)
    rmse = np.sqrt(np.mean((y - x)**2))
    ax.text(0.05, 0.95, f"R²={r**2:.3f}\nRMSE={rmse:.0f}",
            transform=ax.transAxes, fontsize=8, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5))
    ax.text(0.05, 0.05, f"n={len(x):,}", transform=ax.transAxes,
            fontsize=7, va='bottom', ha='left', color='white')

    ax.set_xlim(-absmax, absmax)
    ax.set_ylim(-absmax, absmax)
    ax.set_aspect('equal')


# ── Main ──────────────────────────────────────────────────────────
def main():
    cmap = plt.cm.inferno.copy()

    for sta_name, year_caches in STA_YEARS.items():
        print(f"\n{'='*60}")
        print(f"  {sta_name}")
        print(f"{'='*60}")

        year_data = {}
        for year_label, cache_name in year_caches:
            print(f"  Collecting {year_label} ({cache_name})...")
            out = collect_per_tile(cache_name)
            if out is not None:
                year_data[year_label] = out
                n_tiles = max(len(b) for b in out[1])
                print(f"    ✓ {out[0]}  ({n_tiles} tiles)")

        if not year_data:
            print(f"  ⚠️  No data"); continue

        n_years = len(year_data)
        year_labels = list(year_data.keys())
        max_bands = max(len(v[0]) for v in year_data.values())
        bl_global = BAND_LABELS_5 if max_bands == 5 else BAND_LABELS_4
        n_bands = len(bl_global)

        n_cols = 2 * n_years
        fig, axes = plt.subplots(n_bands, n_cols,
                                 figsize=(4.2 * n_cols, 3.8 * n_bands),
                                 squeeze=False)

        for yi, ylbl in enumerate(year_labels):
            b_labels, b_tiles = year_data[ylbl]

            for bi_g, glbl in enumerate(bl_global):
                ax_bef = axes[bi_g, 2*yi]
                ax_aft = axes[bi_g, 2*yi + 1]

                if glbl in b_labels:
                    bi_l = b_labels.index(glbl)
                    tiles = b_tiles[bi_l]
                else:
                    tiles = []

                if not tiles:
                    for ax in [ax_bef, ax_aft]:
                        ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                                transform=ax.transAxes, fontsize=11, color='gray')
                        ax.set_xlim(-1000, 1000); ax.set_ylim(-1000, 1000)
                        ax.set_aspect('equal')
                else:
                    plot_normalized_panel(ax_bef, tiles, 'raw',  cmap)
                    plot_normalized_panel(ax_aft, tiles, 'harm', cmap)

                if yi == 0:
                    ax_bef.set_ylabel(f"{glbl}\nΔ Planet refl.", fontsize=10)
                if bi_g == n_bands - 1:
                    ax_bef.set_xlabel("Δ S2 refl.", fontsize=9)
                    ax_aft.set_xlabel("Δ S2 refl.", fontsize=9)
                if bi_g == 0:
                    ax_bef.set_title(f"{ylbl}\nBefore", fontsize=11, fontweight='bold')
                    ax_aft.set_title(f"{ylbl}\nAfter",  fontsize=11, fontweight='bold')

        fig.suptitle(f"{sta_name} — Normalized (per-tile mean-centred)",
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout(h_pad=1.0, w_pad=0.6)

        out_path = OUT_DIR / f"scatter_normalized_{sta_name.replace('-','').lower()}.png"
        fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✅ {out_path.name}  ({out_path.stat().st_size/1024:.0f} KB)")

    print(f"\nAll figures saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
