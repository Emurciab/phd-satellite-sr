#!/usr/bin/env python3
"""
Option B: Per-tile R² distribution (box + strip plot).

For each tile, compute R² between Planet and S2 per band.
Show the distribution of within-tile R² as box plots,
before vs after harmonization, per band, per STA, per year.

Output: results/model_evaluation/visualizations/r2_boxplot_{sta}.png
"""

import os, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

TILES_PER_YEAR = 80
MAX_REFL       = 5000

random.seed(42); np.random.seed(42)

def upsample(lr, shape):
    return zoom(lr, (shape[0]/lr.shape[0], shape[1]/lr.shape[1]), order=0)


# ── Collect per-tile R² ───────────────────────────────────────────
def collect_r2_per_tile(cache_name):
    """
    Return (band_labels, r2_before[band][tile], r2_after[band][tile]).
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

    r2_before = [[] for _ in range(n_bands)]
    r2_after  = [[] for _ in range(n_bands)]

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
            n_valid = valid.sum()
            if n_valid < 100:
                continue

            s2_v  = s2_up[valid]
            raw_v = p_raw[valid]
            harm_v = p_harm[valid]

            # Check for constant arrays (pearsonr fails)
            if raw_v.std() < 1 or s2_v.std() < 1:
                continue

            try:
                r_bef, _ = pearsonr(s2_v, raw_v)
                r2_before[band_idx].append(r_bef**2)
            except:
                pass

            if harm_v.std() < 1:
                continue
            try:
                r_aft, _ = pearsonr(s2_v, harm_v)
                r2_after[band_idx].append(r_aft**2)
            except:
                pass

    return band_labels, r2_before, r2_after


# ── Plotting ──────────────────────────────────────────────────────
COL_BEFORE = '#2196F3'   # blue
COL_AFTER  = '#FF9800'   # orange

def main():
    for sta_name, year_caches in STA_YEARS.items():
        print(f"\n{'='*60}")
        print(f"  {sta_name}")
        print(f"{'='*60}")

        year_data = {}
        for year_label, cache_name in year_caches:
            print(f"  Collecting {year_label} ({cache_name})...")
            out = collect_r2_per_tile(cache_name)
            if out is not None:
                year_data[year_label] = out
                n_tiles = max((len(b) for b in out[1]), default=0)
                print(f"    ✓ {out[0]}  ({n_tiles} tiles)")

        if not year_data:
            print(f"  ⚠️  No data"); continue

        n_years = len(year_data)
        year_labels = list(year_data.keys())
        max_bands = max(len(v[0]) for v in year_data.values())
        bl_global = BAND_LABELS_5 if max_bands == 5 else BAND_LABELS_4
        n_bands = len(bl_global)

        # One subplot per year
        fig, axes = plt.subplots(1, n_years, figsize=(7 * n_years, 5),
                                 squeeze=False)

        for yi, ylbl in enumerate(year_labels):
            ax = axes[0, yi]
            b_labels, r2_bef, r2_aft = year_data[ylbl]

            positions_bef = []
            positions_aft = []
            data_bef = []
            data_aft = []
            tick_positions = []
            tick_labels_use = []

            for bi_g, glbl in enumerate(bl_global):
                x_center = bi_g * 3  # spacing between band groups
                tick_positions.append(x_center)
                tick_labels_use.append(glbl)

                if glbl in b_labels:
                    bi_l = b_labels.index(glbl)
                    db = np.array(r2_bef[bi_l]) if r2_bef[bi_l] else np.array([])
                    da = np.array(r2_aft[bi_l]) if r2_aft[bi_l] else np.array([])
                else:
                    db = np.array([]); da = np.array([])

                data_bef.append(db)
                data_aft.append(da)
                positions_bef.append(x_center - 0.45)
                positions_aft.append(x_center + 0.45)

            # Box plots
            bp_kw = dict(widths=0.7, patch_artist=True, showfliers=False,
                         medianprops=dict(color='black', lw=1.5),
                         whiskerprops=dict(lw=1), capprops=dict(lw=1))

            # Before
            bp1 = ax.boxplot([d if len(d) > 0 else [np.nan] for d in data_bef],
                             positions=positions_bef, **bp_kw)
            for patch in bp1['boxes']:
                patch.set_facecolor(COL_BEFORE); patch.set_alpha(0.7)

            # After
            bp2 = ax.boxplot([d if len(d) > 0 else [np.nan] for d in data_aft],
                             positions=positions_aft, **bp_kw)
            for patch in bp2['boxes']:
                patch.set_facecolor(COL_AFTER); patch.set_alpha(0.7)

            # Jittered strip dots
            for i in range(n_bands):
                if len(data_bef[i]) > 0:
                    jitter = np.random.uniform(-0.2, 0.2, len(data_bef[i]))
                    ax.scatter(positions_bef[i] + jitter, data_bef[i],
                               c=COL_BEFORE, s=10, alpha=0.35, edgecolors='none', zorder=3)
                if len(data_aft[i]) > 0:
                    jitter = np.random.uniform(-0.2, 0.2, len(data_aft[i]))
                    ax.scatter(positions_aft[i] + jitter, data_aft[i],
                               c=COL_AFTER, s=10, alpha=0.35, edgecolors='none', zorder=3)

                # Annotate median
                for pos, d, col in [(positions_bef[i], data_bef[i], COL_BEFORE),
                                     (positions_aft[i], data_aft[i], COL_AFTER)]:
                    if len(d) > 0:
                        med = np.median(d)
                        ax.text(pos, med + 0.03, f"{med:.2f}",
                                ha='center', va='bottom', fontsize=7,
                                fontweight='bold', color=col)

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels_use, fontsize=10)
            ax.set_ylabel("Within-tile R²", fontsize=11)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{ylbl}", fontsize=12, fontweight='bold')
            ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
            ax.grid(axis='y', alpha=0.3)

            # Legend
            if yi == 0:
                ax.legend(handles=[
                    mpatches.Patch(facecolor=COL_BEFORE, alpha=0.7, label='Before harmonization'),
                    mpatches.Patch(facecolor=COL_AFTER,  alpha=0.7, label='After harmonization'),
                ], loc='lower right', fontsize=9)

        fig.suptitle(f"{sta_name} — Within-tile R² (Planet vs S2)",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        out_path = OUT_DIR / f"r2_boxplot_{sta_name.replace('-','').lower()}.png"
        fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✅ {out_path.name}  ({out_path.stat().st_size/1024:.0f} KB)")

    print(f"\nAll figures saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
