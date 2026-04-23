#!/usr/bin/env python3
"""
Before / After Harmonization Scatter Plots  —  Curated Training Data Only
=========================================================================

For every curated training sample (the same ones used for model training):
  • BEFORE  → load the raw_downloads NPZ  (pre-harmonisation HR)
  • AFTER   → use the data stored in the pkl dataset  (post-harmonisation HR)
  • LR      → same in both (Sentinel-2 is never modified)

The script reconstructs the cache hash for each frame:
    hash = md5(planet_id|s2_id|tile_bounds|planet_bands|sentinel_bands)[:16]

Outputs
-------
figures/scatter_before_after.png   –  2×5 grid  (before / after  ×  5 bands)
"""

import sys, os, types, hashlib, pickle, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from scipy import stats as sp_stats

# ──────────────────────────────────────────────────────────────────────
# 1.  Unpickling helpers
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TrainingSample5Band:
    lr_sequence:       np.ndarray           = None
    hr_sequence:       np.ndarray           = None
    lr_20m_sequence:   Optional[np.ndarray] = None
    hr_10m_sequence:   Optional[np.ndarray] = None
    hr_20m_sequence:   Optional[np.ndarray] = None
    temporal_dates:    List[str]            = field(default_factory=list)
    planet_dates:      List[str]            = field(default_factory=list)
    region_name:       str                  = ''
    sequence_id:       int                  = 0
    tile_id:           int                  = 0
    tile_bounds:       Dict                 = field(default_factory=dict)
    sensor_type:       str                  = ''
    planet_bands:      List[str]            = field(default_factory=list)
    sentinel_bands:    List[str]            = field(default_factory=list)
    s2_solar_zenith:   Optional[List[float]]= None
    planet_solar_zenith: Optional[List[float]] = None
    frame_quality:     Optional[List[Dict]] = None


def _register_unpickle():
    for mod_name in ('convert_cache_to_5band',
                     'scripts.preprocessing.convert_cache_to_5band',
                     '__main__'):
        fake = types.ModuleType(mod_name)
        fake.TrainingSample5Band = TrainingSample5Band
        sys.modules[mod_name] = fake

_register_unpickle()


# ──────────────────────────────────────────────────────────────────────
# 2.  Load matched-pair look-ups  (planet_date → IDs)
# ──────────────────────────────────────────────────────────────────────
BASE = Path('/media/eamurciab/TOSHIBA_EXT1/PhD/SR')

# Map STA names (as they appear in pkl region_name) to their cache dirs
# and matched-pair pkl files
STA_CONFIG = {
    'STA-1E': {
        'cache': BASE / 'cache/sta1e_grid_parallel',
        'mp_files': [
            BASE / 'cache/sta1e_grid_parallel/matched_pairs_266172a23acc1618.pkl',  # 2023
            BASE / 'cache/sta1e_grid_parallel/matched_pairs_07119e0d32569f66.pkl',  # 2024
        ],
    },
    'STA-1W': {
        'cache': BASE / 'cache/sta1w_grid_parallel',
        'mp_files': [
            BASE / 'cache/sta1w_grid_parallel/matched_pairs_3160b6bf88dbd992.pkl',  # 2022
            BASE / 'cache/sta1w_grid_parallel/matched_pairs_30c02db1138abad0.pkl',  # 2024
        ],
    },
    'STA-2': {
        'cache': BASE / 'cache/sta2_grid_parallel',
        'mp_files': [
            BASE / 'cache/sta2_grid_parallel/matched_pairs_7fdc2b96bfa0f186.pkl',  # 2020
            BASE / 'cache/sta2_grid_parallel/matched_pairs_f4fabbbedca4694f.pkl',  # 2021
            BASE / 'cache/sta2_grid_parallel/matched_pairs_4dd28cfca10df7e5.pkl',  # 2022
            BASE / 'cache/sta2_grid_parallel/matched_pairs_1cea6240859d119a.pkl',  # 2023
            BASE / 'cache/sta2_grid_parallel/matched_pairs_7d197fffbfaf49c0.pkl',  # 2024
        ],
    },
    'STA-3/4': {
        'cache': BASE / 'cache/sta34_grid_parallel',
        'mp_files': [
            BASE / 'cache/sta34_2023/matched_pairs_5c56b1d187c9585d.pkl',  # 2023
        ],
        # sta34_grid_parallel raw_downloads uses same hashes as sta34_2023 run
        'alt_cache': BASE / 'cache/sta34_2023',
    },
    'STA-5/6': {
        'cache': BASE / 'cache/sta56_grid_parallel',
        'mp_files': [
            BASE / 'cache/sta56_grid_parallel/matched_pairs_e6b9f667ac01b2f9.pkl',  # 2023
            BASE / 'cache/sta56_2024/matched_pairs_5651db995e8be8a5.pkl',            # 2024
        ],
    },
}


def load_all_matched_pairs():
    """Build per-STA lookup:  (planet_date, s2_date) → {planet_id, s2_id}."""
    lookup = {}   # sta_name → { (planet_date, s2_date): {planet_id, s2_id} }

    for sta_name, cfg in STA_CONFIG.items():
        sta_lookup = {}
        for mp_path in cfg['mp_files']:
            if not mp_path.exists():
                print(f"  ⚠ Missing: {mp_path}")
                continue
            with open(mp_path, 'rb') as f:
                data = pickle.load(f)
            pairs = data if isinstance(data, list) else data.get('matched_pairs', [])
            for p in pairs:
                key = (p['planet_date'], p['s2_date'])
                sta_lookup[key] = {
                    'planet_id': p['planet_id'],
                    's2_id': p['s2_id'],
                }
        lookup[sta_name] = sta_lookup
        print(f"  {sta_name}: {len(sta_lookup)} date→ID mappings")

    return lookup


# ──────────────────────────────────────────────────────────────────────
# 3.  Reconstruct cache hash and load raw tile
# ──────────────────────────────────────────────────────────────────────
def make_cache_hash(planet_id, s2_id, tile_bounds, planet_bands, sentinel_bands):
    """Reproduce the hash used by generate_*_grid_parallel.py."""
    key = f"{planet_id}|{s2_id}|{tile_bounds}|{planet_bands}|{sentinel_bands}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def load_raw_tile(cache_dir, cache_hash, alt_cache=None):
    """Load pre-harmonisation NPZ from raw_downloads."""
    path = cache_dir / 'raw_downloads' / f'tile_{cache_hash}.npz'
    if path.exists():
        return np.load(path, allow_pickle=True)
    # Try alternative cache
    if alt_cache is not None:
        path2 = alt_cache / 'raw_downloads' / f'tile_{cache_hash}.npz'
        if path2.exists():
            return np.load(path2, allow_pickle=True)
    return None


# ──────────────────────────────────────────────────────────────────────
# 4.  Dataset config (same as prepare_all_stas_dataset.py)
# ──────────────────────────────────────────────────────────────────────
DATASET_PATHS = [
    ('STA-1E', '2023', BASE / 'data/datasets/sta1e_2023_5band'),
    ('STA-1E', '2024', BASE / 'data/datasets/sta1e_2024_5band'),
    ('STA-1W', '2023', BASE / 'data/datasets/sta1w_2023_5band'),
    ('STA-1W', '2024', BASE / 'data/datasets/sta1w_2024_5band'),
    ('STA-2',  '2022', BASE / 'data/sr_training/5band_v3/STA2_2022/superdove_5band'),
    ('STA-2',  '2023', BASE / 'data/sr_training/5band_v3/STA2_2023/superdove_5band'),
    ('STA-2',  '2024', BASE / 'data/sr_training/5band_v3/STA2_2024/superdove_5band'),
    ('STA-3/4','2023', BASE / 'data/datasets/sta34_2023_5band'),
    ('STA-3/4','2024', BASE / 'data/datasets/sta34_2024_5band'),
    ('STA-5/6','2023', BASE / 'data/datasets/sta56_2023_5band'),
    ('STA-5/6','2024', BASE / 'data/datasets/sta56_2024_5band'),
]


def load_pkl_samples(path: Path):
    """Load TrainingSample5Band objects from pkl dataset."""
    samples = []
    # Chunked format
    if (path / 'metadata.pkl').exists():
        for chunk in sorted(path.glob('chunk_*.pkl')):
            with open(chunk, 'rb') as f:
                samples.extend(pickle.load(f))
    else:
        for pkl in sorted(path.glob('*.pkl')):
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                samples.extend(data.get('samples', data.get('data', [])))
            elif isinstance(data, list):
                samples.extend(data)
    return samples


# ──────────────────────────────────────────────────────────────────────
# 5.  Collect pixel pairs  (before & after)
# ──────────────────────────────────────────────────────────────────────
BAND_NAMES = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR']

# Bands: HR has 5 channels [Blue, Green, Red, RedEdge, NIR]
# LR 10m has 3 channels [B2, B3, B4]  →  correspond to HR ch 0,1,2
# LR 20m has 2 channels [B5, B8A]     →  correspond to HR ch 3,4


def collect_pixels(max_samples_per_sta=200, subsample=2000):
    """
    For each curated training sample:
      – load post-harmonisation HR from pkl  (normalised [0,1] → ×10000)
      – reconstruct cache hash → load raw HR from NPZ
      – pair with LR from pkl (same in both cases — S2 is unchanged)

    Returns:
        before_10m, after_10m : dict of band_idx → (lr_pixels, hr_pixels)
        before_20m, after_20m : dict of band_idx → (lr_pixels, hr_pixels)
    """
    id_lookup = load_all_matched_pairs()

    # Accumulators: band_idx → list of (lr, hr) pixels
    before_10m = {i: ([], []) for i in range(3)}
    after_10m  = {i: ([], []) for i in range(3)}
    before_20m = {i: ([], []) for i in range(2)}
    after_20m  = {i: ([], []) for i in range(2)}

    total_matched = 0
    total_missed  = 0

    for sta_name, year, dpath in DATASET_PATHS:
        if not dpath.exists():
            print(f"  ⚠ {sta_name} {year}: not found at {dpath}")
            continue

        samples = load_pkl_samples(dpath)
        if not samples:
            print(f"  ⚠ {sta_name} {year}: 0 samples")
            continue

        # Limit per STA-year to keep balanced
        if len(samples) > max_samples_per_sta:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(samples), max_samples_per_sta, replace=False)
            samples = [samples[i] for i in idx]

        sta_cfg = STA_CONFIG.get(sta_name, {})
        cache_dir = sta_cfg.get('cache')
        alt_cache = sta_cfg.get('alt_cache')
        sta_id_map = id_lookup.get(sta_name, {})

        matched_this = 0
        missed_this  = 0

        for s in samples:
            # Extract metadata
            tb = s.tile_bounds
            if isinstance(tb, dict):
                bounds = tb.get('bounds', [])
                p_bands = tb.get('planet_bands', s.planet_bands)
                s_bands = tb.get('sentinel_bands', s.sentinel_bands)
            else:
                continue

            if not bounds:
                continue

            # Each sample has T=3 temporal frames
            T = 3
            if s.hr_sequence is None or s.lr_sequence is None:
                continue
            if s.hr_sequence.shape[0] < T or s.lr_sequence.shape[0] < T:
                continue

            for t in range(T):
                planet_date = s.planet_dates[t] if t < len(s.planet_dates) else None
                s2_date = s.temporal_dates[t] if t < len(s.temporal_dates) else None
                if planet_date is None or s2_date is None:
                    continue

                # Look up planet_id, s2_id
                key = (planet_date, s2_date)
                ids = sta_id_map.get(key)
                if ids is None:
                    # Sometimes s2_date == planet_date, try both orderings
                    ids = sta_id_map.get((planet_date, planet_date))
                if ids is None:
                    missed_this += 1
                    continue

                # Reconstruct cache hash
                ch = make_cache_hash(
                    ids['planet_id'], ids['s2_id'],
                    bounds, list(p_bands), list(s_bands)
                )
                raw_data = load_raw_tile(cache_dir, ch, alt_cache)
                if raw_data is None:
                    missed_this += 1
                    continue

                matched_this += 1

                # ------- POST-HARMONISATION (from pkl, normalised [0,1]) -------
                hr_after  = s.hr_sequence[t]       # (256,256,5) in [0,1]
                lr_10m    = s.lr_sequence[t]        # (77,77,3)   in [0,1]
                lr_20m_t  = s.lr_20m_sequence[t] if s.lr_20m_sequence is not None else None

                # Convert back to reflectance scale
                hr_after_ref = hr_after * 10000.0
                lr_10m_ref   = lr_10m   * 10000.0
                lr_20m_ref   = lr_20m_t * 10000.0 if lr_20m_t is not None else None

                # ------- PRE-HARMONISATION (from raw NPZ) -------
                hr_before_ref = raw_data['hr_array']  # (256,256,5) raw reflectance
                # LR is identical (S2 not modified by harmonisation)

                # ---------- Subsample pixels ----------
                # 10m bands: spatially average HR to LR resolution
                # HR is 256×256 and LR is 77×77, ratio ≈ 3.32
                # For a fair scatter we compare at LR resolution (77×77)
                # Downsample HR by block averaging 3×3 → ~85×85 then crop to 77×77
                # Actually, simpler: just flatten and subsample
                h, w = lr_10m_ref.shape[:2]
                n_pix = h * w

                for b in range(3):
                    lr_flat = lr_10m_ref[:, :, b].ravel()
                    # For HR, downsample to LR grid (average 256→77)
                    # Use scipy.ndimage or simple block average
                    hr_a_full = hr_after_ref[:, :, b]
                    hr_b_full = hr_before_ref[:, :, b]
                    # Resize HR to LR size via area averaging
                    from skimage.transform import resize
                    hr_a_lr = resize(hr_a_full, (h, w), order=1, anti_aliasing=True,
                                     preserve_range=True).ravel()
                    hr_b_lr = resize(hr_b_full, (h, w), order=1, anti_aliasing=True,
                                     preserve_range=True).ravel()

                    # Mask out bad pixels (near-zero or very high)
                    mask = ((lr_flat > 100) & (lr_flat < 8000) &
                            (hr_a_lr > 100) & (hr_a_lr < 8000) &
                            (hr_b_lr > 100) & (hr_b_lr < 8000))
                    lr_g = lr_flat[mask]
                    hr_ag = hr_a_lr[mask]
                    hr_bg = hr_b_lr[mask]

                    if len(lr_g) > subsample:
                        idx = np.random.choice(len(lr_g), subsample, replace=False)
                        lr_g, hr_ag, hr_bg = lr_g[idx], hr_ag[idx], hr_bg[idx]

                    before_10m[b][0].append(lr_g)
                    before_10m[b][1].append(hr_bg)
                    after_10m[b][0].append(lr_g)
                    after_10m[b][1].append(hr_ag)

                # 20m bands: HR ch 3,4 → LR 20m ch 0,1
                if lr_20m_ref is not None and lr_20m_ref.shape[-1] >= 2:
                    h20, w20 = lr_20m_ref.shape[:2]
                    # HR 20m is at (128,128,2) in the pkl; raw HR is (256,256,5) ch 3,4
                    hr_20m_after = s.hr_20m_sequence[t] * 10000.0 if s.hr_20m_sequence is not None else None
                    hr_20m_before_full = hr_before_ref[:, :, 3:5]  # (256,256,2) raw
                    # Downsample before to 128×128 to match after
                    hr_20m_before = resize(hr_20m_before_full, (h20, w20, 2),
                                           order=1, anti_aliasing=True,
                                           preserve_range=True)

                    for b in range(2):
                        lr_flat = lr_20m_ref[:, :, b].ravel()
                        if hr_20m_after is not None:
                            hr_a = resize(hr_20m_after[:, :, b], (h20, w20),
                                          order=1, anti_aliasing=True,
                                          preserve_range=True).ravel()
                        else:
                            hr_a = resize(hr_20m_before[:, :, b], (h20, w20),
                                          order=1, anti_aliasing=True,
                                          preserve_range=True).ravel()
                        hr_b = hr_20m_before[:, :, b].ravel()

                        mask = ((lr_flat > 50) & (lr_flat < 8000) &
                                (hr_a > 50) & (hr_a < 8000) &
                                (hr_b > 50) & (hr_b < 8000))
                        lg = lr_flat[mask]
                        hag = hr_a[mask]
                        hbg = hr_b[mask]

                        if len(lg) > subsample:
                            idx = np.random.choice(len(lg), subsample, replace=False)
                            lg, hag, hbg = lg[idx], hag[idx], hbg[idx]

                        before_20m[b][0].append(lg)
                        before_20m[b][1].append(hbg)
                        after_20m[b][0].append(lg)
                        after_20m[b][1].append(hag)

        total_matched += matched_this
        total_missed  += missed_this
        print(f"  {sta_name} {year}: {len(samples)} samples, "
              f"{matched_this} frames matched, {missed_this} missed")

    print(f"\n  TOTAL: {total_matched} frames matched, {total_missed} missed")

    # Concatenate
    def cat(d):
        return {k: (np.concatenate(v[0]) if v[0] else np.array([]),
                     np.concatenate(v[1]) if v[1] else np.array([]))
                for k, v in d.items()}

    return cat(before_10m), cat(after_10m), cat(before_20m), cat(after_20m)


# ──────────────────────────────────────────────────────────────────────
# 6.  Plotting
# ──────────────────────────────────────────────────────────────────────
def plot_before_after(before_10m, after_10m, before_20m, after_20m, out_path):
    """2 rows × 5 columns: [Before | After] × [Blue, Green, Red, RedEdge, NIR]."""
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    band_labels = ['Blue\n(B2↔B2)', 'Green\n(B4↔B3)', 'Red\n(B6↔B4)',
                   'RedEdge\n(B7↔B5)', 'NIR\n(B8↔B8A)']
    row_labels = ['Before\nHarmonization', 'After\nHarmonization']

    all_data = [
        # row 0 — before
        [before_10m[0], before_10m[1], before_10m[2], before_20m[0], before_20m[1]],
        # row 1 — after
        [after_10m[0],  after_10m[1],  after_10m[2],  after_20m[0],  after_20m[1]],
    ]

    for row in range(2):
        for col in range(5):
            ax = axes[row, col]
            lr, hr = all_data[row][col]
            if len(lr) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_xlim(0, 5000); ax.set_ylim(0, 5000)
            else:
                # Density scatter
                ax.hist2d(lr, hr, bins=150, cmap='magma', norm=LogNorm(),
                          range=[[0, 5000], [0, 5000]])

                # 1:1 line
                ax.plot([0, 5000], [0, 5000], 'w--', lw=1.5, alpha=0.8)

                # Stats
                slope, intercept, r, p, se = sp_stats.linregress(lr, hr)
                rmse = np.sqrt(np.mean((hr - lr) ** 2))
                bias = np.mean(hr - lr)
                ax.text(0.05, 0.95,
                        f'R² = {r**2:.3f}\nRMSE = {rmse:.0f}\nBias = {bias:+.0f}\n'
                        f'n = {len(lr):,}',
                        transform=ax.transAxes, fontsize=11,
                        va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

            if row == 0:
                ax.set_title(band_labels[col], fontsize=14, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{row_labels[row]}\n\nPlanet HR (reflectance)',
                              fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel('')
            if row == 1:
                ax.set_xlabel('Sentinel-2 LR (reflectance)', fontsize=12)
            else:
                ax.set_xlabel('')
            ax.set_aspect('equal')
            ax.tick_params(labelsize=10)

    fig.suptitle('Spectral Consistency: Planet vs Sentinel-2  —  Curated Training Data\n'
                 '(Before / After CDF Histogram-Matching Harmonization)',
                 fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────
# 7.  Main
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 70)
    print("BEFORE / AFTER HARMONIZATION SCATTER  —  CURATED DATA ONLY")
    print("=" * 70)

    print("\n📂 Loading matched-pair ID lookups …")
    before_10m, after_10m, before_20m, after_20m = collect_pixels(
        max_samples_per_sta=200, subsample=2000
    )

    out_dir = BASE / 'figures'
    out_dir.mkdir(exist_ok=True)

    plot_before_after(before_10m, after_10m, before_20m, after_20m,
                      out_dir / 'scatter_before_after.png')

    # Print summary stats
    print("\n📊 Summary statistics:")
    for label, d10, d20 in [('Before', before_10m, before_20m),
                             ('After',  after_10m,  after_20m)]:
        print(f"\n  {label} harmonization:")
        for i, name in enumerate(BAND_NAMES[:3]):
            lr, hr = d10[i]
            if len(lr):
                r2 = sp_stats.linregress(lr, hr).rvalue ** 2
                rmse = np.sqrt(np.mean((hr - lr) ** 2))
                bias = np.mean(hr - lr)
                print(f"    {name:>8s}: R²={r2:.4f}  RMSE={rmse:.0f}  Bias={bias:+.0f}  n={len(lr):,}")
        for i, name in enumerate(BAND_NAMES[3:]):
            lr, hr = d20[i]
            if len(lr):
                r2 = sp_stats.linregress(lr, hr).rvalue ** 2
                rmse = np.sqrt(np.mean((hr - lr) ** 2))
                bias = np.mean(hr - lr)
                print(f"    {name:>8s}: R²={r2:.4f}  RMSE={rmse:.0f}  Bias={bias:+.0f}  n={len(lr):,}")
