#!/usr/bin/env python3
"""
Comprehensive visualisation suite for the Heteroscedastic Deep Ensemble results.

Produces publication-quality figures from:
  • ensemble_eval_*.csv   (per-sample metrics + uncertainty)
  • ensemble_summary_*.json (aggregate stats, member info)
  • uncertainty_maps/*.npz  (spatial maps)

Outputs saved to results/het_ensemble/visualizations/

Usage:
    conda run -n tf_gpu python scripts/visualize_ensemble_uncertainty.py
"""

import sys, os, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / 'results' / 'het_ensemble'
VIZ_DIR      = RESULTS_DIR / 'visualizations'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Auto-detect the latest CSV / JSON
csv_files = sorted(RESULTS_DIR.glob('ensemble_eval_*.csv'))
json_files = sorted(RESULTS_DIR.glob('ensemble_summary_*.json'))
assert csv_files, f"No ensemble_eval CSV in {RESULTS_DIR}"
assert json_files, f"No ensemble_summary JSON in {RESULTS_DIR}"
CSV_PATH  = csv_files[-1]
JSON_PATH = json_files[-1]
MAPS_DIR  = RESULTS_DIR / 'uncertainty_maps'

print(f"CSV : {CSV_PATH.name}")
print(f"JSON: {JSON_PATH.name}")
print(f"Maps: {MAPS_DIR}  ({len(list(MAPS_DIR.glob('*.npz')))} files)")
print(f"Output: {VIZ_DIR}")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
with open(JSON_PATH) as f:
    summary = json.load(f)

print(f"\nSamples: {len(df)}")
print(f"Columns: {list(df.columns)}")

# ──────────────────────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
C_ALEA = '#2196F3'   # blue
C_EPIS = '#FF9800'   # orange
C_PRED = '#4CAF50'   # green
C_10M  = '#E91E63'   # pink
C_20M  = '#009688'   # teal


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Error vs Uncertainty scatter (the money plot)
# ══════════════════════════════════════════════════════════════════════════════
def plot_error_vs_uncertainty():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res, c in zip(axes, ['10m', '20m'], [C_10M, C_20M]):
        err  = df[f'rmse_{res}']
        unc  = df[f'predictive_{res}']
        # Pearson correlation
        r, p = stats.pearsonr(err, unc)
        # Spearman (rank)
        rho, _ = stats.spearmanr(err, unc)

        ax.scatter(unc, err, s=8, alpha=0.35, c=c, edgecolors='none')
        # Best fit line
        m, b = np.polyfit(unc, err, 1)
        x_line = np.linspace(unc.min(), unc.max(), 100)
        ax.plot(x_line, m * x_line + b, 'k--', lw=1.5, label=f'fit (slope={m:.1f})')
        ax.set_xlabel('Predictive Uncertainty (σ²)')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{res} bands — Error vs Uncertainty')
        ax.legend(title=f'Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}', loc='upper left')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Error–Uncertainty Correlation  (well-calibrated → strong positive correlation)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / 'fig1_error_vs_uncertainty.png')
    plt.close(fig)
    print("✓ Figure 1: Error vs Uncertainty scatter")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Aleatoric vs Epistemic decomposition scatter
# ══════════════════════════════════════════════════════════════════════════════
def plot_aleatoric_vs_epistemic():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res in zip(axes, ['10m', '20m']):
        alea = df[f'aleatoric_{res}']
        epis = df[f'epistemic_{res}']
        ratio = epis / (alea + epis + 1e-12)

        sc = ax.scatter(alea, epis, c=ratio, s=8, alpha=0.5,
                        cmap='coolwarm', vmin=0, vmax=1, edgecolors='none')
        ax.plot([0, max(alea.max(), epis.max())],
                [0, max(alea.max(), epis.max())],
                'k--', lw=1, alpha=0.5, label='1:1 line')
        ax.set_xlabel('Aleatoric Uncertainty')
        ax.set_ylabel('Epistemic Uncertainty')
        ax.set_title(f'{res} bands')
        ax.legend()
        ax.grid(True, alpha=0.3)
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label('Epistemic Fraction')

    fig.suptitle('Uncertainty Decomposition: Aleatoric (data noise) vs Epistemic (model uncertainty)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / 'fig2_aleatoric_vs_epistemic.png')
    plt.close(fig)
    print("✓ Figure 2: Aleatoric vs Epistemic scatter")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Uncertainty component histograms
# ══════════════════════════════════════════════════════════════════════════════
def plot_uncertainty_histograms():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for row, res in enumerate(['10m', '20m']):
        for col, (comp, color, label) in enumerate([
            ('aleatoric', C_ALEA, 'Aleatoric'),
            ('epistemic', C_EPIS, 'Epistemic'),
            ('predictive', C_PRED, 'Predictive (Total)')
        ]):
            ax = axes[row, col]
            vals = df[f'{comp}_{res}']
            ax.hist(vals, bins=80, color=color, alpha=0.75, edgecolor='white', linewidth=0.3)
            ax.axvline(vals.mean(), color='black', ls='--', lw=1.5, label=f'mean={vals.mean():.5f}')
            ax.axvline(vals.median(), color='gray', ls=':', lw=1.5, label=f'median={vals.median():.5f}')
            ax.set_title(f'{label} — {res}')
            ax.set_xlabel('Uncertainty (σ²)')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle('Distribution of Uncertainty Components', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / 'fig3_uncertainty_histograms.png')
    plt.close(fig)
    print("✓ Figure 3: Uncertainty histograms")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Calibration / Reliability diagram
# ══════════════════════════════════════════════════════════════════════════════
def plot_calibration_curve():
    """If the model is well-calibrated, the fraction of errors within
    the predicted ±k·σ interval should match the expected coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res, c in zip(axes, ['10m', '20m'], [C_10M, C_20M]):
        err = df[f'rmse_{res}'].values
        sigma = np.sqrt(df[f'predictive_{res}'].values + 1e-12)

        # Normalised errors (z-scores)
        z = err / sigma

        # Expected quantiles vs observed coverage
        expected_coverages = np.linspace(0.05, 0.95, 19)
        observed_coverages = []
        for q in expected_coverages:
            # For Gaussian, the interval [-z_q, z_q] captures q of the mass
            z_q = stats.norm.ppf(0.5 + q / 2)
            frac_inside = np.mean(z <= z_q)
            observed_coverages.append(frac_inside)

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
        ax.plot(expected_coverages, observed_coverages, 'o-', color=c, lw=2, ms=5,
                label=f'{res} bands')
        ax.fill_between(expected_coverages, expected_coverages,
                        observed_coverages, alpha=0.15, color=c)
        ax.set_xlabel('Expected Coverage')
        ax.set_ylabel('Observed Coverage')
        ax.set_title(f'Calibration Curve — {res}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    fig.suptitle('Reliability Diagram: does predicted uncertainty match true error distribution?',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / 'fig4_calibration_curve.png')
    plt.close(fig)
    print("✓ Figure 4: Calibration curve")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Sparsification plot (oracle error curve)
# ══════════════════════════════════════════════════════════════════════════════
def plot_sparsification():
    """Remove the most uncertain samples first → PSNR of remaining should rise
    monotonically if uncertainty is informative."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res, c in zip(axes, ['10m', '20m'], [C_10M, C_20M]):
        psnr = df[f'psnr_{res}'].values
        unc  = df[f'predictive_{res}'].values
        err  = df[f'rmse_{res}'].values

        # Sort by uncertainty (descending → remove most uncertain first)
        order_unc = np.argsort(-unc)
        # Oracle: sort by actual error (descending)
        order_oracle = np.argsort(-err)
        # Random baseline
        rng = np.random.RandomState(42)
        order_random = rng.permutation(len(psnr))

        fracs = np.linspace(0, 0.95, 50)
        psnr_unc, psnr_oracle, psnr_random = [], [], []

        for frac in fracs:
            n_remove = int(frac * len(psnr))
            remaining_unc = np.delete(psnr, order_unc[:n_remove])
            remaining_oracle = np.delete(psnr, order_oracle[:n_remove])
            remaining_random = np.delete(psnr, order_random[:n_remove])

            psnr_unc.append(remaining_unc.mean() if len(remaining_unc) else np.nan)
            psnr_oracle.append(remaining_oracle.mean() if len(remaining_oracle) else np.nan)
            psnr_random.append(remaining_random.mean() if len(remaining_random) else np.nan)

        ax.plot(fracs * 100, psnr_oracle, 'k-', lw=2, label='Oracle (sort by error)')
        ax.plot(fracs * 100, psnr_unc, '-', color=c, lw=2, label='Uncertainty (sort by σ²)')
        ax.plot(fracs * 100, psnr_random, '--', color='gray', lw=1.5, label='Random')

        # Area between uncertainty and random (AUSE proxy)
        ax.fill_between(fracs * 100, psnr_random, psnr_unc, alpha=0.15, color=c)

        ax.set_xlabel('% Samples Removed (most uncertain first)')
        ax.set_ylabel('Mean PSNR (dB) of Remaining')
        ax.set_title(f'Sparsification — {res}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Sparsification: removing high-uncertainty samples should improve average PSNR',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / 'fig5_sparsification.png')
    plt.close(fig)
    print("✓ Figure 5: Sparsification plot")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Spatial uncertainty maps (example tiles)
# ══════════════════════════════════════════════════════════════════════════════
def plot_spatial_uncertainty_maps():
    """Show prediction, aleatoric, epistemic side-by-side for 3 samples:
    low-uncertainty, medium-uncertainty, high-uncertainty."""
    if not MAPS_DIR.exists():
        print("⚠ No uncertainty_maps dir — skipping Figure 6")
        return

    # Pick samples at different uncertainty levels
    unc_10m = df['predictive_10m'].values
    sorted_idx = np.argsort(unc_10m)
    n = len(sorted_idx)
    picks = [
        ('Low Uncertainty', sorted_idx[n // 10]),         # ~10th percentile
        ('Medium Uncertainty', sorted_idx[n // 2]),        # median
        ('High Uncertainty', sorted_idx[int(n * 0.95)]),  # ~95th percentile
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))

    for row, (label, csv_row_idx) in enumerate(picks):
        sample_idx = df.iloc[csv_row_idx]['sample_idx']
        npz_path = MAPS_DIR / f'sample_{int(sample_idx):05d}.npz'
        if not npz_path.exists():
            print(f"  ⚠ {npz_path.name} missing — skipping")
            continue

        data = np.load(npz_path)
        # Use the middle temporal frame (t=1) and make RGB-like from first 3 bands for 10m
        pred = data['pred_10m'][1]       # (3, 256, 256)
        alea = data['aleatoric_10m'][1]  # (3, 256, 256)
        epis = data['epistemic_10m'][1]  # (3, 256, 256)

        # RGB composite (bands 0,1,2) — clip to [0,1]
        rgb = np.clip(pred.transpose(1, 2, 0), 0, 1)
        # Mean over bands for uncertainty maps
        alea_map = alea.mean(axis=0)
        epis_map = epis.mean(axis=0)
        total_map = alea_map + epis_map

        psnr_val = df.iloc[csv_row_idx][f'psnr_10m']

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f'{label}\nPSNR={psnr_val:.1f} dB', fontsize=10)
        axes[row, 0].axis('off')

        im1 = axes[row, 1].imshow(alea_map, cmap='hot', vmin=0)
        axes[row, 1].set_title('Aleatoric (data noise)', fontsize=10)
        axes[row, 1].axis('off')
        fig.colorbar(im1, ax=axes[row, 1], shrink=0.7)

        im2 = axes[row, 2].imshow(epis_map, cmap='hot', vmin=0)
        axes[row, 2].set_title('Epistemic (model uncertainty)', fontsize=10)
        axes[row, 2].axis('off')
        fig.colorbar(im2, ax=axes[row, 2], shrink=0.7)

        im3 = axes[row, 3].imshow(total_map, cmap='hot', vmin=0)
        axes[row, 3].set_title('Total Uncertainty', fontsize=10)
        axes[row, 3].axis('off')
        fig.colorbar(im3, ax=axes[row, 3], shrink=0.7)

    fig.suptitle('Spatial Uncertainty Maps (10m bands, middle temporal frame)',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / 'fig6_spatial_uncertainty_maps.png')
    plt.close(fig)
    print("✓ Figure 6: Spatial uncertainty maps")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Uncertainty decomposition bar chart (mean proportions)
# ══════════════════════════════════════════════════════════════════════════════
def plot_uncertainty_decomposition_bar():
    fig, ax = plt.subplots(figsize=(8, 5))

    resolutions = ['10m', '20m']
    alea_means = [df[f'aleatoric_{r}'].mean() for r in resolutions]
    epis_means = [df[f'epistemic_{r}'].mean() for r in resolutions]

    x = np.arange(len(resolutions))
    w = 0.35

    bars1 = ax.bar(x - w/2, alea_means, w, label='Aleatoric (irreducible)',
                   color=C_ALEA, edgecolor='white')
    bars2 = ax.bar(x + w/2, epis_means, w, label='Epistemic (reducible)',
                   color=C_EPIS, edgecolor='white')

    # Annotate percentages
    for i, (a, e) in enumerate(zip(alea_means, epis_means)):
        total = a + e
        ax.text(x[i] - w/2, a + total * 0.02, f'{a/total*100:.1f}%',
                ha='center', fontsize=10, fontweight='bold')
        ax.text(x[i] + w/2, e + total * 0.02, f'{e/total*100:.1f}%',
                ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['10m Bands (RGB)', '20m Bands (SWIR)'])
    ax.set_ylabel('Mean Uncertainty (σ²)')
    ax.set_title('Uncertainty Decomposition: How Much is Reducible?')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(VIZ_DIR / 'fig7_uncertainty_decomposition.png')
    plt.close(fig)
    print("✓ Figure 7: Uncertainty decomposition bar chart")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Per-member PSNR comparison (ensemble diversity)
# ══════════════════════════════════════════════════════════════════════════════
def plot_member_comparison():
    members = summary.get('member_info', [])
    if not members:
        print("⚠ No member_info in summary — skipping Figure 8")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    seeds = [m['seed'] for m in members]
    psnr_10m = [m['val_psnr_10m'] for m in members]
    psnr_20m = [m['val_psnr_20m'] for m in members]
    epochs = [m['epoch'] for m in members]

    # Left: PSNR by member
    x = np.arange(len(seeds))
    axes[0].bar(x - 0.2, psnr_10m, 0.35, label='10m', color=C_10M, edgecolor='white')
    axes[0].bar(x + 0.2, psnr_20m, 0.35, label='20m', color=C_20M, edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'M{i}\n(seed {s})' for i, s in enumerate(seeds)], fontsize=9)
    axes[0].set_ylabel('Val PSNR (dB)')
    axes[0].set_title('Per-Member Validation PSNR')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add ensemble mean line
    ens_psnr_10m = summary['metrics']['psnr_10m']['mean']
    ens_psnr_20m = summary['metrics']['psnr_20m']['mean']
    axes[0].axhline(ens_psnr_10m, color=C_10M, ls='--', lw=1.5, alpha=0.7)
    axes[0].axhline(ens_psnr_20m, color=C_20M, ls='--', lw=1.5, alpha=0.7)
    axes[0].annotate(f'Ensemble: {ens_psnr_10m:.1f}',
                     xy=(len(seeds)-0.5, ens_psnr_10m), fontsize=9, color=C_10M)

    # Right: convergence epochs
    axes[1].bar(x, epochs, color='#7E57C2', edgecolor='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'M{i}' for i in range(len(seeds))])
    axes[1].set_ylabel('Best Epoch')
    axes[1].set_title('Convergence Speed per Member')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, ep in enumerate(epochs):
        axes[1].text(i, ep + 1, str(ep), ha='center', fontsize=10)

    fig.suptitle('Ensemble Diversity Analysis (5 Members)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / 'fig8_member_comparison.png')
    plt.close(fig)
    print("✓ Figure 8: Per-member comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — PSNR distribution with uncertainty bands
# ══════════════════════════════════════════════════════════════════════════════
def plot_psnr_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, res, c in zip(axes, ['10m', '20m'], [C_10M, C_20M]):
        psnr = df[f'psnr_{res}']
        unc  = df[f'predictive_{res}']

        ax.hist(psnr, bins=60, color=c, alpha=0.7, edgecolor='white', linewidth=0.3)
        ax.axvline(psnr.mean(), color='black', ls='--', lw=2,
                   label=f'Mean={psnr.mean():.2f} ± {psnr.std():.2f} dB')
        ax.axvline(psnr.median(), color='gray', ls=':', lw=2,
                   label=f'Median={psnr.median():.2f} dB')
        ax.set_xlabel('PSNR (dB)')
        ax.set_ylabel('Count')
        ax.set_title(f'{res} Bands — Ensemble PSNR Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Predictive Quality Distribution (Ensemble Mean)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / 'fig9_psnr_distribution.png')
    plt.close(fig)
    print("✓ Figure 9: PSNR distribution")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Summary dashboard (one-pager)
# ══════════════════════════════════════════════════════════════════════════════
def plot_summary_dashboard():
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Top-left: Error vs uncertainty scatter (10m)
    ax1 = fig.add_subplot(gs[0, 0])
    err = df['rmse_10m']
    unc = df['predictive_10m']
    r, _ = stats.pearsonr(err, unc)
    ax1.scatter(unc, err, s=5, alpha=0.3, c=C_10M, edgecolors='none')
    m, b = np.polyfit(unc, err, 1)
    x_line = np.linspace(unc.min(), unc.max(), 100)
    ax1.plot(x_line, m * x_line + b, 'k--', lw=1.5)
    ax1.set_xlabel('Predictive σ²')
    ax1.set_ylabel('RMSE')
    ax1.set_title(f'Error–Uncertainty (10m)\nr = {r:.3f}')
    ax1.grid(True, alpha=0.3)

    # Top-center: Decomposition bar
    ax2 = fig.add_subplot(gs[0, 1])
    for i, res in enumerate(['10m', '20m']):
        a = df[f'aleatoric_{res}'].mean()
        e = df[f'epistemic_{res}'].mean()
        ax2.bar(i, a, color=C_ALEA, label='Aleatoric' if i == 0 else '')
        ax2.bar(i, e, bottom=a, color=C_EPIS, label='Epistemic' if i == 0 else '')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['10m', '20m'])
    ax2.set_ylabel('Mean σ²')
    ax2.set_title('Uncertainty Decomposition')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Top-right: PSNR histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df['psnr_10m'], bins=50, alpha=0.7, color=C_10M, label='10m', edgecolor='white')
    ax3.hist(df['psnr_20m'], bins=50, alpha=0.5, color=C_20M, label='20m', edgecolor='white')
    ax3.set_xlabel('PSNR (dB)')
    ax3.set_title('PSNR Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom-left: Sparsification (10m)
    ax4 = fig.add_subplot(gs[1, 0])
    psnr = df['psnr_10m'].values
    unc_vals = df['predictive_10m'].values
    err_vals = df['rmse_10m'].values
    order_unc = np.argsort(-unc_vals)
    order_oracle = np.argsort(-err_vals)
    fracs = np.linspace(0, 0.9, 30)
    p_unc, p_orc = [], []
    for frac in fracs:
        n_rm = int(frac * len(psnr))
        p_unc.append(np.delete(psnr, order_unc[:n_rm]).mean())
        p_orc.append(np.delete(psnr, order_oracle[:n_rm]).mean())
    ax4.plot(fracs * 100, p_orc, 'k-', lw=2, label='Oracle')
    ax4.plot(fracs * 100, p_unc, '-', color=C_10M, lw=2, label='Uncertainty')
    ax4.set_xlabel('% Removed')
    ax4.set_ylabel('Mean PSNR (dB)')
    ax4.set_title('Sparsification (10m)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Bottom-center: Calibration (10m)
    ax5 = fig.add_subplot(gs[1, 1])
    sigma = np.sqrt(df['predictive_10m'].values + 1e-12)
    z = df['rmse_10m'].values / sigma
    exp_cov = np.linspace(0.05, 0.95, 19)
    obs_cov = [np.mean(z <= stats.norm.ppf(0.5 + q/2)) for q in exp_cov]
    ax5.plot([0, 1], [0, 1], 'k--', lw=1)
    ax5.plot(exp_cov, obs_cov, 'o-', color=C_10M, lw=2, ms=4)
    ax5.set_xlabel('Expected Coverage')
    ax5.set_ylabel('Observed Coverage')
    ax5.set_title('Calibration (10m)')
    ax5.set_aspect('equal')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)

    # Bottom-right: Key metrics text box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    u = summary.get('uncertainty', {})
    m = summary.get('metrics', {})
    text = (
        f"═══ Ensemble Summary ═══\n"
        f"Members: {summary.get('n_members', '?')}\n"
        f"Test samples: {summary.get('n_samples', '?')}\n\n"
        f"─── Metrics (ensemble mean) ───\n"
        f"PSNR 10m: {m.get('psnr_10m',{}).get('mean',0):.2f} ± {m.get('psnr_10m',{}).get('std',0):.2f} dB\n"
        f"PSNR 20m: {m.get('psnr_20m',{}).get('mean',0):.2f} ± {m.get('psnr_20m',{}).get('std',0):.2f} dB\n"
        f"SSIM 10m: {m.get('ssim_10m',{}).get('mean',0):.4f}\n"
        f"SSIM 20m: {m.get('ssim_20m',{}).get('mean',0):.4f}\n"
        f"RMSE 10m: {m.get('rmse_10m',{}).get('mean',0):.5f}\n"
        f"RMSE 20m: {m.get('rmse_20m',{}).get('mean',0):.5f}\n\n"
        f"─── Calibration ───\n"
        f"Error–Unc corr (10m): {u.get('error_uncertainty_correlation_10m',0):.3f}\n"
        f"Error–Unc corr (20m): {u.get('error_uncertainty_correlation_20m',0):.3f}\n\n"
        f"─── Uncertainty ───\n"
        f"Aleatoric 10m: {u.get('aleatoric_10m',{}).get('mean',0):.6f}\n"
        f"Epistemic 10m: {u.get('epistemic_10m',{}).get('mean',0):.6f}\n"
        f"Aleatoric 20m: {u.get('aleatoric_20m',{}).get('mean',0):.6f}\n"
        f"Epistemic 20m: {u.get('epistemic_20m',{}).get('mean',0):.6f}\n"
    )
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Heteroscedastic Deep Ensemble — Results Dashboard',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.savefig(VIZ_DIR / 'fig10_summary_dashboard.png')
    plt.close(fig)
    print("✓ Figure 10: Summary dashboard")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Generating Ensemble Uncertainty Visualizations")
    print("="*60 + "\n")

    plot_error_vs_uncertainty()       # Fig 1
    plot_aleatoric_vs_epistemic()     # Fig 2
    plot_uncertainty_histograms()     # Fig 3
    plot_calibration_curve()          # Fig 4
    plot_sparsification()             # Fig 5
    plot_spatial_uncertainty_maps()   # Fig 6
    plot_uncertainty_decomposition_bar()  # Fig 7
    plot_member_comparison()          # Fig 8
    plot_psnr_distribution()          # Fig 9
    plot_summary_dashboard()          # Fig 10

    print(f"\n✅ All figures saved to: {VIZ_DIR}/")
    print(f"   Files: {sorted([f.name for f in VIZ_DIR.glob('*.png')])}")
