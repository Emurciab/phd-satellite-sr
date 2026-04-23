#!/usr/bin/env python3
"""
Two single-page summary figures — one per resolution — combining 6 key
uncertainty plots into one cohesive dashboard:

  (a) Error vs Uncertainty scatter       [fig1]
  (b) Aleatoric vs Epistemic scatter     [fig2]
  (c) Calibration / Reliability curve    [fig4]
  (d) Sparsification plot                [fig5]
  (e) Uncertainty Decomposition bar      [fig7]
  (f) PSNR Distribution histogram        [fig9]

Output:
  results/het_ensemble/visualizations/uncertainty_summary_10m.png
  results/het_ensemble/visualizations/uncertainty_summary_20m.png
"""

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RES  = ROOT / 'results' / 'het_ensemble'
VIZ  = RES / 'visualizations'; VIZ.mkdir(exist_ok=True)

CSV  = sorted(RES.glob('ensemble_eval_*.csv'))[-1]
JSON = sorted(RES.glob('ensemble_summary_*.json'))[-1]
df   = pd.read_csv(CSV)
with open(JSON) as f:
    summary = json.load(f)

print(f"CSV : {CSV.name}  ({len(df)} samples)")
print(f"JSON: {JSON.name}")

# ── colours ──────────────────────────────────────────────────────────────────
C_ALEA  = '#2196F3'
C_EPIS  = '#FF9800'
C_PRED  = '#4CAF50'
C_MAIN  = {'10m': '#E91E63', '20m': '#009688'}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── helper: sparsification curves ────────────────────────────────────────────
def _sparsification(psnr, unc, err, n_pts=50):
    order_unc = np.argsort(-unc)
    order_orc = np.argsort(-err)
    rng = np.random.RandomState(42)
    order_rnd = rng.permutation(len(psnr))
    fracs = np.linspace(0, 0.9, n_pts)
    def _curve(order):
        return [np.delete(psnr, order[:int(f*len(psnr))]).mean() for f in fracs]
    return fracs, _curve(order_orc), _curve(order_unc), _curve(order_rnd)


def make_figure(res: str):
    """Build the 6-panel uncertainty dashboard for one resolution."""
    c = C_MAIN[res]
    label = '10 m (RGB)' if res == '10m' else '20 m (SWIR)'

    err   = df[f'rmse_{res}'].values
    psnr  = df[f'psnr_{res}'].values
    ssim  = df[f'ssim_{res}'].values
    alea  = df[f'aleatoric_{res}'].values
    epis  = df[f'epistemic_{res}'].values
    pred  = df[f'predictive_{res}'].values
    sigma = np.sqrt(pred + 1e-12)

    r_pear, _  = stats.pearsonr(err, pred)
    r_spear, _ = stats.spearmanr(err, pred)

    # ── layout: 2 rows × 3 cols ─────────────────────────────────────────
    fig = plt.figure(figsize=(21, 13))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.30)

    # ====================================================================
    # (a) Error vs Uncertainty scatter  [fig1]
    # ====================================================================
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(pred, err, s=6, alpha=0.35, c=c, edgecolors='none')
    m_slope, b_int = np.polyfit(pred, err, 1)
    xl = np.linspace(pred.min(), pred.max(), 100)
    ax.plot(xl, m_slope * xl + b_int, 'k--', lw=1.5)
    ax.set_xlabel('Predictive Uncertainty (σ²)')
    ax.set_ylabel('RMSE')
    ax.set_title('(a)  Error vs Uncertainty', fontweight='bold')
    ax.text(0.03, 0.95,
            f'Pearson  r = {r_pear:.3f}\nSpearman ρ = {r_spear:.3f}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.grid(True, alpha=0.25)

    # ====================================================================
    # (b) Aleatoric vs Epistemic scatter  [fig2]
    # ====================================================================
    ax = fig.add_subplot(gs[0, 1])
    ratio = epis / (alea + epis + 1e-12)
    sc = ax.scatter(alea, epis, c=ratio, s=6, alpha=0.5,
                    cmap='coolwarm', vmin=0, vmax=1, edgecolors='none')
    lim = max(alea.max(), epis.max())
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.4, label='1:1 line')
    ax.set_xlabel('Aleatoric Uncertainty')
    ax.set_ylabel('Epistemic Uncertainty')
    ax.set_title('(b)  Aleatoric vs Epistemic', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label('Epistemic Fraction', fontsize=9)
    ax.grid(True, alpha=0.25)

    # ====================================================================
    # (c) Calibration / Reliability curve  [fig4]
    # ====================================================================
    ax = fig.add_subplot(gs[0, 2])
    z = err / sigma
    exp_cov = np.linspace(0.05, 0.95, 30)
    obs_cov = np.array([np.mean(z <= stats.norm.ppf(0.5 + q / 2))
                        for q in exp_cov])
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    ax.plot(exp_cov, obs_cov, 'o-', color=c, lw=2, ms=4, label=f'{label}')
    ax.fill_between(exp_cov, exp_cov, obs_cov, alpha=0.12, color=c)
    mca = np.mean(np.abs(obs_cov - exp_cov))
    ax.text(0.55, 0.15, f'Miscal. Area = {mca:.3f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_title('(c)  Calibration / Reliability', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)

    # ====================================================================
    # (d) Sparsification plot  [fig5]
    # ====================================================================
    ax = fig.add_subplot(gs[1, 0])
    fracs, p_orc, p_unc, p_rnd = _sparsification(psnr, pred, err)
    ax.plot(fracs * 100, p_orc, 'k-',  lw=2, label='Oracle (sort by error)')
    ax.plot(fracs * 100, p_unc, '-',   lw=2, color=c,
            label='Uncertainty (sort by σ²)')
    ax.plot(fracs * 100, p_rnd, '--',  lw=1.5, color='gray', label='Random')
    ax.fill_between(fracs * 100, p_rnd, p_unc, alpha=0.12, color=c)
    ause = np.trapz(np.array(p_orc) - np.array(p_unc), fracs) / 0.9
    gain = np.trapz(np.array(p_unc) - np.array(p_rnd), fracs) / 0.9
    ax.text(0.03, 0.05,
            f'AUSE = {ause:.3f} dB\nGain vs random = {gain:.2f} dB',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.set_xlabel('% Samples Removed (most uncertain first)')
    ax.set_ylabel('Mean PSNR (dB)')
    ax.set_title('(d)  Sparsification', fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.25)

    # ====================================================================
    # (e) Uncertainty Decomposition bar  [fig7]
    # ====================================================================
    ax = fig.add_subplot(gs[1, 1])

    alea_mean = alea.mean()
    epis_mean = epis.mean()
    total = alea_mean + epis_mean
    alea_pct = alea_mean / total * 100
    epis_pct = epis_mean / total * 100

    bars_a = ax.bar(['Aleatoric\n(irreducible)'], [alea_mean],
                    color=C_ALEA, edgecolor='white', width=0.5)
    bars_e = ax.bar(['Epistemic\n(reducible)'],   [epis_mean],
                    color=C_EPIS, edgecolor='white', width=0.5)
    bars_t = ax.bar(['Total\n(predictive)'],      [total],
                    color=C_PRED, edgecolor='white', width=0.5, alpha=0.7)

    # Percentage labels on top
    for bar, pct in zip([bars_a[0], bars_e[0], bars_t[0]],
                        [alea_pct, epis_pct, 100.0]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + total * 0.02,
                f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # Value labels inside bars
    for bar in [bars_a[0], bars_e[0], bars_t[0]]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h / 2,
                f'{h:.2e}', ha='center', va='center', fontsize=9,
                color='white', fontweight='bold')

    ax.set_ylabel('Mean Uncertainty (σ²)')
    ax.set_title('(e)  Uncertainty Decomposition', fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y')

    # ====================================================================
    # (f) PSNR Distribution histogram  [fig9]
    # ====================================================================
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(psnr, bins=60, color=c, alpha=0.7, edgecolor='white', lw=0.3)
    ax.axvline(psnr.mean(), color='black', ls='--', lw=2,
               label=f'Mean = {psnr.mean():.2f} dB')
    ax.axvline(np.median(psnr), color='gray', ls=':', lw=2,
               label=f'Median = {np.median(psnr):.2f} dB')

    # Add key stats text box
    ax.text(0.03, 0.95,
            f'σ = {psnr.std():.2f} dB\n'
            f'SSIM = {ssim.mean():.4f}\n'
            f'RMSE = {err.mean():.5f}\n'
            f'n = {len(psnr):,}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    ax.set_xlabel('PSNR (dB)')
    ax.set_ylabel('Count')
    ax.set_title('(f)  Ensemble PSNR Distribution', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── super title ──────────────────────────────────────────────────────
    fig.suptitle(
        f'Heteroscedastic Deep Ensemble — Uncertainty Analysis  [{label}]',
        fontsize=16, fontweight='bold', y=0.99)

    out = VIZ_DIR / f'uncertainty_summary_{res}.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {out.name}  ({out.stat().st_size/1024:.0f} KB)")


# ── globals needed inside make_figure ────────────────────────────────────────
VIZ_DIR = VIZ

if __name__ == '__main__':
    make_figure('10m')
    make_figure('20m')
    print(f"\n✅ Both figures in: {VIZ}/")
