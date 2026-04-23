#!/usr/bin/env python3
"""
Regenerate improvement_heatmap_pct.png — same layout as improvement_heatmap.png
but each cell shows the absolute delta AND the percentage change vs. Bilinear baseline.

Format per cell:  +0.38  (+1.2%)
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
csv_path  = "/media/eamurciab/TOSHIBA_EXT1/PhD/SR/results/model_evaluation/evaluation_20260301_004628.csv"
out_path  = "/media/eamurciab/TOSHIBA_EXT1/PhD/SR/results/model_evaluation/visualizations/improvement_heatmap_pct.png"

# ── Load data ──────────────────────────────────────────────────────────────────
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

lookup = {r['Model']: r for r in rows}

METRIC_NAMES   = ['psnr', 'ssim', 'scc', 'rmse', 'ergas', 'sid']
HIGHER_BETTER  = ['psnr', 'ssim', 'scc']
METRIC_DISPLAY = ['PSNR', 'SSIM', 'SCC', 'RMSE', 'ERGAS', 'SID']
CSV_COLS       = {m: f"{m.upper()}_combined" for m in METRIC_NAMES}

# ── Baseline ───────────────────────────────────────────────────────────────────
bl      = lookup['Bilinear']
bl_vals = {m: float(bl[CSV_COLS[m]]) for m in METRIC_NAMES}

# ── All models except baseline ─────────────────────────────────────────────────
model_names = [r['Model'] for r in rows if r['Model'] != 'Bilinear']

# Compute improvements (positive = better) AND percentage relative to baseline
improvements = {}   # absolute delta (sign-normalised so + is always "better")
pct_changes  = {}   # percentage change relative to baseline

for name in model_names:
    r = lookup[name]
    imp = []
    pct = []
    for m in METRIC_NAMES:
        val     = float(r[CSV_COLS[m]])
        bil_val = bl_vals[m]
        if m in HIGHER_BETTER:
            delta = val - bil_val                       # positive = better
        else:
            delta = bil_val - val                       # positive = better

        imp.append(delta)

        # Percentage change relative to baseline absolute value
        if abs(bil_val) > 1e-12:
            pct.append(delta / abs(bil_val) * 100.0)
        else:
            pct.append(0.0)

    improvements[name] = imp
    pct_changes[name]  = pct

# Sort by ΔPSNR, best first
sorted_names = sorted(model_names, key=lambda n: improvements[n][0], reverse=True)
imp_matrix = np.array([improvements[n] for n in sorted_names])
pct_matrix = np.array([pct_changes[n]  for n in sorted_names])

print("Models ranked best → worst (by ΔPSNR):")
for i, name in enumerate(sorted_names):
    print(f"  #{i+1:2d}  {name:<40}  ΔPSNR={improvements[name][0]:+.2f}  "
          f"({pct_changes[name][0]:+.1f}%)")

n_models  = len(sorted_names)
n_metrics = len(METRIC_NAMES)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, max(7, n_models * 0.72)))

vmax = np.max(np.abs(imp_matrix))
im   = ax.imshow(imp_matrix, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)

ax.set_xticks(np.arange(n_metrics))
ax.set_yticks(np.arange(n_models))
ax.set_xticklabels(METRIC_DISPLAY, fontsize=13, fontweight='bold')
ax.set_yticklabels(sorted_names, fontsize=12)
ax.tick_params(axis='y', pad=15)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# ── Cell annotations:  delta  (pct%) ──────────────────────────────────────────
for i in range(n_models):
    for j in range(n_metrics):
        val = imp_matrix[i, j]
        pct = pct_matrix[i, j]

        # Adaptive text colour
        norm_val   = (val + vmax) / (2 * vmax)
        text_color = 'white' if norm_val < 0.25 or norm_val > 0.75 else 'black'

        # Format absolute delta
        if abs(val) >= 1:
            delta_str = f'{val:+.2f}'
        else:
            delta_str = f'{val:+.4f}'

        # Format percentage
        pct_str = f'({pct:+.1f}%)'

        # Two-line annotation: delta on top, percentage below
        ax.text(j, i - 0.13, delta_str, ha='center', va='center',
                color=text_color, fontsize=9.5, fontweight='bold')
        ax.text(j, i + 0.18, pct_str, ha='center', va='center',
                color=text_color, fontsize=8, fontstyle='italic')

# ── Baseline separator ────────────────────────────────────────────────────────
boundary = None
for i, name in enumerate(sorted_names):
    if improvements[name][0] < 0:
        boundary = i
        break

if boundary is not None:
    ax.axhline(y=boundary - 0.5, color='black', linewidth=2, linestyle='--')
    ax.text(n_metrics + 0.1, boundary - 0.5, '← Baseline',
            va='center', ha='left', fontsize=10, fontweight='bold', color='black',
            clip_on=False)

ax.set_title('Improvement over Bilinear Baseline — Absolute Δ & Relative %\n'
             'Green = Better · Red = Worse  (positive = improvement)',
             fontsize=14, fontweight='bold', pad=30)

cbar = fig.colorbar(im, ax=ax, label='Improvement (positive = better)',
                     shrink=0.8, pad=0.02)

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Saved: {out_path}")
