#!/usr/bin/env python3
"""Regenerate improvement_heatmap.png ranked from best to worst performing model."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = "/media/eamurciab/TOSHIBA_EXT1/PhD/SR/results/model_evaluation/evaluation_20260301_004628.csv"
out_path = "/media/eamurciab/TOSHIBA_EXT1/PhD/SR/results/model_evaluation/visualizations/improvement_heatmap.png"

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

lookup = {r['Model']: r for r in rows}

METRIC_NAMES = ['psnr', 'ssim', 'scc', 'rmse', 'ergas', 'sid']
HIGHER_BETTER = ['psnr', 'ssim', 'scc']
METRIC_DISPLAY = ['PSNR', 'SSIM', 'SCC', 'RMSE', 'ERGAS', 'SID']
CSV_COLS = {m: f"{m.upper()}_combined" for m in METRIC_NAMES}

# Baseline
bl = lookup['Bilinear']
bl_vals = {m: float(bl[CSV_COLS[m]]) for m in METRIC_NAMES}

# All models except baseline
model_names = [r['Model'] for r in rows if r['Model'] != 'Bilinear']

# Compute improvements (positive = better than baseline for ALL metrics)
improvements = {}
for name in model_names:
    r = lookup[name]
    imp = []
    for m in METRIC_NAMES:
        val = float(r[CSV_COLS[m]])
        bil_val = bl_vals[m]
        if m in HIGHER_BETTER:
            imp.append(val - bil_val)
        else:
            imp.append(bil_val - val)
    improvements[name] = imp

# Sort by PSNR improvement (index 0), best first
sorted_names = sorted(model_names, key=lambda n: improvements[n][0], reverse=True)
imp_matrix = np.array([improvements[n] for n in sorted_names])

print("Models ranked best → worst (by ΔPSNR):")
for i, name in enumerate(sorted_names):
    print(f"  #{i+1:2d}  {name:<40}  ΔPSNR={improvements[name][0]:+.2f}")

n_models = len(sorted_names)
n_metrics = len(METRIC_NAMES)

# --- Plot ---
fig, ax = plt.subplots(figsize=(16, max(6, n_models * 0.65)))

# Symmetric diverging colormap centered at 0
vmax = np.max(np.abs(imp_matrix))
im = ax.imshow(imp_matrix, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)

ax.set_xticks(np.arange(n_metrics))
ax.set_yticks(np.arange(n_models))
ax.set_xticklabels(METRIC_DISPLAY, fontsize=13, fontweight='bold')
ax.set_yticklabels(sorted_names, fontsize=12)

# Extra padding between y-axis labels and the heatmap cells
ax.tick_params(axis='y', pad=15)

# X labels on top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Text annotations with adaptive color
for i in range(n_models):
    for j in range(n_metrics):
        val = imp_matrix[i, j]
        norm_val = (val + vmax) / (2 * vmax)
        text_color = 'white' if norm_val < 0.25 or norm_val > 0.75 else 'black'
        if abs(val) >= 1:
            fmt = f'{val:+.2f}'
        else:
            fmt = f'{val:+.4f}'
        ax.text(j, i, fmt, ha='center', va='center',
                color=text_color, fontsize=11, fontweight='bold')

# Dashed line at baseline boundary
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

# Rank numbers — pushed further left to avoid overlap with model names
# for i in range(n_models):
    # ax.text(-1.0, i, f'#{i+1}', ha='center', va='center',
            # fontsize=10, fontweight='bold', color='gray', clip_on=False)

ax.set_title('Improvement over Bilinear Baseline (Ranked Best → Worst)\n'
             'Green = Better than Baseline  ·  Red = Worse than Baseline',
             fontsize=14, fontweight='bold', pad=30)

cbar = fig.colorbar(im, ax=ax, label='Improvement (positive = better)',
                     shrink=0.8, pad=0.02)

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")
