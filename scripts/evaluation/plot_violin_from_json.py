#!/usr/bin/env python3
"""
Generate violin plots for each metric per model per branch (10m and 20m)
from real per-sample evaluation data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from pathlib import Path

# Use a clean style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.facecolor': '#FAFAFA',
    'figure.facecolor': 'white',
})

# Color palette
COLORS = {
    'Residual': {'face': '#4C72B0', 'edge': '#2E4A7A', 'dot': '#2B4F8C'},
    'Direct':   {'face': '#DD8452', 'edge': '#A85C30', 'dot': '#B5612A'},
    'Bilinear': {'face': '#AAAAAA', 'edge': '#666666', 'dot': '#555555'},
    'Other':    {'face': '#55A868', 'edge': '#337A48', 'dot': '#2D6B3F'},
}


def _get_color_key(label):
    if 'Residual' in label:
        return 'Residual'
    elif 'Direct' in label:
        return 'Direct'
    elif label == 'Bilinear':
        return 'Bilinear'
    return 'Other'


def load_per_sample(json_path):
    """Load per-sample metrics JSON."""
    with open(json_path) as f:
        return json.load(f)


def make_violin_plots(data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(data.keys())
    metrics = ['psnr', 'ssim', 'scc', 'rmse', 'ergas', 'sid']
    metric_labels = {
        'psnr': 'PSNR (dB)', 'ssim': 'SSIM', 'scc': 'SCC',
        'rmse': 'RMSE', 'ergas': 'ERGAS', 'sid': 'SID'
    }
    higher_better = {'psnr', 'ssim', 'scc'}
    branches = ['10m', '20m']

    for metric in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(30, 10), sharey=True)
        fig.suptitle(f'{metric_labels[metric]} Distribution per Model',
                     fontsize=20, fontweight='bold', y=0.98)

        for ax_idx, branch in enumerate(branches):
            ax = axes[ax_idx]
            sample_key = f'per_sample_{branch}'

            all_samples = []
            labels = []
            means = []

            for model_name in model_names:
                per_sample_list = data[model_name][sample_key]
                samples = np.array([s[metric] for s in per_sample_list])
                all_samples.append(samples)
                labels.append(model_name)
                means.append(np.mean(samples))

            # Sort by mean (best first)
            if metric in higher_better:
                order = np.argsort(means)[::-1]
            else:
                order = np.argsort(means)

            sorted_samples = [all_samples[i] for i in order]
            sorted_labels = [labels[i] for i in order]

            positions = np.arange(len(sorted_labels))

            # --- Violin bodies (no built-in stats, we draw our own) ---
            parts = ax.violinplot(sorted_samples, positions=positions,
                                  showmeans=False, showmedians=False,
                                  showextrema=False)

            for i, (pc, label) in enumerate(zip(parts['bodies'], sorted_labels)):
                ck = _get_color_key(label)
                pc.set_facecolor(COLORS[ck]['face'])
                pc.set_edgecolor(COLORS[ck]['edge'])
                pc.set_linewidth(0.8)
                pc.set_alpha(0.55)

            # --- Overlay: jittered scatter points ---
            rng = np.random.default_rng(42)
            for i, (samples, label) in enumerate(zip(sorted_samples, sorted_labels)):
                ck = _get_color_key(label)
                # Subsample if > 500 points to avoid over-plotting
                if len(samples) > 500:
                    idx = rng.choice(len(samples), 500, replace=False)
                    pts = samples[idx]
                else:
                    pts = samples
                jitter = rng.uniform(-0.15, 0.15, size=len(pts))
                ax.scatter(positions[i] + jitter, pts, s=2, alpha=0.08,
                           color=COLORS[ck]['dot'], zorder=2, rasterized=True)

            # --- Overlay: box-plot style statistics ---
            for i, (samples, label) in enumerate(zip(sorted_samples, sorted_labels)):
                q25, med, q75 = np.percentile(samples, [25, 50, 75])
                mean_val = np.mean(samples)
                iqr = q75 - q25
                whisker_lo = max(samples.min(), q25 - 1.5 * iqr)
                whisker_hi = min(samples.max(), q75 + 1.5 * iqr)
                ck = _get_color_key(label)

                # Whisker lines
                ax.vlines(positions[i], whisker_lo, whisker_hi,
                          color='#333333', linewidth=0.8, zorder=3)

                # IQR box
                box_w = 0.18
                rect = plt.Rectangle((positions[i] - box_w / 2, q25),
                                     box_w, iqr, linewidth=1,
                                     edgecolor='#333333', facecolor=COLORS[ck]['face'],
                                     alpha=0.85, zorder=4)
                ax.add_patch(rect)

                # Median line (white, prominent)
                ax.hlines(med, positions[i] - box_w / 2, positions[i] + box_w / 2,
                          color='white', linewidth=2, zorder=5)

                # Mean marker (diamond)
                ax.scatter(positions[i], mean_val, marker='D', s=22,
                           color='#222222', edgecolors='white', linewidths=0.5,
                           zorder=6)

                # Whisker caps
                cap_w = 0.06
                ax.hlines(whisker_lo, positions[i] - cap_w, positions[i] + cap_w,
                          color='#333333', linewidth=0.8, zorder=3)
                ax.hlines(whisker_hi, positions[i] - cap_w, positions[i] + cap_w,
                          color='#333333', linewidth=0.8, zorder=3)

            # --- Annotate mean + std below model names ---
            # (placed inside bottom margin via transform)
            for i, (samples, label) in enumerate(zip(sorted_samples, sorted_labels)):
                mean_val = np.mean(samples)
                std_val = np.std(samples)
                if metric == 'psnr':
                    txt = f'\u03bc={mean_val:.2f}\n\u03c3={std_val:.2f}'
                elif metric in ('ssim', 'scc'):
                    txt = f'\u03bc={mean_val:.4f}\n\u03c3={std_val:.4f}'
                else:
                    txt = f'\u03bc={mean_val:.4f}\n\u03c3={std_val:.4f}'
                ax.annotate(txt, xy=(positions[i], 0), xycoords=('data', 'axes fraction'),
                            xytext=(0, -58), textcoords='offset points',
                            ha='center', va='top', fontsize=6.5, color='#333333',
                            fontfamily='monospace')

            ax.set_xticks(positions)
            ax.set_xticklabels(sorted_labels, rotation=50, ha='right', fontsize=9,
                               fontweight='medium')
            ax.set_title(f'{branch} bands', fontsize=15, fontweight='bold', pad=10)
            ax.set_ylabel(metric_labels[metric], fontsize=13)
            ax.grid(axis='y', alpha=0.25, linestyle='--', color='#888888')
            ax.grid(axis='x', alpha=0.08, linestyle='-', color='#CCCCCC')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.tick_params(axis='y', labelsize=10)

        # Legend
        legend_elements = [
            Patch(facecolor=COLORS['Residual']['face'], edgecolor=COLORS['Residual']['edge'],
                  alpha=0.75, label='Residual'),
            Patch(facecolor=COLORS['Direct']['face'], edgecolor=COLORS['Direct']['edge'],
                  alpha=0.75, label='Direct'),
            Patch(facecolor=COLORS['Bilinear']['face'], edgecolor=COLORS['Bilinear']['edge'],
                  alpha=0.75, label='Bilinear'),
            plt.Line2D([0], [0], color='white', marker='D', markerfacecolor='#222222',
                       markeredgecolor='white', markersize=6, label='Mean'),
            plt.Line2D([0], [0], color='white', linewidth=2, label='Median'),
        ]
        fig.legend(handles=legend_elements, loc='upper right',
                   bbox_to_anchor=(0.99, 0.95), fontsize=11,
                   framealpha=0.9, edgecolor='#CCCCCC')

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        save_path = output_dir / f'violin_{metric}.png'
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {save_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default=None,
                        help='Path to per-sample JSON (auto-detected from results dir if omitted)')
    parser.add_argument('--output', default='results/model_evaluation/visualizations_models_20260324')
    args = parser.parse_args()

    # Auto-detect latest per_sample JSON if not provided
    if args.json is None:
        results_dir = Path('results/model_evaluation')
        candidates = sorted(results_dir.glob('evaluation_*_per_sample.json'))
        if not candidates:
            raise FileNotFoundError("No per_sample JSON found in results/model_evaluation/")
        args.json = str(candidates[-1])
        print(f"Auto-detected: {args.json}")

    data = load_per_sample(args.json)
    make_violin_plots(data, args.output)
    print('Done.')
