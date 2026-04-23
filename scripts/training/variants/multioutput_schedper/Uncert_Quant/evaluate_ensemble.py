#!/usr/bin/env python3
"""
Heteroscedastic Deep Ensemble — Evaluation & Uncertainty Decomposition
======================================================================

Loads M trained heteroscedastic ensemble members and produces:

1. **Predictive mean** — ensemble-averaged super-resolved images
2. **Aleatoric uncertainty** — mean of the per-member predicted variances
3. **Epistemic uncertainty** — variance of the per-member predicted means
4. **Predictive (total) uncertainty** — aleatoric + epistemic

Outputs:
    • Per-sample CSV with PSNR, SSIM, RMSE + uncertainty statistics
    • Aggregated summary JSON
    • Optional: saved uncertainty maps as .npy for visualisation

Usage:
    python evaluate_ensemble.py \\
        --checkpoint_dirs ckpt/member0 ckpt/member1 ckpt/member2 \\
        --test_dir data/sr_training/all_stas_combined/mmap_test \\
        --output_dir results/het_ensemble

    python evaluate_ensemble.py \\
        --checkpoint_pattern "checkpoints/het_ensemble_direct_*_full" \\
        --test_dir data/sr_training/all_stas_combined/mmap_test
"""

import sys
import os
import argparse
import json
import glob
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# imports — the heteroscedastic model lives in the same Uncert_Quant folder
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = str(SCRIPT_DIR.parent)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_het_ensemble_direct_learnedUps import (
    HetMultiOutputTemporalSRDirectLearnedUps,
)


# =============================================================================
# DATASET  (test-time, no augmentation)
# =============================================================================

class TestMmapDataset(Dataset):
    """Memory-mapped test dataset — no augmentation."""

    def __init__(self, mmap_dir, skip_indices=None):
        mmap_dir = Path(mmap_dir)
        self.lr     = np.load(mmap_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(mmap_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr_10m = np.load(mmap_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(mmap_dir / 'hr_20m.npy', mmap_mode='r')

        self.n_total = self.lr.shape[0]
        skip = set(skip_indices or [])
        self.valid_indices = [i for i in range(self.n_total) if i not in skip]

        print(f"Test Dataset: {self.n_total} total, {len(skip)} skipped, "
              f"{len(self.valid_indices)} valid samples")
        print(f"  LR 10m: {self.lr.shape}, LR 20m: {self.lr_20m.shape}")
        print(f"  HR 10m: {self.hr_10m.shape}, HR 20m: {self.hr_20m.shape}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        lr_10m = torch.from_numpy(np.array(self.lr[real_idx])).float().permute(0, 3, 1, 2)
        lr_20m = torch.from_numpy(np.array(self.lr_20m[real_idx])).float().permute(0, 3, 1, 2)
        hr_10m = torch.from_numpy(np.array(self.hr_10m[real_idx])).float().permute(0, 3, 1, 2)
        hr_20m = torch.from_numpy(np.array(self.hr_20m[real_idx])).float().permute(0, 3, 1, 2)

        return {
            'lr_10m': lr_10m,
            'lr_20m': lr_20m,
            'hr_10m': hr_10m,
            'hr_20m': hr_20m,
            'index': real_idx,
        }


# =============================================================================
# METRICS
# =============================================================================

def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0  # practical inf
    return 10 * np.log10(max_val ** 2 / mse)


def compute_ssim_simple(pred, target, max_val=1.0):
    """Simple SSIM implementation (channel-averaged, 11×11 window)."""
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Use a uniform kernel for simplicity
    kernel_size = 11
    pad = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=pred.device)
    kernel = kernel / (kernel_size ** 2)

    ssim_vals = []
    for c in range(pred.shape[0]):  # per channel
        p = pred[c:c+1].unsqueeze(0)  # (1,1,H,W)
        t = target[c:c+1].unsqueeze(0)

        mu_p = F.conv2d(p, kernel, padding=pad)
        mu_t = F.conv2d(t, kernel, padding=pad)

        mu_p2 = mu_p ** 2
        mu_t2 = mu_t ** 2
        mu_pt = mu_p * mu_t

        sig_p2 = F.conv2d(p ** 2, kernel, padding=pad) - mu_p2
        sig_t2 = F.conv2d(t ** 2, kernel, padding=pad) - mu_t2
        sig_pt = F.conv2d(p * t, kernel, padding=pad) - mu_pt

        num = (2 * mu_pt + C1) * (2 * sig_pt + C2)
        den = (mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2)

        ssim_vals.append(torch.mean(num / den).item())

    return np.mean(ssim_vals)


def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


# =============================================================================
# ENSEMBLE LOADER
# =============================================================================

def load_ensemble_member(checkpoint_path, device):
    """Load a single heteroscedastic model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        ckpt_file = checkpoint_path / 'best_model.pt'
    else:
        ckpt_file = checkpoint_path

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    print(f"  Loading: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)

    cfg = ckpt.get('config', {})
    resize_method = ckpt.get('resize_method',
                              cfg.get('resize_method', 'adaptive'))

    model = HetMultiOutputTemporalSRDirectLearnedUps(
        base_channels=cfg.get('base_channels', 64),
        num_encoder_blocks=cfg.get('num_encoder_blocks', 4),
        num_frames=cfg.get('num_frames', 3),
        num_temporal_blocks=cfg.get('num_temporal_blocks', 2),
        scale_factor_10m=cfg.get('scale_factor_10m', 3.33),
        scale_factor_20m=cfg.get('scale_factor_20m', 3.33),
        in_channels_10m=cfg.get('in_channels_10m', 3),
        in_channels_20m=cfg.get('in_channels_20m', 2),
        resize_method=resize_method,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    member_info = {
        'seed': ckpt.get('seed', 'unknown'),
        'epoch': ckpt.get('epoch', -1),
        'val_psnr_10m': ckpt.get('val_psnr_10m', -1),
        'val_psnr_20m': ckpt.get('val_psnr_20m', -1),
        'resize_method': resize_method,
        'checkpoint': str(ckpt_file),
    }
    return model, member_info


def discover_checkpoints(pattern):
    """Glob for checkpoint directories matching a pattern."""
    dirs = sorted(glob.glob(pattern))
    valid = [d for d in dirs if Path(d, 'best_model.pt').exists()]
    return valid


# =============================================================================
# ENSEMBLE INFERENCE
# =============================================================================

@torch.no_grad()
def ensemble_predict(models, lr_10m, lr_20m, target_size_10m, target_size_20m):
    """
    Run all ensemble members and stack predictions.

    Returns:
        means_10m  : (M, B, T, C, H, W)  — mean predictions per member
        logvars_10m: (M, B, T, C, H, W)  — log-variance per member
        means_20m  : (M, B, T, C, H', W')
        logvars_20m: (M, B, T, C, H', W')
    """
    all_means_10m, all_logvars_10m = [], []
    all_means_20m, all_logvars_20m = [], []

    for model in models:
        m10, lv10, m20, lv20 = model(
            lr_10m, lr_20m,
            target_size_10m=target_size_10m,
            target_size_20m=target_size_20m,
        )
        all_means_10m.append(m10)
        all_logvars_10m.append(lv10)
        all_means_20m.append(m20)
        all_logvars_20m.append(lv20)

    return (
        torch.stack(all_means_10m),
        torch.stack(all_logvars_10m),
        torch.stack(all_means_20m),
        torch.stack(all_logvars_20m),
    )


def decompose_uncertainty(means, logvars):
    """
    Decompose predictive uncertainty for a single resolution.

    Args:
        means  : (M, B, T, C, H, W)
        logvars: (M, B, T, C, H, W)

    Returns:
        pred_mean   : (B, T, C, H, W) — ensemble mean
        aleatoric   : (B, T, C, H, W) — mean of predicted variances
        epistemic   : (B, T, C, H, W) — variance of predicted means
        predictive  : (B, T, C, H, W) — total = aleatoric + epistemic
    """
    # predictive mean
    pred_mean = means.mean(dim=0)                       # (B,T,C,H,W)

    # aleatoric: E_m[σ²_m]
    variances = torch.exp(logvars)                      # (M,B,T,C,H,W)
    aleatoric = variances.mean(dim=0)                   # (B,T,C,H,W)

    # epistemic: Var_m[μ_m]
    epistemic = means.var(dim=0, unbiased=True)         # (B,T,C,H,W)

    # total predictive uncertainty
    predictive = aleatoric + epistemic                  # (B,T,C,H,W)

    return pred_mean, aleatoric, epistemic, predictive


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_ensemble(
    checkpoint_dirs,
    test_dir,
    output_dir,
    skip_indices=None,
    batch_size=1,
    save_maps=False,
    max_samples=None,
):
    """Run full ensemble evaluation on test set."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load all ensemble members ----
    print(f"\nLoading {len(checkpoint_dirs)} ensemble members…")
    models = []
    member_infos = []
    for ckpt_dir in checkpoint_dirs:
        model, info = load_ensemble_member(ckpt_dir, device)
        models.append(model)
        member_infos.append(info)
    M = len(models)
    print(f"Loaded {M} members.\n")

    # ---- Test dataset ----
    dataset = TestMmapDataset(test_dir, skip_indices=skip_indices)
    if max_samples is not None:
        dataset.valid_indices = dataset.valid_indices[:max_samples]
        print(f"  (limited to {max_samples} samples)")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ---- Per-sample results ----
    results = []
    all_uncertainty_stats = {
        'aleatoric_10m': [], 'epistemic_10m': [], 'predictive_10m': [],
        'aleatoric_20m': [], 'epistemic_20m': [], 'predictive_20m': [],
    }

    if save_maps:
        maps_dir = output_dir / 'uncertainty_maps'
        maps_dir.mkdir(exist_ok=True)

    print(f"Evaluating {len(dataset)} samples with {M}-member ensemble…\n")
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        lr_10m = batch['lr_10m'].to(device)   # (B,T,C,H,W)
        lr_20m = batch['lr_20m'].to(device)
        hr_10m = batch['hr_10m'].to(device)
        hr_20m = batch['hr_20m'].to(device)
        indices = batch['index'].numpy()

        target_size_10m = hr_10m.shape[-2:]
        target_size_20m = hr_20m.shape[-2:]

        # ensemble forward pass
        means_10m, logvars_10m, means_20m, logvars_20m = ensemble_predict(
            models, lr_10m, lr_20m, target_size_10m, target_size_20m,
        )

        # decompose uncertainty
        pred_10m, alea_10m, epis_10m, total_10m = decompose_uncertainty(
            means_10m, logvars_10m,
        )
        pred_20m, alea_20m, epis_20m, total_20m = decompose_uncertainty(
            means_20m, logvars_20m,
        )

        # per-sample metrics
        B = lr_10m.shape[0]
        for b in range(B):
            sample_idx = int(indices[b])

            # use middle frame (T//2) for metrics, consistent with evaluation
            t_mid = pred_10m.shape[1] // 2

            p10 = pred_10m[b, t_mid]
            h10 = hr_10m[b, t_mid]
            p20 = pred_20m[b, t_mid]
            h20 = hr_20m[b, t_mid]

            psnr_10m = compute_psnr(p10, h10)
            psnr_20m = compute_psnr(p20, h20)
            ssim_10m = compute_ssim_simple(p10, h10)
            ssim_20m = compute_ssim_simple(p20, h20)
            rmse_10m = compute_rmse(p10, h10)
            rmse_20m = compute_rmse(p20, h20)

            # uncertainty summaries (spatial mean)
            a10 = alea_10m[b, t_mid].mean().item()
            e10 = epis_10m[b, t_mid].mean().item()
            t10 = total_10m[b, t_mid].mean().item()
            a20 = alea_20m[b, t_mid].mean().item()
            e20 = epis_20m[b, t_mid].mean().item()
            t20 = total_20m[b, t_mid].mean().item()

            results.append({
                'sample_idx': sample_idx,
                'psnr_10m': psnr_10m,
                'psnr_20m': psnr_20m,
                'ssim_10m': ssim_10m,
                'ssim_20m': ssim_20m,
                'rmse_10m': rmse_10m,
                'rmse_20m': rmse_20m,
                'aleatoric_10m': a10,
                'epistemic_10m': e10,
                'predictive_10m': t10,
                'aleatoric_20m': a20,
                'epistemic_20m': e20,
                'predictive_20m': t20,
            })

            all_uncertainty_stats['aleatoric_10m'].append(a10)
            all_uncertainty_stats['epistemic_10m'].append(e10)
            all_uncertainty_stats['predictive_10m'].append(t10)
            all_uncertainty_stats['aleatoric_20m'].append(a20)
            all_uncertainty_stats['epistemic_20m'].append(e20)
            all_uncertainty_stats['predictive_20m'].append(t20)

            # save spatial maps
            if save_maps:
                np.savez_compressed(
                    maps_dir / f'sample_{sample_idx:05d}.npz',
                    pred_10m=pred_10m[b].cpu().numpy(),
                    pred_20m=pred_20m[b].cpu().numpy(),
                    aleatoric_10m=alea_10m[b].cpu().numpy(),
                    epistemic_10m=epis_10m[b].cpu().numpy(),
                    aleatoric_20m=alea_20m[b].cpu().numpy(),
                    epistemic_20m=epis_20m[b].cpu().numpy(),
                )

        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            elapsed = time.time() - t0
            done = (batch_idx + 1) * batch_size
            print(f"  [{done}/{len(dataset)}] "
                  f"PSNR 10m: {psnr_10m:.2f} 20m: {psnr_20m:.2f} | "
                  f"alea_10m: {a10:.5f} epis_10m: {e10:.5f} | "
                  f"{elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone! {len(results)} samples in {elapsed:.1f}s")

    # ---- Aggregate statistics ----
    psnr_10m_all = [r['psnr_10m'] for r in results]
    psnr_20m_all = [r['psnr_20m'] for r in results]
    ssim_10m_all = [r['ssim_10m'] for r in results]
    ssim_20m_all = [r['ssim_20m'] for r in results]
    rmse_10m_all = [r['rmse_10m'] for r in results]
    rmse_20m_all = [r['rmse_20m'] for r in results]

    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_members': M,
        'n_samples': len(results),
        'member_info': member_infos,
        'metrics': {
            'psnr_10m': {
                'mean': float(np.mean(psnr_10m_all)),
                'std': float(np.std(psnr_10m_all)),
                'median': float(np.median(psnr_10m_all)),
            },
            'psnr_20m': {
                'mean': float(np.mean(psnr_20m_all)),
                'std': float(np.std(psnr_20m_all)),
                'median': float(np.median(psnr_20m_all)),
            },
            'ssim_10m': {
                'mean': float(np.mean(ssim_10m_all)),
                'std': float(np.std(ssim_10m_all)),
            },
            'ssim_20m': {
                'mean': float(np.mean(ssim_20m_all)),
                'std': float(np.std(ssim_20m_all)),
            },
            'rmse_10m': {
                'mean': float(np.mean(rmse_10m_all)),
                'std': float(np.std(rmse_10m_all)),
            },
            'rmse_20m': {
                'mean': float(np.mean(rmse_20m_all)),
                'std': float(np.std(rmse_20m_all)),
            },
        },
        'uncertainty': {},
    }

    for key in all_uncertainty_stats:
        vals = all_uncertainty_stats[key]
        summary['uncertainty'][key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'median': float(np.median(vals)),
            'q5': float(np.percentile(vals, 5)),
            'q95': float(np.percentile(vals, 95)),
        }

    # ---- Uncertainty quality: correlation with error ----
    # Good uncertainty should correlate with actual prediction error
    for res in ['10m', '20m']:
        errors = np.array([r[f'rmse_{res}'] for r in results])
        pred_unc = np.array([r[f'predictive_{res}'] for r in results])
        if len(errors) > 2:
            corr = float(np.corrcoef(errors, pred_unc)[0, 1])
            summary['uncertainty'][f'error_uncertainty_correlation_{res}'] = corr

    # ---- Save results ----
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # per-sample CSV
    csv_path = output_dir / f'ensemble_eval_{timestamp}.csv'
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"\nPer-sample CSV: {csv_path}")
    except ImportError:
        # fallback: write CSV manually
        with open(csv_path, 'w') as f:
            headers = results[0].keys()
            f.write(','.join(headers) + '\n')
            for r in results:
                f.write(','.join(str(r[h]) for h in headers) + '\n')
        print(f"\nPer-sample CSV: {csv_path}")

    # summary JSON
    json_path = output_dir / f'ensemble_summary_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {json_path}")

    # ---- Print summary ----
    print(f"\n{'=' * 70}")
    print(f"ENSEMBLE EVALUATION SUMMARY  ({M} members, {len(results)} samples)")
    print(f"{'=' * 70}")
    m = summary['metrics']
    print(f"  PSNR 10m: {m['psnr_10m']['mean']:.2f} ± {m['psnr_10m']['std']:.2f} dB")
    print(f"  PSNR 20m: {m['psnr_20m']['mean']:.2f} ± {m['psnr_20m']['std']:.2f} dB")
    print(f"  SSIM 10m: {m['ssim_10m']['mean']:.4f} ± {m['ssim_10m']['std']:.4f}")
    print(f"  SSIM 20m: {m['ssim_20m']['mean']:.4f} ± {m['ssim_20m']['std']:.4f}")
    print(f"  RMSE 10m: {m['rmse_10m']['mean']:.4f} ± {m['rmse_10m']['std']:.4f}")
    print(f"  RMSE 20m: {m['rmse_20m']['mean']:.4f} ± {m['rmse_20m']['std']:.4f}")

    u = summary['uncertainty']
    print(f"\n  --- Uncertainty (spatial mean) ---")
    for res in ['10m', '20m']:
        print(f"  [{res}]")
        print(f"    Aleatoric : {u[f'aleatoric_{res}']['mean']:.5f} "
              f"(median {u[f'aleatoric_{res}']['median']:.5f})")
        print(f"    Epistemic : {u[f'epistemic_{res}']['mean']:.5f} "
              f"(median {u[f'epistemic_{res}']['median']:.5f})")
        print(f"    Predictive: {u[f'predictive_{res}']['mean']:.5f} "
              f"(median {u[f'predictive_{res}']['median']:.5f})")
        corr_key = f'error_uncertainty_correlation_{res}'
        if corr_key in u:
            print(f"    Error↔Uncertainty corr: {u[corr_key]:.3f}")

    return summary


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate heteroscedastic deep ensemble with uncertainty decomposition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explicit checkpoint directories
  python evaluate_ensemble.py \\
      --checkpoint_dirs checkpoints/member0_full checkpoints/member1_full \\
      --test_dir data/sr_training/all_stas_combined/mmap_test

  # Auto-discover from pattern
  python evaluate_ensemble.py \\
      --checkpoint_pattern "checkpoints/het_ensemble_direct_learnedUps_adaptive_*_full" \\
      --test_dir data/sr_training/all_stas_combined/mmap_test

  # Save uncertainty maps for visualization
  python evaluate_ensemble.py \\
      --checkpoint_pattern "checkpoints/het_ensemble_*_full" \\
      --test_dir data/sr_training/all_stas_combined/mmap_test \\
      --save_maps
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint_dirs', nargs='+',
                       help='Paths to ensemble member checkpoint directories')
    group.add_argument('--checkpoint_pattern', type=str,
                       help='Glob pattern to discover checkpoint directories')

    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to mmap test directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results/het_ensemble/)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--skip_indices', nargs='*', type=int,
                        default=[1, 8, 223, 272, 1088, 1103, 1175],
                        help='Sample indices to skip (NaN samples)')
    parser.add_argument('--save_maps', action='store_true',
                        help='Save per-sample uncertainty maps as .npz')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit evaluation to N samples (for debugging)')
    args = parser.parse_args()

    # ---- Discover checkpoints ----
    if args.checkpoint_dirs:
        ckpt_dirs = args.checkpoint_dirs
    else:
        ckpt_dirs = discover_checkpoints(args.checkpoint_pattern)
        if not ckpt_dirs:
            print(f"ERROR: No checkpoints found matching: {args.checkpoint_pattern}")
            sys.exit(1)
        print(f"Discovered {len(ckpt_dirs)} checkpoint directories:")
        for d in ckpt_dirs:
            print(f"  {d}")

    # ---- Output directory ----
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Uncert_Quant -> multioutput_schedper -> variants -> training -> scripts -> SR
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        output_dir = PROJECT_ROOT / 'results' / 'het_ensemble'

    # ---- Run ----
    evaluate_ensemble(
        checkpoint_dirs=ckpt_dirs,
        test_dir=args.test_dir,
        output_dir=output_dir,
        skip_indices=args.skip_indices,
        batch_size=args.batch_size,
        save_maps=args.save_maps,
        max_samples=args.max_samples,
    )
