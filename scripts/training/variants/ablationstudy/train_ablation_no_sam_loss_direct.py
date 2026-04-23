#!/usr/bin/env python3
"""
Ablation variant: disable Compound Spectral Loss (SAM) only — DIRECT prediction.

Base model: MultiOutputTemporalSRDirectLearnedUps (full direct model with adaptive interp.)
Ablation:   alpha_spectral = 0.0  (SAM loss disabled)
Kept:       SpectralConsistencyModule (SE-style, untouched in architecture)

This isolates the contribution of the SAM loss term to direct-prediction
performance, independent of the spectral consistency module.

Ablation matrix (Direct prediction):
  Full model (direct)         : SpectralConsistencyModule ✓  +  SAM Loss ✓
  No SAM Loss (this)          : SpectralConsistencyModule ✓  +  SAM Loss ✗
  No SE Module (separate)     : SpectralConsistencyModule ✗  +  SAM Loss ✓
  No SE + No SAM (separate)   : SpectralConsistencyModule ✗  +  SAM Loss ✗
"""

import sys
import os
from pathlib import Path
import importlib.util

# ─── Load the DIRECT training module ─────────────────────────────────────────
SCHEDPER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'multioutput_schedper')
)
if SCHEDPER_DIR not in sys.path:
    sys.path.insert(0, SCHEDPER_DIR)

TRAIN_SCRIPT_PATH = os.path.join(SCHEDPER_DIR,
                                 'train_multires_temporal_multiOutput_schedPer_direct_learnedUps.py')
spec = importlib.util.spec_from_file_location('train_direct_learned_up', TRAIN_SCRIPT_PATH)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

# Bind symbols from the DIRECT training module
MultiOutputTemporalSRDirectLearnedUps = train_mod.MultiOutputTemporalSRDirectLearnedUps
ORIGINAL_BEST_CONFIG = train_mod.BEST_CONFIG
MultiOutputLoss = train_mod.MultiOutputLoss
count_parameters = train_mod.count_parameters

import torch
import torch.nn as nn

# ─── Ablation config: same as original but SAM loss disabled ─────────────────
BEST_CONFIG = ORIGINAL_BEST_CONFIG.copy()
BEST_CONFIG['alpha_spectral'] = 0.0  # <-- THE ABLATION: disable SAM loss
# NOTE: We do NOT touch the model architecture — SpectralConsistencyModule stays


def smoke_test(device=None):
    """Quick smoke-test: instantiate direct model (with spectral module!) + criterion
    (with SAM disabled) and run a small synthetic forward+loss."""
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    cfg = BEST_CONFIG.copy()
    cfg['base_channels'] = 32
    cfg['num_encoder_blocks'] = 2
    cfg['num_frames'] = 3

    # Full direct model — SpectralConsistencyModule is kept
    model = MultiOutputTemporalSRDirectLearnedUps(
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=cfg['num_frames'],
        resize_method=cfg.get('resize_method', 'adaptive'),
    ).to(device)

    # Verify spectral modules are PRESENT
    has_spectral_10m = hasattr(model, 'spectral_10m') and not isinstance(model.spectral_10m, nn.Identity)
    has_spectral_20m = hasattr(model, 'spectral_20m') and not isinstance(model.spectral_20m, nn.Identity)
    print(f"spectral_10m present: {has_spectral_10m}")
    print(f"spectral_20m present: {has_spectral_20m}")

    n_params = count_parameters(model)
    print(f"Model params: {n_params:,}  (full DIRECT architecture, SpectralConsistencyModule KEPT)")

    # Criterion with SAM weight = 0
    criterion = MultiOutputLoss(
        alpha_pixel=cfg.get('alpha_pixel', 1.0),
        alpha_spectral=0.0,  # <-- ablation
        alpha_gradient=cfg.get('alpha_gradient', 0.01),
        alpha_temporal=cfg.get('alpha_temporal', 0.5),
        alpha_perceptual=cfg.get('alpha_perceptual', 0.1),
        alpha_perceptual_start=cfg.get('alpha_perceptual_start', 0.01),
        perceptual_warmup_epochs=cfg.get('perceptual_warmup_epochs', 30),
    )

    # Synthetic batch
    B, T = 2, cfg['num_frames']
    lr_10m = torch.rand(B, T, 3, 32, 32, device=device)
    lr_20m = torch.rand(B, T, 2, 16, 16, device=device)
    hr_10m = torch.rand(B, T, 3, 96, 96, device=device)
    hr_20m = torch.rand(B, T, 2, 48, 48, device=device)

    with torch.no_grad():
        pred_10m, pred_20m = model(
            lr_10m, lr_20m,
            target_size_10m=hr_10m.shape[-2:],
            target_size_20m=hr_20m.shape[-2:],
        )

    print(f"pred_10m: {pred_10m.shape}, pred_20m: {pred_20m.shape}")

    loss, loss_dict = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
    print(f"Smoke loss: {loss.item():.6f}")
    print("Smoke test PASSED — Direct No SAM Loss ablation is valid.")


def ablation_train_model(data_dir, checkpoint_dir, experiment_name, cfg,
                         resume=False):
    """Train the full DIRECT model but without SAM loss.

    Architecture is identical to the full direct model (SpectralConsistencyModule kept).
    Only the loss function changes: alpha_spectral = 0.0.
    """
    import time, json as _json
    from torch.utils.data import DataLoader, random_split

    _MultiOutputMmapDataset = train_mod.MultiOutputMmapDataset
    _EarlyStopping          = train_mod.EarlyStopping
    _compute_temporal_psnr  = train_mod.compute_temporal_psnr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── dataset ──────────────────────────────────────────────────────────
    dataset = _MultiOutputMmapDataset(
        mmap_dir=data_dir,
        sequence_length=cfg.get('num_frames', 3),
        augment=True,
    )
    n_val   = min(100, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    val_dataset.dataset.augment = False
    print(f"Train: {n_train}, Val: {n_val}")

    # ── model (FULL DIRECT architecture — spectral modules KEPT) ─────────
    model = MultiOutputTemporalSRDirectLearnedUps(
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=cfg.get('num_frames', 3),
        num_temporal_blocks=cfg.get('num_temporal_blocks', 2),
        resize_method=cfg['resize_method'],
    ).to(device)
    # NOTE: No make_no_spectral() call — model is unchanged
    print(f"Model: {count_parameters(model):,} params  (DIRECT, SpectralConsistencyModule KEPT)")

    # Verify spectral module presence
    assert hasattr(model, 'spectral_10m') and not isinstance(model.spectral_10m, nn.Identity), \
        "spectral_10m must be a SpectralConsistencyModule, not Identity!"
    assert hasattr(model, 'spectral_20m') and not isinstance(model.spectral_20m, nn.Identity), \
        "spectral_20m must be a SpectralConsistencyModule, not Identity!"

    # ── loss (SAM disabled) ──────────────────────────────────────────────
    criterion = MultiOutputLoss(
        alpha_pixel=cfg['alpha_pixel'],
        alpha_spectral=0.0,  # <-- THE ABLATION
        alpha_gradient=cfg['alpha_gradient'],
        alpha_temporal=cfg.get('alpha_temporal', 0.5),
        alpha_perceptual=cfg.get('alpha_perceptual', 0.1),
        alpha_perceptual_start=cfg.get('alpha_perceptual_start', 0.01),
        perceptual_warmup_epochs=cfg.get('perceptual_warmup_epochs', 30),
    )
    print(f"Loss weights: Pixel={cfg['alpha_pixel']:.3f}, SAM=0.000 (ABLATED), "
          f"Grad={cfg['alpha_gradient']:.4f}, Temporal={cfg.get('alpha_temporal', 0.5):.3f}")

    # ── optimizer / scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=cfg['patience_scheduler'],
    )
    early_stopping = _EarlyStopping(patience=cfg['patience'])

    # ── resume from checkpoint ───────────────────────────────────────────
    start_epoch   = 0
    history       = {'train_loss': [], 'val_loss': [],
                     'val_psnr_10m': [], 'val_psnr_20m': [], 'lr': []}
    best_val_loss = float('inf')
    best_psnr     = 0.0

    latest_ckpt = checkpoint_dir / 'latest_model.pt'
    if resume and latest_ckpt.exists():
        print(f"\n► Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1

        hist_path = checkpoint_dir / 'training_history.json'
        if hist_path.exists():
            with open(hist_path) as _f:
                history = _json.load(_f)
            for vl in history['val_loss']:
                scheduler.step(vl)
            best_val_loss = min(history['val_loss'])
            best_idx = history['val_loss'].index(best_val_loss)
            best_psnr = (history['val_psnr_10m'][best_idx] +
                         history['val_psnr_20m'][best_idx]) / 2
            for vl_i, vl in enumerate(history['val_loss']):
                early_stopping(vl, vl_i)

        print(f"  Restored epoch {ckpt['epoch']+1}, resuming from epoch {start_epoch+1}")
        print(f"  Best val_loss so far: {best_val_loss:.4f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")
    elif resume:
        print("  No checkpoint found — starting fresh.")

    # ── training loop ────────────────────────────────────────────────────
    total_epochs = cfg['epochs']
    print(f"\nTraining epochs {start_epoch+1} → {total_epochs} ...")

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True)

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        current_perceptual_weight = criterion.update_perceptual_weight(epoch)

        # ── train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            lr_10m = batch['lr_10m'].to(device)
            lr_20m = batch['lr_20m'].to(device)
            hr_10m = batch['hr_10m'].to(device)
            hr_20m = batch['hr_20m'].to(device)

            optimizer.zero_grad()
            pred_10m, pred_20m = model(
                lr_10m, lr_20m,
                target_size_10m=hr_10m.shape[-2:],
                target_size_20m=hr_20m.shape[-2:],
            )
            loss, loss_dict = criterion(
                pred_10m, pred_20m, hr_10m, hr_20m,
                weight_10m=cfg['weight_10m'],
                weight_20m=cfg['weight_20m'],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss = val_psnr_10m = val_psnr_20m = val_temporal_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                lr_10m = batch['lr_10m'].to(device)
                lr_20m = batch['lr_20m'].to(device)
                hr_10m = batch['hr_10m'].to(device)
                hr_20m = batch['hr_20m'].to(device)

                pred_10m, pred_20m = model(
                    lr_10m, lr_20m,
                    target_size_10m=hr_10m.shape[-2:],
                    target_size_20m=hr_20m.shape[-2:],
                )
                loss, loss_dict = criterion(
                    pred_10m, pred_20m, hr_10m, hr_20m,
                    weight_10m=cfg['weight_10m'],
                    weight_20m=cfg['weight_20m'],
                )
                val_loss += loss.item()
                val_temporal_loss += (loss_dict['temporal_10m'] +
                                     loss_dict['temporal_20m']) / 2
                val_psnr_10m += _compute_temporal_psnr(pred_10m, hr_10m).item()
                val_psnr_20m += _compute_temporal_psnr(pred_20m, hr_20m).item()

        val_loss         /= len(val_loader)
        val_psnr_10m     /= len(val_loader)
        val_psnr_20m     /= len(val_loader)
        val_temporal_loss /= len(val_loader)
        combined_psnr = (val_psnr_10m + val_psnr_20m) / 2

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr_10m'].append(val_psnr_10m)
        history['val_psnr_20m'].append(val_psnr_20m)
        history['lr'].append(current_lr)

        # ── checkpointing ────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_psnr = combined_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_psnr_10m': val_psnr_10m,
                'val_psnr_20m': val_psnr_20m,
                'config': cfg,
            }, checkpoint_dir / 'best_model.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': cfg,
        }, checkpoint_dir / 'latest_model.pt')

        with open(checkpoint_dir / 'training_history.json', 'w') as _f:
            _json.dump(history, _f, indent=2)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{total_epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PSNR 10m: {val_psnr_10m:.2f} | PSNR 20m: {val_psnr_20m:.2f} | "
              f"Temp: {val_temporal_loss:.4f} | "
              f"α_p: {current_perceptual_weight:.4f} | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best epoch: {early_stopping.best_epoch+1} "
                  f"with val_loss: {best_val_loss:.4f}")
            break

    print(f"\nTraining complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}, Best PSNR: {best_psnr:.2f} dB")
    return best_val_loss, best_psnr


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Ablation: Disable SAM Loss — DIRECT prediction, keep SpectralConsistencyModule',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation study (DIRECT prediction): removes ONLY the SAM loss term (alpha_spectral=0.0).
The SpectralConsistencyModule remains active in the architecture.
Base model: MultiOutputTemporalSRDirectLearnedUps (full direct with adaptive interp.)

Examples:
  python train_ablation_no_sam_loss_direct.py --smoke
  python train_ablation_no_sam_loss_direct.py --data_dir data/sr_training/all_stas_combined/mmap_train_clean --epochs 200 --batch_size 2
  python train_ablation_no_sam_loss_direct.py --data_dir data/sr_training/all_stas_combined/mmap_train_clean --resume --batch_size 2
        """
    )
    parser.add_argument('--smoke', action='store_true', help='Run quick smoke-test and exit')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to mmap training data directory (required for training)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Where to save checkpoints')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--resize_method', choices=['meta', 'adaptive', 'bilinear'],
                        default='adaptive')
    parser.add_argument('--alpha_temporal', type=float, default=0.5)
    parser.add_argument('--alpha_perceptual', type=float, default=0.1)
    parser.add_argument('--alpha_perceptual_start', type=float, default=0.01)
    parser.add_argument('--perceptual_warmup', type=int, default=30)
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    else:
        if args.data_dir is None:
            parser.error('--data_dir is required for training (use --smoke for quick test)')

        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
        data_path = Path(args.data_dir)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path

        if not data_path.exists():
            parser.error(f'Data directory does not exist: {data_path}')

        cfg = BEST_CONFIG.copy()
        cfg['alpha_spectral'] = 0.0          # <-- THE ABLATION
        cfg['epochs'] = args.epochs
        cfg['batch_size'] = args.batch_size
        cfg['resize_method'] = args.resize_method
        cfg['alpha_temporal'] = args.alpha_temporal
        cfg['alpha_perceptual'] = args.alpha_perceptual
        cfg['alpha_perceptual_start'] = args.alpha_perceptual_start
        cfg['perceptual_warmup_epochs'] = args.perceptual_warmup

        if args.checkpoint_dir:
            ckpt_dir = Path(args.checkpoint_dir)
        else:
            ckpt_dir = PROJECT_ROOT / 'checkpoints' / f'ablation_no_sam_loss_direct_{args.resize_method}'

        print("\n" + "="*70)
        print("  ABLATION STUDY: No SAM Loss — DIRECT Prediction")
        print("  SpectralConsistencyModule: KEPT")
        print("  SAM Loss (alpha_spectral): DISABLED (0.0)")
        print("  Prediction mode: DIRECT (no bilinear baseline)")
        print("="*70)
        print(f"  Data: {data_path}")
        print(f"  Checkpoints: {ckpt_dir}")
        print(f"  Epochs: {cfg['epochs']}, Batch: {cfg['batch_size']}")
        print(f"  Resize method: {cfg['resize_method']}")
        print("="*70 + "\n")

        val_loss, best_psnr = ablation_train_model(
            data_dir=data_path,
            checkpoint_dir=ckpt_dir,
            experiment_name='Ablation: No SAM Loss — Direct (Spectral Module Kept)',
            cfg=cfg,
            resume=args.resume,
        )

        print("\n" + "="*70)
        print(f"ABLATION COMPLETE — Val Loss: {val_loss:.4f}, Best PSNR: {best_psnr:.2f} dB")
        print("="*70)
