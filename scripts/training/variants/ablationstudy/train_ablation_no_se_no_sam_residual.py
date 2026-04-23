#!/usr/bin/env python3
"""
Ablation variant: remove SpectralConsistencyModule and disable SAM loss.

This file restores the ablation script that replaces the spectral modules
with identity mappings and sets the spectral loss weight to 0.0. It re-uses
the main model implementation from the learned-upscale training script and
provides a small smoke_test() helper that instantiates the ablated model
and runs a synthetic forward+loss to verify everything loads correctly.

This is intended as a drop-in ablation variant: to run full training, call
train_model from the original training script but pass a modified config
with 'alpha_spectral': 0.0 and a checkpoint folder/suffix indicating
'ablation_no_spectral'.
"""

import sys
import os
from pathlib import Path
import importlib.util

# Load the original training module directly from its file to avoid import path
# issues when running this file as a script. This ensures we get the model and
# training utilities without depending on package import resolution.
SCHEDPER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'multioutput_schedper')
)
# The training script does bare imports like `from satellite_sr_model_v2 import ...`
# so its parent directory must be on sys.path before we exec the module.
if SCHEDPER_DIR not in sys.path:
    sys.path.insert(0, SCHEDPER_DIR)

TRAIN_SCRIPT_PATH = os.path.join(SCHEDPER_DIR,
                                 'train_multires_temporal_multiOutput_schedPer_learnedUps.py')
spec = importlib.util.spec_from_file_location('train_learned_up', TRAIN_SCRIPT_PATH)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

# Bind the symbols we need from the training module
MultiOutputTemporalSRLearnedUps = train_mod.MultiOutputTemporalSRLearnedUps
ORIGINAL_BEST_CONFIG = train_mod.BEST_CONFIG
MultiOutputLoss = train_mod.MultiOutputLoss
count_parameters = train_mod.count_parameters

import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a working BEST_CONFIG for ablation (same as original but SAM disabled)
BEST_CONFIG = ORIGINAL_BEST_CONFIG.copy()
BEST_CONFIG['alpha_spectral'] = 0.0

# Helper to convert an instantiated model into the "no spectral" variant by
# replacing spectral modules with identity layers. This preserves all other
# weights and structure so training hyperparameters remain identical.
def make_no_spectral(model: nn.Module) -> nn.Module:
    # If attributes exist, replace them with identity
    if hasattr(model, 'spectral_10m'):
        model.spectral_10m = nn.Identity()
    if hasattr(model, 'spectral_20m'):
        model.spectral_20m = nn.Identity()
    return model


def smoke_test(device=None):
    """Quick smoke-test: instantiate ablated model + criterion and run a
    small synthetic forward pass to verify shapes and no import/runtime errors.
    Prints parameter counts and output shapes on success.
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    cfg = BEST_CONFIG.copy()
    # Keep config small for fast smoke test
    cfg['base_channels'] = 32
    cfg['num_encoder_blocks'] = 2
    cfg['num_frames'] = 3

    # Instantiate original model then patch spectral modules away
    model = MultiOutputTemporalSRLearnedUps(
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=cfg['num_frames'],
        resize_method=cfg.get('resize_method', 'adaptive'),
    ).to(device)

    model = make_no_spectral(model)

    n_params = count_parameters(model)
    print(f"Ablated model params: {n_params:,}")

    # Create criterion with SAM weight set to zero
    criterion = MultiOutputLoss(
        alpha_pixel=cfg.get('alpha_pixel', 1.0),
        alpha_spectral=0.0,
        alpha_gradient=cfg.get('alpha_gradient', 0.01),
        alpha_temporal=cfg.get('alpha_temporal', 0.5),
        alpha_perceptual=cfg.get('alpha_perceptual', 0.1),
        alpha_perceptual_start=cfg.get('alpha_perceptual_start', 0.01),
        perceptual_warmup_epochs=cfg.get('perceptual_warmup_epochs', 30),
    )

    # Synthetic batch: B=2, T=3, C10m=3, H=32, W=32 ; C20m=2, H/2, W/2
    B = 2
    T = cfg['num_frames']
    lr_10m = torch.rand(B, T, 3, 32, 32, device=device)
    lr_20m = torch.rand(B, T, 2, 16, 16, device=device)

    hr_10m = torch.rand(B, T, 3, 96, 96, device=device)  # target upscales ~3x in smoke test
    hr_20m = torch.rand(B, T, 2, 48, 48, device=device)

    with torch.no_grad():
        pred_10m, pred_20m = model(
            lr_10m, lr_20m,
            target_size_10m=hr_10m.shape[-2:],
            target_size_20m=hr_20m.shape[-2:],
        )

    print(f"pred_10m shape: {pred_10m.shape}")
    print(f"pred_20m shape: {pred_20m.shape}")

    # Compute a quick loss (should run even if SAM is identity since alpha_spectral=0)
    loss, loss_dict = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
    print(f"Smoke loss: {loss.item():.6f}")
    print("Smoke test completed successfully.")


def ablation_train_model(data_dir, checkpoint_dir, experiment_name, cfg,
                         resume=False):
    """Train the ablated model (no spectral modules / SAM loss).

    If *resume* is True and a ``latest_model.pt`` checkpoint exists in
    *checkpoint_dir*, the model, optimizer, scheduler, history, and epoch
    counter are all restored so training continues seamlessly.
    """
    import time, json as _json
    from torch.utils.data import DataLoader, random_split

    # Bind symbols from the original training module
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

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── model (with spectral modules replaced) ───────────────────────────
    model = MultiOutputTemporalSRLearnedUps(
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=cfg.get('num_frames', 3),
        num_temporal_blocks=cfg.get('num_temporal_blocks', 2),
        resize_method=cfg['resize_method'],
    ).to(device)
    model = make_no_spectral(model)
    print(f"Model: {count_parameters(model):,} params  (spectral → Identity)")

    # ── loss ─────────────────────────────────────────────────────────────
    criterion = MultiOutputLoss(
        alpha_pixel=cfg['alpha_pixel'],
        alpha_spectral=0.0,
        alpha_gradient=cfg['alpha_gradient'],
        alpha_temporal=cfg.get('alpha_temporal', 0.5),
        alpha_perceptual=cfg.get('alpha_perceptual', 0.1),
        alpha_perceptual_start=cfg.get('alpha_perceptual_start', 0.01),
        perceptual_warmup_epochs=cfg.get('perceptual_warmup_epochs', 30),
    )
    print(f"Loss weights: Pixel={cfg['alpha_pixel']:.3f}, SAM=0.000, "
          f"Grad={cfg['alpha_gradient']:.4f}, Temporal={cfg.get('alpha_temporal', 0.5):.3f}")
    print(f"Perceptual: START={cfg.get('alpha_perceptual_start', 0.01):.3f} → "
          f"FINAL={cfg.get('alpha_perceptual', 0.1):.3f} "
          f"(warmup over {cfg.get('perceptual_warmup_epochs', 30)} epochs)")

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
        start_epoch = ckpt['epoch'] + 1  # saved epoch is 0-indexed, resume from next

        # Restore scheduler state by replaying history
        hist_path = checkpoint_dir / 'training_history.json'
        if hist_path.exists():
            with open(hist_path) as _f:
                history = _json.load(_f)
            # Replay scheduler steps so LR is correct
            for vl in history['val_loss']:
                scheduler.step(vl)
            # Restore best val loss from history
            best_val_loss = min(history['val_loss'])
            best_idx = history['val_loss'].index(best_val_loss)
            best_psnr = (history['val_psnr_10m'][best_idx] +
                         history['val_psnr_20m'][best_idx]) / 2
            # Replay early stopping
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
        description='Ablation: Remove SpectralConsistencyModule and disable SAM loss',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test only
  python train_ablation_no_spectral.py --smoke

  # Full training on the clean dataset
  python train_ablation_no_spectral.py --data_dir data/sr_training/all_stas_combined/mmap_train_clean

  # Full training with custom epochs / batch size
  python train_ablation_no_spectral.py \\
      --data_dir data/sr_training/all_stas_combined/mmap_train_clean \\
      --epochs 200 --batch_size 2

  # Resume interrupted training
  python train_ablation_no_spectral.py \\
      --data_dir data/sr_training/all_stas_combined/mmap_train_clean \\
      --resume --batch_size 2
        """
    )
    parser.add_argument('--smoke', action='store_true', help='Run quick smoke-test and exit')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to mmap training data directory (required for training)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Where to save checkpoints (default: checkpoints/ablation_no_spectral)')
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

        # Build config from BEST_CONFIG with ablation overrides
        cfg = BEST_CONFIG.copy()
        cfg['alpha_spectral'] = 0.0          # <-- THE ABLATION
        cfg['epochs'] = args.epochs
        cfg['batch_size'] = args.batch_size
        cfg['resize_method'] = args.resize_method
        cfg['alpha_temporal'] = args.alpha_temporal
        cfg['alpha_perceptual'] = args.alpha_perceptual
        cfg['alpha_perceptual_start'] = args.alpha_perceptual_start
        cfg['perceptual_warmup_epochs'] = args.perceptual_warmup

        # Checkpoint dir
        if args.checkpoint_dir:
            ckpt_dir = Path(args.checkpoint_dir)
        else:
            ckpt_dir = PROJECT_ROOT / 'checkpoints' / f'ablation_no_spectral_{args.resize_method}'

        print("\n" + "="*70)
        print("  ABLATION STUDY: No Spectral Consistency Module / SAM Loss")
        print("="*70)
        print(f"  alpha_spectral = {cfg['alpha_spectral']}")
        print(f"  Spectral modules → nn.Identity()")
        print(f"  Data: {data_path}")
        print(f"  Checkpoints: {ckpt_dir}")
        print(f"  Epochs: {cfg['epochs']}, Batch: {cfg['batch_size']}")
        print(f"  Resize method: {cfg['resize_method']}")
        print("="*70 + "\n")

        val_loss, best_psnr = ablation_train_model(
            data_dir=data_path,
            checkpoint_dir=ckpt_dir,
            experiment_name='Ablation: No SpectralConsistency / SAM',
            cfg=cfg,
            resume=args.resume,
        )

        print("\n" + "="*70)
        print(f"ABLATION COMPLETE — Val Loss: {val_loss:.4f}, Best PSNR: {best_psnr:.2f} dB")
        print("="*70)
