#!/usr/bin/env python3
"""
Ablation variant: remove Temporal Cross-Attention AND Cross-Resolution Fusion
                  — DIRECT prediction.

Base model: MultiOutputTemporalSRDirectLearnedUps (full direct model with adaptive interp.)
Ablation:   (a) TemporalFusionBlock → TemporalIdentity  (no cross-frame interaction)
            (b) Cross-resolution fusion → skip  (each branch processes independently)
Kept:       Encoder, SE module, SAM loss, Decoder, Learned Upscaler, Output Heads

These two components are ablated TOGETHER because they are tightly coupled:
cross-resolution fusion relies on temporally-refined features, and temporal
attention is designed to improve coherence before fusion.  Removing them
jointly isolates the contribution of "multi-frame + multi-resolution
information sharing" to the reconstruction quality.

Architecture comparison:
  Full model:   Encode → TemporalCrossAttn → CrossResolutionFusion → SE → Decode → Up
  This ablation: Encode → (identity)        → (skip)               → SE → Decode → Up
"""

import sys
import os
from pathlib import Path
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ─── Ablation config: unchanged ─────────────────────────────────────────────
BEST_CONFIG = ORIGINAL_BEST_CONFIG.copy()


# =============================================================================
# TEMPORAL IDENTITY — replaces TemporalFusionBlock
# =============================================================================

class TemporalIdentity(nn.Module):
    """Pass-through module that preserves (B,T,C,H,W) shape with NO cross-frame
    interaction.  Replaces TemporalFusionBlock to ablate temporal cross-attention.

    Includes a lightweight per-frame refinement conv so that parameter count
    differences reflect ONLY the removal of cross-attention, not the removal
    of all learnable processing at this stage.
    """

    def __init__(self, channels, num_frames=3):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """x: (B, T, C, H, W) → (B, T, C, H, W) — each frame refined independently."""
        B, T, C, H, W = x.shape
        outputs = []
        for t in range(T):
            frame = x[:, t]
            refined = self.refine(frame)
            outputs.append(frame + self.gamma * refined)
        return torch.stack(outputs, dim=1)


# =============================================================================
# ABLATED MODEL: No Temporal CrossAttn + No Cross-Resolution Fusion (DIRECT)
# =============================================================================

class AblatedNoTemporalNoFusionDirect(MultiOutputTemporalSRDirectLearnedUps):
    """Direct model with temporal cross-attention and cross-resolution fusion
    REMOVED.  Inherits full model and overrides __init__ (replace temporal
    blocks) and forward (skip fusion step)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        base_channels = kwargs.get('base_channels', 64)
        num_frames    = kwargs.get('num_frames', 3)
        num_temporal  = kwargs.get('num_temporal_blocks', 2)

        # ── Replace temporal blocks with identity (no cross-frame) ───────
        self.temporal_blocks_10m = nn.ModuleList([
            TemporalIdentity(base_channels, num_frames)
            for _ in range(num_temporal)
        ])
        self.temporal_blocks_20m = nn.ModuleList([
            TemporalIdentity(base_channels, num_frames)
            for _ in range(num_temporal)
        ])

        # ── Zero-init fusion gate as safety (not used in forward) ────────
        with torch.no_grad():
            for p in self.fusion_gate.parameters():
                nn.init.zeros_(p)
            for p in self.fusion_10m_from_20m.parameters():
                nn.init.zeros_(p)
            for p in self.fusion_20m_from_10m.parameters():
                nn.init.zeros_(p)

    def forward(
        self,
        lr_10m: torch.Tensor,
        lr_20m: torch.Tensor,
        target_size_10m: tuple = None,
        target_size_20m: tuple = None,
    ) -> tuple:
        """Forward WITHOUT temporal cross-attention and WITHOUT cross-resolution
        fusion.  Each resolution branch processes completely independently."""

        B, T, C_10m, H_10m, W_10m = lr_10m.shape
        _, _, C_20m, H_20m, W_20m = lr_20m.shape

        # Compute target sizes
        if target_size_10m is not None:
            tgt_size_10m = target_size_10m
        else:
            tgt_size_10m = (int(H_10m * self.scale_factor_10m),
                            int(W_10m * self.scale_factor_10m))

        if target_size_20m is not None:
            tgt_size_20m = target_size_20m
        else:
            tgt_size_20m = (int(H_20m * self.scale_factor_20m),
                            int(W_20m * self.scale_factor_20m))

        # ========== ENCODE ALL FRAMES ==========
        features_10m = []
        features_20m = []
        for t in range(T):
            features_10m.append(self.encoder_10m(lr_10m[:, t]))
            features_20m.append(self.encoder_20m(lr_20m[:, t]))

        features_10m = torch.stack(features_10m, dim=1)
        features_20m = torch.stack(features_20m, dim=1)

        # ========== TEMPORAL (IDENTITY — no cross-frame interaction) ==========
        for block in self.temporal_blocks_10m:
            features_10m = block(features_10m)

        for block in self.temporal_blocks_20m:
            features_20m = block(features_20m)

        # ========== CROSS-RESOLUTION FUSION: **SKIPPED** ==========
        # Features pass directly without any inter-resolution information exchange.

        # ========== DECODE AND UPSCALE WITH LEARNED RESIZE ==========
        outputs_10m = []
        outputs_20m = []

        for t in range(T):
            # Spectral consistency (KEPT)
            feat_10m = self.spectral_10m(features_10m[:, t])
            feat_20m = self.spectral_20m(features_20m[:, t])

            # Decode (KEPT)
            feat_10m = self.decoder_10m(feat_10m)
            feat_20m = self.decoder_20m(feat_20m)

            # Upscale with LEARNED resize (KEPT) — DIRECT: output directly
            sr_10m = self.upscaler_10m(feat_10m, tgt_size_10m)
            sr_20m = self.upscaler_20m(feat_20m, tgt_size_20m)

            # Refine (KEPT)
            sr_10m = self.residual_refine_10m(sr_10m)
            sr_20m = self.residual_refine_20m(sr_20m)

            # Output heads (KEPT)
            sr_10m = self.output_head_10m(sr_10m)
            sr_20m = self.output_head_20m(sr_20m)

            outputs_10m.append(sr_10m)
            outputs_20m.append(sr_20m)

        sr_10m = torch.stack(outputs_10m, dim=1)
        sr_20m = torch.stack(outputs_20m, dim=1)

        return sr_10m, sr_20m


# =============================================================================
# SMOKE TEST & TRAINING
# =============================================================================

def smoke_test(device=None):
    """Quick smoke-test: ablated model forward + loss."""
    device = device or (torch.device('cuda') if torch.cuda.is_available()
                        else torch.device('cpu'))

    cfg = BEST_CONFIG.copy()
    cfg['base_channels'] = 32
    cfg['num_encoder_blocks'] = 2
    cfg['num_frames'] = 3

    model = AblatedNoTemporalNoFusionDirect(
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=cfg['num_frames'],
        resize_method=cfg.get('resize_method', 'adaptive'),
    ).to(device)

    # Verify temporal blocks are TemporalIdentity
    for blk in model.temporal_blocks_10m:
        assert isinstance(blk, TemporalIdentity), \
            f"Expected TemporalIdentity, got {type(blk).__name__}"
    for blk in model.temporal_blocks_20m:
        assert isinstance(blk, TemporalIdentity), \
            f"Expected TemporalIdentity, got {type(blk).__name__}"

    n_params = count_parameters(model)
    print(f"Model params: {n_params:,}  (no temporal cross-attn, no cross-res fusion)")

    criterion = MultiOutputLoss(
        alpha_pixel=cfg.get('alpha_pixel', 1.0),
        alpha_spectral=cfg.get('alpha_spectral', 0.137),
        alpha_gradient=cfg.get('alpha_gradient', 0.01),
        alpha_temporal=cfg.get('alpha_temporal', 0.5),
        alpha_perceptual=cfg.get('alpha_perceptual', 0.1),
        alpha_perceptual_start=cfg.get('alpha_perceptual_start', 0.01),
        perceptual_warmup_epochs=cfg.get('perceptual_warmup_epochs', 30),
    )

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
    print("Smoke test PASSED — Direct No-Temporal-No-Fusion ablation is valid.")


def ablation_train_model(data_dir, checkpoint_dir, experiment_name, cfg,
                         resume=False):
    """Train the direct model WITHOUT temporal cross-attention and WITHOUT
    cross-resolution fusion."""
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

    # ── model ────────────────────────────────────────────────────────────
    model = AblatedNoTemporalNoFusionDirect(
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=cfg.get('num_frames', 3),
        num_temporal_blocks=cfg.get('num_temporal_blocks', 2),
        resize_method=cfg['resize_method'],
    ).to(device)
    print(f"Model: {count_parameters(model):,} params  "
          "(temporal cross-attn REMOVED, cross-res fusion REMOVED)")

    # ── loss ─────────────────────────────────────────────────────────────
    criterion = MultiOutputLoss(
        alpha_pixel=cfg['alpha_pixel'],
        alpha_spectral=cfg['alpha_spectral'],
        alpha_gradient=cfg['alpha_gradient'],
        alpha_temporal=cfg.get('alpha_temporal', 0.5),
        alpha_perceptual=cfg.get('alpha_perceptual', 0.1),
        alpha_perceptual_start=cfg.get('alpha_perceptual_start', 0.01),
        perceptual_warmup_epochs=cfg.get('perceptual_warmup_epochs', 30),
    )
    print(f"Loss weights: Pixel={cfg['alpha_pixel']:.3f}, SAM={cfg['alpha_spectral']:.3f}, "
          f"Grad={cfg['alpha_gradient']:.4f}, Temporal={cfg.get('alpha_temporal', 0.5):.3f}")

    # ── optimizer / scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=cfg['patience_scheduler'],
    )
    early_stopping = _EarlyStopping(patience=cfg['patience'])

    # ── resume ───────────────────────────────────────────────────────────
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
        description='Ablation: Remove Temporal CrossAttn + Cross-Resolution Fusion — DIRECT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation study (DIRECT prediction): removes both Temporal Cross-Attention
(TemporalFusionBlock → TemporalIdentity) and Cross-Resolution Fusion
(gated feature exchange skipped). All other components unchanged.

Examples:
  python train_ablation_no_temporal_no_fusion_direct.py --smoke
  python train_ablation_no_temporal_no_fusion_direct.py --data_dir data/sr_training/all_stas_combined/mmap_train_clean --epochs 200 --batch_size 2
        """
    )
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
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
            ckpt_dir = PROJECT_ROOT / 'checkpoints' / f'ablation_no_temporal_no_fusion_direct_{args.resize_method}'

        print("\n" + "="*70)
        print("  ABLATION STUDY: No Temporal CrossAttn + No Cross-Res Fusion — DIRECT")
        print("  Temporal Cross-Attention: REMOVED (→ per-frame TemporalIdentity)")
        print("  Cross-Resolution Fusion: REMOVED (→ independent branches)")
        print("  Prediction mode: DIRECT (no residual learning)")
        print("  Everything else: KEPT (Encoder, SE, SAM, Decoder, Adaptive Upscaler)")
        print("="*70)
        print(f"  Data: {data_path}")
        print(f"  Checkpoints: {ckpt_dir}")
        print(f"  Epochs: {cfg['epochs']}, Batch: {cfg['batch_size']}")
        print(f"  Resize method: {cfg['resize_method']}")
        print("="*70 + "\n")

        val_loss, best_psnr = ablation_train_model(
            data_dir=data_path,
            checkpoint_dir=ckpt_dir,
            experiment_name='Ablation: No Temporal + No Fusion — Direct',
            cfg=cfg,
            resume=args.resume,
        )

        print("\n" + "="*70)
        print(f"ABLATION COMPLETE — Val Loss: {val_loss:.4f}, Best PSNR: {best_psnr:.2f} dB")
        print("="*70)
