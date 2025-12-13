#!/usr/bin/env python3
"""
Hyperparameter Tuning for MultiRes Model with Perceptual Loss.

Uses Optuna to optimize:
- Model architecture: base_channels, num_encoder_blocks
- Learning rate, weight decay
- Loss weights: alpha_pixel, alpha_spectral, alpha_gradient, alpha_perceptual
- Branch weights: weight_10m, weight_20m

The winner model (MultiRes Full Dataset) achieved:
- Test PSNR: 36.73 dB (+2.02 dB over bilinear)
- Test SSIM: 0.8935
- Best model so far

This tuning aims to find even better hyperparameters with the perceptual loss.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import optuna
from optuna.trial import TrialState
import argparse
import time
from pathlib import Path
import numpy as np
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Import the multires model
from models.satellite_sr_model_v2 import MultiResolutionSatelliteSRv2 as MultiResolutionSatelliteSR

# Current best configuration (baseline to beat)
CURRENT_BEST = {
    'base_channels': 64,
    'num_encoder_blocks': 4,
    'optimizer': 'adamw',
    'lr': 3.064e-05,
    'weight_decay': 5.235e-06,
    'patience_scheduler': 4,
    'alpha_pixel': 1.0,
    'alpha_spectral': 0.137,
    'alpha_gradient': 0.01,
    'alpha_perceptual': 0.1,
    'weight_10m': 1.0,
    'weight_20m': 1.0,
    'batch_size': 4,
}


class FiveBandMmapDataset(Dataset):
    """Memory-mapped dataset for 5-band SR training."""
    
    def __init__(self, mmap_dir, sequence_length=3, augment=True):
        mmap_dir = Path(mmap_dir)
        self.lr = np.load(mmap_dir / 'lr.npy', mmap_mode='r')
        self.lr_20m = np.load(mmap_dir / 'lr_20m.npy', mmap_mode='r')
        self.hr_10m = np.load(mmap_dir / 'hr_10m.npy', mmap_mode='r')
        self.hr_20m = np.load(mmap_dir / 'hr_20m.npy', mmap_mode='r')
        self.n_samples = self.lr.shape[0]
        self.sequence_length = sequence_length
        self.augment = augment
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        lr_10m = self.lr[idx].copy()
        lr_20m = self.lr_20m[idx].copy()
        hr_10m = self.hr_10m[idx].copy()
        hr_20m = self.hr_20m[idx].copy()
        
        if self.augment:
            import random
            if random.random() > 0.5:
                lr_10m = np.flip(lr_10m, axis=2).copy()
                lr_20m = np.flip(lr_20m, axis=2).copy()
                hr_10m = np.flip(hr_10m, axis=2).copy()
                hr_20m = np.flip(hr_20m, axis=2).copy()
            if random.random() > 0.5:
                lr_10m = np.flip(lr_10m, axis=1).copy()
                lr_20m = np.flip(lr_20m, axis=1).copy()
                hr_10m = np.flip(hr_10m, axis=1).copy()
                hr_20m = np.flip(hr_20m, axis=1).copy()
            k = random.randint(0, 3)
            if k > 0:
                lr_10m = np.rot90(lr_10m, k, axes=(1, 2)).copy()
                lr_20m = np.rot90(lr_20m, k, axes=(1, 2)).copy()
                hr_10m = np.rot90(hr_10m, k, axes=(1, 2)).copy()
                hr_20m = np.rot90(hr_20m, k, axes=(1, 2)).copy()
        
        lr_10m = torch.from_numpy(lr_10m).permute(0, 3, 1, 2).float()
        lr_20m = torch.from_numpy(lr_20m).permute(0, 3, 1, 2).float()
        mid_idx = self.sequence_length // 2
        hr_10m = torch.from_numpy(hr_10m[mid_idx]).permute(2, 0, 1).float()
        hr_20m = torch.from_numpy(hr_20m[mid_idx]).permute(2, 0, 1).float()
        
        return {'lr_10m': lr_10m, 'lr_20m': lr_20m, 'hr_10m': hr_10m, 'hr_20m': hr_20m}


class MultiResSpectralLossWithPerceptual(nn.Module):
    """Combined loss with perceptual term for multires model."""
    
    def __init__(self, cfg):
        super().__init__()
        self.alpha_pixel = cfg['alpha_pixel']
        self.alpha_spectral = cfg['alpha_spectral']
        self.alpha_gradient = cfg['alpha_gradient']
        self.alpha_perceptual = cfg['alpha_perceptual']
        self.weight_10m = cfg['weight_10m']
        self.weight_20m = cfg['weight_20m']
        
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
        # VGG Perceptual loss
        from losses.perceptual_loss import VGGPerceptualLoss
        self.perceptual_loss = VGGPerceptualLoss()
        
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m):
        # 10m branch: pixel + SAM + gradient + perceptual
        pixel_10m = self.l1(pred_10m, target_10m) + 0.1 * self.mse(pred_10m, target_10m)
        sam_10m = self._sam_loss(pred_10m, target_10m)
        grad_10m = self._grad_loss(pred_10m, target_10m)
        perc_10m = self.perceptual_loss(pred_10m, target_10m)
        
        loss_10m = (self.alpha_pixel * pixel_10m + 
                    self.alpha_spectral * sam_10m + 
                    self.alpha_gradient * grad_10m +
                    self.alpha_perceptual * perc_10m)
        
        # 20m branch: pixel + SAM + gradient (no perceptual - only 2 bands)
        pixel_20m = self.l1(pred_20m, target_20m) + 0.1 * self.mse(pred_20m, target_20m)
        sam_20m = self._sam_loss(pred_20m, target_20m)
        grad_20m = self._grad_loss(pred_20m, target_20m)
        
        loss_20m = (self.alpha_pixel * pixel_20m + 
                    self.alpha_spectral * sam_20m + 
                    self.alpha_gradient * grad_20m)
        
        total = self.weight_10m * loss_10m + self.weight_20m * loss_20m
        return total, loss_10m, loss_20m
    
    def _sam_loss(self, pred, target):
        pred_norm = nn.functional.normalize(pred, dim=1, eps=1e-8)
        target_norm = nn.functional.normalize(target, dim=1, eps=1e-8)
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        return 1.0 - cos_sim.mean()
    
    def _grad_loss(self, pred, target):
        loss = 0
        for c in range(pred.shape[1]):
            pred_gx = nn.functional.conv2d(pred[:, c:c+1], self.sobel_x, padding=1)
            pred_gy = nn.functional.conv2d(pred[:, c:c+1], self.sobel_y, padding=1)
            target_gx = nn.functional.conv2d(target[:, c:c+1], self.sobel_x, padding=1)
            target_gy = nn.functional.conv2d(target[:, c:c+1], self.sobel_y, padding=1)
            loss += self.l1(pred_gx, target_gx) + self.l1(pred_gy, target_gy)
        return loss / pred.shape[1]


def calculate_psnr(pred, target):
    """Calculate PSNR between prediction and target."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)).item()


def train_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    """Train for one epoch with optional batch limit for faster tuning."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in loader:
        lr_10m = batch['lr_10m'].to(device)
        lr_20m = batch['lr_20m'].to(device)
        hr_10m = batch['hr_10m'].to(device)
        hr_20m = batch['hr_20m'].to(device)
        
        optimizer.zero_grad()
        pred_10m, pred_20m = model(lr_10m, lr_20m, 
                                    target_size_10m=(hr_10m.shape[2], hr_10m.shape[3]),
                                    target_size_20m=(hr_20m.shape[2], hr_20m.shape[3]))
        loss, _, _ = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        
        if max_batches and n_batches >= max_batches:
            break
    
    return total_loss / n_batches


def validate(model, loader, criterion, device):
    """Validate model and return loss + PSNR metrics."""
    model.eval()
    total_loss = 0
    psnr_10m_sum, psnr_20m_sum = 0, 0
    
    with torch.no_grad():
        for batch in loader:
            lr_10m = batch['lr_10m'].to(device)
            lr_20m = batch['lr_20m'].to(device)
            hr_10m = batch['hr_10m'].to(device)
            hr_20m = batch['hr_20m'].to(device)
            
            pred_10m, pred_20m = model(lr_10m, lr_20m,
                                        target_size_10m=(hr_10m.shape[2], hr_10m.shape[3]),
                                        target_size_20m=(hr_20m.shape[2], hr_20m.shape[3]))
            loss, _, _ = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
            total_loss += loss.item()
            psnr_10m_sum += calculate_psnr(pred_10m, hr_10m)
            psnr_20m_sum += calculate_psnr(pred_20m, hr_20m)
    
    n = len(loader)
    return total_loss / n, psnr_10m_sum / n, psnr_20m_sum / n


def create_trial_config(trial):
    """Create hyperparameter configuration from Optuna trial."""
    
    cfg = {
        # Model architecture - explore around current best
        'base_channels': trial.suggest_categorical('base_channels', [48, 64, 80, 96]),
        'num_encoder_blocks': trial.suggest_int('num_encoder_blocks', 3, 6),
        
        # Optimizer
        'optimizer': 'adamw',
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True),
        
        # Scheduler
        'patience_scheduler': trial.suggest_int('patience_scheduler', 3, 7),
        
        # Loss weights - explore wider range
        'alpha_pixel': 1.0,  # Keep fixed as reference
        'alpha_spectral': trial.suggest_float('alpha_spectral', 0.05, 0.5, log=True),
        'alpha_gradient': trial.suggest_float('alpha_gradient', 0.001, 0.1, log=True),
        'alpha_perceptual': trial.suggest_float('alpha_perceptual', 0.01, 0.5, log=True),
        
        # Branch weights
        'weight_10m': 1.0,  # Keep 10m fixed
        'weight_20m': trial.suggest_float('weight_20m', 0.5, 2.0),
        
        # Training
        'batch_size': trial.suggest_categorical('batch_size', [2, 4, 8]),
    }
    
    return cfg


def objective(trial, data_dir, tuning_epochs, device):
    """Optuna objective function - returns negative PSNR (we minimize)."""
    
    cfg = create_trial_config(trial)
    
    # Log trial parameters
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"  Model: {cfg['base_channels']} channels, {cfg['num_encoder_blocks']} blocks")
    print(f"  LR: {cfg['lr']:.2e}, WD: {cfg['weight_decay']:.2e}")
    print(f"  Loss: SAM={cfg['alpha_spectral']:.3f}, Grad={cfg['alpha_gradient']:.4f}, Perc={cfg['alpha_perceptual']:.3f}")
    print(f"  20m weight: {cfg['weight_20m']:.2f}, Batch: {cfg['batch_size']}")
    print(f"{'='*60}")
    
    try:
        # Create dataset
        dataset = FiveBandMmapDataset(data_dir, augment=True)
        n_val = max(int(len(dataset) * 0.15), 15)  # Slightly larger val for stable metrics
        n_train = len(dataset) - n_val
        
        train_dataset, val_dataset = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        val_dataset.dataset.augment = False
        
        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], 
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], 
                               shuffle=False, num_workers=4, pin_memory=True)
        
        # Create model
        model = MultiResolutionSatelliteSR(
            in_channels_10m=3,
            in_channels_20m=2,
            base_channels=cfg['base_channels'],
            num_encoder_blocks=cfg['num_encoder_blocks'],
            num_frames=3
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        
        # Create loss and optimizer
        criterion = MultiResSpectralLossWithPerceptual(cfg).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=cfg['patience_scheduler']
        )
        
        best_psnr = 0
        no_improve_count = 0
        patience = 10  # Early stopping for tuning
        
        for epoch in range(1, tuning_epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, psnr_10m, psnr_20m = validate(model, val_loader, criterion, device)
            avg_psnr = (psnr_10m + psnr_20m) / 2
            
            scheduler.step(val_loss)
            
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Report intermediate value for pruning
            trial.report(avg_psnr, epoch)
            
            # Prune trial if not promising
            if trial.should_prune():
                print(f"  Trial pruned at epoch {epoch}")
                raise optuna.TrialPruned()
            
            if epoch % 5 == 0 or epoch == tuning_epochs:
                print(f"  Epoch {epoch:2d}: loss={val_loss:.4f}, PSNR={avg_psnr:.2f} dB (best={best_psnr:.2f})")
            
            # Early stopping for tuning
            if no_improve_count >= patience:
                print(f"  Early stop at epoch {epoch}")
                break
        
        print(f"  Final: Best PSNR = {best_psnr:.2f} dB")
        return best_psnr  # Optuna will maximize this
        
    except Exception as e:
        print(f"  Trial failed: {e}")
        raise optuna.TrialPruned()


def run_full_training(cfg, data_dir, checkpoint_dir, epochs=200):
    """Run full training with best hyperparameters."""
    
    print(f"\n{'='*70}")
    print("FULL TRAINING WITH BEST HYPERPARAMETERS")
    print(f"{'='*70}")
    print(f"Config: {json.dumps(cfg, indent=2)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Full dataset
    dataset = FiveBandMmapDataset(data_dir, augment=True)
    n_val = max(int(len(dataset) * 0.1), 10)
    n_train = len(dataset) - n_val
    
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.augment = False
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = MultiResolutionSatelliteSR(
        in_channels_10m=3,
        in_channels_20m=2,
        base_channels=cfg['base_channels'],
        num_encoder_blocks=cfg['num_encoder_blocks'],
        num_frames=3
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['base_channels']} channels, {cfg['num_encoder_blocks']} blocks, {n_params:,} params")
    
    criterion = MultiResSpectralLossWithPerceptual(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=cfg['patience_scheduler']
    )
    
    best_val_loss = float('inf')
    best_psnr = 0
    no_improve_count = 0
    patience = 25  # Full training patience
    history = []
    
    print(f"\nStarting full training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        start = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, psnr_10m, psnr_20m = validate(model, val_loader, criterion, device)
        avg_psnr = (psnr_10m + psnr_20m) / 2
        
        scheduler.step(val_loss)
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]['lr']
        
        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'psnr_10m': psnr_10m, 'psnr_20m': psnr_20m, 'avg_psnr': avg_psnr, 'lr': lr
        })
        
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_psnr = avg_psnr
            no_improve_count = 0
            improved = " *"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'psnr_10m': psnr_10m,
                'psnr_20m': psnr_20m,
                'config': cfg
            }, checkpoint_dir / 'best_model.pt')
        else:
            no_improve_count += 1
        
        print(f"Epoch {epoch:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PSNR: {avg_psnr:.2f} dB | LR: {lr:.2e} | {elapsed:.1f}s{improved}")
        
        if no_improve_count >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Save history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Full training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"{'='*70}")
    
    return best_psnr


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for MultiRes with Perceptual Loss')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--tuning_epochs', type=int, default=30, help='Epochs per trial')
    parser.add_argument('--full_epochs', type=int, default=200, help='Epochs for full training')
    parser.add_argument('--skip_tuning', action='store_true', help='Skip tuning, use best known config')
    parser.add_argument('--study_name', type=str, default='multires_perceptual_tuning')
    args = parser.parse_args()
    
    # Data paths
    DATA_DIR = Path('data/sr_training/5band_v3/combined/mmap')
    CHECKPOINT_DIR = Path('checkpoints/multires_perceptual_tuned')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data: {DATA_DIR}")
    
    if not args.skip_tuning:
        # Create Optuna study
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER TUNING: {args.n_trials} trials, {args.tuning_epochs} epochs each")
        print(f"{'='*70}")
        
        study = optuna.create_study(
            study_name=args.study_name,
            direction='maximize',  # Maximize PSNR
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=5
            ),
            storage=f'sqlite:///optuna_{args.study_name}.db',
            load_if_exists=True
        )
        
        # Add current best as a baseline trial
        study.enqueue_trial({
            'base_channels': 64,
            'num_encoder_blocks': 4,
            'lr': 3.064e-05,
            'weight_decay': 5.235e-06,
            'patience_scheduler': 4,
            'alpha_spectral': 0.137,
            'alpha_gradient': 0.01,
            'alpha_perceptual': 0.1,
            'weight_20m': 1.0,
            'batch_size': 4,
        })
        
        study.optimize(
            lambda trial: objective(trial, DATA_DIR, args.tuning_epochs, device),
            n_trials=args.n_trials,
            timeout=None,
            gc_after_trial=True,
            show_progress_bar=True
        )
        
        # Print results
        print(f"\n{'='*70}")
        print("TUNING RESULTS")
        print(f"{'='*70}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best PSNR: {study.best_value:.2f} dB")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Compare with current best
        print(f"\n{'='*70}")
        print("COMPARISON WITH CURRENT BEST")
        print(f"{'='*70}")
        print(f"{'Parameter':<25} {'Current':>15} {'Tuned':>15}")
        print("-" * 55)
        for key in study.best_params:
            current = CURRENT_BEST.get(key, 'N/A')
            tuned = study.best_params[key]
            if isinstance(current, float):
                print(f"{key:<25} {current:>15.6f} {tuned:>15.6f}")
            else:
                print(f"{key:<25} {str(current):>15} {str(tuned):>15}")
        
        # Create best config
        best_cfg = {
            'base_channels': study.best_params['base_channels'],
            'num_encoder_blocks': study.best_params['num_encoder_blocks'],
            'optimizer': 'adamw',
            'lr': study.best_params['lr'],
            'weight_decay': study.best_params['weight_decay'],
            'patience_scheduler': study.best_params['patience_scheduler'],
            'alpha_pixel': 1.0,
            'alpha_spectral': study.best_params['alpha_spectral'],
            'alpha_gradient': study.best_params['alpha_gradient'],
            'alpha_perceptual': study.best_params['alpha_perceptual'],
            'weight_10m': 1.0,
            'weight_20m': study.best_params['weight_20m'],
            'batch_size': study.best_params['batch_size'],
        }
        
        # Save study results
        results_dir = Path('tuning_results')
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(results_dir / f'multires_perceptual_{timestamp}.json', 'w') as f:
            json.dump({
                'best_psnr': study.best_value,
                'best_params': study.best_params,
                'best_config': best_cfg,
                'n_trials': len(study.trials),
                'completed_trials': len([t for t in study.trials if t.state == TrialState.COMPLETE]),
            }, f, indent=2)
        
        print(f"\nResults saved to: tuning_results/multires_perceptual_{timestamp}.json")
        
    else:
        print("Using current best configuration (skipping tuning)")
        best_cfg = CURRENT_BEST.copy()
    
    # Full training with best config
    print(f"\n{'='*70}")
    print("STARTING FULL TRAINING WITH BEST HYPERPARAMETERS")
    print(f"{'='*70}")
    
    final_psnr = run_full_training(
        cfg=best_cfg if args.skip_tuning else best_cfg,
        data_dir=DATA_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        epochs=args.full_epochs
    )
    
    print(f"\n{'='*70}")
    print("TUNING + TRAINING COMPLETE")
    print(f"Final model PSNR: {final_psnr:.2f} dB")
    print(f"Model saved to: {CHECKPOINT_DIR / 'best_model.pt'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
