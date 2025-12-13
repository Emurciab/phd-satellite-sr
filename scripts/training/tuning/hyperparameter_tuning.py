#!/usr/bin/env python3
"""
Hyperparameter Tuning Framework for Satellite SR Models

Uses Optuna for efficient Bayesian optimization with:
- Pruning: Early stopping of unpromising trials
- Multi-objective: Optimize both loss and PSNR
- Distributed: Can run multiple trials in parallel
- Resumable: Studies can be saved and resumed

Usage:
    # Basic tuning (50 trials)
    python hyperparameter_tuning.py --n-trials 50
    
    # Quick exploration (fewer epochs per trial)
    python hyperparameter_tuning.py --n-trials 30 --epochs-per-trial 20
    
    # Resume previous study
    python hyperparameter_tuning.py --study-name my_study --resume
    
    # Tune specific model
    python hyperparameter_tuning.py --model-type uniform
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset

try:
    import optuna
    from optuna.trial import TrialState
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: Optuna not installed. Install with: pip install optuna")

# Use V3 model (no sigmoid, direct prediction) - fixes mode collapse
from models.satellite_sr_model_v3 import MultiResolutionSatelliteSRv3


# =============================================================================
# DATASET (Simplified for tuning - uses subset)
# =============================================================================

class MmapTuningDataset(Dataset):
    """Fast memory-mapped dataset for hyperparameter tuning."""
    
    def __init__(self, mmap_dir: str, sequence_length: int = 3, augment: bool = True):
        from pathlib import Path
        mmap_dir = Path(mmap_dir)
        
        # Load memory-mapped arrays
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


class TuningDataset(Dataset):
    """Lightweight dataset for hyperparameter tuning."""
    
    def __init__(self, samples, sequence_length=3, augment=True, uniform_3m=False):
        self.samples = [s for s in samples if self._is_valid(s, uniform_3m)]
        self.sequence_length = sequence_length
        self.augment = augment
        self.uniform_3m = uniform_3m
    
    @staticmethod
    def _is_valid(sample, uniform_3m):
        valid = (sample.lr_sequence is not None and 
                sample.lr_20m_sequence is not None and
                sample.hr_10m_sequence is not None and
                sample.hr_20m_sequence is not None)
        if uniform_3m:
            valid = valid and sample.hr_sequence is not None
        return valid
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        lr_10m = sample.lr_sequence.copy()
        lr_20m = sample.lr_20m_sequence.copy()
        hr_10m = sample.hr_10m_sequence.copy()
        
        if self.uniform_3m:
            hr_full = sample.hr_sequence.copy()
            hr_20m = hr_full[:, :, :, 3:5]  # RedEdge+NIR at 3m (256×256)
        else:
            hr_20m = sample.hr_20m_sequence.copy()  # 6m (128×128)
        
        # Handle temporal dimension
        T = lr_10m.shape[0]
        if T > self.sequence_length:
            start = np.random.randint(0, T - self.sequence_length + 1)
            lr_10m = lr_10m[start:start + self.sequence_length]
            lr_20m = lr_20m[start:start + self.sequence_length]
            hr_10m = hr_10m[start:start + self.sequence_length]
            hr_20m = hr_20m[start:start + self.sequence_length]
        elif T < self.sequence_length:
            while lr_10m.shape[0] < self.sequence_length:
                lr_10m = np.concatenate([lr_10m, lr_10m[-1:]], axis=0)
                lr_20m = np.concatenate([lr_20m, lr_20m[-1:]], axis=0)
                hr_10m = np.concatenate([hr_10m, hr_10m[-1:]], axis=0)
                hr_20m = np.concatenate([hr_20m, hr_20m[-1:]], axis=0)
        
        # Simple augmentation
        if self.augment and np.random.rand() > 0.5:
            lr_10m = np.flip(lr_10m, axis=2).copy()
            lr_20m = np.flip(lr_20m, axis=2).copy()
            hr_10m = np.flip(hr_10m, axis=2).copy()
            hr_20m = np.flip(hr_20m, axis=2).copy()
        
        # To tensors
        lr_10m = torch.from_numpy(lr_10m).permute(0, 3, 1, 2).float()
        lr_20m = torch.from_numpy(lr_20m).permute(0, 3, 1, 2).float()
        hr_10m = torch.from_numpy(hr_10m[self.sequence_length//2]).permute(2, 0, 1).float()
        hr_20m = torch.from_numpy(hr_20m[self.sequence_length//2]).permute(2, 0, 1).float()
        
        return {'lr_10m': lr_10m, 'lr_20m': lr_20m, 'hr_10m': hr_10m, 'hr_20m': hr_20m}


# =============================================================================
# LOSS FUNCTION (Configurable)
# =============================================================================

class ConfigurableLoss(nn.Module):
    """Loss function with tunable weights."""
    
    def __init__(
        self,
        weight_10m: float = 1.0,
        weight_20m: float = 1.0,
        alpha_l1: float = 1.0,
        alpha_mse: float = 0.1,
        alpha_spectral: float = 0.1,
        alpha_gradient: float = 0.01
    ):
        super().__init__()
        self.weight_10m = weight_10m
        self.weight_20m = weight_20m
        self.alpha_l1 = alpha_l1
        self.alpha_mse = alpha_mse
        self.alpha_spectral = alpha_spectral
        self.alpha_gradient = alpha_gradient
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Sobel filters
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def spectral_angle_loss(self, pred, target):
        pred_norm = nn.functional.normalize(pred, dim=1, eps=1e-8)
        target_norm = nn.functional.normalize(target, dim=1, eps=1e-8)
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        return 1.0 - cos_sim.mean()
    
    def gradient_loss(self, pred, target):
        loss = 0.0
        for c in range(pred.shape[1]):
            pred_c = pred[:, c:c+1]
            target_c = target[:, c:c+1]
            
            pred_gx = nn.functional.conv2d(pred_c, self.sobel_x, padding=1)
            pred_gy = nn.functional.conv2d(pred_c, self.sobel_y, padding=1)
            target_gx = nn.functional.conv2d(target_c, self.sobel_x, padding=1)
            target_gy = nn.functional.conv2d(target_c, self.sobel_y, padding=1)
            
            loss += self.l1_loss(pred_gx, target_gx) + self.l1_loss(pred_gy, target_gy)
        return loss / pred.shape[1]
    
    def branch_loss(self, pred, target):
        loss = self.alpha_l1 * self.l1_loss(pred, target)
        loss += self.alpha_mse * self.mse_loss(pred, target)
        loss += self.alpha_spectral * self.spectral_angle_loss(pred, target)
        loss += self.alpha_gradient * self.gradient_loss(pred, target)
        return loss
    
    def forward(self, pred_10m, pred_20m, target_10m, target_20m):
        loss_10m = self.branch_loss(pred_10m, target_10m)
        loss_20m = self.branch_loss(pred_20m, target_20m)
        total = self.weight_10m * loss_10m + self.weight_20m * loss_20m
        return total, loss_10m, loss_20m


# =============================================================================
# METRICS
# =============================================================================

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)).item()


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

def get_search_space(trial: 'optuna.Trial', model_type: str = 'multires', sequence_length: int = 3) -> Dict[str, Any]:
    """Define the hyperparameter search space."""
    
    params = {
        # Architecture
        'base_channels': trial.suggest_categorical('base_channels', [32, 48, 64, 96]),
        'num_encoder_blocks': trial.suggest_int('num_encoder_blocks', 2, 6),
        # num_frames must match dataset sequence_length (fixed at dataset creation time)
        'num_frames': sequence_length,
        
        # Optimizer
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        
        # Scheduler
        'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'step', 'none']),
        
        # Loss weights
        'alpha_l1': trial.suggest_float('alpha_l1', 0.5, 2.0),
        'alpha_mse': trial.suggest_float('alpha_mse', 0.0, 0.5),
        'alpha_spectral': trial.suggest_float('alpha_spectral', 0.0, 0.3),
        'alpha_gradient': trial.suggest_float('alpha_gradient', 0.0, 0.1),
        
        # Branch weights
        'weight_10m': trial.suggest_float('weight_10m', 0.5, 2.0),
        'weight_20m': trial.suggest_float('weight_20m', 0.5, 2.0),
        
        # Training
        'batch_size': trial.suggest_categorical('batch_size', [2, 4, 8]),
    }
    
    # Scheduler-specific params
    if params['scheduler'] == 'step':
        params['step_size'] = trial.suggest_int('step_size', 5, 20)
        params['gamma'] = trial.suggest_float('gamma', 0.1, 0.5)
    elif params['scheduler'] == 'plateau':
        params['patience_scheduler'] = trial.suggest_int('patience_scheduler', 3, 10)
    
    return params


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

class Objective:
    """Optuna objective function for hyperparameter optimization."""
    
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        device: torch.device,
        epochs_per_trial: int = 30,
        model_type: str = 'multires',
        early_stop_patience: int = 10,
        sequence_length: int = 3
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.epochs_per_trial = epochs_per_trial
        self.model_type = model_type
        self.early_stop_patience = early_stop_patience
        self.sequence_length = sequence_length
        
        # Scale factors
        if model_type == 'uniform':
            self.scale_10m = 256 / 77  # 3.32×
            self.scale_20m = 256 / 38  # 6.74×
        else:
            self.scale_10m = 256 / 77  # 3.32×
            self.scale_20m = 128 / 38  # 3.37×
    
    def __call__(self, trial: 'optuna.Trial') -> float:
        # Get hyperparameters
        params = get_search_space(trial, self.model_type, self.sequence_length)
        
        # Create dataloaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=params['batch_size'],
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Create model - USE V3 (no sigmoid, direct prediction)
        model = MultiResolutionSatelliteSRv3(
            in_channels_10m=3,
            in_channels_20m=2,
            base_channels=params['base_channels'],
            num_encoder_blocks=params['num_encoder_blocks'],
            num_frames=params['num_frames'],
            scale_factor_10m=self.scale_10m,
            scale_factor_20m=self.scale_20m,
        ).to(self.device)
        
        # Create loss
        criterion = ConfigurableLoss(
            weight_10m=params['weight_10m'],
            weight_20m=params['weight_20m'],
            alpha_l1=params['alpha_l1'],
            alpha_mse=params['alpha_mse'],
            alpha_spectral=params['alpha_spectral'],
            alpha_gradient=params['alpha_gradient'],
        ).to(self.device)
        
        # Create optimizer
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        elif params['optimizer'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        else:  # sgd
            optimizer = optim.SGD(
                model.parameters(), 
                lr=params['lr'], 
                weight_decay=params['weight_decay'],
                momentum=params.get('momentum', 0.9)
            )
        
        # Create scheduler
        if params['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs_per_trial, eta_min=1e-6)
        elif params['scheduler'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=params.get('patience_scheduler', 5), factor=0.5
            )
        elif params['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=params.get('step_size', 10), gamma=params.get('gamma', 0.1)
            )
        else:
            scheduler = None
        
        # Training loop
        best_val_loss = float('inf')
        best_psnr = 0.0
        patience_counter = 0
        last_improvement_epoch = 0
        loss_history = []
        psnr_history = []
        
        for epoch in range(self.epochs_per_trial):
            # Train
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                lr_10m = batch['lr_10m'].to(self.device)
                lr_20m = batch['lr_20m'].to(self.device)
                hr_10m = batch['hr_10m'].to(self.device)
                hr_20m = batch['hr_20m'].to(self.device)
                
                optimizer.zero_grad()
                pred_10m, pred_20m = model(
                    lr_10m, lr_20m,
                    target_size_10m=(hr_10m.shape[2], hr_10m.shape[3]),
                    target_size_20m=(hr_20m.shape[2], hr_20m.shape[3])
                )
                loss, _, _ = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_psnr_10m = 0.0
            val_psnr_20m = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    lr_10m = batch['lr_10m'].to(self.device)
                    lr_20m = batch['lr_20m'].to(self.device)
                    hr_10m = batch['hr_10m'].to(self.device)
                    hr_20m = batch['hr_20m'].to(self.device)
                    
                    pred_10m, pred_20m = model(
                        lr_10m, lr_20m,
                        target_size_10m=(hr_10m.shape[2], hr_10m.shape[3]),
                        target_size_20m=(hr_20m.shape[2], hr_20m.shape[3])
                    )
                    loss, _, _ = criterion(pred_10m, pred_20m, hr_10m, hr_20m)
                    val_loss += loss.item()
                    val_psnr_10m += calculate_psnr(pred_10m, hr_10m)
                    val_psnr_20m += calculate_psnr(pred_20m, hr_20m)
            
            val_loss /= len(val_loader)
            val_psnr_10m /= len(val_loader)
            val_psnr_20m /= len(val_loader)
            avg_psnr = (val_psnr_10m + val_psnr_20m) / 2
            
            # Update scheduler
            if scheduler is not None:
                if params['scheduler'] == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_psnr = avg_psnr
                patience_counter = 0
                last_improvement_epoch = epoch
            else:
                patience_counter += 1
            
            # Track improvement trend in last N epochs
            loss_history.append(val_loss)
            psnr_history.append(avg_psnr)
            
            # Report to Optuna for pruning
            trial.report(val_loss, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping within trial
            if patience_counter >= self.early_stop_patience:
                break
        
        # Analyze if configuration shows promise for extended training
        # Check if still improving in last 20% of epochs
        final_epochs = max(5, int(self.epochs_per_trial * 0.2))
        if len(psnr_history) >= final_epochs:
            recent_psnr = psnr_history[-final_epochs:]
            # Linear regression to detect trend
            x = np.arange(len(recent_psnr))
            slope = np.polyfit(x, recent_psnr, 1)[0]
            still_improving = bool(slope > 0.01)  # Convert numpy.bool_ to Python bool
            
            # Store in trial's user attributes (convert all to native Python types)
            trial.set_user_attr('still_improving', still_improving)
            trial.set_user_attr('final_slope', float(slope))
            trial.set_user_attr('last_improvement_epoch', int(last_improvement_epoch))
            trial.set_user_attr('best_psnr', float(best_psnr))
            trial.set_user_attr('final_psnr', float(psnr_history[-1]))
            trial.set_user_attr('epochs_completed', int(len(psnr_history)))
        
        # Return best validation loss (Optuna minimizes by default)
        return best_val_loss


# =============================================================================
# MAIN TUNING FUNCTION
# =============================================================================

def run_hyperparameter_tuning(
    data_path: str,
    model_type: str = 'multires',
    n_trials: int = 50,
    epochs_per_trial: int = 30,
    study_name: Optional[str] = None,
    resume: bool = False,
    output_dir: str = './tuning_results',
    subset_fraction: float = 0.5,  # Use subset for faster tuning
    use_mmap: bool = True,  # NEW: Use mmap dataset for faster loading
):
    """
    Run hyperparameter tuning.
    
    Args:
        data_path: Path to mmap directory or dataset.pkl
        model_type: 'multires' or 'uniform'
        n_trials: Number of Optuna trials
        epochs_per_trial: Max epochs per trial
        study_name: Name for the study (for resuming)
        resume: Whether to resume existing study
        output_dir: Directory for results
        subset_fraction: Fraction of data to use (faster tuning)
        use_mmap: Use memory-mapped dataset (much faster)
    """
    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna is required. Install with: pip install optuna")
        return
    
    print("=" * 80)
    print("HYPERPARAMETER TUNING (V3 Model - No Sigmoid)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model type: {model_type}")
    print(f"  Trials: {n_trials}")
    print(f"  Epochs per trial: {epochs_per_trial}")
    print(f"  Data subset: {subset_fraction*100:.0f}%")
    print(f"  Using mmap: {use_mmap}")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    
    if use_mmap:
        # Use memory-mapped dataset (FAST!)
        full_dataset = MmapTuningDataset(data_path, sequence_length=3, augment=True)
        print(f"  Mmap samples: {len(full_dataset)}")
    else:
        # Legacy pickle-based loading
        sys.path.insert(0, '/media/eamurciab/TOSHIBA_EXT1/PhD/SR/scripts/preprocessing')
        from convert_cache_to_5band import TrainingSample5Band
        import __main__
        __main__.TrainingSample5Band = TrainingSample5Band
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        samples = data['samples']
        print(f"  Total samples: {len(samples)}")
        
        # Create dataset
        uniform_3m = (model_type == 'uniform')
        full_dataset = TuningDataset(samples, sequence_length=3, augment=True, uniform_3m=uniform_3m)
        print(f"  Valid samples: {len(full_dataset)}")
    
    # Use subset for faster tuning
    if subset_fraction < 1.0:
        n_subset = int(len(full_dataset) * subset_fraction)
        indices = np.random.permutation(len(full_dataset))[:n_subset]
        full_dataset = Subset(full_dataset, indices)
        print(f"  Using subset: {n_subset} samples")
    
    # Split
    n_val = int(len(full_dataset) * 0.2)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    print(f"  Train: {n_train}, Val: {n_val}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Study name
    if study_name is None:
        study_name = f"sr_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create or load study
    storage_path = output_dir / f"{study_name}.db"
    storage = f"sqlite:///{storage_path}"
    
    # Sampler and Pruner
    sampler = TPESampler(seed=42)
    pruner = HyperbandPruner(
        min_resource=5,  # Min epochs before pruning
        max_resource=epochs_per_trial,
        reduction_factor=3
    )
    
    if resume and storage_path.exists():
        print(f"\nResuming study: {study_name}")
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        print(f"\nCreating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='minimize',  # Minimize validation loss
            sampler=sampler,
            pruner=pruner,
            load_if_exists=resume
        )
    
    # Create objective
    sequence_length = 3  # Fixed to match dataset
    objective = Objective(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        epochs_per_trial=epochs_per_trial,
        model_type=model_type,
        early_stop_patience=10,
        sequence_length=sequence_length
    )
    
    # Run optimization with error handling
    print(f"\nStarting optimization...")
    print(f"  Study stored at: {storage_path}")
    print("-" * 80)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,  # Clean up GPU memory
        catch=(RuntimeError, ValueError)  # Catch and log errors instead of crashing
    )
    
    # Results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    # Best trial
    best_trial = study.best_trial
    print(f"\nBest Trial: #{best_trial.number}")
    print(f"  Value (val_loss): {best_trial.value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        'study_name': study_name,
        'model_type': model_type,
        'n_trials': n_trials,
        'epochs_per_trial': epochs_per_trial,
        'best_trial': {
            'number': best_trial.number,
            'value': best_trial.value,
            'params': best_trial.params,
            'best_psnr': best_trial.user_attrs.get('best_psnr', None),
        },
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state),
                'user_attrs': dict(t.user_attrs) if t.user_attrs else {},
            }
            for t in study.trials
        ]
    }
    
    results_path = output_dir / f"{study_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Print top 5 trials
    print(f"\nTop 5 Trials:")
    print("-" * 80)
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value)[:5]
    
    for i, t in enumerate(sorted_trials):
        print(f"  #{i+1} Trial {t.number}: val_loss={t.value:.4f}")
        print(f"      lr={t.params['lr']:.2e}, base_ch={t.params['base_channels']}, "
              f"blocks={t.params['num_encoder_blocks']}, opt={t.params['optimizer']}")
        # Show PSNR if available
        if 'best_psnr' in t.user_attrs:
            print(f"      PSNR={t.user_attrs['best_psnr']:.2f} dB")
    
    # =========================================================================
    # PROMISING CONFIGURATIONS - Still improving at end of training
    # =========================================================================
    print(f"\n" + "=" * 80)
    print("PROMISING CONFIGURATIONS (Still improving at epoch {})".format(epochs_per_trial))
    print("=" * 80)
    print("These configurations were still showing improvement trend at the end.")
    print("Consider extending training with more epochs for these.\n")
    
    promising_trials = [
        t for t in completed_trials 
        if t.user_attrs.get('still_improving', False)
    ]
    
    if promising_trials:
        # Sort by best PSNR
        promising_sorted = sorted(
            promising_trials, 
            key=lambda t: t.user_attrs.get('best_psnr', 0),
            reverse=True
        )
        
        for i, t in enumerate(promising_sorted[:10]):  # Top 10 promising
            attrs = t.user_attrs
            print(f"  Trial {t.number}:")
            print(f"    Best PSNR: {attrs.get('best_psnr', 0):.2f} dB")
            print(f"    Final PSNR: {attrs.get('final_psnr', 0):.2f} dB")
            print(f"    Trend: +{attrs.get('final_slope', 0):.3f} dB/epoch (last 20%)")
            print(f"    Last improvement: epoch {attrs.get('last_improvement_epoch', 0)}")
            print(f"    Epochs completed: {attrs.get('epochs_completed', 0)}")
            print(f"    Config: lr={t.params['lr']:.2e}, base_ch={t.params['base_channels']}, "
                  f"blocks={t.params['num_encoder_blocks']}")
            print()
        
        # Add promising trials to results
        results['promising_trials'] = [
            {
                'number': t.number,
                'params': t.params,
                'best_psnr': t.user_attrs.get('best_psnr', 0),
                'final_psnr': t.user_attrs.get('final_psnr', 0),
                'final_slope': t.user_attrs.get('final_slope', 0),
                'last_improvement_epoch': t.user_attrs.get('last_improvement_epoch', 0),
            }
            for t in promising_sorted[:10]
        ]
    else:
        print("  No trials showed continuing improvement at the end.")
        print("  This may mean:")
        print("    - Models converged within the epoch budget")
        print("    - Early stopping kicked in appropriately")
        print("    - The search found stable configurations")
    
    # Generate training command for best params
    print(f"\n" + "=" * 80)
    print("RECOMMENDED TRAINING COMMAND")
    print("=" * 80)
    
    bp = best_trial.params
    # Use V3 training script
    script = 'train_5band_v3.py'
    
    print(f"\npython scripts/training/{script} \\")
    print(f"    --epochs 200 \\")
    print(f"    --batch-size {bp['batch_size']} \\")
    print(f"    --lr {bp['lr']:.2e}")
    
    print(f"\n# Best tuned hyperparameters:")
    print(f"#   base_channels: {bp['base_channels']}")
    print(f"#   num_encoder_blocks: {bp['num_encoder_blocks']}")
    print(f"#   optimizer: {bp['optimizer']}")
    print(f"#   loss weights: L1={bp['alpha_l1']:.2f}, MSE={bp['alpha_mse']:.2f}, "
          f"SAM={bp['alpha_spectral']:.2f}, Grad={bp['alpha_gradient']:.2f}")
    
    return study, results


# =============================================================================
# QUICK TUNING MODE
# =============================================================================

def quick_tune(data_path: str, model_type: str = 'multires', use_mmap: bool = False):
    """
    Quick tuning mode with reduced search space.
    Good for initial exploration.
    """
    print("=" * 80)
    print("QUICK HYPERPARAMETER EXPLORATION")
    print("=" * 80)
    
    return run_hyperparameter_tuning(
        data_path=data_path,
        model_type=model_type,
        n_trials=20,
        epochs_per_trial=15,
        subset_fraction=0.3,
        output_dir='./tuning_results/quick',
        use_mmap=use_mmap,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for SR V3 models')
    parser.add_argument('--data-dir', type=str,
                        default='./data/sr_training/5band_v3/combined/mmap',
                        help='Path to mmap dataset directory OR pickle file')
    parser.add_argument('--model-type', type=str, default='multires',
                        choices=['multires', 'uniform'],
                        help='Model type to tune')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--epochs-per-trial', type=int, default=30,
                        help='Max epochs per trial')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (for resuming)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume existing study')
    parser.add_argument('--output-dir', type=str, default='./tuning_results',
                        help='Output directory')
    parser.add_argument('--subset', type=float, default=0.5,
                        help='Fraction of data to use (0.0-1.0)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick exploration mode (20 trials, 15 epochs)')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    
    # Auto-detect data format
    if data_path.is_dir() and (data_path / 'hr_10m.npy').exists():
        # Memory-mapped format
        use_mmap = True
        actual_path = str(data_path)
        print(f"Detected mmap dataset at: {data_path}")
    elif data_path.is_file() and data_path.suffix == '.pkl':
        # Pickle file
        use_mmap = False
        actual_path = str(data_path)
    else:
        # Try adding dataset.pkl
        pkl_path = data_path / 'dataset.pkl'
        if pkl_path.exists():
            use_mmap = False
            actual_path = str(pkl_path)
        else:
            print(f"ERROR: Dataset not found at {data_path}")
            print(f"  Expected either:")
            print(f"    - Mmap directory with hr_10m.npy, hr_20m.npy, lr_10m.npy, lr_20m.npy")
            print(f"    - Pickle file (*.pkl)")
            return
    
    if args.quick:
        quick_tune(actual_path, args.model_type, use_mmap=use_mmap)
    else:
        run_hyperparameter_tuning(
            data_path=actual_path,
            model_type=args.model_type,
            n_trials=args.n_trials,
            epochs_per_trial=args.epochs_per_trial,
            study_name=args.study_name,
            resume=args.resume,
            output_dir=args.output_dir,
            subset_fraction=args.subset,
            use_mmap=use_mmap,
        )


if __name__ == '__main__':
    main()
