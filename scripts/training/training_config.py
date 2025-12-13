#!/usr/bin/env python3
"""
Training Configuration Management

Provides a clean way to manage and apply tuned hyperparameters.
Supports JSON config files and preset configurations.

Usage:
    # Train with config file
    python train_with_config.py --config configs/best_multires.json
    
    # Train with preset
    python train_with_config.py --preset baseline
    
    # Override specific params
    python train_with_config.py --config configs/best.json --lr 5e-5 --epochs 100
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    base_channels: int = 64
    num_encoder_blocks: int = 4
    num_frames: int = 3
    in_channels_10m: int = 3
    in_channels_20m: int = 2


@dataclass
class LossConfig:
    """Loss function configuration."""
    weight_10m: float = 1.0
    weight_20m: float = 1.0
    alpha_l1: float = 1.0
    alpha_mse: float = 0.1
    alpha_spectral: float = 0.1
    alpha_gradient: float = 0.01


@dataclass 
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = 'adamw'
    lr: float = 1e-4
    weight_decay: float = 1e-5
    momentum: float = 0.9  # For SGD
    betas: tuple = (0.9, 0.999)  # For Adam


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str = 'cosine'
    T_max: Optional[int] = None  # For cosine (defaults to epochs)
    eta_min: float = 1e-6
    step_size: int = 10  # For step
    gamma: float = 0.1  # For step
    patience: int = 5  # For plateau
    factor: float = 0.5  # For plateau


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool = True
    patience_loss: int = 15
    patience_psnr: int = 20
    min_delta_loss: float = 1e-4
    min_delta_psnr: float = 0.1


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    # Basic
    epochs: int = 100
    batch_size: int = 4
    val_split: float = 0.2
    
    # Components
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    
    # Data
    data_dir: str = './data/sr_training/5band_full/superdove_5band'
    save_dir: str = './checkpoints'
    augment: bool = True
    
    # Model type
    model_type: str = 'multires'  # 'multires' or 'uniform'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        config = cls()
        
        # Basic params
        for key in ['epochs', 'batch_size', 'val_split', 'data_dir', 'save_dir', 
                    'augment', 'model_type']:
            if key in data:
                setattr(config, key, data[key])
        
        # Nested configs
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'loss' in data:
            config.loss = LossConfig(**data['loss'])
        if 'optimizer' in data:
            # Handle tuple conversion for betas
            opt_data = data['optimizer'].copy()
            if 'betas' in opt_data and isinstance(opt_data['betas'], list):
                opt_data['betas'] = tuple(opt_data['betas'])
            config.optimizer = OptimizerConfig(**opt_data)
        if 'scheduler' in data:
            config.scheduler = SchedulerConfig(**data['scheduler'])
        if 'early_stopping' in data:
            config.early_stopping = EarlyStoppingConfig(**data['early_stopping'])
        
        return config


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

PRESETS = {
    'baseline': TrainingConfig(
        epochs=100,
        batch_size=4,
        model=ModelConfig(base_channels=64, num_encoder_blocks=4, num_frames=3),
        loss=LossConfig(alpha_l1=1.0, alpha_mse=0.1, alpha_spectral=0.1, alpha_gradient=0.01),
        optimizer=OptimizerConfig(name='adamw', lr=1e-4, weight_decay=1e-5),
        scheduler=SchedulerConfig(name='cosine'),
    ),
    
    'lightweight': TrainingConfig(
        epochs=80,
        batch_size=8,
        model=ModelConfig(base_channels=32, num_encoder_blocks=3, num_frames=1),
        loss=LossConfig(alpha_l1=1.0, alpha_mse=0.1, alpha_spectral=0.05, alpha_gradient=0.01),
        optimizer=OptimizerConfig(name='adam', lr=2e-4, weight_decay=1e-6),
        scheduler=SchedulerConfig(name='cosine'),
    ),
    
    'heavy': TrainingConfig(
        epochs=150,
        batch_size=2,
        model=ModelConfig(base_channels=96, num_encoder_blocks=6, num_frames=5),
        loss=LossConfig(alpha_l1=1.0, alpha_mse=0.1, alpha_spectral=0.15, alpha_gradient=0.02),
        optimizer=OptimizerConfig(name='adamw', lr=5e-5, weight_decay=1e-4),
        scheduler=SchedulerConfig(name='cosine'),
    ),
    
    'spectral_focus': TrainingConfig(
        epochs=100,
        batch_size=4,
        model=ModelConfig(base_channels=64, num_encoder_blocks=4, num_frames=3),
        loss=LossConfig(alpha_l1=0.8, alpha_mse=0.1, alpha_spectral=0.25, alpha_gradient=0.01),
        optimizer=OptimizerConfig(name='adamw', lr=1e-4, weight_decay=1e-5),
        scheduler=SchedulerConfig(name='cosine'),
    ),
    
    'fast_convergence': TrainingConfig(
        epochs=50,
        batch_size=4,
        model=ModelConfig(base_channels=48, num_encoder_blocks=4, num_frames=3),
        loss=LossConfig(alpha_l1=1.0, alpha_mse=0.1, alpha_spectral=0.1, alpha_gradient=0.01),
        optimizer=OptimizerConfig(name='adamw', lr=2e-4, weight_decay=1e-5),
        scheduler=SchedulerConfig(name='plateau', patience=5, factor=0.5),
        early_stopping=EarlyStoppingConfig(patience_loss=10, patience_psnr=15),
    ),
}


def get_preset(name: str) -> TrainingConfig:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def list_presets():
    """Print available presets."""
    print("\nAvailable Presets:")
    print("-" * 60)
    for name, config in PRESETS.items():
        print(f"\n{name}:")
        print(f"  epochs={config.epochs}, batch_size={config.batch_size}")
        print(f"  model: channels={config.model.base_channels}, blocks={config.model.num_encoder_blocks}, frames={config.model.num_frames}")
        print(f"  optimizer: {config.optimizer.name}, lr={config.optimizer.lr}")


# =============================================================================
# APPLY CONFIG TO TRAINING
# =============================================================================

def apply_config_to_args(config: TrainingConfig, args: argparse.Namespace) -> argparse.Namespace:
    """Apply configuration to argparse namespace."""
    # Basic training params
    args.epochs = config.epochs
    args.batch_size = config.batch_size
    args.val_split = config.val_split
    args.lr = config.optimizer.lr
    args.data_dir = config.data_dir
    args.save_dir = config.save_dir
    args.num_frames = config.model.num_frames
    args.no_augment = not config.augment
    
    # Early stopping
    args.no_early_stop = not config.early_stopping.enabled
    args.patience_loss = config.early_stopping.patience_loss
    args.patience_psnr = config.early_stopping.patience_psnr
    args.min_delta_loss = config.early_stopping.min_delta_loss
    args.min_delta_psnr = config.early_stopping.min_delta_psnr
    
    # Store full config for access in training script
    args._config = config
    
    return args


# =============================================================================
# GENERATE CONFIGS FROM TUNING RESULTS
# =============================================================================

def create_config_from_tuning_results(tuning_results_path: str, output_path: str = None):
    """
    Create a TrainingConfig from Optuna tuning results JSON.
    
    Args:
        tuning_results_path: Path to tuning results JSON
        output_path: Where to save the config (optional)
    
    Returns:
        TrainingConfig with tuned parameters
    """
    with open(tuning_results_path, 'r') as f:
        results = json.load(f)
    
    params = results['best_trial']['params']
    
    config = TrainingConfig(
        epochs=100,  # Set for full training
        batch_size=params.get('batch_size', 4),
        model_type=results.get('model_type', 'multires'),
        
        model=ModelConfig(
            base_channels=params.get('base_channels', 64),
            num_encoder_blocks=params.get('num_encoder_blocks', 4),
            num_frames=params.get('num_frames', 3),
        ),
        
        loss=LossConfig(
            weight_10m=params.get('weight_10m', 1.0),
            weight_20m=params.get('weight_20m', 1.0),
            alpha_l1=params.get('alpha_l1', 1.0),
            alpha_mse=params.get('alpha_mse', 0.1),
            alpha_spectral=params.get('alpha_spectral', 0.1),
            alpha_gradient=params.get('alpha_gradient', 0.01),
        ),
        
        optimizer=OptimizerConfig(
            name=params.get('optimizer', 'adamw'),
            lr=params.get('lr', 1e-4),
            weight_decay=params.get('weight_decay', 1e-5),
            momentum=params.get('momentum', 0.9),
        ),
        
        scheduler=SchedulerConfig(
            name=params.get('scheduler', 'cosine'),
            step_size=params.get('step_size', 10),
            gamma=params.get('gamma', 0.1),
            patience=params.get('patience_scheduler', 5),
        ),
    )
    
    if output_path:
        config.save(output_path)
    
    return config


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Training configuration management')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List presets
    list_parser = subparsers.add_parser('list', help='List available presets')
    
    # Create config
    create_parser = subparsers.add_parser('create', help='Create config from preset')
    create_parser.add_argument('--preset', type=str, default='baseline', help='Preset name')
    create_parser.add_argument('--output', type=str, required=True, help='Output path')
    create_parser.add_argument('--model-type', type=str, default='multires', 
                               choices=['multires', 'uniform'])
    
    # Convert tuning results
    convert_parser = subparsers.add_parser('from-tuning', help='Create config from tuning results')
    convert_parser.add_argument('--results', type=str, required=True, help='Tuning results JSON')
    convert_parser.add_argument('--output', type=str, required=True, help='Output path')
    
    # Show config
    show_parser = subparsers.add_parser('show', help='Show config contents')
    show_parser.add_argument('--config', type=str, help='Config file path')
    show_parser.add_argument('--preset', type=str, help='Preset name')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_presets()
    
    elif args.command == 'create':
        config = get_preset(args.preset)
        config.model_type = args.model_type
        config.save(args.output)
    
    elif args.command == 'from-tuning':
        create_config_from_tuning_results(args.results, args.output)
        print(f"Config created from tuning results: {args.output}")
    
    elif args.command == 'show':
        if args.config:
            config = TrainingConfig.load(args.config)
        elif args.preset:
            config = get_preset(args.preset)
        else:
            print("Specify --config or --preset")
            return
        
        print(json.dumps(config.to_dict(), indent=2))
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
