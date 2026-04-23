#!/usr/bin/env python3
"""
Master training script for ALL model variants.

Trains the following models using combined data from all STAs:
1. MultiRes (Full) - V3 with all features
2. MultiRes (Filtered) - V3 with quality filtering  
3. BestFrame (Residual)
4. BestFrame (Direct)
5. Median (Residual)
6. Median (Direct)
7. MultiOutput (Residual)
8. MultiOutput (Direct)
9. MultiOutput SchedPer (Residual)
10. MultiOutput SchedPer (Direct)
11. Temporal (Residual)
12. Temporal (Direct)
13. GAN

Usage:
    python train_all_models.py --data-dir data/sr_training/all_stas_combined
    python train_all_models.py --model bestframe_residual  # Train single model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime


# Model configurations
MODEL_CONFIGS = {
    # MultiRes variants (using train_5band_v3.py or train_5band_multires.py)
    'multires_full': {
        'script': 'scripts/training/train_5band_v3.py',
        'name': 'MultiRes_Full_AllSTAs',
        'description': 'MultiRes V3 with all features',
    },
    'multires_filtered': {
        'script': 'scripts/training/train_5band_v3.py',
        'name': 'MultiRes_Filtered_AllSTAs',
        'description': 'MultiRes V3 with quality-filtered samples',
        'extra_args': ['--filter-quality'],
    },
    
    # BestFrame variants
    'bestframe_residual': {
        'script': 'scripts/training/variants/bestframe/train_multires_temporal_bestFrame.py',
        'name': 'BestFrame_Residual_AllSTAs',
        'description': 'Best frame selection with residual learning',
    },
    'bestframe_direct': {
        'script': 'scripts/training/variants/bestframe/train_multires_temporal_bestFrame_direct.py',
        'name': 'BestFrame_Direct_AllSTAs',
        'description': 'Best frame selection with direct prediction',
    },
    
    # Median variants
    'median_residual': {
        'script': 'scripts/training/variants/median/train_multires_temporal_median.py',
        'name': 'Median_Residual_AllSTAs',
        'description': 'Temporal median with residual learning',
    },
    'median_direct': {
        'script': 'scripts/training/variants/median/train_multires_temporal_median_direct.py',
        'name': 'Median_Direct_AllSTAs',
        'description': 'Temporal median with direct prediction',
    },
    
    # MultiOutput variants
    'multioutput_residual': {
        'script': 'scripts/training/variants/multioutput/train_multires_temporal_multiOutput.py',
        'name': 'MultiOutput_Residual_AllSTAs',
        'description': 'Multi-output temporal with residual learning',
    },
    'multioutput_direct': {
        'script': 'scripts/training/variants/multioutput/train_multires_temporal_multiOutput_direct.py',
        'name': 'MultiOutput_Direct_AllSTAs',
        'description': 'Multi-output temporal with direct prediction',
    },
    
    # MultiOutput SchedPer variants
    'multioutput_schedper_residual': {
        'script': 'scripts/training/variants/multioutput_schedper/train_multires_temporal_multiOutput_schedPer.py',
        'name': 'MultiOutput_SchedPer_Residual_AllSTAs',
        'description': 'Multi-output scheduled perceptual with residual learning',
    },
    'multioutput_schedper_direct': {
        'script': 'scripts/training/variants/multioutput_schedper/train_multires_temporal_multiOutput_schedPer_direct.py',
        'name': 'MultiOutput_SchedPer_Direct_AllSTAs',
        'description': 'Multi-output scheduled perceptual with direct prediction',
    },
    
    # Temporal variants
    'temporal_residual': {
        'script': 'scripts/training/variants/temporal/train_multires_temporal.py',
        'name': 'Temporal_Residual_AllSTAs',
        'description': 'Temporal fusion with residual learning',
    },
    'temporal_direct': {
        'script': 'scripts/training/variants/temporal/train_multires_temporal_direct.py',
        'name': 'Temporal_Direct_AllSTAs',
        'description': 'Temporal fusion with direct prediction',
    },
    
    # GAN
    'gan': {
        'script': 'scripts/training/variants/GAN/train_multires_temporal_GAN.py',
        'name': 'GAN_AllSTAs',
        'description': 'GAN-based super resolution',
    },
}


def run_training(model_key: str, config: dict, data_dir: Path, 
                 checkpoint_dir: Path, epochs: int, batch_size: int,
                 dry_run: bool = False) -> dict:
    """Run training for a single model"""
    
    script = config['script']
    name = config['name']
    
    # Build command
    cmd = [
        'python', script,
        '--data-dir', str(data_dir / 'mmap_train'),
        '--test-dir', str(data_dir / 'mmap_test'),
        '--checkpoint-dir', str(checkpoint_dir / name),
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--name', name,
    ]
    
    # Add extra args if specified
    if 'extra_args' in config:
        cmd.extend(config['extra_args'])
    
    print(f"\n{'='*70}")
    print(f"🚀 Training: {name}")
    print(f"   Script: {script}")
    print(f"   Description: {config['description']}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("   [DRY RUN - not executing]")
        return {'model': model_key, 'status': 'dry_run', 'duration': 0}
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent,  # Project root
            capture_output=False,  # Show output in terminal
            text=True,
        )
        
        duration = time.time() - start_time
        status = 'success' if result.returncode == 0 else 'failed'
        
        return {
            'model': model_key,
            'name': name,
            'status': status,
            'returncode': result.returncode,
            'duration': duration,
        }
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            'model': model_key,
            'name': name,
            'status': 'error',
            'error': str(e),
            'duration': duration,
        }


def main():
    parser = argparse.ArgumentParser(description='Train all model variants')
    parser.add_argument('--data-dir', type=str, default='data/sr_training/all_stas_combined',
                        help='Directory with prepared mmap datasets')
    parser.add_argument('--checkpoint-dir', type=str, default='scripts/training/checkpoints/all_stas',
                        help='Directory for model checkpoints')
    parser.add_argument('--model', type=str, default=None,
                        help='Train specific model only (e.g., bestframe_residual)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Max epochs per model')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit')
    parser.add_argument('--skip-to', type=str, default=None,
                        help='Skip models until this one (for resuming)')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable models:")
        print("-" * 50)
        for key, config in MODEL_CONFIGS.items():
            print(f"  {key:30s} - {config['description']}")
        return
    
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    
    # Verify data exists
    if not (data_dir / 'mmap_train').exists():
        print(f"❌ Training data not found at {data_dir / 'mmap_train'}")
        print(f"   Run: python scripts/training/prepare_all_stas_dataset.py")
        return
    
    # Determine which models to train
    if args.model:
        if args.model not in MODEL_CONFIGS:
            print(f"❌ Unknown model: {args.model}")
            print(f"   Available: {', '.join(MODEL_CONFIGS.keys())}")
            return
        models_to_train = {args.model: MODEL_CONFIGS[args.model]}
    else:
        models_to_train = MODEL_CONFIGS
    
    # Skip to specific model if requested
    if args.skip_to:
        skip = True
        filtered = {}
        for key, config in models_to_train.items():
            if key == args.skip_to:
                skip = False
            if not skip:
                filtered[key] = config
        models_to_train = filtered
    
    print("=" * 70)
    print("TRAINING ALL MODEL VARIANTS")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Epochs per model: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Models to train: {len(models_to_train)}")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - Commands will be printed but not executed")
    
    # Train each model
    results = []
    total_start = time.time()
    
    for i, (model_key, config) in enumerate(models_to_train.items(), 1):
        print(f"\n[{i}/{len(models_to_train)}] Starting {model_key}...")
        
        result = run_training(
            model_key, config, data_dir, checkpoint_dir,
            args.epochs, args.batch_size, args.dry_run
        )
        results.append(result)
        
        # Save intermediate results
        results_file = checkpoint_dir / 'training_results.json'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'data_dir': str(data_dir),
                'results': results,
            }, f, indent=2)
    
    total_duration = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"Total models: {len(results)}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Total time: {total_duration/3600:.1f} hours")
    
    print("\nPer-model results:")
    for r in results:
        status_icon = '✅' if r['status'] == 'success' else ('❌' if r['status'] == 'failed' else '⚠️')
        duration_str = f"{r['duration']/60:.1f}min" if r['duration'] > 0 else 'N/A'
        print(f"  {status_icon} {r['model']:35s} {r['status']:10s} {duration_str}")


if __name__ == '__main__':
    main()
