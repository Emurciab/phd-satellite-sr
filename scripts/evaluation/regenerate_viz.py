#!/usr/bin/env python3
"""
Quick script to regenerate visualizations from existing evaluation results.
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training' / 'models'))

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

from evaluate_models import ModelEvaluator, ResultVisualizer

def main():
    # Paths
    test_dir = PROJECT_ROOT / 'data' / 'sr_training' / '5band_v3' / 'combined' / 'mmap_test'
    output_dir = PROJECT_ROOT / 'results' / 'model_evaluation'
    
    # Load the latest evaluation results
    json_files = sorted(output_dir.glob('evaluation_*.json'))
    if not json_files:
        print("No evaluation results found!")
        return 1
    
    latest_json = json_files[-1]
    print(f"Loading results from: {latest_json}")
    
    with open(latest_json, 'r') as f:
        results = json.load(f)
    
    print(f"Found {len(results)} models")
    
    # Initialize evaluator (CPU mode to avoid GPU conflicts)
    evaluator = ModelEvaluator(
        test_dir=test_dir,
        output_dir=output_dir,
        device=torch.device('cpu')  # Use CPU to avoid GPU conflicts
    )
    
    # Load models in CPU mode
    evaluator.add_bilinear_baseline()
    
    config_file = PROJECT_ROOT / 'scripts' / 'evaluation' / 'evaluation_config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    for model_def in config['models']:
        if model_def['checkpoint'] is None:
            continue
        try:
            evaluator.add_model(model_def['name'], Path(model_def['checkpoint']))
            print(f"Loaded: {model_def['name']}")
        except Exception as e:
            print(f"Failed to load {model_def['name']}: {e}")
    
    # Evaluate on a small subset (just for visualization)
    print("\nEvaluating models on 10 samples for visualization...")
    evaluator.evaluate_all(max_samples=10)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = ResultVisualizer(evaluator)
    visualizer.generate_all_visualizations(n_samples=5)
    
    print("\nVisualization regeneration complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
