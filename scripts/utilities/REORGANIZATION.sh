#!/bin/bash

# Documentation files
mv *.md docs/ 2>/dev/null

# Core pipeline scripts
mv numpy_efficient_pipeline.py scripts/preprocessing/ 2>/dev/null
mv sr_efficient_pipeline.py scripts/preprocessing/ 2>/dev/null
mv streaming_pipeline.py scripts/preprocessing/ 2>/dev/null
mv Preprocessing.py scripts/preprocessing/ 2>/dev/null
mv efficient_data_handler.py scripts/preprocessing/ 2>/dev/null
mv gee_export_helper.py scripts/preprocessing/ 2>/dev/null

# Download/chunking scripts
mv chunked_download*.py scripts/preprocessing/ 2>/dev/null

# Dataset creation scripts
mv create_chunked_dataset.py scripts/preprocessing/ 2>/dev/null
mv create_fixed_dataset.py scripts/preprocessing/ 2>/dev/null

# Training scripts
mv train_*.py scripts/training/ 2>/dev/null
mv Model.py scripts/training/ 2>/dev/null
mv ImprovedModel.py scripts/training/ 2>/dev/null

# Validation scripts
mv validate_*.py scripts/validation/ 2>/dev/null
mv analyze_fixed_dataset.py scripts/validation/ 2>/dev/null

# Testing scripts
mv test_*.py scripts/testing/ 2>/dev/null
mv quick_check.py scripts/testing/ 2>/dev/null
mv demo_training_integration.py scripts/testing/ 2>/dev/null
mv working_gee_example.py scripts/testing/ 2>/dev/null

# Utility/analysis scripts
mv *_analysis.py scripts/utilities/ 2>/dev/null
mv calculate_*.py scripts/utilities/ 2>/dev/null
mv check_*.py scripts/utilities/ 2>/dev/null
mv debug_*.py scripts/utilities/ 2>/dev/null
mv diagnose_*.py scripts/utilities/ 2>/dev/null
mv gee_integration_status.py scripts/utilities/ 2>/dev/null
mv monitor_*.py scripts/utilities/ 2>/dev/null
mv quick_start_guide.py scripts/utilities/ 2>/dev/null

# Cache directories
mv *_cache cache/ 2>/dev/null
mv *cache/ cache/ 2>/dev/null

# Dataset directories
mv demo_dataset data/datasets/ 2>/dev/null
mv fixed_dataset data/datasets/ 2>/dev/null
mv quick_test data/datasets/ 2>/dev/null

# Model checkpoints
mv checkpoints data/models/ 2>/dev/null
mv fixed_checkpoints data/models/ 2>/dev/null

# Logs
mv *.log logs_archive/ 2>/dev/null
mv *.txt logs_archive/ 2>/dev/null
mv logs/* logs_archive/ 2>/dev/null
rmdir logs 2>/dev/null

# Setup scripts
mkdir -p scripts/setup 2>/dev/null
mv setup_*.sh scripts/setup/ 2>/dev/null
mv monitor_test.sh scripts/setup/ 2>/dev/null

echo "Reorganization complete!"
