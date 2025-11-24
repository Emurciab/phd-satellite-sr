#!/bin/bash

set -e

echo "========================================================================"
echo "ORGANIZING ROOT DIRECTORY"
echo "========================================================================"
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p docs/organization
mkdir -p docs/analysis
mkdir -p scripts/utilities
mkdir -p scripts/visualization
mkdir -p tests/integration

# ============================================================================
# MOVE MARKDOWN DOCUMENTATION
# ============================================================================

echo ""
echo "📄 Moving markdown documentation to docs/..."

# Organization docs
echo "   Organization docs → docs/organization/"
mv WORKSPACE_ORGANIZATION_COMPLETE.md docs/organization/ 2>/dev/null && echo "      ✓ WORKSPACE_ORGANIZATION_COMPLETE.md"
mv REORGANIZATION_COMPLETE.md docs/organization/ 2>/dev/null && echo "      ✓ REORGANIZATION_COMPLETE.md"
mv REORGANIZATION_PLAN.md docs/organization/ 2>/dev/null && echo "      ✓ REORGANIZATION_PLAN.md"
mv REORGANIZATION_SUMMARY.md docs/organization/ 2>/dev/null && echo "      ✓ REORGANIZATION_SUMMARY.md"

# Bug/analysis docs
echo "   Analysis/bug docs → docs/analysis/"
mv BUG_FIX_LR_HR_DATE_MISMATCH.md docs/analysis/ 2>/dev/null && echo "      ✓ BUG_FIX_LR_HR_DATE_MISMATCH.md"
mv DATASET_GENERATION_ISSUE.md docs/analysis/ 2>/dev/null && echo "      ✓ DATASET_GENERATION_ISSUE.md"
mv NODATA_ANALYSIS.md docs/analysis/ 2>/dev/null && echo "      ✓ NODATA_ANALYSIS.md"
mv PIPELINE_FEATURES_CHECKLIST.md docs/analysis/ 2>/dev/null && echo "      ✓ PIPELINE_FEATURES_CHECKLIST.md"

# Main README stays in root
echo "   Main docs (keeping in root):"
echo "      ✓ README.md (project main)"
echo "      ✓ README_WORKSPACE.md (workspace info)"

# Remove ODT file
if [ -f "PIPELINE_FEATURES_CHECKLIST.md.odt" ]; then
    rm PIPELINE_FEATURES_CHECKLIST.md.odt
    echo "   ✓ Removed PIPELINE_FEATURES_CHECKLIST.md.odt (duplicate)"
fi

# ============================================================================
# MOVE SHELL SCRIPTS
# ============================================================================

echo ""
echo "🔧 Moving shell scripts to scripts/utilities/..."

mv organize_legacy.sh scripts/utilities/ 2>/dev/null && echo "      ✓ organize_legacy.sh"
mv reorganize_workspace.sh scripts/utilities/ 2>/dev/null && echo "      ✓ reorganize_workspace.sh"
mv REORGANIZATION.sh scripts/utilities/ 2>/dev/null && echo "      ✓ REORGANIZATION.sh"
mv monitor_generation.sh scripts/utilities/ 2>/dev/null && echo "      ✓ monitor_generation.sh"

# ============================================================================
# MOVE PYTHON SCRIPTS
# ============================================================================

echo ""
echo "🐍 Moving Python scripts..."

# Test scripts
echo "   Test scripts → tests/integration/"
mv test_complete_pipeline.py tests/integration/ 2>/dev/null && echo "      ✓ test_complete_pipeline.py"
mv test_harmonization_quick.py tests/integration/ 2>/dev/null && echo "      ✓ test_harmonization_quick.py"
mv test_harmonization_standalone.py tests/integration/ 2>/dev/null && echo "      ✓ test_harmonization_standalone.py"

# Visualization scripts
echo "   Visualization scripts → scripts/visualization/"
mv visualize_20samples.py scripts/visualization/ 2>/dev/null && echo "      ✓ visualize_20samples.py"

# Utility scripts
echo "   Utility scripts → scripts/utilities/"
mv generate_text_diagrams.py scripts/utilities/ 2>/dev/null && echo "      ✓ generate_text_diagrams.py"

# ============================================================================
# CREATE INDEX FILES
# ============================================================================

echo ""
echo "📋 Creating index files..."

# Create docs/README.md
cat > docs/README.md << 'EOFREADME'
# Documentation

## Directory Structure

- `organization/` - Workspace organization and cleanup documentation
- `analysis/` - Bug fixes, issues, and pipeline analysis

## Organization Documents

- [Workspace Organization Complete](organization/WORKSPACE_ORGANIZATION_COMPLETE.md) - Latest organization (Nov 23, 2024)
- [Reorganization Complete](organization/REORGANIZATION_COMPLETE.md) - Previous reorganization
- [Reorganization Plan](organization/REORGANIZATION_PLAN.md) - Planning documents
- [Reorganization Summary](organization/REORGANIZATION_SUMMARY.md) - Summary

## Analysis & Bug Fixes

- [LR-HR Date Mismatch Bug Fix](analysis/BUG_FIX_LR_HR_DATE_MISMATCH.md) - Critical bug fix (Nov 23, 2024)
- [Dataset Generation Issue](analysis/DATASET_GENERATION_ISSUE.md) - Generation issues
- [NoData Analysis](analysis/NODATA_ANALYSIS.md) - NoData value analysis
- [Pipeline Features Checklist](analysis/PIPELINE_FEATURES_CHECKLIST.md) - Feature tracking

## Main Documentation

See root directory for:
- [README.md](../README.md) - Project overview and setup
- [README_WORKSPACE.md](../README_WORKSPACE.md) - Workspace information
EOFREADME

echo "      ✓ docs/README.md"

# Create scripts/visualization/README.md
cat > scripts/visualization/README.md << 'EOFREADME'
# Visualization Scripts

Scripts for visualizing training samples and datasets.

## Files

- `visualize_20samples.py` - Quick visualization of 20 samples from a dataset

## Main Visualization Scripts

See `scripts/preprocessing/` for the primary visualization scripts:
- `visualize_sequences.py` - Individual sequence visualization (all 3 frames)
- `visualize_sequences_grid.py` - Grid overview of multiple samples

## Usage

```bash
# Visualize 20 samples
python scripts/visualization/visualize_20samples.py \
    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl

# Full visualization suite (from preprocessing)
python scripts/preprocessing/visualize_sequences.py \
    --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \
    --output-dir visualizations/sequences \
    --num-samples 10
```
EOFREADME

echo "      ✓ scripts/visualization/README.md"

# Create scripts/utilities/README.md
cat > scripts/utilities/README.md << 'EOFREADME'
# Utility Scripts

Helper scripts for workspace organization, monitoring, and utilities.

## Organization Scripts

- `organize_legacy.sh` - Move buggy/old data to legacy folders
- `reorganize_workspace.sh` - General workspace reorganization
- `REORGANIZATION.sh` - Earlier reorganization script

## Monitoring Scripts

- `monitor_generation.sh` - Monitor dataset generation progress

## Utility Scripts

- `generate_text_diagrams.py` - Generate text-based diagrams for documentation

## Usage

```bash
# Organize legacy data
bash scripts/utilities/organize_legacy.sh

# Monitor generation
bash scripts/utilities/monitor_generation.sh

# Generate diagrams
python scripts/utilities/generate_text_diagrams.py
```
EOFREADME

echo "      ✓ scripts/utilities/README.md"

# Create tests/integration/README.md
cat > tests/integration/README.md << 'EOFREADME'
# Integration Tests

End-to-end tests for the complete pipeline.

## Files

- `test_complete_pipeline.py` - Full pipeline test from GEE to processed samples
- `test_harmonization_quick.py` - Quick harmonization test
- `test_harmonization_standalone.py` - Standalone harmonization test

## Usage

```bash
# Run complete pipeline test
cd /media/eamurciab/TOSHIBA_EXT1/PhD/SR
conda activate tf_gpu
python tests/integration/test_complete_pipeline.py

# Quick harmonization test
python tests/integration/test_harmonization_quick.py
```

## Unit Tests

See `tests/` directory root for unit tests:
- Unit test files for specific components
- Test fixtures and utilities
EOFREADME

echo "      ✓ tests/integration/README.md"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "========================================================================"
echo "✅ ROOT DIRECTORY ORGANIZATION COMPLETE"
echo "========================================================================"
echo ""
echo "📊 Summary:"
echo ""
echo "   📄 Markdown docs moved to docs/"
echo "      ├── organization/ (4 files)"
echo "      ├── analysis/ (4 files)"
echo "      └── README.md (index)"
echo ""
echo "   🔧 Shell scripts moved to scripts/utilities/ (4 files)"
echo ""
echo "   🐍 Python scripts organized:"
echo "      ├── tests/integration/ (3 test files)"
echo "      ├── scripts/visualization/ (1 file)"
echo "      └── scripts/utilities/ (1 file)"
echo ""
echo "   📋 Index files created (4 READMEs)"
echo ""
echo "   🗑️  Removed:"
echo "      └── PIPELINE_FEATURES_CHECKLIST.md.odt (duplicate)"
echo ""
echo "✨ Root directory now contains only:"
echo "   • README.md (main project)"
echo "   • README_WORKSPACE.md"
echo "   • Core directories (.git, Auth, data, scripts, tests, etc.)"
echo ""
echo "========================================================================"

