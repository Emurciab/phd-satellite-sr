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
