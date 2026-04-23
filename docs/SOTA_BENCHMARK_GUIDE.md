# SOTA Super-Resolution Benchmark — Step-by-Step Guide

This guide walks through comparing our proposed multi-output temporal SR model
against state-of-the-art (SOTA) SR algorithms using their **original pretrained
weights** (no retraining), following the inter-comparative evaluation methodology
from Aybar et al. (2024) / OpenSR-Test.

---

## Overview

| Model | Type | Scale | Source |
|---|---|---|---|
| **Proposed (Direct)** | Multi-output temporal, learned upsample | ×3.33 | Ours |
| **Proposed (Residual)** | Multi-output temporal, learned upsample | ×3.33 | Ours |
| Bilinear | Interpolation baseline | any | — |
| Bicubic | Interpolation baseline | any | — |
| HAN | SISR pixel-wise regression (DIV2K) | ×4 | Niu et al. 2020 |
| LDM | SISR generative diffusion (OpenImages) | ×4 | Rombach et al. 2022 |
| OpenSR-Model | Satellite SR diffusion (Sentinel-2) | ×4 | Aybar et al. 2024 |
| Swin2-MoSE | Satellite Transformer SR (Sen2Venus) | ×4 | Rossi et al. 2025 |
| SatlasSR | Satellite ESRGAN (S2-NAIP) | ×4 | Wolters et al. 2023 |

---

## Step 1 — Install Dependencies

```bash
# Core dependencies (should already be installed)
pip install torch torchvision numpy pandas matplotlib tqdm scikit-image

# HAN model (from super-image library)
pip install super-image

# LDM model (Latent Diffusion via diffusers)
pip install diffusers transformers accelerate

# OpenSR-Model (Sentinel-2 specialised)
pip install opensr-model

# HuggingFace Hub (for model downloads)
pip install huggingface_hub
```

## Step 2 — Clone External Model Repos (Optional)

These are only needed for SatlasSR and Swin2-MoSE:

```bash
cd /mnt/TOSHIBA_EXT1/PhD/SR
mkdir -p external

# SatlasSR (ESRGAN-based S2→NAIP)
git clone https://github.com/allenai/satlas-super-resolution external/satlas-super-resolution

# Swin2-MoSE (Transformer-based Sen2Venus)
git clone https://github.com/IMPLabUniPr/swin2-mose external/swin2-mose
```

Download their pretrained weights from the respective GitHub releases pages and
place them in the cloned directories.

## Step 3 — Quick Sanity Check (20 samples)

```bash
cd /mnt/TOSHIBA_EXT1/PhD/SR

python scripts/evaluation/benchmark_sota.py \
  --test_dir data/sr_training/all_stas_combined/mmap_test \
  --output_dir results/sota_benchmark \
  --max_samples 20 \
  --models bilinear bicubic proposed_direct proposed_residual han
```

This runs ~20 samples with the easiest-to-install models. Check the console output
for per-metric results. If HAN fails, install `super-image` first.

## Step 4 — Add LDM (Generative Baseline)

```bash
python scripts/evaluation/benchmark_sota.py \
  --test_dir data/sr_training/all_stas_combined/mmap_test \
  --output_dir results/sota_benchmark \
  --max_samples 20 \
  --models bilinear proposed_direct proposed_residual han ldm
```

**Note:** LDM is slow (~2 seconds/sample with 50 diffusion steps on GPU).
For testing, reduce steps inside the script (`num_inference_steps=20`).

## Step 5 — Add Satellite-Specific Models

```bash
# OpenSR-Model (after pip install opensr-model)
python scripts/evaluation/benchmark_sota.py \
  --max_samples 20 \
  --models bilinear proposed_direct proposed_residual han opensr

# SatlasSR (after cloning repo + downloading weights)
python scripts/evaluation/benchmark_sota.py \
  --max_samples 20 \
  --models bilinear proposed_direct proposed_residual han satlas
```

## Step 6 — Full Benchmark (All Samples)

Once all models load correctly:

```bash
python scripts/evaluation/benchmark_sota.py \
  --test_dir data/sr_training/all_stas_combined/mmap_test \
  --output_dir results/sota_benchmark \
  --n_viz 10
```

This evaluates **all available models** on **all 1367 test samples**, saves:
- Metrics JSON: `results/sota_benchmark/sota_benchmark_YYYYMMDD_HHMMSS.json`
- Metrics CSV: `results/sota_benchmark/sota_benchmark_YYYYMMDD_HHMMSS.csv`
- Bar chart: `results/sota_benchmark/sota_benchmark_YYYYMMDD_HHMMSS.png`
- Visual comparisons: `results/sota_benchmark/sota_visual_YYYYMMDD_HHMMSS/`

## Step 7 — Interpret Results

### What to look for

| Metric | Direction | Measures |
|--------|-----------|----------|
| PSNR ↑ | Higher better | Pixel-level fidelity |
| SSIM ↑ | Higher better | Structural similarity |
| SCC ↑ | Higher better | Spatial correlation coefficient |
| RMSE ↓ | Lower better | Pixel error magnitude |
| ERGAS ↓ | Lower better | Global spectral quality |
| SID ↓ | Lower better | Spectral information divergence |

### Expected outcome

Since SOTA models use **pretrained weights on different domains** (DIV2K, OpenImages,
S2-NAIP), while our model is **trained on the exact S2→Planet domain**, our model
should outperform SOTA on fidelity metrics (PSNR, SSIM, RMSE). This is expected
and valid — the comparison demonstrates the value of **domain-specific training**
and our **multi-output temporal architecture**.

The key arguments to make in your thesis:

1. **Domain gap matters**: Generic SR models (HAN, LDM) trained on natural images
   underperform on satellite data → justifies domain-specific approach
2. **Temporal information helps**: Our model uses 3 temporal frames; SISR models
   use only 1 → demonstrates temporal attention value
3. **Multi-resolution design**: Our model jointly processes 10m+20m bands; SOTA
   only handles RGB → shows architectural advantage
4. **Scale factor match**: Our ×3.33 vs SOTA ×4 → our model better handles the
   non-integer upsampling specific to S2→Planet

### How to report

Follow Aybar et al. (2024) framing: "We compare **complete SR frameworks**
evaluated with their original pretrained weights on our test dataset. This
methodology evaluates each framework's out-of-the-box applicability to the
S2→Planet super-resolution task."

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: super_image` | `pip install super-image` |
| `ModuleNotFoundError: diffusers` | `pip install diffusers transformers accelerate` |
| `ModuleNotFoundError: opensr_model` | `pip install opensr-model` |
| HAN produces artifacts | Expected for satellite reflectance (trained on natural images) |
| LDM too slow | Reduce steps: edit `num_inference_steps=20` in benchmark_sota.py |
| CUDA OOM | Use `--device cpu` or reduce `--max_samples` |
| Proposed model fails to load | Verify checkpoint path in `get_available_models()` |

---

## Adding More SOTA Models

To add a new model, create a class inheriting from `SOTAModelWrapper`:

```python
class MyNewModelWrapper(SOTAModelWrapper):
    def __init__(self, device):
        super().__init__('My Model Name', device)

    def load(self):
        # Load pretrained weights
        self.model = ...

    def predict_10m(self, sample, data_loader):
        # Get input: sample['lr_10m_mid'] is (77,77,3) BGR [0,1]
        # Return: (256, 256, 3) BGR [0,1]
        ...
```

Then register it in `get_available_models()`.
