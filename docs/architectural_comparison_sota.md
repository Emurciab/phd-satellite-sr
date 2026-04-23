# Architectural Comparison: Proposed Model vs. State-of-the-Art Super-Resolution Methods

This document provides a comprehensive architectural comparison between our proposed **Multi-Output Temporal SR with Learned Upscaling** (MO-TSR) model and state-of-the-art (SOTA) super-resolution methods spanning 2017–2025. Methods are organized into four categories: **(A)** CNN-based, **(B)** GAN-based, **(C)** Transformer-based, and **(D)** satellite/multi-frame-specific. Each entry is characterized on consistent dimensions to highlight the structural novelty of our approach.

---

## 1. Taxonomy of Compared Methods

| ID | Method | Year | Venue | Category | Domain |
|----|--------|------|-------|----------|--------|
| A1 | EDSR | 2017 | CVPRW | CNN | Natural |
| A2 | RCAN | 2018 | ECCV | CNN | Natural |
| A3 | DRLN | 2020 | TPAMI | CNN | Natural |
| A4 | RFDN | 2020 | AIM/ECCVW | CNN (lightweight) | Natural |
| B1 | SRGAN | 2017 | CVPR | GAN | Natural |
| B2 | ESRGAN | 2018 | ECCVW | GAN | Natural |
| B3 | Real-ESRGAN | 2021 | ICCVW | GAN | Natural |
| B4 | BSRGAN | 2021 | ICCV | GAN | Natural |
| B5 | CEDGAN | 2020 | IJGIS | GAN | Geospatial |
| C1 | SwinIR | 2021 | ICCVW | Transformer | Natural |
| C2 | HAT | 2023 | CVPR | Transformer | Natural |
| C3 | SRFormer | 2023 | ICCV | Transformer | Natural |
| C4 | DAT | 2023 | ICCV | Transformer | Natural |
| C5 | StableSR (DiffIR) | 2024 | IJCV | Diffusion | Natural |
| C6 | MambaIR | 2024 | ECCV | SSM/Mamba | Natural |
| D1 | HighRes-net | 2020 | — | CNN (MFSR) | Satellite |
| D2 | RAMS | 2020 | TGRS | CNN (MFSR) | Satellite |
| D3 | MISR-GRU | 2021 | RS | RNN | Satellite |
| — | **MO-TSR (Ours)** | **2026** | **—** | **Hybrid CNN-Attention** | **Satellite** |

---

## 2. Comprehensive Architectural Comparison

### 2.1 Backbone & Feature Extraction

| Method | Backbone Type | Feature Extractor | Depth | Base Channels | Multi-Scale Input |
|--------|---------------|-------------------|-------|---------------|-------------------|
| EDSR | Residual CNN | Residual blocks (no BN) | 32 res. blocks | 256 | ✗ |
| RCAN | Residual CNN | Residual-in-Residual + Channel Attention | 10 groups × 20 blocks | 64 | ✗ |
| DRLN | Residual CNN | Densely connected Laplacian blocks | 20 cascading blocks | 64 | ✗ |
| RFDN | Residual CNN | Residual feature distillation blocks | 4 RFDB | 50 | ✗ |
| SRGAN | Residual CNN | 16 residual blocks + BN | 16 res. blocks | 64 | ✗ |
| ESRGAN | RRDB | Residual-in-Residual Dense Blocks (no BN) | 23 RRDB | 64 | ✗ |
| Real-ESRGAN | RRDB | Same as ESRGAN + sinc-based degradation | 23 RRDB | 64 | ✗ |
| BSRGAN | RRDB | ESRGAN backbone + random degradation shuffle | 23 RRDB | 64 | ✗ |
| CEDGAN | U-Net style | Conditional encoder-decoder (encoder + decoder + skip) | ~8 conv layers | 64 | ✗ |
| SwinIR | Swin Transformer | RSTB (Residual Swin Transformer Blocks) | 6 RSTB × 6 STL | 180 | ✗ |
| HAT | Hybrid Transformer | RHAG (Residual Hybrid Attention Groups) | 6 RHAG × 6 HAB | 180 | ✗ |
| SRFormer | Transformer | Permuted Self-Attention blocks (RSTB variant) | 6 groups × 6 blocks | 180 | ✗ |
| DAT | Dual Aggregation Transformer | Alternating spatial/channel self-attention | 6 groups × 6 blocks | 180 | ✗ |
| StableSR | Diffusion U-Net | Pre-trained Stable Diffusion + time-aware encoder | U-Net (SD 2.1) | 320–1280 | ✗ |
| MambaIR | State-Space Model | Residual State-Space blocks (RSSB) | 6 RSSB × 6 VSSB | 180 | ✗ |
| HighRes-net | Recursive CNN | Shared recursive fusion encoder (pairwise folding) | ~6 conv layers | 64 | ✗ (single-res, multi-frame) |
| RAMS | CNN + attention | Residual Attention blocks + channel-concat frames | 12 res. blocks | 32 | ✗ (single-res, multi-frame) |
| **MO-TSR (Ours)** | **Hybrid CNN-Attention** | **Dual-branch EncoderBranch (7×7 stem + DepthwiseSep ResBlocks + CBAM)** | **4 ResBlocks/branch** | **64** | **✓ (multi-res: 10m + 20m; multi-frame: T=3)** |

### 2.2 Attention Mechanisms

| Method | Channel Attention | Spatial Attention | Self-Attention | Cross-Attention | Temporal Attention |
|--------|-------------------|-------------------|----------------|-----------------|-------------------|
| EDSR | ✗ | ✗ | ✗ | ✗ | ✗ |
| RCAN | ✓ (SE-style squeeze-excitation) | ✗ | ✗ | ✗ | ✗ |
| DRLN | ✓ (Laplacian attention) | ✗ | ✗ | ✗ | ✗ |
| RFDN | ✓ (ESA – Enhanced Spatial Attention) | ✓ | ✗ | ✗ | ✗ |
| SRGAN | ✗ | ✗ | ✗ | ✗ | ✗ |
| ESRGAN | ✗ | ✗ | ✗ | ✗ | ✗ |
| Real-ESRGAN | ✗ | ✗ | ✗ | ✗ | ✗ |
| BSRGAN | ✗ | ✗ | ✗ | ✗ | ✗ |
| CEDGAN | ✗ | ✗ | ✗ | ✗ | ✗ |
| SwinIR | ✗ | ✗ | ✓ (W-MSA/SW-MSA) | ✗ | ✗ |
| HAT | ✓ (CAB) | ✗ | ✓ (W-MSA/SW-MSA) | ✓ (Overlapping cross-attention) | ✗ |
| SRFormer | ✓ (permuted) | ✓ (permuted) | ✓ (PSA) | ✗ | ✗ |
| DAT | ✓ (inter-block) | ✓ (inter-block) | ✓ (spatial+channel alternating) | ✗ | ✗ |
| StableSR | ✗ (latent diffusion) | ✗ | ✓ (cross-attn with text encoder) | ✓ (CFW module) | ✗ |
| MambaIR | ✓ (via SSM gating) | ✓ (via SSM scanning) | ✗ (linear SSM) | ✗ | ✗ |
| HighRes-net | ✗ | ✗ | ✗ | ✗ | ✗ (implicit via recursion) |
| RAMS | ✓ (SE-style) | ✓ (spatial gates) | ✗ | ✗ | ✗ (concat frames) |
| **MO-TSR (Ours)** | **✓ (SE-style + CBAM)** | **✓ (CBAM 7×7 conv)** | **✗** | **✓ (Gated cross-resolution fusion)** | **✓ (TemporalCrossAttention + TemporalFusionBlock)** |

### 2.3 Upsampling Strategy

| Method | Upsampling Type | Mechanism | Scale Flexibility | Content-Aware |
|--------|----------------|-----------|-------------------|---------------|
| EDSR | Post-upsampling | PixelShuffle (sub-pixel conv) | Fixed per model | ✗ |
| RCAN | Post-upsampling | PixelShuffle | Fixed per model | ✗ |
| DRLN | Post-upsampling | PixelShuffle + Laplacian pyramid | Fixed per model | ✗ |
| RFDN | Post-upsampling | PixelShuffle | Fixed per model | ✗ |
| SRGAN | Post-upsampling | PixelShuffle | Fixed per model | ✗ |
| ESRGAN | Post-upsampling | PixelShuffle (nearest + conv) | Fixed per model | ✗ |
| Real-ESRGAN | Post-upsampling | PixelShuffle | Fixed per model | ✗ |
| BSRGAN | Post-upsampling | PixelShuffle | Fixed per model | ✗ |
| CEDGAN | Decoder | Transposed convolution (learned decoder) | Fixed per model | ✗ |
| SwinIR | Post-upsampling | PixelShuffle | Fixed per model | ✗ |
| HAT | Post-upsampling | PixelShuffle | Fixed per model | ✗ |
| SRFormer | Post-upsampling | PixelShuffle | Fixed per model | ✗ |
| DAT | Post-upsampling | PixelShuffle | Fixed per model | ✗ |
| StableSR | Latent decoding | VAE decoder (pre-trained) + progressive aggregation | Arbitrary (tiled) | ✗ |
| MambaIR | Post-upsampling | PixelShuffle | Fixed per model | ✗ |
| HighRes-net | Post-upsampling | Transposed conv | Fixed per model | ✗ |
| RAMS | Post-upsampling | PixelShuffle | Fixed per model | ✗ |
| **MO-TSR (Ours)** | **Progressive hybrid** | **PixelShuffle 2× → AdaptiveInterpolation (deformable 5×5 kernel with learned offsets)** | **Adaptive (non-integer scales)** | **✓ (content-aware offsets + weights)** |

### 2.4 Loss Functions

| Method | Pixel Loss | Perceptual Loss | Adversarial Loss | Spectral Loss | Gradient/Edge Loss | Temporal Loss |
|--------|------------|-----------------|------------------|---------------|-------------------|---------------|
| EDSR | L1 | ✗ | ✗ | ✗ | ✗ | ✗ |
| RCAN | L1 | ✗ | ✗ | ✗ | ✗ | ✗ |
| DRLN | L1 | ✗ | ✗ | ✗ | ✗ | ✗ |
| RFDN | L1 | ✗ | ✗ | ✗ | ✗ | ✗ |
| SRGAN | L2 (MSE) | ✓ (VGG) | ✓ (BCE) | ✗ | ✗ | ✗ |
| ESRGAN | L1 | ✓ (VGG) | ✓ (Relativistic) | ✗ | ✗ | ✗ |
| Real-ESRGAN | L1 | ✓ (VGG) | ✓ (U-Net discriminator) | ✗ | ✗ | ✗ |
| BSRGAN | L1 | ✓ (VGG) | ✓ (U-Net discriminator) | ✗ | ✗ | ✗ |
| CEDGAN | L1 + L2 | ✗ | ✓ (cGAN) | ✗ | ✗ | ✗ |
| SwinIR | L1 | ✗ (classical) / ✓ (real) | ✗ / ✓ (GAN variant) | ✗ | ✗ | ✗ |
| HAT | L1 | ✗ | ✗ | ✗ | ✗ | ✗ |
| SRFormer | L1 | ✗ | ✗ | ✗ | ✗ | ✗ |
| DAT | L1 | ✗ | ✗ | ✗ | ✗ | ✗ |
| StableSR | ✗ (diffusion) | ✓ (implicit in diffusion) | ✗ | ✗ | ✗ | ✗ |
| MambaIR | L1 | ✗ | ✗ | ✗ | ✗ | ✗ |
| HighRes-net | L1 | ✗ | ✗ | ✗ | ✗ | ✗ |
| RAMS | L1 | ✗ | ✗ | ✗ | ✗ | ✗ |
| **MO-TSR (Ours)** | **L1** | **✓ (VGG-16, scheduled warmup 0.01→0.1 over 30 ep.)** | **✗** | **✓ (Spectral Angle Mapper, α=0.137)** | **✓ (Local Gradient, α=0.01)** | **✓ (Cross-frame consistency, α=0.5)** |

### 2.5 Temporal & Multi-Frame Handling

| Method | Temporal Input | Temporal Mechanism | # Frames | Multi-Frame Fusion | Temporal Consistency |
|--------|----------------|-------------------|----------|--------------------|--------------------|
| EDSR | ✗ | — | 1 | — | — |
| RCAN | ✗ | — | 1 | — | — |
| DRLN | ✗ | — | 1 | — | — |
| SRGAN | ✗ | — | 1 | — | — |
| ESRGAN | ✗ | — | 1 | — | — |
| Real-ESRGAN | ✗ | — | 1 | — | — |
| BSRGAN | ✗ | — | 1 | — | — |
| CEDGAN | ✗ | — | 1 | — | — |
| SwinIR | ✗ | — | 1 | — | — |
| HAT | ✗ | — | 1 | — | — |
| SRFormer | ✗ | — | 1 | — | — |
| DAT | ✗ | — | 1 | — | — |
| StableSR | ✗ | — | 1 | — | — |
| MambaIR | ✗ | — | 1 | — | — |
| HighRes-net | ✓ | Recursive pairwise fusion | 9 (variable) | Learned recursive | ✗ (implicit) |
| RAMS | ✓ | Channel concatenation + attention | 9 (variable) | Attention-weighted | ✗ |
| MISR-GRU | ✓ | GRU recurrence | Variable | Recurrent hidden state | ✗ |
| **MO-TSR (Ours)** | **✓** | **TemporalCrossAttention (Q·K^T/√d) + TemporalFusionBlock (gated)** | **3** | **Attention-weighted + gated residual** | **✓ (explicit temporal loss)** |

### 2.6 Multi-Resolution / Multi-Output Capabilities

| Method | Multi-Resolution Input | Multi-Output | Cross-Resolution Fusion | Spectral Preservation |
|--------|------------------------|-------------|-------------------------|----------------------|
| EDSR | ✗ (single scale) | ✗ (single output) | ✗ | ✗ |
| RCAN | ✗ | ✗ | ✗ | ✗ |
| DRLN | ✗ | ✗ | ✗ | ✗ |
| SRGAN | ✗ | ✗ | ✗ | ✗ |
| ESRGAN | ✗ | ✗ | ✗ | ✗ |
| Real-ESRGAN | ✗ | ✗ | ✗ | ✗ |
| BSRGAN | ✗ | ✗ | ✗ | ✗ |
| CEDGAN | ✗ | ✗ | ✗ | ✗ |
| SwinIR | ✗ | ✗ | ✗ | ✗ |
| HAT | ✗ | ✗ | ✗ | ✗ |
| SRFormer | ✗ | ✗ | ✗ | ✗ |
| DAT | ✗ | ✗ | ✗ | ✗ |
| StableSR | ✗ | ✗ | ✗ | ✗ |
| MambaIR | ✗ | ✗ | ✗ | ✗ |
| HighRes-net | ✗ | ✗ | ✗ | ✗ |
| RAMS | ✗ | ✗ | ✗ | ✗ |
| **MO-TSR (Ours)** | **✓ (10m 3-ch + 20m 2-ch)** | **✓ (256×256 + 128×128)** | **✓ (Gated fusion: σ(W·concat)⊙f₁₀ + (1−gate)⊙f₂₀)** | **✓ (SpectralConsistencyModule + SAM loss)** |

---

## 3. Parameter Efficiency & Computational Profile

| Method | Parameters | Category | Scale | Input Type | Notes |
|--------|-----------|----------|-------|------------|-------|
| EDSR | ~43M | CNN | ×2/×3/×4 | Single RGB | Very deep; no attention |
| RCAN | ~16M | CNN | ×2/×3/×4 | Single RGB | Channel attention only |
| DRLN | ~34M | CNN | ×2/×3/×4 | Single RGB | Laplacian attention pyramid |
| RFDN | ~0.5M | CNN (lightweight) | ×4 | Single RGB | AIM 2020 challenge winner |
| SRGAN | ~1.5M (G) + 8M (D) | GAN | ×4 | Single RGB | First perceptual SR GAN |
| ESRGAN | ~16.7M (G) | GAN | ×4 | Single RGB | RRDB generator |
| Real-ESRGAN | ~16.7M (G) | GAN | ×4 | Single RGB | Practical degradation model |
| BSRGAN | ~16.7M (G) | GAN | ×4 | Single RGB | Blind SR with degradation shuffle |
| CEDGAN | ~2–5M (est.) | GAN | ×2–4 | Single spatial | U-Net cGAN for interpolation |
| SwinIR | ~11.8M | Transformer | ×2/×3/×4 | Single RGB | First Swin-based SR |
| HAT | ~20.8M (base) / ~40.8M (L) | Transformer | ×2/×3/×4 | Single RGB | +1dB over SwinIR (Urban100) |
| SRFormer | ~10.5M | Transformer | ×2/×3/×4 | Single RGB | PSA balances channel/spatial |
| DAT | ~11.2M / ~14.8M | Transformer | ×2/×3/×4 | Single RGB | Dual spatial/channel aggregation |
| StableSR | ~1.2B (SD) + ~50M (encoder) | Diffusion | Arbitrary | Single RGB | Leverages pretrained SD 2.1 |
| MambaIR | ~11.5M | SSM | ×2/×3/×4 | Single RGB | Linear complexity via Mamba |
| HighRes-net | ~1.5M | CNN (MFSR) | ×3 | Multi-frame grayscale | ESA PROBA-V challenge winner |
| RAMS | ~0.4M | CNN (MFSR) | ×3 | Multi-frame grayscale | Lightweight multi-frame |
| **MO-TSR (Ours)** | **~1.57M** | **Hybrid** | **~3.3× (non-integer)** | **Multi-frame, multi-resolution, multi-spectral** | **5-band input (3+2), dual output** |

---

## 4. Key Architectural Innovations: What Sets MO-TSR Apart

### 4.1 Unique Structural Features (not found in any single SOTA method)

| Feature | Description | Present In SOTA? |
|---------|-------------|-----------------|
| **Dual-resolution encoder** | Separate EncoderBranch for 10m (3-ch) and 20m (2-ch) bands; each with tailored stem and 4 ResBlocks + CBAM | ✗ — all SOTA use single-resolution input |
| **Gated cross-resolution fusion** | Learnable gating mechanism: $\mathbf{g} = \sigma(W[\mathbf{f}_{10} \| \mathbf{f}_{20}])$, output $= \mathbf{g} \odot \mathbf{f}_{10} + (1-\mathbf{g}) \odot \mathbf{f}_{20}$ | ✗ — no cross-resolution fusion in any listed method |
| **Temporal cross-attention** | Explicit Q·K^T/√d attention across T=3 temporal frames + gated temporal fusion block | Partially in HighRes-net (recursive, no attention), RAMS (concat, no cross-attn) |
| **Content-aware adaptive upsampling** | AdaptiveInterpolation: 5×5 deformable kernel with learned offsets (Tanh·scale bounded) + content-dependent weights | ✗ — all SOTA use fixed PixelShuffle or transposed conv |
| **Progressive non-integer upscaling** | PixelShuffle 2× → adaptive resize to arbitrary target (e.g., 77→256 = 3.32×) | ✗ — all SOTA restricted to integer scales (2×, 3×, 4×) |
| **Spectral consistency module** | Dedicated channel attention module preserving band relationships: AdaptiveAvgPool→Conv1×1→GELU→Conv1×1→Sigmoid | ✗ — no spectral-specific module in any listed method |
| **Multi-output heads** | Simultaneous generation of both HR-10m (256×256×3) and HR-20m (128×128×2) outputs | ✗ — all SOTA produce single output |
| **Compound domain-specific loss** | 5-component: L1 + SAM + gradient + temporal + scheduled perceptual | Partially (ESRGAN: L1+VGG+GAN = 3 terms; none use SAM or temporal) |
| **Scheduled perceptual warmup** | α_perceptual increases 0.01→0.1 over first 30 epochs to stabilize early training | ✗ — novel training strategy |

### 4.2 Shared Features with SOTA (backbone elements)

| Feature | SOTA Methods Using It | Our Implementation |
|---------|----------------------|-------------------|
| Residual learning | EDSR, RCAN, ESRGAN, SwinIR, HAT, etc. | Global residual + local residual blocks |
| Channel attention (SE-style) | RCAN, CBAM, HAT, RAMS | SE block within CBAM |
| Spatial attention | CBAM, DAT, SRFormer | CBAM 7×7 spatial attention |
| PixelShuffle | Nearly all SOTA | First stage (2× only) |
| Depthwise separable convolutions | MobileNet-derived; rare in SR | All ResBlocks use DepthwiseSep |
| GroupNorm | Rare in SR (most use BN or none) | All normalization via GroupNorm (8 groups) |
| Perceptual loss | SRGAN, ESRGAN, Real-ESRGAN | VGG-16 features (scheduled) |

---

## 5. Paradigm-Level Comparison

| Aspect | CNN-based (EDSR, RCAN) | GAN-based (ESRGAN, Real-ESRGAN) | Transformer-based (SwinIR, HAT) | Diffusion (StableSR) | SSM (MambaIR) | **MO-TSR (Ours)** |
|--------|------------------------|----------------------------------|--------------------------------|---------------------|---------------|-------------------|
| **Strengths** | Fast inference; stable training | Perceptual quality; sharp textures | Global context; SOTA on benchmarks | Diverse outputs; handles complex degradation | Linear complexity; long-range | **Domain-specialized; multi-temporal; spectral fidelity** |
| **Weaknesses** | Limited receptive field | Training instability; mode collapse risk | Quadratic complexity; large models | Very slow inference (~minutes); 1B+ params | Still maturing; limited SR evidence | Task-specific (satellite only) |
| **Typical params** | 16M–43M | 16M+ (G) | 11M–41M | ~1.2B | ~11M | **1.57M** |
| **Temporal** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Spectral** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Multi-res** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Inference speed** | Fast | Medium | Slow | Very slow | Fast | **Fast (1.57M)** |

---

## 6. Detailed Method Descriptions

### A. CNN-Based Methods

**EDSR** (Lim et al., 2017) — Enhanced Deep Super-Resolution. Removes batch normalization from residual blocks, enabling much deeper (32-block) networks with 256 base channels. Uses L1 loss only. ~43M parameters. Established the residual-learning paradigm for SR. No attention mechanisms.

**RCAN** (Zhang et al., 2018) — Residual Channel Attention Networks. Introduces channel attention via squeeze-and-excitation into a Residual-in-Residual structure (10 residual groups × 20 residual blocks). First to show channel attention significantly improves SR. ~16M parameters. No spatial or temporal attention.

**DRLN** (Anwar & Barnes, 2020) — Densely Residual Laplacian Network. Cascading residual-on-residual structure with dense connections and a novel Laplacian attention mechanism for learning inter- and intra-level feature dependencies. ~34M parameters. Addresses multi-scale feature fusion through Laplacian pyramids.

**RFDN** (Liu et al., 2020) — Residual Feature Distillation Network. Lightweight design (~0.5M params) using feature distillation connections. Winner of AIM 2020 Efficient SR challenge. Demonstrates that compact networks can approach heavyweight performance.

### B. GAN-Based Methods

**SRGAN** (Ledig et al., 2017) — Super-Resolution GAN. Pioneered perceptual loss (VGG feature matching) + adversarial training for SR. Generator uses 16 ResBlocks with BN; discriminator is a VGG-style classifier. First to prioritize perceptual quality over PSNR. ~1.5M generator params.

**ESRGAN** (Wang et al., 2018) — Enhanced SRGAN. Replaces ResBlocks with Residual-in-Residual Dense Blocks (RRDB), removes BN, uses relativistic discriminator and VGG perceptual loss on pre-activation features. ~16.7M generator params. Established the dominant GAN-SR architecture.

**Real-ESRGAN** (Wang et al., 2021) — Extends ESRGAN with a practical high-order degradation model (blur → resize → noise → JPEG, applied twice) and a U-Net discriminator. Designed for real-world images with unknown degradations. Same RRDB backbone (~16.7M).

**BSRGAN** (Zhang et al., 2021) — Blind SR via randomly shuffled degradation sequences (blur, downsampling, noise in random order). ESRGAN backbone with improved degradation modeling. Addresses the gap between synthetic and real-world degradation.

**CEDGAN** (Zhu et al., 2020) — Conditional Encoder-Decoder GAN. Originally proposed for spatial interpolation of geospatial data (including DEM). Uses a U-Net-style conditional generator with encoder-decoder skip connections and a PatchGAN discriminator. Captures spatial distribution patterns. Applied to remote sensing tasks but not specifically designed for spectral image SR. Estimated ~2–5M parameters.

### C. Transformer & Modern Methods

**SwinIR** (Liang et al., 2021) — First application of Swin Transformer to image restoration. Architecture: shallow feature extraction (3×3 conv) → 6 Residual Swin Transformer Blocks (RSTB) with shifted window attention → PixelShuffle reconstruction. Window size 8. ~11.8M params. +0.14–0.45dB over CNN SOTA while reducing parameters by up to 67%.

**HAT** (Chen et al., 2023) — Hybrid Attention Transformer. Combines channel attention blocks (CAB) with window self-attention within each Hybrid Attention Block (HAB), plus an Overlapping Cross-Attention (OCAB) module for inter-window information exchange. Uses same-task pre-training strategy. ~20.8M (base) to ~40.8M (HAT-L). +1dB over SwinIR on Urban100.

**SRFormer** (Zhou et al., 2023) — Introduces Permuted Self-Attention (PSA) that balances channel and spatial dimensions within attention computation, achieving large window benefits without quadratic cost. ~10.5M params. +0.46dB over SwinIR on Urban100 with fewer parameters.

**DAT** (Chen et al., 2023) — Dual Aggregation Transformer. Alternates between spatial-window and channel self-attention within each block, with inter-block aggregation for information exchange. Captures both spatial and channel dependencies. ~11.2M–14.8M params.

**StableSR** (Wang et al., 2024) — Leverages pre-trained Stable Diffusion 2.1 as a generative prior. Time-aware encoder injects LR information without modifying the pre-trained model. Controllable Feature Wrapping (CFW) module balances quality/fidelity. Progressive aggregation enables arbitrary-size inputs. ~1.2B inherited params. State-of-the-art perceptual quality but very slow inference.

**MambaIR** (Guo et al., 2024) — Applies Mamba (Selective State Space Models) to image restoration. Residual State Space Blocks (RSSB) with Vision State Space Blocks (VSSB) enable long-range dependency modeling with linear complexity. ~11.5M params. Competitive with Transformer-based methods at lower computational cost.

### D. Satellite / Multi-Frame Specific

**HighRes-net** (Deudon et al., 2020) — First deep learning approach to multi-frame SR for satellite imagery. Learns co-registration, fusion, and upsampling end-to-end. Recursive pairwise fusion of an arbitrary number of LR views through a shared encoder. Includes registration-at-the-loss via ShiftNet. Winner of ESA PROBA-V SR challenge. ~1.5M params. Operates on grayscale satellite imagery.

**RAMS** (Salvetti et al., 2020) — Residual Attention Multi-image SR. Lightweight (~0.4M params) multi-frame SR for satellite imagery using channel/spatial attention within residual blocks. Concatenates multiple frames along channel dimension before processing. Designed for PROBA-V ×3 SR task.

**MISR-GRU** (Various, 2021) — Multi-image SR with GRU-based temporal fusion. Processes frames sequentially through a recurrent unit, accumulating information across time steps. Limited by sequential processing bottleneck and lack of explicit attention.

---

## 7. Summary of Distinguishing Characteristics

Our MO-TSR model is architecturally unique across six dimensions that no existing SOTA method addresses simultaneously:

1. **Multi-temporal fusion via attention** — Whereas HighRes-net uses implicit recursive fusion and RAMS uses simple concatenation, MO-TSR employs explicit temporal cross-attention (Q·K^T/√d) followed by gated temporal fusion, enabling learned temporal weighting across T=3 satellite acquisitions.

2. **Multi-resolution dual-branch processing** — No existing method processes multiple native resolutions (10m and 20m) through separate encoder branches with subsequent cross-resolution fusion. All SOTA methods assume a single homogeneous input resolution.

3. **Multi-output generation** — MO-TSR jointly generates HR images at two resolution levels (256×256 and 128×128) from a shared representation, exploiting cross-resolution complementarity. All SOTA methods produce a single output.

4. **Content-aware non-integer upsampling** — The AdaptiveInterpolation module with deformable 5×5 kernels and learned offsets enables content-dependent resampling at non-integer scales (3.32×), unlike the fixed PixelShuffle-only approaches used universally in SOTA.

5. **Spectral preservation by design** — The SpectralConsistencyModule and SAM loss explicitly preserve inter-band spectral relationships, critical for satellite remote sensing but absent from all natural-image SOTA methods.

6. **Extreme parameter efficiency** — At 1.57M parameters, MO-TSR is 7–27× smaller than SOTA Transformer methods (SwinIR: 11.8M, HAT: 20.8M), 10–27× smaller than CNN heavyweights (RCAN: 16M, EDSR: 43M), and ~760× smaller than diffusion approaches (StableSR: ~1.2B), while handling a significantly more complex task (multi-temporal, multi-resolution, multi-output, multi-spectral).

---

*Document generated for PhD thesis: Satellite Image Super-Resolution*  
*Last updated: 2026-03-13*
