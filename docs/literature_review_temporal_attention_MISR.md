# Literature Review: Cross-Attention and Temporal Fusion Mechanisms for Multi-Frame Satellite Super-Resolution

## 1. Overview of Your Architecture (MO-TSR)

Your model, **MO-TSR** (Multi-Output Temporal Super-Resolution with Learned Upscaling), processes **T=3 temporal frames** of dual-resolution satellite imagery (Planet 10m RGB + Sentinel-2 20m bands) and produces super-resolved outputs for both resolution branches simultaneously. Key architectural components:

| Component | Mechanism | Details |
|---|---|---|
| **TemporalCrossAttention** | Pooled global attention + local spatial mixing | AdaptiveAvgPool → Q/K/V linear projections → softmax attention O(T²); local Conv2d(T×C→C); gated per-frame combination |
| **TemporalFusionBlock** | Residual refinement wrapper | Conv refinement + learnable γ scaling around TemporalCrossAttention |
| **CrossResolutionFusion** | Gated dual-branch exchange | Per-frame gated fusion: upsample/downsample + sigmoid gate between 10m and 20m branches |
| **AdaptiveInterpolation** | Learned upscaling | Per-pixel K×K=25 interpolation kernel weights + deformable offsets + residual refinement |
| **Loss** | Multi-component | L1 (α=1.0) + SAM (α=0.137) + Gradient (α=0.01) + Temporal Consistency (α=0.5) + Scheduled Perceptual (0.01→0.1 over 30 epochs) |

**Key design choices**: 1.57M parameters, fixed T=3 input frames, cross-sensor (Planet→S2), multi-output heads, CNN-based backbone (not Transformer).

---

## 2. TAMRF-SITS — The Reference Article

**Michel, J. & Inglada, J. (2025).** *"Temporal Attention Multi-Resolution Fusion of Satellite Image Time-Series, Applied to Landsat-8/9 and Sentinel-2: All Bands, Any Time, at Best Spatial Resolution."* Remote Sensing of Environment, 115159.  
DOI: [10.1016/j.rse.2025.115159](https://doi.org/10.1016/j.rse.2025.115159)

### 2.1 Architecture Summary

TAMRF follows an **encoder–decoder** scheme combining CNNs for spatial processing with a **Transformer for temporal processing**:

- **Spatial Encoders**: Separate date-wise CNN encoders (SRResNet/RRDB-based) for HR (Sentinel-2, 10 bands) and LR (Landsat-8/9, 8 bands). The LR encoder includes bicubic upsampling to align spatial dimensions. Output: F=64 features per pixel per date.

- **Temporal & Modality Encoder**: Per-pixel, the date-wise features from both sensors are concatenated into a single temporal sequence. Each token receives:
  - **Temporal positional encoding** (sinusoidal, 64 features) via the standard Vaswani (2017) scheme
  - **Learnable sensor tokens** (8 features each, one per sensor)
  - Result: a sequence of (N_LR + N_HR) tokens × 136 features
  - Processed by a **3-layer multi-head self-attention Transformer encoder** (4 heads, 136-dim tokens, 256-dim FFN)

- **Temporal Decoder**: Query-based cross-attention. Target acquisition dates are converted to tokens via temporal positional encoding + a learnable placeholder. These queries attend to the latent SITS via **multi-head cross-attention** (keys/values from encoder output).

- **Spatial Decoder**: Two Conv layers to output all predicted bands (18 channels = 10 S2 + 8 Landsat bands).

**Training**: Masked Auto-Encoder (MAE) self-supervised strategy with cloud-contrastive loss + LPIPS-based spatial term (via a learned linear regression projection between sensor spectral bands).

**Model size**: 2.3M parameters.

### 2.2 Key Design Differences with MO-TSR

| Aspect | TAMRF-SITS | MO-TSR (Yours) |
|---|---|---|
| **Temporal mechanism** | Full Transformer self-attention (multi-head, pixel-wise, O(N²) in seq length) | Pooled global attention O(T²) + local Conv mixing; gated combination |
| **Spatial backbone** | RRDB (SRResNet lineage) | Custom CNN with SE modules |
| **Temporal input length** | Variable-length (any N_LR + N_HR dates) | Fixed T=3 frames |
| **Sensors handled** | Multi-sensor via learnable sensor tokens (Landsat+S2) | Cross-sensor via separate encoder branches (Planet→S2) |
| **Upsampling** | Bicubic (supports non-integer ratios) | **AdaptiveInterpolation** with learned per-pixel K×K kernels + deformable offsets |
| **Decoder temporal** | Cross-attention with date queries (can predict **any** target date, including unseen) | No temporal query mechanism; outputs at input frame dates only |
| **Training paradigm** | Self-supervised (MAE) — no HR labels needed | Fully supervised with paired HR ground-truth |
| **Loss function** | Reconstruction (L1) + cloud contrastive + LPIPS spatial | L1 + SAM + Gradient + Temporal Consistency + Scheduled Perceptual |
| **Multi-resolution** | Predicts all bands of both sensors at best (10m) resolution | Dedicated dual branches with **CrossResolutionFusion** (gated exchange, preserves both resolutions) |
| **Parameters** | 2.3M | 1.57M |
| **Scale invariance** | Avoided (uses LPIPS cross-sensor term instead) | Not assumed (uses real cross-sensor pairs) |
| **Cloud handling** | Implicit via contrastive loss (no masks needed) | Explicit via quality filtering in data pipeline |

### 2.3 Similarities

1. **Temporal attention as core mechanism**: Both architectures are fundamentally built around temporal attention to fuse information across multiple acquisitions.
2. **Separate spatial encoding per resolution/sensor**: Both use dedicated spatial encoders before temporal fusion.
3. **Multi-resolution awareness**: Both handle bands at different spatial resolutions (10m/20m for MO-TSR; 10m/30m for TAMRF).
4. **Perceptual loss component**: Both incorporate perceptual similarity losses (LPIPS in TAMRF; scheduled VGG-based perceptual in MO-TSR).
5. **Relatively compact models**: Both < 3M parameters, emphasizing efficiency.

### 2.4 Your Innovations vs. TAMRF

1. **Pooled global attention** is much more memory-efficient than full pixel-wise Transformer attention. TAMRF processes each pixel independently as a temporal sequence, achieving O(N²) in the number of dates but only for each pixel. Your approach pools spatially first, reducing the attention computation dramatically.

2. **Gated cross-resolution fusion** is a novel mechanism not present in TAMRF. TAMRF projects everything to the same resolution and predicts jointly. Your per-frame gated exchange explicitly preserves the information integrity of each resolution branch while enabling cross-resolution feature transfer.

3. **AdaptiveInterpolation with deformable offsets** is a learned upsampling strategy significantly more powerful than TAMRF's fixed bicubic interpolation. The per-pixel kernel weights + deformable offsets allow spatially-varying, content-aware upscaling.

4. **Spectral Angle Mapper (SAM) loss** directly optimizes spectral fidelity, which is critical for remote sensing applications. TAMRF does not use a spectral loss.

5. **Temporal consistency loss** explicitly penalizes temporal flickering between consecutive frames, which TAMRF does not address (it processes each date independently once latent).

6. **Scheduled perceptual loss** (warmup from 0.01→0.1) prevents early training instability — a practical innovation not seen in TAMRF.

---

## 3. Other Related Multi-Frame Satellite SR Approaches

### 3.1 HighRes-net (Deudon et al., 2020)

**"HighRes-net: Recursive Fusion for Multi-Frame Super-Resolution of Satellite Imagery."** arXiv:2002.06460. Winner of ESA's Proba-V Super-Resolution challenge.

- **Fusion**: Recursive pairwise fusion of frames via a learned operator (not attention-based). Frames are fused sequentially: F(F(f₁, f₂), f₃)...  
- **Registration**: Implicit co-registration via reference-frame channel + ShiftNet for loss alignment.
- **Upsampling**: Standard learned upsampling.
- **Loss**: L1 with registered loss alignment.
- **No explicit temporal attention or positional encoding.**

| vs. MO-TSR | Comparison |
|---|---|
| Temporal fusion | Recursive pairwise CNN vs. your **global attention + local mixing** |
| Order invariance | Order-dependent (sequential fusion) vs. your attention-based (approximately permutation-equivariant) |
| Registration | Explicit ShiftNet vs. not needed (your data are coregistered via GEE) |
| Multi-resolution | Single-resolution (Proba-V only) vs. your dual-branch with CrossResolutionFusion |
| Loss | Single L1 vs. your 5-component composite loss |

**Your innovation**: Global attention allows all frames to influence each other simultaneously rather than through a sequential bottleneck.

### 3.2 DeepSUM (Molini et al., 2020)

**"DeepSUM: Deep Neural Network for Super-Resolution of Unregistered Multitemporal Images."** IEEE TGRS, 58(5), 3644-3656.

- **Architecture**: Registration module + feature extraction + 2D spatial attention maps per frame + fusion via weighted averaging.
- **Attention**: Spatial attention maps (channel squeeze → spatial weights per frame). Not temporal attention in the Q/K/V sense.
- **Registration**: Explicit learned affine + non-rigid registration.
- **Dataset**: Proba-V (multi-temporal, single sensor).

| vs. MO-TSR | Comparison |
|---|---|
| Attention type | Spatial attention masks (per-frame quality weighting) vs. your **cross-temporal Q/K/V attention** |
| Registration | Explicit learned registration module vs. pre-registered data |
| Temporal modeling | No temporal positional awareness vs. your temporal attention captures inter-frame dynamics |
| Cross-sensor | Single-sensor (Proba-V) vs. your cross-sensor (Planet→S2) |
| Multi-resolution | Single vs. dual-branch |

**Your innovation**: True cross-temporal attention (Q/K/V mechanism) is fundamentally more expressive than DeepSUM's per-frame spatial attention masks, as it can model inter-frame relationships directly.

### 3.3 RAMS (Salvetti et al., 2020)

**"Multi-Image Super Resolution of Remote Sensing Images Using Residual Attention Deep Neural Networks."** Remote Sensing, 12(14), 2207.

- **Architecture**: Recursive attention across multiple LR frames with multi-scale feature extraction.
- **Attention**: Channel attention (SE-like blocks) applied within a recursive multi-frame processing scheme.
- **Multi-scale**: Features extracted at multiple scales and fused.
- **Dataset**: Proba-V.

| vs. MO-TSR | Comparison |
|---|---|
| Attention | Channel attention (SE blocks) for feature recalibration | Your SE modules + cross-temporal attention (richer) |
| Multi-scale | Multiple receptive fields via dilated convs | Your CrossResolutionFusion between actual multi-res branches |
| Temporal fusion | Recursive attention-weighted aggregation | Your explicit temporal cross-attention with global+local branches |
| Cross-sensor | No | Yes (Planet→S2) |

**Your innovation**: You use SE modules similarly for channel recalibration, but add temporal cross-attention on top, directly modeling inter-frame dependencies rather than just weighting frames.

### 3.4 Okabayashi et al. (2024)

**"Cross-Sensor Super-Resolution of Irregularly Sampled Sentinel-2 Time Series."** CVPR-W EarthVision 2024.

- **Key contribution**: Extended HighRes-net and SRDiff for temporal series with **temporal positional encoding** for irregular sampling.
- **Architecture**: Adapts HighRes-net's recursive fusion and SRDiff's diffusion-based SR with date-aware conditioning.
- **Cross-sensor**: S2 → SPOT-6 (4×).
- **Dataset**: BreizhSR (new, Brittany region).

| vs. MO-TSR | Comparison |
|---|---|
| Temporal encoding | **Sinusoidal temporal positional encoding** for irregular dates | Your approach uses fixed T=3 contiguous frames without explicit date encoding |
| Fusion | Inherits recursive (HighRes-net) | Your explicit attention mechanism |
| Cross-sensor | Yes (S2→SPOT-6) | Yes (Planet→S2) |
| Attention | No self/cross-attention | Your global + local temporal cross-attention |

**Your innovation**: While Okabayashi et al. pioneered temporal positional encoding for irregular satellite MISR, they do not use attention-based temporal fusion — they merely condition existing architectures with date information. Your attention mechanism directly models inter-frame feature relationships.

### 3.5 HyDA-Net (Ibrahim et al., 2025)

**"HyDA-Net: A Hybrid Dense Attention Network for Remote Sensing Multi-Image Super-Resolution."** IEEE JSTARS, 2025.

- **Architecture**: Hybrid dense connections + attention mechanisms for multi-frame SR.
- **Attention**: Dense attention blocks combining spatial and channel attention.
- **Multi-frame**: Processes variable number of LR inputs.
- **Cross-sensor**: Sentinel-2 focused.

| vs. MO-TSR | Comparison |
|---|---|
| Dense connections | DenseNet-style feature reuse | Your residual connections + SE modules |
| Attention | Spatial + channel attention within dense blocks | Your discrete **temporal cross-attention** between frames |
| Multi-output | Single output | Your dual 10m + 20m outputs |
| Temporal modeling | Implicit via feature concatenation | Explicit temporal attention with global weighting |

### 3.6 U-TILISE (Stucker et al., 2023)

**"U-TILISE: A Sequence-to-Sequence Model for Cloud Removal in Optical Satellite Time Series."** IEEE TGRS, 61, 1-16.

- **Architecture**: Hierarchical convolutional spatial encoder + **temporal Transformer** for sequence-to-sequence reconstruction.
- **Purpose**: Cloud removal / gap-filling (not SR per se, but processes temporal satellite SITS).
- **Temporal**: Transformer along temporal dimension (similar concept to TAMRF).
- **Resolution**: Operates at native Sentinel-2 resolution (no super-resolution).

| vs. MO-TSR | Comparison |
|---|---|
| Temporal mechanism | Transformer along time (pixel-wise) | Your cross-attention with pooled global + local mixing |
| Purpose | Cloud removal / gap-filling | Super-resolution with temporal fusion |
| Spatial resolution change | None (10m → 10m) | Upscaling (10m → finer, 20m → 10m) |
| Multi-resolution | Single | Dual-branch with gated exchange |

---

## 4. Thematic Comparison Table

| Feature | MO-TSR (Yours) | TAMRF | HighRes-net | DeepSUM | RAMS | Okabayashi+ | HyDA-Net | U-TILISE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Temporal attention (Q/K/V)** | ✓ (pooled global) | ✓ (full Transformer) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (Transformer) |
| **Cross-frame interaction** | ✓ (global + local) | ✓ (self-attention) | ✓ (recursive) | ✓ (spatial attn) | ✓ (recursive) | ✓ (recursive) | ✓ (dense) | ✓ (self-attention) |
| **Multi-resolution input** | ✓ (10m + 20m) | ✓ (10m + 30m) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Cross-resolution fusion** | ✓ (gated exchange) | Projection to HR | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Cross-sensor** | ✓ (Planet→S2) | ✓ (Landsat↔S2) | ✗ | ✗ | ✗ | ✓ (S2→SPOT) | ✓ (S2) | ✗ |
| **Temporal positional encoding** | ✗ (fixed T=3) | ✓ (sinusoidal) | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **Variable-length inputs** | ✗ (fixed T=3) | ✓ (any N) | Variable | Variable | Variable | Variable | Variable | Variable |
| **Learned upsampling** | ✓ (AdaptiveInterp + deformable) | ✗ (bicubic) | ✓ | ✓ | ✓ | ✓ | ✓ | N/A |
| **SAM / spectral loss** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Temporal consistency loss** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Self-supervised** | ✗ (supervised) | ✓ (MAE) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Multi-output (dual-res)** | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Parameters** | 1.57M | 2.3M | ~1M | ~1.5M | ~1M | ~1M | N/A | N/A |

---

## 5. Summary of Your Innovations

When contrasted against the surveyed literature, MO-TSR introduces several novel contributions:

### 5.1 Unique to MO-TSR (not found in any surveyed method)

1. **Gated Cross-Resolution Fusion**: No other method performs per-frame bidirectional gated feature exchange between two resolution branches. TAMRF projects all bands to a single resolution; all other methods operate at a single resolution. Your approach preserves each branch's information fidelity while enabling mutual enhancement.

2. **AdaptiveInterpolation with Deformable Offsets**: The combination of learned per-pixel K×K (5×5) interpolation weights with learned deformable offset positions is unique. TAMRF uses fixed bicubic; others use PixelShuffle or sub-pixel convolution. Your approach allows content-adaptive, spatially-varying upscaling.

3. **SAM loss for spectral fidelity**: No other multi-frame satellite SR method uses spectral angle as a loss component. This directly optimizes the angular similarity in spectral space, critical for remote sensing applications where spectral integrity is paramount.

4. **Temporal consistency loss**: Explicitly penalizing differences between consecutive frame predictions is not found in TAMRF or any other surveyed method.

5. **Scheduled perceptual loss**: Gradually increasing the perceptual loss weight during training (0.01→0.1 over 30 epochs) is a practical training stabilization technique not found elsewhere.

### 5.2 Shared with TAMRF only

1. **Temporal attention applied to satellite MISR**: Both MO-TSR and TAMRF are among the few methods to use attention-based temporal fusion (Q/K/V mechanism) for satellite image super-resolution. However, the implementations differ fundamentally — TAMRF uses full pixel-wise Transformer attention, while MO-TSR uses pooled global attention.

2. **Multi-resolution input processing with dedicated encoders**: Both handle multi-resolution satellite bands via separate spatial encoders.

3. **Multi-output prediction**: Both predict bands at multiple resolutions/for multiple sensors.

### 5.3 Key Architectural Trade-offs

| Trade-off | MO-TSR | TAMRF |
|---|---|---|
| **Generality vs. efficiency** | Fixed T=3, optimized CNN → fast inference | Variable-length Transformer → flexible but heavier |
| **Supervised vs. SSL** | Paired data required | Self-supervised (no HR labels) |
| **Temporal extrapolation** | Cannot predict unseen dates | Can predict any query date |
| **Memory footprint** | Efficient (pooled attention) | Pixel-wise Transformer (heavier for large images) |
| **Spectral fidelity** | Explicit SAM loss | No direct spectral loss |
| **Upscaling quality** | Learned content-adaptive | Fixed bicubic |

---

## 6. Positioning Statement

MO-TSR occupies a unique position in the landscape: it is one of very few methods that combines **explicit temporal cross-attention** with **multi-resolution cross-sensor processing** for satellite SR, while remaining computationally efficient through pooled attention. Unlike TAMRF, which pursues maximal generality (any sensor, any date, any band) at the cost of requiring full Transformer processing, MO-TSR targets a focused pipeline (Planet→S2, T=3 frames) with innovations in upscaling (AdaptiveInterpolation), spectral fidelity (SAM loss), and temporal stability (consistency loss) that are not addressed by any other method in the literature.

The closest competitor, TAMRF-SITS, excels in versatility but sacrifices upscaling quality (bicubic) and spectral fidelity optimization. Earlier methods (HighRes-net, DeepSUM, RAMS) lack temporal attention entirely. Okabayashi et al. introduced temporal positional encoding for irregular MISR but did not adopt attention-based fusion. MO-TSR bridges these gaps with a compact, efficient architecture that directly targets the key quality dimensions for remote sensing SR: spatial detail (learned upscaling), spectral accuracy (SAM), and temporal coherence (consistency loss).

---

## References

1. Michel, J. & Inglada, J. (2025). Temporal Attention Multi-Resolution Fusion of Satellite Image Time-Series. *Remote Sensing of Environment*, 115159.
2. Deudon, M. et al. (2020). HighRes-net: Recursive Fusion for Multi-Frame Super-Resolution of Satellite Imagery. arXiv:2002.06460.
3. Molini, A.B. et al. (2020). DeepSUM: Deep Neural Network for Super-Resolution of Unregistered Multitemporal Images. *IEEE TGRS*, 58(5), 3644-3656.
4. Salvetti, F. et al. (2020). Multi-Image Super Resolution of Remote Sensing Images Using Residual Attention Deep Neural Networks. *Remote Sensing*, 12(14), 2207.
5. Okabayashi, A. et al. (2024). Cross-Sensor Super-Resolution of Irregularly Sampled Sentinel-2 Time Series. *CVPR-W EarthVision 2024*.
6. Ibrahim, M.R. et al. (2025). HyDA-Net: A Hybrid Dense Attention Network for Remote Sensing Multi-Image Super-Resolution. *IEEE JSTARS*.
7. Stucker, C. et al. (2023). U-TILISE: A Sequence-to-Sequence Model for Cloud Removal in Optical Satellite Time Series. *IEEE TGRS*, 61, 1-16.
8. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
