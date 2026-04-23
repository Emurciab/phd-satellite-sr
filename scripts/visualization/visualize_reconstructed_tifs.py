#!/usr/bin/env python3
"""
Visualize the reconstructed TIF outputs from the SR model.

Produces two figure panels:
  Figure 1 – 10 m branch:  Sentinel-2 input (bilinear upscaled) | EMO-TSR SR | Planet reference
  Figure 2 – 20 m branch:  Sentinel-2 input (bilinear upscaled) | EMO-TSR SR | Planet reference
             Red-edge (band 0) on top row, NIR (band 1) on bottom row.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.transform import resize
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
RECON_DIR = Path("/mnt/TOSHIBA_EXT1/PhD/SR/outputs/reconstructed")
OUT_DIR    = RECON_DIR          # save figures next to the TIFs

PREFIX = "STA1E_2023_seq0"

files = {
    "sr_10m"  : RECON_DIR / f"{PREFIX}_10m.tif",
    "sr_20m"  : RECON_DIR / f"{PREFIX}_20m.tif",
    "pl_10m"  : RECON_DIR / f"{PREFIX}_planet_10m.tif",
    "pl_20m"  : RECON_DIR / f"{PREFIX}_planet_20m.tif",
    "s2_10m"  : RECON_DIR / f"{PREFIX}_sentinel_10m.tif",
    "s2_20m"  : RECON_DIR / f"{PREFIX}_sentinel_20m.tif",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def read_tif(path):
    """Read a GeoTIFF as (C, H, W) float32 array.

    Uses GDAL's ReadRaster() → raw bytes → numpy, avoiding the gdal_array
    extension that breaks under numpy 2.x.
    """
    from osgeo import gdal
    ds = gdal.Open(str(path))
    cols, rows = ds.RasterXSize, ds.RasterYSize
    bands = []
    for i in range(1, ds.RasterCount + 1):
        raw = ds.GetRasterBand(i).ReadRaster(
            0, 0, cols, rows, buf_type=gdal.GDT_Float32
        )
        bands.append(np.frombuffer(raw, dtype=np.float32).reshape(rows, cols))
    ds = None
    arr = np.stack(bands, axis=0)   # (C, H, W)
    arr[arr == 0.0] = np.nan        # treat nodata=0 as NaN
    return arr


def percentile_stretch(img, lo=2, hi=98):
    """Per-band percentile stretch; returns float in [0, 1]."""
    out = np.empty_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        band = img[i]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            out[i] = 0.0
            continue
        p_lo, p_hi = np.percentile(valid, [lo, hi])
        if p_hi == p_lo:
            out[i] = 0.0
        else:
            out[i] = np.clip((band - p_lo) / (p_hi - p_lo), 0, 1)
    return out


def to_rgb(arr):
    """(3, H, W) float [0,1] → (H, W, 3) uint8 for imshow."""
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.moveaxis(np.clip(arr * 255, 0, 255).astype(np.uint8), 0, -1)


def resize_to(arr, target_shape):
    """(C, H, W) → resized to target (H, W) using nearest, keeps same C."""
    C = arr.shape[0]
    out = np.empty((C, *target_shape), dtype=arr.dtype)
    for c in range(C):
        out[c] = resize(arr[c], target_shape, order=1,
                        mode='reflect', anti_aliasing=True,
                        preserve_range=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – 10 m branch  (RGB: bands B, G, R  → display as R, G, B)
# ══════════════════════════════════════════════════════════════════════════════
print("Reading 10 m branch …")
sr_10  = read_tif(files["sr_10m"])   # (3, H, W)
pl_10  = read_tif(files["pl_10m"])   # (3, H', W')
s2_10  = read_tif(files["s2_10m"])   # (3, H, W)  — same grid as SR but Sentinel res

# Band order in the TIF: Blue(0), Green(1), Red(2) → RGB display wants [2,1,0]
BAND_ORDER_10M = [2, 1, 0]   # R, G, B

target_h, target_w = sr_10.shape[1], sr_10.shape[2]

# Sentinel was saved at the same pixel grid as SR (already upscaled by GEE/rasterio)
s2_10_disp = percentile_stretch(s2_10[BAND_ORDER_10M])

# Planet is at native 3 m (smaller tile → resize up for display comparison)
pl_10_rs   = resize_to(pl_10, (target_h, target_w))
pl_10_disp = percentile_stretch(pl_10_rs[BAND_ORDER_10M])

sr_10_disp = percentile_stretch(sr_10[BAND_ORDER_10M])

# ── plot ──────────────────────────────────────────────────────────────────────
labels_10 = ["Sentinel-2 input\n(10 m → bilinear up)",
             "EMO-TSR SR output\n(3 m)", "Planet reference\n(3 m)"]
images_10 = [s2_10_disp, sr_10_disp, pl_10_disp]

fig1, axes = plt.subplots(1, 3, figsize=(18, 7),
                          gridspec_kw={"wspace": 0.03})
fig1.suptitle(f"10 m Branch – RGB composite  ({PREFIX})", fontsize=14, y=1.01)

for ax, img, label in zip(axes, images_10, labels_10):
    ax.imshow(to_rgb(img), interpolation="nearest")
    ax.set_title(label, fontsize=10)
    ax.axis("off")

fig1.tight_layout()
out_10m = OUT_DIR / f"{PREFIX}_vis_10m_branch.png"
fig1.savefig(out_10m, dpi=150, bbox_inches="tight")
print(f"  Saved → {out_10m}")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – 20 m branch  (band 0 = red-edge, band 1 = NIR)
# ══════════════════════════════════════════════════════════════════════════════
print("Reading 20 m branch …")
sr_20  = read_tif(files["sr_20m"])   # (2, H, W)
pl_20  = read_tif(files["pl_20m"])   # (2, H', W')
s2_20  = read_tif(files["s2_20m"])   # (2, H, W)

target_h20, target_w20 = sr_20.shape[1], sr_20.shape[2]
pl_20_rs  = resize_to(pl_20, (target_h20, target_w20))

BAND_NAMES_20M = ["Red-Edge (B5)", "NIR (B8A)"]
COLORMAPS       = ["RdYlGn",        "YlGn"]

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10),
                            gridspec_kw={"wspace": 0.04, "hspace": 0.12})
fig2.suptitle(f"20 m Branch – Red-Edge & NIR  ({PREFIX})", fontsize=14, y=1.01)

col_labels = ["Sentinel-2 input\n(20 m → bilinear up)",
              "EMO-TSR SR output\n(6 m)", "Planet reference\n(6 m)"]

for b_idx, (bname, cmap) in enumerate(zip(BAND_NAMES_20M, COLORMAPS)):
    srcs = [
        percentile_stretch(s2_20[[b_idx]])[0],
        percentile_stretch(sr_20[[b_idx]])[0],
        percentile_stretch(pl_20_rs[[b_idx]])[0],
    ]
    for c_idx, (ax, img) in enumerate(zip(axes2[b_idx], srcs)):
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        if b_idx == 0:
            ax.set_title(col_labels[c_idx], fontsize=10)
        ax.set_ylabel(bname, fontsize=9, labelpad=4)
        ax.axis("off")
    fig2.colorbar(im, ax=axes2[b_idx, -1], fraction=0.03, pad=0.02,
                  label="Scaled reflectance")

fig2.tight_layout()
out_20m = OUT_DIR / f"{PREFIX}_vis_20m_branch.png"
fig2.savefig(out_20m, dpi=150, bbox_inches="tight")
print(f"  Saved → {out_20m}")
plt.show()

print("Done.")
