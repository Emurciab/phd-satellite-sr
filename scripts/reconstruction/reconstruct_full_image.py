#!/usr/bin/env python3
"""
Reconstruct full AOI images from tiled SR patches.

Takes a dataset of tiled TrainingSamples (with geographic tile_bounds), runs SR
inference on each tile, and stitches the outputs into a georeferenced full-AOI
image (GeoTIFF).

Supports two stitching modes:
  1. Simple (no overlap): tiles placed directly at their geographic positions.
  2. Overlap + Gaussian blend: tiles re-inferred with overlapping stride and
     blended using Gaussian weights to eliminate seam artifacts.

Usage:
    # Simple reconstruction (no overlap)
    python -m scripts.reconstruction.reconstruct_full_image \
        --dataset data/datasets/sta1e_2023_5band/dataset_5band.pkl \
        --checkpoint checkpoints/multires_multioutput_schedPer_t0.5_adaptive_direct_learnedUps_full/best_model.pt \
        --sequence-id 0 \
        --output outputs/reconstructed/STA1E_2023_seq0.tif

    # With overlap blending
    python -m scripts.reconstruction.reconstruct_full_image \
        --dataset data/datasets/sta1e_2023_5band/dataset_5band.pkl \
        --checkpoint checkpoints/multires_multioutput_schedPer_t0.5_adaptive_direct_learnedUps_full/best_model.pt \
        --sequence-id 0 \
        --overlap 32 \
        --output outputs/reconstructed/STA1E_2023_seq0_blended.tif
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
import types
import dataclasses
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch

try:
    from osgeo import gdal, osr
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


# ============================================================================
# Pickle compatibility — register fake modules so old .pkl files load
# ============================================================================

@dataclass
class TrainingSample5Band:
    """Dummy for unpickling 5-band dataset samples."""
    pass

@dataclass
class TrainingSample:
    """Dummy for unpickling raw dataset samples."""
    pass

@dataclass
class FrameQualityStats:
    """Dummy for unpickling frame quality."""
    pass

def _setup_unpickle_modules():
    # Modules where TrainingSample5Band may have been defined when pkl was created
    for mod_name in ['__main__', 'convert_cache_to_5band',
                     'scripts.preprocessing.convert_cache_to_5band',
                     'scripts.preprocessing.convert_dataset_to_5band',
                     'convert_dataset_to_5band',
                     'download_inference_data',
                     'scripts.preprocessing.download_inference_data']:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
        else:
            mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = mod
        if not hasattr(mod, 'TrainingSample5Band'):
            mod.TrainingSample5Band = TrainingSample5Band

    # Modules where TrainingSample / FrameQualityStats may have been defined
    for mod_name in ['__main__',
                     'generate_sta2_grid_parallel', 'scripts.preprocessing.generate_sta2_grid_parallel',
                     'generate_sta56_grid_parallel', 'scripts.preprocessing.generate_sta56_grid_parallel',
                     'generate_sta1e_grid_parallel', 'scripts.preprocessing.generate_sta1e_grid_parallel',
                     'generate_sta1w_grid_parallel', 'scripts.preprocessing.generate_sta1w_grid_parallel',
                     'generate_sta34_grid_parallel', 'scripts.preprocessing.generate_sta34_grid_parallel',
                     'download_inference_data', 'scripts.preprocessing.download_inference_data',
                     'scripts.preprocessing.numpy_efficient_pipeline', 'numpy_efficient_pipeline']:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
        else:
            mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = mod
        if not hasattr(mod, 'TrainingSample'):
            mod.TrainingSample = TrainingSample
        if not hasattr(mod, 'FrameQualityStats'):
            mod.FrameQualityStats = FrameQualityStats


# ============================================================================
# Dataset loading
# ============================================================================

def load_samples(dataset_path: Path) -> List:
    """Load TrainingSample objects from a .pkl dataset (single file or chunked)."""
    _setup_unpickle_modules()
    dataset_path = Path(dataset_path)

    if dataset_path.is_file():
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get('samples', data.get('data', []))
        return [data]

    # Directory — try chunked then single-file
    samples = []
    chunks = sorted(dataset_path.glob('chunk_*.pkl'))
    if chunks:
        for cf in chunks:
            with open(cf, 'rb') as f:
                samples.extend(pickle.load(f))
        return samples

    for pf in sorted(dataset_path.glob('*.pkl')):
        with open(pf, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, list):
            samples.extend(data)
        elif isinstance(data, dict):
            samples.extend(data.get('samples', data.get('data', [])))
    return samples


def get_sample_arrays(sample) -> Dict[str, Optional[np.ndarray]]:
    """Extract LR/HR arrays from a sample regardless of attribute naming convention.

    Returns dict with keys: lr_10m, lr_20m, hr_10m, hr_20m (each (T,H,W,C) or None).
    """
    def _get(obj, *attrs):
        for a in attrs:
            v = getattr(obj, a, None)
            if v is not None:
                return v
        return None

    lr_10m = _get(sample, 'lr_sequence', 'LR_10m')
    lr_20m = _get(sample, 'lr_20m_sequence', 'LR_20m')
    # 5-band format stores matching-resolution arrays separately
    hr_10m = _get(sample, 'hr_10m_sequence', 'HR_10m')
    hr_20m = _get(sample, 'hr_20m_sequence', 'HR_20m')
    return {'lr_10m': lr_10m, 'lr_20m': lr_20m, 'hr_10m': hr_10m, 'hr_20m': hr_20m}


def get_tile_bounds(sample) -> Optional[List[float]]:
    """Extract geographic [min_lon, min_lat, max_lon, max_lat] from a sample."""
    tb = getattr(sample, 'tile_bounds', None)
    if tb is None:
        return None
    if isinstance(tb, dict):
        return tb.get('bounds', None)
    if isinstance(tb, (list, tuple)) and len(tb) == 4:
        return list(tb)
    return None


# ============================================================================
# Model loading & inference (thin wrappers over evaluate_models.py)
# ============================================================================

def load_sr_model(checkpoint_path: Path, device: torch.device):
    """Load SR model from checkpoint. Returns (model, checkpoint_dict)."""
    # Add training variant dirs to sys.path so model classes can be imported
    project_root = Path(__file__).resolve().parents[2]
    variant_dirs = [
        project_root / 'scripts' / 'training' / 'variants' / 'multioutput_schedper',
        project_root / 'scripts' / 'training',
        project_root / 'legacy',
    ]
    for d in variant_dirs:
        d_str = str(d)
        if d_str not in sys.path:
            sys.path.insert(0, d_str)

    from scripts.evaluation.evaluate_models import load_model
    return load_model(checkpoint_path, device)


def run_sr_inference(model, lr_10m: np.ndarray, lr_20m: Optional[np.ndarray],
                     device: torch.device,
                     target_size_10m: Tuple[int, int] = (256, 256),
                     target_size_20m: Tuple[int, int] = (128, 128),
                     ) -> Dict[str, np.ndarray]:
    """Run SR on a single tile.

    Args:
        lr_10m: (T, H, W, C) float32 [0, 1]
        lr_20m: (T, H, W, C) float32 [0, 1] or None
        target_size_10m: output spatial dims for 10m branch
        target_size_20m: output spatial dims for 20m branch

    Returns:
        dict with 'sr_10m' (H, W, C) and optionally 'sr_20m' (H, W, C), float32 [0,1]
    """
    # (T, H, W, C) → (1, T, C, H, W)
    lr_10m_t = torch.from_numpy(lr_10m).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    if lr_20m is not None:
        lr_20m_t = torch.from_numpy(lr_20m).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    else:
        # Create zeros with expected shape if 20m data is missing
        lr_20m_t = torch.zeros(1, lr_10m.shape[0], 2,
                               lr_10m.shape[1] // 2, lr_10m.shape[2] // 2,
                               device=device)

    with torch.no_grad():
        try:
            sr_10m, sr_20m = model(lr_10m_t, lr_20m_t,
                                   target_size_10m=target_size_10m,
                                   target_size_20m=target_size_20m)
        except TypeError:
            sr_10m, sr_20m = model(lr_10m_t, lr_20m_t)

        # Multi-temporal → middle frame
        if sr_10m.dim() == 5:
            mid = sr_10m.shape[1] // 2
            sr_10m = sr_10m[:, mid]
            sr_20m = sr_20m[:, mid]

        sr_10m_np = sr_10m.cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, C)
        sr_20m_np = sr_20m.cpu().numpy()[0].transpose(1, 2, 0)

    return {'sr_10m': sr_10m_np, 'sr_20m': sr_20m_np}


# ============================================================================
# Tile grid / canvas geometry
# ============================================================================

def compute_canvas_geometry(tile_bounds_list: List[List[float]],
                            hr_pixel_size: float,
                            ) -> Dict:
    """Compute the output canvas extent and dimensions.

    Args:
        tile_bounds_list: list of [min_lon, min_lat, max_lon, max_lat] per tile.
        hr_pixel_size: GSD of HR output in degrees (tile_size_deg / tile_px).

    Returns:
        dict with min_lon, max_lon, min_lat, max_lat, width_px, height_px, pixel_size.
    """
    all_min_lon = min(b[0] for b in tile_bounds_list)
    all_min_lat = min(b[1] for b in tile_bounds_list)
    all_max_lon = max(b[2] for b in tile_bounds_list)
    all_max_lat = max(b[3] for b in tile_bounds_list)

    width_px = int(round((all_max_lon - all_min_lon) / hr_pixel_size))
    height_px = int(round((all_max_lat - all_min_lat) / hr_pixel_size))

    return {
        'min_lon': all_min_lon, 'max_lon': all_max_lon,
        'min_lat': all_min_lat, 'max_lat': all_max_lat,
        'width_px': width_px, 'height_px': height_px,
        'pixel_size': hr_pixel_size,
    }


def tile_pixel_offset(bounds: List[float], canvas: Dict) -> Tuple[int, int]:
    """Compute (row, col) pixel offset for a tile on the canvas.

    Row 0 = top (max_lat), col 0 = left (min_lon).
    """
    col = int(round((bounds[0] - canvas['min_lon']) / canvas['pixel_size']))
    row = int(round((canvas['max_lat'] - bounds[3]) / canvas['pixel_size']))
    return row, col


def tile_pixel_extent(bounds: List[float], canvas: Dict) -> Tuple[int, int]:
    """Compute the pixel width and height this tile should occupy on the canvas."""
    w = int(round((bounds[2] - bounds[0]) / canvas['pixel_size']))
    h = int(round((bounds[3] - bounds[1]) / canvas['pixel_size']))
    return h, w


# ============================================================================
# Gaussian blending weights
# ============================================================================

def make_gaussian_weight(h: int, w: int, sigma_frac: float = 0.4) -> np.ndarray:
    """Create a 2D Gaussian weight map peaking at the tile center.

    Args:
        h, w: tile dimensions in pixels.
        sigma_frac: sigma as fraction of half-size (0.4 → smooth falloff).

    Returns:
        (H, W) float32 weight array, peak = 1.0.
    """
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32)
    x = np.arange(w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')

    sy = max(cy * sigma_frac, 1.0)
    sx = max(cx * sigma_frac, 1.0)

    weight = np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    return weight


# ============================================================================
# GeoTIFF output
# ============================================================================

def save_geotiff(filepath: Path, array: np.ndarray, canvas: Dict,
                 band_names: Optional[List[str]] = None,
                 nodata: float = 0.0):
    """Save a (H, W, C) array as a GeoTIFF with EPSG:4326 georeferencing.

    Falls back to raw .npy if GDAL is unavailable.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not HAS_GDAL:
        npy_path = filepath.with_suffix('.npy')
        np.save(npy_path, array)
        # Save georef info as sidecar
        meta = {
            'min_lon': canvas['min_lon'], 'max_lat': canvas['max_lat'],
            'pixel_size': canvas['pixel_size'],
            'width_px': canvas['width_px'], 'height_px': canvas['height_px'],
            'crs': 'EPSG:4326',
        }
        import json
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  GDAL not available — saved as {npy_path} + sidecar .json")
        return

    h, w = array.shape[:2]
    n_bands = array.shape[2] if array.ndim == 3 else 1

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(filepath), w, h, n_bands, gdal.GDT_Float32,
                       options=['COMPRESS=LZW', 'TILED=YES'])

    # Geotransform: (top-left-x, pixel-width, 0, top-left-y, 0, -pixel-height)
    gt = (canvas['min_lon'], canvas['pixel_size'], 0,
          canvas['max_lat'], 0, -canvas['pixel_size'])
    ds.SetGeoTransform(gt)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

    for b in range(n_bands):
        band_data = array[:, :, b] if array.ndim == 3 else array
        band_data = np.ascontiguousarray(band_data, dtype=np.float32)
        rb = ds.GetRasterBand(b + 1)
        # Use WriteRaster with raw bytes to avoid gdal_array / numpy ABI issues
        rb.WriteRaster(0, 0, w, h, band_data.tobytes(), w, h, gdal.GDT_Float32)
        rb.SetNoDataValue(nodata)
        if band_names and b < len(band_names):
            rb.SetDescription(band_names[b])

    ds.FlushCache()
    ds = None
    print(f"  Saved GeoTIFF: {filepath} ({w}x{h}, {n_bands} bands)")


# ============================================================================
# Source image mosaicking (PlanetScope + Sentinel-2, no model needed)
# ============================================================================

def mosaic_source_images(samples: List,
                         sequence_id: int,
                         ) -> Dict[str, Tuple[np.ndarray, Dict]]:
    """Mosaic the raw PlanetScope (LR) and Sentinel-2 (HR) source tiles.

    Extracts the middle temporal frame from each tile and places it on a
    geographic canvas at the tile's native resolution. Produces four mosaics:
      - planet_10m:   PlanetScope 3-band (B2,B3,B4) at ~3m
      - planet_20m:   PlanetScope 2-band (B5,B8A)   at ~5m
      - sentinel_10m: Sentinel-2  3-band (B2,B3,B4) at 10m
      - sentinel_20m: Sentinel-2  2-band (B5,B8A)   at 20m

    Returns:
        dict mapping output name to (array (H,W,C) float32, canvas dict).
    """
    seq_samples = [s for s in samples if getattr(s, 'sequence_id', -1) == sequence_id]
    if not seq_samples:
        raise ValueError(f"No samples found for sequence_id={sequence_id}")

    # Collect valid tile data
    tile_data = []
    for s in seq_samples:
        bounds = get_tile_bounds(s)
        if bounds is None:
            continue
        arrays = get_sample_arrays(s)
        if arrays['lr_10m'] is None:
            continue
        tile_data.append({'bounds': bounds, 'arrays': arrays})

    if not tile_data:
        raise ValueError("No valid tiles found")

    print(f"  Source mosaic — {len(tile_data)} tiles")

    first = tile_data[0]['arrays']
    # Geographic tile extent (same for all tiles)
    fb = tile_data[0]['bounds']
    tile_deg_lon = fb[2] - fb[0]
    tile_deg_lat = fb[3] - fb[1]

    all_bounds = [td['bounds'] for td in tile_data]

    # --- Build one canvas per (source, resolution) ---
    configs = {}  # name → {tile_shape, n_bands, array_key, pixel_size, canvas}

    # PlanetScope 10m-equiv bands
    lr10_shape = first['lr_10m'].shape  # (T, H, W, C)
    ps_10m = tile_deg_lon / lr10_shape[2]  # pixel size from tile width
    ps_10m_lat = tile_deg_lat / lr10_shape[1]
    ps_10m_avg = (ps_10m + ps_10m_lat) / 2.0
    configs['planet_10m'] = {
        'array_key': 'lr_10m', 'n_bands': lr10_shape[3],
        'pixel_size': ps_10m_avg,
        'canvas': compute_canvas_geometry(all_bounds, ps_10m_avg),
    }

    # PlanetScope 20m-equiv bands
    if first['lr_20m'] is not None:
        lr20_shape = first['lr_20m'].shape
        ps_20m = tile_deg_lon / lr20_shape[2]
        ps_20m_lat = tile_deg_lat / lr20_shape[1]
        ps_20m_avg = (ps_20m + ps_20m_lat) / 2.0
        configs['planet_20m'] = {
            'array_key': 'lr_20m', 'n_bands': lr20_shape[3],
            'pixel_size': ps_20m_avg,
            'canvas': compute_canvas_geometry(all_bounds, ps_20m_avg),
        }

    # Sentinel-2 10m bands
    if first['hr_10m'] is not None:
        hr10_shape = first['hr_10m'].shape
        s2_10m = tile_deg_lon / hr10_shape[2]
        s2_10m_lat = tile_deg_lat / hr10_shape[1]
        s2_10m_avg = (s2_10m + s2_10m_lat) / 2.0
        configs['sentinel_10m'] = {
            'array_key': 'hr_10m', 'n_bands': hr10_shape[3],
            'pixel_size': s2_10m_avg,
            'canvas': compute_canvas_geometry(all_bounds, s2_10m_avg),
        }

    # Sentinel-2 20m bands
    if first['hr_20m'] is not None:
        hr20_shape = first['hr_20m'].shape
        s2_20m = tile_deg_lon / hr20_shape[2]
        s2_20m_lat = tile_deg_lat / hr20_shape[1]
        s2_20m_avg = (s2_20m + s2_20m_lat) / 2.0
        configs['sentinel_20m'] = {
            'array_key': 'hr_20m', 'n_bands': hr20_shape[3],
            'pixel_size': s2_20m_avg,
            'canvas': compute_canvas_geometry(all_bounds, s2_20m_avg),
        }

    for name, cfg in configs.items():
        c = cfg['canvas']
        print(f"    {name}: {c['width_px']}x{c['height_px']} px  (pixel={cfg['pixel_size']:.8f}°)")

    # Allocate canvases
    canvases = {}
    weights = {}
    for name, cfg in configs.items():
        c = cfg['canvas']
        canvases[name] = np.zeros((c['height_px'], c['width_px'], cfg['n_bands']), dtype=np.float64)
        weights[name] = np.zeros((c['height_px'], c['width_px']), dtype=np.float64)

    # Place middle frame of each tile
    for idx, td in enumerate(tile_data):
        bounds = td['bounds']
        arrays = td['arrays']

        for name, cfg in configs.items():
            arr = arrays[cfg['array_key']]
            if arr is None:
                continue

            # Middle temporal frame → (H, W, C)
            mid = arr.shape[0] // 2
            tile_img = arr[mid]

            canvas = cfg['canvas']
            row, col = tile_pixel_offset(bounds, canvas)
            exp_h, exp_w = tile_pixel_extent(bounds, canvas)

            place_h = min(tile_img.shape[0], exp_h, canvas['height_px'] - row)
            place_w = min(tile_img.shape[1], exp_w, canvas['width_px'] - col)
            if place_h <= 0 or place_w <= 0:
                continue

            tile_img = tile_img[:place_h, :place_w]
            w = np.ones((place_h, place_w), dtype=np.float32)

            canvases[name][row:row + place_h, col:col + place_w] += (
                tile_img.astype(np.float64) * w[:, :, np.newaxis]
            )
            weights[name][row:row + place_h, col:col + place_w] += w

    # Normalize
    results = {}
    for name, cfg in configs.items():
        valid = weights[name] > 0
        canvases[name][valid] /= weights[name][valid, np.newaxis]
        canvases[name] = np.clip(canvases[name], 0, 1).astype(np.float32)
        coverage = np.count_nonzero(valid) / valid.size * 100
        print(f"    {name} coverage: {coverage:.1f}%")
        results[name] = (canvases[name], cfg['canvas'])

    return results


# ============================================================================
# Core SR reconstruction
# ============================================================================

def reconstruct_aoi(samples: List,
                    model: torch.nn.Module,
                    device: torch.device,
                    sequence_id: int,
                    overlap_px: int = 0,
                    target_size_10m: Tuple[int, int] = (256, 256),
                    target_size_20m: Tuple[int, int] = (128, 128),
                    ) -> Dict[str, Tuple[np.ndarray, Dict]]:
    """Reconstruct full-AOI SR images for a given temporal sequence.

    Args:
        samples: list of TrainingSample objects from a single region.
        model: loaded SR model in eval mode.
        device: torch device.
        sequence_id: which temporal sequence to reconstruct.
        overlap_px: if >0, apply Gaussian-weighted blending in overlap regions.
            In this mode overlap_px specifies how many HR pixels of overlap to
            use between adjacent tiles.  The tile stride on the canvas becomes
            (tile_px - overlap_px).
        target_size_10m: model output spatial size for 10m branch.
        target_size_20m: model output spatial size for 20m branch.

    Returns:
        dict mapping output name ('sr_10m', 'sr_20m') to (array, canvas) tuples.
        array is (H, W, C) float32 [0, 1], canvas is the geometry dict.
    """
    # Filter to requested sequence
    seq_samples = [s for s in samples if getattr(s, 'sequence_id', -1) == sequence_id]
    if not seq_samples:
        raise ValueError(f"No samples found for sequence_id={sequence_id}")

    # Extract bounds and validate
    tile_data = []
    for s in seq_samples:
        bounds = get_tile_bounds(s)
        if bounds is None:
            print(f"  WARNING: sample tile_id={getattr(s, 'tile_id', '?')} has no tile_bounds, skipping")
            continue
        arrays = get_sample_arrays(s)
        if arrays['lr_10m'] is None:
            print(f"  WARNING: sample tile_id={getattr(s, 'tile_id', '?')} has no LR data, skipping")
            continue
        tile_data.append({'bounds': bounds, 'arrays': arrays, 'sample': s})

    if not tile_data:
        raise ValueError("No valid tiles with both bounds and LR data found")

    print(f"  Tiles to process: {len(tile_data)}")

    # Determine pixel size from first tile's bounds and expected HR output
    first_bounds = tile_data[0]['bounds']
    tile_deg_lon = first_bounds[2] - first_bounds[0]
    tile_deg_lat = first_bounds[3] - first_bounds[1]
    tile_h, tile_w = target_size_10m
    hr_pixel_size_lon = tile_deg_lon / tile_w
    hr_pixel_size_lat = tile_deg_lat / tile_h
    # Use average — should be nearly identical for small tiles
    hr_pixel_size = (hr_pixel_size_lon + hr_pixel_size_lat) / 2.0

    all_bounds = [td['bounds'] for td in tile_data]
    canvas_10m = compute_canvas_geometry(all_bounds, hr_pixel_size)

    # 20m canvas: half the resolution
    canvas_20m = compute_canvas_geometry(
        all_bounds,
        hr_pixel_size * (target_size_10m[0] / target_size_20m[0])
    )

    print(f"  Canvas 10m: {canvas_10m['width_px']}x{canvas_10m['height_px']} px")
    print(f"  Canvas 20m: {canvas_20m['width_px']}x{canvas_20m['height_px']} px")
    print(f"  HR pixel size: {hr_pixel_size:.8f}°")

    # Determine number of bands from first inference
    first_arrays = tile_data[0]['arrays']
    n_bands_10m = first_arrays['lr_10m'].shape[-1]  # C from (T,H,W,C)
    n_bands_20m = first_arrays['lr_20m'].shape[-1] if first_arrays['lr_20m'] is not None else 2

    # Allocate output canvases + weight accumulator
    sr_10m_canvas = np.zeros((canvas_10m['height_px'], canvas_10m['width_px'], n_bands_10m), dtype=np.float64)
    wt_10m_canvas = np.zeros((canvas_10m['height_px'], canvas_10m['width_px']), dtype=np.float64)

    sr_20m_canvas = np.zeros((canvas_20m['height_px'], canvas_20m['width_px'], n_bands_20m), dtype=np.float64)
    wt_20m_canvas = np.zeros((canvas_20m['height_px'], canvas_20m['width_px']), dtype=np.float64)

    use_blending = overlap_px > 0

    # Process each tile
    for idx, td in enumerate(tile_data):
        bounds = td['bounds']
        arrays = td['arrays']
        tid = getattr(td['sample'], 'tile_id', idx)

        # Run SR inference
        sr_out = run_sr_inference(
            model, arrays['lr_10m'], arrays['lr_20m'], device,
            target_size_10m=target_size_10m,
            target_size_20m=target_size_20m,
        )

        # -- Place 10m output --
        row_10m, col_10m = tile_pixel_offset(bounds, canvas_10m)
        exp_h_10m, exp_w_10m = tile_pixel_extent(bounds, canvas_10m)
        sr_tile_10m = sr_out['sr_10m']

        # Crop SR output to expected geographic extent (handles edge tiles / minor size mismatches)
        place_h = min(sr_tile_10m.shape[0], exp_h_10m, canvas_10m['height_px'] - row_10m)
        place_w = min(sr_tile_10m.shape[1], exp_w_10m, canvas_10m['width_px'] - col_10m)
        if place_h <= 0 or place_w <= 0:
            continue
        sr_tile_10m = sr_tile_10m[:place_h, :place_w]

        if use_blending:
            weight = make_gaussian_weight(place_h, place_w)
        else:
            weight = np.ones((place_h, place_w), dtype=np.float32)

        sr_10m_canvas[row_10m:row_10m + place_h, col_10m:col_10m + place_w] += (
            sr_tile_10m.astype(np.float64) * weight[:, :, np.newaxis]
        )
        wt_10m_canvas[row_10m:row_10m + place_h, col_10m:col_10m + place_w] += weight

        # -- Place 20m output --
        row_20m, col_20m = tile_pixel_offset(bounds, canvas_20m)
        exp_h_20m, exp_w_20m = tile_pixel_extent(bounds, canvas_20m)
        sr_tile_20m = sr_out['sr_20m']

        place_h_20 = min(sr_tile_20m.shape[0], exp_h_20m, canvas_20m['height_px'] - row_20m)
        place_w_20 = min(sr_tile_20m.shape[1], exp_w_20m, canvas_20m['width_px'] - col_20m)
        if place_h_20 > 0 and place_w_20 > 0:
            sr_tile_20m = sr_tile_20m[:place_h_20, :place_w_20]

            if use_blending:
                weight_20 = make_gaussian_weight(place_h_20, place_w_20)
            else:
                weight_20 = np.ones((place_h_20, place_w_20), dtype=np.float32)

            sr_20m_canvas[row_20m:row_20m + place_h_20, col_20m:col_20m + place_w_20] += (
                sr_tile_20m.astype(np.float64) * weight_20[:, :, np.newaxis]
            )
            wt_20m_canvas[row_20m:row_20m + place_h_20, col_20m:col_20m + place_w_20] += weight_20

        if (idx + 1) % 10 == 0 or idx == len(tile_data) - 1:
            print(f"    [{idx + 1}/{len(tile_data)}] tile_id={tid} placed at row={row_10m}, col={col_10m}")

    # Normalize by accumulated weights (avoid division by zero → leave as 0 = NoData)
    valid_10m = wt_10m_canvas > 0
    sr_10m_canvas[valid_10m] /= wt_10m_canvas[valid_10m, np.newaxis]

    valid_20m = wt_20m_canvas > 0
    sr_20m_canvas[valid_20m] /= wt_20m_canvas[valid_20m, np.newaxis]

    sr_10m_canvas = np.clip(sr_10m_canvas, 0, 1).astype(np.float32)
    sr_20m_canvas = np.clip(sr_20m_canvas, 0, 1).astype(np.float32)

    coverage_10m = np.count_nonzero(valid_10m) / valid_10m.size * 100
    coverage_20m = np.count_nonzero(valid_20m) / valid_20m.size * 100
    print(f"  Coverage — 10m: {coverage_10m:.1f}%, 20m: {coverage_20m:.1f}%")

    return {
        'sr_10m': (sr_10m_canvas, canvas_10m),
        'sr_20m': (sr_20m_canvas, canvas_20m),
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Reconstruct full AOI image from tiled SR patches.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--dataset', type=str, required=True,
                   help='Path to dataset .pkl file or directory with chunk_*.pkl files.')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to model checkpoint (.pt).')
    p.add_argument('--sequence-id', type=int, default=0,
                   help='Temporal sequence index to reconstruct (default: 0).')
    p.add_argument('--output', type=str, default=None,
                   help='Output path for GeoTIFF. Defaults to outputs/reconstructed/<dataset_name>_seq<N>.tif')
    p.add_argument('--overlap', type=int, default=0,
                   help='Overlap in HR pixels for Gaussian blending (0 = no overlap, recommended: 32).')
    p.add_argument('--device', type=str, default=None,
                   help='Device: "cuda", "cpu", or auto-detect.')
    p.add_argument('--target-size-10m', type=int, nargs=2, default=[256, 256],
                   help='Target spatial size for 10m SR output (H W). Default: 256 256')
    p.add_argument('--target-size-20m', type=int, nargs=2, default=[128, 128],
                   help='Target spatial size for 20m SR output (H W). Default: 128 128')
    p.add_argument('--band-names-10m', type=str, nargs='+', default=None,
                   help='Band names for 10m GeoTIFF (e.g. B2 B3 B4).')
    p.add_argument('--band-names-20m', type=str, nargs='+', default=None,
                   help='Band names for 20m GeoTIFF (e.g. B5 B8A).')
    p.add_argument('--list-sequences', action='store_true',
                   help='List available sequence IDs and exit.')
    p.add_argument('--also-sources', action='store_true',
                   help='Also produce PlanetScope (LR) and Sentinel-2 (HR) mosaics '
                        'from the same tiles/dates for classification comparison.')
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    samples = load_samples(dataset_path)
    print(f"  Loaded {len(samples)} samples")

    if not samples:
        print("ERROR: No samples found")
        sys.exit(1)

    # Filter to the requested region (use the first sample's region if not specified)
    regions = set(getattr(s, 'region_name', 'unknown') for s in samples)
    print(f"  Regions: {regions}")

    seq_ids = sorted(set(getattr(s, 'sequence_id', -1) for s in samples))
    print(f"  Sequence IDs: {seq_ids}")
    for sid in seq_ids:
        count = sum(1 for s in samples if getattr(s, 'sequence_id', -1) == sid)
        dates = None
        for s in samples:
            if getattr(s, 'sequence_id', -1) == sid:
                dates = getattr(s, 'temporal_dates', None)
                break
        print(f"    seq {sid}: {count} tiles, dates={dates}")

    if args.list_sequences:
        return

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Load model
    print(f"Loading model: {checkpoint_path}")
    model, _ = load_sr_model(checkpoint_path, device)
    model.eval()

    # Reconstruct
    target_10m = tuple(args.target_size_10m)
    target_20m = tuple(args.target_size_20m)

    print(f"\nReconstructing sequence {args.sequence_id}...")
    results = reconstruct_aoi(
        samples, model, device,
        sequence_id=args.sequence_id,
        overlap_px=args.overlap,
        target_size_10m=target_10m,
        target_size_20m=target_20m,
    )

    # Determine output path
    if args.output:
        out_base = Path(args.output)
        if not out_base.is_absolute():
            out_base = project_root / out_base
    else:
        ds_name = dataset_path.stem if dataset_path.is_file() else dataset_path.name
        out_base = project_root / 'outputs' / 'reconstructed' / f"{ds_name}_seq{args.sequence_id}.tif"

    out_10m = out_base.with_name(out_base.stem + '_10m' + out_base.suffix)
    out_20m = out_base.with_name(out_base.stem + '_20m' + out_base.suffix)

    print(f"\nSaving outputs...")
    sr_10m_arr, canvas_10m = results['sr_10m']
    sr_20m_arr, canvas_20m = results['sr_20m']

    save_geotiff(out_10m, sr_10m_arr, canvas_10m,
                 band_names=args.band_names_10m, nodata=0.0)
    save_geotiff(out_20m, sr_20m_arr, canvas_20m,
                 band_names=args.band_names_20m, nodata=0.0)

    # Optionally produce co-registered PlanetScope and Sentinel-2 mosaics
    if args.also_sources:
        print(f"\nBuilding source mosaics (PlanetScope + Sentinel-2)...")
        src_results = mosaic_source_images(samples, sequence_id=args.sequence_id)

        source_band_names = {
            'planet_10m': ['B2', 'B3', 'B4'],
            'planet_20m': ['B5', 'B8A'],
            'sentinel_10m': ['B2', 'B3', 'B4'],
            'sentinel_20m': ['B5', 'B8A'],
        }

        for name, (arr, canvas) in src_results.items():
            src_path = out_base.with_name(out_base.stem + f'_{name}' + out_base.suffix)
            save_geotiff(src_path, arr, canvas,
                         band_names=source_band_names.get(name), nodata=0.0)

    print("\nDone.")


if __name__ == '__main__':
    main()
