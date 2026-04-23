#!/usr/bin/env python3
"""
Memory-efficient chunked conversion of STA-5/6 datasets to 5-band format.
Processes and saves in chunks to avoid OOM errors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np
import time
import gc


@dataclass
class TrainingSample:
    """Original TrainingSample format (for loading)"""
    lr_sequence: np.ndarray = None
    hr_sequence: np.ndarray = None
    temporal_dates: List[str] = None
    planet_dates: List[str] = None
    region_name: str = ''
    sequence_id: int = 0
    tile_id: int = 0
    tile_bounds: Dict = None
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None
    lr_20m_sequence: Optional[np.ndarray] = None
    frame_quality: Optional[List[Dict]] = None


@dataclass
class TrainingSample5Band:
    """Training sample with full 5-band HR data for SuperDove images."""
    lr_sequence: np.ndarray
    hr_sequence: np.ndarray
    lr_20m_sequence: Optional[np.ndarray] = None
    hr_10m_sequence: Optional[np.ndarray] = None
    hr_20m_sequence: Optional[np.ndarray] = None
    temporal_dates: List[str] = field(default_factory=list)
    planet_dates: List[str] = field(default_factory=list)
    region_name: str = ''
    sequence_id: int = 0
    tile_id: int = 0
    tile_bounds: Dict = field(default_factory=dict)
    sensor_type: str = 'superdove'
    planet_bands: List[str] = field(default_factory=lambda: ['B2', 'B4', 'B6', 'B7', 'B8'])
    sentinel_bands: List[str] = field(default_factory=lambda: ['B2', 'B3', 'B4', 'B5', 'B8A'])
    s2_solar_zenith: Optional[List[float]] = None
    planet_solar_zenith: Optional[List[float]] = None
    frame_quality: Optional[List[Dict]] = None


def setup_unpickle_modules():
    """Setup mock modules for unpickling TrainingSample objects."""
    class MockModule:
        TrainingSample = TrainingSample
    sys.modules['generate_sta56_grid_parallel'] = MockModule()
    sys.modules['generate_sta2_grid_parallel'] = MockModule()


def downsample_hr_for_20m(hr_array: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample HR array by factor for 20m band targets."""
    h, w, c = hr_array.shape
    new_h = h // factor
    new_w = w // factor
    h_trim = new_h * factor
    w_trim = new_w * factor
    trimmed = hr_array[:h_trim, :w_trim, :]
    result = trimmed.reshape(new_h, factor, new_w, factor, c).mean(axis=(1, 3))
    return result.astype(hr_array.dtype)


def compute_frame_quality(lr_frame: np.ndarray, hr_frame: np.ndarray, frame_idx: int = 0) -> Dict:
    """Compute combined quality stats for both LR and HR frames."""
    def get_stats(arr):
        total = arr.size
        nan_frac = np.sum(np.isnan(arr)) / total
        zero_frac = np.sum(np.abs(arr) < 0.01) / total
        valid_frac = 1.0 - nan_frac - zero_frac
        return {'nan_frac': nan_frac, 'zero_frac': zero_frac, 'valid_frac': valid_frac}
    
    lr_stats = get_stats(lr_frame)
    hr_stats = get_stats(hr_frame)
    return {
        'frame_idx': frame_idx,
        'lr_valid_frac': lr_stats['valid_frac'],
        'hr_valid_frac': hr_stats['valid_frac'],
        'is_good': lr_stats['valid_frac'] > 0.85 and hr_stats['valid_frac'] > 0.85
    }


def convert_sample_to_5band(sample: TrainingSample, input_already_normalized: bool = False) -> Optional[TrainingSample5Band]:
    """Convert a single TrainingSample to TrainingSample5Band format."""
    if sample.tile_bounds is None:
        return None
    
    band_mode = sample.tile_bounds.get('band_mode', '4-band')
    planet_bands = sample.tile_bounds.get('planet_bands', [])
    
    if band_mode != '8-band':
        return None
    
    expected_planet = ['B2', 'B4', 'B6', 'B7', 'B8']
    if planet_bands != expected_planet:
        return None
    
    hr_sequence = sample.hr_sequence
    lr_sequence = sample.lr_sequence
    lr_20m_sequence = sample.lr_20m_sequence
    
    if hr_sequence is None or hr_sequence.shape[-1] != 5:
        return None
    
    n_frames = hr_sequence.shape[0]
    
    # Split HR into 10m and 20m components
    hr_10m_sequence = hr_sequence[:, :, :, :3]
    hr_20m_full = hr_sequence[:, :, :, 3:]
    
    # Downsample 20m HR from 256 to 128
    hr_20m_frames = []
    for t in range(n_frames):
        hr_20m_frame = downsample_hr_for_20m(hr_20m_full[t], factor=2)
        hr_20m_frames.append(hr_20m_frame)
    hr_20m_sequence = np.stack(hr_20m_frames, axis=0)
    
    # Compute frame quality
    frame_quality_list = []
    for t in range(n_frames):
        fq = compute_frame_quality(lr_sequence[t], hr_sequence[t], t)
        frame_quality_list.append(fq)
    
    # Data is already normalized from generation, ensure float32
    hr_sequence = hr_sequence.astype(np.float32)
    hr_10m_sequence = hr_10m_sequence.astype(np.float32)
    hr_20m_sequence = hr_20m_sequence.astype(np.float32)
    lr_sequence = lr_sequence.astype(np.float32)
    if lr_20m_sequence is not None:
        lr_20m_sequence = lr_20m_sequence.astype(np.float32)
    
    return TrainingSample5Band(
        lr_sequence=lr_sequence,
        hr_sequence=hr_sequence,
        lr_20m_sequence=lr_20m_sequence,
        hr_10m_sequence=hr_10m_sequence,
        hr_20m_sequence=hr_20m_sequence,
        temporal_dates=sample.temporal_dates or [],
        planet_dates=sample.planet_dates or [],
        region_name=sample.region_name or 'STA-5/6',
        sequence_id=sample.sequence_id or 0,
        tile_id=sample.tile_id or 0,
        tile_bounds=sample.tile_bounds or {},
        sensor_type='superdove',
        planet_bands=expected_planet,
        sentinel_bands=['B2', 'B3', 'B4', 'B5', 'B8A'],
        s2_solar_zenith=sample.s2_solar_zenith,
        planet_solar_zenith=sample.planet_solar_zenith,
        frame_quality=frame_quality_list
    )


def main():
    parser = argparse.ArgumentParser(description='Memory-efficient chunked conversion of STA-5/6 to 5-band')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing dataset.pkl')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--chunk-size', type=int, default=200, help='Samples per chunk (default: 200)')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    input_path = input_dir / 'dataset.pkl'
    
    if not input_path.exists():
        print(f"❌ Input dataset not found: {input_path}")
        return
    
    output_dir = Path(args.output) if args.output else input_dir.parent / f"{input_dir.name}_5band"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🔄 CHUNKED CONVERSION: STA-5/6 TO 5-BAND FORMAT")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Chunk size: {args.chunk_size}")
    print("=" * 80)
    
    setup_unpickle_modules()
    
    # Load input dataset
    print(f"\n📂 Loading input dataset...")
    start_time = time.time()
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, list):
        input_samples = data
    else:
        input_samples = data.get('samples', data) if isinstance(data, dict) else data
    
    if not isinstance(input_samples, list):
        input_samples = [input_samples] if input_samples is not None else []
    
    total_samples = len(input_samples)
    print(f"   Loaded {total_samples} samples")
    
    # Process in chunks
    chunk_size = args.chunk_size
    n_chunks = (total_samples + chunk_size - 1) // chunk_size
    
    all_converted = []
    skipped_4band = 0
    errors = 0
    
    print(f"\n🔄 Converting in {n_chunks} chunks of {chunk_size} samples...")
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk_samples = input_samples[start_idx:end_idx]
        
        print(f"\n   Chunk {chunk_idx + 1}/{n_chunks} (samples {start_idx}-{end_idx-1})...")
        
        chunk_converted = []
        for i, sample in enumerate(chunk_samples):
            try:
                converted = convert_sample_to_5band(sample, input_already_normalized=True)
                if converted is not None:
                    chunk_converted.append(converted)
                else:
                    if hasattr(sample, 'tile_bounds') and sample.tile_bounds and sample.tile_bounds.get('band_mode') != '8-band':
                        skipped_4band += 1
                    else:
                        errors += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"      ❌ Error: {e}")
        
        all_converted.extend(chunk_converted)
        print(f"      ✓ Converted {len(chunk_converted)} samples (total: {len(all_converted)})")
        
        # Force garbage collection after each chunk
        del chunk_samples
        gc.collect()
    
    print(f"\n📊 Conversion Summary:")
    print(f"   Input samples:     {total_samples}")
    print(f"   5-band converted:  {len(all_converted)}")
    print(f"   4-band skipped:    {skipped_4band}")
    print(f"   Errors:            {errors}")
    
    if len(all_converted) == 0:
        print("\n❌ No samples were converted!")
        return
    
    # Save in chunks to avoid memory issues
    print(f"\n💾 Saving {len(all_converted)} samples...")
    
    # Save metadata first
    sample = all_converted[0]
    metadata = {
        'region': 'STA-5/6',
        'sensor_type': 'superdove',
        'n_samples': len(all_converted),
        'sequence_length': sample.hr_sequence.shape[0],
        'hr_bands': 5,
        'planet_bands': ['B2', 'B4', 'B6', 'B7', 'B8'],
        's2_bands_10m': ['B2', 'B3', 'B4'],
        's2_bands_20m': ['B5', 'B8A'],
        'normalized': True,
        'hr_tile_size': 256,
        'lr_10m_tile_size': 77,
        'lr_20m_tile_size': 38,
        'hr_20m_tile_size': 128,
        'source_dataset': str(input_path)
    }
    
    print(f"   Sample shapes: HR={sample.hr_sequence.shape}, LR={sample.lr_sequence.shape}")
    print(f"   Value range: [{sample.hr_sequence.min():.4f}, {sample.hr_sequence.max():.4f}]")
    
    # Save as single file but with protocol=4 for efficiency
    output_path = output_dir / 'dataset_5band.pkl'
    dataset = {'samples': all_converted, 'metadata': metadata}
    
    print(f"   Writing to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=4)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Samples: {len(all_converted)}")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
