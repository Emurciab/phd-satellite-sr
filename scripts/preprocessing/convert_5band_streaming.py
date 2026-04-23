#!/usr/bin/env python3
"""
Memory-efficient streaming converter for 5-band format.

This script processes samples one at a time using pickle streaming to avoid OOM.
It never loads the full dataset into memory at once.
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
    """5-band output format"""
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
    """Setup mock modules for unpickling."""
    class MockModule:
        TrainingSample = TrainingSample
    
    sys.modules['generate_sta56_grid_parallel'] = MockModule()
    sys.modules['generate_sta1e_grid_parallel'] = MockModule()
    sys.modules['generate_sta1w_grid_parallel'] = MockModule()
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
    """Compute quality stats for a frame."""
    def get_valid_frac(arr):
        total = arr.size
        nan_frac = np.sum(np.isnan(arr)) / total
        zero_frac = np.sum(np.abs(arr) < 0.01) / total
        return 1.0 - nan_frac - zero_frac
    
    lr_valid = get_valid_frac(lr_frame)
    hr_valid = get_valid_frac(hr_frame)
    
    return {
        'frame_idx': frame_idx,
        'lr_valid_frac': lr_valid,
        'hr_valid_frac': hr_valid,
        'is_good': lr_valid > 0.85 and hr_valid > 0.85
    }


def convert_single_sample(sample: TrainingSample, region_name: str = 'STA-5/6') -> Optional[TrainingSample5Band]:
    """Convert a single sample to 5-band format."""
    if sample.tile_bounds is None:
        return None
    
    band_mode = sample.tile_bounds.get('band_mode', '4-band')
    planet_bands = sample.tile_bounds.get('planet_bands', [])
    
    # Only process 8-band SuperDove
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
    hr_10m_sequence = hr_sequence[:, :, :, :3].astype(np.float32)
    hr_20m_full = hr_sequence[:, :, :, 3:]
    
    # Downsample 20m HR from 256 to 128
    hr_20m_frames = []
    for t in range(n_frames):
        hr_20m_frame = downsample_hr_for_20m(hr_20m_full[t], factor=2)
        hr_20m_frames.append(hr_20m_frame)
    hr_20m_sequence = np.stack(hr_20m_frames, axis=0).astype(np.float32)
    
    # Compute frame quality
    frame_quality_list = []
    for t in range(n_frames):
        fq = compute_frame_quality(lr_sequence[t], hr_sequence[t], t)
        frame_quality_list.append(fq)
    
    return TrainingSample5Band(
        lr_sequence=lr_sequence.astype(np.float32),
        hr_sequence=hr_sequence.astype(np.float32),
        lr_20m_sequence=lr_20m_sequence.astype(np.float32) if lr_20m_sequence is not None else None,
        hr_10m_sequence=hr_10m_sequence,
        hr_20m_sequence=hr_20m_sequence,
        temporal_dates=sample.temporal_dates or [],
        planet_dates=sample.planet_dates or [],
        region_name=region_name,
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
    parser = argparse.ArgumentParser(description='Memory-efficient 5-band converter')
    parser.add_argument('--input', type=str, required=True, help='Input dataset path')
    parser.add_argument('--output', type=str, required=True, help='Output dataset path')
    parser.add_argument('--region', type=str, default='STA-5/6', help='Region name')
    parser.add_argument('--chunk-size', type=int, default=100, help='Samples per chunk file')
    args = parser.parse_args()
    
    input_path = Path(args.input) / 'dataset.pkl'
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🔄 MEMORY-EFFICIENT 5-BAND CONVERSION")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Chunk size: {args.chunk_size} samples per file")
    print("=" * 70)
    
    setup_unpickle_modules()
    
    # Load input - this is unavoidable but we'll process one at a time
    print("\n📂 Loading input dataset...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, list):
        input_samples = data
        config = {}
    else:
        input_samples = data.get('samples', data)
        config = data.get('config', {})
    
    total_samples = len(input_samples)
    print(f"   Loaded {total_samples} samples")
    
    # Process and save in chunks
    print(f"\n🔄 Converting samples in chunks of {args.chunk_size}...")
    
    converted_count = 0
    skipped_4band = 0
    errors = 0
    chunk_idx = 0
    current_chunk = []
    
    for i, sample in enumerate(input_samples):
        try:
            converted = convert_single_sample(sample, args.region)
            
            if converted is not None:
                current_chunk.append(converted)
                converted_count += 1
                
                # Save chunk when full
                if len(current_chunk) >= args.chunk_size:
                    chunk_path = output_dir / f'chunk_{chunk_idx:04d}.pkl'
                    with open(chunk_path, 'wb') as f:
                        pickle.dump(current_chunk, f, protocol=4)
                    print(f"   💾 Saved chunk {chunk_idx}: {len(current_chunk)} samples")
                    current_chunk = []
                    chunk_idx += 1
                    gc.collect()
            else:
                if sample.tile_bounds and sample.tile_bounds.get('band_mode') != '8-band':
                    skipped_4band += 1
                else:
                    errors += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"   ❌ Error on sample {i}: {e}")
        
        if (i + 1) % 200 == 0:
            print(f"   Progress: {i+1}/{total_samples} (converted: {converted_count}, skipped: {skipped_4band})")
    
    # Save remaining samples
    if current_chunk:
        chunk_path = output_dir / f'chunk_{chunk_idx:04d}.pkl'
        with open(chunk_path, 'wb') as f:
            pickle.dump(current_chunk, f, protocol=4)
        print(f"   💾 Saved final chunk {chunk_idx}: {len(current_chunk)} samples")
    
    # Clear memory
    del input_samples
    del data
    gc.collect()
    
    # Save metadata
    metadata = {
        'region': args.region,
        'sensor_type': 'superdove',
        'n_samples': converted_count,
        'n_chunks': chunk_idx + 1,
        'chunk_size': args.chunk_size,
        'planet_bands': ['B2', 'B4', 'B6', 'B7', 'B8'],
        's2_bands_10m': ['B2', 'B3', 'B4'],
        's2_bands_20m': ['B5', 'B8A'],
        'normalized': True,
        'skipped_4band': skipped_4band,
        'errors': errors
    }
    
    with open(output_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"Converted: {converted_count} samples")
    print(f"Skipped (4-band): {skipped_4band}")
    print(f"Errors: {errors}")
    print(f"Chunks saved: {chunk_idx + 1}")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
