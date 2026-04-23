#!/usr/bin/env python3
"""
PIPELINE PROFILER - Detailed timing analysis for SR data generation

This script profiles each component of the pipeline to identify bottlenecks:
- GEE image fetching (collection queries)
- Coverage checking
- Tile downloading (Planet + S2)
- Coregistration (AROSICS)
- Histogram matching (harmonization)
- I/O operations

Run: python scripts/testing/profile_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import numpy as np
import ee
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

# Initialize timing storage
@dataclass
class ProfileResults:
    """Store profiling results"""
    operation_times: Dict[str, List[float]] = field(default_factory=dict)
    operation_counts: Dict[str, int] = field(default_factory=dict)
    
    def add_time(self, operation: str, elapsed: float):
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
        self.operation_times[operation].append(elapsed)
        self.operation_counts[operation] += 1
    
    def summary(self) -> Dict:
        summary = {}
        total_time = 0
        for op, times in self.operation_times.items():
            summary[op] = {
                'count': len(times),
                'total': sum(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'min': min(times),
                'max': max(times)
            }
            total_time += sum(times)
        summary['_total_profiled_time'] = total_time
        return summary


class PipelineProfiler:
    """Profile each component of the SR pipeline"""
    
    def __init__(self, cache_dir: str = './profile_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results = ProfileResults()
        self._init_gee()
    
    def _init_gee(self):
        """Initialize Google Earth Engine"""
        t0 = time.time()
        try:
            credentials = ee.ServiceAccountCredentials(
                'stas-50@remotesensingeduart.iam.gserviceaccount.com',
                '/media/eamurciab/TOSHIBA_EXT1/PhD/SR/Auth/remotesensingeduart-222d07dd6689.json'
            )
            ee.Initialize(credentials)
        except:
            ee.Initialize()
        self.results.add_time('gee_init', time.time() - t0)
        print(f"✓ GEE initialized in {time.time() - t0:.2f}s")
    
    def profile_find_pairs(self, aoi, start_date: str, end_date: str) -> dict:
        """Profile the find_pairs operation"""
        from Auth.AuthGEE import find_pairs, PLANET_COLLECTION_ID
        
        print(f"\n📊 Profiling find_pairs...")
        t0 = time.time()
        
        result = find_pairs(aoi, start_date, end_date, PLANET_COLLECTION_ID)
        
        elapsed = time.time() - t0
        self.results.add_time('find_pairs', elapsed)
        print(f"   ✓ find_pairs: {elapsed:.2f}s ({len(result.get('matched_pairs', []))} pairs)")
        
        return result
    
    def profile_coverage_check(self, img: ee.Image, geom: ee.Geometry, 
                               band_name: str, min_percent: float = 90.0) -> bool:
        """Profile coverage checking for a single tile"""
        t0 = time.time()
        
        try:
            valid_mask = img.select(band_name).mask()
            stats = valid_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=30,
                maxPixels=1e6,
                bestEffort=True
            )
            valid_fraction = stats.get(band_name).getInfo()
            result = (valid_fraction or 0) * 100.0 >= min_percent
        except:
            result = False
        
        elapsed = time.time() - t0
        self.results.add_time('coverage_check_single', elapsed)
        return result
    
    def profile_tile_download(self, planet_img: ee.Image, s2_img: ee.Image,
                              tile_bounds: List, planet_bands: List, 
                              sentinel_bands: List) -> tuple:
        """Profile downloading a single tile (Planet + S2)"""
        
        # Profile Planet download
        t0 = time.time()
        try:
            tile_geom = ee.Geometry.Rectangle(tile_bounds)
            
            hr_selected = planet_img.select(planet_bands).clip(tile_geom).unmask(0)
            hr_resampled = hr_selected.reproject(crs='EPSG:4326', scale=3)
            hr_rect = hr_resampled.sampleRectangle(region=tile_geom, defaultValue=0)
            hr_info = hr_rect.getInfo()
            
            if hr_info and 'properties' in hr_info:
                hr_props = hr_info['properties']
                hr_arrays = [np.array(hr_props[b], dtype=np.float32) for b in planet_bands]
                hr_array = np.stack(hr_arrays, axis=0)
                hr_array = np.transpose(hr_array, (1, 2, 0))
            else:
                hr_array = None
        except Exception as e:
            print(f"   Planet download error: {e}")
            hr_array = None
        
        planet_time = time.time() - t0
        self.results.add_time('download_planet_tile', planet_time)
        
        # Profile S2 download
        t0 = time.time()
        try:
            tile_geom = ee.Geometry.Rectangle(tile_bounds)
            
            lr_selected = s2_img.select(sentinel_bands).clip(tile_geom).unmask(0)
            lr_resampled = lr_selected.reproject(crs='EPSG:4326', scale=10)
            lr_rect = lr_resampled.sampleRectangle(region=tile_geom, defaultValue=0)
            lr_info = lr_rect.getInfo()
            
            if lr_info and 'properties' in lr_info:
                lr_props = lr_info['properties']
                lr_arrays = [np.array(lr_props[b], dtype=np.float32) for b in sentinel_bands]
                lr_array = np.stack(lr_arrays, axis=0)
                lr_array = np.transpose(lr_array, (1, 2, 0))
            else:
                lr_array = None
        except Exception as e:
            print(f"   S2 download error: {e}")
            lr_array = None
        
        s2_time = time.time() - t0
        self.results.add_time('download_s2_tile', s2_time)
        
        return hr_array, lr_array
    
    def profile_coregistration(self, hr_array: np.ndarray, lr_array: np.ndarray,
                               tile_bounds: List) -> np.ndarray:
        """Profile AROSICS coregistration"""
        from scripts.preprocessing.numpy_efficient_pipeline import coregister_lr_to_hr
        
        aoi_bounds_coords = [
            [tile_bounds[0], tile_bounds[1]],
            [tile_bounds[2], tile_bounds[1]],
            [tile_bounds[2], tile_bounds[3]],
            [tile_bounds[0], tile_bounds[3]]
        ]
        
        t0 = time.time()
        try:
            hr_coreg = coregister_lr_to_hr(hr_array, lr_array, aoi_bounds_coords)
        except Exception as e:
            print(f"   Coregistration error: {e}")
            hr_coreg = hr_array
        
        elapsed = time.time() - t0
        self.results.add_time('coregistration', elapsed)
        return hr_coreg
    
    def profile_harmonization(self, hr_array: np.ndarray, 
                              lr_array: np.ndarray) -> np.ndarray:
        """Profile histogram matching harmonization"""
        from scripts.preprocessing.numpy_efficient_pipeline import harmonize_planet_to_sentinel2
        
        t0 = time.time()
        try:
            hr_harm = harmonize_planet_to_sentinel2(hr_array, lr_array)
        except Exception as e:
            print(f"   Harmonization error: {e}")
            hr_harm = hr_array
        
        elapsed = time.time() - t0
        self.results.add_time('harmonization', elapsed)
        return hr_harm
    
    def profile_band_detection(self, planet_collection: ee.ImageCollection,
                               planet_dates: List[str]) -> dict:
        """Profile band detection for multiple dates"""
        date_band_config = {}
        
        for pdate in planet_dates:
            t0 = time.time()
            try:
                planet_img = planet_collection.filter(ee.Filter.eq('date_key', pdate)).first()
                available_bands = planet_img.bandNames().getInfo()
                
                has_extended_bands = all(band in available_bands for band in ['B5', 'B6', 'B7', 'B8'])
                
                if has_extended_bands:
                    date_band_config[pdate] = {'mode': '8-band'}
                else:
                    date_band_config[pdate] = {'mode': '4-band'}
            except:
                date_band_config[pdate] = {'mode': '4-band'}
            
            elapsed = time.time() - t0
            self.results.add_time('band_detection_single', elapsed)
        
        return date_band_config
    
    def profile_full_sample(self, planet_img: ee.Image, s2_img: ee.Image,
                            tile_bounds: List, planet_bands: List,
                            sentinel_bands: List) -> dict:
        """Profile generating a complete sample (download + coreg + harm)"""
        
        sample_times = {}
        
        # Download
        t0 = time.time()
        hr_array, lr_array = self.profile_tile_download(
            planet_img, s2_img, tile_bounds, planet_bands, sentinel_bands
        )
        sample_times['download_total'] = time.time() - t0
        
        if hr_array is None or lr_array is None:
            return {'error': 'download_failed', **sample_times}
        
        # Coregistration
        t0 = time.time()
        hr_coreg = self.profile_coregistration(hr_array, lr_array, tile_bounds)
        sample_times['coregistration'] = time.time() - t0
        
        # Harmonization
        t0 = time.time()
        hr_harm = self.profile_harmonization(hr_coreg, lr_array)
        sample_times['harmonization'] = time.time() - t0
        
        sample_times['total'] = sum(sample_times.values())
        
        return sample_times
    
    def run_full_profile(self, num_samples: int = 3, region_idx: int = 0):
        """Run a complete profiling session"""
        from Auth.AuthGEE import regions, PLANET_COLLECTION_ID
        
        print("=" * 80)
        print("🔬 PIPELINE PROFILER - Full Analysis")
        print("=" * 80)
        
        region_config = regions[region_idx]
        print(f"\n📍 Region: {region_config['name']}")
        print(f"   Samples to profile: {num_samples}")
        
        # Profile find_pairs
        pair_results = self.profile_find_pairs(
            region_config['aoi'],
            '2024-01-01',
            '2024-12-31'
        )
        
        matched_pairs = pair_results.get('matched_pairs', [])
        s2_collection = pair_results.get('s2_collection')
        planet_collection = pair_results.get('planet_collection')
        
        if len(matched_pairs) < 3:
            print("❌ Insufficient pairs for profiling")
            return
        
        # Profile band detection
        print(f"\n📊 Profiling band detection...")
        t0 = time.time()
        planet_dates = list(set([p['planet_date'] for p in matched_pairs[:10]]))
        date_band_config = self.profile_band_detection(planet_collection, planet_dates)
        print(f"   ✓ Band detection: {time.time() - t0:.2f}s for {len(planet_dates)} dates")
        
        # Profile coverage checking
        print(f"\n📊 Profiling coverage checking...")
        sequence = matched_pairs[:3]
        
        # Generate grid
        t0 = time.time()
        bounds = region_config['aoi'].bounds().getInfo()['coordinates'][0]
        lons = [p[0] for p in bounds]
        lats = [p[1] for p in bounds]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        tile_size = 0.007
        tiles = []
        for row in range(5):  # Just check first 25 tiles
            for col in range(5):
                tile_min_lon = min_lon + col * tile_size
                tile_min_lat = min_lat + row * tile_size
                tile_max_lon = tile_min_lon + tile_size
                tile_max_lat = tile_min_lat + tile_size
                tiles.append([tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat])
        
        grid_time = time.time() - t0
        self.results.add_time('grid_generation', grid_time)
        print(f"   ✓ Grid generation: {grid_time:.2f}s")
        
        # Profile coverage checks
        planet_img = planet_collection.filter(ee.Filter.eq("id_pref", sequence[0]["planet_id"])).first()
        s2_img = s2_collection.filter(ee.Filter.eq("id_pref", sequence[0]["s2_id"])).first()
        
        print(f"   Checking {len(tiles)} tiles...")
        valid_count = 0
        for i, tile_bounds in enumerate(tiles):
            tile_geom = ee.Geometry.Rectangle(tile_bounds)
            
            t0 = time.time()
            planet_valid = self.profile_coverage_check(planet_img, tile_geom, 'B1')
            s2_valid = self.profile_coverage_check(s2_img, tile_geom, 'B2')
            
            if planet_valid and s2_valid:
                valid_count += 1
            
            if (i + 1) % 10 == 0:
                avg_time = np.mean(self.results.operation_times['coverage_check_single'])
                print(f"   Progress: {i+1}/{len(tiles)}, avg coverage check: {avg_time*1000:.0f}ms")
        
        print(f"   ✓ Valid tiles: {valid_count}/{len(tiles)}")
        
        # Profile full sample generation
        print(f"\n📊 Profiling full sample generation ({num_samples} samples)...")
        
        planet_bands = ['B1', 'B2', 'B3', 'B4']
        sentinel_bands = ['B2', 'B3', 'B4', 'B8A']
        
        # Find valid tiles for sampling
        valid_tiles = []
        for tile_bounds in tiles:
            tile_geom = ee.Geometry.Rectangle(tile_bounds)
            if (self.profile_coverage_check(planet_img, tile_geom, 'B1') and
                self.profile_coverage_check(s2_img, tile_geom, 'B2')):
                valid_tiles.append(tile_bounds)
            if len(valid_tiles) >= num_samples:
                break
        
        if len(valid_tiles) == 0:
            print("   ❌ No valid tiles found")
            return
        
        sample_results = []
        for i, tile_bounds in enumerate(valid_tiles[:num_samples]):
            print(f"\n   Sample {i+1}/{num_samples}...")
            
            sample_time = self.profile_full_sample(
                planet_img, s2_img, tile_bounds,
                planet_bands, sentinel_bands
            )
            sample_results.append(sample_time)
            
            if 'error' not in sample_time:
                print(f"      Download: {sample_time['download_total']:.1f}s")
                print(f"      Coregistration: {sample_time['coregistration']:.1f}s")
                print(f"      Harmonization: {sample_time['harmonization']:.1f}s")
                print(f"      Total: {sample_time['total']:.1f}s")
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print profiling summary"""
        print("\n" + "=" * 80)
        print("📊 PROFILING SUMMARY")
        print("=" * 80)
        
        summary = self.results.summary()
        total_profiled = summary.pop('_total_profiled_time', 0)
        
        # Sort by total time
        sorted_ops = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)
        
        print(f"\n{'Operation':<30} {'Count':>8} {'Total':>10} {'Mean':>10} {'%':>8}")
        print("-" * 80)
        
        for op, stats in sorted_ops:
            pct = (stats['total'] / total_profiled * 100) if total_profiled > 0 else 0
            print(f"{op:<30} {stats['count']:>8} {stats['total']:>9.1f}s {stats['mean']:>9.2f}s {pct:>7.1f}%")
        
        print("-" * 80)
        print(f"{'TOTAL PROFILED TIME':<30} {'':<8} {total_profiled:>9.1f}s")
        
        # Breakdown by category
        print("\n" + "=" * 80)
        print("📊 BREAKDOWN BY CATEGORY")
        print("=" * 80)
        
        categories = {
            'GEE Requests': ['find_pairs', 'coverage_check_single', 'download_planet_tile', 
                           'download_s2_tile', 'band_detection_single'],
            'Local Processing': ['coregistration', 'harmonization', 'grid_generation'],
            'Initialization': ['gee_init']
        }
        
        for cat_name, ops in categories.items():
            cat_time = sum(summary.get(op, {'total': 0})['total'] for op in ops)
            cat_pct = (cat_time / total_profiled * 100) if total_profiled > 0 else 0
            print(f"\n{cat_name}: {cat_time:.1f}s ({cat_pct:.1f}%)")
            for op in ops:
                if op in summary:
                    print(f"   - {op}: {summary[op]['total']:.1f}s ({summary[op]['count']} calls)")
        
        # Key insights
        print("\n" + "=" * 80)
        print("💡 KEY INSIGHTS")
        print("=" * 80)
        
        # Identify bottlenecks
        if 'download_planet_tile' in summary and 'download_s2_tile' in summary:
            download_time = summary['download_planet_tile']['total'] + summary['download_s2_tile']['total']
            download_pct = (download_time / total_profiled * 100) if total_profiled > 0 else 0
            print(f"\n1. GEE Downloads: {download_time:.1f}s ({download_pct:.1f}% of total)")
            print(f"   - Planet: avg {summary['download_planet_tile']['mean']:.1f}s/tile")
            print(f"   - S2: avg {summary['download_s2_tile']['mean']:.1f}s/tile")
        
        if 'coregistration' in summary:
            coreg_time = summary['coregistration']['total']
            coreg_pct = (coreg_time / total_profiled * 100) if total_profiled > 0 else 0
            print(f"\n2. Coregistration: {coreg_time:.1f}s ({coreg_pct:.1f}% of total)")
            print(f"   - Avg: {summary['coregistration']['mean']:.1f}s/sample")
        
        if 'coverage_check_single' in summary:
            cov_time = summary['coverage_check_single']['total']
            cov_pct = (cov_time / total_profiled * 100) if total_profiled > 0 else 0
            print(f"\n3. Coverage Checking: {cov_time:.1f}s ({cov_pct:.1f}% of total)")
            print(f"   - Avg: {summary['coverage_check_single']['mean']*1000:.0f}ms/check")
            print(f"   - Total checks: {summary['coverage_check_single']['count']}")
        
        # Recommendations
        print("\n" + "=" * 80)
        print("🚀 OPTIMIZATION RECOMMENDATIONS")
        print("=" * 80)
        
        if 'download_planet_tile' in summary:
            download_pct = (summary['download_planet_tile']['total'] + 
                          summary.get('download_s2_tile', {'total': 0})['total']) / total_profiled * 100
            if download_pct > 50:
                print("\n⚡ GEE Downloads are the main bottleneck:")
                print("   - Consider batching multiple tiles in single requests")
                print("   - Cache raw downloads to avoid re-fetching")
                print("   - Use larger chunks (512px) if training allows")
        
        if 'coregistration' in summary and summary['coregistration']['mean'] > 10:
            print("\n⚡ Coregistration is slow:")
            print("   - Consider skipping when shift < threshold")
            print("   - Use simpler phase correlation for similar scenes")
            print("   - Cache coregistered results aggressively")
        
        if 'coverage_check_single' in summary and summary['coverage_check_single']['total'] > 30:
            print("\n⚡ Coverage checking is expensive:")
            print("   - Cache valid tile lists per sequence")
            print("   - Increase parallelism for coverage checks")
            print("   - Use coarser scale (50-100m) for initial checks")


def main():
    """Run the profiler"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Profile the SR pipeline')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of samples to profile')
    parser.add_argument('--region', type=int, default=0,
                       help='Region index to profile')
    
    args = parser.parse_args()
    
    profiler = PipelineProfiler()
    profiler.run_full_profile(num_samples=args.samples, region_idx=args.region)


if __name__ == '__main__':
    main()
