"""
Benchmark Script: Generate 5 samples using generate_sta2_grid_parallel and estimate total time

This script:
1. Uses the grid-based parallel pipeline with coverage pre-checking
2. Generates 5 samples to measure per-sample processing time
3. Queries all STAs to count total available matched pairs
4. Estimates total time required to process all samples across all STAs

Configuration:
- Coverage workers: 15 (for parallel coverage checking)
- Tile workers: 2 (for parallel tile processing)
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# SSL fixes
import certifi
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pickle

# Import pipeline components
from Auth.AuthGEE import find_pairs, PLANET_COLLECTION_ID, regions


# Configuration for benchmark
COVERAGE_WORKERS = 15  # Parallel workers for coverage checking
TILE_WORKERS = 2       # Parallel workers for tile processing
N_SAMPLES = 5          # Number of samples to generate for benchmark
SEQUENCE_LENGTH = 3    # Temporal sequence length


def count_pairs_for_region(region_config: dict) -> dict:
    """
    Count the number of matched pairs available for a region.
    
    Returns:
        dict with counts and metadata
    """
    try:
        planet_coll = region_config.get("planet_collection_id", PLANET_COLLECTION_ID)
        result = find_pairs(
            region_config["aoi"], 
            region_config["start"], 
            region_config["end"], 
            planet_coll
        )
        
        matched_pairs = result.get('matched_pairs', [])
        
        return {
            "name": region_config["name"],
            "start": region_config["start"],
            "end": region_config["end"],
            "matched_pairs": len(matched_pairs),
            "s2_unmatched": len(result.get('s2_unmatched', [])),
            "planet_unmatched": len(result.get('planet_unmatched', [])),
            "success": True
        }
    except Exception as e:
        return {
            "name": region_config["name"],
            "start": region_config["start"],
            "end": region_config["end"],
            "matched_pairs": 0,
            "error": str(e),
            "success": False
        }


def run_grid_parallel_benchmark(region_name: str, n_samples: int = 5,
                                coverage_workers: int = 15, tile_workers: int = 2,
                                sequence_length: int = 3,
                                start_date: str = "2024-01-01",
                                end_date: str = "2025-01-01") -> dict:
    """
    Run the generate_sta2_grid_parallel.py script to generate n_samples
    and measure the time taken.
    
    Args:
        region_name: Name of the region to process
        n_samples: Number of samples to generate
        coverage_workers: Parallel workers for coverage checking
        tile_workers: Parallel workers for tile processing
        sequence_length: Number of temporal frames per sequence
        start_date: Start date for image search (YYYY-MM-DD)
        end_date: End date for image search (YYYY-MM-DD)
    
    Returns:
        dict with timing statistics and sample info
    """
    print(f"\n{'='*70}")
    print(f"📊 BENCHMARK: Grid-Parallel Generation ({n_samples} samples)")
    print(f"{'='*70}")
    print(f"   Region: {region_name}")
    print(f"   Date range: {start_date} → {end_date}")
    print(f"   Coverage workers: {coverage_workers}")
    print(f"   Tile workers: {tile_workers}")
    print(f"   Sequence length: {sequence_length}")
    print(f"{'='*70}")
    
    # Create unique output/cache directories for this benchmark
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"./benchmark_output_{timestamp}"
    cache_dir = f"./benchmark_cache_{timestamp}"
    
    # Build command to run the grid parallel script
    script_path = Path(__file__).parent.parent / "preprocessing" / "generate_sta2_grid_parallel.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--max-samples", str(n_samples),
        "--output-dir", output_dir,
        "--cache-dir", cache_dir,
        "--sequence-length", str(sequence_length),
        "--coverage-threshold", "90.0",
        "--tile-workers", str(tile_workers),
        "--coverage-workers", str(coverage_workers),
        "--region", region_name,
        "--start-date", start_date,
        "--end-date", end_date
    ]
    
    print(f"\n   Running command:")
    print(f"   {' '.join(cmd)}")
    print(f"\n" + "-"*70)
    
    # Run the benchmark
    total_start = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=str(Path(__file__).parent.parent.parent)
        )
        
        exit_code = result.returncode
        
    except Exception as e:
        print(f"\n❌ Error running benchmark: {e}")
        exit_code = -1
    
    total_end = time.time()
    total_time = total_end - total_start
    
    # Check for generated samples
    samples_generated = 0
    sample_shapes = []
    
    dataset_path = Path(output_dir) / f"{region_name.replace('/', '-')}_dataset.pkl"
    if dataset_path.exists():
        try:
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            samples = dataset.get('samples', [])
            samples_generated = len(samples)
            
            # Get sample shapes
            for sample in samples[:5]:  # First 5 samples
                sample_shapes.append({
                    'lr_shape': sample.lr_sequence.shape,
                    'hr_shape': sample.hr_sequence.shape,
                    'dates': sample.temporal_dates,
                    'band_mode': sample.tile_bounds.get('band_mode', 'unknown')
                })
        except Exception as e:
            print(f"   ⚠️ Could not load dataset: {e}")
    
    # Calculate statistics
    results = {
        "region": region_name,
        "requested_samples": n_samples,
        "generated_samples": samples_generated,
        "total_time_seconds": total_time,
        "avg_time_per_sample": total_time / max(samples_generated, 1),
        "coverage_workers": coverage_workers,
        "tile_workers": tile_workers,
        "sequence_length": sequence_length,
        "exit_code": exit_code,
        "sample_shapes": sample_shapes
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"   Samples generated: {samples_generated}/{n_samples}")
    print(f"   Total time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    print(f"   Avg time per sample: {results['avg_time_per_sample']:.1f} seconds")
    
    if sample_shapes:
        print(f"\n   Sample shapes (first {len(sample_shapes)}):")
        for i, shape in enumerate(sample_shapes):
            print(f"      [{i+1}] LR: {shape['lr_shape']}, HR: {shape['hr_shape']}, Mode: {shape['band_mode']}")
    
    # Cleanup benchmark directories
    try:
        import shutil
        if Path(output_dir).exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
    except:
        pass
    
    return results


def estimate_total_processing_time(benchmark_result: dict, all_sta_counts: list) -> dict:
    """
    Estimate total processing time based on benchmark results.
    """
    avg_time_per_sample = benchmark_result['avg_time_per_sample']
    sequence_length = benchmark_result['sequence_length']
    
    # Calculate total samples per STA
    sta_summaries = []
    total_pairs = 0
    total_sequences = 0
    
    for sta in all_sta_counts:
        if sta.get('success', False):
            pairs = sta['matched_pairs']
            # Number of sequences = pairs - (sequence_length - 1)
            sequences = max(0, pairs - (sequence_length - 1))
            
            # Estimate tiles per sequence (based on typical coverage ~50 tiles per sequence)
            # This is conservative - actual may be more or less depending on coverage
            tiles_per_seq = 50
            total_tiles = sequences * tiles_per_seq
            
            sta_summaries.append({
                "name": sta["name"],
                "period": f"{sta['start']} → {sta['end']}",
                "matched_pairs": pairs,
                "potential_sequences": sequences,
                "estimated_tiles": total_tiles,
                "estimated_time_minutes": (total_tiles * avg_time_per_sample) / 60
            })
            
            total_pairs += pairs
            total_sequences += sequences
    
    # Estimate total tiles (conservative: ~50 tiles per sequence)
    tiles_per_seq = 50
    total_tiles = total_sequences * tiles_per_seq
    
    total_time_seconds = total_tiles * avg_time_per_sample
    total_time_minutes = total_time_seconds / 60
    total_time_hours = total_time_minutes / 60
    
    return {
        "avg_time_per_sample_seconds": avg_time_per_sample,
        "total_matched_pairs": total_pairs,
        "total_sequences": total_sequences,
        "estimated_total_tiles": total_tiles,
        "tiles_per_sequence_estimate": tiles_per_seq,
        "estimated_time_seconds": total_time_seconds,
        "estimated_time_minutes": total_time_minutes,
        "estimated_time_hours": total_time_hours,
        "sta_details": sta_summaries
    }


def main():
    print("\n" + "="*80)
    print("🚀 SUPER RESOLUTION DATA GENERATION BENCHMARK")
    print("   Using: generate_sta2_grid_parallel.py")
    print("="*80)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Objective: Generate {N_SAMPLES} samples, estimate total processing time")
    print(f"\n   ⚙️  Configuration:")
    print(f"       Coverage workers: {COVERAGE_WORKERS}")
    print(f"       Tile workers: {TILE_WORKERS}")
    print(f"       Sequence length: {SEQUENCE_LENGTH}")
    print("="*80)
    
    # Step 1: Count pairs for all STAs
    print("\n" + "="*70)
    print("📋 STEP 1: Counting matched pairs for all STAs...")
    print("="*70)
    
    all_sta_counts = []
    for i, region in enumerate(regions):
        print(f"\n   [{i+1}/{len(regions)}] {region['name']} ({region['start']} → {region['end']})...", end=" ", flush=True)
        counts = count_pairs_for_region(region)
        all_sta_counts.append(counts)
        
        if counts.get('success'):
            print(f"✅ {counts['matched_pairs']} pairs")
        else:
            print(f"❌ Error: {counts.get('error', 'unknown')}")
    
    # Print STA summary
    print("\n" + "-"*70)
    print("📊 STA PAIR SUMMARY:")
    print("-"*70)
    
    total_all_pairs = sum(c.get('matched_pairs', 0) for c in all_sta_counts if c.get('success'))
    
    # Group by STA name
    sta_groups = {}
    for sta in all_sta_counts:
        name = sta['name']
        if name not in sta_groups:
            sta_groups[name] = []
        sta_groups[name].append(sta)
    
    for sta_name, entries in sta_groups.items():
        pairs_sum = sum(e.get('matched_pairs', 0) for e in entries if e.get('success'))
        years = [e['start'][:4] for e in entries]
        print(f"   {sta_name}: {pairs_sum} total pairs across {len(entries)} years ({', '.join(years)})")
    
    print(f"\n   📊 GRAND TOTAL: {total_all_pairs} matched pairs across all STAs")
    
    # Step 2: Run benchmark on first STA with data (SINGLE YEAR to avoid memory issues)
    print("\n" + "="*70)
    print(f"📋 STEP 2: Running benchmark ({N_SAMPLES} samples) with grid-parallel pipeline...")
    print("   ⚠️  Using SINGLE YEAR (2024) to avoid GEE memory limits")
    print("="*70)
    
    # Find first region with sufficient pairs - use 2024 year specifically
    benchmark_region_name = None
    benchmark_start_date = "2024-01-01"
    benchmark_end_date = "2025-01-01"
    
    for i, region in enumerate(regions):
        if (all_sta_counts[i].get('success') and 
            all_sta_counts[i].get('matched_pairs', 0) >= SEQUENCE_LENGTH and
            region['start'] == benchmark_start_date):
            benchmark_region_name = region['name']
            break
    
    # Fallback: use first region with data regardless of year
    if benchmark_region_name is None:
        for i, region in enumerate(regions):
            if all_sta_counts[i].get('success') and all_sta_counts[i].get('matched_pairs', 0) >= SEQUENCE_LENGTH:
                benchmark_region_name = region['name']
                benchmark_start_date = region['start']
                benchmark_end_date = region['end']
                break
    
    if benchmark_region_name is None:
        print("❌ No regions with sufficient pairs found!")
        return
    
    benchmark_result = run_grid_parallel_benchmark(
        region_name=benchmark_region_name,
        n_samples=N_SAMPLES,
        coverage_workers=COVERAGE_WORKERS,
        tile_workers=TILE_WORKERS,
        sequence_length=SEQUENCE_LENGTH,
        start_date=benchmark_start_date,
        end_date=benchmark_end_date
    )
    
    # Step 3: Estimate total time
    print("\n" + "="*70)
    print("📋 STEP 3: Estimating total processing time...")
    print("="*70)
    
    estimates = estimate_total_processing_time(benchmark_result, all_sta_counts)
    
    print("\n📊 PER-STA ESTIMATES:")
    print("-"*70)
    
    for sta in estimates['sta_details']:
        print(f"   {sta['name']} ({sta['period']}):")
        print(f"      Matched pairs: {sta['matched_pairs']}")
        print(f"      Potential sequences: {sta['potential_sequences']}")
        print(f"      Estimated tiles: ~{sta['estimated_tiles']}")
        print(f"      Estimated time: {sta['estimated_time_minutes']:.1f} minutes")
    
    print("\n" + "="*70)
    print("📊 TOTAL ESTIMATION SUMMARY")
    print("="*70)
    print(f"   Avg time per tile/sample: {estimates['avg_time_per_sample_seconds']:.1f} seconds")
    print(f"   Total matched pairs: {estimates['total_matched_pairs']}")
    print(f"   Total sequences: {estimates['total_sequences']}")
    print(f"   Estimated tiles per sequence: ~{estimates['tiles_per_sequence_estimate']}")
    print(f"   Estimated total tiles: ~{estimates['estimated_total_tiles']}")
    print(f"\n   ⏱️  ESTIMATED TOTAL TIME:")
    print(f"       {estimates['estimated_time_seconds']:.0f} seconds")
    print(f"       {estimates['estimated_time_minutes']:.1f} minutes")
    print(f"       {estimates['estimated_time_hours']:.2f} hours")
    
    # Note about configuration
    print("\n" + "-"*70)
    print("📝 NOTES:")
    print("-"*70)
    print(f"   • Using grid-based parallel pipeline with coverage pre-checking")
    print(f"   • Coverage workers: {COVERAGE_WORKERS} (parallel coverage checks)")
    print(f"   • Tile workers: {TILE_WORKERS} (parallel tile downloads)")
    print(f"   • Tiles per sequence estimate: ~{estimates['tiles_per_sequence_estimate']} (depends on coverage)")
    print(f"   • Coverage threshold: 90% (tiles with <90% valid pixels are skipped)")
    print(f"   • Actual time may vary based on GEE server load and network")
    
    # Completion
    print("\n" + "="*80)
    print(f"✅ BENCHMARK COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return {
        "benchmark": benchmark_result,
        "estimates": estimates,
        "all_sta_counts": all_sta_counts
    }


if __name__ == "__main__":
    results = main()
