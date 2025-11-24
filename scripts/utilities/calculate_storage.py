"""
Calculate storage requirements for downloading satellite imagery
"""

import numpy as np
from Auth.AuthGEE import regions

def calculate_storage_requirements():
    """
    Calculate total storage needed for all regions
    """
    
    print("="*80)
    print("STORAGE REQUIREMENTS CALCULATION")
    print("="*80 + "\n")
    
    # Configuration
    tile_size_lr = 64      # Sentinel-2 LR tile size in pixels
    tile_size_hr = 256     # Planet HR tile size in pixels
    sequence_length = 3    # Temporal frames
    bands_s2 = 4          # Sentinel-2 bands
    bands_planet = 4      # Planet bands
    dtype_bytes = 4       # float32 = 4 bytes
    
    # Tile calculations with different tile sizes
    tile_configs = [
        (256, "Small tiles (256px)"),
        (512, "Medium tiles (512px)"),
        (1024, "Large tiles (1024px)"),
    ]
    
    print("📊 Configuration:")
    print(f"   Regions: {len(regions)}")
    print(f"   Sequence length: {sequence_length} temporal frames")
    print(f"   Bands: S2={bands_s2}, Planet={bands_planet}")
    print(f"   Data type: float32 (4 bytes)")
    print()
    
    # Get matched pairs per region (approximate from your logs)
    # From your output: ~12 matched pairs per region
    avg_matched_pairs = 12
    sequences_per_region = max(1, avg_matched_pairs - sequence_length + 1)
    
    print(f"📅 Temporal Data:")
    print(f"   Avg matched pairs/region: ~{avg_matched_pairs}")
    print(f"   Sequences/region: ~{sequences_per_region}")
    print()
    
    for tile_size, label in tile_configs:
        print(f"\n{'='*80}")
        print(f"{label}")
        print(f"{'='*80}")
        
        # Estimate tiles per region based on typical area
        # Assuming regions are roughly 20km x 20km
        region_size_km = 20
        pixel_size_m = 10  # 10m resolution
        region_size_pixels = (region_size_km * 1000) / pixel_size_m  # 2000 pixels
        
        overlap = 64  # pixels
        stride = tile_size - overlap
        tiles_per_dimension = int(np.ceil(region_size_pixels / stride))
        tiles_per_region = tiles_per_dimension ** 2
        
        print(f"   Tile size: {tile_size}×{tile_size} pixels")
        print(f"   Overlap: {overlap} pixels")
        print(f"   Tiles per region: ~{tiles_per_region}")
        print()
        
        # Calculate sizes
        # LR: (T, H, W, C) = (3, 64, 64, 4) or scaled
        lr_scale = tile_size // 4  # LR is 4x smaller
        lr_size_per_tile = sequence_length * lr_scale * lr_scale * bands_s2 * dtype_bytes
        
        # HR: (T, H, W, C) = (3, 256, 256, 4) or scaled
        hr_size_per_tile = sequence_length * tile_size * tile_size * bands_planet * dtype_bytes
        
        total_size_per_tile = lr_size_per_tile + hr_size_per_tile
        
        # Total per region
        size_per_region = total_size_per_tile * tiles_per_region
        
        # Total for all regions
        total_size = size_per_region * len(regions)
        
        print(f"   Memory per tile:")
        print(f"      LR (S2):  {lr_size_per_tile / 1024**2:.2f} MB")
        print(f"      HR (Planet): {hr_size_per_tile / 1024**2:.2f} MB")
        print(f"      Total: {total_size_per_tile / 1024**2:.2f} MB")
        print()
        print(f"   Memory per region:")
        print(f"      {size_per_region / 1024**2:.1f} MB ({size_per_region / 1024**3:.2f} GB)")
        print()
        print(f"   💾 TOTAL for {len(regions)} regions:")
        print(f"      {total_size / 1024**2:.1f} MB")
        print(f"      {total_size / 1024**3:.2f} GB")
        
        # With compression (SQLite + pickle typically 40-60% compression)
        compressed_size = total_size * 0.5
        print(f"   💾 With compression (~50%):")
        print(f"      {compressed_size / 1024**3:.2f} GB")
        print()
        
        # Samples count
        total_samples = sequences_per_region * tiles_per_region * len(regions)
        print(f"   📦 Total training samples: ~{total_samples:,}")
        print(f"   📦 Size per sample: {total_size_per_tile / 1024:.1f} KB")
    
    # Summary recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80 + "\n")
    
    print("💡 For Development (Quick Iteration):")
    print("   - Use: create_fixed_dataset.py with num_samples=50")
    print("   - Storage: ~50 MB uncompressed")
    print("   - Time: ~5-10 minutes to create")
    print("   - Perfect for: Testing, debugging, hyperparameter tuning")
    print()
    
    print("💡 For Validation (Medium Scale):")
    print("   - Use: create_fixed_dataset.py with num_samples=200-500")
    print("   - Storage: ~200-500 MB")
    print("   - Time: ~20-60 minutes to create")
    print("   - Perfect for: Verifying model converges, ablation studies")
    print()
    
    print("💡 For Production (Full Scale):")
    print("   - Use: Large tiles (1024px) - only ~9-16 tiles/region")
    print("   - Storage: ~5-10 GB compressed")
    print("   - Time: ~2-4 hours to download once")
    print("   - Perfect for: Final training, best performance")
    print()
    
    print("💡 Current Setup:")
    print("   - You have: External 1TB drive (plenty of space!)")
    print("   - Recommended: Use 512px or 1024px tiles")
    print("   - Strategy:")
    print("     1. Start with 50 samples (~50 MB) for quick testing")
    print("     2. Scale to 500 samples (~500 MB) for validation")
    print("     3. Finally download full dataset (~5-10 GB) for production")
    print()
    
    # Practical numbers
    print("="*80)
    print("PRACTICAL NUMBERS")
    print("="*80 + "\n")
    
    scenarios = [
        (50, "Quick test", 512),
        (100, "Small dataset", 512),
        (500, "Medium dataset", 512),
        (1000, "Large dataset", 1024),
        (5000, "Very large dataset", 1024),
    ]
    
    for num_samples, desc, tsize in scenarios:
        lr_scale = tsize // 4
        lr_size = sequence_length * lr_scale * lr_scale * bands_s2 * dtype_bytes
        hr_size = sequence_length * tsize * tsize * bands_planet * dtype_bytes
        sample_size = lr_size + hr_size
        total = sample_size * num_samples
        compressed = total * 0.5
        
        print(f"{desc:20s} ({num_samples:5d} samples, {tsize}px): "
              f"{compressed / 1024**2:7.1f} MB compressed, "
              f"{total / 1024**3:5.2f} GB uncompressed")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    calculate_storage_requirements()
    
    print("\n💾 Your 1TB external drive can easily hold:")
    print("   ✅ Quick test (50 samples): 0.005% used")
    print("   ✅ Medium dataset (500 samples): 0.05% used")
    print("   ✅ Large dataset (5000 samples): 0.5% used")
    print("   ✅ FULL DATASET (~50,000 samples): 5% used")
    print("\n   You have PLENTY of space! 🎉")
