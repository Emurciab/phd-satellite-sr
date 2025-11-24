"""
Comparison: Interpolated vs Multi-Resolution Approach
Demonstrates why interpolating B8A is problematic
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def simulate_interpolation_artifact():
    """
    Simulate how interpolation corrupts NIR signals
    """
    
    print("="*70)
    print("DEMONSTRATION: INTERPOLATION ARTIFACTS IN B8A")
    print("="*70)
    print()
    
    # Simulate native 20m B8A values (vegetation with spatial variation)
    # Values represent actual NIR reflectance (0-10000)
    native_20m = np.array([
        [8500, 8200, 7900, 7600],
        [8300, 8000, 7700, 7400],
        [8100, 7800, 7500, 7200],
        [7900, 7600, 7300, 7000]
    ])
    
    print("1️⃣  NATIVE 20m B8A (True Sensor Data)")
    print("   Shape:", native_20m.shape)
    print("   Values:\n", native_20m)
    print()
    
    # Interpolate to 10m (current wrong approach)
    # This is what happens in GEE when we resample to scale=10
    interpolated_10m = ndimage.zoom(native_20m, 2, order=1)  # Bilinear
    
    print("2️⃣  INTERPOLATED to 10m (Current Approach - WRONG)")
    print("   Shape:", interpolated_10m.shape)
    print("   Values (first 4x4):\n", interpolated_10m[:4, :4].astype(int))
    print()
    print("   ⚠️  Notice: Artificial values created (e.g., 8350, 8050)")
    print("   ⚠️  These don't represent real measurements!")
    print()
    
    # What SR model learns with interpolated data
    print("3️⃣  PROBLEM: SR Model Training")
    print("   Input (LR):  Interpolated 10m values [8500, 8350, 8200, ...]")
    print("   Target (HR): Interpolated 3m values (even more artificial)")
    print("   Learned:     How to reproduce interpolation, not true SR!")
    print()
    
    # Multi-resolution approach
    print("4️⃣  SOLUTION: Multi-Resolution Approach")
    print("   10m Model:")
    print("     Input:  B2,B3,B4 at native 10m")
    print("     Target: B2,B3,B4 at native 3m")
    print("     Learns: True 10m→3m degradation pattern")
    print()
    print("   20m Model:")
    print("     Input:  B8A at native 20m (no interpolation!)")
    print("     Target: B8A at 6.67m (Planet 3m downsampled 2x)")
    print("     Learns: True 20m→6.67m degradation pattern")
    print()
    
    # Impact on vegetation indices
    print("5️⃣  IMPACT ON VEGETATION INDICES")
    print()
    
    # Simulate Red band at 10m (for NDVI)
    red_10m = np.array([
        [2500, 2400, 2300],
        [2450, 2350, 2250],
        [2400, 2300, 2200]
    ])
    
    # Native 20m NIR (downsampled properly)
    nir_20m_native = native_20m[:3, :3]
    
    # Interpolated 10m NIR (wrong)
    nir_10m_interp = interpolated_10m[:3, :3]
    
    # NDVI with native resolution (need to align)
    # Downsample Red to 20m for fair comparison
    red_20m = red_10m[::2, ::2]
    nir_20m_for_ndvi = nir_20m_native[::2, ::2]
    
    ndvi_native = (nir_20m_for_ndvi - red_20m) / (nir_20m_for_ndvi + red_20m)
    
    # NDVI with interpolated (wrong)
    # Use interpolated NIR at 10m directly
    red_10m_subset = red_10m[:2, :2]
    nir_10m_subset = nir_10m_interp[:2, :2]
    ndvi_interpolated = (nir_10m_subset - red_10m_subset) / (nir_10m_subset + red_10m_subset)
    
    print("   NDVI from Native 20m:")
    print("   ", ndvi_native)
    print()
    print("   NDVI from Interpolated 10m:")
    print("   ", ndvi_interpolated)
    print()
    print("   ⚠️  Difference: Up to", np.max(np.abs(ndvi_native - ndvi_interpolated[:2, :2])))
    print("   ⚠️  This affects biomass estimation, vegetation health analysis!")
    print()
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Native 20m
    im1 = axes[0].imshow(native_20m, cmap='RdYlGn', interpolation='nearest')
    axes[0].set_title('Native 20m B8A\n(True Sensor Data)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('4x4 pixels at 20m resolution')
    plt.colorbar(im1, ax=axes[0], label='Reflectance')
    
    # Interpolated 10m
    im2 = axes[1].imshow(interpolated_10m, cmap='RdYlGn', interpolation='nearest')
    axes[1].set_title('Interpolated to 10m\n(Current Approach - WRONG)', 
                      fontsize=12, fontweight='bold', color='red')
    axes[1].set_xlabel('8x8 pixels at 10m resolution')
    plt.colorbar(im2, ax=axes[1], label='Reflectance')
    
    # Difference (artifact magnitude)
    # Downsample interpolated back to 20m to compare
    downsampled = interpolated_10m[::2, ::2]
    diff = np.abs(downsampled - native_20m)
    im3 = axes[2].imshow(diff, cmap='Reds', interpolation='nearest')
    axes[2].set_title('Interpolation Error\n(Artifact Magnitude)', 
                      fontsize=12, fontweight='bold')
    axes[2].set_xlabel('4x4 pixels - Error from interpolation')
    plt.colorbar(im3, ax=axes[2], label='Absolute Error')
    
    plt.tight_layout()
    plt.savefig('interpolation_artifact_demo.png', dpi=150, bbox_inches='tight')
    print("📊 Visualization saved to: interpolation_artifact_demo.png")
    print()
    
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print()
    print("❌ INTERPOLATED APPROACH:")
    print("   • Creates artificial NIR values")
    print("   • Model learns to reproduce interpolation, not true SR")
    print("   • Corrupts vegetation indices (NDVI, NDRE)")
    print("   • Inflates metrics on interpolated test data")
    print("   • Poor generalization to native 20m data")
    print()
    print("✅ MULTI-RESOLUTION APPROACH:")
    print("   • Preserves native 20m B8A characteristics")
    print("   • Separate models for 10m and 20m bands")
    print("   • Learns true sensor degradation patterns")
    print("   • Accurate vegetation indices for STA analysis")
    print("   • Realistic metrics, better generalization")
    print()
    print("🎯 RECOMMENDATION:")
    print("   Use train_multi_resolution_sr.py for training")
    print("   Train separate models: 10m (RGB) + 20m (NIR)")
    print()
    print("="*70)


def compare_scale_factors():
    """
    Compare scale factors for different approaches
    """
    
    print("\n" + "="*70)
    print("SCALE FACTOR COMPARISON")
    print("="*70)
    print()
    
    print("📏 INTERPOLATED APPROACH (Current - Wrong):")
    print("   All bands trained together at 10m:")
    print("   • B2,B3,B4:  10m → 3m    (scale: 3.33x) ✓")
    print("   • B8A:       10m → 3m    (scale: 3.33x) ✗ WRONG!")
    print("                ↑              ↑")
    print("           interpolated    interpolated")
    print("           from 20m!       from 3m!")
    print()
    print("   Problem: B8A never had true 10m data!")
    print()
    
    print("📏 MULTI-RESOLUTION APPROACH (Correct):")
    print("   10m Model (RGB):")
    print("   • B2,B3,B4:  10m → 3m       (scale: 3.33x) ✓")
    print("                ↑       ↑")
    print("              native  native")
    print()
    print("   20m Model (NIR):")
    print("   • B8A:       20m → 6.67m    (scale: 3.00x) ✓")
    print("                ↑       ↑")
    print("              native  Planet 3m downsampled 2x")
    print()
    print("   Benefits:")
    print("   • Each model learns native resolution patterns")
    print("   • Similar scale factors (~3x) for both models")
    print("   • No interpolation artifacts in training data")
    print("   • Spectral integrity preserved")
    print()
    
    print("🎯 KEY INSIGHT:")
    print("   Sentinel-2 B8A is acquired at 20m resolution by the sensor.")
    print("   Any 10m B8A data you see has been interpolated somewhere.")
    print("   Training SR on interpolated data teaches the model to")
    print("   reproduce interpolation, not perform true super-resolution!")
    print()
    print("="*70)


def main():
    """Run demonstrations"""
    simulate_interpolation_artifact()
    compare_scale_factors()
    
    print("\n" + "="*70)
    print("📚 NEXT STEPS")
    print("="*70)
    print()
    print("1. Review visualization: interpolation_artifact_demo.png")
    print()
    print("2. After dataset regeneration (bug fix), convert to multi-resolution:")
    print("   python scripts/training/train_multi_resolution_sr.py \\")
    print("       --dataset-path data/datasets/sta2_grid_parallel/dataset.pkl \\")
    print("       --output-dir sr_models_multi_res \\")
    print("       --epochs 50")
    print()
    print("3. Compare training curves: Interpolated vs Multi-Resolution")
    print()
    print("4. Evaluate on native 20m test data (not interpolated!)")
    print()
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
