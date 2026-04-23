"""
Test integrated pipeline with chunked downloads
Quick test with 2 samples to verify everything works
"""
import os
import sys
import certifi

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from numpy_efficient_pipeline import TensorFlowFreeDataPipeline
from Auth.AuthGEE import regions

print("="*70)
print("INTEGRATED PIPELINE TEST")
print("Testing chunked download integration with 2 samples")
print("="*70)

# Use first region
region = regions[0]

print(f"\nRegion: {region['name']}")
print(f"Start: {region['start']}")
print(f"End: {region['end']}")
print(f"\nProcessing with 2 samples (quick test)...\n")

# Create pipeline instance
pipeline = TensorFlowFreeDataPipeline(
    cache_dir="test_chunked_pipeline_cache",
    tile_size=256
)

try:
    # Process region (it's a generator, so we need to collect samples)
    print("Starting pipeline processing...")
    sample_generator = pipeline.process_region_numpy(
        region,
        sequence_length=1  # Single frame for speed
    )
    
    # Collect first 2 samples
    results = []
    for i, sample in enumerate(sample_generator):
        if i >= 2:  # Stop after 2 samples
            break
        results.append(sample)
        print(f"   Collected sample {i+1}/2")
    
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}\n")
    
    if results and len(results) > 0:
        print(f"✅ SUCCESS! Generated {len(results)} training samples\n")
        
        # Validate first sample
        sample = results[0]
        print(f"Sample 1 validation:")
        print(f"  HR shape: {sample.hr_sequence.shape}")
        print(f"  LR shape: {sample.lr_sequence.shape}")
        print(f"  HR stats: min={sample.hr_sequence.min():.1f}, max={sample.hr_sequence.max():.1f}, mean={sample.hr_sequence.mean():.1f}")
        print(f"  LR stats: min={sample.lr_sequence.min():.1f}, max={sample.lr_sequence.max():.1f}, mean={sample.lr_sequence.mean():.1f}")
        
        # Check for real data (not zeros or fallbacks)
        hr_valid_pct = 100 * (sample.hr_sequence.sum(axis=-1) > 0).sum() / (sample.hr_sequence.shape[0] * sample.hr_sequence.shape[1])
        lr_valid_pct = 100 * (sample.lr_sequence.sum(axis=-1) > 0).sum() / (sample.lr_sequence.shape[0] * sample.lr_sequence.shape[1])
        
        print(f"  HR valid pixels: {hr_valid_pct:.1f}%")
        print(f"  LR valid pixels: {lr_valid_pct:.1f}%")
        
        # Check for fallback pattern
        hr_means = [sample.hr_sequence[:,:,:,i].mean() for i in range(sample.hr_sequence.shape[-1])]
        import numpy as np
        fallback_pattern = np.array([300, 500, 700, 600])
        
        if np.allclose(hr_means[:4], fallback_pattern, rtol=0.1):
            print(f"  ❌ WARNING: HR matches fallback pattern {fallback_pattern}")
        else:
            print(f"  ✅ HR has real values (not fallback)")
        
        # Final verdict
        if (hr_valid_pct > 30 and lr_valid_pct > 30 and 
            not np.allclose(hr_means[:4], fallback_pattern, rtol=0.1)):
            print(f"\n🎉 ALL CHECKS PASSED!")
            print(f"✅ Pipeline successfully integrated with chunked downloads")
            print(f"✅ HR and LR tiles contain real satellite data")
            print(f"✅ Ready for production dataset generation")
        else:
            print(f"\n⚠️  Some validation checks failed")
            print(f"   Review data quality above")
            
    else:
        print(f"❌ FAILED: No samples generated")
        
except Exception as e:
    print(f"\n❌ ERROR during processing:")
    print(f"   {str(e)}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
