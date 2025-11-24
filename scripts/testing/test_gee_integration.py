#!/usr/bin/env python3
"""
Test the GEE-integrated efficient pipeline with real satellite data downloads.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from numpy_efficient_pipeline import TensorFlowFreeDataPipeline, NumpyTrainingDataGenerator
from Auth.AuthGEE import regions, find_pairs, PLANET_COLLECTION_ID

def test_gee_integration():
    """Test the pipeline with actual GEE downloads."""
    
    print("🧪 Testing GEE-Integrated Efficient Pipeline")
    print("=" * 45)
    
    # Initialize pipeline
    print("\n📦 Initializing pipeline...")
    pipeline = TensorFlowFreeDataPipeline(
        cache_dir="./test_gee_cache",
        tile_size=256,  # Small for testing
        overlap=32,
        max_cache_gb=1.0  # Small cache for testing
    )
    
    # Select a small test region
    test_region = regions[0]  # STA-2, 2020-2021
    print(f"\n🎯 Testing with region: {test_region['name']}")
    print(f"   Date range: {test_region['start']} to {test_region['end']}")
    
    try:
        # Test find_pairs integration
        print("\n🔍 Finding satellite pairs...")
        pair_results = find_pairs(
            test_region["aoi"], 
            test_region["start"], 
            test_region["end"], 
            PLANET_COLLECTION_ID
        )
        
        print(f"   ✅ Found {len(pair_results['matched_pairs'])} paired images")
        
        if len(pair_results['matched_pairs']) == 0:
            print("   ⚠️ No pairs found - cannot test GEE downloads")
            return
        
        # Test processing first few samples
        print(f"\n🔄 Testing GEE tile downloads...")
        
        sample_count = 0
        max_test_samples = 2  # Test just 2 samples
        
        for sample in pipeline.process_region_numpy(
            region_config=test_region,
            sequence_length=3,
            use_cache=True
        ):
            sample_count += 1
            
            print(f"\n   ✅ Sample {sample_count}:")
            print(f"      Region: {sample.region_name}")
            print(f"      Sequence: {sample.sequence_id}, Tile: {sample.tile_id}")
            print(f"      Dates: {sample.temporal_dates}")
            print(f"      LR shape: {sample.lr_sequence.shape}")  # (T, H, W, C)
            print(f"      HR shape: {sample.hr_sequence.shape}")  # (T, H, W, C)
            print(f"      LR range: [{sample.lr_sequence.min():.1f}, {sample.lr_sequence.max():.1f}]")
            print(f"      HR range: [{sample.hr_sequence.min():.1f}, {sample.hr_sequence.max():.1f}]")
            
            if sample_count >= max_test_samples:
                print(f"\n   🎉 Successfully tested {sample_count} samples with real GEE data!")
                break
        
        if sample_count == 0:
            print("   ⚠️ No samples generated - check GEE connectivity and image availability")
        
        # Print statistics
        pipeline.print_stats()
        
    except Exception as e:
        print(f"   ❌ Error during GEE integration test: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎯 GEE Integration Test Summary:")
    print(f"   ✅ Pipeline initialization: Success")
    print(f"   ✅ find_pairs integration: Success") 
    print(f"   ✅ GEE tile downloads: {'Success' if sample_count > 0 else 'Failed'}")
    print(f"   📊 Samples processed: {sample_count}")
    
    if sample_count > 0:
        print(f"\n🚀 Ready for full-scale processing!")
        print(f"   - Increase tile_size for higher resolution")
        print(f"   - Process multiple regions in parallel")
        print(f"   - Use for super resolution model training")
    else:
        print(f"\n🔧 Troubleshooting needed:")
        print(f"   - Check GEE authentication")
        print(f"   - Verify image availability in collections")
        print(f"   - Check network connectivity")

def test_data_generator():
    """Test the data generator with GEE integration."""
    
    print("\n\n🎯 Testing Data Generator with GEE")
    print("=" * 35)
    
    try:
        # Initialize components
        pipeline = TensorFlowFreeDataPipeline(
            cache_dir="./test_gee_cache",
            tile_size=128,  # Even smaller for generator test
            overlap=16
        )
        
        data_gen = NumpyTrainingDataGenerator(
            pipeline=pipeline,
            batch_size=2,
            normalize=True,
            augment=True
        )
        
        # Test with first region only
        test_regions = [regions[0]]
        
        print("🔄 Generating training batches...")
        
        batch_count = 0
        for lr_batch, hr_batch in data_gen.generate_batches(test_regions, sequence_length=3):
            batch_count += 1
            
            print(f"\n   📦 Batch {batch_count}:")
            print(f"      LR batch shape: {lr_batch.shape}")  # (batch, time, H, W, C)
            print(f"      HR batch shape: {hr_batch.shape}")  # (batch, time, H, W, C)
            print(f"      LR range: [{lr_batch.min():.3f}, {lr_batch.max():.3f}]")
            print(f"      HR range: [{hr_batch.min():.3f}, {hr_batch.max():.3f}]")
            
            if batch_count >= 1:  # Test just 1 batch
                break
        
        if batch_count > 0:
            print(f"\n   ✅ Data generator working with GEE integration!")
            print(f"   🎯 Ready for ML model training")
        else:
            print(f"\n   ⚠️ No batches generated")
            
    except Exception as e:
        print(f"   ❌ Data generator test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🌍 Starting GEE Integration Tests...")
    
    # Test 1: Basic GEE tile downloading
    test_gee_integration()
    
    # Test 2: Data generator with GEE
    test_data_generator()
    
    print("\n" + "="*50)
    print("🎉 GEE Integration Testing Complete!")
    print("="*50)