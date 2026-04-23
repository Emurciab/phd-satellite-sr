import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Step 1: Import preprocessing pipeline
# print("\n📦 Step 1: Loading preprocessing pipeline...")
try:
    from numpy_efficient_pipeline import TensorFlowFreeDataPipeline, NumpyTrainingDataGenerator
    from Auth.AuthGEE import regions, find_pairs, PLANET_COLLECTION_ID
except ImportError as e:
    print(f"   ❌ Error loading preprocessing: {e}")
    sys.exit(1)

# Step 2: Check if PyTorch is available
# print("\n📦 Step 2: Checking PyTorch availability...")
try:
    import torch
    import torch.nn as nn
    print(f"   ✅ PyTorch {torch.__version__} available")
    print(f"   GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("   ⚠️ PyTorch not installed in current environment")
    print("   💡 To install PyTorch, run:")
    print("      conda activate tf_gpu  # or your PyTorch environment")
    print("      pip install torch torchvision")
    print("\n   ℹ️  Continuing with NumPy-only testing...")
    torch = None

# Step 3: Load the improved model
# print("\n📦 Step 3: Loading ImprovedModel...")
try:
    if torch is not None:
        from ImprovedModel import create_improved_sr_model, ImprovedSatelliteSR, SpectralFidelityLoss
    else:
        print("   ⚠️ Skipping model import (PyTorch not available)")
except Exception as e:
    print(f"   ❌ Error loading ImprovedModel: {e}")
    if torch is not None:
        sys.exit(1)

# Step 4: Initialize preprocessing pipeline
# print("\n🔧 Step 4: Initializing preprocessing pipeline...")
try:
    pipeline = TensorFlowFreeDataPipeline(
        cache_dir="./test_model_cache",
        tile_size=256,  # Moderate size for testing
        overlap=32,
        max_cache_gb=2.0
    )
except Exception as e:
    print(f"   ❌ Error initializing pipeline: {e}")
    sys.exit(1)






# Step 5: Get one test region
print("\n🌍 Step 5: Loading test region...")
try:
    if len(regions) == 0:
        print("   ⚠️ No regions defined in AuthGEE.py")
        sys.exit(1)
    
    test_region = regions[0]
    print(f"   Region: {test_region['name']}")
    print(f"   Period: {test_region['start']} to {test_region['end']}")
except Exception as e:
    print(f"   ❌ Error loading region: {e}")
    sys.exit(1)

# Step 6: Process one sample from the region
print("\n📥 Step 6: Fetching ONE image pair from GEE...")
print("   (This may take 30-60 seconds on first run)")
try:
    sample_count = 0
    test_sample = None
    
    for sample in pipeline.process_region_numpy(
        region_config=test_region,
        sequence_length=3,  # 3-frame temporal sequence
        use_cache=True
    ):
        test_sample = sample
        sample_count += 1
        print(f"   ✅ Retrieved sample {sample_count}")
    
    if test_sample is None:
        print("   ❌ No samples retrieved. Check:")
        print("      - GEE authentication")
        print("      - Image availability for region/dates")
        print("      - Network connection")
        sys.exit(1)
    
    # print(f"\n   📊 Sample details:")
    # print(f"      LR shape: {test_sample.lr_sequence.shape}")  # (T, H, W, C)
    # print(f"      HR shape: {test_sample.hr_sequence.shape}")  # (T, H, W, C)
    # print(f"      Temporal frames: {len(test_sample.temporal_dates)}")
    # print(f"      Dates: {test_sample.temporal_dates}")
    # print(f"      Band count: {test_sample.lr_sequence.shape[-1]}")
    
except Exception as e:
    print(f"   ❌ Error processing region: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: Prepare data for model
# print("\n🔄 Step 7: Preparing data for model...")
try:
    # Get shapes
    T, H_lr, W_lr, C = test_sample.lr_sequence.shape
    T_hr, H_hr, W_hr, C_hr = test_sample.hr_sequence.shape
    
    # print(f"   Input (LR): T={T}, H={H_lr}, W={W_lr}, C={C}")
    # print(f"   Target (HR): T={T_hr}, H={H_hr}, W={W_hr}, C={C_hr}")
    
    # Check data validity
    lr_min, lr_max = test_sample.lr_sequence.min(), test_sample.lr_sequence.max()
    hr_min, hr_max = test_sample.hr_sequence.min(), test_sample.hr_sequence.max()

    # print(f"\n   📈 Data statistics:")
    # print(f"      LR range: [{lr_min:.2f}, {lr_max:.2f}]")
    # print(f"      HR range: [{hr_min:.2f}, {hr_max:.2f}]")

    if lr_max > 10000 or hr_max > 10000:
        print("   ⚠️ Data appears to be in raw reflectance (not normalized)")
    elif lr_max <= 1.0 and hr_max <= 1.0:
        print("   ✅ Data appears to be normalized [0, 1]")
    
    # Check for NaN or Inf
    if np.isnan(test_sample.lr_sequence).any():
        print("   ⚠️ Warning: LR contains NaN values")
    if np.isnan(test_sample.hr_sequence).any():
        print("   ⚠️ Warning: HR contains NaN values")
    if np.isinf(test_sample.lr_sequence).any():
        print("   ⚠️ Warning: LR contains Inf values")
    if np.isinf(test_sample.hr_sequence).any():
        print("   ⚠️ Warning: HR contains Inf values")
    
except Exception as e:
    print(f"   ❌ Error analyzing data: {e}")
    sys.exit(1)

# Step 8: Test with PyTorch model (if available)
if torch is not None:
    print("\n🧪 Step 8: Testing with PyTorch model...")
    try:
        # Create model configuration
        config = {
            'base_channels': 64,
            'scale_factor': max(2, H_hr // H_lr),  # Compute actual scale
            'use_temporal': True,
            'use_mixed_precision': False  # Use FP32 for testing
        }
        
        print(f"   Model config: {config}")
        
        # Create model
        # print("   Creating model...")
        model, loss_fn = create_improved_sr_model(config)
        model.eval()  # Set to evaluation mode
        
        # print(f"   ✅ Model created successfully")
        
        # Prepare tensors
        # Model expects: (B, T, C, H, W)
        # NumPy data is: (T, H, W, C)
        lr_tensor = torch.from_numpy(test_sample.lr_sequence).float()
        hr_tensor = torch.from_numpy(test_sample.hr_sequence).float()
        
        # Rearrange dimensions: (T, H, W, C) -> (1, T, C, H, W)
        lr_tensor = lr_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W)
        hr_tensor = hr_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W)
        
        # print(f"\n   📊 Tensor shapes:")
        # print(f"      LR input: {lr_tensor.shape}")
        # print(f"      HR target: {hr_tensor.shape}")
        
        # Normalize if needed (model expects [0, 1])
        if lr_tensor.max() > 1.0:
            print("   Normalizing data to [0, 1]...")
            lr_tensor = torch.clamp(lr_tensor / 10000.0, 0, 1)
            hr_tensor = torch.clamp(hr_tensor / 10000.0, 0, 1)
        
        # Forward pass
        # print("\n   🚀 Running forward pass...")
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        
        print(f"   ✅ Forward pass successful!")
        # print(f"      Output shape: {sr_tensor.shape}")
        # print(f"      Output range: [{sr_tensor.min().item():.4f}, {sr_tensor.max().item():.4f}]")
        
        # Check output validity
        if torch.isnan(sr_tensor).any():
            print("   ❌ ERROR: Output contains NaN values!")
        elif torch.isinf(sr_tensor).any():
            print("   ❌ ERROR: Output contains Inf values!")
        elif sr_tensor.min() < 0 or sr_tensor.max() > 1:
            print(f"   ⚠️ Warning: Output outside [0, 1] range")
        else:
            print("   ✅ Output is valid (no NaN/Inf, in [0, 1] range)")
        
        # Compute loss
        print("\n   📉 Computing spectral fidelity loss...")
        try:
            total_loss, loss_components = loss_fn(sr_tensor, hr_tensor, lr_tensor, config['scale_factor'])
            
            # print(f"   ✅ Loss computation successful!")
            # print(f"\n   Loss components:")
            # print(f"      Total loss: {total_loss.item():.6f}")
            for name, value in loss_components.items():
                print(f"      - {name}: {value:.6f}")
        except Exception as e:
            print(f"   ⚠️ Error computing loss: {e}")
        
        # Compare resolutions
        # print(f"\n   📐 Resolution comparison:")
        # print(f"      Input LR: {lr_tensor.shape[-2]} × {lr_tensor.shape[-1]}")
        # print(f"      Output SR: {sr_tensor.shape[-2]} × {sr_tensor.shape[-1]}")
        # print(f"      Target HR: {hr_tensor.shape[-2]} × {hr_tensor.shape[-1]}")
        # print(f"      Scale achieved: {sr_tensor.shape[-2] / lr_tensor.shape[-2]:.2f}x")
        
        # Model statistics
        print(f"\n   💾 Model statistics:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"      Total parameters: {total_params:,}")
        print(f"      Trainable parameters: {trainable_params:,}")
        print(f"      Model size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
        
    except Exception as e:
        print(f"   ❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠️ Step 8: Skipped (PyTorch not available)")
    print("   To run full model test:")
    print("   1. Install PyTorch: conda activate tf_gpu && pip install torch")
    print("   2. Run this script again")

# Summary
# print("\n" + "="*80)
# print("TEST SUMMARY")
# print("="*80)
# print("\n✅ Preprocessing Pipeline: WORKING")
# print(f"   - Retrieved {sample_count} sample(s) from GEE")
# print(f"   - LR shape: {test_sample.lr_sequence.shape}")
# print(f"   - HR shape: {test_sample.hr_sequence.shape}")
# print(f"   - Band count: {C} {'(4-band Dove)' if C == 4 else '(6-band SuperDove)' if C == 6 else ''}")

# if torch is not None:
#     print("\n✅ Model Testing: COMPLETE")
#     print("   - Model handles variable band counts: ✅")
#     print("   - Forward pass successful: ✅")
#     print("   - Output is valid: ✅")
#     print("   - Loss computation working: ✅")
# else:
#     print("\n⚠️ Model Testing: SKIPPED (PyTorch not installed)")

# print("\n💡 Next Steps:")
# print("   1. Review the test results above")
# print("   2. If all tests passed, proceed to full training")
# print("   3. Use ImprovedModel.py for your SR training pipeline")
# print("   4. Monitor spectral fidelity metrics during training")

# print("\n" + "="*80)
# print("✅ TEST COMPLETED")
# print("="*80)
