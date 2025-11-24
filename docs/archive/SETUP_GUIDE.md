# 🚀 Installation & Setup Guide for Efficient SR Pipeline

## **Problem Summary**
Your current environment (coreg) has Python 3.14.0, which is too new for TensorFlow. TensorFlow requires Python 3.8-3.11.

## **✅ Solution Options**

### **Option 1: Use Your Existing TensorFlow Environment (RECOMMENDED)**

You already have a working TensorFlow environment! 

```bash
# Switch to your existing TensorFlow environment
conda activate tf_gpu

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"

# Install additional dependencies for the efficient pipeline
pip install scikit-image lmdb zarr dask[array] h5py

# Now you can use the full pipeline with TensorFlow support
cd /media/eamurciab/TOSHIBA_EXT1/PhD/SR
python sr_efficient_pipeline.py
```

### **Option 2: TensorFlow-Free Pipeline (CURRENT ENVIRONMENT)**

Use the pure NumPy version that works in your current environment:

```bash
# Stay in current coreg environment
# Install only the minimal dependencies
pip install scikit-image

# Use the TensorFlow-free version
python numpy_efficient_pipeline.py
```

### **Option 3: Create New Environment with Compatible Python**

```bash
# Create new environment with Python 3.10 (TensorFlow compatible)
conda create -n sr_env python=3.10

# Activate and install dependencies
conda activate sr_env
pip install tensorflow==2.15.0 earthengine-api pandas numpy scikit-image lmdb zarr dask[array] h5py

# Use full pipeline
python sr_efficient_pipeline.py
```

## **🎯 Quick Start (Recommended Path)**

### Step 1: Use Your TensorFlow Environment
```bash
conda activate tf_gpu
```

### Step 2: Install Missing Dependencies
```bash
pip install lmdb zarr dask[array] h5py
```

### Step 3: Test the Pipeline
```bash
# Test with the efficient pipeline
cd /media/eamurciab/TOSHIBA_EXT1/PhD/SR
python -c "
from sr_efficient_pipeline import EfficientSRDataPipeline
print('✅ Pipeline imported successfully!')
"
```

### Step 4: Integration with Your Existing Code
```python
# In your existing AuthGEE.py or new script
from sr_efficient_pipeline import EfficientSRDataPipeline, SuperResolutionDataGenerator

# Use your existing regions
from Auth.AuthGEE import regions

# Initialize efficient pipeline
pipeline = EfficientSRDataPipeline(
    cache_dir="./sr_training_cache",
    tile_size=512,  # Adjust based on GPU memory
    overlap=64
)

# Create data generator
data_gen = SuperResolutionDataGenerator(
    pipeline=pipeline,
    batch_size=4,
    sequence_length=3
)

# Generate training data
for lr_batch, hr_batch in data_gen.generate_training_batches(regions[:2]):
    print(f"LR: {lr_batch.shape}, HR: {hr_batch.shape}")
    # Train your model here
    break
```

## **📊 Performance Comparison**

| Approach | Environment | Memory Usage | Setup Time | Compatibility |
|----------|-------------|--------------|------------|---------------|
| **Option 1: tf_gpu** | Existing TF env | ~2GB | 2 minutes | ✅ Full features |
| **Option 2: NumPy** | Current coreg | ~1GB | 30 seconds | ✅ Core features |
| **Option 3: New env** | Fresh sr_env | ~2GB | 10 minutes | ✅ Full features |

## **🔧 Troubleshooting**

### If TensorFlow Installation Still Fails:
```bash
# Alternative TensorFlow installation methods
conda activate tf_gpu

# Try conda-forge channel
conda install -c conda-forge tensorflow

# Or use specific version
pip install tensorflow==2.13.0

# Check GPU support (if you have GPU)
python -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"
```

### For Earth Engine Authentication:
```bash
# Initialize Earth Engine (if not done already)
earthengine authenticate
```

### Memory Issues:
```python
# Reduce tile size if running out of memory
pipeline = EfficientSRDataPipeline(tile_size=256, batch_size=2)
```

## **🎯 Next Steps After Setup**

1. **Test with Small Dataset**: Start with 1-2 regions to validate
2. **Adjust Parameters**: Tune tile_size based on your GPU memory
3. **Monitor Performance**: Use the built-in statistics tracking
4. **Scale Up**: Add more regions once validated

## **💡 Pro Tips**

- **Start with Option 1** (tf_gpu environment) - it's already configured
- **Use smaller tile sizes** (256px) if you have limited GPU memory  
- **Enable caching** to speed up repeated training runs
- **Monitor memory usage** with `htop` or `nvidia-smi` (for GPU)

Choose Option 1 (tf_gpu environment) for the fastest start! 🚀