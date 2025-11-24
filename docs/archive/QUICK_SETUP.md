# Quick Setup Guide - Using tf_gpu Environment

## ✅ Recommended Approach
**Use your existing `tf_gpu` environment** which already has GPU/CUDA configured, then add PyTorch and Earth Engine.

---

## Step-by-Step Installation

### Option 1: Run the Setup Script (Easiest)
```bash
# Make sure you're in the SR directory
cd /media/eamurciab/TOSHIBA_EXT1/PhD/SR

# Run the automated setup
./setup_tf_gpu_env.sh
```

### Option 2: Manual Installation
```bash
# Activate tf_gpu environment
conda activate tf_gpu

# Install PyTorch with CUDA 12.1 support (matches your CUDA 12.8 driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Google Earth Engine
pip install earthengine-api

# Verify PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Verify Earth Engine (if not authenticated yet, see below)
python -c "import ee; print('EE version:', ee.__version__)"
```

---

## Earth Engine Authentication

If Earth Engine is not authenticated yet:

```bash
conda activate tf_gpu

# Authenticate (opens browser)
earthengine authenticate

# Or use service account (if you have one)
# This is what your Auth environment uses
python -c "
import ee
credentials = ee.ServiceAccountCredentials(
    'your-service-account@project.iam.gserviceaccount.com',
    '/path/to/your/credentials.json'
)
ee.Initialize(credentials)
print('✅ EE initialized with service account')
"
```

---

## Test the Complete Pipeline

Once installed, test everything:

```bash
conda activate tf_gpu
python test_improved_model.py
```

This will:
- ✅ Load preprocessing pipeline (requires Earth Engine)
- ✅ Fetch real satellite data from GEE
- ✅ Load ImprovedModel (requires PyTorch)
- ✅ Run forward pass on GPU
- ✅ Compute spectral fidelity loss
- ✅ Display full validation results

---

## Why tf_gpu is Better for This

| Feature | tf_gpu ✅ | Auth | Reason |
|---------|----------|------|--------|
| **GPU/CUDA** | Pre-configured | No GPU | Already working |
| **TensorFlow** | 2.19.0 | No | Optional for data aug |
| **Python** | 3.10.16 | 3.14.0 | More stable |
| **Environment** | Stable | Newer | Production ready |

---

## Your Current Setup

```
System:
├── NVIDIA GPU: Yes (detected)
├── CUDA Driver: 12.8
└── nvidia-smi: Working

tf_gpu Environment:
├── Python: 3.10.16 ✅
├── TensorFlow: 2.19.0 ✅
├── GPU Support: Active ✅
├── PyTorch: [To be installed]
└── Earth Engine: [To be installed]
```

---

## Installation Commands (Copy-Paste)

```bash
# One-liner to install everything
conda activate tf_gpu && \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
pip install earthengine-api && \
python -c "import torch; print('✅ PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" && \
python -c "import ee; print('✅ Earth Engine:', ee.__version__)"

# Then test
python test_improved_model.py
```

---

## Expected Output After Installation

```
✅ Python: 3.10.16
✅ PyTorch: 2.x.x
   - CUDA available: True
   - CUDA version: 12.1
   - GPU count: 1
   - GPU name: [Your GPU name]
✅ Earth Engine API: 0.x.x
✅ TensorFlow: 2.19.0
   - GPU devices: 1

Installation complete!
```

---

## Troubleshooting

### PyTorch doesn't see GPU
```bash
# Check CUDA is accessible
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# If False, reinstall with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Earth Engine authentication fails
```bash
# Re-authenticate
earthengine authenticate

# Or copy credentials from Auth environment
cp ~/path/to/auth/credentials.json ~/.config/earthengine/
```

### Import errors
```bash
# Make sure tf_gpu is activated
conda activate tf_gpu
which python  # Should show: .../envs/tf_gpu/bin/python
```

---

## After Installation

1. **Test preprocessing pipeline**:
   ```bash
   python -c "from numpy_efficient_pipeline import TensorFlowFreeDataPipeline; print('✅ Pipeline OK')"
   ```

2. **Test model**:
   ```bash
   python -c "from ImprovedModel import create_improved_sr_model; print('✅ Model OK')"
   ```

3. **Run full test**:
   ```bash
   python test_improved_model.py
   ```

---

## Next Steps After Successful Test

1. Review test results
2. Adjust model hyperparameters if needed
3. Set up training loop
4. Start training on your ~1000 image pairs
5. Monitor spectral fidelity metrics

---

## Summary

✅ **Use tf_gpu environment** - already has GPU/CUDA working  
✅ **Install PyTorch + Earth Engine** - two simple pip commands  
✅ **No need to modify Auth environment** - keep it clean for other projects  
✅ **Everything in one place** - training, inference, and data processing  

**Total setup time: ~2-3 minutes** 🚀
