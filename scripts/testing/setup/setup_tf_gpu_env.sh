#!/bin/bash
# Setup script to install PyTorch and Earth Engine in tf_gpu environment

echo "=================================="
echo "Setting up tf_gpu environment"
echo "=================================="

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo ""
echo "Current environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

echo ""
echo "Step 1: Installing PyTorch with CUDA 12.1 support..."
echo "-------------------------------------------------------"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Step 2: Installing Google Earth Engine API..."
echo "-----------------------------------------------"
pip install earthengine-api

echo ""
echo "Step 3: Verifying installations..."
echo "------------------------------------"
python << EOF
import sys
print("\n✅ Python:", sys.version.split()[0])

try:
    import torch
    print("✅ PyTorch:", torch.__version__)
    print("   - CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("   - CUDA version:", torch.version.cuda)
        print("   - GPU count:", torch.cuda.device_count())
        print("   - GPU name:", torch.cuda.get_device_name(0))
except ImportError as e:
    print("❌ PyTorch not installed:", e)

try:
    import ee
    print("✅ Earth Engine API:", ee.__version__)
except ImportError as e:
    print("❌ Earth Engine not installed:", e)

try:
    import tensorflow as tf
    print("✅ TensorFlow:", tf.__version__)
    print("   - GPU devices:", len(tf.config.list_physical_devices('GPU')))
except ImportError as e:
    print("❌ TensorFlow not installed:", e)

print("\n" + "="*50)
print("Installation complete!")
print("="*50)
EOF

echo ""
echo "Step 4: Earth Engine Authentication Check..."
echo "----------------------------------------------"
python -c "import ee; ee.Initialize()" 2>&1 && echo "✅ EE authenticated" || echo "⚠️  EE needs authentication - run: earthengine authenticate"

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To test the model, run:"
echo "  conda activate tf_gpu"
echo "  python test_improved_model.py"
