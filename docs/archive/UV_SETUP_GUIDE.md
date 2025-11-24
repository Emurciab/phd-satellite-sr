# Using UV Package Manager in Auth Environment

## Quick Start Guide

### Step 1: Install UV
```bash
# Install uv globally (recommended approach)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or install via pip in your Auth environment
conda activate Auth
pip install uv
```

### Step 2: Verify Installation
```bash
conda activate Auth
uv --version
```

### Step 3: Install PyTorch with UV
```bash
conda activate Auth

# For CUDA 11.8 (GPU support)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (newer GPUs)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## UV Command Reference

### Basic Package Management

```bash
# Activate environment first
conda activate Auth

# Install packages (much faster than pip!)
uv pip install package_name

# Install specific version
uv pip install package_name==1.2.3

# Install from requirements.txt
uv pip install -r requirements.txt

# Install with extras
uv pip install "package[extra,deps]"

# Uninstall
uv pip uninstall package_name

# List installed packages
uv pip list

# Show package info
uv pip show package_name

# Freeze dependencies
uv pip freeze > requirements.txt
```

### Advanced Usage

```bash
# Install from GitHub
uv pip install git+https://github.com/user/repo.git

# Install in editable mode (development)
uv pip install -e .

# Upgrade package
uv pip install --upgrade package_name

# Upgrade all packages
uv pip install --upgrade $(uv pip list --outdated --format=freeze | cut -d= -f1)

# Compile dependencies (like pip-compile)
uv pip compile requirements.in -o requirements.txt

# Sync environment to requirements
uv pip sync requirements.txt
```

---

## Why UV is Better

1. **Speed**: 10-100x faster than pip
2. **Deterministic**: Lock files for reproducibility
3. **Resolver**: Better dependency resolution
4. **Caching**: Efficient package caching
5. **Compatibility**: Works with existing pip workflows

---

## UV vs Conda/Mamba

| Feature | UV | Conda/Mamba | Use Case |
|---------|----|----|----------|
| **Python packages** | ✅ Excellent | ✅ Good | Python-only deps |
| **System packages** | ❌ No | ✅ Yes | C/C++ libraries |
| **Speed** | ⚡ Fastest | 🚀 Fast | Package installs |
| **Environment creation** | ✅ Via venv | ✅ Native | New projects |

**Recommended workflow**: 
- Use **conda/mamba** for environment creation and system dependencies
- Use **uv** for Python packages within environments

---

## For Your SR Project

### Install All Dependencies with UV

```bash
# Activate Auth environment
conda activate Auth

# Install uv if not already installed
pip install uv

# Install PyTorch (choose your CUDA version)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other ML dependencies
uv pip install numpy pandas matplotlib scikit-learn

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Run your test
python test_improved_model.py
```

### Create a Requirements File

```bash
# After installing all dependencies, freeze them
conda activate Auth
uv pip freeze > requirements-auth.txt

# To recreate environment later
conda create -n Auth_new python=3.14
conda activate Auth_new
uv pip install -r requirements-auth.txt
```

---

## Troubleshooting

### UV not found after installation
```bash
# Add to PATH (if installed via curl)
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### CUDA version mismatch
```bash
# Check your NVIDIA driver CUDA version
nvidia-smi

# Then use matching PyTorch:
# CUDA 11.x → cu118
# CUDA 12.x → cu121
```

### SSL/Network issues
```bash
# Use uv with custom index
uv pip install torch --index-url https://download.pytorch.org/whl/cu118 --trusted-host download.pytorch.org
```

### Conda conflicts
```bash
# UV installs into conda environment seamlessly
# If conflicts occur, try:
conda activate Auth
conda install pip  # Ensure pip is from conda
uv pip install torch  # Then use uv
```

---

## Performance Comparison

```bash
# Time comparison example (your mileage may vary)

# With pip:
time pip install torch torchvision
# Real: ~2-5 minutes

# With uv:
time uv pip install torch torchvision  
# Real: ~10-30 seconds

# Speedup: 5-20x faster! ⚡
```

---

## Next Steps for Your Project

1. **Install UV**:
   ```bash
   conda activate Auth
   pip install uv
   ```

2. **Install PyTorch**:
   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Test the model**:
   ```bash
   python test_improved_model.py
   ```

4. **Create requirements**:
   ```bash
   uv pip freeze > requirements-sr.txt
   ```

---

## Additional Resources

- UV Documentation: https://github.com/astral-sh/uv
- PyTorch Installation: https://pytorch.org/get-started/locally/
- UV vs pip comparison: https://astral.sh/blog/uv

---

## Quick Command Cheatsheet

```bash
# Setup
conda activate Auth
pip install uv

# Install PyTorch
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other packages
uv pip install numpy pandas matplotlib

# Check what's installed
uv pip list

# Save dependencies
uv pip freeze > requirements.txt

# Test PyTorch
python -c "import torch; print(torch.__version__)"

# Run SR model test
python test_improved_model.py
```
