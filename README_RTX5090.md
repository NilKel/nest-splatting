# RTX 5090 Installation Guide for nest-splatting

This directory contains modified files for installing nest-splatting on RTX 5090 (Blackwell/SM90) with CUDA 12.8.

## Files Created

1. **`environment_rtx5090.yml`** - Modified conda environment file
   - Removed PyTorch (will be installed via pip nightly)
   - Removed tiny-cuda-nn (will be built from source)
   - Kept all other dependencies

2. **`install_rtx5090.sh`** - Automated installation script
   - Creates conda environment
   - Installs PyTorch nightly with CUDA 12.8
   - Builds tiny-cuda-nn from source with SM90 support
   - Installs all nest-splatting submodules

## Quick Start

### Option 1: Automated Installation (Recommended)

```bash
cd ~/Projects/nest-splatting

# Step 1: Run the installation script
bash install_rtx5090.sh

# Step 2: Activate the environment
conda activate nest_splatting

# Step 3: Continue installation
bash install_rtx5090.sh --continue
```

### Option 2: Manual Installation

```bash
cd ~/Projects/nest-splatting

# Step 1: Create conda environment
conda env create -f environment_rtx5090.yml
conda activate nest_splatting

# Step 2: Install PyTorch Nightly with CUDA 12.8
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Step 3: Verify PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Step 4: Build tiny-cuda-nn from source
export TCNN_CUDA_ARCHITECTURES=90
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn /tmp/tiny-cuda-nn-rtx5090
cd /tmp/tiny-cuda-nn-rtx5090/bindings/torch
python3 setup.py install

# Step 5: Install nest-splatting submodules
cd ~/Projects/nest-splatting
pip install -e ./submodules/diff-surfel-rasterization
pip install -e ./submodules/simple-knn
pip install -e ./gridencoder
```

## Key Configuration

### GPU Architecture
- **RTX 5090 Compute Capability**: SM90 (9.0)
- **Environment Variable**: `TCNN_CUDA_ARCHITECTURES=90`

### CUDA Version
- **System CUDA**: 12.8
- **PyTorch**: Nightly build with CUDA 12.8 support

### Why This is Needed

1. **No pre-built wheels**: tiny-cuda-nn doesn't have pre-built wheels for SM90 architecture yet
2. **CUDA 12.8 is very new**: Most packages only support up to CUDA 12.4 stable
3. **PyTorch nightly required**: Stable PyTorch doesn't support CUDA 12.8 yet

## Verification

After installation, verify everything works:

```bash
conda activate nest_splatting

# Check PyTorch
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Check tiny-cuda-nn  
python3 -c "import tinycudann as tcnn; print('tiny-cuda-nn:', tcnn.__version__)"

# Check GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')"
```

Expected output for GPU check: `Compute Capability: (9, 0)` for RTX 5090

## Troubleshooting

### Build fails with "unsupported GPU architecture"
- Make sure `TCNN_CUDA_ARCHITECTURES=90` is set before building
- Check CUDA version: `nvcc --version` should show 12.8

### PyTorch can't find CUDA
- Verify driver: `nvidia-smi` should show CUDA 12.8
- Reinstall PyTorch nightly: `pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall`

### Compilation takes too long / runs out of memory
- Build without parallel compilation (already done in script)
- Close other applications to free up RAM

## Notes

- **Build time**: tiny-cuda-nn may take 10-30 minutes to compile
- **Disk space**: Requires ~5GB temporary space for building
- **RAM**: At least 16GB recommended during compilation

## For instant-ngp

The instant-ngp project in `~/Projects/instant-ngp` already auto-detects SM90 and should work without modifications. Just build normally:

```bash
cd ~/Projects/instant-ngp
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
```

The CMake will automatically detect your RTX 5090 and compile for SM90.

However, I noticed your instant-ngp detected **SM120** instead of SM90! This needs investigation - check your build output.
