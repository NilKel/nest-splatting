#!/bin/bash
set -e

echo "=========================================="
echo "RTX 5090 (SM90) Setup for nest-splatting"
echo "CUDA 12.8 + PyTorch Nightly"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Create conda environment
echo -e "${GREEN}Step 1: Creating conda environment...${NC}"
if conda env list | grep -q "nest_splatting"; then
    echo -e "${YELLOW}Environment 'nest_splatting' already exists. Remove it? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        conda env remove -n nest_splatting
        conda env create -f environment_rtx5090.yml
    else
        echo "Skipping environment creation..."
    fi
else
    conda env create -f environment_rtx5090.yml
fi

echo ""
echo -e "${GREEN}Step 2: Activating environment...${NC}"
echo "Run: conda activate nest_splatting"
echo "Then run this script again with --continue flag"
echo ""

if [ "$1" != "--continue" ]; then
    echo -e "${YELLOW}Please run:${NC}"
    echo "  conda activate nest_splatting"
    echo "  bash install_rtx5090.sh --continue"
    exit 0
fi

# Check if in correct conda env
if [[ "$CONDA_DEFAULT_ENV" != "nest_splatting" ]]; then
    echo -e "${YELLOW}Error: Please activate the nest_splatting environment first!${NC}"
    echo "Run: conda activate nest_splatting"
    exit 1
fi

echo ""
echo -e "${GREEN}Step 3: Installing PyTorch Nightly with CUDA 12.8...${NC}"
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

echo ""
echo -e "${GREEN}Step 4: Verifying PyTorch CUDA...${NC}"
python3 << 'PYCODE'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
PYCODE

echo ""
echo -e "${GREEN}Step 5: Setting up tiny-cuda-nn build environment...${NC}"
export TCNN_CUDA_ARCHITECTURES=90
echo "TCNN_CUDA_ARCHITECTURES=90"

echo ""
echo -e "${GREEN}Step 6: Cloning tiny-cuda-nn...${NC}"
TCNN_DIR="/tmp/tiny-cuda-nn-rtx5090"
if [ -d "$TCNN_DIR" ]; then
    echo "Removing existing tiny-cuda-nn directory..."
    rm -rf "$TCNN_DIR"
fi
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn "$TCNN_DIR"

echo ""
echo -e "${GREEN}Step 7: Building and installing tiny-cuda-nn PyTorch bindings...${NC}"
cd "$TCNN_DIR/bindings/torch"
python3 setup.py install

echo ""
echo -e "${GREEN}Step 8: Verifying tiny-cuda-nn installation...${NC}"
python3 << 'PYCODE'
try:
    import tinycudann as tcnn
    print(f"✓ tiny-cuda-nn successfully installed: {tcnn.__version__}")
except ImportError as e:
    print(f"✗ Failed to import tiny-cuda-nn: {e}")
    exit(1)
PYCODE

echo ""
echo -e "${GREEN}Step 9: Installing nest-splatting submodules...${NC}"
cd ~/Projects/nest-splatting

echo "Installing diff-surfel-rasterization..."
pip install -e ./submodules/diff-surfel-rasterization

echo "Installing simple-knn..."
pip install -e ./submodules/simple-knn

echo "Installing gridencoder..."
pip install -e ./gridencoder

echo ""
echo -e "${GREEN}=========================================="
echo "Installation Complete!"
echo "==========================================${NC}"
echo ""
echo "To use the environment:"
echo "  conda activate nest_splatting"
echo ""
echo "To verify everything works:"
echo "  python3 -c 'import torch; import tinycudann; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\"); print(f\"tiny-cuda-nn: {tinycudann.__version__}\")'"
echo ""
