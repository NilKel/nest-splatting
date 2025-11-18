#!/bin/bash
# Proper rebuild script that ensures Python uses the new build
set -e

echo "==== Rebuilding diff-surfel-rasterization ===="
cd "$(dirname "$0")"

# Clean old builds
echo "Cleaning old builds..."
rm -rf build/ dist/ *.egg-info
rm -f diff_surfel_rasterization/_C*.so

# Build
echo "Building extension..."
python setup.py build_ext --inplace

# Copy to site-packages (force Python to use new version)
echo "Installing to site-packages..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp -v diff_surfel_rasterization/_C*.so "$SITE_PACKAGES/diff_surfel_rasterization/"
cp -v diff_surfel_rasterization/__init__.py "$SITE_PACKAGES/diff_surfel_rasterization/"

echo "==== Rebuild complete! ===="
echo "New .so installed to: $SITE_PACKAGES/diff_surfel_rasterization/"

