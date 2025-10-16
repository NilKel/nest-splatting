#!/bin/bash
# Quick surface potential comparison script
# Trains 2DGS once, then compares baseline vs surface INGP methods

SCENE=${1:-"/home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums"}
ITERATIONS=${2:-10000}
SCENE_NAME=$(basename $SCENE)

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       SURFACE POTENTIAL vs BASELINE COMPARISON                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Scene: $SCENE"
echo "Iterations: $ITERATIONS"
echo ""

# Check if 2DGS Gaussians already exist
GAUSSIAN_INIT="$SCENE/gaussian_init.pth"
if [ -f "$GAUSSIAN_INIT" ]; then
    echo "✓ Found existing 2DGS Gaussians at $GAUSSIAN_INIT"
    echo "  Will skip 2DGS training phase"
else
    echo "⚠ No pre-trained Gaussians found"
    echo "  Will train 2DGS first (this will take ~2-3 minutes)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Training BASELINE method"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python train.py \
  -s "$SCENE" \
  -m "./output/${SCENE_NAME}_baseline_${ITERATIONS}" \
  --yaml ./configs/nerfsyn.yaml \
  --method baseline \
  --eval \
  --iterations $ITERATIONS \
  --test_render_stride 25

BASELINE_EXIT=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Training SURFACE method"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python train.py \
  -s "$SCENE" \
  -m "./output/${SCENE_NAME}_surface_${ITERATIONS}" \
  --yaml ./configs/nerfsyn.yaml \
  --method surface \
  --eval \
  --iterations $ITERATIONS \
  --test_render_stride 25

SURFACE_EXIT=$?

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                      TRAINING COMPLETE                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [ $BASELINE_EXIT -eq 0 ] && [ $SURFACE_EXIT -eq 0 ]; then
    echo "✓ Both methods trained successfully!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "RESULTS COMPARISON"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    echo "BASELINE:"
    grep -E "(Average PSNR|Average SSIM)" "./output/${SCENE_NAME}_baseline_${ITERATIONS}/final_test_renders/metrics.txt" | sed 's/^/  /'
    
    echo ""
    echo "SURFACE:"
    grep -E "(Average PSNR|Average SSIM)" "./output/${SCENE_NAME}_surface_${ITERATIONS}/final_test_renders/metrics.txt" | sed 's/^/  /'
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📁 Output directories:"
    echo "  Baseline: ./output/${SCENE_NAME}_baseline_${ITERATIONS}"
    echo "  Surface:  ./output/${SCENE_NAME}_surface_${ITERATIONS}"
    echo ""
    echo "📊 View metrics:"
    echo "  cat ./output/${SCENE_NAME}_baseline_${ITERATIONS}/final_test_renders/metrics.txt"
    echo "  cat ./output/${SCENE_NAME}_surface_${ITERATIONS}/final_test_renders/metrics.txt"
    echo ""
else
    echo "✗ Training failed"
    [ $BASELINE_EXIT -ne 0 ] && echo "  Baseline exit code: $BASELINE_EXIT"
    [ $SURFACE_EXIT -ne 0 ] && echo "  Surface exit code: $SURFACE_EXIT"
fi

echo "💾 2DGS Gaussians saved to: $GAUSSIAN_INIT"
echo "   (Subsequent runs will skip 2DGS training)"
echo ""

