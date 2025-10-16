#!/bin/bash
# Quick test of final rendering feature

conda activate nest_splatting

python train.py \
  -s /home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums \
  -m ./output/test_final_render \
  --yaml ./configs/nerfsyn.yaml \
  --method baseline \
  --eval \
  --iterations 50 \
  --test_render_stride 5

echo ""
echo "Check output at: ./output/test_final_render/final_test_renders/"
