# Test Commands

## Quick Test (1000 iterations)

### Baseline Mode:
```bash
conda activate nest_splatting

python train.py \
  -s /home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums \
  -m ./output/test_baseline \
  --yaml ./configs/nerfsyn.yaml \
  --method baseline \
  --eval \
  --iterations 1000
```

### Surface Mode:
```bash
conda activate nest_splatting

python train.py \
  -s /home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums \
  -m ./output/test_surface \
  --yaml ./configs/nerfsyn.yaml \
  --method surface \
  --eval \
  --iterations 1000
```

## Full Training (5000 iterations)

### Single Scene:
```bash
python train.py \
  -s /home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/lego \
  -m ./output/lego_surface_5k \
  --yaml ./configs/nerfsyn.yaml \
  --method surface \
  --eval \
  --iterations 5000
```

### Batch (All 8 Scenes):
```bash
# Modify scripts/nerfsyn_eval.py to add --method surface
python scripts/nerfsyn_eval.py --yaml ./configs/nerfsyn.yaml
```

## Verify Method Argument Works

```bash
# Should print "Method: BASELINE"
python train.py \
  -s /home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums \
  -m ./output/test \
  --yaml ./configs/nerfsyn.yaml \
  --method baseline \
  --iterations 10

# Should print "Method: SURFACE"
python train.py \
  -s /home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/drums \
  -m ./output/test \
  --yaml ./configs/nerfsyn.yaml \
  --method surface \
  --iterations 10
```

## Rendering Evaluation

After training, render test views:
```bash
python eval_render.py \
  --iteration 5000 \
  -s /home/nilkel/Projects/data/nest_synthetic/nerf_synthetic/lego \
  -m ./output/lego_surface_5k \
  --yaml ./configs/nerfsyn.yaml \
  --skip_train --skip_mesh
```

## Compare Metrics

```bash
# Compare PSNR/SSIM between baseline and surface
python scripts/metric_eval.py ./output/comparison 5000
```
