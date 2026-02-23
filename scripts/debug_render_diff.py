#!/usr/bin/env python3
"""Compare SH-only vs SH+48D rendered images to debug the residual effect."""
import os, sys, torch, numpy as np
from PIL import Image

baked_dir = "outputs/nerf_synthetic/chair/3D_SH_TC/biasfixedwmma/baked/renders"

# Load a few test images from both modes
sh_dir = os.path.join(baked_dir, "sh_only")
res_dir = os.path.join(baked_dir, "sh_shared")

# List images
images = sorted(os.listdir(sh_dir))[:5]

for img_name in images:
    sh_img = np.array(Image.open(os.path.join(sh_dir, img_name))).astype(np.float32) / 255.0
    res_img = np.array(Image.open(os.path.join(res_dir, img_name))).astype(np.float32) / 255.0

    diff = res_img - sh_img
    abs_diff = np.abs(diff)

    print(f"\n{img_name}:")
    print(f"  SH-only:  mean={sh_img.mean():.4f}, min={sh_img.min():.4f}, max={sh_img.max():.4f}")
    print(f"  SH+48D:   mean={res_img.mean():.4f}, min={res_img.min():.4f}, max={res_img.max():.4f}")
    print(f"  Diff:     mean={diff.mean():.6f}, std={diff.std():.6f}")
    print(f"  |Diff|:   mean={abs_diff.mean():.6f}, max={abs_diff.max():.6f}")

    # Check if difference is zero (residual not applied)
    if abs_diff.max() < 1e-6:
        print(f"  WARNING: Residual has NO effect!")
    else:
        # Where are the biggest differences?
        flat_idx = np.argmax(abs_diff.sum(axis=2))
        max_y, max_x = np.unravel_index(flat_idx, abs_diff.shape[:2])
        print(f"  Max diff at pixel ({max_x}, {max_y}): "
              f"SH={sh_img[max_y,max_x]}, 48D={res_img[max_y,max_x]}, diff={diff[max_y,max_x]}")

        # Save 10x amplified diff image
        diff_vis = np.clip(abs_diff * 10, 0, 1)
        Image.fromarray((diff_vis * 255).astype(np.uint8)).save(
            os.path.join(baked_dir, f"diff_48d_{img_name}"))
