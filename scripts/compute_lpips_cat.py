#!/usr/bin/env python3
"""
Compute LPIPS for all cat-mode nerf_synthetic experiments and update test_metrics.txt.

Iterates through outputs/nerf_synthetic/<scene>/cat/<experiment>/final_test_renders/
and computes LPIPS(VGG) between *_render.png and *_gt.png pairs.
Appends 'Average LPIPS:   X.XXXX' to test_metrics.txt if not already present.

Uses batched processing and persistent LPIPS model for speed.
"""

import os
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lpipsPyTorch.modules.lpips import LPIPS

BASE_DIR = Path("outputs/nerf_synthetic")
SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
BATCH_SIZE = 8  # VGG features are large, 8 x 800x800 fits in 32GB


@torch.no_grad()
def compute_lpips_for_experiment(exp_dir, lpips_model):
    """Compute average LPIPS for an experiment using batched inference."""
    renders_dir = exp_dir / "final_test_renders"
    if not renders_dir.exists():
        return None

    render_files = sorted(f for f in os.listdir(renders_dir) if f.endswith("_render.png"))
    if not render_files:
        return None

    all_lpips = []

    for batch_start in range(0, len(render_files), BATCH_SIZE):
        batch_files = render_files[batch_start:batch_start + BATCH_SIZE]
        renders_batch = []
        gts_batch = []

        for rf in batch_files:
            idx = rf.replace("_render.png", "")
            gt_path = renders_dir / f"{idx}_gt.png"
            render_path = renders_dir / rf
            if not gt_path.exists():
                continue
            renders_batch.append(tf.to_tensor(Image.open(render_path))[:3])
            gts_batch.append(tf.to_tensor(Image.open(gt_path))[:3])

        if not renders_batch:
            continue

        renders_t = torch.stack(renders_batch).cuda()
        gts_t = torch.stack(gts_batch).cuda()

        # LPIPS model returns [N,1,1,1] tensor
        lpips_vals = lpips_model(renders_t, gts_t)
        all_lpips.extend(lpips_vals.squeeze().tolist() if lpips_vals.numel() > 1
                         else [lpips_vals.item()])

    if not all_lpips:
        return None
    return sum(all_lpips) / len(all_lpips)


def update_test_metrics(exp_dir, avg_lpips):
    """Append Average LPIPS line to test_metrics.txt."""
    metrics_file = exp_dir / "test_metrics.txt"
    if not metrics_file.exists():
        return False

    content = metrics_file.read_text()
    if "Average LPIPS:" in content:
        return False

    lines = content.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if "Average SSIM:" in line:
            insert_idx = i + 1
        elif "Average L1:" in line:
            insert_idx = i
            break

    if insert_idx is None:
        for i, line in enumerate(lines):
            if line.startswith("Average"):
                insert_idx = i + 1

    if insert_idx is None:
        return False

    lpips_line = f"Average LPIPS:   {avg_lpips:.4f}"
    lines.insert(insert_idx, lpips_line)
    metrics_file.write_text("\n".join(lines))
    return True


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Load LPIPS model once
    print("Loading LPIPS VGG model...")
    lpips_model = LPIPS('vgg', '0.1').to(device).eval()

    # Collect all experiments to process
    MODES = ["baseline", "cat"]
    experiments = []
    for scene in SCENES:
        for mode in MODES:
            mode_dir = BASE_DIR / scene / mode
            if not mode_dir.exists():
                continue
            for exp_name in sorted(os.listdir(mode_dir)):
                exp_dir = mode_dir / exp_name
                if not exp_dir.is_dir():
                    continue
                metrics_file = exp_dir / "test_metrics.txt"
                if not metrics_file.exists():
                    continue
                content = metrics_file.read_text()
                if "Average LPIPS:" in content:
                    continue
                if not (exp_dir / "final_test_renders").exists():
                    continue
                experiments.append((scene, mode, exp_name, exp_dir))

    print(f"Found {len(experiments)} experiments needing LPIPS computation")
    print(f"Batch size: {BATCH_SIZE}")

    for i, (scene, mode, exp_name, exp_dir) in enumerate(experiments):
        print(f"[{i+1}/{len(experiments)}] {scene}/{mode}/{exp_name}", end=" ", flush=True)
        avg_lpips = compute_lpips_for_experiment(exp_dir, lpips_model)
        if avg_lpips is not None:
            updated = update_test_metrics(exp_dir, avg_lpips)
            print(f"LPIPS={avg_lpips:.4f} {'✓' if updated else '(skip)'}")
        else:
            print("no renders")

    print(f"\nDone! Processed {len(experiments)} experiments.")


if __name__ == "__main__":
    main()
