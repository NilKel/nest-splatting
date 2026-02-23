#!/usr/bin/env python3
"""
Check if baked texture resolution meets Nyquist criterion wrt hashgrid cell size.

For each Gaussian:
  - UV extent = 4 * scale (the surfel sampling range is [-4s, 4s])
  - Texel spacing at grid_size G: dx = 2 * 4 * max(sx,sy) / G  (world units)
  - Hash cell size = voxel_range / finest_resolution
  - Nyquist: need dx < cell_size / 2  (2 samples per cell)
  - Samples per cell = cell_size / dx
"""
import os, sys, glob, pickle, torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams

model_path = "outputs/nerf_synthetic/chair/3D_SH_TC/biasfixedwmma"

with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
args.model_path = model_path
args.eval = True
cfg_model = Config(os.path.join(model_path, "config.yaml"))

iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                for f in glob.glob(os.path.join(model_path, "ngp_*.pth")))

temp_parser = ArgumentParser()
model_params = ModelParams(temp_parser, sentinel=True)
dataset = model_params.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

# Load baked PLY
baked_ply = os.path.join(model_path, "baked", "baked.ply")
gaussians.load_ply(baked_ply)
N = len(gaussians.get_xyz)

# Get hash grid params
ingp = INGP(cfg_model, args=args).to('cuda')
ingp.load_model(model_path, iteration)

hash_enc = ingp.hash_encoding
embeddings, offsets, num_levels, per_level_scale, base_resolution, align_corners, interp_id = hash_enc.get_params()
voxel_min, voxel_max = -1.5, 1.5
voxel_range = voxel_max - voxel_min  # 3.0

finest_resolution = base_resolution * (per_level_scale ** (num_levels - 1))
cell_size = voxel_range / finest_resolution

print(f"Hash grid: {num_levels} levels, base={base_resolution}, "
      f"finest={finest_resolution:.0f}, cell_size={cell_size:.6f}")
print(f"Voxel range: [{voxel_min}, {voxel_max}] = {voxel_range}")

# Per-Gaussian scales
scales = gaussians.get_scaling.detach()  # [N, 2] or [N, 3]
# For 2DGS surfels, scales are [N, 2] (sx, sy on the surfel plane)
print(f"Scales shape: {scales.shape}")
sx = scales[:, 0]
sy = scales[:, 1]
max_scale = torch.max(sx, sy)  # [N]

UV_EXTENT = 4.0  # bake samples in [-4*s, 4*s]

for grid_size in [8, 16, 32, 64]:
    # Texel spacing in world units
    # The bake kernel samples at UV range [-extent, extent] with G texels
    # extent = UV_EXTENT * scale, so world range = 2 * UV_EXTENT * scale
    # texel spacing = 2 * UV_EXTENT * max_scale / grid_size
    texel_spacing = (2.0 * UV_EXTENT * max_scale / grid_size).cpu().numpy()

    # Samples per hash cell
    samples_per_cell = cell_size / texel_spacing

    # Nyquist: need >= 2 samples per cell
    meets_nyquist = samples_per_cell >= 2.0

    print(f"\n{'='*60}")
    print(f"GRID SIZE: {grid_size}×{grid_size} ({grid_size**2} texels)")
    print(f"{'='*60}")
    print(f"  Texel spacing (world units):")
    print(f"    Min:    {texel_spacing.min():.6f}")
    print(f"    Max:    {texel_spacing.max():.6f}")
    print(f"    Mean:   {texel_spacing.mean():.6f}")
    print(f"    Median: {np.median(texel_spacing):.6f}")
    print(f"  Hash cell size: {cell_size:.6f}")
    print(f"  Samples per hash cell:")
    print(f"    Min:    {samples_per_cell.min():.3f}")
    print(f"    Max:    {samples_per_cell.max():.3f}")
    print(f"    Mean:   {samples_per_cell.mean():.3f}")
    print(f"    Median: {np.median(samples_per_cell):.3f}")
    print(f"  Meets Nyquist (>=2 samples/cell): "
          f"{meets_nyquist.sum():,} / {N:,} ({100*meets_nyquist.mean():.1f}%)")
    print(f"  Has >=1 sample/cell: "
          f"{(samples_per_cell >= 1).sum():,} / {N:,} ({100*(samples_per_cell >= 1).mean():.1f}%)")
    print(f"  Sub-cell (<1 sample/cell): "
          f"{(samples_per_cell < 1).sum():,} / {N:,} ({100*(samples_per_cell < 1).mean():.1f}%)")

# Histogram of samples_per_cell at grid_size=8
print(f"\n{'='*60}")
print(f"SAMPLES PER HASH CELL HISTOGRAM (grid_size=8)")
print(f"{'='*60}")
texel_spacing_8 = (2.0 * UV_EXTENT * max_scale / 8).cpu().numpy()
spc_8 = cell_size / texel_spacing_8

bins = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 100.0]
labels = ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-2", "2-4", "4-8", "8-16", "16+"]
counts, _ = np.histogram(spc_8, bins=bins)
max_count = counts.max()
for label, count in zip(labels, counts):
    bar = "#" * int(50 * count / max_count) if max_count > 0 else ""
    pct = 100 * count / N
    print(f"  {label:>9s}: {count:>7,} ({pct:>5.1f}%) {bar}")

# What grid_size would each Gaussian need for Nyquist?
# Need: 2 * UV_EXTENT * max_scale / G <= cell_size / 2
# G >= 4 * UV_EXTENT * max_scale / cell_size
needed_G = (4.0 * UV_EXTENT * max_scale / cell_size).cpu().numpy()
print(f"\n{'='*60}")
print(f"GRID SIZE NEEDED FOR NYQUIST (per Gaussian)")
print(f"{'='*60}")
print(f"  Min:    {needed_G.min():.1f}")
print(f"  Max:    {needed_G.max():.1f}")
print(f"  Mean:   {needed_G.mean():.1f}")
print(f"  Median: {np.median(needed_G):.1f}")

g_bins = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 100000]
g_labels = ["<=2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-256", "257-512", "512+"]
g_counts, _ = np.histogram(needed_G, bins=g_bins)
max_gc = g_counts.max()
for label, count in zip(g_labels, g_counts):
    bar = "#" * int(50 * count / max_gc) if max_gc > 0 else ""
    pct = 100 * count / N
    print(f"  {label:>9s}: {count:>7,} ({pct:>5.1f}%) {bar}")
