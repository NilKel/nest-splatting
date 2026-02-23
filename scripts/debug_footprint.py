#!/usr/bin/env python3
"""
Compute per-Gaussian pixel footprint statistics.
For each Gaussian, count how many pixels it intersects (alpha > threshold).
Report min, max, mean, median, and a histogram.
"""
import os, sys, math, glob, pickle, torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams
from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

model_path = "outputs/nerf_synthetic/chair/3D_SH_TC/biasfixedwmma"
baked_dir = os.path.join(model_path, "baked")

# Load config
with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
args.model_path = model_path
args.eval = True
cfg_model = Config(os.path.join(model_path, "config.yaml"))

iteration = max(int(os.path.basename(f).replace("ngp_", "").replace(".pth", ""))
                for f in glob.glob(os.path.join(model_path, "ngp_*.pth")))

# Setup
temp_parser = ArgumentParser()
model_params = ModelParams(temp_parser, sentinel=True)
dataset = model_params.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
test_cameras = scene.getTestCameras()

# Load baked PLY
baked_ply = os.path.join(baked_dir, "baked.ply")
gaussians.load_ply(baked_ply)
gaussians.active_sh_degree = 3
gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
N = len(gaussians.get_xyz)

beta = cfg_model.surfel.tg_beta
kernel_name = getattr(args, 'kernel', 'gaussian')
kernel_map = {'gaussian': 0, 'beta': 1, 'flex': 2, 'general': 3, 'beta_scaled': 4}
kernel_type = kernel_map.get(kernel_name, 0)
bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')

# Strategy: render each Gaussian individually is too slow (107k Gaussians).
# Instead, use the rasterizer's tile-based approach. We can approximate the
# footprint by projecting each Gaussian's 2D surfel onto the image.
#
# For a 2D Gaussian surfel, the pixel footprint is roughly:
#   area = pi * scale_screen_x * scale_screen_y
# where scale_screen is the projected scale in pixels.
#
# More precisely: project the surfel's transformation matrix to get the
# screen-space ellipse, then count pixels in the bounding box (tile-based).
#
# Simplest accurate approach: use the rasterizer's radii output, which gives
# the bounding circle radius in tiles for each Gaussian.

print(f"Computing footprints for {N} Gaussians across test views...")

# We'll compute footprints from the projected 2D covariance.
# The rasterizer returns radii (in tiles). Each tile is BLOCK_X=16 pixels.
# So pixel radius ≈ radii * 16, and footprint ≈ pi * (radii*16)^2
# But this is a conservative bounding circle. Let's be more precise.

# Actually, let's compute the projected surfel area directly.
# The transMat maps local (u,v,1) to screen (x,y,w).
# The surfel extent in UV is controlled by scales sx, sy.
# Screen-space Jacobian = d(screen)/d(uv) at center.
# Area = sx * sy * |det(J)| * pi (for ellipse)

# But the simplest approach: just use the rasterizer and collect per-Gaussian
# intersection counts. We can render with a custom "count" mode, or use
# the radii to estimate.

# Let's compute from projected geometry directly.
# For each test view, compute projected bounding box area per Gaussian.

# Compute screen-space footprint analytically
VIEW_IDX = 125
cam = test_cameras[VIEW_IDX]
W, H = int(cam.image_width), int(cam.image_height)

print(f"View {VIEW_IDX}: {W}×{H}")

tanfovx = math.tan(cam.FoVx * 0.5)
tanfovy = math.tan(cam.FoVy * 0.5)

# Get the projected 2D positions and radii from the rasterizer
settings = GaussianRasterizationSettings(
    image_height=H, image_width=W,
    tanfovx=tanfovx, tanfovy=tanfovy,
    bg=bg_color, scale_modifier=1.0,
    viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
    sh_degree=gaussians.active_sh_degree, campos=cam.camera_center,
    prefiltered=False, debug=False, beta=beta,
)
rasterizer = GaussianRasterizer(raster_settings=settings)

with torch.no_grad():
    result = rasterizer(
        means3D=gaussians.get_xyz, means2D=torch.zeros_like(gaussians.get_xyz[:, :2]),
        opacities=gaussians.get_opacity, shs=gaussians.get_features,
        scales=gaussians.get_scaling, rotations=gaussians.get_rotation,
        kernel_type=kernel_type,
    )

# result is (rendered, out_color, out_others, radii, ...)
# The rasterizer returns radii as the 4th element
rendered_img = result[0]
radii = result[1]  # [N] int tensor, radius in pixels (or tiles?)

print(f"Radii tensor shape: {radii.shape}, dtype: {radii.dtype}")
radii_cpu = radii.cpu().numpy()

# Check what fraction are visible (radius > 0)
visible = radii_cpu > 0
n_visible = visible.sum()
print(f"Visible Gaussians: {n_visible} / {N} ({100*n_visible/N:.1f}%)")

if n_visible == 0:
    print("No visible Gaussians!")
    sys.exit(1)

vis_radii = radii_cpu[visible]

# The radii from the rasterizer are in pixels (bounding circle radius).
# Pixel footprint of the bounding box = (2*radius+1)^2
# Actual ellipse footprint ≈ pi * a * b, but we only have the bounding circle.
# Approximate with circle: footprint ≈ pi * r^2
footprints_bbox = (2 * vis_radii.astype(np.float64) + 1) ** 2
footprints_circle = np.pi * vis_radii.astype(np.float64) ** 2

print(f"\n{'='*60}")
print(f"PIXEL FOOTPRINT STATISTICS (view {VIEW_IDX})")
print(f"{'='*60}")
print(f"  Using bounding box (2r+1)^2:")
print(f"    Min:    {footprints_bbox.min():.0f} px")
print(f"    Max:    {footprints_bbox.max():.0f} px")
print(f"    Mean:   {footprints_bbox.mean():.1f} px")
print(f"    Median: {np.median(footprints_bbox):.0f} px")
print(f"    Std:    {footprints_bbox.std():.1f} px")

print(f"\n  Using circle (pi*r^2):")
print(f"    Min:    {footprints_circle.min():.0f} px")
print(f"    Max:    {footprints_circle.max():.0f} px")
print(f"    Mean:   {footprints_circle.mean():.1f} px")
print(f"    Median: {np.median(footprints_circle):.0f} px")

print(f"\n  Raw radii (pixels):")
print(f"    Min:    {vis_radii.min()}")
print(f"    Max:    {vis_radii.max()}")
print(f"    Mean:   {vis_radii.mean():.1f}")
print(f"    Median: {np.median(vis_radii):.0f}")

# Also compute: how many texels does an 8x8 grid give per pixel?
# texels_per_pixel = 64 / footprint
tex_per_px_8 = 64.0 / footprints_circle
tex_per_px_16 = 256.0 / footprints_circle
print(f"\n  Texels per pixel (8×8 = 64 texels):")
print(f"    Min:    {tex_per_px_8.min():.4f}")
print(f"    Max:    {tex_per_px_8.max():.4f}")
print(f"    Mean:   {tex_per_px_8.mean():.4f}")
print(f"    Median: {np.median(tex_per_px_8):.4f}")
print(f"  Gaussians with <1 texel/pixel (undersampled at 8×8): "
      f"{(tex_per_px_8 < 1).sum()} ({100*(tex_per_px_8 < 1).mean():.1f}%)")
print(f"  Gaussians with <1 texel/pixel (undersampled at 16×16): "
      f"{(tex_per_px_16 < 1).sum()} ({100*(tex_per_px_16 < 1).mean():.1f}%)")

# Histogram of radii
print(f"\n{'='*60}")
print(f"RADIUS HISTOGRAM (pixels)")
print(f"{'='*60}")
bins = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 100000]
labels = ["0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127",
          "128-255", "256-511", "512-1023", "1024-2047", "2048+"]
counts, _ = np.histogram(vis_radii, bins=bins)
max_count = counts.max()
for i, (label, count) in enumerate(zip(labels, counts)):
    bar = "#" * int(50 * count / max_count) if max_count > 0 else ""
    pct = 100 * count / n_visible
    print(f"  {label:>9s}: {count:>7,} ({pct:>5.1f}%) {bar}")

# Histogram of footprint (circle area)
print(f"\n{'='*60}")
print(f"FOOTPRINT HISTOGRAM (pi*r^2 pixels)")
print(f"{'='*60}")
fp_bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 1e9]
fp_labels = ["<10", "10-49", "50-99", "100-499", "500-999", "1k-5k",
             "5k-10k", "10k-50k", "50k-100k", "100k+"]
fp_counts, _ = np.histogram(footprints_circle, bins=fp_bins)
max_fp = fp_counts.max()
for label, count in zip(fp_labels, fp_counts):
    bar = "#" * int(50 * count / max_fp) if max_fp > 0 else ""
    pct = 100 * count / n_visible
    print(f"  {label:>9s}: {count:>7,} ({pct:>5.1f}%) {bar}")

# Average across multiple views
print(f"\n{'='*60}")
print(f"MULTI-VIEW STATISTICS (all {len(test_cameras)} views)")
print(f"{'='*60}")
all_max_radii = []
all_mean_radii = []
all_median_radii = []
all_n_visible = []

with torch.no_grad():
    for i, cam in enumerate(test_cameras):
        settings = GaussianRasterizationSettings(
            image_height=int(cam.image_height), image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg_color, scale_modifier=1.0,
            viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
            sh_degree=gaussians.active_sh_degree, campos=cam.camera_center,
            prefiltered=False, debug=False, beta=beta,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)
        result = rasterizer(
            means3D=gaussians.get_xyz, means2D=torch.zeros_like(gaussians.get_xyz[:, :2]),
            opacities=gaussians.get_opacity, shs=gaussians.get_features,
            scales=gaussians.get_scaling, rotations=gaussians.get_rotation,
            kernel_type=kernel_type,
        )
        r = result[1].cpu().numpy()
        vis = r > 0
        if vis.sum() > 0:
            all_max_radii.append(r[vis].max())
            all_mean_radii.append(r[vis].mean())
            all_median_radii.append(np.median(r[vis]))
            all_n_visible.append(vis.sum())

print(f"  Visible Gaussians: mean={np.mean(all_n_visible):.0f}, "
      f"min={np.min(all_n_visible)}, max={np.max(all_n_visible)}")
print(f"  Max radius (px):   mean={np.mean(all_max_radii):.0f}, "
      f"min={np.min(all_max_radii)}, max={np.max(all_max_radii)}")
print(f"  Mean radius (px):  mean={np.mean(all_mean_radii):.1f}, "
      f"min={np.min(all_mean_radii):.1f}, max={np.max(all_mean_radii):.1f}")
print(f"  Median radius (px): mean={np.mean(all_median_radii):.1f}, "
      f"min={np.min(all_median_radii):.1f}, max={np.max(all_median_radii):.1f}")
