"""
Render test views using vanilla 2DGS rasterizer (../2d-gaussian-splatting env).
Loads PLY from a nest-splatting checkpoint, renders with vanilla 2DGS, saves results.

Usage (from nest-splatting dir):
    conda run -n 2dgs python scripts/test_vanilla_2dgs.py \
        --ply_path outputs/mip_360/bonsai/3D_SH_res/35kiterfps5H01bc001op0sc1e3nsGS_10kfx_05wodr10_01thr/point_cloud/iteration_35000/point_cloud.ply \
        --source_path /home/nilkel/Projects/data/mip_360/bonsai/ \
        -i images_2 \
        --num_views 5
"""

import sys
import os

# Add vanilla 2DGS to path
VANILLA_2DGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', '2d-gaussian-splatting')
VANILLA_2DGS = os.path.abspath(VANILLA_2DGS)
sys.path.insert(0, VANILLA_2DGS)

import torch
import numpy as np
import math
from argparse import ArgumentParser, Namespace

from scene.gaussian_model import GaussianModel
from scene import Scene
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams


def main():
    parser = ArgumentParser()
    parser.add_argument("--ply_path", required=True, help="Path to point_cloud.ply from nest-splatting")
    parser.add_argument("--source_path", "-s", required=True, help="Path to dataset")
    parser.add_argument("-i", "--images", default="images")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: next to PLY)")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # Setup model params via vanilla 2DGS parser
    temp_parser = ArgumentParser()
    lp = ModelParams(temp_parser)
    pp = PipelineParams(temp_parser)

    # Create a fake model_path for Scene loading
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ply_path), '..', '..', 'sanity_test')
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # We need a model_path with point_cloud/iteration_X structure for Scene to load
    # Instead, load the scene for cameras only, then load PLY manually
    fake_args = Namespace(
        source_path=args.source_path,
        model_path=out_dir,
        images=args.images,
        resolution=-1,
        white_background=args.white_background,
        data_device='cuda',
        eval=True,
        sh_degree=args.sh_degree,
        convert_SHs_python=False,
        compute_cov3D_python=False,
        depth_ratio=0.0,
        debug=False,
    )

    dataset = lp.extract(fake_args)
    pipe = pp.extract(fake_args)

    gaussians = GaussianModel(args.sh_degree)

    # Load scene (for cameras) without loading a checkpoint
    scene = Scene(dataset, gaussians, shuffle=False)

    # Now load our PLY manually
    print(f"Loading PLY from: {args.ply_path}")
    gaussians.load_ply(args.ply_path)
    gaussians.active_sh_degree = args.sh_degree
    print(f"Loaded {len(gaussians.get_xyz)} Gaussians, SH degree={gaussians.active_sh_degree}")

    bg_color = torch.ones(3, device="cuda") if args.white_background else torch.zeros(3, device="cuda")

    test_cams = scene.getTestCameras()
    n_views = min(args.num_views, len(test_cams))

    print(f"\nRendering {n_views} test views with vanilla 2DGS rasterizer...")
    print("-" * 70)
    print(f"{'View':>6} | {'Mean RGB':>12} | {'PSNR':>8}")
    print("-" * 70)

    from utils.image_utils import psnr

    for i in range(n_views):
        cam = test_cams[i]
        gt = cam.original_image.cuda()

        render_pkg = render(cam, gaussians, pipe, bg_color)
        img = torch.clamp(render_pkg['render'], 0.0, 1.0)

        p = psnr(img, gt).mean().item()
        print(f"{i:>6} | {img.mean().item():>12.6f} | {p:>8.2f}")

        # Save rendered image as torch tensor for comparison
        torch.save(img.cpu(), os.path.join(out_dir, f'vanilla_2dgs_view{i}.pt'))

        # Save as PNG
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(img_np).save(os.path.join(out_dir, f'vanilla_2dgs_view{i}.png'))

    print("-" * 70)
    print(f"Saved renders to {out_dir}/")
    print("Run the nest-splatting comparison script to compare these with 3D_SH_res output.")


if __name__ == "__main__":
    main()
