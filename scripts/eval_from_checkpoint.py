#!/usr/bin/env python3
"""
Evaluation script that loads training config from checkpoint directory.

Usage:
    python scripts/eval_from_checkpoint.py --model_path outputs/nerf_synthetic/chair/cat/run_name_5_levels
    python scripts/eval_from_checkpoint.py --model_path outputs/nerf_synthetic/chair/cat/run_name_5_levels --fps_only
    python scripts/eval_from_checkpoint.py --model_path outputs/nerf_synthetic/chair/cat/run_name_5_levels --num_iters 200
"""

import os
import sys
import json
import pickle
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from scene.background import LearnableSkybox
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.render_utils import save_img_u8, convert_gray_to_cmap
from lpipsPyTorch import lpips


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""

    # Try to load args.pkl (exact reproduction)
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        print(f"[CONFIG] Loaded args from {args_pkl_path}")
        return args

    # Fallback to args.json
    args_json_path = os.path.join(model_path, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            args_dict = json.load(f)
        args = Namespace(**args_dict)
        print(f"[CONFIG] Loaded args from {args_json_path}")
        return args

    raise FileNotFoundError(f"No training config found in {model_path}. "
                           f"Expected args.pkl or args.json")


def benchmark_fps(gaussians, cameras, pipe, ingp, cfg_model, iteration, background,
                 num_warmup=10, num_iters=100, skybox=None, background_mode="none"):
    """Benchmark FPS over all cameras."""
    beta = cfg_model.surfel.tg_beta

    with torch.no_grad():
        # Warmup
        for i in range(num_warmup):
            cam = cameras[i % len(cameras)]
            _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                      iteration=iteration, cfg=cfg_model, skybox=skybox,
                      background_mode=background_mode)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for i in range(num_iters):
            cam = cameras[i % len(cameras)]
            torch.cuda.synchronize()
            t0 = time.time()
            _ = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                      iteration=iteration, cfg=cfg_model, skybox=skybox,
                      background_mode=background_mode)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

    mean_time = sum(times) / len(times)
    fps = 1.0 / mean_time
    return fps, mean_time * 1000


def compute_metrics(gaussians, cameras, pipe, ingp, cfg_model, iteration, background,
                   sort_by_name=True, skybox=None, background_mode="none",
                   save_renders=False, save_depths=False, output_dir=None, camera_type="test"):
    """Compute PSNR, SSIM, LPIPS metrics over cameras.

    Args:
        sort_by_name: If True, sort cameras by image_name for consistent ordering
        save_renders: If True, save rendered images and GT to output_dir
        save_depths: If True, save depth maps to output_dir
        output_dir: Directory to save outputs (model_path if None)
        camera_type: "test" or "train" for output subdirectory naming
    """
    beta = cfg_model.surfel.tg_beta

    # Sort cameras by image_name if requested
    if sort_by_name:
        cameras = sorted(cameras, key=lambda c: c.image_name)

    # Create output directories if saving
    if save_renders or save_depths:
        if output_dir is None:
            raise ValueError("output_dir must be specified when saving renders/depths")

        render_dir = os.path.join(output_dir, f"final_{camera_type}_renders")
        depth_dir = os.path.join(output_dir, f"final_{camera_type}_depths")

        if save_renders:
            os.makedirs(render_dir, exist_ok=True)
        if save_depths:
            os.makedirs(depth_dir, exist_ok=True)

    psnr_values = []
    ssim_values = []
    lpips_values = []
    l1_values = []
    cam_names = []

    with torch.no_grad():
        for idx, cam in enumerate(cameras):
            render_pkg = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                               iteration=iteration, cfg=cfg_model, skybox=skybox,
                               background_mode=background_mode)

            rendered = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)

            psnr_val = psnr(rendered, gt).mean().item()
            ssim_val = ssim(rendered, gt).mean().item()
            lpips_val = lpips(rendered.unsqueeze(0), gt.unsqueeze(0), net_type='vgg').item()
            l1_val = l1_loss(rendered, gt).mean().item()

            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            lpips_values.append(lpips_val)
            l1_values.append(l1_val)
            cam_names.append(cam.image_name)

            print(f"[EVAL] {cam.image_name}: PSNR={psnr_val:.2f} SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f}")

            # Save renders
            if save_renders:
                cam_name = cam.image_name
                rendered_np = rendered.permute(1, 2, 0).cpu().numpy()
                gt_np = gt.permute(1, 2, 0).cpu().numpy()
                save_img_u8(gt_np, os.path.join(render_dir, f"{idx:03d}_{cam_name}_gt.png"))
                save_img_u8(rendered_np, os.path.join(render_dir, f"{idx:03d}_{cam_name}_render.png"))

            # Save depth maps
            if save_depths:
                cam_name = cam.image_name
                depth_expected = render_pkg['depth_expected']  # (1, H, W)
                depth_median = render_pkg['depth_median']  # (1, H, W)

                depth_expected_np = depth_expected.squeeze(0).cpu().numpy()
                depth_median_np = depth_median.squeeze(0).cpu().numpy()

                depth_expected_color = convert_gray_to_cmap(depth_expected_np, map_mode='turbo', revert=False)
                depth_median_color = convert_gray_to_cmap(depth_median_np, map_mode='turbo', revert=False)

                save_img_u8(depth_expected_color, os.path.join(depth_dir, f"{idx:03d}_{cam_name}_depth_expected.png"))
                save_img_u8(depth_median_color, os.path.join(depth_dir, f"{idx:03d}_{cam_name}_depth_median.png"))

    return {
        'psnr': psnr_values,
        'ssim': ssim_values,
        'lpips': lpips_values,
        'l1': l1_values,
        'names': cam_names,
        'avg_psnr': sum(psnr_values) / len(psnr_values),
        'avg_ssim': sum(ssim_values) / len(ssim_values),
        'avg_lpips': sum(lpips_values) / len(lpips_values),
        'avg_l1': sum(l1_values) / len(l1_values),
    }


def main():
    parser = ArgumentParser(description="Evaluate checkpoint with saved config")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Iteration to load (-1 for latest)")
    parser.add_argument("--fps_only", action="store_true",
                       help="Only benchmark FPS, skip metrics")
    parser.add_argument("--metrics_only", action="store_true",
                       help="Only compute metrics, skip FPS benchmark")
    parser.add_argument("--num_iters", type=int, default=100,
                       help="Number of iterations for FPS benchmark")
    parser.add_argument("--test", action="store_true", default=True,
                       help="Evaluate on test cameras (default)")
    parser.add_argument("--train", action="store_true",
                       help="Evaluate on train cameras")
    parser.add_argument("--sort_by_name", action="store_true", default=True,
                       help="Sort cameras by image_name for consistent ordering")
    parser.add_argument("--save_renders", action="store_true",
                       help="Save rendered images and GT to final_{test/train}_renders/")
    parser.add_argument("--save_depths", action="store_true",
                       help="Save depth maps to final_{test/train}_depths/")

    eval_args = parser.parse_args()

    # Load training config
    print(f"\n[EVAL] Loading config from: {eval_args.model_path}")
    args = load_training_config(eval_args.model_path)

    # Override model_path to the actual directory
    args.model_path = eval_args.model_path

    # Load YAML config
    config_yaml_path = os.path.join(eval_args.model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
        print(f"[CONFIG] Loaded config from {config_yaml_path}")
    else:
        cfg_model = Config(args.yaml)
        print(f"[CONFIG] Loaded config from {args.yaml}")

    # Print key training parameters
    print(f"\n[CONFIG] Training parameters:")
    print(f"  Method: {args.method}")
    print(f"  Kernel: {args.kernel}")
    print(f"  Hybrid levels: {getattr(args, 'hybrid_levels', 'N/A')}")
    print(f"  Source path: {args.source_path}")

    # Setup model
    iteration = eval_args.iteration

    # If iteration is -1, find the latest checkpoint
    if iteration == -1:
        import glob
        ngp_files = glob.glob(os.path.join(eval_args.model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            iteration = max(iterations)
            print(f"[CONFIG] Auto-detected latest iteration: {iteration}")
        else:
            raise FileNotFoundError(f"No ngp_*.pth checkpoints found in {eval_args.model_path}")

    # Create parser for ModelParams/PipelineParams extraction
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)

    # Force eval=True to load test cameras even if model was trained without --eval
    args.eval = True

    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    # Load INGP model
    ingp_model = INGP(cfg_model, args=args).to('cuda')
    ingp_model.load_model(eval_args.model_path, iteration)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp_model.set_active_levels(iteration)

    # Set kernel type
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Prune dead Gaussians
    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
    n_dead = dead_mask.sum().item()
    if n_dead > 0:
        valid_mask = ~dead_mask
        gaussians._xyz = gaussians._xyz[valid_mask]
        gaussians._features_dc = gaussians._features_dc[valid_mask]
        gaussians._features_rest = gaussians._features_rest[valid_mask]
        gaussians._opacity = gaussians._opacity[valid_mask]
        gaussians._scaling = gaussians._scaling[valid_mask]
        gaussians._rotation = gaussians._rotation[valid_mask]
        gaussians._appearance_level = gaussians._appearance_level[valid_mask]
        if hasattr(gaussians, '_gaussian_features') and gaussians._gaussian_features is not None and gaussians._gaussian_features.numel() > 0:
            gaussians._gaussian_features = gaussians._gaussian_features[valid_mask.to(gaussians._gaussian_features.device)]
        if hasattr(gaussians, '_adaptive_features') and gaussians._adaptive_features is not None and gaussians._adaptive_features.numel() > 0:
            gaussians._adaptive_features = gaussians._adaptive_features[valid_mask.to(gaussians._adaptive_features.device)]
        if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid_mask.to(gaussians._shape.device)]

    num_gaussians = len(gaussians.get_xyz)
    print(f"\n[EVAL] Gaussians after pruning: {num_gaussians:,}")

    # Load skybox if trained with background mode
    skybox = None
    background_mode = getattr(args, 'background', 'none')
    if background_mode is None:
        background_mode = 'none'
    if background_mode in ['skybox_dense', 'skybox_sparse']:
        skybox_res = getattr(args, 'skybox_res', 512)
        skybox = LearnableSkybox(resolution_h=skybox_res, resolution_w=skybox_res * 2).cuda()
        if skybox.load_model(eval_args.model_path, iteration):
            print(f"[EVAL] Loaded skybox checkpoint (mode={background_mode})")
        else:
            print(f"[EVAL] WARNING: No skybox checkpoint found, using default white skybox")

    # Get cameras
    if eval_args.train:
        cameras = scene.getTrainCameras()
        camera_type = "train"
    else:
        cameras = scene.getTestCameras()
        camera_type = "test"

    print(f"[EVAL] Using {len(cameras)} {camera_type} cameras")

    if len(cameras) > 0:
        H, W = cameras[0].image_height, cameras[0].image_width
        print(f"[EVAL] Resolution: {W}x{H}")

    # Background
    background = torch.zeros(3, device="cuda")

    # Results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Model: {eval_args.model_path}")
    print(f"Gaussians: {num_gaussians:,}")
    print(f"Method: {args.method}, Kernel: {args.kernel}")

    # FPS benchmark
    if not eval_args.metrics_only:
        print(f"\n[FPS] Benchmarking ({eval_args.num_iters} iterations)...")
        fps, ms_per_frame = benchmark_fps(
            gaussians, cameras, pipe, ingp_model, cfg_model, iteration, background,
            num_iters=eval_args.num_iters, skybox=skybox, background_mode=background_mode
        )
        print(f"[FPS] {fps:.1f} FPS ({ms_per_frame:.2f} ms/frame)")

    # Metrics
    if not eval_args.fps_only:
        save_info = ""
        if eval_args.save_renders:
            save_info += " (saving renders)"
        if eval_args.save_depths:
            save_info += " (saving depths)"
        print(f"\n[METRICS] Computing PSNR/SSIM/LPIPS on {camera_type} set{save_info}...")
        metrics = compute_metrics(
            gaussians, cameras, pipe, ingp_model, cfg_model, iteration, background,
            sort_by_name=eval_args.sort_by_name, skybox=skybox, background_mode=background_mode,
            save_renders=eval_args.save_renders, save_depths=eval_args.save_depths,
            output_dir=eval_args.model_path, camera_type=camera_type
        )

        print(f"\n[SUMMARY]")
        print(f"  Average PSNR:  {metrics['avg_psnr']:.2f} dB")
        print(f"  Average SSIM:  {metrics['avg_ssim']:.4f}")
        print(f"  Average LPIPS: {metrics['avg_lpips']:.4f}")
        print(f"  Average L1:    {metrics['avg_l1']:.6f}")

        # Save results
        results_path = os.path.join(eval_args.model_path, f"eval_{camera_type}_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'model_path': eval_args.model_path,
                'iteration': iteration,
                'num_gaussians': num_gaussians,
                'method': args.method,
                'kernel': args.kernel,
                'camera_type': camera_type,
                'num_cameras': len(cameras),
                'fps': fps if not eval_args.metrics_only else None,
                'ms_per_frame': ms_per_frame if not eval_args.metrics_only else None,
                'avg_psnr': metrics['avg_psnr'],
                'avg_ssim': metrics['avg_ssim'],
                'avg_lpips': metrics['avg_lpips'],
                'avg_l1': metrics['avg_l1'],
                'per_image': {
                    name: {
                        'psnr': metrics['psnr'][i],
                        'ssim': metrics['ssim'][i],
                        'lpips': metrics['lpips'][i],
                        'l1': metrics['l1'][i],
                    }
                    for i, name in enumerate(metrics['names'])
                }
            }, f, indent=2)
        print(f"\n[EVAL] Results saved to: {results_path}")

    print("="*80)


if __name__ == "__main__":
    main()
