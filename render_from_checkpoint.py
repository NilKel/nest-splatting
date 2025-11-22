#!/usr/bin/env python3
"""
Render test images from a trained checkpoint with configurable stride.
Keeps image indices consistent (renders idx 0, stride, 2*stride, etc.)

Usage:
    python render_from_checkpoint.py --checkpoint_dir outputs/add/nerf_synthetic/drums/retryall --stride 1
"""

import torch
import os
import sys
import numpy as np
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from pathlib import Path

# Add nest-splatting to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from utils.render_utils import save_img_u8
from utils.image_utils import psnr
from utils.loss_utils import ssim, l1_loss


def load_config_from_checkpoint(checkpoint_dir):
    """
    Load configuration from checkpoint directory.
    Reads cfg_args file and infers configuration from checkpoint structure.
    """
    cfg_args_path = os.path.join(checkpoint_dir, 'cfg_args')
    
    if not os.path.exists(cfg_args_path):
        raise FileNotFoundError(f"cfg_args not found at {cfg_args_path}")
    
    # Read the cfg_args file (it's a Namespace repr)
    with open(cfg_args_path, 'r') as f:
        cfg_str = f.read()
    
    # Parse the Namespace string
    args_dict = eval(cfg_str)
    
    # Convert dict to Namespace if it's a dict
    if isinstance(args_dict, dict):
        args = Namespace(**args_dict)
    else:
        args = args_dict
    
    return args


def infer_method_from_checkpoint(checkpoint_dir):
    """
    Infer the method (baseline/add/cat) from the checkpoint directory structure.
    """
    # Check the path for method indicators
    path_lower = checkpoint_dir.lower()
    
    if 'add' in path_lower:
        return 'add'
    elif 'cat' in path_lower:
        return 'cat'
    else:
        return 'baseline'


def infer_yaml_config(checkpoint_dir):
    """
    Infer YAML config file based on dataset name in path.
    """
    # Check for common dataset names
    if 'nerf_synthetic' in checkpoint_dir or 'nerfsyn' in checkpoint_dir:
        return 'configs/nerfsyn.yaml'
    elif 'dtu' in checkpoint_dir.lower():
        return 'configs/dtu.yaml'
    elif 'tanks' in checkpoint_dir.lower() or 'tandt' in checkpoint_dir.lower():
        return 'configs/tandt.yaml'
    else:
        # Default to nerfsyn
        return 'configs/nerfsyn.yaml'


def find_latest_iteration(checkpoint_dir):
    """
    Find the latest iteration checkpoint in the directory.
    """
    # Check for PLY files in point_cloud directory
    point_cloud_dir = os.path.join(checkpoint_dir, 'point_cloud')
    if os.path.exists(point_cloud_dir):
        iteration_dirs = [d for d in os.listdir(point_cloud_dir) if d.startswith('iteration_')]
        if iteration_dirs:
            iterations = [int(d.split('_')[1]) for d in iteration_dirs]
            return max(iterations)
    
    # Check for INGP checkpoint files
    ngp_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('ngp_') and f.endswith('.pth')]
    if ngp_files:
        iterations = [int(f.split('_')[1].split('.')[0]) for f in ngp_files]
        return max(iterations)
    
    return None


def render_checkpoint(checkpoint_dir, stride=1, output_dir=None, iteration=None, yaml_config=None, method=None):
    """
    Render test images from a checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., outputs/add/nerf_synthetic/drums/retryall)
        stride: Render every Nth test image (default: 1 = all images)
        output_dir: Custom output directory (default: checkpoint_dir/rerender_stride_{stride})
        iteration: Checkpoint iteration to load (default: auto-detect latest)
        yaml_config: Path to YAML config file (default: auto-infer from dataset)
        method: Rendering method - baseline/add/cat (default: auto-infer from path)
    """
    
    print(f"\n{'='*70}")
    print(f"  Rendering from Checkpoint")
    print(f"{'='*70}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    
    # Auto-detect iteration if not provided
    if iteration is None:
        iteration = find_latest_iteration(checkpoint_dir)
        if iteration is None:
            raise ValueError(f"Could not find checkpoint in {checkpoint_dir}")
        print(f"Auto-detected iteration: {iteration}")
    else:
        print(f"Using iteration: {iteration}")
    
    # Auto-infer YAML config if not provided
    if yaml_config is None:
        yaml_config = infer_yaml_config(checkpoint_dir)
        print(f"Auto-inferred YAML config: {yaml_config}")
    
    # Auto-infer method if not provided
    if method is None:
        method = infer_method_from_checkpoint(checkpoint_dir)
        print(f"Auto-inferred method: {method.upper()}")
    
    # Load saved config args
    try:
        saved_args = load_config_from_checkpoint(checkpoint_dir)
        print(f"Loaded saved args from checkpoint")
        source_path = saved_args.source_path
        sh_degree = saved_args.sh_degree
    except Exception as e:
        print(f"Warning: Could not load cfg_args: {e}")
        # Try to infer from path
        source_path = None
        sh_degree = 3
    
    # Load YAML config
    cfg_model = Config(yaml_config)
    print(f"Loaded YAML config: {yaml_config}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(checkpoint_dir, f'rerender_stride_{stride}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Stride: {stride} (render every {stride} images)")
    print(f"{'='*70}\n")
    
    # Create model arguments
    model_args = Namespace(
        sh_degree=sh_degree,
        source_path=source_path if source_path else checkpoint_dir.split('/outputs/')[1].split('/')[0],
        model_path=checkpoint_dir,
        images='images',
        resolution=1,
        white_background=False,
        data_device='cuda',
        eval=True,
        load_allres=False,
    )
    
    # Add method-specific arguments
    class Args:
        def __init__(self, method):
            self.method = method
            self.hybrid_levels = 3  # Default for cat mode
            self.cat_coarse2fine = False
    
    method_args = Args(method)
    
    # Setup pipeline
    pipe_args = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        depth_ratio=0.0,
        debug=False,
    )
    
    # Load INGP model
    print("Loading INGP model...")
    ingp_model = INGP(cfg_model, method_args).to('cuda')
    ingp_model.load_model(checkpoint_dir, iteration)
    ingp_model.set_active_levels(iteration)  # Set active levels for c2f
    print(f"✓ Loaded INGP checkpoint from iteration {iteration}")
    
    # Load Gaussian model
    print("Loading Gaussian model...")
    gaussians = GaussianModel(sh_degree)
    scene = Scene(model_args, gaussians, load_iteration=iteration, shuffle=False)
    print(f"✓ Loaded {len(gaussians.get_xyz)} Gaussians")
    
    # Setup rendering
    bg_color = [1, 1, 1] if model_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussians.base_opacity = cfg_model.surfel.base_opacity
    beta = cfg_model.surfel.base_beta
    
    # Get test cameras
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("ERROR: No test cameras found!")
        return
    
    print(f"\nFound {len(test_cameras)} test cameras")
    print(f"Will render {len(range(0, len(test_cameras), stride))} images with stride {stride}")
    print(f"\nRendering...")
    
    # Metrics accumulation
    psnr_values = []
    ssim_values = []
    l1_values = []
    
    # Render test images
    with torch.no_grad():
        for idx in tqdm(range(0, len(test_cameras), stride), desc="Rendering"):
            viewpoint = test_cameras[idx]
            
            # Render the image
            render_pkg = render(
                viewpoint, gaussians, pipe_args, background,
                ingp=ingp_model, beta=beta, iteration=iteration,
                cfg=cfg_model, render_mode=method
            )
            
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            
            # Compute metrics
            psnr_value = psnr(rendered_image, gt_image).mean().item()
            ssim_value = ssim(rendered_image, gt_image).mean().item()
            l1_value = l1_loss(rendered_image, gt_image).mean().item()
            
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            l1_values.append(l1_value)
            
            # Convert to numpy for saving
            rendered_np = rendered_image.permute(1, 2, 0).detach().cpu().numpy()
            gt_np = gt_image.permute(1, 2, 0).detach().cpu().numpy()
            
            # Save with consistent indices
            gt_name = os.path.join(output_dir, f"{idx:03d}_gt.png")
            render_name = os.path.join(output_dir, f"{idx:03d}_render.png")
            
            save_img_u8(gt_np, gt_name)
            save_img_u8(rendered_np, render_name)
    
    # Compute and save metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_l1 = np.mean(l1_values)
    
    print(f"\n{'='*70}")
    print(f"  Rendering Complete!")
    print(f"{'='*70}")
    print(f"Rendered {len(psnr_values)} images")
    print(f"\nMetrics:")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    print(f"  Average L1:   {avg_l1:.6f}")
    print(f"{'='*70}\n")
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Checkpoint: {checkpoint_dir}\n")
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Stride: {stride}\n")
        f.write(f"Images rendered: {len(psnr_values)}\n")
        f.write(f"\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average L1: {avg_l1:.6f}\n")
        f.write(f"\n")
        f.write(f"Per-image metrics:\n")
        for i, (idx, p, s, l) in enumerate(zip(range(0, len(test_cameras), stride), psnr_values, ssim_values, l1_values)):
            f.write(f"Image {idx:03d}: PSNR={p:.2f} SSIM={s:.4f} L1={l:.6f}\n")
    
    print(f"Metrics saved to: {metrics_file}")
    print(f"Images saved to: {output_dir}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render test images from a trained checkpoint")
    
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint directory (e.g., outputs/add/nerf_synthetic/drums/retryall)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Render every Nth test image (default: 1 = all images)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory (default: checkpoint_dir/rerender_stride_{stride})")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Checkpoint iteration to load (default: auto-detect latest)")
    parser.add_argument("--yaml", type=str, default=None,
                        help="Path to YAML config file (default: auto-infer from dataset)")
    parser.add_argument("--method", type=str, default=None, choices=["baseline", "add", "cat"],
                        help="Rendering method (default: auto-infer from checkpoint path)")
    
    args = parser.parse_args()
    
    # Validate checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)
    
    # Render
    render_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        stride=args.stride,
        output_dir=args.output_dir,
        iteration=args.iteration,
        yaml_config=args.yaml,
        method=args.method
    )



