#!/usr/bin/env python3
"""
Render views around a scene center on a circular camera trajectory.

Adapted from https://github.com/KeKsBoTer/onion/blob/master/utils/video.py

Usage:
    python scripts/render_circle.py --model_path outputs/mip_360/bicycle/cat/us01bc001op001sc1e4nsgeneral001decayadrrec_5_levels --num_views 10
    python scripts/render_circle.py --model_path outputs/mip_360/bicycle/cat/us01bc001op001sc1e4nsgeneral001decayadrrec_5_levels --num_views 60 --video
"""

import os
import sys
import math
import json
import pickle
import glob
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from scene.cameras import Camera
from scene.background import LearnableSkybox
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.render_utils import save_img_u8
from torchvision.utils import save_image


def load_training_config(model_path):
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            args = pickle.load(f)
        return args
    args_json_path = os.path.join(model_path, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            args_dict = json.load(f)
        return Namespace(**args_dict)
    raise FileNotFoundError(f"No training config found in {model_path}")


def estimate_scene_params(cameras):
    """Estimate scene center, radius, and up vector from training cameras."""
    cam_positions = []
    cam_ups = []
    for cam in cameras:
        cam_positions.append(cam.camera_center.cpu().numpy())
        # R column 1 is the camera's up direction in world coords
        R = cam.R.cpu().numpy() if isinstance(cam.R, torch.Tensor) else np.array(cam.R)
        cam_ups.append(R[:, 1])
    cam_positions = np.array(cam_positions)
    cam_ups = np.array(cam_ups)

    center = cam_positions.mean(axis=0)

    dists = np.linalg.norm(cam_positions - center, axis=1)
    radius = np.median(dists)

    # Average camera up vector (robust to individual camera tilt)
    up = cam_ups.mean(axis=0)
    up = up / np.linalg.norm(up)

    return center, radius, up


def make_circle_camera(progress, center, radius, up, look_target, ref_camera):
    """Create a camera on a circle around the scene center.

    Args:
        progress: float in [0, 1) for position on circle
        center: np.array [3] - center of camera orbit
        radius: float - orbit radius
        up: np.array [3] - up vector
        look_target: np.array [3] - point to look at
        ref_camera: reference Camera for FoV and resolution
    """
    up = up / np.linalg.norm(up)

    # Build orthonormal basis for the circle plane
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(up, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])

    u = np.cross(up, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(up, u)
    v = v / np.linalg.norm(v)

    theta = progress * 2 * math.pi
    cam_pos = center + radius * math.cos(theta) * u + radius * math.sin(theta) * v

    # Build view matrix (R, T) in COLMAP convention
    forward = look_target - cam_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    cam_up = np.cross(right, forward)
    cam_up = cam_up / np.linalg.norm(cam_up)

    # Camera class stores R as c2w rotation: columns = [right, up, forward]
    # getWorld2View2 transposes R to get w2c rotation
    R = np.stack([right, cam_up, forward], axis=1)  # c2w rotation [3,3]
    T = -R.T @ cam_pos  # w2c translation

    # Create a dummy image (black) with same resolution
    H, W = ref_camera.image_height, ref_camera.image_width
    dummy_image = torch.zeros(3, H, W)

    return Camera(
        colmap_id=0, R=R, T=T,
        FoVx=ref_camera.FoVx, FoVy=ref_camera.FoVy,
        image=dummy_image, gt_alpha_mask=None,
        image_name=f"circle_{progress:.3f}", uid=0,
    )


def main():
    parser = ArgumentParser(description="Render circular camera trajectory")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, default=None,
                       help="Override source data path (for cameras)")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--radius_scale", type=float, default=1.0,
                       help="Scale factor for orbit radius (1.0 = median camera distance)")
    parser.add_argument("--elevation", type=float, default=0.0,
                       help="Elevation of camera orbit above center (in units of radius)")
    parser.add_argument("--tilt", type=float, default=0.0,
                       help="Tilt look target down (positive = look lower, in units of radius)")
    parser.add_argument("--decompose", action="store_true",
                       help="Also render gaussian-only and ngp-only decompositions")
    parser.add_argument("--video", action="store_true",
                       help="Save as MP4 video instead of individual images")
    parser.add_argument("--fps", type=int, default=30)

    eval_args = parser.parse_args()

    # Load config
    args = load_training_config(eval_args.model_path)
    args.model_path = eval_args.model_path
    args.eval = True
    if eval_args.source_path:
        args.source_path = eval_args.source_path

    config_yaml_path = os.path.join(eval_args.model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
    else:
        cfg_model = Config(args.yaml)

    # Find iteration
    iteration = eval_args.iteration
    if iteration == -1:
        ngp_files = glob.glob(os.path.join(eval_args.model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            iteration = max(iterations)
            print(f"[CONFIG] Latest iteration: {iteration}")
        else:
            raise FileNotFoundError(f"No ngp_*.pth found in {eval_args.model_path}")

    # Load model
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    ingp = INGP(cfg_model, args=args).to('cuda')
    ingp.load_model(eval_args.model_path, iteration)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp.set_active_levels(iteration)
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Prune dead Gaussians
    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
    if dead_mask.sum().item() > 0:
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
        if hasattr(gaussians, '_shape') and gaussians._shape is not None and gaussians._shape.numel() > 0:
            gaussians._shape = gaussians._shape[valid_mask.to(gaussians._shape.device)]

    print(f"[RENDER] {len(gaussians.get_xyz):,} Gaussians, kernel={gaussians.kernel_type}")

    # Load skybox if needed
    skybox = None
    background_mode = getattr(args, 'background', 'none') or 'none'
    if background_mode in ['skybox_dense', 'skybox_sparse']:
        skybox_res = getattr(args, 'skybox_res', 512)
        skybox = LearnableSkybox(resolution_h=skybox_res, resolution_w=skybox_res * 2).cuda()
        skybox.load_model(eval_args.model_path, iteration)

    # Estimate scene parameters from training cameras
    train_cameras = scene.getTrainCameras()
    center, radius, up = estimate_scene_params(train_cameras)
    print(f"[SCENE] Center: {center}, Radius: {radius:.2f}, Up: {up}")

    # Adjust orbit
    orbit_radius = radius * eval_args.radius_scale
    orbit_center = center.copy()
    orbit_center += up * radius * eval_args.elevation  # Elevate orbit

    look_target = center.copy()
    look_target += up * radius * eval_args.tilt  # Positive tilt = look lower (raise target above cam)

    ref_camera = train_cameras[0]
    background = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    # Output directories
    out_dir = os.path.join(eval_args.model_path, "circle_renders")
    os.makedirs(out_dir, exist_ok=True)

    render_modes = [('full', None)]
    if eval_args.decompose:
        render_modes += [('gaussian_only', 'gaussian_only'), ('ngp_only', 'ngp_only')]
        for name, _ in render_modes[1:]:
            os.makedirs(os.path.join(eval_args.model_path, f"circle_renders_{name}"), exist_ok=True)

    # Warmup render (exclude from timing)
    warmup_cam = make_circle_camera(0.0, orbit_center, orbit_radius, up, look_target, ref_camera)
    with torch.no_grad():
        render(warmup_cam, gaussians, pipe, background, ingp=ingp, beta=beta,
               iteration=iteration, cfg=cfg_model, skybox=skybox,
               background_mode=background_mode, aabb_mode=getattr(args, 'aabb', '2dgs'))
    torch.cuda.synchronize()

    # Render views
    all_frames = {name: [] for name, _ in render_modes}
    render_times = []
    with torch.no_grad():
        for i in range(eval_args.num_views):
            progress = i / eval_args.num_views
            cam = make_circle_camera(
                progress, orbit_center, orbit_radius, up, look_target, ref_camera
            )

            for mode_name, decompose_mode in render_modes:
                if mode_name == 'full':
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()

                render_pkg = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                                   iteration=iteration, cfg=cfg_model, skybox=skybox,
                                   background_mode=background_mode,
                                   aabb_mode=getattr(args, 'aabb', '2dgs'),
                                   decompose_mode=decompose_mode)
                rendered = torch.clamp(render_pkg["render"], 0.0, 1.0)

                if mode_name == 'full':
                    torch.cuda.synchronize()
                    render_times.append(time.perf_counter() - t0)
                    save_dir = out_dir
                else:
                    save_dir = os.path.join(eval_args.model_path, f"circle_renders_{mode_name}")

                all_frames[mode_name].append(rendered)
                out_path = os.path.join(save_dir, f"frame_{i:04d}.png")
                save_image(rendered, out_path)

            mode_str = f" (+decompose)" if eval_args.decompose else ""
            print(f"[RENDER] Frame {i+1}/{eval_args.num_views}{mode_str}")

    if render_times:
        avg_ms = np.mean(render_times) * 1000
        avg_fps = 1.0 / np.mean(render_times)
        print(f"[BENCHMARK] Avg render: {avg_ms:.1f} ms ({avg_fps:.1f} FPS) over {len(render_times)} frames")

    # Optionally save videos
    if eval_args.video:
        import cv2
        for mode_name, _ in render_modes:
            frames = all_frames[mode_name]
            if len(frames) < 2:
                continue
            if mode_name == 'full':
                video_path = os.path.join(out_dir, "circle.mp4")
            else:
                video_dir = os.path.join(eval_args.model_path, f"circle_renders_{mode_name}")
                video_path = os.path.join(video_dir, "circle.mp4")
            h, w = frames[0].shape[1], frames[0].shape[2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, eval_args.fps, (w, h))
            for frame in frames:
                frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"[VIDEO] {mode_name}: {video_path}")

    print(f"\n[DONE] {len(all_frames['full'])} frames saved to {out_dir}")


if __name__ == "__main__":
    main()
