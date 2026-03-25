#!/usr/bin/env python3
"""
Shared utilities for circular camera trajectory rendering.

Handles model loading, scene parameter estimation, and camera generation.
Used by `preview_circle.py` (adjust POV) and `render_circle_video.py` (final videos).
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
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# Config & model loading
# ---------------------------------------------------------------------------

def load_training_config(model_path):
    """Load training args from args.pkl or args.json in the model directory."""
    args_pkl_path = os.path.join(model_path, "args.pkl")
    if os.path.exists(args_pkl_path):
        with open(args_pkl_path, 'rb') as f:
            return pickle.load(f)
    args_json_path = os.path.join(model_path, "args.json")
    if os.path.exists(args_json_path):
        with open(args_json_path, 'r') as f:
            return Namespace(**json.load(f))
    raise FileNotFoundError(f"No training config found in {model_path}")


def load_model(model_path, source_path=None, iteration=-1):
    """Load a trained NEST-splatting model.

    Args:
        model_path: Path to the model directory (containing args.pkl, ngp_*.pth, point_cloud/).
        source_path: Override for the training data source path. Required if the original
                     data location is unavailable (e.g. trained on a cluster).
        iteration: Checkpoint iteration to load. -1 = latest.

    Returns:
        dict with keys: gaussians, ingp, pipe, cfg, scene, iteration, args,
                        skybox, background_mode, beta
    """
    args = load_training_config(model_path)
    args.model_path = model_path
    args.eval = True
    if source_path:
        args.source_path = source_path

    config_yaml_path = os.path.join(model_path, "config.yaml")
    cfg = Config(config_yaml_path) if os.path.exists(config_yaml_path) else Config(args.yaml)

    # Find iteration
    if iteration == -1:
        ngp_files = glob.glob(os.path.join(model_path, "ngp_*.pth"))
        if not ngp_files:
            raise FileNotFoundError(f"No ngp_*.pth found in {model_path}")
        iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
        iteration = max(iterations)
        print(f"[CONFIG] Latest iteration: {iteration}")

    # Load INGP (hash encoder + MLP)
    temp_parser = ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    ingp = INGP(cfg, args=args).to('cuda')
    ingp.load_model(model_path, iteration)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp.set_active_levels(iteration)
    if hasattr(args, 'kernel'):
        gaussians.kernel_type = args.kernel

    # Prune dead Gaussians (opacity <= 0.005)
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

    print(f"[MODEL] {len(gaussians.get_xyz):,} Gaussians, kernel={gaussians.kernel_type}")

    # Load skybox if needed
    skybox = None
    background_mode = getattr(args, 'background', 'none') or 'none'
    if background_mode in ['skybox_dense', 'skybox_sparse']:
        skybox_res = getattr(args, 'skybox_res', 512)
        skybox = LearnableSkybox(resolution_h=skybox_res, resolution_w=skybox_res * 2).cuda()
        skybox.load_model(model_path, iteration)

    return dict(
        gaussians=gaussians, ingp=ingp, pipe=pipe, cfg=cfg,
        scene=scene, iteration=iteration, args=args,
        skybox=skybox, background_mode=background_mode,
        beta=cfg.surfel.tg_beta,
    )


# ---------------------------------------------------------------------------
# Camera trajectory
# ---------------------------------------------------------------------------

def estimate_scene_params(cameras):
    """Estimate scene center, radius, and up vector from training cameras.

    Returns:
        center: np.array [3] — mean camera position
        radius: float — median distance from center to cameras
        up: np.array [3] — average camera up direction (unit vector)
    """
    cam_positions = []
    cam_ups = []
    for cam in cameras:
        cam_positions.append(cam.camera_center.cpu().numpy())
        R = cam.R.cpu().numpy() if isinstance(cam.R, torch.Tensor) else np.array(cam.R)
        cam_ups.append(R[:, 1])  # R column 1 = camera up in world coords
    cam_positions = np.array(cam_positions)
    cam_ups = np.array(cam_ups)

    center = cam_positions.mean(axis=0)
    radius = np.median(np.linalg.norm(cam_positions - center, axis=1))
    up = cam_ups.mean(axis=0)
    up = up / np.linalg.norm(up)

    return center, radius, up


def make_circle_camera(progress, center, radius, up, look_target, ref_camera):
    """Create a camera positioned on a circle, looking at a target point.

    Args:
        progress: float in [0, 1) — position along the circle (0=start, 1=full loop)
        center: np.array [3] — center of the camera orbit circle
        radius: float — orbit radius
        up: np.array [3] — scene up vector (defines the orbit plane)
        look_target: np.array [3] — world point the camera looks at
        ref_camera: Camera — reference for FoV and image resolution
    """
    up = up / np.linalg.norm(up)

    # Orthonormal basis for the orbit plane (perpendicular to up)
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(up, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(up, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(up, u)
    v = v / np.linalg.norm(v)

    theta = progress * 2 * math.pi
    cam_pos = center + radius * math.cos(theta) * u + radius * math.sin(theta) * v

    # View matrix (COLMAP convention)
    forward = look_target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    cam_up = cam_up / np.linalg.norm(cam_up)

    R = np.stack([right, cam_up, forward], axis=1)  # c2w rotation [3,3]
    T = -R.T @ cam_pos  # w2c translation

    H, W = ref_camera.image_height, ref_camera.image_width
    return Camera(
        colmap_id=0, R=R, T=T,
        FoVx=ref_camera.FoVx, FoVy=ref_camera.FoVy,
        image=torch.zeros(3, H, W), gt_alpha_mask=None,
        image_name=f"circle_{progress:.3f}", uid=0,
    )


def compute_orbit_params(scene, radius_scale=1.0, elevation=0.0, tilt=0.0):
    """Compute orbit parameters from training cameras + user adjustments.

    Args:
        scene: Scene object with training cameras.
        radius_scale: Multiplier for orbit radius (>1 = zoom out, <1 = zoom in).
        elevation: Shift camera orbit vertically (units of scene radius).
                   Positive = move cameras up.
        tilt: Shift the look-at target along the up vector (units of scene radius).
              The sign convention depends on the scene's up vector orientation.
              Try positive first; if the camera looks the wrong way, flip the sign.

    Returns:
        dict with: orbit_center, orbit_radius, up, look_target, ref_camera,
                   center, radius (original scene params)
    """
    train_cameras = scene.getTrainCameras()
    center, radius, up = estimate_scene_params(train_cameras)
    print(f"[SCENE] Center: {center}, Radius: {radius:.2f}, Up: {up}")

    orbit_radius = radius * radius_scale
    orbit_center = center + up * radius * elevation
    look_target = center + up * radius * tilt

    return dict(
        orbit_center=orbit_center, orbit_radius=orbit_radius,
        up=up, look_target=look_target,
        ref_camera=train_cameras[0],
        center=center, radius=radius,
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_frames(model, orbit, num_views, decompose=False):
    """Render frames around the circular trajectory.

    Args:
        model: dict from load_model()
        orbit: dict from compute_orbit_params()
        num_views: number of frames around the circle
        decompose: if True, also render gaussian_only and ngp_only decompositions

    Returns:
        all_frames: dict mapping mode name -> list of [3,H,W] tensors
        avg_fps: average rendering FPS (full mode only)
    """
    gaussians = model['gaussians']
    pipe, ingp, cfg = model['pipe'], model['ingp'], model['cfg']
    beta, iteration = model['beta'], model['iteration']
    skybox, background_mode = model['skybox'], model['background_mode']
    aabb_mode = getattr(model['args'], 'aabb', '2dgs')
    background = torch.zeros(3, device="cuda")

    orbit_center = orbit['orbit_center']
    orbit_radius = orbit['orbit_radius']
    up, look_target = orbit['up'], orbit['look_target']
    ref_camera = orbit['ref_camera']

    render_modes = [('full', None)]
    if decompose:
        render_modes += [('gaussian_only', 'gaussian_only'), ('ngp_only', 'ngp_only')]

    # Warmup (exclude from timing)
    warmup_cam = make_circle_camera(0.0, orbit_center, orbit_radius, up, look_target, ref_camera)
    with torch.no_grad():
        render(warmup_cam, gaussians, pipe, background, ingp=ingp, beta=beta,
               iteration=iteration, cfg=cfg, skybox=skybox,
               background_mode=background_mode, aabb_mode=aabb_mode)
    torch.cuda.synchronize()

    all_frames = {name: [] for name, _ in render_modes}
    render_times = []

    with torch.no_grad():
        for i in range(num_views):
            progress = i / num_views
            cam = make_circle_camera(progress, orbit_center, orbit_radius, up, look_target, ref_camera)

            for mode_name, decompose_mode in render_modes:
                if mode_name == 'full':
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()

                render_pkg = render(cam, gaussians, pipe, background, ingp=ingp, beta=beta,
                                   iteration=iteration, cfg=cfg, skybox=skybox,
                                   background_mode=background_mode, aabb_mode=aabb_mode,
                                   decompose_mode=decompose_mode)
                rendered = torch.clamp(render_pkg["render"], 0.0, 1.0)

                if mode_name == 'full':
                    torch.cuda.synchronize()
                    render_times.append(time.perf_counter() - t0)

                all_frames[mode_name].append(rendered)

            decompose_str = " (+decompose)" if decompose else ""
            print(f"[RENDER] Frame {i+1}/{num_views}{decompose_str}")

    avg_fps = 1.0 / np.mean(render_times) if render_times else 0
    avg_ms = np.mean(render_times) * 1000 if render_times else 0
    print(f"[BENCHMARK] Avg render: {avg_ms:.1f} ms ({avg_fps:.1f} FPS) over {len(render_times)} frames")

    return all_frames, avg_fps


def save_frames_to_disk(all_frames, model_path):
    """Save rendered frames as PNGs.

    Saves to:
        <model_path>/circle_renders/          — full renders
        <model_path>/circle_renders_gaussian_only/  — gaussian-only decomposition
        <model_path>/circle_renders_ngp_only/       — ngp-only decomposition

    Args:
        all_frames: dict from render_frames()
        model_path: base model directory
    """
    for mode_name, frames in all_frames.items():
        if mode_name == 'full':
            out_dir = os.path.join(model_path, "circle_renders")
        else:
            out_dir = os.path.join(model_path, f"circle_renders_{mode_name}")
        os.makedirs(out_dir, exist_ok=True)

        for i, frame in enumerate(frames):
            save_image(frame, os.path.join(out_dir, f"frame_{i:04d}.png"))

    n = len(all_frames.get('full', []))
    print(f"[SAVED] {n} frames × {len(all_frames)} modes to {model_path}/circle_renders*/")


def save_videos(all_frames, model_path, fps=20):
    """Encode frame lists as MP4 videos using OpenCV.

    Saves circle.mp4 in each output directory.

    Args:
        all_frames: dict from render_frames()
        model_path: base model directory
        fps: video playback framerate
    """
    import cv2

    for mode_name, frames in all_frames.items():
        if len(frames) < 2:
            continue

        if mode_name == 'full':
            video_dir = os.path.join(model_path, "circle_renders")
        else:
            video_dir = os.path.join(model_path, f"circle_renders_{mode_name}")
        os.makedirs(video_dir, exist_ok=True)

        video_path = os.path.join(video_dir, "circle.mp4")
        h, w = frames[0].shape[1], frames[0].shape[2]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for frame in frames:
            frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[VIDEO] {mode_name}: {video_path}")
