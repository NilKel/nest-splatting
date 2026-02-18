#!/usr/bin/env python3
"""
Render a circular camera trajectory around a scene and create a video.

Usage:
    python scripts/render_trajectory.py --model_path outputs/mip_360/treehill/cat/...
    python scripts/render_trajectory.py --model_path outputs/mip_360/treehill/cat/... --ref_camera _DSC8954
"""

import os
import sys
import json
import pickle
import argparse
import glob
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import Namespace
from scene import Scene, GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams
from utils.graphics_utils import focal2fov, getWorld2View2


def load_training_config(model_path):
    """Load training configuration from checkpoint directory."""
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


def create_look_at_matrix(camera_pos, target, up=np.array([-0.07, -0.93, -0.36])):
    """
    Create a camera rotation matrix that looks from camera_pos toward target.
    Returns R in the convention used by this codebase.

    Convention: R.T goes into view matrix, so R transforms camera coords to world coords.
    R columns are camera basis vectors in world coordinates:
      - Column 0: camera X (right)
      - Column 1: camera Y (down, since Y points down in image)
      - Column 2: camera Z (backward, since camera looks down -Z in OpenGL convention)
    """
    # Forward vector (from camera to target)
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)

    # Right vector (forward cross up gives right)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        # Handle case where forward is parallel to up
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Recompute up to be orthogonal
    up_vec = np.cross(right, forward)
    up_vec = up_vec / np.linalg.norm(up_vec)

    # R columns: [right, down, backward]
    # down = -up_vec (since up_vec points up in world, camera Y points down)
    # backward = -forward (since camera looks down -Z)
    R = np.column_stack([right, -up_vec, -forward])

    return R.astype(np.float32)


def create_circular_trajectory(center, radius, height, n_frames, ref_camera=None,
                                start_angle=0, end_angle=360, look_at_z=None):
    """
    Create a circular camera trajectory around a center point.

    Args:
        center: [x, y, z] center point to orbit around
        radius: distance from center
        height: z-coordinate of camera
        n_frames: number of frames
        ref_camera: reference camera to extract FoV and image size from
        start_angle: starting angle in degrees (0 = +X direction)
        end_angle: ending angle in degrees
        look_at_z: z-coordinate to look at (default: center[2])
    """
    # Get camera parameters from reference
    if ref_camera is not None:
        FoVx = ref_camera.FoVx
        FoVy = ref_camera.FoVy
        width = ref_camera.image_width
        height_img = ref_camera.image_height
    else:
        FoVx = 1.08
        FoVy = 0.75
        width = 1267
        height_img = 832

    # Look-at target
    if look_at_z is None:
        look_at_z = center[2]
    look_at = np.array([center[0], center[1], look_at_z])

    cameras = []
    for i in range(n_frames):
        # Interpolate angle from start to end
        t = i / max(n_frames - 1, 1)
        angle_deg = start_angle + t * (end_angle - start_angle)
        angle = np.radians(angle_deg)

        # Camera position on circle around center
        pos = np.array([
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            height
        ], dtype=np.float32)

        # Create rotation matrix looking at center
        R = create_look_at_matrix(pos, look_at)

        # Translation: T = -R.T @ pos (since R.T transforms world to camera)
        T = -R.T @ pos

        cameras.append({
            'R': R,
            'T': T,
            'FoVx': FoVx,
            'FoVy': FoVy,
            'width': width,
            'height': height_img,
            'frame_id': i,
            'pos': pos  # For debugging
        })

    return cameras


def make_camera(cam_info, uid):
    """Create a Camera object from camera info dict."""
    image = torch.zeros(3, cam_info['height'], cam_info['width'])

    return Camera(
        colmap_id=uid,
        R=cam_info['R'],
        T=cam_info['T'],
        FoVx=cam_info['FoVx'],
        FoVy=cam_info['FoVy'],
        image=image,
        gt_alpha_mask=None,
        image_name=f"traj_{uid:04d}",
        uid=uid,
        data_device='cuda'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--n_frames', type=int, default=120)
    parser.add_argument('--radius', type=float, default=None, help='Orbit radius (auto from ref camera if not set)')
    parser.add_argument('--height', type=float, default=None, help='Camera Z height (auto from ref camera if not set)')
    parser.add_argument('--center_x', type=float, default=0.0)
    parser.add_argument('--center_y', type=float, default=0.0)
    parser.add_argument('--center_z', type=float, default=0.0, help='Z coordinate of orbit center')
    parser.add_argument('--look_at_z', type=float, default=None, help='Z coordinate to look at (default: center_z)')
    parser.add_argument('--ref_camera', type=str, default=None, help='Reference camera name substring (e.g., _DSC8954)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--start_angle', type=float, default=0.0, help='Start angle in degrees (0 = +X)')
    parser.add_argument('--end_angle', type=float, default=360.0, help='End angle in degrees')
    parser.add_argument('--skip_video', action='store_true', help='Skip video creation, just render frames')

    args = parser.parse_args()

    # Find iteration
    if args.iteration == -1:
        ngp_files = glob.glob(os.path.join(args.model_path, "ngp_*.pth"))
        if ngp_files:
            iterations = [int(os.path.basename(f).replace("ngp_", "").replace(".pth", "")) for f in ngp_files]
            args.iteration = max(iterations)
        else:
            raise FileNotFoundError("No ngp_*.pth checkpoints found")

    print(f"Loading model from {args.model_path}, iteration {args.iteration}")

    # Load training config
    train_args = load_training_config(args.model_path)
    train_args.model_path = args.model_path
    train_args.eval = True

    # Load YAML config
    config_yaml_path = os.path.join(args.model_path, "config.yaml")
    if os.path.exists(config_yaml_path):
        cfg_model = Config(config_yaml_path)
    else:
        cfg_model = Config(train_args.yaml)

    # Setup models
    temp_parser = argparse.ArgumentParser()
    model_params = ModelParams(temp_parser, sentinel=True)
    pipeline_params = PipelineParams(temp_parser)

    dataset = model_params.extract(train_args)
    pipe = pipeline_params.extract(train_args)

    # Load INGP
    ingp_model = INGP(cfg_model, args=train_args).to('cuda')
    ingp_model.load_model(args.model_path, args.iteration)

    # Load Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    gaussians.XYZ_TYPE = "UV"
    ingp_model.set_active_levels(args.iteration)

    if hasattr(train_args, 'kernel'):
        gaussians.kernel_type = train_args.kernel

    # Load skybox if available
    skybox = None
    background_mode = getattr(train_args, 'background', 'none')
    if background_mode is None:
        background_mode = 'none'
    if background_mode in ['skybox_dense', 'skybox_sparse']:
        from scene.background import LearnableSkybox
        skybox_res = getattr(train_args, 'skybox_res', 512)
        skybox = LearnableSkybox(resolution_h=skybox_res, resolution_w=skybox_res * 2).cuda()
        if skybox.load_model(args.model_path, args.iteration):
            print(f"Loaded skybox (mode={background_mode})")

    num_gaussians = len(gaussians.get_xyz)
    print(f"Gaussians: {num_gaussians:,}")

    # Find reference camera
    test_cameras = scene.getTestCameras()
    ref_cam = None
    if args.ref_camera:
        for cam in test_cameras:
            if args.ref_camera in cam.image_name:
                ref_cam = cam
                print(f"Using reference camera: {cam.image_name}")
                break
        if ref_cam is None:
            print(f"Warning: Reference camera '{args.ref_camera}' not found, using first test camera")
            ref_cam = test_cameras[0] if test_cameras else None
    elif test_cameras:
        ref_cam = test_cameras[0]
        print(f"Using first test camera as reference: {ref_cam.image_name}")

    # Extract radius and height from reference camera if not specified
    if ref_cam is not None:
        ref_pos = ref_cam.camera_center.cpu().numpy()
        print(f"Reference camera position: {ref_pos}")

        center = np.array([args.center_x, args.center_y, args.center_z])

        if args.radius is None:
            # Distance from center in XY plane
            args.radius = np.sqrt((ref_pos[0] - center[0])**2 + (ref_pos[1] - center[1])**2)
            print(f"Auto radius from ref camera: {args.radius:.2f}")

        if args.height is None:
            args.height = ref_pos[2]
            print(f"Auto height from ref camera: {args.height:.2f}")
    else:
        if args.radius is None:
            args.radius = 3.5
        if args.height is None:
            args.height = -0.75

    # Create trajectory
    center = np.array([args.center_x, args.center_y, args.center_z])
    print(f"\nCreating circular trajectory:")
    print(f"  Center: {center}")
    print(f"  Radius: {args.radius:.2f}")
    print(f"  Height: {args.height:.2f}")
    print(f"  Frames: {args.n_frames}")

    # Look-at Z (default to center_z if not specified)
    look_at_z = args.look_at_z if args.look_at_z is not None else args.center_z

    traj_cameras = create_circular_trajectory(
        center, args.radius, args.height, args.n_frames, ref_camera=ref_cam,
        start_angle=args.start_angle, end_angle=args.end_angle, look_at_z=look_at_z
    )

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, 'trajectory_video')
    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    # Render
    background = torch.zeros(3, device="cuda")
    beta = cfg_model.surfel.tg_beta

    print(f"\nRendering {args.n_frames} frames to {frames_dir}...")
    with torch.no_grad():
        for i, cam_info in enumerate(traj_cameras):
            cam = make_camera(cam_info, i)

            render_pkg = render(cam, gaussians, pipe, background,
                               ingp=ingp_model, beta=beta,
                               iteration=args.iteration, cfg=cfg_model,
                               skybox=skybox, background_mode=background_mode)

            image = render_pkg["render"].clamp(0, 1)
            image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            Image.fromarray(image_np).save(os.path.join(frames_dir, f'{i:04d}.png'))

            if (i + 1) % 10 == 0:
                print(f"  Rendered {i + 1}/{args.n_frames}")

    print(f"\nFrames saved to: {frames_dir}")

    # Create video with ffmpeg
    if not args.skip_video:
        video_path = os.path.join(args.output_dir, 'trajectory.mp4')
        ffmpeg_cmd = f'ffmpeg -y -framerate {args.fps} -i "{frames_dir}/%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "{video_path}"'
        print(f"\nCreating video with command:")
        print(f"  {ffmpeg_cmd}")
        ret = os.system(ffmpeg_cmd)
        if ret == 0:
            print(f"\nVideo saved to: {video_path}")
        else:
            print(f"\nffmpeg failed (return code {ret}). You may need to install ffmpeg.")
            print("To install ffmpeg on Ubuntu: sudo apt install ffmpeg")
            print(f"Or create video manually: {ffmpeg_cmd}")
    else:
        print("\nSkipped video creation (--skip_video)")
        print(f"To create video manually:")
        print(f"  ffmpeg -y -framerate {args.fps} -i \"{frames_dir}/%04d.png\" -c:v libx264 -pix_fmt yuv420p -crf 18 \"{args.output_dir}/trajectory.mp4\"")


if __name__ == '__main__':
    main()
