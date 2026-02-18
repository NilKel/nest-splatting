#!/usr/bin/env python3
"""
Render an interpolated camera trajectory through test cameras using spline interpolation.

Usage:
    python scripts/render_interpolated_trajectory.py --model_path outputs/mip_360/treehill/cat/...
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
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import Namespace
from scene import Scene, GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from hash_encoder.modules import INGP
from hash_encoder.config import Config
from arguments import ModelParams, PipelineParams


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


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    rot = Rotation.from_matrix(R)
    return rot.as_quat()  # Returns [x, y, z, w], we'll handle this in slerp


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    rot = Rotation.from_quat(q)
    return rot.as_matrix()


def interpolate_cameras(cameras, n_frames, loop=True):
    """
    Interpolate between cameras using cubic splines for position and SLERP for rotation.

    Args:
        cameras: list of Camera objects
        n_frames: total number of output frames
        loop: if True, create a looping trajectory
    """
    n_cams = len(cameras)

    # Extract camera data
    positions = np.array([c.camera_center.cpu().numpy() for c in cameras])
    rotations = [Rotation.from_matrix(c.R.cpu().numpy()) for c in cameras]

    # Get FoV and image size from first camera
    FoVx = cameras[0].FoVx
    FoVy = cameras[0].FoVy
    width = cameras[0].image_width
    height = cameras[0].image_height

    if loop:
        # Append first camera to end for looping
        positions = np.vstack([positions, positions[0:1]])
        rotations = rotations + [rotations[0]]
        n_cams += 1

    # Parameter t for each camera (0 to 1)
    t_cameras = np.linspace(0, 1, n_cams)

    # Create cubic spline for positions
    cs_x = CubicSpline(t_cameras, positions[:, 0], bc_type='periodic' if loop else 'natural')
    cs_y = CubicSpline(t_cameras, positions[:, 1], bc_type='periodic' if loop else 'natural')
    cs_z = CubicSpline(t_cameras, positions[:, 2], bc_type='periodic' if loop else 'natural')

    # Create SLERP for rotations
    slerp = Slerp(t_cameras, Rotation.concatenate(rotations))

    # Interpolate
    t_interp = np.linspace(0, 1, n_frames, endpoint=not loop)

    interp_cameras = []
    for i, t in enumerate(t_interp):
        pos = np.array([cs_x(t), cs_y(t), cs_z(t)])
        rot = slerp(t)
        R = rot.as_matrix().astype(np.float32)

        # Compute T from R and position
        # T = -R.T @ pos
        T = -R.T @ pos

        interp_cameras.append({
            'R': R,
            'T': T.astype(np.float32),
            'FoVx': FoVx,
            'FoVy': FoVy,
            'width': width,
            'height': height,
            'frame_id': i
        })

    return interp_cameras


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
        image_name=f"interp_{uid:04d}",
        uid=uid,
        data_device='cuda'
    )


def sort_cameras_by_angle(cameras):
    """Sort cameras by their angle around the scene center."""
    positions = np.array([c.camera_center.cpu().numpy() for c in cameras])
    angles = np.arctan2(positions[:, 1], positions[:, 0])
    sorted_indices = np.argsort(angles)
    return [cameras[i] for i in sorted_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--n_frames', type=int, default=120)
    parser.add_argument('--use_train', action='store_true', help='Use train cameras instead of test')
    parser.add_argument('--no_loop', action='store_true', help='Do not create looping trajectory')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--skip_video', action='store_true', help='Skip video creation')

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

    # Get cameras
    if args.use_train:
        cameras = scene.getTrainCameras()
        print(f"Using {len(cameras)} train cameras")
    else:
        cameras = scene.getTestCameras()
        print(f"Using {len(cameras)} test cameras")

    # Sort cameras by angle for smooth trajectory
    cameras = sort_cameras_by_angle(cameras)
    print("Sorted cameras by angle around scene center")

    # Print camera order
    for i, cam in enumerate(cameras):
        pos = cam.camera_center.cpu().numpy()
        angle = np.arctan2(pos[1], pos[0]) * 180 / np.pi
        print(f"  {i:2d}. {cam.image_name}: angle={angle:.1f}°")

    # Create interpolated trajectory
    print(f"\nCreating interpolated trajectory with {args.n_frames} frames...")
    loop = not args.no_loop
    traj_cameras = interpolate_cameras(cameras, args.n_frames, loop=loop)

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, 'interpolated_trajectory')
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
            print(f"\nffmpeg failed. Install with: sudo apt install ffmpeg")
            print(f"Then run: {ffmpeg_cmd}")
    else:
        print("\nSkipped video creation (--skip_video)")
        print(f"To create video: ffmpeg -y -framerate {args.fps} -i \"{frames_dir}/%04d.png\" -c:v libx264 -pix_fmt yuv420p -crf 18 \"{args.output_dir}/trajectory.mp4\"")


if __name__ == '__main__':
    main()
