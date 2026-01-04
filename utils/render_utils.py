# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import enum
import types
from typing import List, Mapping, Optional, Text, Tuple, Union
import copy
from PIL import Image
import mediapy as media
from matplotlib import cm
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

def normalize(x: np.ndarray) -> np.ndarray:
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = average_pose(poses)
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """Construct lookat view matrix."""
  vec2 = normalize(lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]
  t_mean = t.mean(axis=0)
  t = t - t_mean

  eigval, eigvec = np.linalg.eig(t.T @ t)
  # Sort eigenvectors in order of largest to smallest eigenvalue.
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # Flip coordinate system if z component of y-axis is negative
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  return poses_recentered, transform
  # points = np.random.rand(3,100)
  # points_h = np.concatenate((points,np.ones_like(points[:1])), axis=0)
  # (poses_recentered @ points_h)[0]
  # (transform @ pad_poses(poses) @ points_h)[0,:3]
  # import pdb; pdb.set_trace()

  # # Just make sure it's it in the [-1, 1]^3 cube
  # scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
  # poses_recentered[:, :3, 3] *= scale_factor
  # transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

  # return poses_recentered, transform

def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
  """Generate an elliptical render path based on the given poses."""
  # Calculate the focal point for the path (cameras point toward this).
  center = focus_point_fn(poses)
  # Path height sits at z=0 (in middle of zero-mean capture pattern).
  offset = np.array([center[0], center[1], 0])

  # Calculate scaling for ellipse axes based on input camera positions.
  sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
  # Use ellipse that is symmetric about the focal point in xy.
  low = -sc + offset
  high = sc + offset
  # Optional height variation need not be symmetric
  z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
  z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

  def get_positions(theta):
    # Interpolate between bounds with trig functions to get ellipse in x-y.
    # Optionally also interpolate in z to change camera height along path.
    return np.stack([
        low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
        low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
        z_variation * (z_low[2] + (z_high - z_low)[2] *
                       (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
    ], -1)

  theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
  positions = get_positions(theta)

  #if const_speed:

  # # Resample theta angles so that the velocity is closer to constant.
  # lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
  # theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
  # positions = get_positions(theta)

  # Throw away duplicated last position.
  positions = positions[:-1]

  # Set path's up vector to axis closest to average of input pose up vectors.
  avg_up = poses[:, :3, 1].mean(0)
  avg_up = avg_up / np.linalg.norm(avg_up)
  ind_up = np.argmax(np.abs(avg_up))
  up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

  return np.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_path(viewpoint_cameras, n_frames=480):
  c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
  pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
  pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

  # generate new poses
  new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
  # warp back to orignal scale
  new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

  traj = []
  for c2w in new_poses:
      c2w = c2w @ np.diag([1, -1, -1, 1])
      cam = copy.deepcopy(viewpoint_cameras[0])
      cam.image_height = int(cam.image_height / 2) * 2
      cam.image_width = int(cam.image_width / 2) * 2
      cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
      cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
      cam.camera_center = cam.world_view_transform.inverse()[3, :3]
      traj.append(cam)

  return traj

def load_img(pth: str) -> np.ndarray:
  """Load an image and cast to float32."""
  with open(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)
  return image


def create_videos(base_dir, input_dir, out_name, num_frames=480):
  """Creates videos out of the images saved to disk."""
  # Last two parts of checkpoint path are experiment name and scene name.
  video_prefix = f'{out_name}'

  zpad = max(5, len(str(num_frames - 1)))
  idx_to_str = lambda idx: str(idx).zfill(zpad)

  os.makedirs(base_dir, exist_ok=True)
  render_dist_curve_fn = np.log
  
  # Load one example frame to get image shape and depth range.
  depth_file = os.path.join(input_dir, 'vis', f'depth_{idx_to_str(0)}.tiff')
  depth_frame = load_img(depth_file)
  shape = depth_frame.shape
  p = 3
  distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
  lo, hi = [render_dist_curve_fn(x) for x in distance_limits]
  print(f'Video shape is {shape[:2]}')

  video_kwargs = {
      'shape': shape[:2],
      'codec': 'h264',
      'fps': 60,
      'crf': 18,
  }
  
  for k in ['depth', 'normal', 'color']:
    video_file = os.path.join(base_dir, f'{video_prefix}_{k}.mp4')
    input_format = 'gray' if k == 'alpha' else 'rgb'
    

    file_ext = 'png' if k in ['color', 'normal'] else 'tiff'
    idx = 0

    if k == 'color':
      file0 = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.{file_ext}')
    else:
      file0 = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(0)}.{file_ext}')

    if not os.path.exists(file0):
      print(f'Images missing for tag {k}')
      continue
    print(f'Making video {video_file}...')
    with media.VideoWriter(
        video_file, **video_kwargs, input_format=input_format) as writer:
      for idx in tqdm(range(num_frames)):
        # img_file = os.path.join(input_dir, f'{k}_{idx_to_str(idx)}.{file_ext}')
        if k == 'color':
          img_file = os.path.join(input_dir, 'renders', f'{idx_to_str(idx)}.{file_ext}')
        else:
          img_file = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(idx)}.{file_ext}')

        if not os.path.exists(img_file):
          ValueError(f'Image file {img_file} does not exist.')
        img = load_img(img_file)
        if k in ['color', 'normal']:
          img = img / 255.
        elif k.startswith('depth'):
          img = render_dist_curve_fn(img)
          img = np.clip((img - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
          img = cm.get_cmap('turbo')(img)[..., :3]

        frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
        writer.add_image(frame)
        idx += 1

def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')

def gsnum_trans_color(gs_num, MAX_N = 30):
  # print('gs_num: ', gs_num.shape)
  gs_num = gs_num.squeeze()
  input_shape = gs_num.shape
  
  vis_img = torch.zeros((3,) + input_shape) 
  
  normed_gs = gs_num / MAX_N
  normed_gs = torch.clamp(normed_gs, 0, 1)

  # 前一半的颜色: 红色 -> 黄色 (红色 (1, 0, 0) -> 黄色 (1, 1, 0))
  mask_first_half = normed_gs <= 0.5
  vis_img[0][mask_first_half] = 1.0  # 红色和黄色都有红色分量
  vis_img[1][mask_first_half] = 2 * normed_gs[mask_first_half]  # 绿色分量从 0 -> 1
  vis_img[2][mask_first_half] = 0.0  # 没有蓝色分量

  # 后一半的颜色: 黄色 -> 绿色 (黄色 (1, 1, 0) -> 绿色 (0, 1, 0))
  mask_second_half = normed_gs > 0.5
  vis_img[0][mask_second_half] = 2 * (1 - normed_gs[mask_second_half])  # 红色分量从 1 -> 0
  vis_img[1][mask_second_half] = 1.0  # 全绿
  vis_img[2][mask_second_half] = 0.0  # 没有蓝色分量

  return vis_img


def convert_gray_to_cmap(img_gray, map_mode = 'jet', revert = False, vmax = None):

    img_gray = copy.deepcopy(img_gray)
    shape = img_gray.shape
    cmap = plt.get_cmap(map_mode)
    if vmax is not None:
        img_gray = (img_gray / vmax).clip(0,1)
    else:
        img_gray = img_gray / (np.max(img_gray)+1e-6)
    if revert:
        img_gray = 1- img_gray
    colors = cmap(img_gray.reshape(-1))[:, :3]
    # visualization
    colors = colors.reshape(shape+tuple([3])) #*255
    return colors


def create_intersection_heatmap(gaussian_num, legend_width=80, max_display=200, colormap='turbo'):
    """
    Create a heatmap visualization of per-pixel Gaussian intersection counts.

    Args:
        gaussian_num: Tensor or array of shape (1, H, W) or (H, W) containing
                     the number of Gaussians that contributed to each pixel.
        legend_width: Width of the legend bar on the right side (default 80px).
        max_display: Fixed maximum for colormap normalization (default 200).
                    Values above this are clipped and shown as "outliers".
        colormap: Matplotlib colormap name (default 'turbo' - perceptually uniform).

    Returns:
        heatmap_img: numpy array of shape (H, W + legend_width, 3) in [0, 1] range
        min_count: minimum intersection count (excluding background)
        max_count: maximum intersection count (actual, may exceed max_display)

    Colormap: Black (BG, count=0) -> colormap[0..max_display]
    """
    # Convert to numpy if tensor
    if hasattr(gaussian_num, 'cpu'):
        gs_count = gaussian_num.squeeze().cpu().numpy()
    else:
        gs_count = np.squeeze(gaussian_num)

    H, W = gs_count.shape

    # Find actual min/max excluding zeros (background)
    valid_mask = gs_count > 0
    if valid_mask.any():
        min_count = int(gs_count[valid_mask].min())
        max_count = int(gs_count[valid_mask].max())
    else:
        min_count, max_count = 0, 1

    # Normalize to [0, 1] using fixed max_display (not actual max)
    # This ensures consistent coloring across all views
    normalized = np.zeros_like(gs_count, dtype=np.float32)
    normalized[valid_mask] = gs_count[valid_mask] / max_display
    normalized = np.clip(normalized, 0, 1)

    # Apply colormap (turbo is perceptually uniform and good for heatmaps)
    cmap = plt.get_cmap(colormap)
    heatmap_rgba = cmap(normalized)  # (H, W, 4)
    heatmap = heatmap_rgba[..., :3].astype(np.float32)

    # Set background (count=0) to black
    heatmap[~valid_mask] = 0.0

    # Create legend bar with colormap gradient
    legend = np.zeros((H, legend_width, 3), dtype=np.float32)

    # Gradient bar (centered, with padding)
    bar_start = int(H * 0.08)
    bar_end = int(H * 0.92)
    bar_height = bar_end - bar_start
    bar_x_start = 15
    bar_x_end = legend_width - 15

    # Create vertical gradient using the colormap
    for y in range(bar_start, bar_end):
        t = 1.0 - (y - bar_start) / max(bar_height - 1, 1)  # 1 at top, 0 at bottom
        color = cmap(t)[:3]
        legend[y, bar_x_start:bar_x_end, :] = color

    # Combine heatmap and legend
    heatmap_with_legend = np.concatenate([heatmap, legend], axis=1)

    # Add text labels using matplotlib figure
    fig, ax = plt.subplots(figsize=(((W + legend_width) / 100), (H / 100)), dpi=100)
    ax.imshow(heatmap_with_legend)
    ax.axis('off')

    # Add labels on the legend
    text_x = W + legend_width // 2

    # Show max_display at top, 0 at bottom
    ax.text(text_x, bar_start - 8, f'{max_display}+', fontsize=9, ha='center', va='bottom',
            color='white', fontweight='bold')
    ax.text(text_x, bar_end + 8, '1', fontsize=9, ha='center', va='top',
            color='white', fontweight='bold')

    # Title
    ax.text(text_x, H * 0.03, 'Intersections', fontsize=8, ha='center', va='center', color='white')

    # Show actual stats at bottom
    outlier_count = np.sum(gs_count > max_display)
    outlier_pct = 100.0 * outlier_count / np.sum(valid_mask) if valid_mask.any() else 0
    ax.text(text_x, H * 0.96, f'max:{max_count}', fontsize=7, ha='center', va='center', color='yellow')
    if outlier_pct > 0.1:
        ax.text(text_x, H * 0.99, f'>{max_display}: {outlier_pct:.1f}%', fontsize=6,
                ha='center', va='center', color='red')

    # Render figure to numpy array
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Get the RGBA buffer and convert to RGB
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    result = buf[..., :3].astype(np.float32) / 255.0

    plt.close(fig)

    # Resize to original dimensions if needed (matplotlib may change size slightly)
    if result.shape[0] != H or result.shape[1] != W + legend_width:
        from PIL import Image as PILImage
        result_pil = PILImage.fromarray((result * 255).astype(np.uint8))
        result_pil = result_pil.resize((W + legend_width, H), PILImage.LANCZOS)
        result = np.array(result_pil).astype(np.float32) / 255.0

    return result, min_count, max_count


def create_flex_beta_heatmap(flex_beta_map, alpha_map, legend_width=80, min_display=0.0, max_display=10.0):
    """
    Create a heatmap visualization of per-pixel flex kernel beta values.
    Uses coolwarm colormap: blue (high beta, hard) -> red (low beta, soft).

    Args:
        flex_beta_map: Tensor of shape (1, H, W) containing alpha-weighted flex beta values
        alpha_map: Tensor of shape (1, H, W) containing alpha values for masking background
        legend_width: Width of the legend bar on the right side (default 80px)
        min_display: Minimum beta for colormap (default 0.0)
        max_display: Maximum beta for colormap (default 10.0, values above clipped)

    Returns:
        heatmap_img: numpy array of shape (H, W + legend_width, 3) in [0, 1] range
        min_beta: minimum beta value (excluding background)
        max_beta: maximum beta value
    """
    # Convert to numpy
    if hasattr(flex_beta_map, 'cpu'):
        beta_vals = flex_beta_map.squeeze().cpu().numpy()
    else:
        beta_vals = np.squeeze(flex_beta_map)

    if hasattr(alpha_map, 'cpu'):
        alpha_vals = alpha_map.squeeze().cpu().numpy()
    else:
        alpha_vals = np.squeeze(alpha_map)

    H, W = beta_vals.shape

    # Valid mask: where alpha > threshold (not background)
    valid_mask = alpha_vals > 0.01

    # Find actual min/max excluding background
    if valid_mask.any():
        min_beta = float(beta_vals[valid_mask].min())
        max_beta = float(beta_vals[valid_mask].max())
    else:
        min_beta, max_beta = 0.0, 1.0

    # Normalize to [0, 1] using display range
    normalized = np.zeros_like(beta_vals, dtype=np.float32)
    normalized[valid_mask] = (beta_vals[valid_mask] - min_display) / (max_display - min_display)
    normalized = np.clip(normalized, 0, 1)

    # Apply coolwarm colormap (reversed: high beta = blue, low beta = red)
    # coolwarm: red (0) -> white (0.5) -> blue (1)
    # We want: low beta (soft) = red, high beta (hard) = blue
    cmap = plt.get_cmap('coolwarm')
    heatmap_rgba = cmap(normalized)  # (H, W, 4)
    heatmap = heatmap_rgba[..., :3].astype(np.float32)

    # Set background to black
    heatmap[~valid_mask] = 0.0

    # Create legend bar
    legend = np.zeros((H, legend_width, 3), dtype=np.float32)

    bar_start = int(H * 0.08)
    bar_end = int(H * 0.92)
    bar_height = bar_end - bar_start
    bar_x_start = 15
    bar_x_end = legend_width - 15

    # Create vertical gradient (high beta at top = blue, low at bottom = red)
    for y in range(bar_start, bar_end):
        t = 1.0 - (y - bar_start) / max(bar_height - 1, 1)  # 1 at top, 0 at bottom
        color = cmap(t)[:3]
        legend[y, bar_x_start:bar_x_end, :] = color

    # Combine heatmap and legend
    heatmap_with_legend = np.concatenate([heatmap, legend], axis=1)

    # Add text labels
    fig, ax = plt.subplots(figsize=(((W + legend_width) / 100), (H / 100)), dpi=100)
    ax.imshow(heatmap_with_legend)
    ax.axis('off')

    text_x = W + legend_width // 2

    # Labels: high beta (hard) at top, low beta (soft) at bottom
    ax.text(text_x, bar_start - 8, f'{max_display:.1f}', fontsize=9, ha='center', va='bottom',
            color='white', fontweight='bold')
    ax.text(text_x, bar_end + 8, f'{min_display:.1f}', fontsize=9, ha='center', va='top',
            color='white', fontweight='bold')

    # Title
    ax.text(text_x, H * 0.03, 'Flex β', fontsize=8, ha='center', va='center', color='white')

    # Show actual stats
    ax.text(text_x, H * 0.96, f'range:{min_beta:.2f}-{max_beta:.2f}', fontsize=6,
            ha='center', va='center', color='yellow')

    # Render figure to numpy array
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    result = buf[..., :3].astype(np.float32) / 255.0

    plt.close(fig)

    # Resize if needed
    if result.shape[0] != H or result.shape[1] != W + legend_width:
        from PIL import Image as PILImage
        result_pil = PILImage.fromarray((result * 255).astype(np.uint8))
        result_pil = result_pil.resize((W + legend_width, H), PILImage.LANCZOS)
        result = np.array(result_pil).astype(np.float32) / 255.0

    return result, min_beta, max_beta


def create_intersection_histogram(gaussian_num, max_display=200, num_bins=50):
    """
    Create a histogram visualization showing the distribution of intersection counts.
    Useful for identifying outliers and understanding the distribution.

    Args:
        gaussian_num: Tensor or array of shape (1, H, W) or (H, W)
        max_display: X-axis maximum (values beyond shown in overflow bin)
        num_bins: Number of histogram bins

    Returns:
        histogram_img: numpy array of shape (400, 600, 3) in [0, 1] range
        stats: dict with min, max, mean, median, std, outlier_pct
    """
    # Convert to numpy if tensor
    if hasattr(gaussian_num, 'cpu'):
        gs_count = gaussian_num.squeeze().cpu().numpy()
    else:
        gs_count = np.squeeze(gaussian_num)

    # Get valid (non-background) pixels
    valid_mask = gs_count > 0
    valid_counts = gs_count[valid_mask].flatten()

    if len(valid_counts) == 0:
        # Return empty histogram
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.text(0.5, 0.5, 'No valid pixels', ha='center', va='center', transform=ax.transAxes)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        result = buf[..., :3].astype(np.float32) / 255.0
        plt.close(fig)
        return result, {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0, 'outlier_pct': 0}

    # Compute statistics
    stats = {
        'min': int(valid_counts.min()),
        'max': int(valid_counts.max()),
        'mean': float(valid_counts.mean()),
        'median': float(np.median(valid_counts)),
        'std': float(valid_counts.std()),
        'outlier_pct': 100.0 * np.sum(valid_counts > max_display) / len(valid_counts)
    }

    # Create histogram
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

    # Clip values for histogram but track overflow
    clipped_counts = np.clip(valid_counts, 0, max_display)

    # Create bins
    bins = np.linspace(0, max_display, num_bins + 1)

    # Plot histogram
    n, bins_out, patches = ax.hist(clipped_counts, bins=bins, color='steelblue',
                                    edgecolor='black', alpha=0.7)

    # Color the last bin red if there are outliers
    if stats['outlier_pct'] > 0:
        patches[-1].set_facecolor('red')

    # Add vertical lines for statistics
    ax.axvline(stats['mean'], color='green', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.1f}")
    ax.axvline(stats['median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.1f}")

    # Add percentile lines
    p90 = np.percentile(valid_counts, 90)
    p99 = np.percentile(valid_counts, 99)
    ax.axvline(p90, color='yellow', linestyle=':', linewidth=1.5, label=f"P90: {p90:.0f}")
    ax.axvline(p99, color='red', linestyle=':', linewidth=1.5, label=f"P99: {p99:.0f}")

    ax.set_xlabel('Intersection Count', fontsize=10)
    ax.set_ylabel('Pixel Count', fontsize=10)
    ax.set_title(f'Distribution of Gaussian Intersections per Pixel\n'
                 f'Range: [{stats["min"]}, {stats["max"]}], Outliers (>{max_display}): {stats["outlier_pct"]:.2f}%',
                 fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, max_display)

    # Use log scale for y-axis if range is large
    if n.max() / (n[n > 0].min() + 1) > 100:
        ax.set_yscale('log')

    fig.tight_layout()
    fig.canvas.draw()

    # Get the RGBA buffer and convert to RGB
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    result = buf[..., :3].astype(np.float32) / 255.0

    plt.close(fig)

    return result, stats