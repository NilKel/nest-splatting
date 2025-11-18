import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height

    K = depthmap.shape[0]

    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    # print(f'rays_o {rays_o.shape} rays_d {rays_d.shape} depth {depthmap.shape}')
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points, rays_d, rays_o

def cam2rays(view):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height

    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3].detach()
    return rays_d, rays_o

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap
    """
    points, _, _ = depths_to_points(view, depth)
    points = points.reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def depth_to_gradient_unnormalized(view, depth):
    """
    Compute unnormalized depth gradients (preserves magnitude for surface potential).
    Returns cross product of spatial gradients without normalization.

    view: view camera
    depth: depthmap
    """
    points, _, _ = depths_to_points(view, depth)
    points = points.reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    # Cross product WITHOUT normalization - preserves |∇f| magnitude
    gradient_map = torch.cross(dx, dy, dim=-1)
    output[1:-1, 1:-1, :] = gradient_map
    return output

def save_points(points, save_path):
    print(f'save points at: {save_path}, shape {points.shape}')
    obj_file = open(save_path, 'w')
    N = points.shape[0]
    for i in tqdm(range(N)):
        c = np.array([1., 0., 0.])
        obj_file.write("v {0} {1} {2} {3} {4} {5}\n".format \
            (points[i,0],points[i,1],points[i,2], c[0], c[1], c[2]))
    obj_file.close()