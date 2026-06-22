#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal
import os

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, rays = None, depth_path = None, HWK = None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_path = None,
                 clip_plane = None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        # Source image path (full). Used by --decomp to key its guided-filter cache.
        self.image_path = image_path
        # Populated by utils.image_decomp.load_gt_low_for_cameras when --decomp is set.
        # [3, H, W] float32 on data_device, matching original_image. None otherwise.
        self.gt_low = None

        self.depth_path = depth_path
        self.depth_map = None

        # --method clip_relight: per-frame world-space clip plane [a,b,c,d]
        # (kept half n·x+d<=0). None for non-clip datasets.
        self.clip_plane = (torch.as_tensor(clip_plane, dtype=torch.float32, device="cuda")
                           if clip_plane is not None else None)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        # `--data_device cpu`: pin host memory so per-iter `.cuda(non_blocking=True)`
        # can do a real async DMA transfer that overlaps with the prior iter's
        # backward / Adam step. Without pinning, the transfer is a synchronous
        # bounce-buffer copy and blocks the GPU. At 4K (192 MB / image FP32),
        # pinning cuts the perceived transfer cost from ~10-20 ms to ~0 ms.
        if self.data_device.type == "cpu":
            try:
                self.original_image = self.original_image.pin_memory()
            except RuntimeError:
                pass  # pin failed (rare; e.g. exhausted pinned-memory pool)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
            if self.data_device.type == "cpu":
                try:
                    self.gt_alpha_mask = self.gt_alpha_mask.pin_memory()
                except RuntimeError:
                    pass
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            # self.gt_alpha_mask = None
            self.gt_alpha_mask = torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        
        self.HWK = HWK

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.R = torch.tensor(self.R, dtype=torch.float32, device='cuda')
        self.T = torch.tensor(self.T, dtype=torch.float32, device='cuda')
    
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

