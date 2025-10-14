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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    HWK = None
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

        HWK = None
        if cam_info.K is not None:
            K = cam_info.K.copy()
            K[:2] = K[:2] * scale
            HWK = (resolution[1], resolution[0], K)


    if len(cam_info.image.split()) > 3:
        # import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
        
        gt_image = gt_image * loaded_mask

        # if cam_info.image_name == '0026':
        #     print("!!!")
        #     print("loaded_mask: ", loaded_mask.shape)


    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
    
    # if cam_info.alpha is not None:
    #     loaded_mask = torch.from_numpy(cam_info.alpha).permute(2,0,1)
    

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, 
                  data_device=args.data_device,
                  HWK=HWK)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


# def get_center_and_ray(pose, intr, image_size):
#     """
#     Args:
#         pose (tensor [3,4]/[B,3,4]): Camera pose.
#         intr (tensor [3,3]/[B,3,3]): Camera intrinsics.
#         image_size (list of int): Image size.
#     Returns:
#         center_3D (tensor [HW,3]/[B,HW,3]): Center of the camera.
#         ray (tensor [HW,3]/[B,HW,3]): Ray of the camera with depth=1 (note: not unit ray).
#     """
#     H, W = image_size
#     # Given the intrinsic/extrinsic matrices, get the camera center and ray directions.
#     with torch.no_grad():
#         # Compute image coordinate grid.
#         y_range = torch.arange(H, dtype=torch.float32, device=pose.device).add_(0.5)
#         x_range = torch.arange(W, dtype=torch.float32, device=pose.device).add_(0.5)
#         Y, X = torch.meshgrid(y_range, x_range, indexing="ij")  # [H,W]
#         xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
#     # Compute center and ray.
#     if len(pose.shape) == 3:
#         batch_size = len(pose)
#         xy_grid = xy_grid.repeat(batch_size, 1, 1)  # [B,HW,2]
#     grid_3D = img2cam(to_hom(xy_grid), intr)  # [HW,3]/[B,HW,3]
#     center_3D = torch.zeros_like(grid_3D)  # [HW,3]/[B,HW,3]
#     # Transform from camera to world coordinates.
#     grid_3D = cam2world(grid_3D, pose)  # [HW,3]/[B,HW,3]
#     center_3D = cam2world(center_3D, pose)  # [HW,3]/[B,HW,3]
#     ray = grid_3D - center_3D  # [B,HW,3]
#     return center_3D, ray
