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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_H
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.env_map = None
        self.base_opacity = 0.0
        self._appearance_level = torch.empty(0)
        self.feat_gradient_accum = torch.empty(0)

        # Per-Gaussian features for "add" mode (added to hashgrid features)
        self._gaussian_features = torch.empty(0)

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._gaussian_features,
            self._appearance_level,  # Add appearance_level to checkpoint
        )
    
    def restore(self, model_args, training_args):
        # Handle both old format (13 elements) and new format (14 elements with _appearance_level)
        if len(model_args) == 14:
            # New format with _appearance_level
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self._gaussian_features,
            self._appearance_level) = model_args
        else:
            # Old format without _appearance_level - create it before training_setup()
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self._gaussian_features) = model_args
            # Create _appearance_level (needed by training_setup)
            init_level = 24
            ap_level = init_level * torch.ones((self._xyz.shape[0], 1), device="cuda").float()
            self._appearance_level = nn.Parameter(ap_level.requires_grad_(True))
            print(f"[Restore] Warning: Old checkpoint format - created _appearance_level with init_level={init_level}")
        
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

        # Load optimizer state (may not have gaussian_features if loading old checkpoint)
        try:
            self.optimizer.load_state_dict(opt_dict)
        except:
            # If loading fails (e.g., param_groups mismatch), load what we can
            # Load the state dict for existing parameters
            for i, param_group in enumerate(self.optimizer.param_groups):
                if i < len(opt_dict['param_groups']):
                    # Copy over the parameter group settings (lr, etc) but not params list
                    param_group['lr'] = opt_dict['param_groups'][i]['lr']
            # Load state for existing params
            for param_key in opt_dict['state'].keys():
                if param_key in [id(p) for pg in self.optimizer.param_groups for p in pg['params']]:
                    self.optimizer.state[param_key] = opt_dict['state'][param_key]

        # Ensure gaussian_features has proper optimizer state initialized
        # Find the param_group for gaussian_features and ensure it has state
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "gaussian_features":
                param = param_group['params'][0]
                # Initialize optimizer state for gaussian_features if not present
                if param not in self.optimizer.state:
                    # Adam optimizer state initialization
                    self.optimizer.state[param] = {
                        'step': torch.tensor(0.),
                        'exp_avg': torch.zeros_like(param),
                        'exp_avg_sq': torch.zeros_like(param)
                    }
                break

    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_appearance_level(self):
        return self._appearance_level

    @property
    def get_gaussian_features(self):
        return self._gaussian_features

    @property
    def get_envmap(self): #
        return self.env_map
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity) * (1.0 - self.base_opacity) + self.base_opacity
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def get_homotrans(self):
        rots = build_rotation(self._rotation)
        scales = self.get_scaling
        xyzs = self.get_xyz

        return build_H(rots, scales, xyzs)


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, args):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        opacities = self.inverse_opacity_activation(0.05 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # disable, set level MAX
        init_level = 24
        ap_level = init_level * torch.ones((self.get_xyz.shape[0], 1), device="cuda").float()
        self._appearance_level = nn.Parameter(ap_level.requires_grad_(True))

        # Initialize per-Gaussian features (for "add" and "cat" modes)
        # In "add" mode: gaussian_feat_dim = total_levels * per_level_dim = 6 * 4 = 24
        # In "cat" mode: gaussian_feat_dim = hybrid_levels * per_level_dim
        if hasattr(args, 'method') and args.method == "cat" and hasattr(args, 'hybrid_levels'):
            gaussian_feat_dim = args.hybrid_levels * 4  # per_level_dim = 4
        else:
            gaussian_feat_dim = 24  # 6 levels * 4 dim (default)
        
        # Create Gaussian features parameter (even if 0D for hybrid_levels=0, for consistency)
        # PyTorch handles 0D parameters gracefully
        gaussian_feats = torch.zeros((self.get_xyz.shape[0], gaussian_feat_dim), device="cuda").float()
        self._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.feat_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._appearance_level], 'lr': 0, "name": "ap_level"},
            {'params': [self._gaussian_features], 'lr': training_args.feature_lr, "name": "gaussian_features"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, save_gaussian_features=False):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if self.env_map is not None:
            save_path = path.replace('.ply', '.map')
            torch.save(self.env_map.state_dict(), save_path)
        
        # Save per-Gaussian features separately (for add/cat modes)
        if save_gaussian_features and hasattr(self, '_gaussian_features') and self._gaussian_features.numel() > 0:
            features_path = path.replace('.ply', '_gaussian_features.pth')
            torch.save({
                'gaussian_features': self._gaussian_features.detach().cpu(),
                'shape': self._gaussian_features.shape,
            }, features_path)
            print(f"[Checkpoint] Saved per-Gaussian features ({self._gaussian_features.shape}) to {features_path}")
           
    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.05))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, args = None):
        print(f'load ply file from {path}')
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # Load per-Gaussian features if available (for add/cat modes)
        features_path = path.replace('.ply', '_gaussian_features.pth')
        if os.path.exists(features_path):
            checkpoint = torch.load(features_path, map_location='cpu')
            gaussian_features = checkpoint['gaussian_features'].cuda()
            self._gaussian_features = nn.Parameter(gaussian_features.requires_grad_(True))
            print(f"[Checkpoint] Loaded per-Gaussian features {gaussian_features.shape} from {features_path}")
        elif args and hasattr(args, 'method') and args.method in ['add', 'cat']:
            # Initialize with zeros if not found but method requires it
            if hasattr(args, 'method') and args.method == 'cat' and hasattr(args, 'hybrid_levels'):
                gaussian_feat_dim = args.hybrid_levels * 4  # 4 = level_dim
            else:
                gaussian_feat_dim = 24  # Default: 6 levels * 4 dim
            self._gaussian_features = nn.Parameter(
                torch.zeros((xyz.shape[0], gaussian_feat_dim), device='cuda').requires_grad_(True)
            )
            print(f"[Checkpoint] Initialized per-Gaussian features to zeros ({gaussian_feat_dim}D) - features not found in checkpoint")
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        init_level = 6
        ap_level = init_level * torch.ones((self.get_xyz.shape[0], 1), device="cuda").float()
        self._appearance_level = nn.Parameter(ap_level.requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "mlp" or group["name"] == "env": continue

            # Special handling for empty tensors (e.g., gaussian_features with 0 dimensions)
            param = group["params"][0]
            if param.shape[1] == 0:  # Empty feature dimension
                # For tensors with shape [N, 0], just update the first dimension
                new_param = torch.zeros((mask.sum().item(), 0), device=param.device, dtype=param.dtype)
                group["params"][0] = nn.Parameter(new_param.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
                # Clear optimizer state for this param
                if param in self.optimizer.state:
                    del self.optimizer.state[param]
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._appearance_level = optimizable_tensors["ap_level"]
        if "gaussian_features" in optimizable_tensors:
            self._gaussian_features = optimizable_tensors["gaussian_features"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.feat_gradient_accum = self.feat_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "mlp" or group["name"] == "env": continue
            # Skip param groups that aren't being updated (e.g., gaussian_features when hybrid_levels=0)
            if group["name"] not in tensors_dict:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_ap_level, new_gaussian_features=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "ap_level" : new_ap_level}

        # Add gaussian_features if available
        if new_gaussian_features is not None:
            d["gaussian_features"] = new_gaussian_features

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._appearance_level = optimizable_tensors["ap_level"]
        if "gaussian_features" in optimizable_tensors:
            self._gaussian_features = optimizable_tensors["gaussian_features"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.feat_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        new_ap_level = self._appearance_level[selected_pts_mask].repeat(N,1)

        # Clone gaussian features for split gaussians
        if self._gaussian_features.numel() > 0:
            new_gaussian_features = self._gaussian_features[selected_pts_mask].repeat(N,1)
        else:
            new_gaussian_features = None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_ap_level, new_gaussian_features)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_ap_level = self._appearance_level[selected_pts_mask]

        # Clone gaussian features for cloned gaussians
        if self._gaussian_features.numel() > 0:
            new_gaussian_features = self._gaussian_features[selected_pts_mask]
        else:
            new_gaussian_features = None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_ap_level, new_gaussian_features)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, ap_update, act_level, densify_tag = True, prune_tag = True, iteration=None):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        appearance_grads = self.feat_gradient_accum / self.denom
        appearance_grads[appearance_grads.isnan()] = 0.0

        if ap_update > 0 and act_level is not None:
            update_mask = torch.where(appearance_grads >= ap_update, True, False)
            self._appearance_level[update_mask] = torch.clamp(self._appearance_level[update_mask] + 1, max=6)

        if densify_tag == False:
            grads = grads * 0

        num_before = self.get_xyz.shape[0]

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        num_after_densify = self.get_xyz.shape[0]

        # prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_mask = (self.get_opacity < min_opacity * (1.0 - self.base_opacity) + self.base_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            # big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            big_points_ws = (self.get_scaling.max(dim=1).values > self.get_scaling.mean() * 10.)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        num_to_prune = prune_mask.sum().item()

        if prune_tag:
            self.prune_points(prune_mask)

        num_after_prune = self.get_xyz.shape[0]

        # Debug logging every 500 iterations
        if iteration is not None and iteration % 500 == 0:
            print(f"[Densify] Iter {iteration}: {num_before} -> +{num_after_densify - num_before} densify -> {num_after_densify} -> -{num_to_prune if prune_tag else 0} prune -> {num_after_prune} (prune_tag={prune_tag})")

        torch.cuda.empty_cache()

        assert(self._xyz.shape[0] == self._appearance_level.shape[0])

    def add_densification_stats(self, viewspace_point_tensor, update_filter, pixels = None):

        if pixels is not None:
            # print(viewspace_point_tensor.grad.shape, pixels.shape, torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True).shape, update_filter.shape)
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True) * pixels[update_filter]#.unsqueeze(-1)
            self.denom[update_filter] += pixels[update_filter]
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
            self.denom[update_filter] += 1
