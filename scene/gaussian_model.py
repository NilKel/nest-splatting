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
        
        # Per-Gaussian features for cat mode
        self._gaussian_features = torch.empty(0)
        self._gaussian_feat_dim = 0  # Will be set in create_from_pcd
        
        # Adaptive mode parameters
        self._gamma = torch.empty(0)  # (N, 1) learnable blend parameter
        self._adaptive_features = torch.empty(0)  # (N, total_levels * per_level_dim)
        self._adaptive_feat_dim = 0
        self._adaptive_num_levels = 0
        self.temperature = 1.0
        self.min_temperature = 0.01

        # Adaptive_zero mode weight (per-Gaussian, controls hash vs zeros for fine levels)
        self._adaptive_zero_weight = torch.empty(0)  # (N, 1) learnable weight

        # Adaptive_gate mode: gate logits for binary hash selection
        # Gumbel-STE always binary: sigmoid(logit) > 0.5 → use hash, else zeros
        self._gate_logits = torch.empty(0)  # (N, 1) gate logits

        # Relocation mode for adaptive weights: 'clone' (copy from source) or 'reset' (initialize to 0)
        self._relocation_mode = "clone"

        # Frozen beta shape value (raw _shape value) - if not None, shape is frozen and this value is used for new Gaussians
        self._frozen_beta_raw = None

        # Beta/General kernel shape parameter (per-Gaussian, controls kernel falloff)
        # Beta kernel: sigmoid(_shape) * 4.0 + 0.001 gives range [0.001, 4.001]
        #   shape≈0 = hard flat disk, shape≈4 = soft Gaussian cloud
        # General kernel: sigmoid(_shape) * 6.0 + 2.0 gives range [2.0, 8.0]
        #   beta=2.0 = standard Gaussian, beta=8.0 = super-Gaussian (box)
        self._shape = torch.empty(0)
        self.kernel_type = "gaussian"  # "gaussian", "beta", "flex", or "general"

        # Flex kernel: per-Gaussian learnable beta for Gaussian sharpening
        # softplus(_flex_beta) gives range [0, inf), typically [0, ~50]
        # beta=0 = standard Gaussian, beta>0 = sharper/more opaque
        self._flex_beta = torch.empty(0)

        # Diffuse mode flag (uses SH degree 0, no hashgrid)
        self._diffuse_mode = False
        # Specular mode flag (full 2DGS with SH, no hashgrid)
        self._specular_mode = False
        # Diffuse+NGP mode flag (diffuse SH + hashgrid on unprojected depth)
        self._diffuse_ngp_mode = False
        # Diffuse+Offset mode flag (diffuse SH as xyz offset for hashgrid query)
        self._diffuse_offset_mode = False

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
            self._gaussian_feat_dim,
            self._gamma,
            self._adaptive_features,
            self._adaptive_feat_dim,
            self._adaptive_num_levels,
            self.temperature,
            self._adaptive_cat_weight if hasattr(self, '_adaptive_cat_weight') else torch.empty(0, device="cuda"),
        )
    
    def restore(self, model_args, training_args):
        # Handle multiple checkpoint formats
        if len(model_args) == 20:
            # Format with adaptive_cat mode
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
            self._gaussian_feat_dim,
            self._gamma,
            self._adaptive_features,
            self._adaptive_feat_dim,
            self._adaptive_num_levels,
            self.temperature,
            self._adaptive_cat_weight) = model_args
        elif len(model_args) == 19:
            # Format with adaptive mode (no adaptive_cat)
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
            self._gaussian_feat_dim,
            self._gamma,
            self._adaptive_features,
            self._adaptive_feat_dim,
            self._adaptive_num_levels,
            self.temperature) = model_args
            self._adaptive_cat_weight = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
        elif len(model_args) == 14:
            # Cat mode format
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
            self._gaussian_feat_dim) = model_args
            self._gamma = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._adaptive_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._adaptive_feat_dim = 0
            self._adaptive_num_levels = 0
            self._adaptive_cat_weight = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
        else:
            # Old format without gaussian_features
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
            self.spatial_lr_scale) = model_args
            self._gaussian_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._gaussian_feat_dim = 0
            self._gamma = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._adaptive_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._adaptive_feat_dim = 0
            self._adaptive_num_levels = 0
            self._adaptive_cat_weight = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
        
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    
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
    def get_gamma(self):
        return self._gamma
    
    @property
    def get_adaptive_features(self):
        return self._adaptive_features
    
    @property
    def get_envmap(self): # 
        return self.env_map
    
    @property
    def get_opacity(self):
        op = self.opacity_activation(self._opacity) * (1.0 - self.base_opacity) + self.base_opacity
        # Energy normalization for beta kernel: boost opacity as shape increases (softer kernels)
        # This preserves brightness as kernels harden (shape → 0)
        # if self.kernel_type == "beta" and self._shape.numel() > 0:
        #     return op * (1.0 + self.get_shape)
        return op

    @property
    def get_shape(self):
        """Returns shape parameter for beta or general kernel.

        Beta kernel: range [0.5, 4.0]
            shape = 0.5: hard flat disk (minimum to prevent gradient collapse)
            shape = 4.0: soft Gaussian cloud

        General kernel (Isotropic Generalized Gaussian): range [2.0, 8.0]
            beta = 2.0: standard Gaussian
            beta = 8.0: super-Gaussian (flat top, steep edges)
        """
        if self.kernel_type == "general":
            # Sigmoid * 6.0 + 2.0 maps (-inf, inf) -> (2.0, 8.0)
            return torch.sigmoid(self._shape) * 6.0 + 2.0
        else:
            # Beta kernel activation
            # β = sigmoid(_shape) * 5.0, range [0, 5]
            # β=0: flat disk, β~4: Gaussian-like, β=5: sharper
            return torch.sigmoid(self._shape) * 5.0

    @property
    def get_flex_beta(self):
        """Returns per-Gaussian beta for flex kernel.
        Uses softplus to ensure non-negative values [0, inf).
        beta = 0: standard Gaussian
        beta > 0: sharper, more opaque Gaussian
        """
        return torch.nn.functional.softplus(self._flex_beta)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def get_homotrans(self):
        rots = build_rotation(self._rotation)
        scales = self.get_scaling
        xyzs = self.get_xyz

        return build_H(rots, scales, xyzs)

    def update_temperature(self, iteration, max_iter):
        """Exponential decay of temperature for adaptive mode."""
        if max_iter > 0:
            ratio = iteration / max_iter
            self.temperature = 1.0 * (self.min_temperature / 1.0) ** ratio
        return self.temperature

    def get_adaptive_mask(self, level_dim):
        """
        Compute soft mask for adaptive feature blending.
        mask[i] = 1 means use per-Gaussian, mask[i] = 0 means use hashgrid.
        """
        if self._adaptive_num_levels == 0:
            return None
        
        N = self._gamma.shape[0]
        num_levels = self._adaptive_num_levels
        
        # Create level indices: [0, 1, ..., num_levels-1]
        level_indices = torch.arange(num_levels, device=self._gamma.device, dtype=self._gamma.dtype)
        
        # Compute per-level mask: sigmoid((gamma - level_idx) / temperature)
        mask_per_level = torch.sigmoid((self._gamma - level_indices) / self.temperature)
        
        # Expand mask to full feature dimension
        mask_expanded = mask_per_level.repeat_interleave(level_dim, dim=1)
        
        return mask_expanded

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
        
        # Initialize per-Gaussian features for cat mode
        # Dimension = hybrid_levels * per_level_dim (default: 3 * 4 = 12)
        if hasattr(args, 'method') and args.method == "cat" and hasattr(args, 'hybrid_levels'):
            per_level_dim = 4  # From config encoding.hashgrid.dim
            self._gaussian_feat_dim = args.hybrid_levels * per_level_dim
        else:
            self._gaussian_feat_dim = 0
        
        if self._gaussian_feat_dim > 0:
            gaussian_feats = torch.zeros((self.get_xyz.shape[0], self._gaussian_feat_dim), device="cuda").float()
            self._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))
        else:
            self._gaussian_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
        
        # Initialize adaptive mode parameters
        if hasattr(args, 'method') and args.method == "adaptive":
            per_level_dim = 4
            num_levels = getattr(args, 'adaptive_levels', 6)
            self._adaptive_feat_dim = num_levels * per_level_dim
            self._adaptive_num_levels = num_levels
            
            # Initialize gamma to -1.0 (favors hashgrid initially)
            gamma_init = -1.0 * torch.ones((self.get_xyz.shape[0], 1), device="cuda").float()
            self._gamma = nn.Parameter(gamma_init.requires_grad_(True))
            
            # Initialize adaptive features to small random values
            adaptive_feats = torch.randn((self.get_xyz.shape[0], self._adaptive_feat_dim), device="cuda").float() * 0.01
            self._adaptive_features = nn.Parameter(adaptive_feats.requires_grad_(True))
        elif hasattr(args, 'method') and args.method == "adaptive_add":
            # adaptive_add mode: per-Gaussian features + weight for weighted blending
            per_level_dim = 4
            num_levels = getattr(args, 'adaptive_levels', 6)
            self._adaptive_feat_dim = num_levels * per_level_dim
            self._adaptive_num_levels = num_levels
            
            # Initialize gamma (blend weight) to 0.0 (sigmoid(0) = 0.5, equal blend)
            gamma_init = torch.zeros((self.get_xyz.shape[0], 1), device="cuda").float()
            self._gamma = nn.Parameter(gamma_init.requires_grad_(True))
            
            # Initialize adaptive features to small random values
            adaptive_feats = torch.randn((self.get_xyz.shape[0], self._adaptive_feat_dim), device="cuda").float() * 0.01
            self._adaptive_features = nn.Parameter(adaptive_feats.requires_grad_(True))
        else:
            self._adaptive_feat_dim = 0
            self._adaptive_num_levels = 0
            self._gamma = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._adaptive_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
        
        # Initialize adaptive_cat mode parameters (similar to cat, but with blend weight)
        if hasattr(args, 'method') and args.method == "adaptive_cat":
            per_level_dim = 4  # From config encoding.hashgrid.dim
            num_levels = 6  # Total levels from config (will be overridden by warmup if loaded)
            self._gaussian_feat_dim = num_levels * per_level_dim  # 24D
            
            # Initialize per-Gaussian features to small random values
            gaussian_feats = torch.randn((self.get_xyz.shape[0], self._gaussian_feat_dim), device="cuda").float() * 0.01
            self._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))
            
            # Initialize blend weight to 0.0 (sigmoid(0) = 0.5, equal blend initially)
            blend_weight = torch.zeros((self.get_xyz.shape[0], 1), device="cuda").float()
            self._adaptive_cat_weight = nn.Parameter(blend_weight.requires_grad_(True))
        else:
            self._adaptive_cat_weight = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
        
        # Diffuse mode flag (uses SH degree 0, no hashgrid)
        self._diffuse_mode = hasattr(args, 'method') and args.method == "diffuse"

        # Initialize beta kernel shape parameter if using beta kernel
        if hasattr(args, 'kernel') and args.kernel in ["beta", "beta_scaled"]:
            self.kernel_type = args.kernel  # "beta" or "beta_scaled"
            # β = sigmoid(_shape) * 5.0, init to β≈3 (semisoft)
            # sigmoid(0.405) ≈ 0.6, so β ≈ 3
            shape_init = 0.405 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            self._shape = nn.Parameter(shape_init.requires_grad_(True))
            print(f"[DEBUG] {args.kernel} kernel shape initialized: requires_grad={self._shape.requires_grad}")
        elif hasattr(args, 'kernel') and args.kernel == "flex":
            self.kernel_type = "flex"
            self._shape = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
        else:
            self.kernel_type = "gaussian"
            self._shape = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.feat_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # For diffuse mode, don't train features_rest (SH degree 0 only)
        f_rest_lr = 0.0 if self._diffuse_mode else training_args.feature_lr / 20.0
        
        # For diffuse_offset mode, reduce learning rates for stability
        f_dc_lr = training_args.feature_lr
        xyz_lr_scale = 1.0
        if self._diffuse_offset_mode:
            f_dc_lr = training_args.feature_lr * 0.01  # 100x smaller for offset stability
            xyz_lr_scale = 0.1  # 10x smaller for position stability (reduce overpruning)
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale * xyz_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': f_dc_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': f_rest_lr, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._appearance_level], 'lr': 0, "name": "ap_level"},
        ]
        
        # Add per-Gaussian features for cat mode (if present)
        if self._gaussian_feat_dim > 0:
            l.append({'params': [self._gaussian_features], 'lr': training_args.feature_lr, "name": "gaussian_features"})
        
        # Add adaptive mode parameters (if present)
        if self._adaptive_feat_dim > 0:
            l.append({'params': [self._gamma], 'lr': training_args.opacity_lr, "name": "gamma"})
            l.append({'params': [self._adaptive_features], 'lr': training_args.feature_lr, "name": "adaptive_features"})
        
        # Add adaptive_cat blend weight (if present)
        if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0:
            l.append({'params': [self._adaptive_cat_weight], 'lr': training_args.opacity_lr, "name": "adaptive_cat_weight"})

        # Add adaptive_zero weight (if present)
        if hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0:
            l.append({'params': [self._adaptive_zero_weight], 'lr': training_args.opacity_lr, "name": "adaptive_zero_weight"})

        # Add adaptive_gate parameter (if present)
        if hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0:
            l.append({'params': [self._gate_logits], 'lr': training_args.opacity_lr, "name": "gate_logits"})

        # Add beta kernel shape parameter (if present and requires grad)
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            print(f"[DEBUG] training_setup: _shape.requires_grad={self._shape.requires_grad}")
            if self._shape.requires_grad:
                l.append({'params': [self._shape], 'lr': 0.001, "name": "shape"})

        # Add flex kernel per-Gaussian beta parameter (if present)
        if hasattr(self, '_flex_beta') and self._flex_beta.numel() > 0:
            l.append({'params': [self._flex_beta], 'lr': training_args.opacity_lr, "name": "flex_beta"})
            print(f"[DEBUG] Added flex_beta to optimizer: shape={self._flex_beta.shape}, lr={training_args.opacity_lr}")
        else:
            print(f"[DEBUG] flex_beta NOT added: hasattr={hasattr(self, '_flex_beta')}, numel={self._flex_beta.numel() if hasattr(self, '_flex_beta') else 'N/A'}")

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale*xyz_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale*xyz_lr_scale,
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
        # Add per-Gaussian features for cat mode
        if self._gaussian_feat_dim > 0:
            for i in range(self._gaussian_feat_dim):
                l.append('gf_{}'.format(i))
        # Add beta kernel shape parameter
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            l.append('shape')
        return l

    def save_ply(self, path):
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
        
        # Build list of attributes to save
        attr_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]

        # Include gaussian_features if present (cat mode)
        if self._gaussian_feat_dim > 0:
            gaussian_feats = self._gaussian_features.detach().cpu().numpy()
            attr_list.append(gaussian_feats)

        # Include shape parameter if present (beta kernel)
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            shapes = self._shape.detach().cpu().numpy()
            attr_list.append(shapes)

        attributes = np.concatenate(attr_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if self.env_map is not None:
            save_path = path.replace('.ply', '.map')
            torch.save(self.env_map.state_dict(), save_path)
           
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
        
        # Load per-Gaussian features for cat mode (if present in PLY and args specify cat mode)
        gf_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("gf_")]
        if len(gf_names) > 0:
            gf_names = sorted(gf_names, key = lambda x: int(x.split('_')[-1]))
            gaussian_feats = np.zeros((xyz.shape[0], len(gf_names)))
            for idx, attr_name in enumerate(gf_names):
                gaussian_feats[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self._gaussian_feat_dim = len(gf_names)
            self._gaussian_features = nn.Parameter(torch.tensor(gaussian_feats, dtype=torch.float, device="cuda").requires_grad_(True))
            print(f"Loaded {self._gaussian_feat_dim}D per-Gaussian features for cat mode")
        elif args is not None and hasattr(args, 'method') and args.method == "cat" and hasattr(args, 'hybrid_levels'):
            # Cat mode but no features in PLY - initialize them (for old checkpoints)
            per_level_dim = 4
            self._gaussian_feat_dim = args.hybrid_levels * per_level_dim
            gaussian_feats = torch.zeros((xyz.shape[0], self._gaussian_feat_dim), device="cuda").float()
            self._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))
            print(f"Warning: No per-Gaussian features in PLY, initialized {self._gaussian_feat_dim}D zeros for cat mode")
        else:
            self._gaussian_feat_dim = 0
            self._gaussian_features = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))

        init_level = 6
        ap_level = init_level * torch.ones((self.get_xyz.shape[0], 1), device="cuda").float()
        self._appearance_level = nn.Parameter(ap_level.requires_grad_(True))

        # Load beta kernel shape parameter (if present in PLY)
        if "shape" in [p.name for p in plydata.elements[0].properties]:
            shapes = np.asarray(plydata.elements[0]["shape"])[..., np.newaxis]
            self._shape = nn.Parameter(torch.tensor(shapes, dtype=torch.float, device="cuda").requires_grad_(True))
            # Determine kernel type from args if available, otherwise default to "beta"
            if args is not None and hasattr(args, 'kernel') and args.kernel == "beta_scaled":
                self.kernel_type = "beta_scaled"
            else:
                self.kernel_type = "beta"
            print(f"Loaded {self.kernel_type} kernel shape parameter")
        elif args is not None and hasattr(args, 'kernel') and args.kernel in ["beta", "beta_scaled"]:
            # Beta kernel requested but no shape in PLY - init to β≈3 (semisoft)
            # sigmoid(0.405) ≈ 0.6, so β ≈ 3
            shape_init = 0.405 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda")
            self._shape = nn.Parameter(shape_init.requires_grad_(True))
            self.kernel_type = args.kernel
            print(f"Warning: No shape in PLY, initialized {args.kernel} kernel to β≈3 (semisoft)")
        elif args is not None and hasattr(args, 'kernel') and args.kernel == "flex":
            # Flex kernel requested
            self._shape = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self.kernel_type = "flex"
        else:
            self._shape = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self.kernel_type = "gaussian"

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
        if "gamma" in optimizable_tensors:
            self._gamma = optimizable_tensors["gamma"]
        if "adaptive_features" in optimizable_tensors:
            self._adaptive_features = optimizable_tensors["adaptive_features"]
        if "adaptive_cat_weight" in optimizable_tensors:
            self._adaptive_cat_weight = optimizable_tensors["adaptive_cat_weight"]
        if "adaptive_zero_weight" in optimizable_tensors:
            self._adaptive_zero_weight = optimizable_tensors["adaptive_zero_weight"]
        if "gate_logits" in optimizable_tensors:
            self._gate_logits = optimizable_tensors["gate_logits"]
        if "shape" in optimizable_tensors:
            self._shape = optimizable_tensors["shape"]
        elif hasattr(self, '_shape') and self._shape.numel() > 0:
            # Handle frozen shape parameter (not in optimizer) - prune manually
            self._shape = nn.Parameter(self._shape.data[valid_points_mask].clone(), requires_grad=False)
        if "flex_beta" in optimizable_tensors:
            self._flex_beta = optimizable_tensors["flex_beta"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.feat_gradient_accum = self.feat_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        # Debug: verify shape tensor size matches xyz after pruning
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            xyz_size = self.get_xyz.shape[0]
            shape_size = self._shape.shape[0]
            if xyz_size != shape_size:
                print(f"[ERROR] Size mismatch after prune_points: xyz={xyz_size}, shape={shape_size}")
                raise RuntimeError(f"Shape tensor size mismatch after prune: xyz={xyz_size}, shape={shape_size}")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "mlp" or group["name"] == "env": continue
            assert len(group["params"]) == 1
            
            # Skip param groups that aren't in the tensors_dict
            if group["name"] not in tensors_dict:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
                
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_ap_level, new_gaussian_features=None, new_gamma=None, new_adaptive_features=None, new_adaptive_cat_weight=None, new_adaptive_zero_weight=None, new_gate_logits=None, new_shape=None, new_flex_beta=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "ap_level" : new_ap_level}

        # Add gaussian_features if present (cat mode)
        if new_gaussian_features is not None and self._gaussian_feat_dim > 0:
            d["gaussian_features"] = new_gaussian_features

        # Add adaptive mode parameters
        if new_gamma is not None and self._adaptive_feat_dim > 0:
            d["gamma"] = new_gamma
        if new_adaptive_features is not None and self._adaptive_feat_dim > 0:
            d["adaptive_features"] = new_adaptive_features

        # Add adaptive_cat blend weight
        if new_adaptive_cat_weight is not None and hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0:
            d["adaptive_cat_weight"] = new_adaptive_cat_weight

        # Add adaptive_zero blend weight
        if new_adaptive_zero_weight is not None and hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0:
            d["adaptive_zero_weight"] = new_adaptive_zero_weight

        # Add adaptive_gate parameter
        if new_gate_logits is not None and hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0:
            d["gate_logits"] = new_gate_logits

        # Add beta kernel shape parameter
        if new_shape is not None and hasattr(self, '_shape') and self._shape.numel() > 0:
            d["shape"] = new_shape

        # Add flex kernel per-Gaussian beta parameter
        if new_flex_beta is not None and hasattr(self, '_flex_beta') and self._flex_beta.numel() > 0:
            d["flex_beta"] = new_flex_beta

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
        if "gamma" in optimizable_tensors:
            self._gamma = optimizable_tensors["gamma"]
        if "adaptive_features" in optimizable_tensors:
            self._adaptive_features = optimizable_tensors["adaptive_features"]
        if "adaptive_cat_weight" in optimizable_tensors:
            self._adaptive_cat_weight = optimizable_tensors["adaptive_cat_weight"]
        if "adaptive_zero_weight" in optimizable_tensors:
            self._adaptive_zero_weight = optimizable_tensors["adaptive_zero_weight"]
        if "gate_logits" in optimizable_tensors:
            self._gate_logits = optimizable_tensors["gate_logits"]
        if "shape" in optimizable_tensors:
            self._shape = optimizable_tensors["shape"]
        elif new_shape is not None and hasattr(self, '_shape') and self._shape.numel() > 0:
            # Handle frozen shape parameter (not in optimizer) - manually concatenate
            self._shape = nn.Parameter(torch.cat([self._shape.data, new_shape], dim=0), requires_grad=False)
        if "flex_beta" in optimizable_tensors:
            self._flex_beta = optimizable_tensors["flex_beta"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.feat_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Debug: verify shape tensor size matches xyz
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            xyz_size = self.get_xyz.shape[0]
            shape_size = self._shape.shape[0]
            if xyz_size != shape_size:
                print(f"[ERROR] Size mismatch after densification_postfix: xyz={xyz_size}, shape={shape_size}")
                raise RuntimeError(f"Shape tensor size mismatch: xyz={xyz_size}, shape={shape_size}")

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
        
        # Handle gaussian_features for cat mode
        new_gaussian_features = None
        if self._gaussian_feat_dim > 0:
            new_gaussian_features = self._gaussian_features[selected_pts_mask].repeat(N,1)
        
        # Handle adaptive mode parameters
        new_gamma = None
        new_adaptive_features = None
        if self._adaptive_feat_dim > 0:
            new_gamma = self._gamma[selected_pts_mask].repeat(N,1)
            new_adaptive_features = self._adaptive_features[selected_pts_mask].repeat(N,1)
        
        # Handle adaptive_cat blend weight
        new_adaptive_cat_weight = None
        if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0:
            if self._relocation_mode == "reset":
                num_new = selected_pts_mask.sum().item() * N
                new_adaptive_cat_weight = torch.zeros((num_new, 1), device="cuda")
            else:  # clone
                new_adaptive_cat_weight = self._adaptive_cat_weight[selected_pts_mask].repeat(N, 1)

        # Handle adaptive_zero blend weight
        new_adaptive_zero_weight = None
        if hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0:
            if self._relocation_mode == "reset":
                num_new = selected_pts_mask.sum().item() * N
                new_adaptive_zero_weight = torch.zeros((num_new, 1), device="cuda")
            else:  # clone
                new_adaptive_zero_weight = self._adaptive_zero_weight[selected_pts_mask].repeat(N, 1)

        # Handle adaptive_gate parameter
        new_gate_logits = None
        if hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0:
            if self._relocation_mode == "reset":
                num_new = selected_pts_mask.sum().item() * N
                new_gate_logits = torch.zeros((num_new, 1), device="cuda")
            else:  # clone
                new_gate_logits = self._gate_logits[selected_pts_mask].repeat(N, 1)

        # Handle beta kernel shape parameter
        new_shape = None
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            if self._frozen_beta_raw is not None:
                num_new = selected_pts_mask.sum().item() * N
                new_shape = torch.full((num_new, 1), self._frozen_beta_raw, device="cuda")
            else:
                new_shape = self._shape[selected_pts_mask].repeat(N, 1)

        # Handle flex kernel per-Gaussian beta parameter
        new_flex_beta = None
        if hasattr(self, '_flex_beta') and self._flex_beta.numel() > 0:
            new_flex_beta = self._flex_beta[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_ap_level, new_gaussian_features, new_gamma, new_adaptive_features, new_adaptive_cat_weight, new_adaptive_zero_weight, new_gate_logits, new_shape, new_flex_beta)

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
        
        # Handle gaussian_features for cat mode
        new_gaussian_features = None
        if self._gaussian_feat_dim > 0:
            new_gaussian_features = self._gaussian_features[selected_pts_mask]
        
        # Handle adaptive mode parameters
        new_gamma = None
        new_adaptive_features = None
        if self._adaptive_feat_dim > 0:
            new_gamma = self._gamma[selected_pts_mask]
            new_adaptive_features = self._adaptive_features[selected_pts_mask]
        
        # Handle adaptive_cat blend weight
        new_adaptive_cat_weight = None
        if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0:
            if self._relocation_mode == "reset":
                num_new = selected_pts_mask.sum().item()
                new_adaptive_cat_weight = torch.zeros((num_new, 1), device="cuda")
            else:  # clone
                new_adaptive_cat_weight = self._adaptive_cat_weight[selected_pts_mask]

        # Handle adaptive_zero blend weight
        new_adaptive_zero_weight = None
        if hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0:
            if self._relocation_mode == "reset":
                num_new = selected_pts_mask.sum().item()
                new_adaptive_zero_weight = torch.zeros((num_new, 1), device="cuda")
            else:  # clone
                new_adaptive_zero_weight = self._adaptive_zero_weight[selected_pts_mask]

        # Handle adaptive_gate parameter
        new_gate_logits = None
        if hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0:
            if self._relocation_mode == "reset":
                num_new = selected_pts_mask.sum().item()
                new_gate_logits = torch.zeros((num_new, 1), device="cuda")
            else:  # clone
                new_gate_logits = self._gate_logits[selected_pts_mask]

        # Handle beta kernel shape parameter
        new_shape = None
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            if self._frozen_beta_raw is not None:
                num_new = selected_pts_mask.sum().item()
                new_shape = torch.full((num_new, 1), self._frozen_beta_raw, device="cuda")
            else:
                new_shape = self._shape[selected_pts_mask]

        # Handle flex kernel per-Gaussian beta parameter
        new_flex_beta = None
        if hasattr(self, '_flex_beta') and self._flex_beta.numel() > 0:
            new_flex_beta = self._flex_beta[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_ap_level, new_gaussian_features, new_gamma, new_adaptive_features, new_adaptive_cat_weight, new_adaptive_zero_weight, new_gate_logits, new_shape, new_flex_beta)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, ap_update, act_level, densify_tag = True, prune_tag = True):
        
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        appearance_grads = self.feat_gradient_accum / self.denom
        appearance_grads[appearance_grads.isnan()] = 0.0

        if ap_update > 0 and act_level is not None:
            update_mask = torch.where(appearance_grads >= ap_update, True, False)
            self._appearance_level[update_mask] = torch.clamp(self._appearance_level[update_mask] + 1, max=6)

        if densify_tag == False:
            grads = grads * 0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        
        # prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_mask = (self.get_opacity < min_opacity * (1.0 - self.base_opacity) + self.base_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            # big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            big_points_ws = (self.get_scaling.max(dim=1).values > self.get_scaling.mean() * 10.)
            
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if prune_tag:
            self.prune_points(prune_mask)
        
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

    # ==================== MCMC Methods ====================
    # Based on "3D Gaussian Splatting as Markov Chain Monte Carlo"
    
    def _mcmc_update_params(self, idxs, ratio):
        """
        Compute new opacity and scale for relocated Gaussians using the MCMC relocation kernel.
        
        Args:
            idxs: Indices of Gaussians to update
            ratio: [N, 1] tensor of relocation ratios (how many children each Gaussian produces)
        
        Returns:
            Tuple of (xyz, features_dc, features_rest, opacity, scaling, rotation, ap_level, gaussian_features, gamma, adaptive_features)
        """
        from utils.reloc_utils import compute_relocation_cuda
        
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )
        
        # Clamp opacity to valid range and convert back to logit space
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        
        # Convert scale back to log space
        new_scaling = self.scaling_inverse_activation(new_scaling)
        
        # Handle gaussian_features for cat mode
        gaussian_features = None
        if self._gaussian_feat_dim > 0:
            gaussian_features = self._gaussian_features[idxs]
        
        # Handle adaptive mode parameters
        gamma = None
        adaptive_features = None
        if self._adaptive_feat_dim > 0:
            gamma = self._gamma[idxs]
            adaptive_features = self._adaptive_features[idxs]
        
        # Handle adaptive_cat blend weight
        adaptive_cat_weight = None
        if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0:
            if self._relocation_mode == "reset":
                adaptive_cat_weight = torch.zeros((len(idxs), 1), device="cuda")
            else:  # clone
                adaptive_cat_weight = self._adaptive_cat_weight[idxs]

        # Handle adaptive_zero weight
        adaptive_zero_weight = None
        if hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0:
            if self._relocation_mode == "reset":
                adaptive_zero_weight = torch.zeros((len(idxs), 1), device="cuda")
            else:  # clone
                adaptive_zero_weight = self._adaptive_zero_weight[idxs]

        # Handle adaptive_gate parameter
        gate_logits = None
        if hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0:
            if self._relocation_mode == "reset":
                gate_logits = torch.zeros((len(idxs), 1), device="cuda")
            else:  # clone
                gate_logits = self._gate_logits[idxs]

        # Handle beta kernel shape parameter
        shape = None
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            if self._frozen_beta_raw is not None:
                # Use frozen value for new Gaussians
                shape = torch.full((len(idxs), 1), self._frozen_beta_raw, device="cuda")
            else:
                # Clone from source
                shape = self._shape[idxs]

        return (
            self._xyz[idxs],
            self._features_dc[idxs],
            self._features_rest[idxs],
            new_opacity,
            new_scaling,
            self._rotation[idxs],
            self._appearance_level[idxs],
            gaussian_features,
            gamma,
            adaptive_features,
            adaptive_cat_weight,
            adaptive_zero_weight,
            gate_logits,
            shape
        )

    def _mcmc_sample_alives(self, probs, num, alive_indices=None):
        """
        Sample Gaussian indices based on opacity probabilities.
        
        Args:
            probs: Probability weights for sampling (typically opacity values)
            num: Number of samples to draw
            alive_indices: Optional indices to sample from (if None, samples from all)
        
        Returns:
            sampled_idxs: Indices of sampled Gaussians
            ratio: Bincount tensor showing how many times each index was sampled
        """
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        
        # Create ratio tensor with proper size
        ratio = torch.zeros(self._xyz.shape[0], 1, device="cuda", dtype=torch.int32)
        bincount = torch.bincount(sampled_idxs, minlength=self._xyz.shape[0])
        ratio[:, 0] = bincount
        
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask):
        """
        Relocate dead Gaussians by sampling from alive ones.
        
        Dead Gaussians (those with very low opacity) are replaced with copies of
        alive Gaussians, with their opacity and scale adjusted using the MCMC
        relocation kernel to preserve the overall contribution.
        
        Args:
            dead_mask: Boolean tensor indicating which Gaussians are "dead" (low opacity)
        """
        if dead_mask.sum() == 0:
            return
        
        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        
        if alive_indices.shape[0] <= 0:
            return
        
        # Sample from alive Gaussians based on opacity
        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._mcmc_sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])
        
        # Get updated parameters for the sampled Gaussians
        (
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_ap_level,
            new_gaussian_features,
            new_gamma,
            new_adaptive_features,
            new_adaptive_cat_weight,
            new_adaptive_zero_weight,
            new_gate_logits,
            new_shape
        ) = self._mcmc_update_params(reinit_idx, ratio=ratio)

        # Reset optimizer state for sampled indices FIRST (before updating)
        # This is critical - we zero out momentum for source Gaussians that are giving away mass
        self._reset_optimizer_state_for_indices(reinit_idx.unique())

        # Update the source Gaussians (they gave away some of their "mass")
        self._opacity.data[reinit_idx] = new_opacity
        self._scaling.data[reinit_idx] = new_scaling

        # Replace dead Gaussians with the new parameters (copy from sampled)
        self._xyz.data[dead_indices] = new_xyz
        self._features_dc.data[dead_indices] = new_features_dc
        self._features_rest.data[dead_indices] = new_features_rest
        self._opacity.data[dead_indices] = new_opacity
        self._scaling.data[dead_indices] = new_scaling
        self._rotation.data[dead_indices] = new_rotation
        self._appearance_level.data[dead_indices] = new_ap_level

        if self._gaussian_feat_dim > 0 and new_gaussian_features is not None:
            self._gaussian_features.data[dead_indices] = new_gaussian_features

        if self._adaptive_feat_dim > 0:
            if new_gamma is not None:
                self._gamma.data[dead_indices] = new_gamma
            if new_adaptive_features is not None:
                self._adaptive_features.data[dead_indices] = new_adaptive_features

        if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0 and new_adaptive_cat_weight is not None:
            self._adaptive_cat_weight.data[dead_indices] = new_adaptive_cat_weight

        if hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0 and new_adaptive_zero_weight is not None:
            self._adaptive_zero_weight.data[dead_indices] = new_adaptive_zero_weight

        if hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0 and new_gate_logits is not None:
            self._gate_logits.data[dead_indices] = new_gate_logits

        if hasattr(self, '_shape') and self._shape.numel() > 0 and new_shape is not None:
            self._shape.data[dead_indices] = new_shape

        # Reset optimizer state for dead indices (they got completely new values)
        self._reset_optimizer_state_for_indices(dead_indices)

    def add_new_gs(self, cap_max):
        """
        Add new Gaussians by sampling from existing ones, up to a capacity limit.
        
        New Gaussians are created by sampling from existing Gaussians based on
        their opacity. The source Gaussians have their opacity and scale adjusted
        using the MCMC relocation kernel.
        
        Args:
            cap_max: Maximum total number of Gaussians allowed
        
        Returns:
            Number of new Gaussians added
        """
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        
        if num_gs <= 0:
            return 0
        
        # Sample from all Gaussians based on opacity
        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._mcmc_sample_alives(probs=probs, num=num_gs)
        
        # Get parameters for new Gaussians
        (
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_ap_level,
            new_gaussian_features,
            new_gamma,
            new_adaptive_features,
            new_adaptive_cat_weight,
            new_adaptive_zero_weight,
            new_gate_logits,
            new_shape
        ) = self._mcmc_update_params(add_idx, ratio=ratio)

        # Update source Gaussians (they gave away some of their "mass")
        self._opacity.data[add_idx] = new_opacity
        self._scaling.data[add_idx] = new_scaling

        # Add new Gaussians using existing densification_postfix
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity,
            new_scaling, new_rotation, new_ap_level,
            new_gaussian_features, new_gamma, new_adaptive_features, new_adaptive_cat_weight,
            new_adaptive_zero_weight, new_gate_logits, new_shape
        )
        
        # Reset optimizer state for modified source indices
        self._reset_optimizer_state_for_indices(add_idx.unique())
        
        return num_gs

    def _reset_optimizer_state_for_indices(self, inds):
        """
        Reset optimizer state (momentum) for specific Gaussian indices.
        
        Args:
            inds: Indices of Gaussians whose optimizer state should be reset
        """
        if inds.numel() == 0:
            return
            
        for group in self.optimizer.param_groups:
            if group["name"] in ["mlp", "env"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
