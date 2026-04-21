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


def _fibonacci_sphere(num_points: int) -> torch.Tensor:
    """Uniformly-distributed unit vectors on the sphere via golden-angle spiral.
    Returns [num_points, 3] on CPU (caller moves to device)."""
    indices = torch.arange(0, num_points, dtype=torch.float32) + 0.5
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (indices / num_points) * 2.0
    r = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    theta = phi * indices
    x = torch.cos(theta) * r
    z = torch.sin(theta) * r
    return torch.stack([x, y, z], dim=1)


class SparseGaussianAdam(torch.optim.Adam):
    """Adam optimizer that only updates visible Gaussians (MSv2).

    Each parameter group holds one tensor of shape [N, ...] where N = num Gaussians.
    On step(), a visibility mask (bool[N]) selects which Gaussians get updated.
    Invisible Gaussians keep their current parameters and Adam states unchanged.
    """

    def __init__(self, params, lr=0.0, eps=1e-15, betas=(0.9, 0.999)):
        super().__init__(params, lr=lr, eps=eps, betas=betas)

    @torch.no_grad()
    def step(self, visibility=None, N=None):
        """Sparse Adam step: only update visible Gaussians.

        Args:
            visibility: bool tensor [N] — True for visible Gaussians. If None, updates all.
            N: total number of Gaussians. If None, inferred from visibility.
        """
        if visibility is None:
            return super().step()

        for group in self.param_groups:
            lr = group["lr"]
            if lr == 0:
                continue
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None or p.grad.numel() == 0:
                    continue

                # Lazy state init
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"].item()

                # Slice visibility to match param's first dim
                vis = visibility[:p.shape[0]]

                # Index visible rows
                g = p.grad[vis]
                ea = state["exp_avg"][vis]
                easq = state["exp_avg_sq"][vis]

                # Adam moment updates
                ea.mul_(beta1).add_(g, alpha=1 - beta1)
                easq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Write back
                state["exp_avg"][vis] = ea
                state["exp_avg_sq"][vis] = easq

                # Bias-corrected step (matches MSv2 CUDA kernel)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                corrected_lr = lr * (bias_correction2 ** 0.5) / bias_correction1

                p[vis] -= corrected_lr * ea / (easq.sqrt() + eps)


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
        self.active_sh_degree = sh_degree  # Full SH degree from start
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.env_map = None
        self.base_opacity = 0.0
        self._appearance_level = torch.empty(0)
        self.feat_gradient_accum = torch.empty(0)

        # --minimc: per-Gaussian photometric error + pixel-ownership accumulators.
        # See train loop for per-step scatter_add. Lifecycle mirrors xyz_gradient_accum.
        self.minimc_error_accum = torch.empty(0)
        self.minimc_win_count = torch.empty(0)
        
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

        # --feature beta: spherical beta parameterization of view-dependent color.
        # Replaces the higher-order SH bands with K primitive "lobes" per Gaussian.
        # Storage:
        #   _sb_params     [N, sb_number, 6] = (r, g, b, theta, phi, beta_raw_per_primitive)
        # (removed _sb_sharpness — reference uses per-primitive sb_params[...,5] only)
        # View-dep color at each Gaussian: see eval_sb() in gaussian_renderer/__init__.py.
        # Default feature is "sh"; these tensors stay empty until create_from_pcd
        # is called with sb_number > 0.
        self.feature_mode = "sh"     # "sh" or "beta"
        self.sb_number = 0           # number of beta primitives per Gaussian (K)
        self._sb_params = torch.empty(0)

        # MEGS-2 Spherical Gaussian (SG) feature — --feature sg
        # Per-Gaussian: K SG axes, each (direction, sharpness, rgb)
        #   _sg_directions [N, sg_number, 3]  — unit vectors on access (raw stored)
        #   _sg_sharpness  [N, sg_number, 1]  — |λ| on access (raw stored)
        #   _sg_rgb        [N, sg_number, 3]  — additive RGB modulated by the lobe
        # Color: rgb_out = sum_k sg_rgb[k] * exp(|λ[k]| * (cos θ_k − 1))
        self.sg_number = 0
        self._sg_directions = torch.empty(0)
        self._sg_sharpness_sg = torch.empty(0)   # separate from _sb_sharpness
        self._sg_rgb = torch.empty(0)

        # Spherical Voronoi (SV) feature — --feature voronoi
        # Per-Gaussian: K sites on sphere, each with an RGB.
        #   _sv_sites   [N, K, 3]  raw 3D vectors (normalized at eval, magnitude = τ).
        #   _sv_colors  [N, K, 3]  per-site RGB (no activation).
        # Color: logits = -τ_k * ||sites_k − view_dir||,  W = softmax(logits),
        #        V = clamp_min(Σ_k W_k * color_k, 0).
        self.sv_number = 0
        self._sv_sites = torch.empty(0)
        self._sv_colors = torch.empty(0)

        # Flex kernel: per-Gaussian learnable beta for Gaussian sharpening
        # softplus(_flex_beta) gives range [0, inf), typically [0, ~50]
        # beta=0 = standard Gaussian, beta>0 = sharper/more opaque
        self._flex_beta = torch.empty(0)

        self.mini_factor_culling = None

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
    def get_sg_directions(self):
        """Returns unit-normalized SG direction vectors [N, sg_number, 3]."""
        d = self._sg_directions
        return d / (d.norm(dim=-1, keepdim=True) + 1e-8)

    @property
    def get_sg_sharpness(self):
        """Returns non-negative SG sharpness [N, sg_number, 1] via abs()."""
        return torch.abs(self._sg_sharpness_sg)

    @property
    def get_sg_rgb(self):
        """Returns raw SG RGB coefficients [N, sg_number, 3] (no activation)."""
        return self._sg_rgb

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
        elif self.kernel_type == "nexel":
            # Nexel: per-axis gamma = exp(raw) + 1, range [1, inf).
            # γ=1: standard Gaussian. γ>1: softer/broader. γ<1 impossible (clamped).
            # Returns [N, 2] for (gamma_x, gamma_y).
            return torch.exp(self._shape) + 1.0
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
        
        # Initialize per-Gaussian features for cat mode, 3D mode, and 3D_direct mode
        # Dimension = hybrid_levels * per_level_dim (default: 3 * 4 = 12)
        if hasattr(args, 'method') and args.method in ["cat", "3D", "3D_direct", "3D_direct_fused", "3D_direct_lean", "3D_direct_fp16", "3D_direct_TC", "3D_SH_TC"] and hasattr(args, 'hybrid_levels'):
            per_level_dim = 4  # From config encoding.hashgrid.dim
            self._gaussian_feat_dim = args.hybrid_levels * per_level_dim
        else:
            self._gaussian_feat_dim = 0
        
        if self._gaussian_feat_dim > 0:
            gaussian_feats = torch.randn((self.get_xyz.shape[0], self._gaussian_feat_dim), device="cuda").float() * 0.01
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
        elif hasattr(args, 'kernel') and args.kernel == "nexel":
            self.kernel_type = "nexel"
            # Per-axis gamma exponents: γ = exp(raw) + 1, so γ ≥ 1.
            # Init raw = -5 → γ ≈ 1.007 (near-Gaussian at start).
            # [N, 2] for (gamma_x, gamma_y).
            gamma_init = -5.0 * torch.ones((fused_point_cloud.shape[0], 2), dtype=torch.float, device="cuda")
            self._shape = nn.Parameter(gamma_init.requires_grad_(True))
            print(f"[NEXEL] Initialized {fused_point_cloud.shape[0]} Gaussians with per-axis gamma")
        else:
            self.kernel_type = "gaussian"
            self._shape = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))

        # Initialize spherical-beta params if --feature beta
        self.feature_mode = getattr(args, 'feature', 'sh')
        if self.feature_mode == "beta":
            self.sb_number = int(getattr(args, 'sb_number', 2))
            N = fused_point_cloud.shape[0]
            # Per-primitive: [r, g, b, theta, phi, beta_raw]
            # r,g,b: small random (softplus-activated → small positive colors)
            # theta: uniform [0, pi], phi: uniform [0, 2*pi]  → lobe directions on sphere
            # beta_raw: 0 → activated beta = 4*exp(0) = 4 (moderate sharpness)
            # Reference beta-splatting init: RGB=0 (→ steep softplus ≈ 0.12),
            # theta∈[0,π], phi∈[0,2π], per-primitive beta=0 (→ exp(4·0)=exponent 4).
            sb = torch.zeros((N, self.sb_number, 6), dtype=torch.float, device="cuda")
            sb[..., 3] = torch.rand(N, self.sb_number, device="cuda") * np.pi
            sb[..., 4] = torch.rand(N, self.sb_number, device="cuda") * 2.0 * np.pi
            # sb[..., 5] per-primitive beta stays 0 → 4·exp(0)=4 (default exponent)
            self._sb_params = nn.Parameter(sb.requires_grad_(True))
            print(f"[FEATURE beta] Initialized {N} Gaussians with sb_number={self.sb_number} primitives")
        elif self.feature_mode == "sg":
            # MEGS-2 Spherical Gaussian feature
            self.sg_number = int(getattr(args, 'sb_number', 2))  # reuse --sb_number flag
            N = fused_point_cloud.shape[0]
            # Random unit-vector directions
            d = torch.randn(N, self.sg_number, 3, device="cuda")
            d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
            self._sg_directions = nn.Parameter(d.requires_grad_(True))
            # Sharpness: init to 0.1 (moderate lobes)
            self._sg_sharpness_sg = nn.Parameter(
                0.1 * torch.ones((N, self.sg_number, 1), device="cuda").requires_grad_(True)
            )
            # RGB: small random
            self._sg_rgb = nn.Parameter(
                (0.1 * torch.randn(N, self.sg_number, 3, device="cuda")).requires_grad_(True)
            )
            print(f"[FEATURE sg] Initialized {N} Gaussians with sg_number={self.sg_number} axes")
            # Empty SB / SV params
            self.sb_number = 0
            self._sb_params = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self.sv_number = 0
            self._sv_sites = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._sv_colors = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
        elif self.feature_mode == "voronoi":
            # Spherical Voronoi: K sites on sphere (fibonacci init), RGB per site.
            # Hybrid with SH DC: SH degree-0 carries view-independent base color,
            # SV adds view-dependent refinement on top. SV colors init to 0 so at
            # t=0 the color equals the SH DC (pcd RGB) and SV learns the delta.
            self.sv_number = int(getattr(args, 'sb_number', 8))  # reuse --sb_number (default 8)
            N = fused_point_cloud.shape[0]
            fib = _fibonacci_sphere(self.sv_number).to("cuda")
            sites = fib.unsqueeze(0).expand(N, -1, -1).contiguous()
            self._sv_sites = nn.Parameter(sites.requires_grad_(True))
            colors = torch.zeros((N, self.sv_number, 3), device="cuda")
            self._sv_colors = nn.Parameter(colors.requires_grad_(True))
            print(f"[FEATURE voronoi] Initialized {N} Gaussians with sv_number={self.sv_number} sites "
                  f"(hybrid: SH DC view-independent + SV view-dependent, SV colors init to 0)")
            # Empty SB / SG params
            self.sb_number = 0
            self._sb_params = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self.sg_number = 0
            self._sg_directions = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._sg_sharpness_sg = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._sg_rgb = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
        else:
            self.sb_number = 0
            self._sb_params = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self.sg_number = 0
            self._sg_directions = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._sg_sharpness_sg = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._sg_rgb = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self.sv_number = 0
            self._sv_sites = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))
            self._sv_colors = nn.Parameter(torch.empty(0, device="cuda").requires_grad_(False))

    def reinitial_pts(self, pts, rgb):
        """Reinitialize Gaussian parameters from point positions and RGB colors.

        Used by GaussianSpa Phase 1: after importance-based pruning, reinitialize
        scales from nearest-neighbor distances, reset rotations and opacities.
        """
        fused_color = RGB2SH(rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(pts), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.zeros((pts.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((pts.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(pts.detach().clone().requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # CRITICAL: ap_level must be 24, NOT 0. See hashgrid.h: max_level = min(ap_level, L).
        self._appearance_level = nn.Parameter(torch.ones(pts.shape[0], 1, device="cuda").float() * 24, requires_grad=False)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.feat_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # --minimc: per-Gaussian error + win-count accumulators (reset each training_setup
        # and each cull cycle). Safe to (re)allocate here after reinits.
        self.minimc_error_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.minimc_win_count = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # For diffuse mode, don't train features_rest (SH degree 0 only)
        f_rest_lr = 0.0 if self._diffuse_mode else training_args.feature_lr / 20.0

        # Freeze SH: zero LR for both DC and rest (isolate hashgrid learning)
        if getattr(training_args, 'freeze_sh', False):
            f_rest_lr = 0.0

        # For diffuse_offset mode, reduce learning rates for stability
        f_dc_lr = training_args.feature_lr
        if getattr(training_args, 'freeze_sh', False):
            f_dc_lr = 0.0
        xyz_lr_scale = 1.0
        if self._diffuse_offset_mode:
            f_dc_lr = training_args.feature_lr * 0.01  # 100x smaller for offset stability
            xyz_lr_scale = 0.1  # 10x smaller for position stability (reduce overpruning)

        # --feature beta/sg/voronoi: DC + directional lobes decomposition (matches
        # beta-splatting's sh_degree=0 + sb_number=2 and MEGS-2's rgb_base + SG).
        # DC is trainable (view-independent), higher-order SH frozen at zero,
        # directional feature adds view-dependence on top.
        if self.feature_mode in ("voronoi", "beta", "sg"):
            f_rest_lr = 0.0  # only DC is trainable for directional-lobe features

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

        # --feature beta: spherical-beta parameter LR groups.
        # Matches beta-splatting reference: only sb_params (per-primitive
        # r,g,b,θ,φ,β_per_prim) is optimized. No shared sharpness parameter.
        if self.feature_mode == "beta" and self._sb_params.numel() > 0:
            _sb_lr   = float(getattr(training_args, 'sb_params_lr', 0.0025))
            l.append({'params': [self._sb_params], 'lr': _sb_lr, "name": "sb_params"})

        # --feature sg: MEGS-2 Spherical Gaussian LR groups.
        # Sharpness LR reduced to 0.25x feature_lr for stability (MEGS-2 used 4x
        # but that's janky here; lower LR avoids oscillation).
        if self.feature_mode == "sg" and self._sg_directions.numel() > 0:
            _sg_lr = training_args.feature_lr
            l.append({'params': [self._sg_directions], 'lr': _sg_lr,         "name": "sg_directions"})
            l.append({'params': [self._sg_sharpness_sg], 'lr': _sg_lr * 0.25, "name": "sg_sharpness"})
            l.append({'params': [self._sg_rgb],        'lr': _sg_lr,         "name": "sg_rgb"})

        # --feature voronoi: Spherical Voronoi (radiance) LR groups.
        # Matches sphericalvoronoi/radiance: sites use an exponential schedule
        # (5e-2 → 1e-4 over training), colors at a fixed small LR (0.000125 blender).
        # Defaults come from training_args; see train.py --sites_lr / --sv_color_lr.
        if self.feature_mode == "voronoi" and self._sv_sites.numel() > 0:
            _sites_lr_init = float(getattr(training_args, 'sites_lr', 5e-2))
            _color_lr = float(getattr(training_args, 'sv_color_lr', 1.25e-4))
            l.append({'params': [self._sv_sites],  'lr': _sites_lr_init, "name": "sv_sites"})
            l.append({'params': [self._sv_colors], 'lr': _color_lr,       "name": "sv_colors"})
            # Exponential scheduler for sv_sites LR (applied in update_learning_rate).
            _sites_lr_final = float(getattr(training_args, 'sites_lr_final', 1e-4))
            _max_steps = int(getattr(training_args, 'iterations', 30000))
            self.sv_sites_scheduler_args = get_expon_lr_func(
                lr_init=_sites_lr_init,
                lr_final=_sites_lr_final,
                lr_delay_mult=training_args.position_lr_delay_mult,
                max_steps=_max_steps,
            )
        
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

        if getattr(training_args, 'mini', False):
            self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale*xyz_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale*xyz_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        # Iteration offset for the xyz exponential schedule. Depth reinit resets this so
        # each fresh cohort of Gaussians gets a full xyz LR budget instead of being born
        # into a late-decay regime. See reset_xyz_lr_schedule().
        if not hasattr(self, 'xyz_lr_offset'):
            self.xyz_lr_offset = 0


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        xyz_lr = None
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                xyz_lr = self.xyz_scheduler_args(iteration - self.xyz_lr_offset)
                param_group['lr'] = xyz_lr
            # --feature voronoi: exponential decay on sv_sites LR (matches reference intent).
            elif param_group["name"] == "sv_sites" and hasattr(self, 'sv_sites_scheduler_args'):
                param_group['lr'] = self.sv_sites_scheduler_args(iteration)
        return xyz_lr

    def reset_xyz_lr_schedule(self, current_iteration):
        """Reset the xyz exponential LR schedule's t=0 to `current_iteration`.
        Call this after a depth reinit so the fresh Gaussians start with position_lr_init,
        not the heavily-decayed LR implied by the absolute iteration counter.
        """
        self.xyz_lr_offset = int(current_iteration)

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
        elif args is not None and hasattr(args, 'method') and args.method in ["cat", "3D", "3D_direct"] and hasattr(args, 'hybrid_levels'):
            # Cat/3D mode but no features in PLY - initialize them (for old checkpoints)
            per_level_dim = 4
            self._gaussian_feat_dim = args.hybrid_levels * per_level_dim
            gaussian_feats = torch.randn((xyz.shape[0], self._gaussian_feat_dim), device="cuda").float() * 0.01
            self._gaussian_features = nn.Parameter(gaussian_feats.requires_grad_(True))
            print(f"Warning: No per-Gaussian features in PLY, initialized {self._gaussian_feat_dim}D randn*0.01 for {args.method} mode")
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
                if stored_state is None:
                    # No Adam state yet (e.g. param was freshly recreated by
                    # cat_tensors_to_optimizer/_prune_optimizer on a cold
                    # optimizer). Just swap the param; Adam will initialize
                    # fresh zero moments on the next step.
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                else:
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
        # --feature beta: sb params go through the optimizer pruner like any other param.
        if "sb_params" in optimizable_tensors:
            self._sb_params = optimizable_tensors["sb_params"]
        # --feature sg: directions / sharpness / rgb also go through the pruner.
        if "sg_directions" in optimizable_tensors:
            self._sg_directions = optimizable_tensors["sg_directions"]
        if "sg_sharpness" in optimizable_tensors:
            self._sg_sharpness_sg = optimizable_tensors["sg_sharpness"]
        if "sg_rgb" in optimizable_tensors:
            self._sg_rgb = optimizable_tensors["sg_rgb"]
        # --feature voronoi
        if "sv_sites" in optimizable_tensors:
            self._sv_sites = optimizable_tensors["sv_sites"]
        if "sv_colors" in optimizable_tensors:
            self._sv_colors = optimizable_tensors["sv_colors"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.feat_gradient_accum = self.feat_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        # --minimc accumulators (only if they've been allocated this run)
        if self.minimc_error_accum.numel() > 0 and self.minimc_error_accum.shape[0] == valid_points_mask.shape[0]:
            self.minimc_error_accum = self.minimc_error_accum[valid_points_mask]
            self.minimc_win_count = self.minimc_win_count[valid_points_mask]

        if self.mini_factor_culling is not None:
            if self.mini_factor_culling.shape[0] == valid_points_mask.shape[0]:
                self.mini_factor_culling = self.mini_factor_culling[valid_points_mask]
            else:
                self.mini_factor_culling = None  # Size mismatch, invalidate

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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_ap_level=None, new_gaussian_features=None, new_gamma=None, new_adaptive_features=None, new_adaptive_cat_weight=None, new_adaptive_zero_weight=None, new_gate_logits=None, new_shape=None, new_flex_beta=None, new_sb_params=None, new_sg_directions=None, new_sg_sharpness=None, new_sg_rgb=None, new_sv_sites=None, new_sv_colors=None):
        # CRITICAL: ap_level defaults to 24 (all hash levels active).
        # ap_level=0 silently disables hash encoding — see hashgrid.h: max_level = min(ap_level, L).
        if new_ap_level is None:
            new_ap_level = torch.ones(new_xyz.shape[0], 1, device="cuda") * 24
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

        # --feature beta: sb params (fresh children inherit donor values if provided,
        # otherwise zero-init; normally densify_and_clone/split passes them through).
        if new_sb_params is not None and self._sb_params.numel() > 0:
            d["sb_params"] = new_sb_params
        # --feature sg: per-axis direction/sharpness/rgb
        if new_sg_directions is not None and self._sg_directions.numel() > 0:
            d["sg_directions"] = new_sg_directions
        if new_sg_sharpness is not None and self._sg_sharpness_sg.numel() > 0:
            d["sg_sharpness"] = new_sg_sharpness
        if new_sg_rgb is not None and self._sg_rgb.numel() > 0:
            d["sg_rgb"] = new_sg_rgb
        # --feature voronoi: per-site direction (+tau via magnitude) and rgb
        if new_sv_sites is not None and self._sv_sites.numel() > 0:
            d["sv_sites"] = new_sv_sites
        if new_sv_colors is not None and self._sv_colors.numel() > 0:
            d["sv_colors"] = new_sv_colors

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
        if "sb_params" in optimizable_tensors:
            self._sb_params = optimizable_tensors["sb_params"]
        if "sg_directions" in optimizable_tensors:
            self._sg_directions = optimizable_tensors["sg_directions"]
        if "sg_sharpness" in optimizable_tensors:
            self._sg_sharpness_sg = optimizable_tensors["sg_sharpness"]
        if "sg_rgb" in optimizable_tensors:
            self._sg_rgb = optimizable_tensors["sg_rgb"]
        if "sv_sites" in optimizable_tensors:
            self._sv_sites = optimizable_tensors["sv_sites"]
        if "sv_colors" in optimizable_tensors:
            self._sv_colors = optimizable_tensors["sv_colors"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.feat_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # --minimc accumulators: zero-reset after densification/split (same as xyz_gradient_accum)
        if self.minimc_error_accum.numel() > 0:
            self.minimc_error_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.minimc_win_count = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if self.mini_factor_culling is not None:
            if self.mini_factor_culling.shape[0] + new_xyz.shape[0] == self.get_xyz.shape[0]:
                self.mini_factor_culling = torch.cat([self.mini_factor_culling, torch.ones(new_xyz.shape[0], 1, device='cuda')])
            else:
                self.mini_factor_culling = None  # Size mismatch, invalidate

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

        # Handle spherical-beta params (--feature beta)
        new_sb_params = None
        if self._sb_params.numel() > 0:
            new_sb_params = self._sb_params[selected_pts_mask].repeat(N, 1, 1)

        # SG lobes — children inherit donor's direction/sharpness/rgb
        new_sg_directions = None
        new_sg_sharpness_split = None
        new_sg_rgb_split = None
        if self._sg_directions.numel() > 0:
            new_sg_directions = self._sg_directions[selected_pts_mask].repeat(N, 1, 1)
            new_sg_sharpness_split = self._sg_sharpness_sg[selected_pts_mask].repeat(N, 1, 1)
            new_sg_rgb_split = self._sg_rgb[selected_pts_mask].repeat(N, 1, 1)

        # SV sites — children inherit donor's sites/colors
        new_sv_sites = None
        new_sv_colors = None
        if self._sv_sites.numel() > 0:
            new_sv_sites = self._sv_sites[selected_pts_mask].repeat(N, 1, 1)
            new_sv_colors = self._sv_colors[selected_pts_mask].repeat(N, 1, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_ap_level, new_gaussian_features, new_gamma, new_adaptive_features, new_adaptive_cat_weight, new_adaptive_zero_weight, new_gate_logits, new_shape, new_flex_beta, new_sb_params, new_sg_directions, new_sg_sharpness_split, new_sg_rgb_split, new_sv_sites, new_sv_colors)

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

        # Handle spherical-beta params (--feature beta)
        new_sb_params = None
        if self._sb_params.numel() > 0:
            new_sb_params = self._sb_params[selected_pts_mask]

        # Handle spherical-gaussian params (--feature sg)
        new_sg_directions_c = None
        new_sg_sharpness_c = None
        new_sg_rgb_c = None
        if self._sg_directions.numel() > 0:
            new_sg_directions_c = self._sg_directions[selected_pts_mask]
            new_sg_sharpness_c = self._sg_sharpness_sg[selected_pts_mask]
            new_sg_rgb_c = self._sg_rgb[selected_pts_mask]

        # Handle spherical-voronoi params (--feature voronoi)
        new_sv_sites_c = None
        new_sv_colors_c = None
        if self._sv_sites.numel() > 0:
            new_sv_sites_c = self._sv_sites[selected_pts_mask]
            new_sv_colors_c = self._sv_colors[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_ap_level, new_gaussian_features, new_gamma, new_adaptive_features, new_adaptive_cat_weight, new_adaptive_zero_weight, new_gate_logits, new_shape, new_flex_beta, new_sb_params, new_sg_directions_c, new_sg_sharpness_c, new_sg_rgb_c, new_sv_sites_c, new_sv_colors_c)

    def _clone_by_mask_fastgs(self, selected_pts_mask):
        """Clone Gaussians flagged by ``selected_pts_mask`` (bool, [N]).
        Field handling mirrors ``densify_and_clone`` but takes a pre-built mask.
        """
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_ap_level = self._appearance_level[selected_pts_mask]

        new_gaussian_features = None
        if self._gaussian_feat_dim > 0:
            new_gaussian_features = self._gaussian_features[selected_pts_mask]

        new_gamma = None
        new_adaptive_features = None
        if self._adaptive_feat_dim > 0:
            new_gamma = self._gamma[selected_pts_mask]
            new_adaptive_features = self._adaptive_features[selected_pts_mask]

        num_new = int(selected_pts_mask.sum().item())
        new_adaptive_cat_weight = None
        if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0:
            if self._relocation_mode == "reset":
                new_adaptive_cat_weight = torch.zeros((num_new, 1), device="cuda")
            else:
                new_adaptive_cat_weight = self._adaptive_cat_weight[selected_pts_mask]

        new_adaptive_zero_weight = None
        if hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0:
            if self._relocation_mode == "reset":
                new_adaptive_zero_weight = torch.zeros((num_new, 1), device="cuda")
            else:
                new_adaptive_zero_weight = self._adaptive_zero_weight[selected_pts_mask]

        new_gate_logits = None
        if hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0:
            if self._relocation_mode == "reset":
                new_gate_logits = torch.zeros((num_new, 1), device="cuda")
            else:
                new_gate_logits = self._gate_logits[selected_pts_mask]

        new_shape = None
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            if self._frozen_beta_raw is not None:
                new_shape = torch.full((num_new, 1), self._frozen_beta_raw, device="cuda")
            else:
                new_shape = self._shape[selected_pts_mask]

        new_flex_beta = None
        if hasattr(self, '_flex_beta') and self._flex_beta.numel() > 0:
            new_flex_beta = self._flex_beta[selected_pts_mask]

        new_sb_params = None
        if self._sb_params.numel() > 0:
            new_sb_params = self._sb_params[selected_pts_mask]

        new_sg_directions_c = None
        new_sg_sharpness_c = None
        new_sg_rgb_c = None
        if self._sg_directions.numel() > 0:
            new_sg_directions_c = self._sg_directions[selected_pts_mask]
            new_sg_sharpness_c = self._sg_sharpness_sg[selected_pts_mask]
            new_sg_rgb_c = self._sg_rgb[selected_pts_mask]

        new_sv_sites_c = None
        new_sv_colors_c = None
        if self._sv_sites.numel() > 0:
            new_sv_sites_c = self._sv_sites[selected_pts_mask]
            new_sv_colors_c = self._sv_colors[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities,
            new_scaling, new_rotation, new_ap_level, new_gaussian_features,
            new_gamma, new_adaptive_features, new_adaptive_cat_weight,
            new_adaptive_zero_weight, new_gate_logits, new_shape, new_flex_beta,
            new_sb_params, new_sg_directions_c, new_sg_sharpness_c,
            new_sg_rgb_c, new_sv_sites_c, new_sv_colors_c)
        return num_new

    def _split_by_mask_fastgs(self, selected_pts_mask, N=2):
        """Split Gaussians flagged by ``selected_pts_mask`` into ``N`` children.
        Mirrors ``densify_and_split`` but takes a pre-built mask.
        """
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                   + self.get_xyz[selected_pts_mask].repeat(N, 1))
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_ap_level = self._appearance_level[selected_pts_mask].repeat(N, 1)

        new_gaussian_features = None
        if self._gaussian_feat_dim > 0:
            new_gaussian_features = self._gaussian_features[selected_pts_mask].repeat(N, 1)

        new_gamma = None
        new_adaptive_features = None
        if self._adaptive_feat_dim > 0:
            new_gamma = self._gamma[selected_pts_mask].repeat(N, 1)
            new_adaptive_features = self._adaptive_features[selected_pts_mask].repeat(N, 1)

        num_new = int(selected_pts_mask.sum().item()) * N
        new_adaptive_cat_weight = None
        if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0:
            if self._relocation_mode == "reset":
                new_adaptive_cat_weight = torch.zeros((num_new, 1), device="cuda")
            else:
                new_adaptive_cat_weight = self._adaptive_cat_weight[selected_pts_mask].repeat(N, 1)

        new_adaptive_zero_weight = None
        if hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0:
            if self._relocation_mode == "reset":
                new_adaptive_zero_weight = torch.zeros((num_new, 1), device="cuda")
            else:
                new_adaptive_zero_weight = self._adaptive_zero_weight[selected_pts_mask].repeat(N, 1)

        new_gate_logits = None
        if hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0:
            if self._relocation_mode == "reset":
                new_gate_logits = torch.zeros((num_new, 1), device="cuda")
            else:
                new_gate_logits = self._gate_logits[selected_pts_mask].repeat(N, 1)

        new_shape = None
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            if self._frozen_beta_raw is not None:
                new_shape = torch.full((num_new, 1), self._frozen_beta_raw, device="cuda")
            else:
                new_shape = self._shape[selected_pts_mask].repeat(N, 1)

        new_flex_beta = None
        if hasattr(self, '_flex_beta') and self._flex_beta.numel() > 0:
            new_flex_beta = self._flex_beta[selected_pts_mask].repeat(N, 1)

        new_sb_params = None
        if self._sb_params.numel() > 0:
            new_sb_params = self._sb_params[selected_pts_mask].repeat(N, 1, 1)

        new_sg_directions = None
        new_sg_sharpness_split = None
        new_sg_rgb_split = None
        if self._sg_directions.numel() > 0:
            new_sg_directions = self._sg_directions[selected_pts_mask].repeat(N, 1, 1)
            new_sg_sharpness_split = self._sg_sharpness_sg[selected_pts_mask].repeat(N, 1, 1)
            new_sg_rgb_split = self._sg_rgb[selected_pts_mask].repeat(N, 1, 1)

        new_sv_sites = None
        new_sv_colors = None
        if self._sv_sites.numel() > 0:
            new_sv_sites = self._sv_sites[selected_pts_mask].repeat(N, 1, 1)
            new_sv_colors = self._sv_colors[selected_pts_mask].repeat(N, 1, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity,
            new_scaling, new_rotation, new_ap_level, new_gaussian_features,
            new_gamma, new_adaptive_features, new_adaptive_cat_weight,
            new_adaptive_zero_weight, new_gate_logits, new_shape, new_flex_beta,
            new_sb_params, new_sg_directions, new_sg_sharpness_split,
            new_sg_rgb_split, new_sv_sites, new_sv_colors)

        # Prune the parents (children inherit at the tail of the tensor).
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * int(selected_pts_mask.sum()), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        return num_new

    def densify_and_prune_fastgs(self, min_opacity, extent, max_screen_size,
                                 importance_score, pruning_score,
                                 grad_thresh=0.0002, grad_abs_thresh=0.0012,
                                 dense=0.001, importance_thresh=1,
                                 prune_budget_frac=0.5):
        """FastGS densification + pruning. Grad-qualifiers combined with the
        multi-view consistency metric mask. Pruning uses the pruning_score to
        bias-sample within the opacity-pruned set (50% budget by default).
        """
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.norm(grad_vars, dim=-1) >= grad_thresh
        grad_qualifiers_abs = torch.norm(grads_abs, dim=-1) >= grad_abs_thresh
        max_scale = torch.max(self.get_scaling, dim=1).values
        clone_qualifiers = max_scale <= dense * extent
        split_qualifiers = max_scale > dense * extent

        all_clones = grad_qualifiers & clone_qualifiers
        all_splits = grad_qualifiers_abs & split_qualifiers

        # Multi-view consistency filter: require the Gaussian to appear in
        # high-error pixels on average across the K sampled views.
        N = self.get_xyz.shape[0]
        metric_mask = torch.zeros(N, dtype=torch.bool, device="cuda")
        if importance_score is not None and importance_score.numel() == N:
            metric_mask = importance_score > importance_thresh

        # Diagnostics: report the funnel at every call so starved densification is obvious.
        _n_grad = int(grad_qualifiers.sum().item())
        _n_grad_abs = int(grad_qualifiers_abs.sum().item())
        _n_small = int(clone_qualifiers.sum().item())
        _n_big = int(split_qualifiers.sum().item())
        _n_metric = int(metric_mask.sum().item())
        if importance_score is not None and importance_score.numel() == N:
            _imp_max = float(importance_score.max().item())
            _imp_mean = float(importance_score.float().mean().item())
            _imp_nonzero = int((importance_score > 0).sum().item())
        else:
            _imp_max = _imp_mean = 0.0
            _imp_nonzero = 0
        print(f"[FASTGS/funnel] N={N}, grad>={grad_thresh}:{_n_grad}, "
              f"grad_abs>={grad_abs_thresh}:{_n_grad_abs}, "
              f"small(<={dense}*ext):{_n_small}, big:{_n_big}, "
              f"metric>{importance_thresh}:{_n_metric} (imp max={_imp_max:.1f}, mean={_imp_mean:.2f}, nonzero={_imp_nonzero})")

        # Percentile dump — pick thresholds empirically. "What percentile does
        # my current threshold cut at?" tells you whether densification catches
        # the right slice of Gaussians.
        def _qdump(tensor, name, current_thresh=None, nonzero_only=False):
            if tensor.numel() == 0:
                return
            t = tensor.detach().flatten()
            if nonzero_only:
                t = t[t > 0]
                if t.numel() == 0:
                    print(f"[FASTGS/dist] {name}: all zeros")
                    return
            qs = torch.tensor([0.50, 0.75, 0.90, 0.95, 0.99], device=t.device)
            vals = torch.quantile(t.float(), qs).tolist()
            extra = ""
            if current_thresh is not None:
                pct_above = float((t > current_thresh).float().mean().item() * 100.0)
                extra = f"  | >{current_thresh:g}: {pct_above:.2f}% pass"
            print(f"[FASTGS/dist] {name:<14} "
                  f"q50={vals[0]:.6g}  q75={vals[1]:.6g}  q90={vals[2]:.6g}  "
                  f"q95={vals[3]:.6g}  q99={vals[4]:.6g}{extra}")

        _grad_norm = torch.norm(grad_vars, dim=-1)
        _grad_abs_norm = torch.norm(grads_abs, dim=-1)
        _qdump(_grad_norm,     "grad_norm",     grad_thresh,     nonzero_only=True)
        _qdump(_grad_abs_norm, "grad_abs_norm", grad_abs_thresh, nonzero_only=True)
        _qdump(max_scale,      f"max_scale (dense·ext={dense * extent:.5f})", dense * extent)
        if importance_score is not None and importance_score.numel() == N:
            _qdump(importance_score, "importance_sc", importance_thresh, nonzero_only=True)

        n_cloned = self._clone_by_mask_fastgs(metric_mask & all_clones)
        # After clone, tensor grew → keep masks of original length only for split.
        split_mask = metric_mask & all_splits
        n_split_parents = int(split_mask.sum().item())
        if n_split_parents > 0:
            padded_split = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            padded_split[:split_mask.shape[0]] = split_mask
            self._split_by_mask_fastgs(padded_split, N=2)

        # Opacity / size based pruning, bias-sampled by (1 / pruning_score).
        prune_mask = (self.get_opacity < min_opacity).squeeze(-1)
        if max_screen_size:
            big_vs = self.max_radii2D > max_screen_size
            big_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_vs | big_ws

        to_remove = int(prune_mask.sum().item())
        remove_budget = int(prune_budget_frac * to_remove)
        n_pruned = 0
        if remove_budget > 0 and pruning_score is not None:
            n_cur = self.get_xyz.shape[0]
            scores = 1.0 - pruning_score
            padded = torch.zeros(n_cur, dtype=torch.float32, device="cuda")
            padded[:scores.shape[0]] = 1.0 / (1e-6 + scores.squeeze())
            sampled = torch.multinomial(padded, remove_budget, replacement=False)
            sampled_mask = torch.zeros(n_cur, dtype=torch.bool, device="cuda")
            sampled_mask[sampled] = True
            final_prune = prune_mask & sampled_mask
            n_pruned = int(final_prune.sum().item())
            self.prune_points(final_prune)

        n_split_children = 2 * n_split_parents  # N=2 in _split_by_mask_fastgs
        n_net = n_cloned + n_split_children - n_split_parents - n_pruned
        print(f"[FASTGS/action] cloned={n_cloned}, split_parents={n_split_parents} "
              f"(→{n_split_children} children), opacity_pruneable={to_remove} "
              f"(budget={remove_budget}), actually_pruned={n_pruned}, "
              f"ΔN={n_net:+d} -> N={self.get_xyz.shape[0]}")
        torch.cuda.empty_cache()
        return {"cloned": n_cloned, "split_parents": n_split_parents, "pruned": n_pruned}

    def final_prune_fastgs(self, min_opacity, pruning_score, score_thresh=0.9):
        """FastGS final-stage prune: remove by opacity OR high pruning_score.
        Runs every 3000 iters after 15k in the paper.
        """
        prune_mask = (self.get_opacity < min_opacity).squeeze(-1)
        if pruning_score is not None and pruning_score.numel() == prune_mask.numel():
            prune_mask = prune_mask | (pruning_score > score_thresh)
        n_before = self.get_xyz.shape[0]
        self.prune_points(prune_mask)
        return n_before - self.get_xyz.shape[0]

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

        # AbsGS: use abs gradient for split (large over-reconstructed Gaussians),
        # signed gradient for clone (small under-reconstructed Gaussians).
        # Per AbsGS paper: gradient cancellation only affects large Gaussians.
        use_abs = getattr(self, 'use_absgs', False)
        if use_abs and self.xyz_gradient_accum_abs.sum() > 0:
            split_grads = self.xyz_gradient_accum_abs / self.denom
            split_grads[split_grads.isnan()] = 0.0
            if densify_tag == False:
                split_grads = split_grads * 0
        else:
            split_grads = grads

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(split_grads, max_grad, extent)
        
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

    @torch.no_grad()
    def morton_sort(self):
        """Sort all Gaussian parameters by Morton (z-order) code of their 3D positions.
        Improves spatial locality for hash table lookups and cache efficiency."""
        xyz = self._xyz.detach()
        N = xyz.shape[0]
        if N == 0:
            return

        # Normalize positions to [0, 1023] range for 10-bit Morton encoding
        xyz_min = xyz.min(dim=0).values
        xyz_max = xyz.max(dim=0).values
        xyz_range = (xyz_max - xyz_min).clamp(min=1e-6)
        xyz_norm = ((xyz - xyz_min) / xyz_range * 1023.0).clamp(0, 1023).long()

        # Compute 30-bit Morton code (interleave 10 bits of x, y, z)
        def expand_bits(v):
            # Spread 10 bits across 30 bits: 0b...zyx -> 0b...z00y00x00
            v = (v | (v << 16)) & 0x030000FF
            v = (v | (v << 8))  & 0x0300F00F
            v = (v | (v << 4))  & 0x030C30C3
            v = (v | (v << 2))  & 0x09249249
            return v

        x_exp = expand_bits(xyz_norm[:, 0])
        y_exp = expand_bits(xyz_norm[:, 1])
        z_exp = expand_bits(xyz_norm[:, 2])
        morton_codes = x_exp | (y_exp << 1) | (z_exp << 2)

        order = torch.argsort(morton_codes)

        # Reorder all optimizer param groups (params + Adam states)
        for group in self.optimizer.param_groups:
            if group["name"] in ("mlp", "env"):
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            new_param = nn.Parameter(group["params"][0].data[order].contiguous().requires_grad_(True))
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][order].contiguous()
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][order].contiguous()
                del self.optimizer.state[group['params'][0]]
                self.optimizer.state[new_param] = stored_state
            else:
                del self.optimizer.state[group['params'][0]]
            group["params"][0] = new_param

        # Update member references
        for group in self.optimizer.param_groups:
            name = group["name"]
            param = group["params"][0]
            if name == "xyz": self._xyz = param
            elif name == "f_dc": self._features_dc = param
            elif name == "f_rest": self._features_rest = param
            elif name == "opacity": self._opacity = param
            elif name == "scaling": self._scaling = param
            elif name == "rotation": self._rotation = param
            elif name == "ap_level": self._appearance_level = param
            elif name == "gaussian_features": self._gaussian_features = param
            elif name == "gamma": self._gamma = param
            elif name == "adaptive_features": self._adaptive_features = param
            elif name == "adaptive_cat_weight" and hasattr(self, '_adaptive_cat_weight'): self._adaptive_cat_weight = param
            elif name == "adaptive_zero_weight": self._adaptive_zero_weight = param
            elif name == "gate_logits": self._gate_logits = param
            elif name == "shape": self._shape = param
            elif name == "flex_beta": self._flex_beta = param
            elif name == "sb_params": self._sb_params = param
            elif name == "sg_directions": self._sg_directions = param
            elif name == "sg_sharpness": self._sg_sharpness_sg = param
            elif name == "sg_rgb": self._sg_rgb = param
            elif name == "sv_sites": self._sv_sites = param
            elif name == "sv_colors": self._sv_colors = param

        # Handle frozen shape (not in optimizer)
        if hasattr(self, '_shape') and self._shape.numel() > 0 and not self._shape.requires_grad:
            self._shape = nn.Parameter(self._shape.data[order].contiguous(), requires_grad=False)

        # Reorder non-optimizer tensors
        self.xyz_gradient_accum = self.xyz_gradient_accum[order].contiguous()
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[order].contiguous()
        self.feat_gradient_accum = self.feat_gradient_accum[order].contiguous()
        self.denom = self.denom[order].contiguous()
        self.max_radii2D = self.max_radii2D[order].contiguous()
        if self.minimc_error_accum.numel() > 0:
            self.minimc_error_accum = self.minimc_error_accum[order].contiguous()
            self.minimc_win_count = self.minimc_win_count[order].contiguous()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, pixels = None):
        # factor_culling: per-Gaussian importance scaling from MSv2 (count_vis / count_rad)
        # Suppresses gradients for Gaussians visible in many views but important in few
        factor = getattr(self, 'mini_factor_culling', None)

        full_grad = viewspace_point_tensor.grad[update_filter]
        # Signed gradient (channels 0:2) — always accumulated, used for clone decisions
        signed_grad = full_grad[:, :2] if full_grad.shape[-1] > 2 else full_grad

        if pixels is not None:
            grad_norm = torch.norm(signed_grad, dim=-1, keepdim=True) * pixels[update_filter]
            if factor is not None:
                N = self.xyz_gradient_accum.shape[0]
                if factor.shape[0] == N:
                    grad_norm = grad_norm * factor[update_filter]
            self.xyz_gradient_accum[update_filter] += grad_norm
            self.denom[update_filter] += pixels[update_filter]
        else:
            grad_norm = torch.norm(signed_grad, dim=-1, keepdim=True)
            if factor is not None:
                N = self.xyz_gradient_accum.shape[0]
                if factor.shape[0] == N:
                    grad_norm = grad_norm * factor[update_filter]
            self.xyz_gradient_accum[update_filter] += grad_norm
            self.denom[update_filter] += 1

        # AbsGS (--grads abs): also accumulate abs gradient (channels 2:4) for splits.
        # Per the AbsGS paper: abs gradients are only used for split decisions (large
        # over-reconstructed Gaussians), NOT clone decisions (small Gaussians don't
        # suffer from gradient cancellation).
        use_abs = getattr(self, 'use_absgs', False)
        if use_abs and full_grad.shape[-1] >= 4:
            abs_grad = full_grad[:, 2:4]
            if pixels is not None:
                abs_norm = torch.norm(abs_grad, dim=-1, keepdim=True) * pixels[update_filter]
            else:
                abs_norm = torch.norm(abs_grad, dim=-1, keepdim=True)
            if factor is not None:
                N = self.xyz_gradient_accum_abs.shape[0]
                if factor.shape[0] == N:
                    abs_norm = abs_norm * factor[update_filter]
            self.xyz_gradient_accum_abs[update_filter] += abs_norm

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

    def _mcmc_donor_probs(self, indices: torch.Tensor, mode: str) -> torch.Tensor:
        """Unnormalized donor-sampling probabilities for MCMC relocation/add.

        mode='opacity': probs = activated opacity of each indexed Gaussian.
        mode='gradient': probs = accumulated xyz gradient magnitude (mean per step
            via xyz_gradient_accum / denom). Targets high-error regions. Falls
            back to opacity if the gradient accum isn't populated yet.
        """
        if mode == "gradient" and self.xyz_gradient_accum.numel() > 0 and self.denom.numel() > 0:
            denom = self.denom.clamp_min(1e-12)
            g = (self.xyz_gradient_accum / denom).squeeze(-1)  # [N]
            g = g[indices].clamp_min(0.0)
            if g.sum() > 1e-12:
                return g
            # Fall through to opacity if gradient is all zeros (e.g., early iters).
        return self.get_opacity[indices, 0] if self.get_opacity.dim() == 2 else self.get_opacity.squeeze(-1)[indices]

    def relocate_gs(self, dead_mask, probs_mode: str = "opacity"):
        """
        Relocate dead Gaussians by sampling from alive ones.

        Dead Gaussians (those with very low opacity) are replaced with copies of
        alive Gaussians, with their opacity and scale adjusted using the MCMC
        relocation kernel to preserve the overall contribution.

        Args:
            dead_mask: Boolean tensor indicating which Gaussians are "dead" (low opacity)
            probs_mode: 'opacity' (default) samples donors proportional to opacity;
                        'gradient' samples proportional to the accumulated xyz gradient
                        magnitude (requires xyz_gradient_accum populated).
        """
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # Sample donor indices from alive Gaussians.
        probs = self._mcmc_donor_probs(alive_indices, probs_mode)
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

    def add_new_gs(self, cap_max, probs_mode: str = "opacity"):
        """
        Add new Gaussians by sampling from existing ones, up to a capacity limit.

        Args:
            cap_max: Maximum total number of Gaussians allowed.
            probs_mode: 'opacity' (default) or 'gradient' — see relocate_gs docstring.
        Returns:
            Number of new Gaussians added
        """
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        # Sample donor indices from all Gaussians.
        all_idx = torch.arange(current_num_points, device=self._opacity.device)
        probs = self._mcmc_donor_probs(all_idx, probs_mode)
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

    @torch.no_grad()
    def consolidate_primitives(self,
                               tau_dist=None,
                               tau_normal=0.95,
                               tau_planar=None,
                               tau_feat=0.01,
                               verbose=True):
        """Mode-agnostic geometric consolidation: merge redundant primitives
        that pass the 4-gate test (normal alignment, coplanarity, spatial
        overlap, feature similarity). Works for 2DGS surfels; 3DGS ellipsoids
        would pick the smallest-scale axis as normal (not our case).

        Pipeline:
          1. scipy.spatial.cKDTree.query_pairs(τ_dist) gives all within-radius pairs.
          2. Vectorized 4-gate check (gate 3 is implicit via cKDTree radius).
          3. Greedy pairing (each primitive in at most one merge this cycle),
             preferring pairs with lowest feature MSE.
          4. For each pair (i, j): update i in-place with opacity-weighted
             averages (alpha-composition opacity), reset Adam momentum on i,
             and mark j for pruning.
          5. prune_points(j_mask) drops the absorbed primitives and shrinks
             every per-Gaussian tensor + Adam state via _prune_optimizer.

        Returns number of pairs merged.
        """
        N = self._xyz.shape[0]
        if N < 2:
            return 0

        # Extract current state (GPU).
        xyz_t = self._xyz.detach()
        rot_t = self._rotation.detach()
        scaling_t = self.get_scaling.detach()              # [N, 2] (2DGS) or [N, 3] (3DGS)
        opacity_t = self.get_opacity.squeeze(-1).detach()   # [N], sigmoid-activated

        max_scale = scaling_t.max(dim=1).values  # [N]
        if tau_dist is None or tau_dist <= 0.0:
            tau_dist = float(max_scale.median().item() * 0.5)
        if tau_planar is None or tau_planar <= 0.0:
            tau_planar = 0.5 * tau_dist

        # Normal: for 2DGS surfels the rotation's z-axis (column 2). For 3DGS
        # ellipsoids we'd pick the smallest-scale axis; our code path is 2DGS.
        R = build_rotation(rot_t)    # [N, 3, 3]
        normals = R[:, :, 2]         # [N, 3]

        xyz_np = xyz_t.cpu().numpy()
        normals_np = normals.cpu().numpy()
        opacity_np = opacity_t.cpu().numpy()
        features_dc_flat = self._features_dc.detach().reshape(N, -1).cpu().numpy()

        from scipy.spatial import cKDTree
        tree = cKDTree(xyz_np)
        pairs = tree.query_pairs(tau_dist, output_type="ndarray")
        P = int(len(pairs))
        if P == 0:
            if verbose:
                print(f"[MERGE] No pairs within τ_dist={tau_dist:.5g}")
            return 0

        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        ni, nj = normals_np[i_idx], normals_np[j_idx]
        pi, pj = xyz_np[i_idx], xyz_np[j_idx]

        # Gate 1: normal alignment.
        dot = np.sum(ni * nj, axis=1)
        gate1 = dot > tau_normal
        # Gate 2: coplanarity (projection of (pi - pj) onto avg normal).
        navg = ni + nj
        navg /= np.maximum(np.linalg.norm(navg, axis=1, keepdims=True), 1e-8)
        gate2 = np.abs(np.sum((pi - pj) * navg, axis=1)) < tau_planar
        # Gate 4: feature (SH DC) similarity.
        feat_mse = np.mean((features_dc_flat[i_idx] - features_dc_flat[j_idx]) ** 2, axis=1)
        gate4 = feat_mse < tau_feat
        passes = gate1 & gate2 & gate4
        n_pass = int(passes.sum())

        if n_pass == 0:
            if verbose:
                print(f"[MERGE] {P} candidate pairs, 0 pass 4 gates "
                      f"(~normal:{int((~gate1).sum())}, "
                      f"~planar:{int((~gate2).sum())}, "
                      f"~feat:{int((~gate4).sum())})")
            return 0

        # Greedy: prefer lowest feature MSE first; each primitive paired at most once.
        ordered_idx = np.argsort(feat_mse[passes])
        merge_pairs = pairs[passes][ordered_idx]
        used = np.zeros(N, dtype=bool)
        absorb_i, absorb_j = [], []
        for (i, j) in merge_pairs:
            if used[i] or used[j]:
                continue
            used[i] = True
            used[j] = True
            absorb_i.append(int(i))
            absorb_j.append(int(j))
        if len(absorb_i) == 0:
            return 0

        i_t = torch.tensor(absorb_i, dtype=torch.long, device="cuda")
        j_t = torch.tensor(absorb_j, dtype=torch.long, device="cuda")
        a_i = opacity_t[i_t]
        a_j = opacity_t[j_t]
        w_i = a_i / (a_i + a_j + 1e-8)  # [M]

        def _bcast(tensor, w):
            return w.view([-1] + [1] * (tensor.dim() - 1))

        # Opacity (alpha-composition, correct optical-thickness rule).
        a_new = (1.0 - (1.0 - a_i) * (1.0 - a_j)).clamp(1e-6, 1.0 - 1e-6)
        self._opacity.data[i_t, 0] = self.inverse_opacity_activation(a_new)

        # Position: opacity-weighted average.
        w = _bcast(self._xyz, w_i)
        self._xyz.data[i_t] = w * self._xyz.data[i_t] + (1.0 - w) * self._xyz.data[j_t]

        # Scaling: weighted average in raw (log) space.
        w = _bcast(self._scaling, w_i)
        self._scaling.data[i_t] = w * self._scaling.data[i_t] + (1.0 - w) * self._scaling.data[j_t]

        # Rotation: can't average quaternions; keep the higher-opacity partner's quat.
        flip = (a_j > a_i)
        if flip.any():
            self._rotation.data[i_t[flip]] = self._rotation.data[j_t[flip]]

        # All additional per-Gaussian tensors: opacity-weighted average in raw space.
        _feat_tensors = [
            "_features_dc", "_features_rest",
            "_gaussian_features", "_gamma",
            "_adaptive_features", "_adaptive_cat_weight", "_adaptive_zero_weight",
            "_gate_logits", "_shape", "_flex_beta",
            "_sb_params",
            "_sg_directions", "_sg_sharpness_sg", "_sg_rgb",
            "_sv_sites", "_sv_colors",
        ]
        for attr_name in _feat_tensors:
            t = getattr(self, attr_name, None)
            if t is None or t.numel() == 0 or t.shape[0] != N:
                continue
            w = _bcast(t, w_i)
            t.data[i_t] = w * t.data[i_t] + (1.0 - w) * t.data[j_t]

        # Reset Adam momentum on updated primitives; old momentum is stale.
        self._reset_optimizer_state_for_indices(i_t.unique())

        # Prune absorbed primitives (prune_points handles optimizer state).
        prune_mask = torch.zeros(N, dtype=torch.bool, device="cuda")
        prune_mask[j_t] = True
        self.prune_points(prune_mask)

        if verbose:
            print(f"[MERGE] {P} pairs (τ_dist={tau_dist:.5g}), {n_pass} pass 4 gates → "
                  f"{len(absorb_i)} merged after greedy pairing. "
                  f"N: {N} → {self._xyz.shape[0]}.")
        return len(absorb_i)

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

    # ==================== Zero-Waste RJ-MCMC (--minimc) ====================

    @torch.no_grad()
    def accumulate_minimc_error(self, max_contrib_idx, image, gt_image):
        """Per-step error + win-count routing for --minimc.

        Each pixel contributes its L1 RGB error to `minimc_error_accum[max_idx]` and
        bumps `minimc_win_count[max_idx]` by 1, where `max_idx` is the id of the
        Gaussian that was the max-weight contributor at that pixel.

        Args:
            max_contrib_idx: [1, H, W] or [H, W] int tensor (rasterizer out_index).
                             -1 marks pixels with no contributor.
            image:    [3, H, W] rendered image (float).
            gt_image: [3, H, W] ground-truth image (float).
        """
        if self.minimc_error_accum.numel() == 0:
            return
        if max_contrib_idx is None:
            return
        N = self._xyz.shape[0]
        # Lazily resize if the Gaussian count has changed since last allocation.
        if self.minimc_error_accum.shape[0] != N:
            self.minimc_error_accum = torch.zeros((N, 1), device="cuda")
            self.minimc_win_count = torch.zeros((N, 1), device="cuda")

        pixel_err = (image - gt_image).abs().mean(dim=0).reshape(-1)        # [H*W]
        ids = max_contrib_idx.long().reshape(-1)                            # [H*W]
        valid = (ids >= 0) & (ids < N)
        if not valid.any():
            return
        v_ids = ids[valid]
        v_err = pixel_err[valid]
        self.minimc_error_accum.view(-1).scatter_add_(0, v_ids, v_err)
        self.minimc_win_count.view(-1).scatter_add_(0, v_ids, torch.ones_like(v_err))

    @torch.no_grad()
    def minimc_cull_losers_to_dead_pool(self, dead_thresh=0.005):
        """Phase-2 mini cull: Gaussians that didn't win a single pixel since the
        last cull get their opacity snapped below `dead_thresh`, placing them in
        the MCMC dead pool (to be consumed by the next relocate_by_error call).
        Resets both accumulators for the next cycle.
        """
        if self.minimc_win_count.numel() == 0:
            return 0
        losers = (self.minimc_win_count.squeeze(-1) == 0)
        n = int(losers.sum().item())
        if n > 0:
            target = inverse_sigmoid(
                torch.tensor(dead_thresh * 0.5, device=self._opacity.device)
            )
            self._opacity.data[losers] = target
        self.minimc_error_accum.zero_()
        self.minimc_win_count.zero_()
        return n

    @torch.no_grad()
    def minimc_sweep_and_relocate(self, scene, render_fn, pipe, background, beta,
                                  iteration, cfg, ingp=None, imp_metric="indoor",
                                  dead_thresh=0.005,
                                  cull_single_view=True, low_imp_cdf_thres=0.999,
                                  growth_frac=0.05):
        """Phase 2+3 fused: full-view sweep that simultaneously computes per-Gaussian
        importance, single-view membership, AND photometric error attribution; then
        applies multi-criterion culling and MCMC relocation onto donors that are
        BOTH high-importance AND associated with high-error pixels.

        Each call:
          1. Renders every training view once with `record_transmittance=True`.
             Accumulates THREE per-Gaussian quantities across views:
               - `imp`        = sum of `accum_weights` (= sum α·T) per Gaussian
                                (used for cull criteria and as a sanity floor)
               - `count_vis`  = number of views where this Gaussian sits in the
                                per-view top-99% by importance (single-view detector)
               - `error_acc`  = sum of per-pixel L1 RGB error at pixels where this
                                Gaussian was the max contributor (the "I am the
                                dominant Gaussian at a pixel that is still wrong"
                                metric — used for donor sampling)
          2. Culls Gaussians via three criteria, ORed together:
               (a) `imp == 0` (no contribution at all)
               (b) `count_vis <= 1` (single-view → likely view-specific overfitting),
                    if `cull_single_view=True`
               (c) global low-importance (CDF-thresholded bottom of `imp`),
                    if `low_imp_cdf_thres < 1.0`
             Culling = snap `_opacity` below `dead_thresh`.
          3. Samples L = (#dead) donors from the alive pool with probability
             proportional to `error_acc + eps`. Bin-counts → caps at `max_per_donor`.
             High-importance Gaussians dominating high-error pixels get the most clones.
          4. Volume-preserving MCMC opacity/scale split via `_mcmc_update_params`.
          5. Donor state copied into the dead slots; Adam reset for both sets.
          6. Leftover dead slots (cap-clamp leftovers) stay dead → natural sparsification.

        Returns:
            dict with diagnostic counts.
        """
        N_total = self._xyz.shape[0]

        # ---------- Step 1: full-view sweep — importance + per-view visibility ----------
        # (The photometric error accumulation path is BENCHED — we now clone the
        # top-importance alive Gaussians deterministically, matching --mini v2's
        # `mini_culling_with_clone` aggressive-clone policy.)
        imp = torch.zeros(N_total, device="cuda")
        count_vis = torch.zeros(N_total, device="cuda")
        views = scene.getTrainCameras().copy()
        for view in views:
            render_pkg = render_fn(view, self, pipe, background, beta=beta,
                                   iteration=iteration, cfg=cfg, ingp=ingp,
                                   record_transmittance=True, is_training=False)
            accum_weights = render_pkg.get('accum_weights', None)
            if accum_weights is None:
                accum_weights = render_pkg.get('transmittance_avg', None)
            if accum_weights is None:
                del render_pkg
                continue

            accum_weights = accum_weights.squeeze()
            P = accum_weights.shape[0]
            if P < N_total:
                padded = torch.zeros(N_total, device="cuda")
                padded[:P] = accum_weights
                accum_weights = padded
            elif P > N_total:
                accum_weights = accum_weights[:N_total]

            imp += accum_weights
            vis_mask = self._cdf_mask(accum_weights, thres=0.99)
            count_vis[vis_mask] += 1
            del render_pkg
            torch.cuda.empty_cache()

        if imp.numel() != N_total:
            return {"n_total": N_total, "n_alive_before": 0,
                    "n_culled_zero": 0, "n_culled_single_view": 0,
                    "n_culled_low_imp": 0, "n_culled_total": 0,
                    "n_dead_in": 0, "n_candidates": 0,
                    "n_clones": 0, "n_natural_dead_left": 0}

        # ---------- Step 2: cull mask (zero-imp | single-view | low-CDF-imp) ----------
        target_dead_op = inverse_sigmoid(
            torch.tensor(dead_thresh * 0.5, device=self._opacity.device)
        )
        cur_op_pre = self.get_opacity.squeeze(-1)
        already_dead = (cur_op_pre <= dead_thresh)

        zero_imp = (imp <= 0) & (~already_dead)
        n_culled_zero = int(zero_imp.sum().item())

        if cull_single_view:
            single_view = (count_vis <= 1) & (~already_dead) & (~zero_imp)
            n_culled_single_view = int(single_view.sum().item())
        else:
            single_view = torch.zeros_like(zero_imp)
            n_culled_single_view = 0

        if 0.0 < low_imp_cdf_thres < 1.0:
            keep_mask = self._cdf_mask(imp, thres=low_imp_cdf_thres)
            low_imp = (~keep_mask) & (~already_dead) & (~zero_imp) & (~single_view)
            n_culled_low_imp = int(low_imp.sum().item())
        else:
            low_imp = torch.zeros_like(zero_imp)
            n_culled_low_imp = 0

        cull_mask = zero_imp | single_view | low_imp
        n_culled = int(cull_mask.sum().item())
        if n_culled > 0:
            self._opacity.data[cull_mask] = target_dead_op

        # Step 3: rebuild alive/dead masks after the cull.
        cur_op = self.get_opacity.squeeze(-1)
        dead_mask = (cur_op <= dead_thresh)
        alive_mask = ~dead_mask
        alive_idx = alive_mask.nonzero(as_tuple=True)[0]
        dead_idx = dead_mask.nonzero(as_tuple=True)[0]
        K = alive_idx.numel()
        L = dead_idx.numel()
        n_alive_before = K
        if L == 0 or K == 0:
            return {"n_total": N_total, "n_alive_before": n_alive_before,
                    "n_culled_zero": n_culled_zero,
                    "n_culled_single_view": n_culled_single_view,
                    "n_culled_low_imp": n_culled_low_imp,
                    "n_culled_total": n_culled, "n_dead_in": L,
                    "n_candidates": 0, "n_clones": 0, "n_natural_dead_left": L}

        # ---------- Step 4: build the clone candidate pool (top-CDF alive) ----------
        # Matches --mini v2's aggressive clone: everything responsible for the top
        # `clone_cdf_thres` fraction of cumulative importance is a clone donor.
        # Intersected with the post-cull alive set. With the default thres=0.99 this
        # is "every alive Gaussian except the weakest few by cumulative mass".
        alive_imp = imp[alive_idx]
        if 0.0 < low_imp_cdf_thres < 1.0:
            clone_cdf_keep = self._cdf_mask(imp, thres=low_imp_cdf_thres)
        else:
            clone_cdf_keep = torch.ones(N_total, dtype=torch.bool, device="cuda")
        candidate_global_mask = alive_mask & clone_cdf_keep
        candidate_idx = candidate_global_mask.nonzero(as_tuple=True)[0]
        n_candidates = int(candidate_idx.numel())
        if n_candidates == 0:
            return {"n_total": N_total, "n_alive_before": n_alive_before,
                    "n_culled_zero": n_culled_zero,
                    "n_culled_single_view": n_culled_single_view,
                    "n_culled_low_imp": n_culled_low_imp,
                    "n_culled_total": n_culled, "n_dead_in": L,
                    "n_candidates": 0, "n_clones": 0, "n_natural_dead_left": L}

        # ---------- Step 5: clone budget + select top-K candidates by importance ----------
        # Clone budget = min(5% of total, n_candidates, n_dead). Matches vanilla
        # MCMC's `add_new_gs` growth rate, but written into dead slots instead of
        # growing the tensor. Deterministic top-K selection (no sampling).
        growth_target = max(0, int(growth_frac * N_total))
        n_clones = min(growth_target, n_candidates, L)
        if n_clones == 0:
            return {"n_total": N_total, "n_alive_before": n_alive_before,
                    "n_culled_zero": n_culled_zero,
                    "n_culled_single_view": n_culled_single_view,
                    "n_culled_low_imp": n_culled_low_imp,
                    "n_culled_total": n_culled, "n_dead_in": L,
                    "n_candidates": n_candidates, "n_clones": 0,
                    "n_natural_dead_left": L}
        cand_imp = imp[candidate_idx]
        _, top_order = torch.topk(cand_imp, n_clones, largest=True, sorted=False)
        donor_global = candidate_idx[top_order]                       # [n_clones]
        target_dead = dead_idx[:n_clones]                             # [n_clones]

        # ---------- Step 6: MCMC volume-preserving opacity/scale split ----------
        # Each selected donor spawns exactly one clone (ratio=1 → N=2 particles).
        # α_new = 1 - (1 - α_old)^(1/2); scale_new follows matching covariance-
        # preserving formula. Computed via _mcmc_update_params → compute_relocation_cuda.
        ratio = torch.zeros(self._xyz.shape[0], 1, device="cuda", dtype=torch.int32)
        ratio[donor_global, 0] = 1  # one child per donor → N = 2
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
            new_shape,
        ) = self._mcmc_update_params(donor_global, ratio=ratio)

        # ---------- Step 7: Adam reset for donors (mass split + co-located sibling) ----------
        self._reset_optimizer_state_for_indices(donor_global)

        # ---------- Step 8: write donor opacity/scale in place (volume-preserving) ----------
        self._opacity.data[donor_global] = new_opacity
        self._scaling.data[donor_global] = new_scaling

        # ---------- Step 9: stamp clones into dead slots ----------
        # Donor state copied via _mcmc_update_params's return tuple; opacity/scale
        # already hold the volume-preserving split values.
        self._xyz.data[target_dead]          = new_xyz
        self._features_dc.data[target_dead]  = new_features_dc
        self._features_rest.data[target_dead] = new_features_rest
        self._opacity.data[target_dead]      = new_opacity
        self._scaling.data[target_dead]      = new_scaling
        self._rotation.data[target_dead]     = new_rotation
        self._appearance_level.data[target_dead] = new_ap_level

        if self._gaussian_feat_dim > 0 and new_gaussian_features is not None:
            self._gaussian_features.data[target_dead] = new_gaussian_features
        if self._adaptive_feat_dim > 0:
            if new_gamma is not None:
                self._gamma.data[target_dead] = new_gamma
            if new_adaptive_features is not None:
                self._adaptive_features.data[target_dead] = new_adaptive_features
        if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0 and new_adaptive_cat_weight is not None:
            self._adaptive_cat_weight.data[target_dead] = new_adaptive_cat_weight
        if hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0 and new_adaptive_zero_weight is not None:
            self._adaptive_zero_weight.data[target_dead] = new_adaptive_zero_weight
        if hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0 and new_gate_logits is not None:
            self._gate_logits.data[target_dead] = new_gate_logits
        if hasattr(self, '_shape') and self._shape.numel() > 0 and new_shape is not None:
            self._shape.data[target_dead] = new_shape

        # ---------- Step 10: Adam reset for the clone slots (brand-new values) ----------
        self._reset_optimizer_state_for_indices(target_dead)

        return {
            "n_total": N_total,
            "n_alive_before": n_alive_before,
            "n_culled_zero": n_culled_zero,
            "n_culled_single_view": n_culled_single_view,
            "n_culled_low_imp": n_culled_low_imp,
            "n_culled_total": n_culled,
            "n_dead_in": L,
            "n_candidates": n_candidates,
            "n_clones": n_clones,
            "n_natural_dead_left": L - n_clones,
        }

    # -- BENCHED (kept for reference) -------------------------------------------
    # Per-step photometric error accumulation flow. Replaced by the sweep-based
    # `minimc_sweep_and_relocate` above. The methods below are still functional
    # but not called from the train loop. Re-enable by reinstating
    # `accumulate_minimc_error` in train.py and dispatching `minimc_relocate_by_error`
    # / `minimc_cull_losers_to_dead_pool` separately.

    @torch.no_grad()
    def minimc_relocate_by_error(self, dead_mask, max_per_donor=3, eps=1e-8):
        """Phase-3 capped RJ-MCMC relocation: teleport dead Gaussians onto alive
        donors sampled proportionally to accumulated photometric error, with a
        strict per-donor clone cap. Uses the existing MCMC volume-preserving
        opacity/scale split kernel via `_mcmc_update_params`.

        Dead slots beyond the capped total-clone count remain dead (natural
        sparsification).

        Args:
            dead_mask: [N] bool. True where opacity <= dead_thresh.
            max_per_donor: int. Strict cap on clones produced by any single donor.
            eps: small floor added to error probs so zero-error alive donors can
                 still receive if the whole alive pool has zero accumulated error.

        Returns:
            total_clones (int): number of dead Gaussians that were moved.
        """
        if self.minimc_error_accum.numel() == 0:
            return 0
        alive_mask = ~dead_mask
        alive_idx = alive_mask.nonzero(as_tuple=True)[0]
        dead_idx = dead_mask.nonzero(as_tuple=True)[0]
        K = alive_idx.numel()
        L = dead_idx.numel()
        if L == 0 or K == 0:
            return 0

        # Sync minimc accumulators to current point count if something upstream resized.
        if self.minimc_error_accum.shape[0] != self._xyz.shape[0]:
            return 0

        # 1. Error-weighted donor distribution (eps floor so zero-error donors aren't forbidden).
        err = self.minimc_error_accum[alive_idx].squeeze(-1).clamp_min(0.0) + eps
        probs = err / err.sum()

        # 2. Sample L donors with replacement, bin-count, clamp per donor.
        sampled_local = torch.multinomial(probs, num_samples=L, replacement=True)  # [L] in [0, K)
        spawn = torch.bincount(sampled_local, minlength=K)                         # [K]
        spawn = spawn.clamp_max(max_per_donor)
        if not (spawn > 0).any():
            return 0
        # Donors that actually produce at least one clone (local indices into alive_idx).
        donor_local = (spawn > 0).nonzero(as_tuple=True)[0]
        donor_counts = spawn[donor_local]                                          # [D]
        total_clones = int(donor_counts.sum().item())
        if total_clones == 0:
            return 0

        # 3. Build the flat [total_clones] donor-global index list that drives
        #    _mcmc_update_params (one entry per clone). Duplicates are the donors
        #    that got multiple clones.
        donor_global = alive_idx[donor_local]                                      # [D]
        donor_repeat = torch.repeat_interleave(donor_global, donor_counts)         # [total_clones]

        # 4. Build the [N, 1] int32 `ratio` tensor expected by _mcmc_update_params.
        #    ratio[i, 0] = number of children donor i produces. Non-donors: 0.
        #    Inside _mcmc_update_params the volume split uses N = ratio[i] + 1 = clones + 1.
        ratio = torch.zeros(self._xyz.shape[0], 1, device="cuda", dtype=torch.int32)
        ratio[donor_global, 0] = donor_counts.to(torch.int32)

        # 5. Compute post-split params for every clone position (one entry per clone).
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
            new_shape,
        ) = self._mcmc_update_params(donor_repeat, ratio=ratio)

        # 6. Reset Adam state for donors first (they give away mass).
        self._reset_optimizer_state_for_indices(donor_global)

        # 7. Update donor opacity/scaling in place. Use per-donor post-split value
        #    (pull from the first clone of each donor in the flat tuple).
        donor_first_clone_offset = torch.cumsum(
            torch.cat([torch.zeros(1, device="cuda", dtype=donor_counts.dtype), donor_counts[:-1]]),
            dim=0
        ).long()
        self._opacity.data[donor_global] = new_opacity[donor_first_clone_offset]
        self._scaling.data[donor_global] = new_scaling[donor_first_clone_offset]

        # 8. Stamp every clone onto a dead slot (first `total_clones` of dead_idx).
        target_dead = dead_idx[:total_clones]
        self._xyz.data[target_dead] = new_xyz
        self._features_dc.data[target_dead] = new_features_dc
        self._features_rest.data[target_dead] = new_features_rest
        self._opacity.data[target_dead] = new_opacity
        self._scaling.data[target_dead] = new_scaling
        self._rotation.data[target_dead] = new_rotation
        self._appearance_level.data[target_dead] = new_ap_level

        if self._gaussian_feat_dim > 0 and new_gaussian_features is not None:
            self._gaussian_features.data[target_dead] = new_gaussian_features
        if self._adaptive_feat_dim > 0:
            if new_gamma is not None:
                self._gamma.data[target_dead] = new_gamma
            if new_adaptive_features is not None:
                self._adaptive_features.data[target_dead] = new_adaptive_features
        if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0 and new_adaptive_cat_weight is not None:
            self._adaptive_cat_weight.data[target_dead] = new_adaptive_cat_weight
        if hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0 and new_adaptive_zero_weight is not None:
            self._adaptive_zero_weight.data[target_dead] = new_adaptive_zero_weight
        if hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0 and new_gate_logits is not None:
            self._gate_logits.data[target_dead] = new_gate_logits
        if hasattr(self, '_shape') and self._shape.numel() > 0 and new_shape is not None:
            self._shape.data[target_dead] = new_shape

        # 9. Reset Adam state for the clones (they received brand-new values).
        self._reset_optimizer_state_for_indices(target_dead)

        # 10. Clear accumulators at touched indices so the next cycle starts fresh.
        self.minimc_error_accum[donor_global] = 0
        self.minimc_error_accum[target_dead] = 0
        self.minimc_win_count[donor_global] = 0
        self.minimc_win_count[target_dead] = 0

        return total_clones

    # ==================== Mini-Splatting v2 Methods ====================

    @staticmethod
    def _cdf_mask(importance, thres=0.99):
        """CDF-based importance mask: keep Gaussians in the top `thres` fraction of total importance.
        Matches MSv2 init_cdf_mask exactly."""
        importance = importance.flatten()
        if thres != 1.0:
            vals, idx = torch.sort(importance + 1e-6)
            cumsum_val = torch.cumsum(vals, dim=0)
            split_index = ((cumsum_val / vals.sum()) > (1 - thres)).nonzero().min()
            split_val = vals[split_index]
            non_prune_mask = importance > split_val
        else:
            non_prune_mask = torch.ones_like(importance, dtype=torch.bool)
        return non_prune_mask

    @torch.no_grad()
    def mini_compute_importance(self, scene, render_fn, pipe, background, beta, iteration, cfg, ingp=None, imp_metric="indoor"):
        """Render all training views and accumulate per-Gaussian importance scores.
        Matches MSv2's importance accumulation loop."""
        N = self._xyz.shape[0]
        imp_score = torch.zeros(N, device="cuda")
        views = scene.getTrainCameras().copy()

        for view in views:
            render_pkg = render_fn(view, self, pipe, background, beta=beta,
                                   iteration=iteration, cfg=cfg, ingp=ingp,
                                   record_transmittance=True, is_training=False)
            # accum_weights = sum(alpha*T) per Gaussian
            # Main rasterizer: separate 'accum_weights' key
            # diff_surfel_3D_sh_res: 'transmittance_avg' IS accum_weights (atomicAdd of w=alpha*T)
            accum_weights = render_pkg.get('accum_weights', None)
            if accum_weights is None:
                accum_weights = render_pkg.get('transmittance_avg', None)
            if accum_weights is None:
                continue

            accum_weights = accum_weights.squeeze()
            P = accum_weights.shape[0]
            if P < N:
                padded = torch.zeros(N, device="cuda")
                padded[:P] = accum_weights
                accum_weights = padded
            elif P > N:
                accum_weights = accum_weights[:N]

            imp_score += accum_weights
            del render_pkg
            torch.cuda.empty_cache()

        return imp_score

    @torch.no_grad()
    def mini_intersection_preserving(self, scene, render_fn, pipe, background, beta, iteration, cfg, ingp=None, imp_metric="indoor"):
        """CDF-based pruning keeping top 99% of importance. Matches MSv2 culling_with_interesction_preserving."""
        n_before = self._xyz.shape[0]
        imp_score = self.mini_compute_importance(scene, render_fn, pipe, background, beta, iteration, cfg, ingp, imp_metric)

        imp_score[imp_score == 0] = 0  # zero-importance Gaussians get pruned
        non_prune_mask = self._cdf_mask(imp_score, thres=0.99)

        prune_mask = ~non_prune_mask
        n_pruned = prune_mask.sum().item()
        if n_pruned > 0 and n_pruned < n_before:
            self.prune_points(prune_mask)
        return n_pruned

    @torch.no_grad()
    def mini_intersection_sampling(self, scene, render_fn, pipe, background, beta, iteration, cfg, ingp=None, imp_metric="indoor", sampling_factor=0.6):
        """Importance-weighted sampling keeping ~sampling_factor of Gaussians. Matches MSv2 culling_with_interesction_sampling."""
        n_before = self._xyz.shape[0]
        imp_score = self.mini_compute_importance(scene, render_fn, pipe, background, beta, iteration, cfg, ingp, imp_metric)

        imp_score[imp_score == 0] = 0
        prob = imp_score / (imp_score.sum() + 1e-8)
        prob_np = prob.cpu().numpy()

        N = self._xyz.shape[0]
        non_zero_frac = (prob_np != 0).sum() / prob_np.shape[0]
        num_sampled = int(N * sampling_factor * non_zero_frac)
        num_sampled = max(1, min(num_sampled, N - 1))

        indices = np.random.choice(N, size=num_sampled, p=prob_np, replace=False)
        non_prune_mask = np.zeros(N, dtype=bool)
        non_prune_mask[indices] = True

        prune_mask = torch.tensor(~non_prune_mask, device="cuda")
        n_pruned = prune_mask.sum().item()
        if n_pruned > 0 and n_pruned < n_before:
            self.prune_points(prune_mask)
        return n_pruned

    @torch.no_grad()
    def mini_clone(self, selected_pts_mask):
        """Clone selected Gaussians with opacity/scale adjustment. Matches MSv2 clone()."""
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        temp_opacity_old = self.get_opacity[selected_pts_mask]
        new_opacity = 1 - (1 - temp_opacity_old) ** 0.5

        temp_scale_old = self.get_scaling[selected_pts_mask]
        new_scaling = (temp_opacity_old / (2 * new_opacity - 0.5 ** 0.5 * new_opacity ** 2)) * temp_scale_old

        new_opacity = torch.clamp(new_opacity, max=1.0 - torch.finfo(torch.float32).eps, min=0.0051)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling)

        # Update originals in-place
        self._opacity.data[selected_pts_mask] = new_opacity
        self._scaling.data[selected_pts_mask] = new_scaling

        new_rotation = self._rotation[selected_pts_mask]

        # Per-Gaussian features for cat/fused modes
        new_gaussian_features = None
        if hasattr(self, '_gaussian_features') and self._gaussian_features.numel() > 0:
            new_gaussian_features = self._gaussian_features[selected_pts_mask]

        # CRITICAL: ap_level must be 24, NOT 0. See hashgrid.h: max_level = min(ap_level, L).
        # ap_level=0 → zero hash levels queried → dead hash gradients.
        new_ap_level = self._appearance_level[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacity, new_scaling, new_rotation,
                                   new_ap_level,
                                   new_gaussian_features=new_gaussian_features)

    @torch.no_grad()
    def mini_culling_with_clone(self, scene, render_fn, pipe, background, beta, iteration, cfg, ingp=None, imp_metric="indoor",
                                do_clone=True):
        """Aggressive cloning + single-view pruning. Mirrors MSv2 `culling_with_clone`:
        (1) full-view importance sweep, (2) prune `count_vis<=1 OR bottom-0.1% CDF`,
        (3) if `do_clone=True`, clone the top-99% CDF survivors with volume-preserving
        N=2 opacity/scale split (α_new = 1 − √(1 − α_old)), extending the tensor via
        `densification_postfix`. When `do_clone=False`, this is a prune-only operation
        and the count monotonically decreases.

        Returns (n_pruned, n_cloned) as a tuple.
        """
        N = self._xyz.shape[0]
        imp_score = torch.zeros(N, device="cuda")
        count_vis = torch.zeros(N, 1, device="cuda")
        count_rad = torch.zeros(N, 1, device="cuda")
        accum_area_max = torch.zeros(N, device="cuda")
        views = scene.getTrainCameras().copy()

        for view in views:
            render_pkg = render_fn(view, self, pipe, background, beta=beta,
                                   iteration=iteration, cfg=cfg, ingp=ingp,
                                   record_transmittance=True, is_training=False)
            accum_weights = render_pkg.get('accum_weights', None)
            if accum_weights is None:
                accum_weights = render_pkg.get('transmittance_avg', None)
            if accum_weights is None:
                continue

            accum_weights = accum_weights.squeeze()
            radii = render_pkg['radii']
            P = accum_weights.shape[0]
            L = min(N, P)

            imp_score[:L] += accum_weights[:L]
            count_rad[:L][radii[:L] > 0] += 1
            accum_area_max[:L] += (radii[:L] > 0).float()

            # Per-view: which Gaussians are in top 99% importance?
            padded_aw = torch.zeros(N, device="cuda")
            padded_aw[:L] = accum_weights[:L]
            vis_mask = self._cdf_mask(padded_aw, thres=0.99)
            count_vis[vis_mask] += 1

            del render_pkg
            torch.cuda.empty_cache()

        # Store factor_culling for gradient scaling during densification
        # Gaussians important in few views get amplified, well-covered ones get suppressed
        self.mini_factor_culling = count_vis / (count_rad + 1e-1)

        # Step 1: PRUNE — single-view Gaussians OR bottom-0.1% by cumulative importance.
        non_prune_mask = self._cdf_mask(imp_score, thres=0.999)
        prune_mask = (count_vis <= 1).squeeze()
        prune_mask = torch.logical_or(prune_mask, ~non_prune_mask)
        n_pruned = int(prune_mask.sum().item())

        if n_pruned <= 0 or n_pruned >= N:
            return n_pruned, 0

        # Compute clone mask on the ORIGINAL tensor ([N]). Zero-area Gaussians
        # are excluded from the clone pool (ms2 does the same: sets imp_score to 0
        # for `accum_area_max == 0`). Then intersect with survivors.
        imp_for_clone = imp_score.clone()
        imp_for_clone[accum_area_max == 0] = 0.0
        clone_mask_full = self._cdf_mask(imp_for_clone, thres=0.99)   # [N]

        # Step 2: PRUNE now. After this, tensor is shape [N - n_pruned].
        survivor_clone_mask = clone_mask_full[~prune_mask]            # [N - n_pruned]
        self.prune_points(prune_mask)

        # factor_culling was already pruned inside prune_points; extend below after cloning.

        # Step 3: CLONE — volume-preserving N=2 split of every survivor in clone_mask.
        # Skipped entirely when do_clone=False → this function becomes aggressive prune-only.
        if not do_clone:
            return n_pruned, 0
        n_clone = int(survivor_clone_mask.sum().item())
        if n_clone == 0:
            return n_pruned, 0

        # Volume-preserving opacity/scale split: α_new = 1 − √(1 − α_old).
        temp_opacity_old = self.get_opacity[survivor_clone_mask]                      # [K, 1]
        new_opacity_activated = 1.0 - torch.pow(1.0 - temp_opacity_old, 0.5)
        new_opacity_activated = torch.clamp(new_opacity_activated,
                                            max=1.0 - torch.finfo(torch.float32).eps,
                                            min=0.0051)
        # Scale correction (same as ms2 clone, reloc_utils N=2 closed form):
        # scale_new = α_old / (2*α_new − √0.5 * α_new²) * scale_old
        temp_scale_old = self.get_scaling[survivor_clone_mask]                        # [K, 2]
        denom = (2.0 * new_opacity_activated
                 - (0.5 ** 0.5) * new_opacity_activated * new_opacity_activated)
        new_scaling_activated = (temp_opacity_old / denom) * temp_scale_old
        new_opacity_raw = self.inverse_opacity_activation(new_opacity_activated)     # [K, 1]
        new_scaling_raw = self.scaling_inverse_activation(new_scaling_activated)     # [K, 2]

        # Update donor slots in place (donor and clone share the new value).
        self._opacity.data[survivor_clone_mask] = new_opacity_raw
        self._scaling.data[survivor_clone_mask] = new_scaling_raw

        # Clone fields to append via densification_postfix.
        new_xyz = self._xyz[survivor_clone_mask]
        new_features_dc = self._features_dc[survivor_clone_mask]
        new_features_rest = self._features_rest[survivor_clone_mask]
        new_rotation = self._rotation[survivor_clone_mask]
        new_ap_level = self._appearance_level[survivor_clone_mask]
        new_opacity_cat = new_opacity_raw
        new_scaling_cat = new_scaling_raw

        new_gaussian_features = None
        if self._gaussian_feat_dim > 0 and self._gaussian_features.numel() > 0:
            new_gaussian_features = self._gaussian_features[survivor_clone_mask]

        new_gamma = None
        new_adaptive_features = None
        if self._adaptive_feat_dim > 0:
            if self._gamma.numel() > 0:
                new_gamma = self._gamma[survivor_clone_mask]
            if self._adaptive_features.numel() > 0:
                new_adaptive_features = self._adaptive_features[survivor_clone_mask]

        new_adaptive_cat_weight = None
        if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0:
            new_adaptive_cat_weight = self._adaptive_cat_weight[survivor_clone_mask]
        new_adaptive_zero_weight = None
        if hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0:
            new_adaptive_zero_weight = self._adaptive_zero_weight[survivor_clone_mask]
        new_gate_logits = None
        if hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0:
            new_gate_logits = self._gate_logits[survivor_clone_mask]
        new_shape = None
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            new_shape = self._shape[survivor_clone_mask]
        new_flex_beta = None
        if hasattr(self, '_flex_beta') and self._flex_beta.numel() > 0:
            new_flex_beta = self._flex_beta[survivor_clone_mask]
        new_sb_params = None
        if self._sb_params.numel() > 0:
            new_sb_params = self._sb_params[survivor_clone_mask]
        new_sv_sites_mc = None
        new_sv_colors_mc = None
        if self._sv_sites.numel() > 0:
            new_sv_sites_mc = self._sv_sites[survivor_clone_mask]
            new_sv_colors_mc = self._sv_colors[survivor_clone_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity_cat,
            new_scaling_cat, new_rotation, new_ap_level,
            new_gaussian_features, new_gamma, new_adaptive_features,
            new_adaptive_cat_weight, new_adaptive_zero_weight, new_gate_logits,
            new_shape, new_flex_beta, new_sb_params,
            None, None, None,  # no SG here
            new_sv_sites_mc, new_sv_colors_mc,
        )

        # Extend factor_culling with the cloned donors' factors so downstream
        # gradient scaling keeps matching the new tensor shape.
        if self.mini_factor_culling is not None and self.mini_factor_culling.shape[0] == self._xyz.shape[0] - n_clone:
            # factor_culling still matches the post-prune size; append the cloned ones.
            cloned_factor = self.mini_factor_culling[survivor_clone_mask]
            self.mini_factor_culling = torch.cat([self.mini_factor_culling, cloned_factor])

        return n_pruned, n_clone

    @staticmethod
    def _compute_safe_radius_world(depth, alpha, normal, fx, fy,
                                    rel_depth_thresh=0.05,
                                    alpha_thresh=0.01,
                                    use_normal_edge=False,
                                    normal_dot_thresh=0.70):
        """Per-pixel world-space safe radius: distance to the nearest silhouette /
        depth-discontinuity / alpha edge, converted to world units via depth·(1/focal).

        Loose variant (used by --minispa): normal-crease edges OFF by default,
        depth threshold 5%, alpha threshold 0.01. Interior points will typically
        have huge safe radii (unclamped); only points near silhouettes get bound.
        """
        depth = depth.squeeze()
        alpha = alpha.squeeze()
        H, W = depth.shape
        device = depth.device

        dz_x = torch.abs(depth[:, 1:] - depth[:, :-1])
        dz_y = torch.abs(depth[1:, :] - depth[:-1, :])
        dep_x = depth[:, :-1].clamp_min(1e-6)
        dep_y = depth[:-1, :].clamp_min(1e-6)
        rel_x = dz_x > rel_depth_thresh * dep_x
        rel_y = dz_y > rel_depth_thresh * dep_y
        depth_edge = torch.zeros_like(depth, dtype=torch.bool)
        depth_edge[:, :-1] |= rel_x
        depth_edge[:, 1:]  |= rel_x
        depth_edge[:-1, :] |= rel_y
        depth_edge[1:, :]  |= rel_y

        if use_normal_edge and normal is not None:
            n_dot_x = (normal[:, :, 1:] * normal[:, :, :-1]).sum(dim=0)
            n_dot_y = (normal[:, 1:, :] * normal[:, :-1, :]).sum(dim=0)
            n_edge_x = n_dot_x < normal_dot_thresh
            n_edge_y = n_dot_y < normal_dot_thresh
            normal_edge = torch.zeros_like(depth, dtype=torch.bool)
            normal_edge[:, :-1] |= n_edge_x
            normal_edge[:, 1:]  |= n_edge_x
            normal_edge[:-1, :] |= n_edge_y
            normal_edge[1:, :]  |= n_edge_y
        else:
            normal_edge = torch.zeros_like(depth, dtype=torch.bool)

        alpha_edge = alpha < alpha_thresh
        boundary = depth_edge | normal_edge | alpha_edge

        try:
            from scipy.ndimage import distance_transform_edt
            dist_px = distance_transform_edt((~boundary).cpu().numpy())
            dist_px = torch.from_numpy(dist_px).to(device=device, dtype=torch.float32)
        except ImportError:
            import torch.nn.functional as F
            dist_px = torch.zeros_like(depth, dtype=torch.float32)
            frontier = boundary.clone()
            for step in range(1, 21):
                grown = F.max_pool2d(frontier.float().unsqueeze(0).unsqueeze(0),
                                      kernel_size=3, stride=1, padding=1)[0, 0] > 0
                newly_hit = grown & ~frontier
                dist_px[newly_hit] = step
                frontier = grown
                if frontier.all():
                    break
            dist_px[~boundary] = dist_px[~boundary].clamp_min(1.0)

        inv_focal = max(1.0 / float(fx), 1.0 / float(fy))
        safe_radius_world = dist_px * depth * inv_focal
        return safe_radius_world

    def mini_depth_reinit(self, depth_maps, alpha_maps, viewpoint_cameras, gt_images=None, normal_maps=None, num_total_views=None,
                          max_idx_maps=None, src_features_dc=None, src_features_rest=None,
                          total_count_override=None, compute_safe_radius=False,
                          safe_radius_rel_depth_thresh=0.05,
                          safe_radius_use_normal_edge=False):
        """Collect 3D points from rendered depth maps for reinitialization.

        Uses max-contributor depth for crisp surface placement.
        Uniform sampling from valid pixels.
        Returns xyz, GT colors, per-pixel rendered normals, and (if available) the
        full SH coefficients of the max-weight Gaussian sampled at each reinit pixel.

        Args:
            depth_maps: list of [1, H, W] depth tensors (depth_max_contributor from render)
            alpha_maps: list of [1, H, W] alpha tensors (rend_alpha from render)
            viewpoint_cameras: list of camera objects
            gt_images: list of [3, H, W] GT image tensors (for color init)
            normal_maps: list of [3, H, W] rendered normal tensors (world space, for surfel orientation)
            num_total_views: total number of views (for budget calculation when called per-view)
            max_idx_maps: optional list of [1, H, W] (or [H, W]) int32 per-pixel max-contributor
                          Gaussian ids (from rasterizer's out_index buffer). When provided alongside
                          src_features_dc/src_features_rest, the SH of the max-weight Gaussian at
                          each sampled pixel is harvested into 'sh_dc' / 'sh_rest'.
            src_features_dc:   optional [N, 1, K] snapshot of OLD _features_dc to index into
            src_features_rest: optional [N, R, K] snapshot of OLD _features_rest to index into
        Returns:
            dict with 'xyz' [M, 3], 'colors' [M, 3] (RGB 0-1), 'normals' [M, 3],
            and optionally 'sh_dc' [M, 1, K], 'sh_rest' [M, R, K]
            or None if no valid points
        """
        all_pts = []
        all_colors = []
        all_normals = []
        all_sh_dc = []
        all_sh_rest = []
        all_footprints = []  # per-point world-space size of one pixel at that point's depth
        all_safe_r = []      # optional per-point silhouette-aware safe radius (world units)
        # Whether we will harvest SH from the max-contributor id
        do_sh_transfer = (max_idx_maps is not None
                          and src_features_dc is not None
                          and src_features_rest is not None)
        # Budget: by default match the current tensor size. Callers can override
        # (e.g. --minimc reinit passes only the alive Gaussian count so the reinit
        # doesn't spawn into the full cap_max budget).
        N_total = total_count_override if total_count_override is not None else len(self._xyz)
        n_views = num_total_views if num_total_views is not None else len(viewpoint_cameras)

        for i, (depth, alpha, cam) in enumerate(zip(depth_maps, alpha_maps, viewpoint_cameras)):
            H, W = cam.image_height, cam.image_width

            # Sample from pixels with valid depth, weighted by (1 - alpha) to fill holes
            alpha_flat = alpha.squeeze().reshape(-1)
            depth_flat = depth.squeeze().reshape(-1)
            valid_mask = (alpha_flat > 0.01) & (depth_flat > 0.01)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            if valid_indices.shape[0] == 0:
                continue

            num_sample = max(1, int(N_total / n_views))
            num_sample = min(num_sample, valid_indices.shape[0])

            # MSv2: weight by 1-alpha so uncovered regions get more samples
            prob = (1.0 - alpha_flat[valid_indices]).clamp(min=0)
            prob_sum = prob.sum()
            if prob_sum > 0:
                prob = prob / prob_sum
                sampled = torch.multinomial(prob, num_sample, replacement=False)
            else:
                sampled = torch.randperm(valid_indices.shape[0], device="cuda")[:num_sample]
            indices = valid_indices[sampled]

            iy = (indices // W).float()
            ix = (indices % W).float()
            d = depth.squeeze().reshape(-1)[indices]

            # Camera intrinsics
            if hasattr(cam, 'focal_x'):
                fx, fy = cam.focal_x, cam.focal_y
                cx, cy = W / 2.0, H / 2.0
            else:
                fx = W / (2.0 * np.tan(cam.FoVx / 2.0))
                fy = H / (2.0 * np.tan(cam.FoVy / 2.0))
                cx, cy = W / 2.0, H / 2.0

            # Pixel to camera space
            x_cam = (ix - cx) / fx * d
            y_cam = (iy - cy) / fy * d
            z_cam = d
            pts_cam = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1)

            # Camera to world
            w2c = cam.world_view_transform.T
            c2w = torch.inverse(w2c)
            pts_world = (c2w @ pts_cam.T).T[:, :3]

            all_pts.append(pts_world)

            # Per-pixel world-space footprint: size in world units of one pixel
            # at this point's depth. Used by reinit scale init to prevent NN-distance
            # ballooning in sparse regions.
            #    footprint = depth / focal   (world units per pixel)
            # We use the larger of (1/fx, 1/fy) for isotropic surfel scale init.
            inv_focal = max(1.0 / float(fx), 1.0 / float(fy))
            all_footprints.append(d * inv_focal)

            # Optional silhouette-aware safe radius (loose variant).
            # Computed per-view on the full image, then sampled at the chosen pixels.
            if compute_safe_radius and normal_maps is not None and i < len(normal_maps):
                safe_r_world = self._compute_safe_radius_world(
                    depth, alpha, normal_maps[i], fx, fy,
                    rel_depth_thresh=safe_radius_rel_depth_thresh,
                    use_normal_edge=safe_radius_use_normal_edge,
                )
                all_safe_r.append(safe_r_world.reshape(-1)[indices])

            # Sample GT colors at these pixels
            if gt_images is not None and i < len(gt_images):
                gt = gt_images[i]  # [3, H, W]
                colors = gt.reshape(3, -1)[:, indices].T  # [M, 3]
                all_colors.append(colors)

            # Use rendered per-pixel normals if available, else fall back to camera-facing
            if normal_maps is not None and i < len(normal_maps):
                nmap = normal_maps[i]  # [3, H, W] world-space normals
                normals = nmap.reshape(3, -1)[:, indices].T  # [M, 3]
                normals = torch.nn.functional.normalize(normals, dim=-1)
            else:
                cam_pos = cam.camera_center.cuda()
                normals = cam_pos.unsqueeze(0) - pts_world
                normals = torch.nn.functional.normalize(normals, dim=-1)
            all_normals.append(normals)

            # Harvest the max-weight Gaussian's full SH at each sampled pixel.
            # The rasterizer wrote -1 where no Gaussian contributed; we mask those out
            # and fall back to RGB2SH(GT) downstream for the missing rows.
            if do_sh_transfer and i < len(max_idx_maps):
                idx_map = max_idx_maps[i].to(indices.device).long().reshape(-1)  # [H*W]
                pixel_max_idx = idx_map[indices]  # [M]
                N_src = src_features_dc.shape[0]
                valid = (pixel_max_idx >= 0) & (pixel_max_idx < N_src)
                # Clamp to keep the index lookup safe; we overwrite invalid rows next.
                safe_idx = pixel_max_idx.clamp(min=0, max=max(N_src - 1, 0))
                sh_dc_view = src_features_dc[safe_idx]    # [M, 1, K]
                sh_rest_view = src_features_rest[safe_idx]  # [M, R, K]
                # Where there was no contributor, fall back to RGB2SH(GT pixel).
                if (~valid).any():
                    if gt_images is not None and i < len(gt_images):
                        fallback_rgb = gt_images[i].reshape(3, -1)[:, indices].T  # [M, 3]
                        fallback_dc = RGB2SH(fallback_rgb)  # [M, 3]
                        # Place into [M, 1, 3] DC slot, leave rest at 0 for invalid rows.
                        sh_dc_view = sh_dc_view.clone()
                        sh_dc_view[~valid] = fallback_dc[~valid].unsqueeze(1)
                        sh_rest_view = sh_rest_view.clone()
                        sh_rest_view[~valid] = 0.0
                    else:
                        sh_dc_view = sh_dc_view.clone()
                        sh_dc_view[~valid] = 0.0
                        sh_rest_view = sh_rest_view.clone()
                        sh_rest_view[~valid] = 0.0
                all_sh_dc.append(sh_dc_view)
                all_sh_rest.append(sh_rest_view)

        if len(all_pts) == 0:
            return None

        result = {'xyz': torch.cat(all_pts, dim=0)}
        if all_colors:
            result['colors'] = torch.cat(all_colors, dim=0)
        if all_normals:
            result['normals'] = torch.cat(all_normals, dim=0)
        if all_sh_dc:
            result['sh_dc'] = torch.cat(all_sh_dc, dim=0)
        if all_sh_rest:
            result['sh_rest'] = torch.cat(all_sh_rest, dim=0)
        if all_footprints:
            result['pixel_footprint'] = torch.cat(all_footprints, dim=0)  # [M] world units
        if all_safe_r:
            result['safe_radius'] = torch.cat(all_safe_r, dim=0)  # [M] world units
        return result

    @staticmethod
    def _normal_to_quaternion(normals):
        """Convert normal vectors to quaternions where surfel z-axis aligns with normal.

        Args:
            normals: [M, 3] unit normal vectors
        Returns:
            quaternions: [M, 4] (w, x, y, z)
        """
        # Target: rotate [0, 0, 1] to match normal direction
        # Using the half-vector quaternion formula: q = [1 + dot(z, n), cross(z, n)]
        z = torch.tensor([0.0, 0.0, 1.0], device=normals.device)
        dot = normals[:, 2]  # dot([0,0,1], normal) = normal.z
        cross_x = -normals[:, 1]  # cross([0,0,1], n) = [-n.y, n.x, 0]
        cross_y = normals[:, 0]

        w = 1.0 + dot  # [M]
        quat = torch.stack([w, cross_x, cross_y, torch.zeros_like(w)], dim=-1)  # [M, 4]

        # Handle anti-parallel case (normal ≈ [0, 0, -1])
        anti = (dot < -0.999)
        if anti.any():
            quat[anti] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=normals.device)

        # Normalize
        quat = torch.nn.functional.normalize(quat, dim=-1)
        return quat

    @torch.no_grad()
    def minimc_despawn_non_contributors(self, scene, render_fn, pipe, background, beta,
                                        iteration, cfg, ingp=None, dead_thresh=0.005):
        """Sweep every training view, find the union of per-pixel max-contributors
        across all views, and despawn every alive Gaussian that is NOT in that
        union into the MCMC dead pool (opacity snapped below `dead_thresh`).

        Max contributors are preserved exactly — no reinit, no opacity change.
        Non-contributor alives become dead and will be re-cloned by the next
        Phase-B sweep+clone step. The tensor size is invariant.

        Returns:
            dict with n_total, n_alive_before, n_winners, n_despawned.
        """
        N_total = self._xyz.shape[0]
        if N_total == 0:
            return {"n_total": 0, "n_alive_before": 0, "n_winners": 0, "n_despawned": 0}

        # Boolean mask: True for any Gaussian that was the max contributor at
        # ≥1 pixel across the full training set.
        has_won = torch.zeros(N_total, dtype=torch.bool, device="cuda")
        views = scene.getTrainCameras().copy()
        for view in views:
            render_pkg = render_fn(view, self, pipe, background, beta=beta,
                                   iteration=iteration, cfg=cfg, ingp=ingp,
                                   record_transmittance=False, is_training=False)
            max_idx_map = render_pkg.get('max_contrib_idx', None)
            if max_idx_map is None:
                del render_pkg
                continue
            ids = max_idx_map.long().reshape(-1)
            valid = (ids >= 0) & (ids < N_total)
            if valid.any():
                v_ids = ids[valid].unique()
                has_won[v_ids] = True
            del render_pkg
            torch.cuda.empty_cache()

        cur_op = self.get_opacity.squeeze(-1)
        alive_mask = (cur_op > dead_thresh)
        n_alive_before = int(alive_mask.sum().item())
        n_winners = int((alive_mask & has_won).sum().item())

        # Despawn: alive AND never won a pixel → set opacity below dead_thresh.
        loser_mask = alive_mask & (~has_won)
        n_despawned = int(loser_mask.sum().item())
        if n_despawned > 0:
            target_op = inverse_sigmoid(
                torch.tensor(dead_thresh * 0.5, device=self._opacity.device)
            )
            self._opacity.data[loser_mask] = target_op
            # Reset Adam state for the losers (they're being moved to the dead pool).
            loser_idx = loser_mask.nonzero(as_tuple=True)[0]
            self._reset_optimizer_state_for_indices(loser_idx)

        return {
            "n_total": N_total,
            "n_alive_before": n_alive_before,
            "n_winners": n_winners,
            "n_despawned": n_despawned,
        }

    @torch.no_grad()
    def reinitial_alive_inplace(self, reinit_data, alive_indices, new_opacity_value=0.8,
                                footprint_scale_cap=4.0):
        """Reinitialize a subset of the alive Gaussians in-place from depth-sampled
        points, leaving all other slots (including the dead pool) untouched.
        Used by `--minimc` to refresh the scene without destroying the MCMC budget.

        The number of slots written is `min(M, K)` where M = reinit point count
        and K = len(alive_indices). If the reinit sweep under-produces (M < K),
        the trailing `K - M` alive slots stay as they were — no padding, no
        duplication. This preserves the "don't exceed alive count, don't shrink
        total budget" invariant.

        Args:
            reinit_data: dict with 'xyz' [M, 3], optionally 'colors' [M, 3],
                         'normals' [M, 3], 'sh_dc' [M, 1, 3], 'sh_rest' [M, R, 3].
            alive_indices: [K] long tensor of row indices into this model's
                           parameter tensors. Only the first min(M, K) of these
                           are overwritten; any remaining alive slots are untouched.
            new_opacity_value: scalar opacity for refreshed Gaussians (default 0.8).
        """
        new_xyz = reinit_data['xyz']
        new_colors = reinit_data.get('colors', None)
        new_normals = reinit_data.get('normals', None)
        new_sh_dc = reinit_data.get('sh_dc', None)
        new_sh_rest = reinit_data.get('sh_rest', None)
        new_footprint = reinit_data.get('pixel_footprint', None)

        if alive_indices.numel() == 0 or new_xyz is None or new_xyz.shape[0] == 0:
            return

        M = int(new_xyz.shape[0])
        K_alive = int(alive_indices.numel())
        K = min(M, K_alive)
        if K == 0:
            return

        # Slice (not pad) the reinit data to K rows.
        def _slice(t, K):
            if t is None:
                return None
            t = t.to("cuda")
            return t[:K]

        new_xyz    = _slice(new_xyz, K)
        new_colors = _slice(new_colors, K)
        new_normals = _slice(new_normals, K)
        new_sh_dc  = _slice(new_sh_dc, K)
        new_sh_rest = _slice(new_sh_rest, K)
        new_footprint = _slice(new_footprint, K)

        # Use the first K alive slots (arbitrary choice — any subset of K alive
        # indices would work since "we don't need the same exact Gaussians").
        alive_indices = alive_indices[:K]

        # Build SH features for the K points (same preference order as reinitial_from_depth):
        # 1) sh_dc/sh_rest transfer → 2) RGB2SH(colors) → 3) zeros.
        features = torch.zeros((K, 3, (self.max_sh_degree + 1) ** 2), device="cuda")
        if new_sh_dc is not None and new_sh_dc.shape[0] == K:
            features[:, :, 0:1] = new_sh_dc.float().transpose(1, 2)
            if new_sh_rest is not None and new_sh_rest.shape[0] == K and new_sh_rest.shape[1] > 0:
                R = min(new_sh_rest.shape[1], features.shape[2] - 1)
                features[:, :, 1:1 + R] = new_sh_rest[:, :R].float().transpose(1, 2)
        elif new_colors is not None and new_colors.shape[0] == K:
            features[:, :, 0] = RGB2SH(new_colors.float())

        # NN-distance scales, clamped by pixel footprint to prevent dilation in
        # sparse regions. scale = min(nn_dist, footprint_scale_cap * pixel_footprint)
        dist2 = torch.clamp_min(distCUDA2(new_xyz), 0.0000001)
        nn_dist = torch.sqrt(dist2)
        if new_footprint is not None and new_footprint.shape[0] == K:
            cap = footprint_scale_cap * new_footprint.to(nn_dist.device).float()
            nn_dist = torch.minimum(nn_dist, cap.clamp_min(1e-7))
        log_scales = torch.log(nn_dist)[..., None].repeat(1, 2)  # [K, 2]

        # Rotations from normals (camera-facing via quaternion), random fallback.
        if new_normals is not None:
            rots = self._normal_to_quaternion(new_normals)
        else:
            rots = torch.rand(K, 4, device="cuda")

        new_op_logit = inverse_sigmoid(
            torch.tensor(float(new_opacity_value), device="cuda")
        ).expand(K, 1).clone()

        # In-place writes to alive slots (dead slots are untouched).
        ai = alive_indices.to(self._xyz.device).long()
        self._xyz.data[ai] = new_xyz
        # _features_dc is [N, 1, 3] = features[:, :, 0:1] transposed.
        self._features_dc.data[ai] = features[:, :, 0:1].transpose(1, 2).contiguous()
        self._features_rest.data[ai] = features[:, :, 1:].transpose(1, 2).contiguous()
        self._scaling.data[ai] = log_scales
        self._rotation.data[ai] = rots
        self._opacity.data[ai] = new_op_logit
        if self._appearance_level.numel() > 0:
            self._appearance_level.data[ai] = torch.full((K, 1), 24.0,
                                                         device=self._appearance_level.device)

        # Reset auxiliary per-kernel state for the refreshed slots (so they start
        # from the fully-Gaussian-like shape, matching reinitial_from_depth).
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            if self.kernel_type == "general":
                self._shape.data[ai] = -10.0
            else:
                self._shape.data[ai] = 1.386  # → β ≈ 4.0

        # Zero the per-Gaussian learnable feature banks at alive slots (cold start).
        if hasattr(self, '_gaussian_features') and self._gaussian_features.numel() > 0:
            self._gaussian_features.data[ai] = 0.0
        if hasattr(self, '_gamma') and self._gamma.numel() > 0:
            self._gamma.data[ai] = 0.0
        if hasattr(self, '_adaptive_features') and self._adaptive_features.numel() > 0:
            self._adaptive_features.data[ai] = 0.0
        if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0:
            self._adaptive_cat_weight.data[ai] = 0.0
        if hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0:
            self._adaptive_zero_weight.data[ai] = 0.0
        if hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0:
            self._gate_logits.data[ai] = 0.0

        # Reset Adam moments only for the refreshed slots.
        self._reset_optimizer_state_for_indices(ai)

        # Zero the per-Gaussian accumulators so the fresh cohort starts clean.
        if self.xyz_gradient_accum.numel() > 0:
            self.xyz_gradient_accum[ai] = 0.0
        if self.xyz_gradient_accum_abs.numel() > 0:
            self.xyz_gradient_accum_abs[ai] = 0.0
        if self.feat_gradient_accum.numel() > 0:
            self.feat_gradient_accum[ai] = 0.0
        if self.denom.numel() > 0:
            self.denom[ai] = 0.0
        if self.max_radii2D.numel() > 0:
            self.max_radii2D[ai] = 0.0
        if self.minimc_error_accum.numel() > 0:
            self.minimc_error_accum[ai] = 0.0
            self.minimc_win_count[ai] = 0.0

    @torch.no_grad()
    def reinitial_from_depth(self, reinit_data):
        """Reinitialize all Gaussians from depth-sampled 3D points.

        Args:
            reinit_data: dict with 'xyz' [M, 3], optionally 'colors' [M, 3] (RGB 0-1),
                         'normals' [M, 3] (unit camera-facing normals)
                         OR a plain [M, 3] tensor (legacy: xyz only)
        """
        if isinstance(reinit_data, dict):
            new_xyz = reinit_data['xyz']
            new_colors = reinit_data.get('colors', None)
            new_normals = reinit_data.get('normals', None)
            new_sh_dc = reinit_data.get('sh_dc', None)
            new_sh_rest = reinit_data.get('sh_rest', None)
            new_footprint = reinit_data.get('pixel_footprint', None)
            new_safe_radius = reinit_data.get('safe_radius', None)
            # Optional knobs used by the mesh-reinit path to reduce overshoot:
            #   scale_factor: multiply NN-distance before log (default 1.0)
            #   init_opacity: starting opacity (default 0.8)
            scale_factor = float(reinit_data.get('scale_factor', 1.0))
            init_opacity = float(reinit_data.get('init_opacity', 0.8))
        else:
            new_xyz = reinit_data
            new_colors = None
            new_normals = None
            new_sh_dc = None
            new_sh_rest = None
            new_footprint = None
            new_safe_radius = None
            scale_factor = 1.0
            init_opacity = 0.8

        M = new_xyz.shape[0]
        # How many pixels of slop we allow per surfel in sparse regions.
        # scale = min(nn_dist, footprint_scale_cap * pixel_footprint)
        footprint_scale_cap = 4.0

        # Initialize SH features. Preference order:
        #   1) sh_dc / sh_rest sampled from the OLD max-weight Gaussian per pixel (full SH transfer)
        #   2) per-pixel GT RGB → DC SH (DC-only fallback)
        #   3) all zeros (cold start)
        # 'features' is [M, 3, (deg+1)^2] in CHANNEL × ORDER layout used by the model.
        features = torch.zeros((M, 3, (self.max_sh_degree + 1) ** 2), device="cuda")
        if new_sh_dc is not None and new_sh_dc.shape[0] == M:
            # _features_dc layout is [N, 1, 3]; transpose to [N, 3, 1] to match `features`.
            features[:, :, 0:1] = new_sh_dc.to(features.device).float().transpose(1, 2)
            if new_sh_rest is not None and new_sh_rest.shape[0] == M and new_sh_rest.shape[1] > 0:
                # _features_rest layout is [N, R, 3]; transpose to [N, 3, R].
                R = min(new_sh_rest.shape[1], features.shape[2] - 1)
                features[:, :, 1:1 + R] = new_sh_rest[:, :R].to(features.device).float().transpose(1, 2)
        elif new_colors is not None and new_colors.shape[0] == M:
            # new_colors is [M, 3] in [0, 1] RGB; convert to SH DC convention used by this codebase.
            features[:, :, 0] = RGB2SH(new_colors.to(features.device).float())

        # Per-point NN-distance scaling, clamped by the pixel-footprint at each
        # point's depth so sparse regions don't explode to the nearest-neighbor
        # distance. Falls back to raw NN when no per-point footprint is provided.
        dist2 = torch.clamp_min(distCUDA2(new_xyz), 0.0000001)
        nn_dist = torch.sqrt(dist2)
        if new_footprint is not None and new_footprint.shape[0] == M:
            cap = footprint_scale_cap * new_footprint.to(nn_dist.device).float()
            nn_dist = torch.minimum(nn_dist, cap.clamp_min(1e-7))
        # Silhouette-aware clamp (loose variant, used by --minispa): distance to
        # the nearest depth/alpha edge in world units. Interior points typically
        # see a large safe radius (unclamped); silhouette points get pulled in.
        if new_safe_radius is not None and new_safe_radius.shape[0] == M:
            sr = new_safe_radius.to(nn_dist.device).float().clamp_min(1e-7)
            nn_dist = torch.minimum(nn_dist, sr)
        if scale_factor != 1.0:
            nn_dist = nn_dist * scale_factor
        log_scales = torch.log(nn_dist.clamp_min(1e-7))[..., None].repeat(1, 2)

        # Camera-facing rotations (surfels face the camera they were unprojected from)
        if new_normals is not None:
            rots = self._normal_to_quaternion(new_normals)
        else:
            rots = torch.rand(M, 4, device="cuda")

        opacities = torch.logit(torch.ones(M, 1, device="cuda") * init_opacity)
        # CRITICAL: ap_level controls how many hash levels are queried per Gaussian.
        # In hashgrid.h: max_level = min(appearance_level, L).
        # If ap_level=0, zero hash levels are queried → hash features are all zeros → zero hash gradients.
        # Must be set to a value >= num_hash_levels (24 is the "disable C2F" sentinel from create_from_pcd).
        ap_levels = torch.ones(M, 1, device="cuda") * 24

        # Reset all parameters
        self._xyz = nn.Parameter(new_xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(log_scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._appearance_level = nn.Parameter(ap_levels.requires_grad_(False))

        # Reset per-gaussian features if they exist
        if hasattr(self, '_gaussian_features') and self._gaussian_features.numel() > 0:
            feat_dim = self._gaussian_features.shape[1]
            self._gaussian_features = nn.Parameter(torch.zeros(M, feat_dim, device="cuda").requires_grad_(True))
        if hasattr(self, '_gamma') and self._gamma.numel() > 0:
            self._gamma = nn.Parameter(torch.zeros(M, self._gamma.shape[1], device="cuda").requires_grad_(True))
        if hasattr(self, '_adaptive_features') and self._adaptive_features.numel() > 0:
            self._adaptive_features = nn.Parameter(torch.zeros(M, self._adaptive_features.shape[1], device="cuda").requires_grad_(True))

        # Reinitialize the per-Gaussian shape parameter to a fully-Gaussian-like value.
        # beta / beta_scaled (β = sigmoid(_shape) * 5): β = 4.0 is "Gaussian-like" → _shape = logit(0.8) ≈ 1.386.
        # general kernel (β = sigmoid(_shape) * 6 + 2, range [2, 8]): β = 2.0 is the standard Gaussian
        # → sigmoid(_shape) = 0 → use a large negative shape (-10) to clamp the activation near β=2.
        if hasattr(self, '_shape') and self._shape.numel() > 0:
            if self.kernel_type == "general":
                shape_init = -10.0 * torch.ones((M, 1), dtype=torch.float, device="cuda")  # → β ≈ 2.0
            else:
                shape_init = 1.386 * torch.ones((M, 1), dtype=torch.float, device="cuda")  # → β ≈ 4.0
            self._shape = nn.Parameter(shape_init.requires_grad_(True))
        if hasattr(self, '_adaptive_cat_weight') and self._adaptive_cat_weight.numel() > 0:
            self._adaptive_cat_weight = nn.Parameter(torch.zeros(M, 1, device="cuda").requires_grad_(True))
        if hasattr(self, '_adaptive_zero_weight') and self._adaptive_zero_weight.numel() > 0:
            self._adaptive_zero_weight = nn.Parameter(torch.zeros(M, 1, device="cuda").requires_grad_(True))
        if hasattr(self, '_gate_logits') and self._gate_logits.numel() > 0:
            self._gate_logits = nn.Parameter(torch.zeros(M, 1, device="cuda").requires_grad_(True))
        if hasattr(self, '_flex_beta') and self._flex_beta.numel() > 0:
            self._flex_beta = nn.Parameter(torch.zeros(M, 1, device="cuda").requires_grad_(True))

        # Directional-feature tensors (SB / SG / SV) also need to be resized to M.
        # Mirror create_from_pcd init so the fresh slots start from a sane default.
        if getattr(self, 'feature_mode', 'sh') == "beta" and self._sb_params.numel() > 0:
            K = self.sb_number
            sb = torch.zeros((M, K, 6), dtype=torch.float, device="cuda")
            sb[..., 3] = torch.rand(M, K, device="cuda") * float(np.pi)
            sb[..., 4] = torch.rand(M, K, device="cuda") * 2.0 * float(np.pi)
            self._sb_params = nn.Parameter(sb.requires_grad_(True))
        if getattr(self, 'feature_mode', 'sh') == "sg" and self._sg_directions.numel() > 0:
            K = self.sg_number
            d = torch.randn(M, K, 3, device="cuda")
            d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
            self._sg_directions = nn.Parameter(d.requires_grad_(True))
            self._sg_sharpness_sg = nn.Parameter(
                0.1 * torch.ones((M, K, 1), device="cuda").requires_grad_(True)
            )
            self._sg_rgb = nn.Parameter(
                (0.1 * torch.randn(M, K, 3, device="cuda")).requires_grad_(True)
            )
        if getattr(self, 'feature_mode', 'sh') == "voronoi" and self._sv_sites.numel() > 0:
            K = self.sv_number
            fib = _fibonacci_sphere(K).to("cuda")
            sites = fib.unsqueeze(0).expand(M, -1, -1).contiguous()
            self._sv_sites = nn.Parameter(sites.requires_grad_(True))
            self._sv_colors = nn.Parameter(
                torch.zeros((M, K, 3), device="cuda").requires_grad_(True)
            )

        # Reset accumulators
        self.xyz_gradient_accum = torch.zeros(M, 1, device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros(M, 1, device="cuda")
        self.feat_gradient_accum = torch.zeros(M, 1, device="cuda")
        self.denom = torch.zeros(M, 1, device="cuda")
        self.max_radii2D = torch.zeros(M, device="cuda")
        if self.minimc_error_accum.numel() > 0:
            self.minimc_error_accum = torch.zeros(M, 1, device="cuda")
            self.minimc_win_count = torch.zeros(M, 1, device="cuda")

        self.mini_factor_culling = None
