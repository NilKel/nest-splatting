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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    homotrans,
    ap_level,
    features,
    offsets,
    gridrange,
    features_diffuse,
    offsets_diffuse,
    gridrange_diffuse,
    raster_settings,
    hashgrid_settings,
    render_mode,
    shapes,
    kernel_type,
    aabb_mode=0
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        homotrans,
        ap_level,
        features,
        offsets,
        gridrange,
        features_diffuse,
        offsets_diffuse,
        gridrange_diffuse,
        raster_settings,
        hashgrid_settings,
        render_mode,
        shapes,
        kernel_type,
        aabb_mode
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        homotrans,
        ap_level,
        features,
        offsets,
        gridrange,
        features_diffuse,
        offsets_diffuse,
        gridrange_diffuse,
        raster_settings,
        hashgrid_settings,
        render_mode,
        shapes,
        kernel_type,
        aabb_mode=0
    ):

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record() 

        # Handle empty shapes tensor
        if shapes is None:
            shapes = torch.Tensor([]).cuda()

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            homotrans,
            ap_level,
            features,
            offsets,
            gridrange,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.beta,
            raster_settings.if_contract,
            raster_settings.record_transmittance,
            hashgrid_settings.L,
            hashgrid_settings.S,
            hashgrid_settings.H,
            hashgrid_settings.align_corners,
            hashgrid_settings.interpolation,
            features_diffuse,
            offsets_diffuse,
            gridrange_diffuse,
            render_mode,
            hashgrid_settings.shape_dims,
            raster_settings.max_intersections,
            shapes,
            kernel_type,
            aabb_mode,
            hashgrid_settings.aa,
            hashgrid_settings.aa_threshold
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, out_index, radii, geomBuffer, binningBuffer, imgBuffer, pixels, transmittance_avg = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, out_index, radii, geomBuffer, binningBuffer, imgBuffer, pixels, transmittance_avg = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.hashgrid_settings = hashgrid_settings
        ctx.num_rendered = num_rendered
        ctx.render_mode = render_mode
        ctx.kernel_type = kernel_type

        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, homotrans, ap_level, features, offsets, gridrange, \
            features_diffuse, offsets_diffuse, gridrange_diffuse, \
            depth, out_index, radii, sh, geomBuffer, binningBuffer, imgBuffer, shapes)

        # if raster_settings.record_transmittance :
        return color, radii, depth, transmittance_avg, pixels

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth, grad_0, grad_1):

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        hashgrid_settings = ctx.hashgrid_settings
        render_mode = ctx.render_mode
        kernel_type = ctx.kernel_type
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, homotrans, ap_level, features, offsets, gridrange, \
            features_diffuse, offsets_diffuse, gridrange_diffuse, \
            depth, out_index, radii, sh, geomBuffer, binningBuffer, imgBuffer, shapes = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                depth,
                out_index,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                homotrans,
                ap_level,
                features,
                offsets,
                gridrange,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_depth,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug,
                raster_settings.beta,
                raster_settings.if_contract,
                hashgrid_settings.L,
                hashgrid_settings.S,
                hashgrid_settings.H,
                hashgrid_settings.align_corners,
                hashgrid_settings.interpolation,
                features_diffuse,
                offsets_diffuse,
                gridrange_diffuse,
                render_mode,
                hashgrid_settings.shape_dims,
                shapes,
                kernel_type)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_features, grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_feat_sum, grad_features_diffuse, grad_shapes = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_features, grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_feat_sum, grad_features_diffuse, grad_shapes = _C.rasterize_gaussians_backward(*args)

        grad_homotrans = None

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_homotrans,
            grad_feat_sum,
            grad_features,
            None,  # offsets
            None,  # gridrange
            grad_features_diffuse,
            None,  # offsets_diffuse
            None,  # gridrange_diffuse
            None,  # raster_settings
            None,  # hashgrid_settings
            None,  # render_mode
            grad_shapes,  # shapes
            None,  # kernel_type
            None,  # aabb_mode
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    beta : float
    if_contract : bool
    record_transmittance : bool
    max_intersections : int = 0  # 0 means no limit

class HashGridSettings(NamedTuple):
    L: int
    S: float
    H : int
    align_corners : bool
    interpolation : int
    shape_dims: torch.Tensor  # [GS, HS, OS] or empty for backward compat
    aa: float = 0.0  # Anti-aliasing scale factor (0=disabled, >0=enabled)
    aa_threshold: float = 0.01  # Skip hash query when avg level weight < threshold

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings, hashgrid_settings):
        super().__init__()
        self.raster_settings = raster_settings
        self.hashgrid_settings = hashgrid_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None,
        cov3D_precomp = None, \
        homotrans = None, ap_level = None, \
        features = None, offsets = None, gridrange = None, \
        features_diffuse = None, offsets_diffuse = None, gridrange_diffuse = None, \
        render_mode = 0, shapes = None, kernel_type = 0, aabb_mode = 0):

        raster_settings = self.raster_settings
        hashgrid_settings = self.hashgrid_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()

        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).cuda()

        if homotrans is None:
            homotrans = torch.Tensor([]).cuda()
        if ap_level is None:
            ap_level = torch.Tensor([]).cuda()

        if features is None:
            features = torch.Tensor([]).cuda()
        if offsets is None:
            offsets = torch.Tensor([]).int().cuda()
        if gridrange is None:
            gridrange = torch.Tensor([]).cuda()

        # For surface_rgb mode: second hashgrid for diffuse RGB
        if features_diffuse is None:
            features_diffuse = torch.Tensor([]).cuda()
        if offsets_diffuse is None:
            offsets_diffuse = torch.Tensor([]).int().cuda()
        if gridrange_diffuse is None:
            gridrange_diffuse = torch.Tensor([]).cuda()

        # Beta kernel shapes (empty tensor if not using beta kernel)
        if shapes is None:
            shapes = torch.Tensor([]).cuda()

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            homotrans,
            ap_level,
            features,
            offsets,
            gridrange,
            features_diffuse,
            offsets_diffuse,
            gridrange_diffuse,
            raster_settings,
            hashgrid_settings,
            render_mode,
            shapes,
            kernel_type,
            aabb_mode
        )

def compute_relocation(opacity_old, scale_old, N, binoms, n_max):
    """
    Compute new opacities and scales for MCMC relocation.
    
    Args:
        opacity_old: [N] tensor of old opacities (after sigmoid)
        scale_old: [N, 2] tensor of old scales (after exp)
        N: [N] int tensor of relocation ratios
        binoms: [n_max, n_max] tensor of binomial coefficients
        n_max: maximum value for N
    
    Returns:
        new_opacity: [N] tensor of new opacities
        new_scale: [N, 2] tensor of new scales
    """
    new_opacity, new_scale = _C.compute_relocation(opacity_old, scale_old, N.int(), binoms, n_max)
    return new_opacity, new_scale

