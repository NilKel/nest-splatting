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
    aabb_mode=0,
    metric_map=None,
    is_textured=None,
    scaling_z=None,
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
        aabb_mode,
        metric_map,
        is_textured,
        scaling_z,
    )

class _RasterizeGaussians(torch.autograd.Function):
    # Class variable to store MLP gradients from last backward pass
    _last_mlp_grads = None

    @staticmethod
    def get_mlp_grads():
        """Get MLP gradients from last backward pass (for 3D_SH_res mode, bias-free, all [16×16]).
        Returns tuple of (grad_W1, grad_W2, grad_W3) or None.
        """
        return _RasterizeGaussians._last_mlp_grads

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
        aabb_mode=0,
        metric_map=None,
        is_textured=None,
        scaling_z=None,
    ):

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # Handle empty shapes tensor
        if shapes is None:
            shapes = torch.Tensor([]).cuda()

        # FastGS metric_map: optional per-pixel int32 [H*W] mask of high-error pixels.
        # Empty tensor when disabled.
        if metric_map is None:
            metric_map = torch.empty(0, dtype=torch.int32, device="cuda")

        # `--method mixed` per-Gauss bool flag [P]. Empty tensor → all-textured.
        if is_textured is None:
            is_textured = torch.empty(0, dtype=torch.bool, device="cuda")
        elif is_textured.dtype != torch.bool:
            is_textured = is_textured.bool()

        # `--method mixed_3d` per-Gauss activated 3rd-axis scale [P]. Empty tensor
        # → nullptr in CUDA → untextured surfels keep the 2DGS ray-splat geometry
        # (bit-identical to `--method mixed`, e.g. pre-texsplit).
        if scaling_z is None:
            scaling_z = torch.empty(0, dtype=torch.float32, device="cuda")

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
            hashgrid_settings.aa_threshold,
            raster_settings.max_intersections_per_pixel,
            metric_map,
            is_textured,
            scaling_z,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, out_index, radii, geomBuffer, binningBuffer, imgBuffer, pixels, transmittance_avg, intersection_buffer, intersection_count, metric_counts = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, out_index, radii, geomBuffer, binningBuffer, imgBuffer, pixels, transmittance_avg, intersection_buffer, intersection_count, metric_counts = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.hashgrid_settings = hashgrid_settings
        ctx.num_rendered = num_rendered
        ctx.render_mode = render_mode
        ctx.kernel_type = kernel_type
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, homotrans, ap_level, features, offsets, gridrange, \
            features_diffuse, offsets_diffuse, gridrange_diffuse, \
            depth, out_index, radii, sh, geomBuffer, binningBuffer, imgBuffer, shapes, is_textured, scaling_z)

        # `--l2` (mixed_3d only): expose a second image-output slot that's
        # numerically a clone of `color` but a separate autograd-graph node.
        # When --l2 is OFF (default), consumers ignore this slot → autograd
        # passes None for its upstream grad in backward, and we convert that
        # to nullptr in the CUDA call → byte-identical to pre-flag behavior.
        # When --l2 is ON, the renderer wires `color` to L1+SSIM(image_tex, gt)
        # and `color_untex` to L2(image_untex, gt); the kernel routes per-Gauss
        # using is_textured to pick between the two upstream image gradients.
        color_untex = color.clone()

        # if raster_settings.record_transmittance :
        # Return geomBuffer for use by 3D mode backward (transMat access).
        # Also surface out_index ([H, W] int32, per-pixel id of the max-weight contributor)
        # for the mini depth-reinit SH-transfer path. New element appended at the END to
        # avoid disturbing existing positional unpacks elsewhere.
        # metric_counts (FastGS VCD/VCP) also appended at the end; [P] int32, zeros
        # when metric_map wasn't provided.
        return color, radii, depth, transmittance_avg, pixels, intersection_buffer, intersection_count, geomBuffer, out_index, metric_counts, color_untex

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth, grad_0, grad_1, grad_intersection_buffer, grad_intersection_count, grad_geomBuffer, grad_out_index, grad_metric_counts, grad_out_color_untex):

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
            depth, out_index, radii, sh, geomBuffer, binningBuffer, imgBuffer, shapes, is_textured, scaling_z = ctx.saved_tensors

        # `--l2` (mixed_3d only): per-Gauss image-grad routing. The second
        # image-output slot (`color_untex`) gets an upstream grad iff the
        # caller actually wired a loss to it. PyTorch passes a ZEROS tensor
        # (not None) for unused outputs of a Function, so we detect "no L2
        # consumer" by checking if the gradient is non-trivial. Zeros →
        # pass empty tensor → CUDA sees nullptr → kernel reverts to single-
        # loss behavior, byte-identical to pre-flag. Real grad → CUDA routes
        # per-Gauss using is_textured.
        # The edge case where the user IS using L2 but the L2-leg gradient
        # is literally all zeros falls back to single-loss for that iter —
        # equivalent (the L2 leg contributes nothing either way).
        _has_l2_grad = (grad_out_color_untex is not None
                        and grad_out_color_untex.is_floating_point()
                        and bool(grad_out_color_untex.abs().sum().item() > 0))
        if _has_l2_grad:
            grad_out_color_untex_pass = grad_out_color_untex
        else:
            grad_out_color_untex_pass = torch.empty(0, dtype=grad_out_color.dtype, device=grad_out_color.device)

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
                grad_out_color_untex_pass,
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
                kernel_type,
                raster_settings.detach_hash_grad,
                is_textured,
                scaling_z)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                (grad_features, grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
                 grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_feat_sum,
                 grad_features_diffuse, grad_shapes,
                 grad_mlp_W1, grad_mlp_W2, grad_mlp_W3, grad_scaling_z
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            (grad_features, grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
             grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_feat_sum,
             grad_features_diffuse, grad_shapes,
             grad_mlp_W1, grad_mlp_W2, grad_mlp_W3, grad_scaling_z
            ) = _C.rasterize_gaussians_backward(*args)

        # Debug: print grad_features stats to diagnose hash gradient flow
        if grad_features is not None and grad_features.numel() > 0:
            if not hasattr(_RasterizeGaussians, '_bw_dbg_count'):
                _RasterizeGaussians._bw_dbg_count = 0
            _RasterizeGaussians._bw_dbg_count += 1
            if _RasterizeGaussians._bw_dbg_count % 500 == 1:
                print(f"[BW_DBG iter~{_RasterizeGaussians._bw_dbg_count}] grad_features: shape={list(grad_features.shape)}, "
                      f"norm={grad_features.norm().item():.8f}, "
                      f"abs_max={grad_features.abs().max().item():.8f}, "
                      f"nonzero={grad_features.count_nonzero().item()}/{grad_features.numel()}")

        # Store MLP gradients for fused MLP modes (render_mode=5)
        # These need to be retrieved by the caller and applied to MLP parameters
        # Use mask to handle bit flags (collaborative GEMM, freeze_mlp)
        if (render_mode & 0xFF) == 5:
            _RasterizeGaussians._last_mlp_grads = (
                grad_mlp_W1, grad_mlp_W2, grad_mlp_W3
            )

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
            None,  # metric_map (int32; no gradient)
            None,  # is_textured (bool; no gradient)
            # scaling_z (mixed_3d EWA 3rd axis) — input 24. Pre-`--texsplit`
            # the renderer doesn't pass scaling_z, so the wrapper substitutes an
            # internal torch.empty(0) that is NOT a grad-requiring Variable;
            # autograd then forbids a non-None grad here. Gate on
            # needs_input_grad so we only return the grad when scaling_z was a
            # real differentiable input (post-texsplit pc.get_scaling_z).
            grad_scaling_z if ctx.needs_input_grad[24] else None,  # scaling_z
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
    detach_hash_grad : bool = False  # Detach positional gradients from hashgrid (for CAT mode frequency separation)
    max_intersections_per_pixel : int = 0  # 3D mode: max intersections per pixel for intersection buffer (0 = disabled)

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
        render_mode = 0, shapes = None, kernel_type = 0, aabb_mode = 0,
        metric_map = None, is_textured = None, scaling_z = None):

        raster_settings = self.raster_settings
        hashgrid_settings = self.hashgrid_settings

        # render_mode 6 (3D_SH_cat) legitimately needs BOTH shs (for computeColorFromSH
        # in preprocess → rgb buffer) AND colors_precomp (= DC_SH used as MLP input
        # at line ~1889 of forward.cu, threaded through the kernel's `features` arg).
        # So we only error when neither is provided.
        if shs is None and colors_precomp is None:
            raise Exception('Please provide either SHs or precomputed colors!')

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
            aabb_mode,
            metric_map,
            is_textured,
            scaling_z,
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

def compute_opacity_gradient_3D(dL_dweight, T_values, G_values, alpha_values,
                                 gaussian_ids, pixel_starts, N):
    """
    Compute opacity gradients for 3D mode with full transmittance chain.

    Args:
        dL_dweight: [M] gradient w.r.t. weights from PyTorch
        T_values: [M] transmittance values (before each intersection)
        G_values: [M] kernel values (Gaussian/Beta kernel)
        alpha_values: [M] alpha values (opacity * G)
        gaussian_ids: [M] Gaussian indices (int tensor)
        pixel_starts: [num_pixels+1] pixel boundary indices (int tensor)
        N: number of Gaussians

    Returns:
        dL_dopacity: [N] gradients for opacity parameters
        dL_dalpha: [M] per-intersection dL/dalpha for geometry gradient computation
    """
    return _C.compute_opacity_gradient_3D(
        dL_dweight, T_values, G_values, alpha_values,
        gaussian_ids, pixel_starts, N
    )

def compute_geometry_gradient_3D(dL_dalpha, opacity_values, G_values,
                                  s_x_values, s_y_values, rho_flag,
                                  gaussian_ids, pixel_ids, transMat,
                                  W, H, N):
    """
    Compute geometry gradients for 3D mode using geomBuffer's transMat.
    Takes dL_dalpha per intersection and computes dL_dtransMat.

    Args:
        dL_dalpha: [M] per-intersection dL/dalpha from opacity gradient kernel
        opacity_values: [M] per-intersection opacity values
        G_values: [M] kernel values (Gaussian kernel)
        s_x_values: [M] intersection s.x coordinates
        s_y_values: [M] intersection s.y coordinates
        rho_flag: [M] 1.0=disk intersection, 0.0=center intersection
        gaussian_ids: [M] Gaussian indices (int tensor)
        pixel_ids: [M] pixel indices (int tensor)
        transMat: [N, 9] transformation matrices from geomBuffer
        W, H: image dimensions
        N: number of Gaussians

    Returns:
        dL_dtransMat: [N, 9] gradients for transformation matrices
    """
    return _C.compute_geometry_gradient_3D(
        dL_dalpha, opacity_values, G_values,
        s_x_values, s_y_values, rho_flag,
        gaussian_ids, pixel_ids, transMat,
        W, H, N
    )

def backward_from_weight_grad(geomBuffer, P, dL_dweight, gaussian_ids, pixel_ids,
                               pixel_starts, T_values, G_values, alpha_values,
                               opacity_values, s_x_values, s_y_values, rho_flag,
                               dL_duv_x, dL_duv_y, W, H):
    """
    Unified 3D mode backward from weight gradients.
    Takes dL_dweight from PyTorch, reads transMat from geomBuffer internally.
    Computes both opacity gradients and geometry gradients in one pass.
    Also accepts dL_duv from hash/xyz gradient path (like cat mode).

    Args:
        geomBuffer: Raw geometry buffer from forward pass
        P: Number of Gaussians (for geomBuffer parsing)
        dL_dweight: [M] gradient w.r.t. weights from PyTorch
        gaussian_ids: [M] Gaussian indices (int tensor)
        pixel_ids: [M] pixel indices (int tensor)
        pixel_starts: [num_pixels+1] pixel boundary indices (int tensor)
        T_values: [M] transmittance values
        G_values: [M] kernel values
        alpha_values: [M] alpha values
        opacity_values: [M] per-intersection opacity values
        s_x_values: [M] intersection s.x coordinates
        s_y_values: [M] intersection s.y coordinates
        rho_flag: [M] 1.0=disk intersection, 0.0=center intersection
        dL_duv_x: [M] hash/xyz gradient contribution to s.x (can be empty tensor)
        dL_duv_y: [M] hash/xyz gradient contribution to s.y (can be empty tensor)
        W, H: image dimensions

    Returns:
        dL_dopacity: [N] gradients for opacity parameters
        dL_dtransMat: [N, 9] gradients for transformation matrices
        dL_dmean2D: [N, 2] gradients for mean2D (for position/densification)
    """
    return _C.backward_from_weight_grad(
        geomBuffer, P, dL_dweight, gaussian_ids, pixel_ids, pixel_starts,
        T_values, G_values, alpha_values, opacity_values,
        s_x_values, s_y_values, rho_flag, dL_duv_x, dL_duv_y, W, H
    )

def transmat_to_scale_rot_grad(dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, means3D, transMat_precomp, scales, rotations, projmatrix, viewmatrix, W, H):
    """
    Convert screen-space dL_dtransMat to world-space dL_dscale and dL_drotation.
    This properly applies the projection matrix (with ndc2pix) to get correct gradients,
    matching what the native CUDA backward does.

    Also incorporates:
    - xyz gradient contribution (dL_dhomoMat) like CAT mode backward
    - 2D mean gradient contribution (dL_dmean2D) for correct xyz gradients
    - Normal gradient contribution (dL_dnormal3D) from depth/normal loss

    The native backward computes:
        P = world2ndc * ndc2pix  (CRITICAL: includes image dimension scaling!)
        dL_dT += dL_dmean2D contribution (CRITICAL: screen-space mean gradient!)
        dL_dM = P * transpose(dL_dT) + dL_dhomoMat  (xyz gradient contribution!)
        dL_dtn = viewmatrix^T * dL_dnormal3D  (normal gradient in world space!)
        dL_dRS = [dL_dM[0], dL_dM[1], dL_dtn]  (includes normal gradient!)
        dL_dscale = [dot(dL_dRS[0], R[0]), dot(dL_dRS[1], R[1])]
        dL_dR = [dL_dRS[0] * scale.x, dL_dRS[1] * scale.y, dL_dRS[2]]
        dL_drot = quat_to_rotmat_vjp(rot, dL_dR)

    This function implements the same transformation for 3D_direct mode.

    Args:
        dL_dtransMat: [N, 9] screen-space transformation matrix gradient
        dL_dhomoMat: [N, 9] xyz gradient contribution (can be empty tensor)
                     Layout: [col0.xyz, col1.xyz, col2.xyz] per Gaussian
                     - col0: sum over intersections of (dL_dxyz * s_x)
                     - col1: sum over intersections of (dL_dxyz * s_y)
                     - col2: sum over intersections of (dL_dxyz)
        dL_dmean2D: [N, 2] 2D mean gradient (can be empty tensor)
        dL_dnormal3D: [N, 3] normal gradient from depth/normal loss (can be empty tensor)
        means3D: [N, 3] world-space positions (needed for dL_dmean2D handling)
        transMat_precomp: [N, 9] forward pass transMat (can be empty tensor)
                          When provided, used for dL_dmean2D handling for numerical accuracy.
        scales: [N, 2] scale parameters
        rotations: [N, 4] quaternion rotation parameters
        projmatrix: [4, 4] or [16] projection matrix
        viewmatrix: [4, 4] or [16] view matrix (for normal gradient transform)
        W: int, image width (for ndc2pix transformation)
        H: int, image height (for ndc2pix transformation)

    Returns:
        dL_dscales: [N, 2] world-space scale gradients
        dL_drots: [N, 4] quaternion rotation gradients
        dL_dmeans: [N, 3] mean position gradients
    """
    return _C.transmat_to_scale_rot_grad(dL_dtransMat, dL_dhomoMat, dL_dmean2D, dL_dnormal3D, means3D, transMat_precomp,
                                          scales, rotations, projmatrix, viewmatrix, W, H)

def get_transmat_from_geombuffer(geomBuffer, P):
    """
    Extract transMat from geomBuffer.

    Args:
        geomBuffer: Raw geometry buffer from forward pass
        P: Number of Gaussians

    Returns:
        transMat: [P, 9] transformation matrices
    """
    return _C.get_transmat_from_geombuffer(geomBuffer, P)

def set_mlp_weights(W1, W2, W3):
    """
    Copy MLP weights to CUDA global memory for fused in-kernel MLP evaluation (bias-free, all [16×16]).
    Call this before each render call if weights have changed.

    Layer 1 uses input padding (input[4]=1.0) for implicit bias via W1[:, 4].
    Layers 2 and 3 have no bias.

    Args:
        W1: [16, 16] Layer 1 weights (input → hidden1)
        W2: [16, 16] Layer 2 weights (hidden1 → hidden2)
        W3: [16, 16] Layer 3 weights (first 3 rows = RGB residual)
    """
    _C.set_mlp_weights(W1.contiguous(),
                       W2.contiguous(),
                       W3.contiguous())

def set_contrib_thresh(val):
    """Set contribution threshold. Skip hash query when w = T*alpha < val (0 = disabled)."""
    _C.set_contrib_thresh(val)

def set_count_thresh(val):
    """Set count threshold. Skip hash after N contributing Gaussians per pixel (0 = disabled)."""
    _C.set_count_thresh(val)

def set_overdraw_lambda(val):
    """Set overdraw regularization lambda. Penalizes per-pixel contributor count with sigmoid relaxation (0 = disabled)."""
    _C.set_overdraw_lambda(val)

def set_weight_reg_lambda(val):
    """Set weight-squared regularization lambda. CUDA backward adds dL_dalpha = -lambda * 2*w*T per Gaussian (0 = disabled)."""
    _C.set_weight_reg_lambda(val)

def set_activation_bias(sh_bias=0.5, res_bias=0.5):
    """Set activation biases: color = ReLU(SH + sh_bias) + ReLU(residual + res_bias).
    Default: 0.5/0.5. For decomposition: sh_only uses res_bias=-999, tex_only uses sh_bias=-999."""
    _C.set_activation_bias(sh_bias, res_bias)

def set_residual_mode(mode=0):
    """Select residual activation:
       0 = 3D_SH_res (default): color = ReLU(ReLU(SH+sh_bias) + residual + res_bias)
       1 = 3D_SH_add:           color = ReLU(SH+sh_bias) + ReLU(residual + res_bias)
    Patches both forward and backward device globals — call once at startup."""
    _C.set_residual_mode(int(mode))


def set_ste_relu(v=0):
    """`--ste`: straight-through estimator on the per-Gauss outer ReLU (mode 0).
    Backward only — gradient = 1 even at clamped activations so the MLP / hash
    keeps receiving signal at clamped pixels. v=1 enables; v=0 = default."""
    _C.set_ste_relu(int(v))


def set_textured_bias_gate(v=0):
    """`--method res_3d_paired`: when v==1, the kernel forces sh_color = 0 for
    TEXTURED Gauss (is_textured[gauss_id] == True). Their feat becomes just
    `residual + res_bias` — no `+0.5` SH-bias baseline floor. Untextured Gauss
    are untouched (EWA branch reads rgb[] directly with the bias applied).

    Default 0 = byte-identical to plain mixed_3d. Set to 1 at the
    `--method res_3d_paired` split event so the textured residual-carriers
    contribute pure residual instead of (0.5 + residual). Mirrors the
    per-Gauss bias gate built into diff_surfel_res_3d.

    Patches both forward and backward device globals."""
    _C.set_textured_bias_gate(int(v))


def set_lru_slope(alpha=0.0):
    """`--lru`: leaky-ReLU slope α for the outer per-Gauss activation (mode 0).
    Forward `feat = (pre>0) ? pre : α·pre`. Backward clamp gate = α (instead
    of 0). α == 0 (default) reduces to standard ReLU."""
    _C.set_lru_slope(float(alpha))

def set_anti_alias(factor=0.0, focal=1.0):
    """Set Nexels-style hash-grid anti-aliasing down-weighting.
    factor=0 disables AA. factor=1.0 matches Nexels' grid_threshold_factor=1.0
    (their paper default).

    Internally we pre-multiply by 2 so the CUDA kernel implements
       ts = (2*factor) * depth * scale_world / focal
    matching Nexels'   ts = grid_threshold_factor * (2*depth/focal) * scale_world.
    The kernel multiplies the per-level normalised scale by grad_scale
    (= d_normalized/d_world, set inside query_feature) so the formula is
    correct under both contract and non-contract paths."""
    _C.set_anti_alias(float(factor) * 2.0, float(focal))

def set_compact_mult(val=1.0):
    """Set FastGS Compact Box Mahalanobis² multiplier for AdR cutoff.

    AdR cutoff formula: cutoff = sqrt(2 * log(opacity * 255) * val).
    - val=1.0 (default): matches our existing AdR cutoff, no change.
    - val=0.5 (FastGS paper): tighter tile AABB → fewer Gaussian–tile pairs,
      faster rasterization. May cause visible alpha discontinuities at tile edges
      for near-threshold Gaussians; FastGS accepts this for speed.
    Only effective when --aabb is one of {adr, adrrect} (aabb_mode in {1, 3}).
    """
    _C.set_compact_mult(float(val))

def set_aa_kernel_size(val=0.0):
    """Set AA-2DGS Jacobian-based mip filter kernel size σ (0 = off, typical 0.1).
    When > 0, replaces the min(rho3d, rho2d) heuristic in the scalar mode 5/6
    standard-Gaussian path with Σ'_local = I + σ·J·Jᵀ, alpha = coef·opa·exp(-0.5·rho)."""
    _C.set_aa_kernel_size(float(val))


def set_skip_mlp_grad(val=True):
    """Periodic-freeze toggle for mode 5 (3D_SH_res) backward.

    When True, the backward skips ALL hash/MLP gradient work this iteration:
      - 3 weight-grad WMMA GEMMs (dL_dW1/W2/W3)
      - Scalar input-chain backprop (W3ᵀ → W2ᵀ → W1ᵀ, i.e. Phase 2/3/4 scalar)
      - query_feature<true> call (hash-table dL_dgrid atomicAdds AND dL/dxyz-from-hash)
      - Tile-level dL_dW flush to global memory

    Geometry backward (transMat, normals, alpha, opacity, shapes) runs
    unchanged. When False (default), the backward runs byte-for-byte
    identically to pre-flag behavior.

    Pair with a Python optimizer.step() skip on hash_encoding / mlp_fused
    param groups on skip iterations.
    """
    _C.set_skip_mlp_grad(bool(val))


def set_depth_sort(val):
    """Set depth sort toggle. True = separated depth sort, False = standard sort (default)."""
    _C.set_depth_sort(val)

def get_mlp_grads():
    """
    Get MLP gradients from last backward pass (for 3D_SH_res mode, bias-free, all [16×16]).

    Returns:
        Tuple of (grad_W1, grad_W2, grad_W3) or None.
        - grad_W1: [16, 16] Layer 1 weight gradients
        - grad_W2: [16, 16] Layer 2 weight gradients
        - grad_W3: [16, 16] Layer 3 weight gradients (first 3 rows = RGB residual)
    """
    return _RasterizeGaussians.get_mlp_grads()


def reset_backward_profile():
    """Reset backward kernel profiling counters."""
    _C.reset_backward_profile()


def read_backward_profile():
    """Read backward kernel profiling data.

    Returns:
        (cycles, counts) where:
        - cycles[0]: Phase A (intersection + MLP fwd + sigmoid bw + GEMM L3 + dL_dz2)
        - cycles[1]: Phase B (GEMM L2 + dL_dz1)
        - cycles[2]: Phase C (GEMM L1 + dL_dinput + feature/hash/geom grads)
        - cycles[3]: Tile flush
        - cycles[4]: Total per-Gaussian cycles
        - counts[0]: Gaussians processed
        - counts[1]: Gaussians skipped (ballot)
        - counts[2]: Tiles processed
        - counts[3]: Total intersections
    """
    return _C.read_backward_profile()

