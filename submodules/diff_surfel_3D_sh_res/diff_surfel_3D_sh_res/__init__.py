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
            hashgrid_settings.aa_threshold,
            raster_settings.max_intersections_per_pixel,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, out_index, radii, geomBuffer, binningBuffer, imgBuffer, pixels, transmittance_avg, intersection_buffer, intersection_count = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, out_index, radii, geomBuffer, binningBuffer, imgBuffer, pixels, transmittance_avg, intersection_buffer, intersection_count = _C.rasterize_gaussians(*args)

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
        # Return geomBuffer for use by 3D mode backward (transMat access)
        return color, radii, depth, transmittance_avg, pixels, intersection_buffer, intersection_count, geomBuffer

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth, grad_0, grad_1, grad_intersection_buffer, grad_intersection_count, grad_geomBuffer):

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
                kernel_type,
                raster_settings.detach_hash_grad)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                (grad_features, grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
                 grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_feat_sum,
                 grad_features_diffuse, grad_shapes,
                 grad_mlp_W1, grad_mlp_W2, grad_mlp_W3
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            (grad_features, grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
             grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_feat_sum,
             grad_features_diffuse, grad_shapes,
             grad_mlp_W1, grad_mlp_W2, grad_mlp_W3
            ) = _C.rasterize_gaussians_backward(*args)

        # Store MLP gradients for 3D_SH_res mode (render_mode=5)
        # These need to be retrieved by the caller and applied to MLP parameters
        # Use mask to handle bit 8 flag (collaborative GEMM enabled)
        if (render_mode & 0xFF) == 5:
            _RasterizeGaussians._last_mlp_grads = (
                grad_mlp_W1, grad_mlp_W2, grad_mlp_W3
            )

        # For 3D mode (render_mode == 3):
        # - CUDA color output is NOT used (Python MLP processes intersection buffer instead)
        # - So grad_out_color = 0, meaning native backward only has gradients from auxiliary losses
        #   (mask_loss via dL_daccum, depth_loss, normal_loss, etc.)
        # - IntersectionOpacityGrad computes gradients from RGB/feature loss only
        # - These are ADDITIVE, not overlapping - no double-counting!
        #
        # Keep ALL gradients from native backward (they contain mask_loss contributions).
        # IMPORTANT: Do NOT zero grad_means2D - it's needed for densification!
        # - Native backward's grad_means2D comes from mask_loss (dL_daccum -> dL_dalpha -> dL_dmean2D)
        # - IntersectionOpacityGrad's grad_screenspace comes from RGB loss
        # - These are additive since they come from different loss sources
        # - grad_means2D is critical for add_densification_stats() to work correctly!
        if render_mode == 3:
            # Keep ALL gradients including grad_means2D for densification
            # Keep: grad_means3D, grad_means2D, grad_opacities, grad_scales, grad_rotations
            # These contain auxiliary loss (mask_loss, etc.) gradients
            pass

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
            aabb_mode,
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

