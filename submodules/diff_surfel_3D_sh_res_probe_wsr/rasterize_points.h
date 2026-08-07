/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& homotrans,
	const torch::Tensor& ap_level,
	const torch::Tensor& features,
	const torch::Tensor& offsets,
	const torch::Tensor& gridrange,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const float beta,
	const bool if_contract,
	const bool record_transmittance,
	const uint32_t Level,
	const float LevelScale,
	const uint32_t Base,
	const bool align_corners,
	const uint32_t interp,
	const torch::Tensor& features_diffuse,
	const torch::Tensor& offsets_diffuse,
	const torch::Tensor& gridrange_diffuse,
	const int render_mode,
	const torch::Tensor& shape_dims,
	const int max_intersections,
	const torch::Tensor& shapes,
	const int kernel_type,
	const int aabb_mode,
	const float aa,
	const float aa_threshold,
	const int max_intersections_per_pixel,
	const torch::Tensor& metric_map);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	 const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& other_maps,
	const torch::Tensor& out_index,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& homotrans,
	const torch::Tensor& ap_level,
	const torch::Tensor& features,
	const torch::Tensor& offsets,
	const torch::Tensor& gridrange,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const float beta,
	const bool if_contract,
	const uint32_t Level,
	const float LevelScale,
	const uint32_t Base,
	const bool align_corners,
	const uint32_t interp,
	const torch::Tensor& features_diffuse,
	const torch::Tensor& offsets_diffuse,
	const torch::Tensor& gridrange_diffuse,
	const int render_mode,
	const torch::Tensor& shape_dims,
	const torch::Tensor& shapes,
	const int kernel_type,
	const bool detach_hash_grad);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);

// 3D mode opacity gradient with full transmittance chain
// Returns: (dL_dopacity [N], dL_dalpha [M])
std::tuple<torch::Tensor, torch::Tensor> ComputeOpacityGradient3DCUDA(
    const torch::Tensor& dL_dweight,
    const torch::Tensor& T_values,
    const torch::Tensor& G_values,
    const torch::Tensor& alpha_values,
    const torch::Tensor& gaussian_ids,
    const torch::Tensor& pixel_starts,
    const int N);

// 3D mode geometry gradient using geomBuffer
// Takes dL_dalpha per intersection and computes dL_dtransMat
// Returns: dL_dtransMat [N, 9]
torch::Tensor ComputeGeometryGradient3DCUDA(
    const torch::Tensor& dL_dalpha,
    const torch::Tensor& opacity_values,
    const torch::Tensor& G_values,
    const torch::Tensor& s_x_values,
    const torch::Tensor& s_y_values,
    const torch::Tensor& rho_flag,
    const torch::Tensor& gaussian_ids,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& transMat,
    const int W, const int H,
    const int N);

// Unified 3D mode backward from weight gradients
// Takes dL_dweight from PyTorch, reads transMat from geomBuffer internally
// Also accepts dL_duv from hash/xyz gradient path (like cat mode)
// Returns: (dL_dopacity [N], dL_dtransMat [N, 9], dL_dmean2D [N, 2])
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BackwardFromWeightGradCUDA(
    const torch::Tensor& geomBuffer,
    const int P,
    const torch::Tensor& dL_dweight,
    const torch::Tensor& gaussian_ids,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& pixel_starts,
    const torch::Tensor& T_values,
    const torch::Tensor& G_values,
    const torch::Tensor& alpha_values,
    const torch::Tensor& opacity_values,
    const torch::Tensor& s_x_values,
    const torch::Tensor& s_y_values,
    const torch::Tensor& rho_flag,
    const torch::Tensor& dL_duv_x,    // [M] hash/xyz gradient contribution (can be empty)
    const torch::Tensor& dL_duv_y,    // [M] hash/xyz gradient contribution (can be empty)
    const int W, const int H);

// Convert screen-space dL_dtransMat to world-space dL_dscale and dL_drotation
// This properly applies the projection matrix (with ndc2pix) to get correct gradients
// Also incorporates xyz gradient contribution (dL_dhomoMat), 2D mean gradient (dL_dmean2D),
// and normal gradient from depth/normal loss (dL_dnormal3D)
// Returns: (dL_dscale [N, 2], dL_drotation [N, 4], dL_dmeans [N, 3])
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> TransMatToScaleRotGradCUDA(
    const torch::Tensor& dL_dtransMat,  // [N, 9] screen-space transMat gradient
    const torch::Tensor& dL_dhomoMat,   // [N, 9] xyz gradient contribution (can be empty)
    const torch::Tensor& dL_dmean2D,    // [N, 2] 2D mean gradient (can be empty)
    const torch::Tensor& dL_dnormal3D,  // [N, 3] normal gradient from depth/normal loss (can be empty)
    const torch::Tensor& means3D,       // [N, 3] world-space positions (needed for dL_dmean2D)
    const torch::Tensor& transMat_precomp, // [N, 9] forward pass transMat (can be empty)
    const torch::Tensor& scales,        // [N, 2]
    const torch::Tensor& rotations,     // [N, 4] quaternions
    const torch::Tensor& projmatrix,    // [4, 4] or [16] projection matrix
    const torch::Tensor& viewmatrix,    // [4, 4] or [16] view matrix (for normal gradient transform)
    const int W, const int H);          // Image dimensions for ndc2pix transformation

// Extract transMat from geomBuffer for use in backward
torch::Tensor GetTransMatFromGeomBufferCUDA(
    const torch::Tensor& geomBuffer,
    const int P);

// ============================================================================
// MLP WEIGHT MANAGEMENT FOR FUSED MODES (3D_fused, 3D_direct_fused)
// ============================================================================

// Copy MLP weights to CUDA global memory for in-kernel MLP evaluation (bias-free, all [16×16])
void SetMlpWeightsCUDA(
    const torch::Tensor& W1,      // [16, 16] - Layer 1 weights
    const torch::Tensor& W2,      // [16, 16] - Layer 2 weights
    const torch::Tensor& W3);     // [16, 16] - Layer 3 weights (only first 3 rows = RGB residual)

// Set contribution threshold (skip hash+MLP when w = T*alpha < val, 0 = disabled)
void SetContribThreshCUDA(float val);

// Set count threshold (skip hash after N contributing Gaussians per pixel, 0 = disabled)
void SetCountThreshCUDA(int val);

// Set opacity threshold (skip hash+MLP when alpha = opa*kernel_val < val, 0 = disabled)
void SetOpacityThreshCUDA(float val);

// Set texture-query dropout rate + per-iteration seed (training only, 0 rate = disabled)
void SetDropoutCUDA(float rate, int seed);

// Set overdraw regularization lambda (0 = disabled)
void SetOverdrawLambdaCUDA(float val);

// Set weight-squared regularization lambda (CUDA backward gradient)
void SetWeightRegLambdaCUDA(float val);

// Set activation biases: color = ReLU(SH + sh_bias) + ReLU(residual + res_bias)
void SetActivationBiasCUDA(float sh_bias, float res_bias);

// Select residual activation mode (mirrors training render method).
//   0 = 3D_SH_res (default): color = ReLU(ReLU(SH+sh_bias) + residual + res_bias)
//   1 = 3D_SH_add:           color = ReLU(SH+sh_bias) + ReLU(residual + res_bias)
void SetResidualModeCUDA(int mode);

// WSR (sort-free weighted-sum) mode: installs occ/occ_grad/aux pointers.
void SetWsrCUDA(int mode, torch::Tensor occ, torch::Tensor occ_grad, torch::Tensor aux);
void SetWsrGateCUDA(float tau, int bins, float zmin, float zmax, torch::Tensor tbin);
void SetWsrDGateCUDA(float margin, torch::Tensor dbuf);

// `--ste`: straight-through estimator on per-Gauss outer ReLU (mode 0).
// 1 = backward bypasses the clamp gate; 0 = exact gradient (default).
void SetSteReluCUDA(int v);

// `--detach_res_shape_grad`: drive the alpha/shape gradient from SV only
// (detach the MLP residual from surfel-shape gradients). Backward-only.
void SetDetachResShapeGradCUDA(int v);

// `--lru`: leaky-ReLU slope α for the outer per-Gauss activation (mode 0).
// α == 0 (default) → standard ReLU. α > 0 → forward + backward leak through
// negative activations scaled by α.
void SetLruSlopeCUDA(float v);

// Set Nexels-style anti-aliasing params (hash-grid down-weighting)
void SetAntiAliasCUDA(float factor, float focal);

// FastGS Compact Box Mahalanobis² scale factor for AdR cutoff.
// val=1.0 = our current AdR; val=0.5 = FastGS paper default (tighter AABB, faster rasterization).
void SetCompactMultCUDA(float val);
void SetBetaMultCUDA(float val);

// Set AA-2DGS mip filter kernel size σ (0 disables, typical 0.1).
void SetAaKernelSizeCUDA(float val);

// Periodic-freeze toggle for mode 5 backward. True = skip all hash/MLP
// gradient work this iter (weight-grad GEMMs, input-chain backprop,
// query_feature<true>, tile flush). False = normal backward.
void SetSkipMlpGradCUDA(bool val);

// Set depth sort toggle (true = separated depth sort, false = standard sort)
void SetDepthSortCUDA(bool val);

// ============================================================================
// BACKWARD KERNEL PROFILING
// ============================================================================

// Reset profiling counters to zero
void ResetBackwardProfileCUDA();

// Read profiling data: returns dict with cycle counts and counters
// cycles[0..4]: Phase A, B, C, flush, total (sum across all blocks)
// counts[0..3]: gaussians_processed, gaussians_skipped, tiles, intersections
std::tuple<torch::Tensor, torch::Tensor> ReadBackwardProfileCUDA();