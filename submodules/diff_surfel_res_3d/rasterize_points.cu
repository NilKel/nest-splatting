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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "cuda_rasterizer/backward.h"
#include "cuda_rasterizer/forward.h"
#include <fstream>
#include <string>
#include <functional>

#define CHECK_INPUT(x)											\
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
	const torch::Tensor& metric_map,
	const torch::Tensor& is_textured,
	const torch::Tensor& scaling_z)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  // Extract dimensions from shape_dims tensor [GS, HS, OS]
  uint32_t GS = 0, HS = 0, OS = 3;  // Defaults
  
  if (shape_dims.numel() == 3) {
    GS = shape_dims[0].item<int>();
    HS = shape_dims[1].item<int>();
    OS = shape_dims[2].item<int>();
  }
  
  // Use OS for output buffer allocation (single or combined output)
  uint32_t C = OS;
  
  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(colors);
  CHECK_INPUT(opacity);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(homotrans);
  CHECK_INPUT(ap_level);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);

  CHECK_INPUT(features);
  CHECK_INPUT(offsets);
  CHECK_INPUT(gridrange);
  
  CHECK_INPUT(features_diffuse);
  CHECK_INPUT(offsets_diffuse);
  CHECK_INPUT(gridrange_diffuse);

  // Beta kernel shapes (optional)
  if (shapes.numel() > 0) {
    CHECK_INPUT(shapes);
  }

  // Keep D for hashgrid feature dimension (still needed for CUDA kernels)
  uint32_t D = 0;
  uint32_t D_diffuse = 0;
  bool has_dual_hashgrid = (features_diffuse.numel() > 0 && offsets_diffuse.numel() > 0);
  
  if(Level > 0){
	D = features.size(1);  // Main hashgrid features per level
	if(has_dual_hashgrid){
		D_diffuse = features.size(1);  // Diffuse hashgrid features per level
	}
  }
  
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  // For dual buffer modes (residual_hybrid), allocate separately using GS and HS
  torch::Tensor out_color;
  torch::Tensor out_gaussian_rgb;  // For residual_hybrid
  bool use_dual_buffers = (render_mode == 11 && GS > 0 && HS > 0 && OS == 0);
  
  if (use_dual_buffers) {
    // Residual_hybrid: two separate buffers
    out_gaussian_rgb = torch::full({GS, H, W}, 0.0, float_opts);
    out_color = torch::full({HS, H, W}, 0.0, float_opts);
    C = HS;  // Set C for hashgrid buffer
  } else {
    // All other modes: single buffer
    out_color = torch::full({C, H, W}, 0.0, float_opts);
  }
  
  // `--method res_3d` dual-cascade output buffers. Allocated only when
  // res_3d's single-pass dispatch is active (render_mode bit signal — TBD;
  // for now always allocated and nullptr-gated downstream when not desired).
  torch::Tensor out_color_sv_aux = torch::full({C, H, W}, 0.0, float_opts);
  torch::Tensor out_color_tex_aux = torch::full({C, H, W}, 0.0, float_opts);

  int out_dim = 3+3+1+1 + 3 + 3 + 1 + 1 + 1 + 1; // + w_square_sum + beta_sum (--w_lambda_perpix)
    if((has_dual_hashgrid && render_mode == 5) || (has_dual_hashgrid && render_mode == 2)) {  // surface_rgb or baseline_double mode
    // No extra channels needed - features are already in out_color
  }
  torch::Tensor out_others = torch::full({out_dim, H, W}, 0.0, float_opts);
  torch::Tensor out_index = torch::full({H, W}, 0.0, int_opts);

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  int record_P = P;
  if(record_transmittance == false) record_P = 0;
  torch::Tensor cover_pixels = torch::full({record_P, 1}, 0, float_opts);
  torch::Tensor trans_avg = torch::full({record_P, 1}, 0, float_opts);

  // FastGS VCD/VCP per-Gaussian counter. Always allocated (shape [P]) so the Python
  // tuple layout is stable; only written when metric_map is provided.
  const bool fastgs_enabled = metric_map.numel() > 0;
  torch::Tensor metric_counts = torch::zeros({P}, int_opts);
  if (fastgs_enabled) {
    TORCH_CHECK(metric_map.numel() == H * W, "metric_map must have H*W entries");
    CHECK_INPUT(metric_map);
  }

  // 3D mode intersection buffer allocation (render_mode == 3)
  torch::Tensor intersection_buffer;
  torch::Tensor intersection_count;
  uint32_t max_intersections_alloc = 0;
  if (render_mode == 3 && max_intersections_per_pixel > 0) {
    max_intersections_alloc = (uint32_t)max_intersections_per_pixel;
    // Padded layout: each pixel gets max_intersections_per_pixel slots
    // Each slot is 12 floats: [gaussian_id, weight, pixel_id, xyz.x, xyz.y, xyz.z, s_x, s_y, rho_flag, alpha, T, G]
    // xyz.x/y/z = world-space intersection point for hash query (exact match with CAT mode)
    // s_x, s_y = disk coordinates for backward gradient computation
    // rho_flag = 1.0 if disk intersection, 0.0 if center fallback
    // alpha = opacity * G, T = transmittance, G = kernel value
    int64_t total_slots = (int64_t)H * W * max_intersections_alloc;
    intersection_buffer = torch::zeros({total_slots, 12}, float_opts);
    intersection_count = torch::zeros({H * W}, int_opts);
  } else {
    // Allocate empty tensors for non-3D modes
    intersection_buffer = torch::empty({0}, float_opts);
    intersection_count = torch::empty({0}, int_opts);
  }

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
	  }

	  // For baseline hashgrid mode, colors is empty - pass nullptr instead of invalid pointer
	  const float* colors_ptr = (colors.numel() > 0) ? colors.contiguous().data<float>() : nullptr;

	  // Beta kernel shapes pointer (nullptr if not using beta kernel)
	  const float* shapes_ptr = (shapes.numel() > 0) ? shapes.contiguous().data<float>() : nullptr;

	  // Convert hash features to FP16 for bandwidth reduction
	  auto features_half = (features.numel() > 0) ? features.contiguous().to(torch::kHalf) : features;

	  // `--method mixed` per-Gauss bool flag. Empty tensor → nullptr (all-textured behavior).
	  const bool* is_textured_ptr = (is_textured.numel() > 0)
	      ? is_textured.contiguous().data<bool>() : nullptr;

	  // `--method mixed_3d` per-Gauss activated 3rd-axis scale [P]. Empty tensor
	  // → nullptr → untextured surfels keep 2DGS ray-splat geometry.
	  const float* scaling_z_ptr = (scaling_z.numel() > 0)
	      ? scaling_z.contiguous().data<float>() : nullptr;

	  rendered = CudaRasterizer::Rasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, M,
		background.contiguous().data<float>(),
		W, H, C, Level, D, LevelScale, Base, align_corners, interp, if_contract, record_transmittance,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors_ptr,
		opacity.contiguous().data<float>(),
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(),
		homotrans.contiguous().data<float>(),
		ap_level.contiguous().data<float>(),
		reinterpret_cast<const __half*>(features_half.data_ptr()),
		offsets.contiguous().data<int>(),
		gridrange.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_color_sv_aux.contiguous().data<float>(),
		out_color_tex_aux.contiguous().data<float>(),
		out_others.contiguous().data<float>(),
		out_index.contiguous().data<int>(),
		radii.contiguous().data<int>(),
		cover_pixels.contiguous().data<float>(),
		trans_avg.contiguous().data<float>(),
		debug,
		beta,
		D_diffuse,
		features_diffuse.contiguous().data<float>(),
		offsets_diffuse.contiguous().data<int>(),
		gridrange_diffuse.contiguous().data<float>(),
		render_mode,
		(uint32_t)max_intersections,
		shapes_ptr,
		kernel_type,
		aabb_mode,
		aa,
		aa_threshold,
		(render_mode == 3) ? intersection_buffer.contiguous().data<float>() : nullptr,
		(render_mode == 3) ? (uint32_t*)intersection_count.contiguous().data<int>() : nullptr,
		max_intersections_alloc,
		fastgs_enabled ? metric_map.contiguous().data<int>() : nullptr,
		fastgs_enabled ? metric_counts.contiguous().data<int>() : nullptr,
		is_textured_ptr,
		scaling_z_ptr);
  }

  return std::make_tuple(rendered, out_color, out_color_sv_aux, out_color_tex_aux, out_others, out_index, radii, geomBuffer, binningBuffer, imgBuffer, cover_pixels, trans_avg, intersection_buffer, intersection_count, metric_counts);
}

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
	// `--l2` (mixed_3d only): per-Gauss image-grad routing. Empty tensor →
	// nullptr → every Gauss reads `dL_dout_color` (byte-identical to pre-flag).
	const torch::Tensor& dL_dout_color_untex,
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
	const bool detach_hash_grad,
	const torch::Tensor& is_textured,
	const torch::Tensor& scaling_z)
{

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(radii);
  CHECK_INPUT(colors);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(homotrans);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);
  CHECK_INPUT(binningBuffer);
  CHECK_INPUT(imageBuffer);
  CHECK_INPUT(geomBuffer);
  
  CHECK_INPUT(features_diffuse);
  CHECK_INPUT(offsets_diffuse);
  CHECK_INPUT(gridrange_diffuse);

  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  // Extract dimensions from shape_dims tensor [GS, HS, OS]
  uint32_t GS = 0, HS = 0, OS = 3;  // Defaults
  
  if (shape_dims.numel() == 3) {
    GS = shape_dims[0].item<int>();
    HS = shape_dims[1].item<int>();
    OS = shape_dims[2].item<int>();
  }
  
  // Use OS for output dimension (should match forward pass)
  uint32_t C = OS;
  
  uint32_t D = 0;
  uint32_t D_diffuse = 0;
  uint32_t table_size = 0;
  uint32_t table_size_diffuse = 0;
  
  bool has_dual_hashgrid = (features_diffuse.numel() > 0 && offsets_diffuse.numel() > 0);

  if(Level > 0){
	table_size = features.size(0);
	D = features.size(1);
	
	if(has_dual_hashgrid){
		D_diffuse = features_diffuse.size(1);
		table_size_diffuse = features_diffuse.size(0);
	}
	
	// NOTE: C is already set from shape_dims[2] (OS) above
	// Do NOT recalculate C here - use the explicit value from Python
  }
  
  
  
  int M = 0;
  if(sh.size(0) != 0)
  {
	M = sh.size(1);
  } else if (sh.dim() >= 2) {
	// P=0 but SH tensor still has shape info — preserve M for gradient shape
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());
  // For gradients, use actual input dimension from colors_precomp
  // This matches whatever was passed in (26D for adaptive_cat, 48D for residual_hybrid, etc.)
  int colors_dim = 0;
  
  if (colors.numel() > 0) {
    colors_dim = colors.size(1);  // Actual input dimension
  }
  
  // For baseline mode with no colors_precomp, allocate dummy dL_dcolors to avoid empty tensor access
  // The kernel will write to it but we won't use the gradients
  // Must allocate at least C columns since kernel indexes as [global_id * C + ch]
  if (colors_dim == 0) {
    colors_dim = C;  // Match output channels to avoid illegal access
  }
  torch::Tensor dL_dcolors = torch::zeros({P, colors_dim}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dhomoMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  // `--method mixed_3d` 3rd-axis scale gradient (untextured EWA rows).
  torch::Tensor dL_dscaling_z = torch::zeros({P, 1}, means3D.options());

  torch::Tensor dL_dfeatures = torch::zeros({table_size, D}, means3D.options());
  torch::Tensor dL_dfeatures_diffuse = torch::zeros({table_size_diffuse, D_diffuse}, means3D.options());

  torch::Tensor dL_gradsum = torch::zeros({P, 1}, means3D.options());

  // Kernel shape gradients: [P, 1] for beta/general/flex, [P, 2] for nexel (gamma_x, gamma_y)
  int shape_dim = (kernel_type == 5) ? 2 : 1;
  torch::Tensor dL_dshapes = torch::zeros({P, shape_dim}, means3D.options());

  // MLP gradient buffers for 3D_SH_res mode (render_mode=5, bias-free, all [16×16])
  // Architecture: 16D input (4D hash + 1.0 bias + 11 zero pad) -> 16 hidden -> 16 hidden -> 16 output (first 3 = RGB residual)
  torch::Tensor dL_dmlp_W1 = torch::zeros({16, 16}, means3D.options());  // Layer 1 weights [16×16]
  torch::Tensor dL_dmlp_W2 = torch::zeros({16, 16}, means3D.options());  // Layer 2 weights [16×16]
  torch::Tensor dL_dmlp_W3 = torch::zeros({16, 16}, means3D.options());  // Layer 3 weights [16×16]

  if(P != 0)
  {
	  // For baseline hashgrid mode, colors is empty - pass nullptr instead of invalid pointer
	  const float* colors_ptr_bw = (colors.numel() > 0) ? colors.contiguous().data<float>() : nullptr;

	  // Beta kernel shapes pointer
	  const float* shapes_ptr_bw = (shapes.numel() > 0) ? shapes.contiguous().data<float>() : nullptr;

	  // Convert hash features to FP16 for bandwidth reduction
	  auto features_half_bw = (features.numel() > 0) ? features.contiguous().to(torch::kHalf) : features;

	  // `--method mixed` per-Gauss bool flag.
	  const bool* is_textured_ptr_bw = (is_textured.numel() > 0)
	      ? is_textured.contiguous().data<bool>() : nullptr;

	  // `--method mixed_3d` activated 3rd-axis scale (nullptr → 2DGS path).
	  const float* scaling_z_ptr_bw = (scaling_z.numel() > 0)
	      ? scaling_z.contiguous().data<float>() : nullptr;

	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, C, Level, D, LevelScale, Base, align_corners, interp, if_contract,
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors_ptr_bw,
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  transMat_precomp.contiguous().data<float>(),
	  homotrans.contiguous().data<float>(),
	  ap_level.contiguous().data<float>(),
	  reinterpret_cast<const __half*>(features_half_bw.data_ptr()),
	  offsets.contiguous().data<int>(),
	  gridrange.contiguous().data<float>(), 
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  other_maps.contiguous().data<float>(),
	  out_index.contiguous().data<int>(),
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  // `--l2` per-Gauss routing: pass nullptr for the empty-tensor default so
	  // the kernel reverts to single-loss behavior.
	  (dL_dout_color_untex.numel() > 0)
		  ? dL_dout_color_untex.contiguous().data<float>()
		  : nullptr,
	  dL_dout_others.contiguous().data<float>(),
	  dL_dfeatures.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dnormal.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dtransMat.contiguous().data<float>(),
	  dL_dhomoMat.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  dL_gradsum.contiguous().data<float>(),
	  debug,
	  beta,
	  D_diffuse,
	  features_diffuse.contiguous().data<float>(),
	  offsets_diffuse.contiguous().data<int>(),
	  gridrange_diffuse.contiguous().data<float>(),
	  dL_dfeatures_diffuse.contiguous().data<float>(),
	  render_mode,
	  shapes_ptr_bw,
	  kernel_type,
	  dL_dshapes.contiguous().data<float>(),
	  detach_hash_grad,
	  // MLP gradient buffers for 3D_SH_res/3D_SH_cat mode (bias-free, all [16×16])
	  // Bit 9 (0x200) = freeze_mlp: skip weight gradients (pass nullptr)
	  ((render_mode & 0xFF) == 5 || (render_mode & 0xFF) == 6) && !(render_mode & 0x200) ? dL_dmlp_W1.contiguous().data<float>() : nullptr,
	  ((render_mode & 0xFF) == 5 || (render_mode & 0xFF) == 6) && !(render_mode & 0x200) ? dL_dmlp_W2.contiguous().data<float>() : nullptr,
	  ((render_mode & 0xFF) == 5 || (render_mode & 0xFF) == 6) && !(render_mode & 0x200) ? dL_dmlp_W3.contiguous().data<float>() : nullptr,
	  is_textured_ptr_bw,
	  scaling_z_ptr_bw,
	  dL_dscaling_z.contiguous().data<float>());
  }

  return std::make_tuple(dL_dfeatures, dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dtransMat, dL_dsh, dL_dscales, dL_drotations, dL_gradsum, dL_dfeatures_diffuse, dL_dshapes,
                         dL_dmlp_W1, dL_dmlp_W2, dL_dmlp_W3, dL_dscaling_z);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

// Forward declaration of kernels from backward.cu
__global__ void compute_opacity_gradient_3D_kernel(
    int M, int N, int num_pixels,
    const float* __restrict__ dL_dweight,
    const float* __restrict__ T_values,
    const float* __restrict__ G_values,
    const float* __restrict__ alpha_values,
    const int* __restrict__ gaussian_ids,
    const int* __restrict__ pixel_starts,
    float* __restrict__ dL_dopacity,
    float* __restrict__ dL_dalpha_out);

__global__ void compute_geometry_gradient_3D_kernel(
    int M, int N, int W, int H,
    const float* __restrict__ dL_dalpha,
    const float* __restrict__ opacity_values,
    const float* __restrict__ G_values,
    const float* __restrict__ s_x_values,
    const float* __restrict__ s_y_values,
    const float* __restrict__ rho_flag,
    const int* __restrict__ gaussian_ids,
    const int* __restrict__ pixel_ids,
    const float* __restrict__ transMat,
    float* __restrict__ dL_dtransMat);

// PyTorch wrapper for 3D mode opacity gradient computation
// Returns: (dL_dopacity [N], dL_dalpha [M])
std::tuple<torch::Tensor, torch::Tensor> ComputeOpacityGradient3DCUDA(
    const torch::Tensor& dL_dweight,      // [M]
    const torch::Tensor& T_values,        // [M]
    const torch::Tensor& G_values,        // [M]
    const torch::Tensor& alpha_values,    // [M]
    const torch::Tensor& gaussian_ids,    // [M] int
    const torch::Tensor& pixel_starts,    // [num_pixels+1] int
    const int N)                          // num Gaussians
{
    CHECK_INPUT(dL_dweight);
    CHECK_INPUT(T_values);
    CHECK_INPUT(G_values);
    CHECK_INPUT(alpha_values);
    CHECK_INPUT(gaussian_ids);
    CHECK_INPUT(pixel_starts);

    int M = dL_dweight.size(0);
    int num_pixels = pixel_starts.size(0) - 1;

    torch::Tensor dL_dopacity = torch::zeros({N}, dL_dweight.options());
    torch::Tensor dL_dalpha = torch::zeros({M}, dL_dweight.options());

    if (M > 0 && num_pixels > 0) {
        int threads = 256;
        int blocks = (num_pixels + threads - 1) / threads;
        compute_opacity_gradient_3D_kernel<<<blocks, threads>>>(
            M, N, num_pixels,
            dL_dweight.contiguous().data_ptr<float>(),
            T_values.contiguous().data_ptr<float>(),
            G_values.contiguous().data_ptr<float>(),
            alpha_values.contiguous().data_ptr<float>(),
            gaussian_ids.contiguous().data_ptr<int>(),
            pixel_starts.contiguous().data_ptr<int>(),
            dL_dopacity.data_ptr<float>(),
            dL_dalpha.data_ptr<float>()
        );
    }

    return std::make_tuple(dL_dopacity, dL_dalpha);
}

// PyTorch wrapper for 3D mode geometry gradient computation
// Takes dL_dalpha per intersection and computes dL_dtransMat using geomBuffer
// Returns: dL_dtransMat [N, 9]
torch::Tensor ComputeGeometryGradient3DCUDA(
    const torch::Tensor& dL_dalpha,       // [M] per-intersection dL_dalpha
    const torch::Tensor& opacity_values,  // [M] per-intersection opacity
    const torch::Tensor& G_values,        // [M] kernel value
    const torch::Tensor& s_x_values,      // [M] intersection s.x
    const torch::Tensor& s_y_values,      // [M] intersection s.y
    const torch::Tensor& rho_flag,        // [M] 1.0=disk, 0.0=center
    const torch::Tensor& gaussian_ids,    // [M] int
    const torch::Tensor& pixel_ids,       // [M] int
    const torch::Tensor& transMat,        // [N, 9] from geomBuffer
    const int W, const int H,             // Image dimensions
    const int N)                          // num Gaussians
{
    CHECK_INPUT(dL_dalpha);
    CHECK_INPUT(opacity_values);
    CHECK_INPUT(G_values);
    CHECK_INPUT(s_x_values);
    CHECK_INPUT(s_y_values);
    CHECK_INPUT(rho_flag);
    CHECK_INPUT(gaussian_ids);
    CHECK_INPUT(pixel_ids);
    CHECK_INPUT(transMat);

    int M = dL_dalpha.size(0);

    torch::Tensor dL_dtransMat = torch::zeros({N, 9}, dL_dalpha.options());

    if (M > 0) {
        int threads = 256;
        int blocks = (M + threads - 1) / threads;
        compute_geometry_gradient_3D_kernel<<<blocks, threads>>>(
            M, N, W, H,
            dL_dalpha.contiguous().data_ptr<float>(),
            opacity_values.contiguous().data_ptr<float>(),
            G_values.contiguous().data_ptr<float>(),
            s_x_values.contiguous().data_ptr<float>(),
            s_y_values.contiguous().data_ptr<float>(),
            rho_flag.contiguous().data_ptr<float>(),
            gaussian_ids.contiguous().data_ptr<int>(),
            pixel_ids.contiguous().data_ptr<int>(),
            transMat.contiguous().data_ptr<float>(),
            dL_dtransMat.data_ptr<float>()
        );
    }

    return dL_dtransMat;
}

// PyTorch wrapper for unified 3D mode backward computation
// Takes dL_dweight from PyTorch, reads transMat from geomBuffer internally
// Also accepts dL_duv from hash/xyz gradient path (like cat mode)
// Returns: (dL_dopacity [N], dL_dtransMat [N, 9], dL_dmean2D [N, 2])
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BackwardFromWeightGradCUDA(
    const torch::Tensor& geomBuffer,      // Raw geomBuffer from forward
    const int P,                          // Number of Gaussians (for geomBuffer parsing)
    const torch::Tensor& dL_dweight,      // [M] gradients from PyTorch
    const torch::Tensor& gaussian_ids,    // [M] int
    const torch::Tensor& pixel_ids,       // [M] int
    const torch::Tensor& pixel_starts,    // [num_pixels+1] int
    const torch::Tensor& T_values,        // [M] transmittance
    const torch::Tensor& G_values,        // [M] kernel value
    const torch::Tensor& alpha_values,    // [M] alpha
    const torch::Tensor& opacity_values,  // [M] per-intersection opacity
    const torch::Tensor& s_x_values,      // [M] intersection s.x
    const torch::Tensor& s_y_values,      // [M] intersection s.y
    const torch::Tensor& rho_flag,        // [M] 1.0=disk, 0.0=center
    const torch::Tensor& dL_duv_x,        // [M] hash/xyz gradient contribution (can be empty)
    const torch::Tensor& dL_duv_y,        // [M] hash/xyz gradient contribution (can be empty)
    const int W, const int H)             // Image dimensions
{
    CHECK_INPUT(dL_dweight);
    CHECK_INPUT(gaussian_ids);
    CHECK_INPUT(pixel_ids);
    CHECK_INPUT(pixel_starts);
    CHECK_INPUT(T_values);
    CHECK_INPUT(G_values);
    CHECK_INPUT(alpha_values);
    CHECK_INPUT(opacity_values);
    CHECK_INPUT(s_x_values);
    CHECK_INPUT(s_y_values);
    CHECK_INPUT(rho_flag);

    int M = dL_dweight.size(0);
    int num_pixels = pixel_starts.size(0) - 1;

    // Allocate output tensors
    torch::Tensor dL_dopacity = torch::zeros({P}, dL_dweight.options());
    torch::Tensor dL_dtransMat = torch::zeros({P, 9}, dL_dweight.options());
    torch::Tensor dL_dmean2D = torch::zeros({P, 2}, dL_dweight.options());

    if (M > 0 && num_pixels > 0) {
        // Parse geomBuffer to get transMat pointer
        char* geom_buffer = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
        CudaRasterizer::GeometryState geomState = CudaRasterizer::GeometryState::fromChunk(geom_buffer, P);

        // Get pointers for dL_duv (can be nullptr if tensors are empty)
        const float* dL_duv_x_ptr = (dL_duv_x.numel() > 0) ? dL_duv_x.contiguous().data_ptr<float>() : nullptr;
        const float* dL_duv_y_ptr = (dL_duv_y.numel() > 0) ? dL_duv_y.contiguous().data_ptr<float>() : nullptr;

        // Call the unified backward kernel
        // Use pre-computed means2D from geomBuffer for consistent 2D fallback gradients
        backward_from_weight_grad(
            num_pixels, P, W, H,
            dL_dweight.contiguous().data_ptr<float>(),
            gaussian_ids.contiguous().data_ptr<int>(),
            pixel_ids.contiguous().data_ptr<int>(),
            pixel_starts.contiguous().data_ptr<int>(),
            T_values.contiguous().data_ptr<float>(),
            G_values.contiguous().data_ptr<float>(),
            alpha_values.contiguous().data_ptr<float>(),
            opacity_values.contiguous().data_ptr<float>(),
            s_x_values.contiguous().data_ptr<float>(),
            s_y_values.contiguous().data_ptr<float>(),
            rho_flag.contiguous().data_ptr<float>(),
            dL_duv_x_ptr,
            dL_duv_y_ptr,
            geomState.transMat,  // Read directly from geomBuffer!
            reinterpret_cast<const float*>(geomState.means2D),  // Pre-computed mean2D [N, 2]
            dL_dopacity.data_ptr<float>(),
            dL_dtransMat.data_ptr<float>(),
            dL_dmean2D.data_ptr<float>()
        );
    }

    return std::make_tuple(dL_dopacity, dL_dtransMat, dL_dmean2D);
}

// PyTorch wrapper for transMat to scale/rotation gradient conversion
// This properly converts screen-space dL_dtransMat to world-space dL_dscale and dL_drotation
// Also incorporates xyz gradient contribution (dL_dhomoMat) and 2D mean gradient (dL_dmean2D)
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
    const int W, const int H)           // Image dimensions for ndc2pix transformation
{
    CHECK_INPUT(dL_dtransMat);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations);
    CHECK_INPUT(projmatrix);

    int N = dL_dtransMat.size(0);

    // Allocate output tensors
    torch::Tensor dL_dscales = torch::zeros({N, 2}, dL_dtransMat.options());
    torch::Tensor dL_drots = torch::zeros({N, 4}, dL_dtransMat.options());
    torch::Tensor dL_dmeans = torch::zeros({N, 3}, dL_dtransMat.options());

    if (N > 0) {
        // Get pointers for optional gradient contributions (can be nullptr if tensor is empty)
        const float* dL_dhomoMat_ptr = (dL_dhomoMat.numel() > 0) ? dL_dhomoMat.contiguous().data_ptr<float>() : nullptr;
        const float* dL_dmean2D_ptr = (dL_dmean2D.numel() > 0) ? dL_dmean2D.contiguous().data_ptr<float>() : nullptr;
        const float* dL_dnormal3D_ptr = (dL_dnormal3D.numel() > 0) ? dL_dnormal3D.contiguous().data_ptr<float>() : nullptr;
        const float* means3D_ptr = (means3D.numel() > 0) ? means3D.contiguous().data_ptr<float>() : nullptr;
        const float* transMat_precomp_ptr = (transMat_precomp.numel() > 0) ? transMat_precomp.contiguous().data_ptr<float>() : nullptr;
        const float* viewmatrix_ptr = (viewmatrix.numel() > 0) ? viewmatrix.contiguous().data_ptr<float>() : nullptr;

        transMat_to_scale_rot_grad(
            N,
            W, H,  // Image dimensions for ndc2pix
            dL_dtransMat.contiguous().data_ptr<float>(),
            dL_dhomoMat_ptr,
            dL_dmean2D_ptr,
            dL_dnormal3D_ptr,
            means3D_ptr,
            transMat_precomp_ptr,
            scales.contiguous().data_ptr<float>(),
            rotations.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            viewmatrix_ptr,
            dL_dscales.data_ptr<float>(),
            dL_drots.data_ptr<float>(),
            dL_dmeans.data_ptr<float>()
        );
    }

    return std::make_tuple(dL_dscales, dL_drots, dL_dmeans);
}

// Extract transMat from geomBuffer
// Returns: transMat [P, 9] as a float tensor
torch::Tensor GetTransMatFromGeomBufferCUDA(
    const torch::Tensor& geomBuffer,
    const int P)
{
    CHECK_INPUT(geomBuffer);

    torch::Tensor transMat = torch::zeros({P, 9}, torch::TensorOptions().dtype(torch::kFloat32).device(geomBuffer.device()));

    if (P > 0) {
        char* geom_buffer = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
        CudaRasterizer::GeometryState geomState = CudaRasterizer::GeometryState::fromChunk(geom_buffer, P);

        // Copy transMat from geomBuffer to output tensor
        cudaMemcpy(transMat.data_ptr<float>(), geomState.transMat, P * 9 * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return transMat;
}

// ============================================================================
// MLP WEIGHT MANAGEMENT FOR FUSED MODES (3D_fused, 3D_direct_fused)
// ============================================================================

// Copy MLP weights from PyTorch tensors to CUDA global memory (bias-free, all [16×16])
// Call this before each render call if weights have changed
void SetMlpWeightsCUDA(
    const torch::Tensor& W1,      // [16, 16] - Layer 1 weights
    const torch::Tensor& W2,      // [16, 16] - Layer 2 weights
    const torch::Tensor& W3       // [16, 16] - Layer 3 weights (first 3 rows = RGB residual)
) {
    CHECK_INPUT(W1);
    CHECK_INPUT(W2);
    CHECK_INPUT(W3);

    // Ensure tensors are contiguous
    auto W1_c = W1.contiguous();
    auto W2_c = W2.contiguous();
    auto W3_c = W3.contiguous();

    // Call wrapper function in forward.cu where global memory is allocated
    FORWARD::setMlpWeights(
        W1_c.data_ptr<float>(),
        W2_c.data_ptr<float>(),
        W3_c.data_ptr<float>());
}

void SetContribThreshCUDA(float val) {
    FORWARD::setContribThresh(val);
    BACKWARD::setContribThresh(val);
}

void SetCountThreshCUDA(int val) {
    FORWARD::setCountThresh(val);
    BACKWARD::setCountThresh(val);
}

void SetOverdrawLambdaCUDA(float val) {
    FORWARD::setOverdrawLambda(val);
    BACKWARD::setOverdrawLambda(val);
}

void SetWeightRegLambdaCUDA(float val) {
    FORWARD::setWeightRegLambda(val);
    BACKWARD::setWeightRegLambda(val);
}

void SetActivationBiasCUDA(float sh_bias, float res_bias) {
    FORWARD::setActivationBias(sh_bias, res_bias);
    BACKWARD::setResBias(res_bias);
}

void SetResidualModeCUDA(int mode) {
    FORWARD::setResidualMode(mode);
    BACKWARD::setResidualMode(mode);
}

// `--ste`: straight-through estimator on the per-Gauss outer ReLU.
void SetSteReluCUDA(int v) {
    FORWARD::setSteRelu(v);
    BACKWARD::setSteRelu(v);
}

// `--method res_3d_double`: toggle the per-Gauss bias gate (tex sh_color → 0).
void SetTexturedBiasGateCUDA(int v) {
    FORWARD::setTexturedBiasGate(v);
    BACKWARD::setTexturedBiasGate(v);
}

// `--lru`: leaky-ReLU slope α for the outer per-Gauss activation (mode 0).
void SetLruSlopeCUDA(float v) {
    FORWARD::setLruSlope(v);
    BACKWARD::setLruSlope(v);
}

void SetAntiAliasCUDA(float factor, float focal) {
    FORWARD::setAntiAlias(factor, focal);
}

void SetCompactMultCUDA(float val) {
    FORWARD::setCompactMult(val);
}

void SetAaKernelSizeCUDA(float val) {
    FORWARD::setAaKernelSize(val);
    BACKWARD::setAaKernelSize(val);
}

// Periodic-freeze toggle: when true, backward skips all hash/MLP-grad work
// (weight-grad GEMMs, input-chain backprop, query_feature<true>, tile flush).
// Geometry backward runs as normal.
void SetSkipMlpGradCUDA(bool val) {
    BACKWARD::setSkipMlpGrad(val);
}

// Defined in rasterizer_impl.cu
extern bool g_depth_sort;

void SetDepthSortCUDA(bool val) {
    g_depth_sort = val;
}

// ============================================================================
// BACKWARD KERNEL PROFILING
// ============================================================================
// Host functions are in backward.cu (same compilation unit as __device__ symbols)

void ResetBackwardProfileCUDA() {
    resetBackwardProfile();  // defined in backward.cu
}

std::tuple<torch::Tensor, torch::Tensor> ReadBackwardProfileCUDA() {
    unsigned long long cycles[6];
    unsigned int counts[4];
    readBackwardProfile(cycles, counts);  // defined in backward.cu

    auto cycles_tensor = torch::zeros({6}, torch::dtype(torch::kInt64).device(torch::kCPU));
    auto counts_tensor = torch::zeros({4}, torch::dtype(torch::kInt32).device(torch::kCPU));

    for (int i = 0; i < 6; i++) cycles_tensor[i] = (int64_t)cycles[i];
    for (int i = 0; i < 4; i++) counts_tensor[i] = (int32_t)counts[i];

    return std::make_tuple(cycles_tensor, counts_tensor);
}
