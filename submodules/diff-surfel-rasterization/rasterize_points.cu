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
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
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

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
	const torch::Tensor& adaptive_mask)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  uint32_t C = 3;

  // For hybrid_features mode (render_mode=4) and adaptive mode (render_mode=6), 
  // C will be calculated after determining D from features
  // For other modes, use colors dimension if available
  if (colors.numel() != 0 && render_mode != 4 && render_mode != 6) C = colors.size(1);
  
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

  uint32_t D = 0;
  uint32_t D_diffuse = 0;
  bool has_dual_hashgrid = (features_diffuse.numel() > 0 && offsets_diffuse.numel() > 0);
  
  if(Level > 0){
	D = features.size(1);  // Main hashgrid features
	
	if(has_dual_hashgrid && render_mode == 5){  // surface_rgb mode
		// Surface RGB mode with two separate hashgrids
		// Hashgrid 1: 12 per level (surface vectors) -> dot product -> 4 per level
		// Hashgrid 2: 4 per level (baseline features)
		// Output: combined features (baseline + surface) per level
		D_diffuse = features_diffuse.size(1);  // Should be 4 (baseline features)
		C = (offsets.size(0) - 1) * 4;  // 6 levels × 4 = 24D combined features
	}
	else if(has_dual_hashgrid && render_mode == 2){  // baseline_double mode
		// baseline_double mode with two 4D hashgrids
		// Hashgrid 1: 4 per level (queried at xyz, per-sample)
		// Hashgrid 2: 4 per level (queried at pk, per-Gaussian)
		// Output: feat1 + feat2 (element-wise), concatenated across levels
		D_diffuse = features_diffuse.size(1);  // Should be 4
		// Calculate output dimension from BOTH hashgrids
		uint32_t num_levels_1 = offsets.size(0) - 1;
		uint32_t num_levels_2 = offsets_diffuse.size(0) - 1;
		C = num_levels_1 * D + num_levels_2 * D_diffuse;  // e.g., 3*4 + 3*4 = 24D
	}
	else if(has_dual_hashgrid && render_mode == 3){  // baseline_blend_double mode
		// baseline_blend_double mode with two 4D hashgrids
		// Hashgrid 1: 4 per level (queried at blended position, AFTER alpha blending)
		// Hashgrid 2: 4 per level (queried at pk, per-Gaussian, alpha blended)
		// Output: blended feat_pk + feat_spatial, concatenated across levels
		D_diffuse = features_diffuse.size(1);  // Should be 4
		// Calculate output dimension from BOTH hashgrids
		uint32_t num_levels_1 = offsets.size(0) - 1;
		uint32_t num_levels_2 = offsets_diffuse.size(0) - 1;
		C = num_levels_1 * D + num_levels_2 * D_diffuse;  // e.g., 3*4 + 3*4 = 24D
	}
	else if(render_mode == 4) {
		// Hybrid features mode: hybrid_levels×D per-Gaussian + active_hashgrid_levels×D hashgrid
		// Output dimension: total_levels×D (always constant, e.g., 24D)
		// Level from Python is encoded as: (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
		// e.g., for total=6, hybrid=3, active=2: (6 << 16) | (2 << 8) | 3 = 393987
		// Buffer is always total_levels×D; unused dimensions are zero-padded during coarse-to-fine

		uint32_t total_levels = Level >> 16;  // Extract bits 16-31
		C = total_levels * D;  // e.g., 6 × 4 = 24D (constant regardless of active_levels)
	}
	else if(render_mode == 6) {
		// Adaptive mode: soft blend per-Gaussian and hashgrid features
		// Input: colors_precomp = [adaptive_features (total_levels×D) | mask (total_levels×D)]
		// Output: blended features (total_levels×D)
		// Output dimension is ALWAYS total_levels * D, same as baseline/cat mode
		uint32_t num_levels = offsets.size(0) - 1;  // total_levels from hashgrid
		C = num_levels * D;  // e.g., 6 × 4 = 24D (same as baseline)
	}
	else {
		// Single hashgrid modes
		// Determine output dimensions based on render_mode
		uint32_t effective_D = D;
		if(render_mode == 1) {
			// Surface mode: 12 vector features -> dot product in CUDA -> 4 scalar features per level
			effective_D = 4;
		} else if(render_mode == 5) {
			// Surface RGB mode: 12 vectors + 3 RGB -> 4 scalars + 3 RGB per level
			effective_D = 7;
		} else {
			// Baseline mode (render_mode == 0): use D as-is (supports surface_blend with D=12)
			effective_D = D;
		}

		C = (offsets.size(0) - 1) * effective_D;
	}
  }
  
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

//   torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_color = torch::full({C, H, W}, 0.0, float_opts);
  
  int out_dim = 3+3+1+1 + 3 + 3; // record mean_pts & appearance vis color
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

	  rendered = CudaRasterizer::Rasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, M,
		background.contiguous().data<float>(),
		W, H, C, Level, D, LevelScale, Base, align_corners, interp, if_contract, record_transmittance,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(), 
		homotrans.contiguous().data<float>(),
		ap_level.contiguous().data<float>(),
		features.contiguous().data<float>(), 
		offsets.contiguous().data<int>(), 
		gridrange.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
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
		render_mode);
  }
  return std::make_tuple(rendered, out_color, out_others, out_index, radii, geomBuffer, binningBuffer, imgBuffer, cover_pixels, trans_avg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
	const int render_mode
	) 
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
  uint32_t C = 3;
  if (colors.numel() != 0) C = colors.size(1);
  
  uint32_t D = 0;
  uint32_t D_diffuse = 0;
  uint32_t table_size = 0;
  uint32_t table_size_diffuse = 0;
  
  bool has_dual_hashgrid = (features_diffuse.numel() > 0 && offsets_diffuse.numel() > 0);

  if(Level > 0){
	table_size = features.size(0);
	D = features.size(1);
	
	if(has_dual_hashgrid && render_mode == 5){  // surface_rgb mode
		// Surface RGB mode with two separate hashgrids
		D_diffuse = features_diffuse.size(1);  // Should be 4 (baseline features)
		table_size_diffuse = features_diffuse.size(0);
		C = (offsets.size(0) - 1) * 4;  // 6 levels × 4 = 24D combined features
	}
	else if(has_dual_hashgrid && render_mode == 2){  // baseline_double mode
		// baseline_double mode with two 4D hashgrids
		D_diffuse = features_diffuse.size(1);  // Should be 4
		table_size_diffuse = features_diffuse.size(0);
		// Calculate output dimension from BOTH hashgrids
		uint32_t num_levels_1 = offsets.size(0) - 1;
		uint32_t num_levels_2 = offsets_diffuse.size(0) - 1;
		C = num_levels_1 * D + num_levels_2 * D_diffuse;  // e.g., 3*4 + 3*4 = 24D
	}
	else if(has_dual_hashgrid && render_mode == 3){  // baseline_blend_double mode
		// baseline_blend_double mode with two 4D hashgrids
		D_diffuse = features_diffuse.size(1);  // Should be 4
		table_size_diffuse = features_diffuse.size(0);
		// Calculate output dimension from BOTH hashgrids
		uint32_t num_levels_1 = offsets.size(0) - 1;
		uint32_t num_levels_2 = offsets_diffuse.size(0) - 1;
		C = num_levels_1 * D + num_levels_2 * D_diffuse;  // e.g., 3*4 + 3*4 = 24D
	}
	else if(render_mode == 4) {
		// Hybrid features mode: hybrid_levels×D per-Gaussian + active_hashgrid_levels×D hashgrid
		// Output dimension: total_levels×D (always constant, e.g., 24D)
		// Level from Python is encoded as: (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
		// e.g., for total=6, hybrid=3, active=2: (6 << 16) | (2 << 8) | 3 = 393987
		// Buffer is always total_levels×D; unused dimensions are zero-padded during coarse-to-fine
		uint32_t total_lvls = Level >> 16;
		C = total_lvls * D;  // e.g., 6 × 4 = 24D
		uint32_t total_levels = Level >> 16;  // Extract bits 16-31
		C = total_levels * D;  // e.g., 6 × 4 = 24D (constant regardless of active_levels)
	}
	else {
	// Single hashgrid modes
	// Determine output dimensions based on render_mode
	uint32_t effective_D = D;
	if(render_mode == 1) {
		// Surface mode: 12 vector features -> dot product in CUDA -> 4 scalar features per level
		effective_D = 4;
	} else if(render_mode == 5) {
		// Surface RGB mode: 12 vectors + 3 RGB -> 4 scalars + 3 RGB per level
		effective_D = 7;
	} else {
		// Baseline mode (render_mode == 0): use D as-is (supports surface_blend with D=12)
		effective_D = D;
	}

	C = (offsets.size(0) - 1) * effective_D;
	}
    // if (Level + 1 != offsets.size(0)) {
	//   AT_ERROR("offsets shoud have same level with hash features.");
    // }
  }
  
  
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
//   torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  // For hybrid_features (render_mode=4): dL_dcolors is hybrid_levels×D (per-Gaussian features only)
  // For other modes: dL_dcolors is C-dimensional
  uint32_t colors_dim = C;
  if (render_mode == 4) {
	// Decode hybrid_levels from Level parameter
	// Level encoding: (total_levels << 16) | (active_hashgrid_levels << 8) | hybrid_levels
	uint32_t hybrid_levels = Level & 0xFF;  // Extract bits 0-7
	colors_dim = hybrid_levels * D;  // e.g., 3 × 4 = 12D
  }
  torch::Tensor dL_dcolors = torch::zeros({P, colors_dim}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dhomoMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  torch::Tensor dL_dfeatures = torch::zeros({table_size, D}, means3D.options());
  torch::Tensor dL_dfeatures_diffuse = torch::zeros({table_size_diffuse, D_diffuse}, means3D.options());

  torch::Tensor dL_gradsum = torch::zeros({P, 1}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, C, Level, D, LevelScale, Base, align_corners, interp, if_contract, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  transMat_precomp.contiguous().data<float>(),
	  homotrans.contiguous().data<float>(),
	  ap_level.contiguous().data<float>(),
	  features.contiguous().data<float>(), 
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
	  render_mode);
  }

  return std::make_tuple(dL_dfeatures, dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dtransMat, dL_dsh, dL_dscales, dL_drotations, dL_gradsum, dL_dfeatures_diffuse);
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
