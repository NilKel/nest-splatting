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
	const torch::Tensor& gaussian_features,
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
	const int render_mode,
	const int hybrid_levels,
	const uint32_t Level,
	const float LevelScale,
	const uint32_t Base,
	const bool align_corners,
	const uint32_t interp)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  uint32_t C = 3;
  
  if (colors.numel() != 0) C = colors.size(1);
  
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
  CHECK_INPUT(gaussian_features);

  uint32_t D = 0;
  uint32_t hash_levels = 0;
  uint32_t effective_Level = Level;  // Level to pass to CUDA kernel

  // Check if we actually have hashgrid features (not just Level > 0)
  bool has_hashgrid = (Level > 0 && features.numel() > 0);
  
  if(has_hashgrid){
	D = features.size(1);
	hash_levels = offsets.size(0) - 1;  // Actual number of hashgrid levels from encoder

	// In cat mode, C is total output dimension (Gaussian + Hash features)
	// In baseline/add mode, C is based on the actual hashgrid levels
	if(render_mode == 2){
		// Cat mode: effective_Level = Level - hybrid_levels (effective hashgrid levels)
		// C = Level * D (total output always equals total_levels * D)
		// Example: Level=6, hybrid_levels=5 → effective_Level=1, C=24
		effective_Level = Level - hybrid_levels;  // Effective hashgrid levels
		C = Level * D;  // Total output dimension (total_levels * per_level_dim)
	} else {
		// Baseline/add mode: output dimension is based on actual hashgrid levels
		C = hash_levels * D;  // Hash feature dimension
		effective_Level = Level;  // Use Level as-is for baseline/add mode
	}

    // if (Level + 1 != offsets.size(0)) {
	//   AT_ERROR("offsets shoud have same level with hash features.");
    // }
  }
  else if(render_mode == 2 && gaussian_features.numel() > 0){
	// Cat mode with Level=0 or hybrid_levels >= Level (all features from Gaussians)
	// This handles the case where there's no hashgrid at all
	C = gaussian_features.size(1);  // e.g., 24D all from Gaussians
	effective_Level = 0;  // No hashgrid levels
	D = (C > 0 && Level > 0) ? (C / Level) : 0;  // Infer D from total dim
  }
  
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

//   torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_color = torch::full({C, H, W}, 0.0, float_opts);
  
  int out_dim = 3+3+1+1 + 3 + 3; // record mean_pts & appearance vis color, 
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
		gaussian_features.contiguous().data<float>(),
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
		render_mode,
		hybrid_levels);
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
	const torch::Tensor& gaussian_features,
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
	const int render_mode,
	const int hybrid_levels,
	const uint32_t Level,
	const float LevelScale,
	const uint32_t Base,
	const bool align_corners,
	const uint32_t interp
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

  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  uint32_t C = 3;
  if (colors.numel() != 0) C = colors.size(1);
  
  uint32_t D = 0;
  uint32_t table_size = 0;
  uint32_t hash_levels = 0;
  uint32_t effective_Level = Level;  // Level to pass to CUDA kernel

  // Check if we actually have hashgrid features (not just Level > 0)
  bool has_hashgrid = (Level > 0 && features.numel() > 0);
  
  if(has_hashgrid){
	table_size = features.size(0);
	D = features.size(1);
	hash_levels = offsets.size(0) - 1;  // Actual number of hashgrid levels from encoder

	// In cat mode, C is total output dimension (Gaussian + Hash features)
	// In baseline/add mode, C is based on the actual hashgrid levels
	if(render_mode == 2){
		// Cat mode: effective_Level = Level - hybrid_levels (effective hashgrid levels)
		// C = Level * D (total output always equals total_levels * D)
		// Example: Level=6, hybrid_levels=5 → effective_Level=1, C=24
		effective_Level = Level - hybrid_levels;  // Effective hashgrid levels
		C = Level * D;  // Total output dimension (total_levels * per_level_dim)
	} else {
		// Baseline/add mode: output dimension is based on actual hashgrid levels
		C = hash_levels * D;
		effective_Level = Level;  // Use Level as-is for baseline/add mode
	}
    // if (Level + 1 != offsets.size(0)) {
	//   AT_ERROR("offsets shoud have same level with hash features.");
    // }
  }
  else if(render_mode == 2 && gaussian_features.numel() > 0){
	// Cat mode with Level=0 or hybrid_levels >= Level (all features from Gaussians)
	// This handles the case where there's no hashgrid at all
	C = gaussian_features.size(1);  // e.g., 24D all from Gaussians
	effective_Level = 0;  // No hashgrid levels
	D = (C > 0 && Level > 0) ? (C / Level) : 0;  // Infer D from total dim
  }
  
  
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
//   torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, C}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dhomoMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  torch::Tensor dL_dfeatures = torch::zeros({table_size, D}, means3D.options());

  torch::Tensor dL_gradsum = torch::zeros({P, 1}, means3D.options());

  // Initialize gradient for per-Gaussian features
  uint32_t gaussian_feat_dim = 0;
  if(gaussian_features.numel() > 0){
    gaussian_feat_dim = gaussian_features.size(1);
  }
  torch::Tensor dL_dgaussian_features = torch::zeros({P, gaussian_feat_dim}, means3D.options());
  
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
	  gaussian_features.contiguous().data<float>(),
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
	  dL_dgaussian_features.contiguous().data<float>(),
	  debug,
	  beta,
	  render_mode,
	  hybrid_levels);
  }

  return std::make_tuple(dL_dfeatures, dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dtransMat, dL_dsh, dL_dscales, dL_drotations, dL_gradsum, dL_dgaussian_features);
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
