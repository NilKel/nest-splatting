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
	const torch::Tensor& shape_dims,
	const int max_intersections,
	const torch::Tensor& shapes,
	const int kernel_type)
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

	  // For baseline hashgrid mode, colors is empty - pass nullptr instead of invalid pointer
	  const float* colors_ptr = (colors.numel() > 0) ? colors.contiguous().data<float>() : nullptr;

	  // Beta kernel shapes pointer (nullptr if not using beta kernel)
	  const float* shapes_ptr = (shapes.numel() > 0) ? shapes.contiguous().data<float>() : nullptr;

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
		render_mode,
		(uint32_t)max_intersections,
		shapes_ptr,
		kernel_type);
  }

  return std::make_tuple(rendered, out_color, out_others, out_index, radii, geomBuffer, binningBuffer, imgBuffer, cover_pixels, trans_avg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
	const int kernel_type) 
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
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
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

  torch::Tensor dL_dfeatures = torch::zeros({table_size, D}, means3D.options());
  torch::Tensor dL_dfeatures_diffuse = torch::zeros({table_size_diffuse, D_diffuse}, means3D.options());

  torch::Tensor dL_gradsum = torch::zeros({P, 1}, means3D.options());

  // Beta kernel shape gradients
  torch::Tensor dL_dshapes = torch::zeros({P, 1}, means3D.options());

  if(P != 0)
  {
	  // For baseline hashgrid mode, colors is empty - pass nullptr instead of invalid pointer
	  const float* colors_ptr_bw = (colors.numel() > 0) ? colors.contiguous().data<float>() : nullptr;

	  // Beta kernel shapes pointer
	  const float* shapes_ptr_bw = (shapes.numel() > 0) ? shapes.contiguous().data<float>() : nullptr;

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
	  render_mode,
	  shapes_ptr_bw,
	  kernel_type,
	  dL_dshapes.contiguous().data<float>());
  }

  return std::make_tuple(dL_dfeatures, dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dtransMat, dL_dsh, dL_dscales, dL_drotations, dL_gradsum, dL_dfeatures_diffuse, dL_dshapes);
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
