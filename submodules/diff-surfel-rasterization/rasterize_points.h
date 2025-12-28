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
	const int max_intersections);

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
	const int render_mode,
	const torch::Tensor& shape_dims);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);