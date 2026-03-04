/*
 * Baked rendering submodule — PyTorch/CUDA binding (forward-only).
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
#include "cuda_rasterizer/forward.h"
#include <functional>

#define CHECK_INPUT(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,         // precomputed per-Gaussian colors [N, C] or empty
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
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
	const torch::Tensor& shapes,
	const int kernel_type,
	const torch::Tensor& residual_textures,
	const torch::Tensor& atlas_texture,
	const torch::Tensor& atlas_rects,
	const int atlas_width,
	const int aabb_mode,
	// Persistent buffers — pass empty on first call, reused on subsequent calls
	torch::Tensor geomBuffer,
	torch::Tensor binningBuffer,
	torch::Tensor imgBuffer)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	CHECK_INPUT(background);
	CHECK_INPUT(means3D);
	CHECK_INPUT(opacity);
	CHECK_INPUT(scales);
	CHECK_INPUT(rotations);
	CHECK_INPUT(viewmatrix);
	CHECK_INPUT(projmatrix);
	CHECK_INPUT(sh);
	CHECK_INPUT(campos);

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	// Kernel writes all inside pixels directly — no zeroing needed
	torch::Tensor out_color = torch::empty({3, H, W}, float_opts);
	// Kernel zeros all entries in preprocess — no zeroing needed
	torch::Tensor radii = torch::empty({P}, int_opts);

	// Persistent buffers: resizeFunctional will grow if needed, no-op if already big enough
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
	if (P != 0)
	{
		int M = 0;
		if (sh.size(0) != 0)
		{
			M = sh.size(1);
		}

		const float* colors_ptr = (colors.numel() > 0) ? colors.contiguous().data<float>() : nullptr;
		const float* shapes_ptr = (shapes.numel() > 0) ? shapes.contiguous().data<float>() : nullptr;

		const __half* atlas_texture_ptr = (atlas_texture.numel() > 0)
			? (const __half*)atlas_texture.contiguous().data_ptr<at::Half>() : nullptr;
		const float* atlas_rects_ptr = (atlas_rects.numel() > 0)
			? atlas_rects.contiguous().data<float>() : nullptr;

		// Infer residual dimension from tensor size: total / (P * 8 * 8)
		int residual_dim = 3;  // default: DC residual
		if (residual_textures.numel() > 0 && P > 0) {
			residual_dim = (int)(residual_textures.numel() / (P * 64));
		}

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			colors_ptr,
			opacity.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			out_color.contiguous().data<float>(),
			nullptr,  // out_others not needed (RENDER_AXUTILITY=0)
			radii.contiguous().data<int>(),
			debug,
			beta,
			shapes_ptr,
			kernel_type,
			(residual_textures.numel() > 0) ? (const __half*)residual_textures.contiguous().data_ptr<at::Half>() : nullptr,
			residual_dim,
			atlas_texture_ptr,
			atlas_rects_ptr,
			atlas_width,
			aabb_mode);
	}

	return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix)
{
	const int P = means3D.size(0);

	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::markVisible(P,
			means3D.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			present.contiguous().data<bool>());
	}

	return present;
}
