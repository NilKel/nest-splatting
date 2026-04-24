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
#include <map>
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

// ---------------------------------------------------------------------------
// Hardware-texture atlas cache.
// The atlas tensor is [H, W, 3] FP16 but CUDA textures need 1/2/4 channels for
// hardware bilinear — so we one-time pad to RGBA (alpha=0), upload to a 2D
// cudaArray, create a `cudaTextureObject_t` with linear filtering, and cache it
// by the atlas data pointer so repeat renders reuse the same binding.
// ---------------------------------------------------------------------------
struct AtlasTex {
	cudaArray_t array = nullptr;
	cudaTextureObject_t tex = 0;
	int W = 0, H = 0;
	// Dequantization: rgb = tex.xyz * scale + offset.
	// uint8 path → computed from atlas ±6σ. half4 path → scale=1, offset=0 (pass-through).
	float scale = 1.0f;
	float offset = 0.0f;
};
static std::map<const void*, AtlasTex> g_atlas_cache;

// Runtime toggle: uint8 quantized (default, memory-efficient) vs half4 (lossless).
// Flipped via SetAtlasUseUint8CUDA; flush cache to force rebuild at next render.
static bool g_atlas_use_uint8 = true;

// Runtime toggle: use hardware texture object (default) vs pre-texture software
// path (raw FP16 global-memory reads + manual bilinear inside the render kernel).
// Useful for A/B benchmarking the GPU-texturing speedup.
static bool g_use_atlas_tex_obj = true;

// Kernel: copy [H*W*3] FP16 RGB → [H, W] half4 with alpha=0.
__global__ void rgb_to_rgba_half4(const __half* __restrict__ rgb,
                                  __half* __restrict__ rgba,
                                  int HxW)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= HxW) return;
	int src = idx * 3;
	int dst = idx * 4;
	rgba[dst + 0] = rgb[src + 0];
	rgba[dst + 1] = rgb[src + 1];
	rgba[dst + 2] = rgb[src + 2];
	rgba[dst + 3] = __float2half(0.0f);
}

// Kernel: quantize [H*W*3] FP16 RGB → [H, W] uchar4 RGBA.
// `offset` = atlas min, `scale` = atlas max − min. Normalized to [0, 255].
__global__ void rgb_to_uchar4(const __half* __restrict__ rgb,
                              uchar4* __restrict__ rgba,
                              int HxW, float offset, float inv_scale)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= HxW) return;
	int src = idx * 3;
	float r = __half2float(rgb[src + 0]);
	float g = __half2float(rgb[src + 1]);
	float b = __half2float(rgb[src + 2]);
	// Map to [0, 255] with clamping.
	float qr = (r - offset) * inv_scale * 255.0f;
	float qg = (g - offset) * inv_scale * 255.0f;
	float qb = (b - offset) * inv_scale * 255.0f;
	qr = fmaxf(0.0f, fminf(255.0f, qr));
	qg = fmaxf(0.0f, fminf(255.0f, qg));
	qb = fmaxf(0.0f, fminf(255.0f, qb));
	uchar4 v;
	v.x = (unsigned char)(qr + 0.5f);
	v.y = (unsigned char)(qg + 0.5f);
	v.z = (unsigned char)(qb + 0.5f);
	v.w = 0;
	rgba[idx] = v;
}

// Build (or fetch cached) a uint8-quantized hardware-bilinear texture for the atlas.
// Caller supplies the dequantization params (offset, scale) — typically
// offset = mean − k·std, scale = 2k·std (k=6 covers ~99.99% of residuals).
// At sample time the kernel does `rgb = tex2D<float4>() * scale + offset`.
static const AtlasTex& get_or_build_atlas_tex(const __half* rgb_ptr, int W, int H,
                                              float offset, float scale) {
	const void* key = (const void*)rgb_ptr;
	auto it = g_atlas_cache.find(key);
	if (it != g_atlas_cache.end()) {
		if (it->second.W == W && it->second.H == H) return it->second;
		// Dim mismatch — destroy stale entry and rebuild.
		cudaDestroyTextureObject(it->second.tex);
		cudaFreeArray(it->second.array);
		g_atlas_cache.erase(it);
	}
	AtlasTex e;
	e.W = W; e.H = H;
	size_t hxw = (size_t)W * (size_t)H;
	const int block = 256;
	const int grid = (int)((hxw + block - 1) / block);

	if (g_atlas_use_uint8) {
		// --- uint8-quantized path (smaller memory, ~±6σ range, hw bilinear) ---
		e.offset = offset;
		e.scale = scale;
		cudaChannelFormatDesc desc = cudaCreateChannelDesc(8, 8, 8, 8,
		                                                   cudaChannelFormatKindUnsigned);
		cudaMallocArray(&e.array, &desc, W, H);
		uchar4* scratch = nullptr;
		cudaMalloc(&scratch, hxw * sizeof(uchar4));
		float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;
		rgb_to_uchar4<<<grid, block>>>(rgb_ptr, scratch, (int)hxw, offset, inv_scale);
		cudaMemcpy2DToArray(e.array, 0, 0, scratch, W * sizeof(uchar4),
		                    W * sizeof(uchar4), H, cudaMemcpyDeviceToDevice);
		cudaFree(scratch);
		cudaResourceDesc res{};
		res.resType = cudaResourceTypeArray;
		res.res.array.array = e.array;
		cudaTextureDesc td{};
		td.addressMode[0] = cudaAddressModeClamp;
		td.addressMode[1] = cudaAddressModeClamp;
		td.filterMode = cudaFilterModeLinear;
		td.readMode = cudaReadModeNormalizedFloat;  // uint8 → [0, 1] float
		td.normalizedCoords = 0;
		cudaCreateTextureObject(&e.tex, &res, &td, nullptr);
	} else {
		// --- half4 path (lossless wrt training FP16 storage) ---
		// Identity dequantization so the render kernel's
		// `rgb = tex * scale + offset` reduces to `tex`.
		e.offset = 0.0f;
		e.scale = 1.0f;
		cudaChannelFormatDesc desc = cudaCreateChannelDesc(16, 16, 16, 16,
		                                                   cudaChannelFormatKindFloat);
		cudaMallocArray(&e.array, &desc, W, H);
		__half* scratch = nullptr;
		cudaMalloc(&scratch, hxw * 4 * sizeof(__half));
		rgb_to_rgba_half4<<<grid, block>>>(rgb_ptr, scratch, (int)hxw);
		cudaMemcpy2DToArray(e.array, 0, 0, scratch, W * 4 * sizeof(__half),
		                    W * 4 * sizeof(__half), H, cudaMemcpyDeviceToDevice);
		cudaFree(scratch);
		cudaResourceDesc res{};
		res.resType = cudaResourceTypeArray;
		res.res.array.array = e.array;
		cudaTextureDesc td{};
		td.addressMode[0] = cudaAddressModeClamp;
		td.addressMode[1] = cudaAddressModeClamp;
		td.filterMode = cudaFilterModeLinear;
		td.readMode = cudaReadModeElementType;  // half4 → float4 via hw unpack
		td.normalizedCoords = 0;
		cudaCreateTextureObject(&e.tex, &res, &td, nullptr);
	}
	g_atlas_cache[key] = e;
	return g_atlas_cache[key];
}

// Flip atlas encoding (true = uint8, false = half4). Also flushes the cache so
// the next render rebuilds with the new format.
void ClearAtlasCacheCUDA();  // forward decl, defined below

void SetAtlasUseUint8CUDA(bool val) {
	if (val == g_atlas_use_uint8) return;
	g_atlas_use_uint8 = val;
	ClearAtlasCacheCUDA();
}

void SetUseAtlasTexObjectCUDA(bool val) {
	g_use_atlas_tex_obj = val;
	// No cache flush needed — disabling tex object just bypasses the cache.
}

// Python-facing helpers.
void ClearAtlasCacheCUDA() {
	for (auto& kv : g_atlas_cache) {
		cudaDestroyTextureObject(kv.second.tex);
		cudaFreeArray(kv.second.array);
	}
	g_atlas_cache.clear();
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
	// Optional Spherical-Beta params [N, K, 6], K=sb_number (empty if SB disabled)
	const torch::Tensor& sb_params,
	const int sb_number,
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

		// Build (or reuse cached) hardware-texture object for the atlas.
		// atlas_texture is shape [H*W*3] FP16 flat; atlas_width gives the 2D
		// reshape. We compute a quantization range once per atlas (cached via
		// the atlas data pointer). Range = mean ± 6·std covers ~99.99% of a
		// normal-distributed residual, well inside uint8 precision for the
		// central mass while clipping outliers.
		cudaTextureObject_t atlas_tex_obj = 0;
		float atlas_offset = 0.0f, atlas_scale = 1.0f;
		// CUDA 2D cudaArray max dimension (per CUDA guide; 65,536 for sm_60+).
		// If either axis exceeds this we silently skip tex-object creation and
		// fall through to the software bilinear path, which has no such limit.
		constexpr int CUDA_ARRAY_2D_MAX = 65536;
		bool atlas_tex_ok = false;
		int atlas_height = 0;
		if (atlas_texture_ptr != nullptr && atlas_width > 0 && g_use_atlas_tex_obj) {
			int64_t total_pixels = atlas_texture.numel() / 3;
			atlas_height = (int)(total_pixels / atlas_width);
			atlas_tex_ok = (atlas_width <= CUDA_ARRAY_2D_MAX && atlas_height <= CUDA_ARRAY_2D_MAX);
			if (!atlas_tex_ok) {
				printf("[BAKE_RENDER] Atlas %dx%d exceeds cudaArray 2D max (%d); "
				       "falling back to software bilinear path.\n",
				       atlas_width, atlas_height, CUDA_ARRAY_2D_MAX);
			}
		}
		if (atlas_tex_ok) {
			// Check cache — avoid the mean/std recompute when reusing.
			const void* key = (const void*)atlas_texture_ptr;
			auto cit = g_atlas_cache.find(key);
			if (cit != g_atlas_cache.end() && cit->second.W == atlas_width && cit->second.H == atlas_height) {
				atlas_tex_obj = cit->second.tex;
				atlas_offset = cit->second.offset;
				atlas_scale = cit->second.scale;
			} else {
				auto f32 = atlas_texture.to(torch::kFloat32);
				float mean = f32.mean().item<float>();
				float std  = f32.std().item<float>();
				float k = 6.0f;
				atlas_offset = mean - k * std;
				atlas_scale = std::max(2.0f * k * std, 1e-6f);
				const AtlasTex& e = get_or_build_atlas_tex(atlas_texture_ptr,
				                                            atlas_width, atlas_height,
				                                            atlas_offset, atlas_scale);
				atlas_tex_obj = e.tex;
			}
		}

		// Infer residual dimension from tensor size: total / (P * 8 * 8)
		int residual_dim = 3;  // default: DC residual
		if (residual_textures.numel() > 0 && P > 0) {
			residual_dim = (int)(residual_textures.numel() / (P * 64));
		}

		const float* sb_params_ptr = (sb_params.numel() > 0 && sb_number > 0)
			? sb_params.contiguous().data<float>() : nullptr;

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
			aabb_mode,
			sb_params_ptr,
			sb_number,
			atlas_tex_obj,
			atlas_offset,
			atlas_scale);
	}

	return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

// Device-global setter bindings (mirror diff_surfel_3D_sh_res).
void SetActivationBiasBakeCUDA(float sh_bias, float res_bias) {
	FORWARD::setActivationBias(sh_bias, res_bias);
}

void SetCompactMultBakeCUDA(float val) {
	FORWARD::setCompactMult(val);
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
