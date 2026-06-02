/*
 * Baked rendering submodule — PyTorch/CUDA binding (forward-only).
 */

#include <math.h>
#include <vector>
#include <cfloat>
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

// BC7 atlas (block-compressed). When set via SetAtlasBC7CUDA, the renderer
// uses the BC7 cudaArray + textureObject directly (hardware decode, full bilinear).
// Single global because we render one atlas at a time.
static cudaArray_t        g_bc7_array  = nullptr;
static cudaTextureObject_t g_bc7_tex   = 0;
static int   g_bc7_W = 0, g_bc7_H = 0;       // padded to multiples of 4
static float g_bc7_offset = 0.0f;
static float g_bc7_scale  = 1.0f;

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

// ---- RVQ atlas install (sets device-global pointers in forward.cu) --------
// Caller is responsible for keeping codebooks/indices/surfel_offsets alive in
// GPU memory until ClearAtlasRVQCUDA() is invoked (or the next SetAtlasRVQCUDA
// with new tensors). The render kernel reads them by raw device pointer.
static torch::Tensor g_rvq_codebooks_hold;
static torch::Tensor g_rvq_indices_hold;
static torch::Tensor g_rvq_offsets_hold;

void ClearAtlasRVQCUDA();   // forward-decl (defined below SetAtlasRVQCUDA)


void SetAtlasRVQCUDA(torch::Tensor codebooks_fp16,
                     torch::Tensor indices_u8,
                     torch::Tensor surfel_offsets_i64,
                     int B,
                     float atlas_scale,
                     float atlas_offset) {
	if (codebooks_fp16.numel() == 0) {
		ClearAtlasRVQCUDA();
		return;
	}
	TORCH_CHECK(codebooks_fp16.dtype() == torch::kFloat16, "codebooks must be FP16");
	TORCH_CHECK(indices_u8.dtype() == torch::kUInt8, "indices must be uint8 (K ≤ 256)");
	TORCH_CHECK(surfel_offsets_i64.dtype() == torch::kInt64, "surfel_offsets must be int64");
	TORCH_CHECK(codebooks_fp16.is_cuda() && indices_u8.is_cuda() && surfel_offsets_i64.is_cuda(),
	            "all RVQ tensors must be on CUDA");
	TORCH_CHECK(codebooks_fp16.dim() == 3,
	            "codebooks must be [L, K, B*B*3], got ", codebooks_fp16.sizes());
	int L = (int)codebooks_fp16.size(0);
	int K = (int)codebooks_fp16.size(1);
	int D = (int)codebooks_fp16.size(2);
	TORCH_CHECK(D == B * B * 3, "codebook dim ", D, " != block_size^2 * 3 = ", B*B*3);
	TORCH_CHECK(indices_u8.dim() == 2 && indices_u8.size(0) == L,
	            "indices must be [L, N_used] with matching L");
	unsigned long long N_used = (unsigned long long)indices_u8.size(1);

	// Hold references so the tensors stay alive while CUDA reads them.
	g_rvq_codebooks_hold = codebooks_fp16.contiguous();
	g_rvq_indices_hold   = indices_u8.contiguous();
	g_rvq_offsets_hold   = surfel_offsets_i64.contiguous();

	FORWARD::setAtlasRVQ(
		reinterpret_cast<const __half*>(g_rvq_codebooks_hold.data_ptr<at::Half>()),
		g_rvq_indices_hold.data_ptr<uint8_t>(),
		g_rvq_offsets_hold.data_ptr<int64_t>(),
		L, K, B, N_used);
	printf("[BAKE_RENDER] RVQ atlas installed: L=%d K=%d B=%d N_used=%llu "
	       "(codebooks=%.1f KB, indices=%.1f MB)\n",
	       L, K, B, N_used,
	       (float)g_rvq_codebooks_hold.numel() * 2 / 1024.0f,
	       (float)g_rvq_indices_hold.numel() / (1024.0f * 1024.0f));

	// Build per-stage min/max + 2D uint8-RGBA codebook texture.
	// Step 1: compute per-stage min and max on the host side. Move codebook
	// to CPU FP32 once for the reduction, then back to GPU for re-quantization.
	auto cb_cpu_f32 = g_rvq_codebooks_hold.to(torch::kCPU, torch::kFloat32).contiguous();
	// Shape: [L, K, D] flattened — reduce over K and D per stage.
	std::vector<float> per_stage_scale(L), per_stage_offset(L);
	auto* cb_data = cb_cpu_f32.data_ptr<float>();
	int64_t stage_elems = (int64_t)K * D;
	for (int l = 0; l < L; ++l) {
		float mn = +FLT_MAX, mx = -FLT_MAX;
		const float* p = cb_data + (int64_t)l * stage_elems;
		for (int64_t i = 0; i < stage_elems; ++i) {
			float v = p[i];
			if (v < mn) mn = v;
			if (v > mx) mx = v;
		}
		per_stage_scale[l]  = (mx - mn > 1e-8f) ? (mx - mn) : 1.0f;
		per_stage_offset[l] = mn;
		FORWARD::setAtlasRVQDequant(l, per_stage_scale[l], per_stage_offset[l]);
	}
	(void)atlas_scale; (void)atlas_offset;     // not used; we use per-stage params

	// Step 2: quantize codebook → uint8 RGBA, lay out into a 2D image
	// (K*B cols × L*B rows). Each codeword's 4×4 region is contiguous.
	int W2D = K * B;
	int H2D = L * B;
	std::vector<uint8_t> u8_image((size_t)W2D * H2D * 4, 0);  // RGBA, A=0
	for (int l = 0; l < L; ++l) {
		float s = per_stage_scale[l]; float o = per_stage_offset[l];
		const float* p = cb_data + (int64_t)l * stage_elems;
		for (int k = 0; k < K; ++k) {
			const float* cw = p + (int64_t)k * D;       // 16 px × 3 chan = 48 floats
			for (int iv = 0; iv < B; ++iv) {
				for (int iu = 0; iu < B; ++iu) {
					int px = iv * B + iu;
					int x = k * B + iu;
					int y = l * B + iv;
					int outo = (y * W2D + x) * 4;
					auto q = [&](float f) {
						float v = (f - o) / s * 255.0f;
						return (uint8_t)fmaxf(0.f, fminf(255.f, v + 0.5f));
					};
					u8_image[outo + 0] = q(cw[px*3 + 0]);
					u8_image[outo + 1] = q(cw[px*3 + 1]);
					u8_image[outo + 2] = q(cw[px*3 + 2]);
					u8_image[outo + 3] = 0;
				}
			}
		}
	}

	// Step 3: upload to a 2D cudaArray + create texture object.
	cudaChannelFormatDesc desc = cudaCreateChannelDesc(8, 8, 8, 8,
	                                                   cudaChannelFormatKindUnsigned);
	cudaArray_t cb_array = nullptr;
	cudaMallocArray(&cb_array, &desc, W2D, H2D);
	cudaMemcpy2DToArray(cb_array, 0, 0, u8_image.data(),
	                    W2D * 4, W2D * 4, H2D, cudaMemcpyHostToDevice);
	cudaResourceDesc res{};
	res.resType = cudaResourceTypeArray;
	res.res.array.array = cb_array;
	cudaTextureDesc td{};
	td.addressMode[0] = cudaAddressModeClamp;
	td.addressMode[1] = cudaAddressModeClamp;
	td.filterMode = cudaFilterModePoint;
	td.readMode = cudaReadModeNormalizedFloat;
	td.normalizedCoords = 0;
	static cudaTextureObject_t s_cb_tex = 0;
	static cudaArray_t         s_cb_array = nullptr;
	if (s_cb_tex)   cudaDestroyTextureObject(s_cb_tex);
	if (s_cb_array) cudaFreeArray(s_cb_array);
	cudaCreateTextureObject(&s_cb_tex, &res, &td, nullptr);
	s_cb_array = cb_array;
	FORWARD::setAtlasRVQTex2D(s_cb_tex);

	printf("[BAKE_RENDER] RVQ-tex codebook 2D installed: %dx%d uint8-RGBA "
	       "(%.1f KB), per-stage min/max dequant\n",
	       W2D, H2D, (float)W2D * H2D * 4 / 1024.0f);

	// ---- PAIRED RVQ: precompute pair0=cb[0]+cb[1] and pair1=cb[2]+cb[3] ----
	if (L != 4 || B != 4) {
		printf("[BAKE_RENDER] paired RVQ requires L=4 B=4 (got L=%d B=%d); "
		       "kernel will read uninitialised paired data. Stick with the "
		       "non-paired fork for non-standard configs.\n", L, B);
		return;
	}
	const int Kp = K * K;        // pair codebook entries (65536 at K=256)
	std::vector<std::vector<float>> pair_cb(2,
		std::vector<float>((size_t)Kp * D, 0.0f));
	for (int p = 0; p < 2; ++p) {
		const float* cb_hi = cb_data + (int64_t)(2*p    ) * stage_elems;
		const float* cb_lo = cb_data + (int64_t)(2*p + 1) * stage_elems;
		float* dst = pair_cb[p].data();
		for (int k_hi = 0; k_hi < K; ++k_hi) {
			for (int k_lo = 0; k_lo < K; ++k_lo) {
				const float* hi = cb_hi + (int64_t)k_hi * D;
				const float* lo = cb_lo + (int64_t)k_lo * D;
				float* o = dst + ((int64_t)k_hi * K + k_lo) * D;
				for (int d = 0; d < D; ++d) o[d] = hi[d] + lo[d];
			}
		}
	}
	std::vector<float> pair_scale(2), pair_offset(2);
	for (int p = 0; p < 2; ++p) {
		float mn = +FLT_MAX, mx = -FLT_MAX;
		for (float v : pair_cb[p]) { if (v < mn) mn = v; if (v > mx) mx = v; }
		pair_scale[p]  = (mx - mn > 1e-8f) ? (mx - mn) : 1.0f;
		pair_offset[p] = mn;
	}

	int Wp = K * B;
	int Hp = 2 * K * B;
	std::vector<uint8_t> pair_image((size_t)Wp * Hp * 4, 0);
	for (int p = 0; p < 2; ++p) {
		float s = pair_scale[p]; float o = pair_offset[p];
		const float* cb_p = pair_cb[p].data();
		for (int c_hi = 0; c_hi < K; ++c_hi) {
			for (int c_lo = 0; c_lo < K; ++c_lo) {
				const float* cw = cb_p + ((int64_t)c_hi * K + c_lo) * D;
				for (int iv = 0; iv < B; ++iv) {
					for (int iu = 0; iu < B; ++iu) {
						int px = iv * B + iu;
						int x = c_lo * B + iu;
						int y = p * (K * B) + c_hi * B + iv;
						int outo = (y * Wp + x) * 4;
						auto q = [&](float f) {
							float v = (f - o) / s * 255.0f;
							return (uint8_t)fmaxf(0.f, fminf(255.f, v + 0.5f));
						};
						pair_image[outo + 0] = q(cw[px*3 + 0]);
						pair_image[outo + 1] = q(cw[px*3 + 1]);
						pair_image[outo + 2] = q(cw[px*3 + 2]);
					}
				}
			}
		}
	}

	cudaChannelFormatDesc pdesc = cudaCreateChannelDesc(8, 8, 8, 8,
	                                                    cudaChannelFormatKindUnsigned);
	static cudaArray_t        s_pair_array = nullptr;
	static cudaTextureObject_t s_pair_tex  = 0;
	if (s_pair_tex)   { cudaDestroyTextureObject(s_pair_tex);  s_pair_tex = 0; }
	if (s_pair_array) { cudaFreeArray(s_pair_array);          s_pair_array = nullptr; }
	cudaMallocArray(&s_pair_array, &pdesc, Wp, Hp);
	cudaMemcpy2DToArray(s_pair_array, 0, 0, pair_image.data(),
	                    Wp * 4, Wp * 4, Hp, cudaMemcpyHostToDevice);
	cudaResourceDesc pres{};
	pres.resType = cudaResourceTypeArray;
	pres.res.array.array = s_pair_array;
	cudaTextureDesc ptd{};
	ptd.addressMode[0] = cudaAddressModeClamp;
	ptd.addressMode[1] = cudaAddressModeClamp;
	ptd.filterMode = cudaFilterModeLinear;       // hw bilinear within codeword for same-block fast path
	ptd.readMode = cudaReadModeNormalizedFloat;
	ptd.normalizedCoords = 0;
	cudaCreateTextureObject(&s_pair_tex, &pres, &ptd, nullptr);
	printf("[BAKE_RENDER] paired-cb 2D tex: %dx%d uint8-RGBA (%.1f MB)\n",
	       Wp, Hp, (float)Wp * Hp * 4 / 1024.0f / 1024.0f);

	// Pack per-block uint32: (pair1<<16)|pair0; pair = (idx_hi<<8)|idx_lo.
	auto idx_cpu = g_rvq_indices_hold.to(torch::kCPU).contiguous();
	const uint8_t* idx_data = idx_cpu.data_ptr<uint8_t>();
	std::vector<uint32_t> packed((size_t)N_used);
	for (uint64_t b = 0; b < N_used; ++b) {
		uint16_t pair0 = ((uint16_t)idx_data[0 * N_used + b] << 8) | idx_data[1 * N_used + b];
		uint16_t pair1 = ((uint16_t)idx_data[2 * N_used + b] << 8) | idx_data[3 * N_used + b];
		packed[b] = ((uint32_t)pair1 << 16) | (uint32_t)pair0;
	}
	static torch::Tensor s_pair_indices_hold;
	auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	s_pair_indices_hold = torch::empty({(long long)N_used}, opts);
	cudaMemcpy(s_pair_indices_hold.data_ptr<int32_t>(), packed.data(),
	           N_used * sizeof(uint32_t), cudaMemcpyHostToDevice);

	FORWARD::setAtlasRVQPaired(
		s_pair_tex,
		reinterpret_cast<const uint32_t*>(s_pair_indices_hold.data_ptr<int32_t>()),
		K,
		pair_scale[0], pair_offset[0],
		pair_scale[1], pair_offset[1]);
	printf("[BAKE_RENDER] paired indices: %.1f MB uint32 packed; "
	       "dequant scale={%.4f, %.4f} offset={%.4f, %.4f}\n",
	       (float)N_used * sizeof(uint32_t) / 1024.0f / 1024.0f,
	       pair_scale[0], pair_scale[1], pair_offset[0], pair_offset[1]);
}

void ClearAtlasRVQCUDA() {
	FORWARD::clearAtlasRVQ();
	g_rvq_codebooks_hold = torch::Tensor();
	g_rvq_indices_hold   = torch::Tensor();
	g_rvq_offsets_hold   = torch::Tensor();
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
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
	const torch::Tensor& atlas_texture,
	const torch::Tensor& atlas_rects,
	const int atlas_width,
	const int aabb_mode,
	// Optional Spherical-Beta params [N, K, 6], K=sb_number (empty if SB disabled)
	const torch::Tensor& sb_params,
	const int sb_number,
	// Optional Spherical-Voronoi params: pre-activated tensors. Empty (numel==0)
	// or voronoi_K==0 ⇒ SH path. When all three are set the rasterizer skips
	// computeColorFromSH entirely and uses computeColorFromVoronoi (fused into
	// preprocessCUDA, no extra kernel launch — same per-Gaussian dispatch as SH).
	//   voronoi_sites  : [N, K, 3] unit vectors  (caller does F.normalize)
	//   voronoi_tau    : [N, K]    post-exp scalars (caller does torch.exp(_sv_tau))
	//   voronoi_colors : [N, K, 3] raw RGB  (no activation; ReLU applied in CUDA)
	const torch::Tensor& voronoi_sites,
	const torch::Tensor& voronoi_tau,
	const torch::Tensor& voronoi_colors,
	const int voronoi_K,
	// Persistent scratch buffers (resized in place when growth is needed).
	torch::Tensor geomBuffer,
	torch::Tensor binningBuffer,
	torch::Tensor imgBuffer,
	// Persistent caller-owned outputs (Python pre-allocates and reuses).
	torch::Tensor out_color,
	torch::Tensor radii,
	const int sort_mode,
	// `--method mixed_3d`: per-Gauss textured flag [P] + activated 3rd-axis
	// scale [P]. Empty tensors → pure 2DGS bake (unchanged).
	const torch::Tensor& is_textured,
	const torch::Tensor& scaling_z)
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

	// Persistent scratch buffers: resizeFunctional grows when needed, no-op otherwise.
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

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

		// RVQ-only submodule: kernel's atlas-sample step is the RVQ decode
		// branch, which doesn't read atlas_texture / atlas_tex_obj. Pass
		// no-op values to Rasterizer::forward so the kernel signature stays
		// compatible with the bake-render baseline.
		cudaTextureObject_t atlas_tex_obj = 0;
		float atlas_offset = 0.0f, atlas_scale = 1.0f;

		// All input tensors are caller-pre-validated contiguous (we only run
		// in inference mode against fixed pre-activated buffers from Python).
		// Skip the redundant .contiguous() calls — they're no-ops on already-
		// contiguous tensors but still cost a Python/C++ shape check per frame.
		const float* sb_params_ptr = (sb_params.numel() > 0 && sb_number > 0)
			? sb_params.contiguous().data<float>() : nullptr;

		// Voronoi pointers — null-safe extraction. Caller passes empty tensors
		// or voronoi_K=0 to disable. We require all three populated together.
		const float* voronoi_sites_ptr = (voronoi_sites.numel() > 0 && voronoi_K > 0)
			? voronoi_sites.contiguous().data<float>() : nullptr;
		const float* voronoi_tau_ptr = (voronoi_tau.numel() > 0 && voronoi_K > 0)
			? voronoi_tau.contiguous().data<float>() : nullptr;
		const float* voronoi_colors_ptr = (voronoi_colors.numel() > 0 && voronoi_K > 0)
			? voronoi_colors.contiguous().data<float>() : nullptr;
		const int voronoi_K_eff =
			(voronoi_sites_ptr && voronoi_tau_ptr && voronoi_colors_ptr) ? voronoi_K : 0;

		// `--method mixed_3d`: empty → nullptr → pure 2DGS bake (unchanged).
		const bool* is_textured_ptr = (is_textured.numel() > 0)
			? is_textured.contiguous().data<bool>() : nullptr;
		const float* scaling_z_ptr = (scaling_z.numel() > 0)
			? scaling_z.contiguous().data<float>() : nullptr;

		// Camera matrices (viewmatrix / projmatrix) come from PyTorch with a
		// transpose applied — they're strided views, NOT contiguous. The
		// kernel assumes row-major dense layout, so .contiguous() is required.
		// Other inputs are already contiguous (gaussian tensors are snapshotted
		// via .contiguous() in prepare_gaussian_inputs) so the call is a no-op.
		CudaRasterizer::Rasterizer::forward(
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
			radii.contiguous().data<int>(),
			debug,
			beta,
			shapes_ptr,
			kernel_type,
			atlas_texture_ptr,
			atlas_rects_ptr,
			atlas_width,
			aabb_mode,
			sb_params_ptr,
			sb_number,
			atlas_tex_obj,
			atlas_offset,
			atlas_scale,
			sort_mode,
			voronoi_sites_ptr,
			voronoi_tau_ptr,
			voronoi_colors_ptr,
			voronoi_K_eff,
			is_textured_ptr,
			scaling_z_ptr);
	}

	return std::make_tuple(geomBuffer, binningBuffer, imgBuffer);
}

// Device-global setter bindings (mirror diff_surfel_3D_sh_res).
void SetActivationBiasBakeCUDA(float sh_bias, float res_bias) {
	FORWARD::setActivationBias(sh_bias, res_bias);
}

void SetCompactMultBakeCUDA(float val) {
	FORWARD::setCompactMult(val);
}

void SetResidualModeBakeCUDA(int mode) {
	FORWARD::setResidualMode(mode);
}

void SetUntexKernelBakeCUDA(int v) {
	FORWARD::setUntexKernel(v);
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
			means3D.data<float>(),
			viewmatrix.data<float>(),
			projmatrix.data<float>(),
			present.data<bool>());
	}

	return present;
}
