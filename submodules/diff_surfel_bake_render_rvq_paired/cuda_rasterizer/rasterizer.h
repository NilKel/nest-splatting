/*
 * Baked rendering submodule — forward-only rasterizer.
 */
#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>  // cudaTextureObject_t

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			int* radii,
			bool debug,
			const float beta,
			const float* shapes,
			const int kernel_type,
			const __half* atlas_texture,
			const float* atlas_rects,
			const int atlas_width,
			const int aabb_mode = 3,
			const float* sb_params = nullptr,
			const int sb_number = 0,
			cudaTextureObject_t atlas_tex_obj = 0,
			float atlas_offset = 0.0f,
			float atlas_scale = 1.0f,
			const int sort_mode = 0,  // 0 = legacy 64-bit single sort, 1 = FastGS two-stage
			// --feature SV: per-Gaussian voronoi state. Pre-activated:
			// sites = unit vectors, tau = exp(_sv_tau), colors = raw RGB.
			// Empty / voronoi_K == 0 ⇒ legacy SH path.
			const float* voronoi_sites = nullptr,
			const float* voronoi_tau = nullptr,
			const float* voronoi_colors = nullptr,
			const int voronoi_K = 0,
			// `--method mixed_3d`: per-Gauss textured flag + activated 3rd-axis
			// scale. nullptr scaling_z → pure 2DGS bake (unchanged).
			const bool* is_textured = nullptr,
			const float* scaling_z = nullptr);
	};
};

#endif
