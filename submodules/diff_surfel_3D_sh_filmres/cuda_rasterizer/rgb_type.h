/*
 * rgb_type.h - FP16 vs FP32 per-Gaussian SH baseline color type alias.
 *
 * Why a separate header (not config.h)?
 * --------------------------------------
 * config.h defines NUM_CHANNELS, which collides with cub's template
 * parameter of the same name in cub/agent/agent_histogram.cuh. cub is
 * included from rasterizer_impl.cu and must NOT see NUM_CHANNELS as a
 * macro at that point. We need the rgb_t typedef in headers consumed by
 * rasterizer_impl.cu (GeometryState), so this minimal header carries only
 * the FP16 toggle + alias and is safe to include alongside cub.
 *
 * Set FP16_RGB to 1 to store per-Gaussian SH baseline colors as __half
 * (6B/Gauss) — halves bandwidth of the inner-loop color fetch. Set to 0
 * for the FP32 baseline. See rasterizer_impl.h::GeometryState::rgb.
 */
#ifndef CUDA_RASTERIZER_RGB_TYPE_H_INCLUDED
#define CUDA_RASTERIZER_RGB_TYPE_H_INCLUDED

#include <cuda_fp16.h>

#define FP16_RGB 1

#if FP16_RGB
	typedef __half rgb_t;
	#define RGB_TO_FLOAT(x) __half2float(x)
	#define FLOAT_TO_RGB(x) __float2half(x)
#else
	typedef float rgb_t;
	#define RGB_TO_FLOAT(x) (x)
	#define FLOAT_TO_RGB(x) (x)
#endif

#endif
