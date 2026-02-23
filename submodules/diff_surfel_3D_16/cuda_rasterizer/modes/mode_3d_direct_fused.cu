/*
 * Mode 5: 3D_direct_fused - In-kernel MLP evaluation with collaborative tile GEMM
 *
 * This mode runs hash lookup + MLP inside the rasterizer kernel, following the
 * per-intersection paradigm: sum(w_i * MLP(f_i))
 *
 * Key optimizations:
 * - "Zeroes Matrix" + Collaborative Tile GEMM (non-participating pixels write zeros)
 * - FP16 weights (global memory) + FP16 shared memory for GEMM buffers
 *   → Single batch of 256 threads (vs 2×128 with FP32), saves 6 syncs/Gaussian
 * - Staged interleaved backward: layer-by-layer backward interleaved with GEMM
 *   to reduce peak register pressure from ~281 to ~139 floats
 * - No pre-activation storage: ReLU derivative uses h_post > 0
 * - Single-pass backprop: merged compute_dL_dz + dL_dinput into one interleaved flow
 * - Bias-free MLP: L1 uses input padding (input[40]=1.0, W1[32×41]),
 *   L2/L3 have no bias. Guarantees W@0=0 for zeroes matrix correctness.
 */

#ifndef MODE_3D_DIRECT_FUSED_CU_INCLUDED
#define MODE_3D_DIRECT_FUSED_CU_INCLUDED

// When included from backward.cu, all declarations already exist
#ifndef BACKWARD_CU_INCLUDES_MODE
#include "modes.h"
#include "../auxiliary.h"
#include "../config.h"
#include "../hashgrid.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Forward declaration of encode_view_direction from forward.cu
__device__ void encode_view_direction(const float3& dir, float* enc);
__device__ void encode_view_direction_bw(const float3& dir, float* enc);
#endif // BACKWARD_CU_INCLUDES_MODE

// ============================================================================
// MLP weight pointers struct (bias-free)
// L1: W1[32×41] (last column acts as implicit bias via input[40]=1.0)
// L2: W2[32×32] (no bias)
// L3: W3[3×32] (no bias)
// ============================================================================
struct MlpWeights {
    const __half* W1;      // [32 * 41] FP16
    const __half* W2;      // [32 * 32] FP16
    const __half* W3_rgb;  // [3 * 32] FP16
};

// ============================================================================
// MLP Forward Pass (bias-free, in registers, stores only post-ReLU activations)
// Input must have input[40] = 1.0f set by caller for implicit L1 bias
// ============================================================================
__device__ __forceinline__ void mlp_forward_inline(
    const float* input,    // [41] (input[40] = 1.0 for implicit bias)
    float* output,         // [3]
    float* h1,             // [32] hidden1 post-ReLU (for backward)
    float* h2,             // [32] hidden2 post-ReLU (for backward)
    bool apply_sigmoid,
    const MlpWeights& mlp  // MLP weight pointers
) {
    // Layer 1: input[41] -> h1[32] with ReLU (FP16 weights, FP32 accumulators)
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        float acc = 0;
        #pragma unroll
        for (int i = 0; i < 41; i++) {
            acc += input[i] * __half2float(mlp.W1[h * 41 + i]);
        }
        h1[h] = fmaxf(0.0f, acc);  // ReLU
    }

    // Layer 2: h1[32] -> h2[32] with ReLU (FP16 weights, FP32 accumulators)
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        float acc = 0;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            acc += h1[i] * __half2float(mlp.W2[h * 32 + i]);
        }
        h2[h] = fmaxf(0.0f, acc);  // ReLU
    }

    // Layer 3: h2[32] -> output[3] with optional sigmoid (FP16 weights, FP32 accumulators)
    #pragma unroll
    for (int o = 0; o < 3; o++) {
        float acc = 0;
        #pragma unroll
        for (int h = 0; h < 32; h++) {
            acc += h2[h] * __half2float(mlp.W3_rgb[o * 32 + h]);
        }
        if (apply_sigmoid) {
            output[o] = 1.0f / (1.0f + expf(-acc));
        } else {
            output[o] = acc;
        }
    }
}

// ============================================================================
// Collaborative Tile GEMM for MLP weight gradients (FP16 shared memory)
// Uses "Zeroes Matrix" strategy: non-participating pixels contribute 0
// Bias-free: W@0=0 guarantees zero contributions from inactive threads
//
// FP16 smem allows single batch of 256 threads (vs 2×128 with FP32):
//   256×73×2 = 37,376 bytes (same budget as 128×73×4)
// Eliminates 6 __syncthreads() per Gaussian (2 per layer × 3 layers)
// ============================================================================

// ============================================================================
// Per-layer collaborative GEMM functions (FP16 smem, single batch of 256)
// Split from monolithic collaborative_mlp_backward_all for interleaved backward
// ============================================================================

// Layer 3: dL_dW3[3,32] = dL_dz3^T @ h2
// Smem: 256×3 + 256×32 = 17,920 bytes (FP16)
__device__ void collaborative_gemm_layer3(
    const float* my_dL_dz3,    // [3] per-pixel (FP32 registers)
    const float* my_h2,        // [32] per-pixel (FP32 registers)
    float* tile_dL_dW3,        // [96] shared mem accumulator (FP32)
    __half* smem_buffer        // dynamic shared memory (FP16)
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    __half* smem_dL_dz3 = smem_buffer;               // [256][3]
    __half* smem_h2 = smem_buffer + 256 * 3;          // [256][32]

    // Store phase: all 256 threads write (float → half)
    smem_dL_dz3[tid * 3 + 0] = __float2half(my_dL_dz3[0]);
    smem_dL_dz3[tid * 3 + 1] = __float2half(my_dL_dz3[1]);
    smem_dL_dz3[tid * 3 + 2] = __float2half(my_dL_dz3[2]);
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        smem_h2[tid * 32 + i] = __float2half(my_h2[i]);
    }
    __syncthreads();

    // Compute phase: 96 outputs (3×32), all fit in one pass
    if (tid < 96) {
        int o = tid / 32;
        int h = tid % 32;
        float sum = 0;
        for (int p = 0; p < 256; p++) {
            sum += __half2float(smem_dL_dz3[p * 3 + o]) * __half2float(smem_h2[p * 32 + h]);
        }
        tile_dL_dW3[tid] += sum;
    }
    __syncthreads();
}

// Layer 2: dL_dW2[32,32] = dL_dz2^T @ h1
// Smem: 256×32 + 256×32 = 32,768 bytes (FP16)
__device__ void collaborative_gemm_layer2(
    const float* my_dL_dz2,    // [32] per-pixel (FP32 registers)
    const float* my_h1,        // [32] per-pixel (FP32 registers)
    float* tile_dL_dW2,        // [1024] shared mem accumulator (FP32)
    __half* smem_buffer        // dynamic shared memory (FP16)
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    __half* smem_dL_dz2 = smem_buffer;                // [256][32]
    __half* smem_h1 = smem_buffer + 256 * 32;          // [256][32]

    // Store phase: all 256 threads write (float → half)
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        smem_dL_dz2[tid * 32 + i] = __float2half(my_dL_dz2[i]);
        smem_h1[tid * 32 + i] = __float2half(my_h1[i]);
    }
    __syncthreads();

    // Compute phase: 1024 outputs (32×32), need 4 passes
    for (int pass = 0; pass < 4; pass++) {
        int out_idx = pass * 256 + tid;
        if (out_idx < 1024) {
            int h = out_idx / 32;
            int i = out_idx % 32;
            float sum = 0;
            for (int p = 0; p < 256; p++) {
                sum += __half2float(smem_dL_dz2[p * 32 + h]) * __half2float(smem_h1[p * 32 + i]);
            }
            tile_dL_dW2[out_idx] += sum;
        }
    }
    __syncthreads();
}

// Layer 1: dL_dW1[32,41] = dL_dz1^T @ input
// Virtual indexing: input[40] is always 1.0 (implicit bias column)
// Smem: 256×32 + 256×41 = 37,376 bytes (FP16) — largest layer
__device__ void collaborative_gemm_layer1(
    const float* my_dL_dz1,    // [32] per-pixel (FP32 registers)
    const float* my_input,     // [41] per-pixel (input[40] = 1.0, FP32 registers)
    float* tile_dL_dW1,        // [1312] shared mem accumulator (32×41, FP32)
    __half* smem_buffer        // dynamic shared memory (FP16)
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    __half* smem_dL_dz1 = smem_buffer;                // [256][32]
    __half* smem_input = smem_buffer + 256 * 32;       // [256][41]

    // Store phase: all 256 threads write (float → half)
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        smem_dL_dz1[tid * 32 + i] = __float2half(my_dL_dz1[i]);
    }
    #pragma unroll
    for (int i = 0; i < 41; i++) {
        smem_input[tid * 41 + i] = __float2half(my_input[i]);
    }
    __syncthreads();

    // Compute phase: 1312 outputs (32×41), need ceil(1312/256) = 6 passes
    for (int pass = 0; pass < 6; pass++) {
        int out_idx = pass * 256 + tid;
        if (out_idx < 1312) {
            int h = out_idx / 41;
            int i = out_idx % 41;
            float sum = 0;
            for (int p = 0; p < 256; p++) {
                sum += __half2float(smem_dL_dz1[p * 32 + h]) * __half2float(smem_input[p * 41 + i]);
            }
            tile_dL_dW1[out_idx] += sum;
        }
    }
    __syncthreads();
}

// All-in-one wrapper (calls per-layer functions sequentially)
__device__ void collaborative_mlp_backward_all(
    const float* my_input,    // [41]
    const float* my_h1,       // [32]
    const float* my_h2,       // [32]
    const float* my_dL_dz1,   // [32]
    const float* my_dL_dz2,   // [32]
    const float* my_dL_dz3,   // [3]
    float* tile_dL_dW1,       // [1312]
    float* tile_dL_dW2,       // [1024]
    float* tile_dL_dW3,       // [96]
    __half* smem_buffer
) {
    collaborative_gemm_layer3(my_dL_dz3, my_h2, tile_dL_dW3, smem_buffer);
    collaborative_gemm_layer2(my_dL_dz2, my_h1, tile_dL_dW2, smem_buffer);
    collaborative_gemm_layer1(my_dL_dz1, my_input, tile_dL_dW1, smem_buffer);
}

// Required dynamic shared memory size for collaborative backward (FP16 single batch)
// Layer 1 requires the most: 256*41*2 + 256*32*2 = 20,992 + 16,384 = 37,376 bytes
constexpr int COLLABORATIVE_SMEM_SIZE = 256 * 41 * sizeof(__half) + 256 * 32 * sizeof(__half);

namespace MODES {

// ============================================================================
// Flush tile-local MLP gradients to global memory (bias-free)
// Called once per tile after processing all Gaussians
// ============================================================================
__device__ void flush_tile_mlp_grads(
    const float* tile_dL_dW1,
    const float* tile_dL_dW2,
    const float* tile_dL_dW3,
    float* global_dL_dW1,
    float* global_dL_dW2,
    float* global_dL_dW3,
    const int thread_id
) {
    // W1: 32×41 = 1312 floats
    for (int idx = thread_id; idx < 1312; idx += 256) {
        if (tile_dL_dW1[idx] != 0.0f)
            atomicAdd(&global_dL_dW1[idx], tile_dL_dW1[idx]);
    }
    // W2: 32×32 = 1024 floats
    for (int idx = thread_id; idx < 1024; idx += 256) {
        if (tile_dL_dW2[idx] != 0.0f)
            atomicAdd(&global_dL_dW2[idx], tile_dL_dW2[idx]);
    }
    // W3: 3×32 = 96 floats
    for (int idx = thread_id; idx < 96; idx += 256) {
        if (tile_dL_dW3[idx] != 0.0f)
            atomicAdd(&global_dL_dW3[idx], tile_dL_dW3[idx]);
    }
}

} // namespace MODES

#endif // MODE_3D_DIRECT_FUSED_CU_INCLUDED
