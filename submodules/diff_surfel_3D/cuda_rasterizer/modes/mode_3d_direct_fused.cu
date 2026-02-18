/*
 * Mode 5: 3D_direct_fused - In-kernel MLP evaluation with collaborative tile GEMM
 *
 * This mode runs hash lookup + MLP inside the rasterizer kernel, following the
 * per-intersection paradigm: sum(w_i * MLP(f_i))
 *
 * Key optimizations:
 * - "Zeroes Matrix" + Collaborative Tile GEMM (non-participating pixels write zeros)
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
    const float* W1;      // [32 * 41]
    const float* W2;      // [32 * 32]
    const float* W3_rgb;  // [3 * 32]
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
    // Layer 1: input[41] -> h1[32] with ReLU (last column of W1 acts as bias)
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        float acc = 0;
        #pragma unroll
        for (int i = 0; i < 41; i++) {
            acc += input[i] * mlp.W1[h * 41 + i];
        }
        h1[h] = fmaxf(0.0f, acc);  // ReLU
    }

    // Layer 2: h1[32] -> h2[32] with ReLU (no bias)
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        float acc = 0;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            acc += h1[i] * mlp.W2[h * 32 + i];
        }
        h2[h] = fmaxf(0.0f, acc);  // ReLU
    }

    // Layer 3: h2[32] -> output[3] with optional sigmoid (no bias)
    #pragma unroll
    for (int o = 0; o < 3; o++) {
        float acc = 0;
        #pragma unroll
        for (int h = 0; h < 32; h++) {
            acc += h2[h] * mlp.W3_rgb[o * 32 + h];
        }
        if (apply_sigmoid) {
            output[o] = 1.0f / (1.0f + expf(-acc));
        } else {
            output[o] = acc;
        }
    }
}

// ============================================================================
// Collaborative Tile GEMM for MLP weight gradients
// Uses "Zeroes Matrix" strategy: non-participating pixels contribute 0
// Bias-free: W@0=0 guarantees zero contributions from inactive threads
// ============================================================================

#define SUB_TILE_SIZE 128

// ============================================================================
// Per-layer collaborative GEMM functions (bias-free)
// Split from monolithic collaborative_mlp_backward_all for interleaved backward
// ============================================================================

// Layer 3: dL_dW3[3,32] = dL_dz3^T @ h2
// Sub-tile: 128×3 + 128×32 = 17KB
__device__ void collaborative_gemm_layer3(
    const float* my_dL_dz3,    // [3] per-pixel
    const float* my_h2,        // [32] per-pixel
    float* tile_dL_dW3,        // [96] shared mem accumulator
    float* smem_buffer         // dynamic shared memory
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float* smem_dL_dz3 = smem_buffer;                      // [128][3]
    float* smem_h2 = smem_buffer + SUB_TILE_SIZE * 3;      // [128][32]

    for (int batch = 0; batch < 2; batch++) {
        const int batch_start = batch * SUB_TILE_SIZE;
        const bool in_batch = (tid >= batch_start) && (tid < batch_start + SUB_TILE_SIZE);
        const int local_idx = tid - batch_start;

        if (in_batch) {
            smem_dL_dz3[local_idx * 3 + 0] = my_dL_dz3[0];
            smem_dL_dz3[local_idx * 3 + 1] = my_dL_dz3[1];
            smem_dL_dz3[local_idx * 3 + 2] = my_dL_dz3[2];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                smem_h2[local_idx * 32 + i] = my_h2[i];
            }
        }
        __syncthreads();

        if (tid < 96) {
            int o = tid / 32;
            int h = tid % 32;
            float sum = 0;
            for (int p = 0; p < SUB_TILE_SIZE; p++) {
                sum += smem_dL_dz3[p * 3 + o] * smem_h2[p * 32 + h];
            }
            tile_dL_dW3[tid] += sum;
        }
        __syncthreads();
    }
}

// Layer 2: dL_dW2[32,32] = dL_dz2^T @ h1
// Sub-tile: 128×32 + 128×32 = 32KB
__device__ void collaborative_gemm_layer2(
    const float* my_dL_dz2,    // [32] per-pixel
    const float* my_h1,        // [32] per-pixel
    float* tile_dL_dW2,        // [1024] shared mem accumulator
    float* smem_buffer         // dynamic shared memory
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float* smem_dL_dz2 = smem_buffer;                      // [128][32]
    float* smem_h1 = smem_buffer + SUB_TILE_SIZE * 32;     // [128][32]

    for (int batch = 0; batch < 2; batch++) {
        const int batch_start = batch * SUB_TILE_SIZE;
        const bool in_batch = (tid >= batch_start) && (tid < batch_start + SUB_TILE_SIZE);
        const int local_idx = tid - batch_start;

        if (in_batch) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                smem_dL_dz2[local_idx * 32 + i] = my_dL_dz2[i];
                smem_h1[local_idx * 32 + i] = my_h1[i];
            }
        }
        __syncthreads();

        for (int pass = 0; pass < 4; pass++) {
            int out_idx = pass * 256 + tid;
            if (out_idx < 1024) {
                int h = out_idx / 32;
                int i = out_idx % 32;
                float sum = 0;
                for (int p = 0; p < SUB_TILE_SIZE; p++) {
                    sum += smem_dL_dz2[p * 32 + h] * smem_h1[p * 32 + i];
                }
                tile_dL_dW2[out_idx] += sum;
            }
        }
        __syncthreads();
    }
}

// Layer 1: dL_dW1[32,41] = dL_dz1^T @ input
// Virtual indexing: input[40] is always 1.0 (implicit bias column)
// Sub-tile: 128×32 + 128×41 = ~37KB (largest)
__device__ void collaborative_gemm_layer1(
    const float* my_dL_dz1,    // [32] per-pixel
    const float* my_input,     // [41] per-pixel (input[40] = 1.0)
    float* tile_dL_dW1,        // [1312] shared mem accumulator (32×41)
    float* smem_buffer         // dynamic shared memory
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float* smem_dL_dz1 = smem_buffer;                      // [128][32]
    float* smem_input = smem_buffer + SUB_TILE_SIZE * 32;   // [128][41]

    for (int batch = 0; batch < 2; batch++) {
        const int batch_start = batch * SUB_TILE_SIZE;
        const bool in_batch = (tid >= batch_start) && (tid < batch_start + SUB_TILE_SIZE);
        const int local_idx = tid - batch_start;

        if (in_batch) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                smem_dL_dz1[local_idx * 32 + i] = my_dL_dz1[i];
            }
            #pragma unroll
            for (int i = 0; i < 41; i++) {
                smem_input[local_idx * 41 + i] = my_input[i];
            }
        }
        __syncthreads();

        // 32×41 = 1312 outputs, need ceil(1312/256) = 6 passes
        for (int pass = 0; pass < 6; pass++) {
            int out_idx = pass * 256 + tid;
            if (out_idx < 1312) {
                int h = out_idx / 41;
                int i = out_idx % 41;
                float sum = 0;
                for (int p = 0; p < SUB_TILE_SIZE; p++) {
                    sum += smem_dL_dz1[p * 32 + h] * smem_input[p * 41 + i];
                }
                tile_dL_dW1[out_idx] += sum;
            }
        }
        __syncthreads();
    }
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
    float* smem_buffer
) {
    collaborative_gemm_layer3(my_dL_dz3, my_h2, tile_dL_dW3, smem_buffer);
    collaborative_gemm_layer2(my_dL_dz2, my_h1, tile_dL_dW2, smem_buffer);
    collaborative_gemm_layer1(my_dL_dz1, my_input, tile_dL_dW1, smem_buffer);
}

// Required dynamic shared memory size for collaborative backward (with sub-tiling)
// Layer 1 requires the most: 128*41*4 + 128*32*4 = 20.5KB + 16KB = 36.5KB
constexpr int COLLABORATIVE_SMEM_SIZE = SUB_TILE_SIZE * 41 * sizeof(float) + SUB_TILE_SIZE * 32 * sizeof(float);

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
