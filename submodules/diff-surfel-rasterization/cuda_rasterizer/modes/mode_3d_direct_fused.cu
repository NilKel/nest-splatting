/*
 * Mode 5: 3D_direct_fused - In-kernel MLP evaluation with collaborative tile GEMM
 *
 * This mode runs hash lookup + MLP inside the rasterizer kernel, following the
 * per-intersection paradigm: sum(w_i * MLP(f_i))
 *
 * Key optimization: "Zeroes Matrix" + Collaborative Tile GEMM
 * - Non-participating pixels write zeros (no divergence issues)
 * - All 256 threads cooperate on matrix multiply for weight gradients
 * - No shared memory atomics during computation
 * - Single flush to global at end of tile
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
// MLP weight pointers struct (passed to inline functions)
// ============================================================================
struct MlpWeights {
    const float* W1;      // [32 * 40]
    const float* b1;      // [32]
    const float* W2;      // [32 * 32]
    const float* b2;      // [32]
    const float* W3_rgb;  // [3 * 32]
    const float* b3_rgb;  // [3]
};

// ============================================================================
// MLP Forward Pass (in registers, no storage)
// ============================================================================
__device__ __forceinline__ void mlp_forward_inline(
    const float* input,    // [40]
    float* output,         // [3]
    float* h1,             // [32] hidden1 post-ReLU (for backward)
    float* h2,             // [32] hidden2 post-ReLU (for backward)
    bool apply_sigmoid,
    const MlpWeights& mlp  // MLP weight pointers
) {
    // Layer 1: input[40] -> h1[32] with ReLU
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        float acc = mlp.b1[h];
        #pragma unroll
        for (int i = 0; i < 40; i++) {
            acc += input[i] * mlp.W1[h * 40 + i];
        }
        h1[h] = fmaxf(0.0f, acc);  // ReLU
    }

    // Layer 2: h1[32] -> h2[32] with ReLU
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        float acc = mlp.b2[h];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            acc += h1[i] * mlp.W2[h * 32 + i];
        }
        h2[h] = fmaxf(0.0f, acc);  // ReLU
    }

    // Layer 3: h2[32] -> output[3] with optional sigmoid
    #pragma unroll
    for (int o = 0; o < 3; o++) {
        float acc = mlp.b3_rgb[o];
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
// MLP Forward with pre-activation storage (for backward)
// ============================================================================
__device__ __forceinline__ void mlp_forward_with_preact(
    const float* input,    // [40]
    float* output,         // [3]
    float* h1_pre,         // [32] pre-ReLU (for backward)
    float* h1_post,        // [32] post-ReLU
    float* h2_pre,         // [32] pre-ReLU (for backward)
    float* h2_post,        // [32] post-ReLU
    bool apply_sigmoid,
    const MlpWeights& mlp  // MLP weight pointers
) {
    // Layer 1: input[40] -> h1[32] with ReLU
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        float acc = mlp.b1[h];
        #pragma unroll
        for (int i = 0; i < 40; i++) {
            acc += input[i] * mlp.W1[h * 40 + i];
        }
        h1_pre[h] = acc;
        h1_post[h] = fmaxf(0.0f, acc);
    }

    // Layer 2: h1[32] -> h2[32] with ReLU
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        float acc = mlp.b2[h];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            acc += h1_post[i] * mlp.W2[h * 32 + i];
        }
        h2_pre[h] = acc;
        h2_post[h] = fmaxf(0.0f, acc);
    }

    // Layer 3: h2[32] -> output[3] with optional sigmoid
    #pragma unroll
    for (int o = 0; o < 3; o++) {
        float acc = mlp.b3_rgb[o];
        #pragma unroll
        for (int h = 0; h < 32; h++) {
            acc += h2_post[h] * mlp.W3_rgb[o * 32 + h];
        }
        if (apply_sigmoid) {
            output[o] = 1.0f / (1.0f + expf(-acc));
        } else {
            output[o] = acc;
        }
    }
}

// ============================================================================
// Compute dL_dinput only (no weight gradients) - for fast feature/hash backward
// ============================================================================
__device__ __forceinline__ void mlp_backward_input_only(
    const float* h1_pre,      // [32] pre-ReLU
    const float* h1_post,     // [32] post-ReLU
    const float* h2_pre,      // [32] pre-ReLU
    const float* h2_post,     // [32] post-ReLU
    const float* output,      // [3] MLP output (after sigmoid)
    const float* dL_dout,     // [3] gradient from loss
    float* dL_dinput,         // [40] gradient to input (output)
    const MlpWeights& mlp     // MLP weight pointers
) {
    // Gradient through sigmoid
    float dL_dpre3[3];
    #pragma unroll
    for (int o = 0; o < 3; o++) {
        float sig = output[o];
        dL_dpre3[o] = dL_dout[o] * sig * (1.0f - sig);
    }

    // Layer 3 backward: dL_dh2
    float dL_dh2[32] = {0};
    #pragma unroll
    for (int o = 0; o < 3; o++) {
        #pragma unroll
        for (int h = 0; h < 32; h++) {
            dL_dh2[h] += dL_dpre3[o] * mlp.W3_rgb[o * 32 + h];
        }
    }

    // ReLU backward for h2
    float dL_dpre2[32];
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        dL_dpre2[h] = (h2_pre[h] > 0) ? dL_dh2[h] : 0;
    }

    // Layer 2 backward: dL_dh1
    float dL_dh1[32] = {0};
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            dL_dh1[i] += dL_dpre2[h] * mlp.W2[h * 32 + i];
        }
    }

    // ReLU backward for h1
    float dL_dpre1[32];
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        dL_dpre1[h] = (h1_pre[h] > 0) ? dL_dh1[h] : 0;
    }

    // Layer 1 backward: dL_dinput
    #pragma unroll
    for (int i = 0; i < 40; i++) {
        dL_dinput[i] = 0;
        #pragma unroll
        for (int h = 0; h < 32; h++) {
            dL_dinput[i] += dL_dpre1[h] * mlp.W1[h * 40 + i];
        }
    }
}

// ============================================================================
// Collaborative Tile GEMM for MLP weight gradients
// Uses "Zeroes Matrix" strategy: non-participating pixels contribute 0
// ============================================================================

// Helper: Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// OPTIMIZED Collaborative GEMM - Parallel output computation
// Uses DYNAMIC shared memory (up to 96KB on modern GPUs like RTX 3090/4090/5090)
// Instead of batching, we load ALL data at once and compute in parallel.
//
// IMPORTANT: The backward kernel must set cudaFuncSetAttribute to unlock
// shared memory beyond 48KB, and pass smem size in launch config.
// ============================================================================

// ============================================================================
// SUB-TILE COLLABORATIVE GEMM
// ============================================================================
// Process 256 threads in 2 batches of 128 to reduce shared memory usage:
//
// With full 256 threads:
//   Layer 1: 256×32 + 256×40 = 72KB  ← exceeds 62KB available
//
// With sub-tiles of 128:
//   Layer 1: 128×32 + 128×40 = 36KB  ✓ fits!
//   Layer 2: 128×32 + 128×32 = 32KB  ✓
//   Layer 3: 128×3  + 128×32 = 17KB  ✓
//
// Maximum needed: 36KB (fits in 62KB = 99KB max - 37KB static)
// ============================================================================

#define SUB_TILE_SIZE 128

// All-in-one collaborative backward using dynamic shared memory with sub-tiling
// This is called from the main backward kernel which sets up smem
__device__ void collaborative_mlp_backward_all(
    // Per-pixel data (in registers)
    const float* my_input,    // [40]
    const float* my_h1,       // [32]
    const float* my_h2,       // [32]
    const float* my_dL_dz1,   // [32]
    const float* my_dL_dz2,   // [32]
    const float* my_dL_dz3,   // [3]
    // Tile accumulators (in shared memory)
    float* tile_dL_dW1,       // [1280]
    float* tile_dL_db1,       // [32]
    float* tile_dL_dW2,       // [1024]
    float* tile_dL_db2,       // [32]
    float* tile_dL_dW3,       // [96]
    float* tile_dL_db3,       // [3]
    // Dynamic shared memory buffer
    float* smem_buffer        // Pointer to dynamic smem
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // ========================================
    // Layer 3: dL_dW3[3,32] = dL_dz3^T @ h2
    // Sub-tile: 128×3 + 128×32 = 17KB
    // ========================================
    {
        float* smem_dL_dz3 = smem_buffer;                      // [128][3]
        float* smem_h2 = smem_buffer + SUB_TILE_SIZE * 3;      // [128][32]

        // Process in 2 batches of 128 threads
        for (int batch = 0; batch < 2; batch++) {
            const int batch_start = batch * SUB_TILE_SIZE;
            const bool in_batch = (tid >= batch_start) && (tid < batch_start + SUB_TILE_SIZE);
            const int local_idx = tid - batch_start;

            // Load data: only threads in this batch write
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

            // Compute: threads 0-95 compute W3 gradients for this batch
            if (tid < 96) {
                int o = tid / 32;
                int h = tid % 32;
                float sum = 0;
                for (int p = 0; p < SUB_TILE_SIZE; p++) {
                    sum += smem_dL_dz3[p * 3 + o] * smem_h2[p * 32 + h];
                }
                tile_dL_dW3[tid] += sum;
            }

            // Bias: threads 0-2 compute b3 gradients for this batch
            if (tid < 3) {
                float sum = 0;
                for (int p = 0; p < SUB_TILE_SIZE; p++) {
                    sum += smem_dL_dz3[p * 3 + tid];
                }
                tile_dL_db3[tid] += sum;
            }
            __syncthreads();
        }
    }

    // ========================================
    // Layer 2: dL_dW2[32,32] = dL_dz2^T @ h1
    // Sub-tile: 128×32 + 128×32 = 32KB
    // ========================================
    {
        float* smem_dL_dz2 = smem_buffer;                      // [128][32]
        float* smem_h1 = smem_buffer + SUB_TILE_SIZE * 32;     // [128][32]

        // Process in 2 batches of 128 threads
        for (int batch = 0; batch < 2; batch++) {
            const int batch_start = batch * SUB_TILE_SIZE;
            const bool in_batch = (tid >= batch_start) && (tid < batch_start + SUB_TILE_SIZE);
            const int local_idx = tid - batch_start;

            // Load data: only threads in this batch write
            if (in_batch) {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    smem_dL_dz2[local_idx * 32 + i] = my_dL_dz2[i];
                    smem_h1[local_idx * 32 + i] = my_h1[i];
                }
            }
            __syncthreads();

            // Compute: threads 0-255 compute W2 gradients (4 passes for 1024 elements)
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

            // Bias: threads 0-31 compute b2 gradients for this batch
            if (tid < 32) {
                float sum = 0;
                for (int p = 0; p < SUB_TILE_SIZE; p++) {
                    sum += smem_dL_dz2[p * 32 + tid];
                }
                tile_dL_db2[tid] += sum;
            }
            __syncthreads();
        }
    }

    // ========================================
    // Layer 1: dL_dW1[32,40] = dL_dz1^T @ input
    // Sub-tile: 128×32 + 128×40 = 36KB (largest)
    // ========================================
    {
        float* smem_dL_dz1 = smem_buffer;                      // [128][32]
        float* smem_input = smem_buffer + SUB_TILE_SIZE * 32;  // [128][40]

        // Process in 2 batches of 128 threads
        for (int batch = 0; batch < 2; batch++) {
            const int batch_start = batch * SUB_TILE_SIZE;
            const bool in_batch = (tid >= batch_start) && (tid < batch_start + SUB_TILE_SIZE);
            const int local_idx = tid - batch_start;

            // Load data: only threads in this batch write
            if (in_batch) {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    smem_dL_dz1[local_idx * 32 + i] = my_dL_dz1[i];
                }
                #pragma unroll
                for (int i = 0; i < 40; i++) {
                    smem_input[local_idx * 40 + i] = my_input[i];
                }
            }
            __syncthreads();

            // Compute: threads compute W1 gradients (5 passes for 1280 elements)
            for (int pass = 0; pass < 5; pass++) {
                int out_idx = pass * 256 + tid;
                if (out_idx < 1280) {
                    int h = out_idx / 40;
                    int i = out_idx % 40;
                    float sum = 0;
                    for (int p = 0; p < SUB_TILE_SIZE; p++) {
                        sum += smem_dL_dz1[p * 32 + h] * smem_input[p * 40 + i];
                    }
                    tile_dL_dW1[out_idx] += sum;
                }
            }

            // Bias: threads 0-31 compute b1 gradients for this batch
            if (tid < 32) {
                float sum = 0;
                for (int p = 0; p < SUB_TILE_SIZE; p++) {
                    sum += smem_dL_dz1[p * 32 + tid];
                }
                tile_dL_db1[tid] += sum;
            }
            __syncthreads();
        }
    }
}

// Legacy function signatures for compatibility (redirect to all-in-one)
__device__ void collaborative_gemm_layer3(
    const float* my_dL_dz3,
    const float* my_h2,
    float* tile_dL_dW3,
    float* tile_dL_db3
) {
    // This is a stub - use collaborative_mlp_backward_all instead
    // Kept for API compatibility during transition
}

// Required dynamic shared memory size for collaborative backward (with sub-tiling)
// Call cudaFuncSetAttribute with cudaFuncAttributeMaxDynamicSharedMemorySize
// set to at least COLLABORATIVE_SMEM_SIZE before launching the kernel
constexpr int COLLABORATIVE_SMEM_SIZE = SUB_TILE_SIZE * 40 * sizeof(float) + SUB_TILE_SIZE * 32 * sizeof(float);
// = 128*40*4 + 128*32*4 = 20KB + 16KB = 36KB (Layer 1 requires the most)

// ============================================================================
// Compute all dL_dz values (pre-activation gradients) for collaborative GEMM
// Returns zeros if pixel doesn't participate (for zeroes matrix strategy)
// ============================================================================
__device__ void compute_dL_dz_all(
    const float* h1_pre,
    const float* h1_post,
    const float* h2_pre,
    const float* h2_post,
    const float* output,
    const float* dL_dout,     // [3] scaled by weight
    float* dL_dz3,            // [3] output
    float* dL_dz2,            // [32] output
    float* dL_dz1,            // [32] output
    const MlpWeights& mlp     // MLP weight pointers
) {
    // Gradient through sigmoid
    #pragma unroll
    for (int o = 0; o < 3; o++) {
        float sig = output[o];
        dL_dz3[o] = dL_dout[o] * sig * (1.0f - sig);
    }

    // Layer 3 backward -> dL_dh2
    float dL_dh2[32] = {0};
    #pragma unroll
    for (int o = 0; o < 3; o++) {
        #pragma unroll
        for (int h = 0; h < 32; h++) {
            dL_dh2[h] += dL_dz3[o] * mlp.W3_rgb[o * 32 + h];
        }
    }

    // ReLU backward -> dL_dz2
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        dL_dz2[h] = (h2_pre[h] > 0) ? dL_dh2[h] : 0;
    }

    // Layer 2 backward -> dL_dh1
    float dL_dh1[32] = {0};
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            dL_dh1[i] += dL_dz2[h] * mlp.W2[h * 32 + i];
        }
    }

    // ReLU backward -> dL_dz1
    #pragma unroll
    for (int h = 0; h < 32; h++) {
        dL_dz1[h] = (h1_pre[h] > 0) ? dL_dh1[h] : 0;
    }
}

namespace MODES {

// ============================================================================
// Forward pass implementation (only needed in forward.cu, not backward.cu)
// ============================================================================
#ifndef BACKWARD_CU_INCLUDES_MODE
__device__ void forward_3d_direct_fused(
    const float3& xyz,
    const float3& pk,
    const float3& sutu,
    const float3& svtv,
    const float2& s,
    const float rho3d,
    const float rho2d,
    const int gauss_id,
    const float* colors,
    const float* hash_features,
    const int* level_offsets,
    const float3& voxel_min,
    const float3& voxel_max,
    const float appearance_level,
    const int level,
    const int l_dim,
    const float l_scale,
    const int Base,
    const bool align_corners,
    const int interp,
    const bool contract,
    const glm::vec3* cam_pos,
    float* feat,
    const bool debug
) {
    // Decode level parameter
    const int hybrid_levels = level & 0xFF;
    const int active_hashgrid_levels = (level >> 8) & 0xFF;
    const int per_gaussian_dim = hybrid_levels * l_dim;

    // 1. Get per-Gaussian features (coarse levels, 20D max)
    float gauss_feat[20];
    if (hybrid_levels > 0 && colors != nullptr) {
        const float* per_gaussian_feat = &colors[gauss_id * per_gaussian_dim];
        for (int i = 0; i < per_gaussian_dim && i < 20; i++) {
            gauss_feat[i] = per_gaussian_feat[i];
        }
        for (int i = per_gaussian_dim; i < 20; i++) {
            gauss_feat[i] = 0.0f;
        }
    } else {
        for (int i = 0; i < 20; i++) gauss_feat[i] = 0.0f;
    }

    // 2. Query hashgrid for fine levels (4D)
    float hash_feat[4] = {0, 0, 0, 0};
    // Note: hash query would go here - for now simplified
    // In full implementation, call query_feature from auxiliary.h

    // 3. Encode view direction (16D)
    glm::vec3 cp = *cam_pos;
    float3 view_dir = {xyz.x - cp.x, xyz.y - cp.y, xyz.z - cp.z};
    float inv_len = rsqrtf(view_dir.x*view_dir.x + view_dir.y*view_dir.y + view_dir.z*view_dir.z + 1e-7f);
    view_dir.x *= inv_len;
    view_dir.y *= inv_len;
    view_dir.z *= inv_len;
    float view_enc[16];
    encode_view_direction(view_dir, view_enc);

    // 4. Build MLP input: [gauss(20) | hash(4) | view(16)] = 40D
    float mlp_input[40];
    for (int i = 0; i < 20; i++) mlp_input[i] = gauss_feat[i];
    for (int i = 0; i < 4; i++)  mlp_input[20 + i] = hash_feat[i];
    for (int i = 0; i < 16; i++) mlp_input[24 + i] = view_enc[i];

    // 5. Run MLP -> RGB with sigmoid
    float h1[32], h2[32];
    mlp_forward_inline(mlp_input, feat, h1, h2, true);
}
#endif // BACKWARD_CU_INCLUDES_MODE

// ============================================================================
// Backward pass implementation - Collaborative Tile GEMM version
// Called ONCE per Gaussian by ALL pixels in the tile
// Non-participating pixels pass zeros (zeroes matrix strategy)
// ============================================================================
__device__ void backward_3d_direct_fused(
    // Pixel participation flag
    const bool participates,
    // Intersection data (only valid if participates)
    const float3& xyz,
    const float3& pk,
    const float3& sutu,
    const float3& svtv,
    const float2& s,
    const float rho3d,
    const float rho2d,
    // Per-Gaussian data
    const int gauss_id,
    const float* colors,
    const float* hash_features,
    const int* level_offsets,
    const float3& voxel_min,
    const float3& voxel_max,
    const float appearance_level,
    const int level,
    const int l_dim,
    const float l_scale,
    const int Base,
    const bool align_corners,
    const int interp,
    const bool contract,
    const glm::vec3* cam_pos,
    // Gradient input (only valid if participates)
    const float* dL_drgb,     // [3] dL_dpixel
    const float weight,       // alpha * T
    // Tile-local accumulators (shared memory)
    float* tile_dL_dW1,
    float* tile_dL_db1,
    float* tile_dL_dW2,
    float* tile_dL_db2,
    float* tile_dL_dW3,
    float* tile_dL_db3,
    // Global gradient outputs
    float* dL_dcolors,
    float* dL_dfeatures,
    float* dL_dxyz,
    const bool debug,
    const MlpWeights& mlp
) {
    // ========================================
    // Phase 1: Per-pixel divergent computation
    // Non-participating pixels get all zeros
    // ========================================

    // Initialize all data to zero (for zeroes matrix strategy)
    float my_input[40] = {0};
    float my_h1_pre[32] = {0};
    float my_h1_post[32] = {0};
    float my_h2_pre[32] = {0};
    float my_h2_post[32] = {0};
    float my_output[3] = {0};
    float my_dL_dout[3] = {0};
    float my_dL_dz3[3] = {0};
    float my_dL_dz2[32] = {0};
    float my_dL_dz1[32] = {0};
    float my_dL_dinput[40] = {0};

    if (participates) {
        // Decode level parameter
        const int hybrid_levels = level & 0xFF;
        const int active_hashgrid_levels = (level >> 8) & 0xFF;
        const int per_gaussian_dim = hybrid_levels * l_dim;

        // 1. Reconstruct per-Gaussian features
        for (int i = 0; i < per_gaussian_dim && i < 20; i++) {
            my_input[i] = colors[gauss_id * per_gaussian_dim + i];
        }

        // 2. Query hash features (simplified - would call query_feature)
        // For now, hash features stay at 0

        // 3. Compute view direction
        glm::vec3 cp = *cam_pos;
        float3 view_dir = {xyz.x - cp.x, xyz.y - cp.y, xyz.z - cp.z};
        float inv_len = rsqrtf(view_dir.x*view_dir.x + view_dir.y*view_dir.y + view_dir.z*view_dir.z + 1e-7f);
        view_dir.x *= inv_len;
        view_dir.y *= inv_len;
        view_dir.z *= inv_len;
        float view_enc[16];
        encode_view_direction_bw(view_dir, view_enc);

        // 4. Build MLP input
        for (int i = 0; i < 16; i++) my_input[24 + i] = view_enc[i];

        // 5. Recompute MLP forward
        mlp_forward_with_preact(my_input, my_output, my_h1_pre, my_h1_post, my_h2_pre, my_h2_post, true, mlp);

        // 6. Scale gradient by weight
        for (int c = 0; c < 3; c++) {
            my_dL_dout[c] = dL_drgb[c] * weight;
        }

        // 7. Compute all dL_dz values for GEMM
        compute_dL_dz_all(my_h1_pre, my_h1_post, my_h2_pre, my_h2_post,
                         my_output, my_dL_dout, my_dL_dz3, my_dL_dz2, my_dL_dz1, mlp);

        // 8. Compute dL_dinput (for feature gradients)
        mlp_backward_input_only(my_h1_pre, my_h1_post, my_h2_pre, my_h2_post,
                               my_output, my_dL_dout, my_dL_dinput, mlp);
    }

    __syncthreads();

    // ========================================
    // Phase 2: Collaborative Tile GEMM
    // All 256 threads participate (zeros contribute nothing)
    // Uses dynamic shared memory (must be allocated by caller)
    // ========================================

    // Get pointer to dynamic shared memory (passed to kernel)
    extern __shared__ float dynamic_smem[];

    // All-in-one collaborative GEMM for all 3 layers
    collaborative_mlp_backward_all(
        my_input, my_h1_post, my_h2_post,
        my_dL_dz1, my_dL_dz2, my_dL_dz3,
        tile_dL_dW1, tile_dL_db1,
        tile_dL_dW2, tile_dL_db2,
        tile_dL_dW3, tile_dL_db3,
        dynamic_smem
    );

    // ========================================
    // Phase 3: Per-pixel gradients (only participating pixels)
    // ========================================

    if (participates) {
        const int hybrid_levels = level & 0xFF;
        const int active_hashgrid_levels = (level >> 8) & 0xFF;
        const int per_gaussian_dim = hybrid_levels * l_dim;

        // Backprop to per-Gaussian features
        for (int i = 0; i < per_gaussian_dim && i < 20; i++) {
            atomicAdd(&(dL_dcolors[gauss_id * per_gaussian_dim + i]), my_dL_dinput[i]);
        }

        // Backprop to hash features would go here (using query_feature<true, ...>)
    }
}

// ============================================================================
// Flush tile-local MLP gradients to global memory
// Called once per tile after processing all Gaussians
// ============================================================================
__device__ void flush_tile_mlp_grads(
    const float* tile_dL_dW1,
    const float* tile_dL_db1,
    const float* tile_dL_dW2,
    const float* tile_dL_db2,
    const float* tile_dL_dW3,
    const float* tile_dL_db3,
    float* global_dL_dW1,
    float* global_dL_db1,
    float* global_dL_dW2,
    float* global_dL_db2,
    float* global_dL_dW3,
    float* global_dL_db3,
    const int thread_id
) {
    // Each thread handles a subset of the weights
    // Use conditional atomicAdd to skip zeros

    // W1: 1280 elements, 256 threads -> 5 elements per thread
    for (int idx = thread_id; idx < 1280; idx += 256) {
        if (tile_dL_dW1[idx] != 0.0f)
            atomicAdd(&global_dL_dW1[idx], tile_dL_dW1[idx]);
    }

    // b1: 32 elements
    for (int idx = thread_id; idx < 32; idx += 256) {
        if (tile_dL_db1[idx] != 0.0f)
            atomicAdd(&global_dL_db1[idx], tile_dL_db1[idx]);
    }

    // W2: 1024 elements
    for (int idx = thread_id; idx < 1024; idx += 256) {
        if (tile_dL_dW2[idx] != 0.0f)
            atomicAdd(&global_dL_dW2[idx], tile_dL_dW2[idx]);
    }

    // b2: 32 elements
    for (int idx = thread_id; idx < 32; idx += 256) {
        if (tile_dL_db2[idx] != 0.0f)
            atomicAdd(&global_dL_db2[idx], tile_dL_db2[idx]);
    }

    // W3: 96 elements
    for (int idx = thread_id; idx < 96; idx += 256) {
        if (tile_dL_dW3[idx] != 0.0f)
            atomicAdd(&global_dL_dW3[idx], tile_dL_dW3[idx]);
    }

    // b3: 3 elements
    for (int idx = thread_id; idx < 3; idx += 256) {
        if (tile_dL_db3[idx] != 0.0f)
            atomicAdd(&global_dL_db3[idx], tile_dL_db3[idx]);
    }
}

} // namespace MODES

#endif // MODE_3D_DIRECT_FUSED_CU_INCLUDED
