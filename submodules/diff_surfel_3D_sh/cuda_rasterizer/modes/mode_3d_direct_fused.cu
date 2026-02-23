/*
 * Mode 5: 3D_SH_TC - In-kernel MLP outputting SH coefficients with WMMA backward
 *
 * MLP outputs 48D SH coefficients (degree 3, 16 per channel × 3 RGB).
 * Per-intersection view direction is used to evaluate SH → RGB after MLP.
 * This enables baking: cache 48D SH coefficients, evaluate cheaply at any viewdir.
 *
 * Key optimizations:
 * - WMMA Tensor Core GEMM for backward weight gradients
 * - "Zeroes Matrix" strategy: non-participating pixels contribute 0
 * - FP16 weights in shared memory, FP32 accumulators for weight gradients
 * - Bias-free MLP: L1 uses input padding (input[24]=1.0, W1[32×32])
 *
 * WMMA-aligned MLP dimensions (all multiples of 16):
 *   Input:  32D (25D real + 7D padding zeros)
 *   Hidden: 32D (unchanged)
 *   Output: 48D (16 SH coefs × 3 RGB channels, already aligned)
 *   W1: [32, 32], W2: [32, 32], W3: [48, 32]
 */

#ifndef MODE_3D_DIRECT_FUSED_CU_INCLUDED
#define MODE_3D_DIRECT_FUSED_CU_INCLUDED

// When included from backward.cu, all declarations already exist
#ifndef BACKWARD_CU_INCLUDES_MODE
#include "modes.h"
#include "../auxiliary.h"
#include "../config.h"
#include "../hashgrid.h"
#include "../mma_utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif // BACKWARD_CU_INCLUDES_MODE

// ============================================================================
// SH evaluation constants (degree 3, matching utils/sh_utils.py)
// SH_C0 and SH_C1 are already defined in auxiliary.h
// ============================================================================
__device__ constexpr float SH_C2_0 =  1.0925484305920792f;
__device__ constexpr float SH_C2_1 = -1.0925484305920792f;
__device__ constexpr float SH_C2_2 =  0.31539156525252005f;
__device__ constexpr float SH_C2_3 = -1.0925484305920792f;
__device__ constexpr float SH_C2_4 =  0.5462742152960396f;
__device__ constexpr float SH_C3_0 = -0.5900435899266435f;
__device__ constexpr float SH_C3_1 =  2.890611442640554f;
__device__ constexpr float SH_C3_2 = -0.4570457994644658f;
__device__ constexpr float SH_C3_3 =  0.3731763325901154f;
__device__ constexpr float SH_C3_4 = -0.4570457994644658f;
__device__ constexpr float SH_C3_5 =  1.445305721320277f;
__device__ constexpr float SH_C3_6 = -0.5900435899266435f;

// ============================================================================
// Evaluate degree-3 SH basis functions at unit direction (x,y,z)
// Returns 16 basis values in basis[0..15]
// ============================================================================
__device__ __forceinline__ void eval_sh_basis_degree3(
    float x, float y, float z, float* basis
) {
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    // Degree 0 (1 basis)
    basis[0] = SH_C0;

    // Degree 1 (3 basis)
    basis[1] = -SH_C1 * y;
    basis[2] =  SH_C1 * z;
    basis[3] = -SH_C1 * x;

    // Degree 2 (5 basis)
    basis[4] = SH_C2_0 * xy;
    basis[5] = SH_C2_1 * yz;
    basis[6] = SH_C2_2 * (2.0f * zz - xx - yy);
    basis[7] = SH_C2_3 * xz;
    basis[8] = SH_C2_4 * (xx - yy);

    // Degree 3 (7 basis)
    basis[9]  = SH_C3_0 * y * (3.0f * xx - yy);
    basis[10] = SH_C3_1 * xy * z;
    basis[11] = SH_C3_2 * y * (4.0f * zz - xx - yy);
    basis[12] = SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
    basis[13] = SH_C3_4 * x * (4.0f * zz - xx - yy);
    basis[14] = SH_C3_5 * z * (xx - yy);
    basis[15] = SH_C3_6 * x * (xx - 3.0f * yy);
}

// ============================================================================
// Evaluate SH for one color channel: dot(sh_coefs[16], basis[16])
// sh points to 16 coefficients for this channel
// ============================================================================
__device__ __forceinline__ float eval_sh_channel(
    const float* sh, const float* basis
) {
    float result = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        result += sh[i] * basis[i];
    }
    return result;
}

// ============================================================================
// MLP weight pointers struct (bias-free, WMMA-padded dimensions)
// L1: W1[32×32] (col 24 = implicit bias via input[24]=1.0, cols 25-31 = zero)
// L2: W2[32×32] (no bias, already WMMA-aligned)
// L3: W3[48×32] (all 48 rows used for SH coefficients)
// ============================================================================
struct MlpWeights {
    const __half* W1;      // [32 * 32] FP16 (WMMA-padded from 32*25)
    const __half* W2;      // [32 * 32] FP16
    const __half* W3_rgb;  // [48 * 32] FP16 (all 48 SH outputs used)
};

// ============================================================================
// MLP Forward Pass (bias-free, WMMA-padded weights, scalar per-pixel)
// Input must have input[24] = 1.0f set by caller for implicit L1 bias
// Input positions 25-31 must be 0.0f (WMMA padding)
// Output: 48 raw SH coefficients (no activation)
// ============================================================================
__device__ __forceinline__ void mlp_forward_inline(
    const float* input,    // [TC_INPUT_DIM=32] (input[24] = 1.0, [25..31] = 0)
    float* output,         // [TC_OUTPUT_DIM=48] raw SH coefficients
    float* h1,             // [TC_HIDDEN_DIM=32] hidden1 post-ReLU (for backward)
    float* h2,             // [TC_HIDDEN_DIM=32] hidden2 post-ReLU (for backward)
    bool apply_sigmoid,    // ignored for SH mode (always identity)
    const MlpWeights& mlp  // MLP weight pointers
) {
    // Layer 1: input[32] -> h1[32] with ReLU
    #pragma unroll
    for (int h = 0; h < TC_HIDDEN_DIM; h++) {
        float acc = 0;
        #pragma unroll
        for (int i = 0; i < TC_INPUT_DIM; i++) {
            acc += input[i] * __half2float(mlp.W1[h * TC_INPUT_DIM + i]);
        }
        h1[h] = fmaxf(0.0f, acc);  // ReLU
    }

    // Layer 2: h1[32] -> h2[32] with ReLU
    #pragma unroll
    for (int h = 0; h < TC_HIDDEN_DIM; h++) {
        float acc = 0;
        #pragma unroll
        for (int i = 0; i < TC_HIDDEN_DIM; i++) {
            acc += h1[i] * __half2float(mlp.W2[h * TC_HIDDEN_DIM + i]);
        }
        h2[h] = fmaxf(0.0f, acc);  // ReLU
    }

    // Layer 3: h2[32] -> output[48] (identity activation, all 48 SH coefs)
    #pragma unroll
    for (int o = 0; o < TC_OUTPUT_DIM; o++) {
        float acc = 0;
        #pragma unroll
        for (int h = 0; h < TC_HIDDEN_DIM; h++) {
            acc += h2[h] * __half2float(mlp.W3_rgb[o * TC_HIDDEN_DIM + h]);
        }
        output[o] = acc;  // No activation — raw SH coefficients
    }
}

// ============================================================================
// WMMA Tensor Core GEMM for MLP weight gradients
//
// Memory layout:
//   dL_dz stored as [256, M] row-major in smem → matrix_a col_major, stride=M
//   activation stored as [256, N] row-major in smem → matrix_b row_major, stride=N
//   tile_dL_dW[M, N] row-major in shared memory → accumulator load/store
// ============================================================================

// Layer 3: dL_dW3[48,32] = dL_dz3^T[48,256] @ h2[256,32]
// Output tiles: 3×2 = 6 (warps 0-5 active)
static __device__ void wmma_gemm_layer3(
    const float* my_dL_dz3,    // [TC_OUTPUT_DIM=48] per-pixel
    const float* my_h2,        // [TC_HIDDEN_DIM=32] per-pixel
    float* tile_dL_dW3,        // [W3_SIZE=1536] shared mem accumulator (FP32)
    __half* smem_buffer        // dynamic shared memory (FP16)
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    __half* smem_dz = smem_buffer;                              // [256, 48]
    __half* smem_h2 = smem_buffer + TC_BATCH * TC_OUTPUT_DIM;   // [256, 32]

    // Store phase: all 256 threads write float→half
    #pragma unroll
    for (int i = 0; i < TC_OUTPUT_DIM; i++) {
        smem_dz[tid * TC_OUTPUT_DIM + i] = __float2half(my_dL_dz3[i]);
    }
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        smem_h2[tid * TC_HIDDEN_DIM + i] = __float2half(my_h2[i]);
    }
    __syncthreads();

    // WMMA compute: 6 output tiles distributed to warps
    for (int tile_idx = warp_id; tile_idx < BW_L3_TILES; tile_idx += WARPS_PER_BLOCK) {
        int m_tile = tile_idx / BW_L3_N_TILES;
        int n_tile = tile_idx % BW_L3_N_TILES;

        // Load existing accumulator (to accumulate across Gaussians)
        frag_acc frag_c;
        wmma::load_matrix_sync(frag_c,
            tile_dL_dW3 + m_tile * WMMA_M * W3_COLS + n_tile * WMMA_N,
            W3_COLS, wmma::mem_row_major);

        // K-loop: 16 chunks of 16 pixels
        for (int k = 0; k < BW_K_CHUNKS; k++) {
            bw_frag_a frag_a;
            bw_frag_b frag_b;
            wmma::load_matrix_sync(frag_a,
                smem_dz + k * WMMA_K * TC_OUTPUT_DIM + m_tile * WMMA_M,
                TC_OUTPUT_DIM);
            wmma::load_matrix_sync(frag_b,
                smem_h2 + k * WMMA_K * TC_HIDDEN_DIM + n_tile * WMMA_N,
                TC_HIDDEN_DIM);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        wmma::store_matrix_sync(
            tile_dL_dW3 + m_tile * WMMA_M * W3_COLS + n_tile * WMMA_N,
            frag_c, W3_COLS, wmma::mem_row_major);
    }
    __syncthreads();
}

// Layer 2: dL_dW2[32,32] = dL_dz2^T[32,256] @ h1[256,32]
// Output tiles: 2×2 = 4 (warps 0-3 active)
static __device__ void wmma_gemm_layer2(
    const float* my_dL_dz2,    // [TC_HIDDEN_DIM=32] per-pixel
    const float* my_h1,        // [TC_HIDDEN_DIM=32] per-pixel
    float* tile_dL_dW2,        // [W2_SIZE=1024] shared mem accumulator (FP32)
    __half* smem_buffer        // dynamic shared memory
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    __half* smem_dz = smem_buffer;                              // [256, 32]
    __half* smem_h1 = smem_buffer + TC_BATCH * TC_HIDDEN_DIM;   // [256, 32]

    // Store phase: all 256 threads write float→half
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        smem_dz[tid * TC_HIDDEN_DIM + i] = __float2half(my_dL_dz2[i]);
        smem_h1[tid * TC_HIDDEN_DIM + i] = __float2half(my_h1[i]);
    }
    __syncthreads();

    // WMMA compute: 4 output tiles distributed to warps
    for (int tile_idx = warp_id; tile_idx < BW_L2_TILES; tile_idx += WARPS_PER_BLOCK) {
        int m_tile = tile_idx / BW_L2_N_TILES;
        int n_tile = tile_idx % BW_L2_N_TILES;

        frag_acc frag_c;
        wmma::load_matrix_sync(frag_c,
            tile_dL_dW2 + m_tile * WMMA_M * W2_COLS + n_tile * WMMA_N,
            W2_COLS, wmma::mem_row_major);

        for (int k = 0; k < BW_K_CHUNKS; k++) {
            bw_frag_a frag_a;
            bw_frag_b frag_b;
            wmma::load_matrix_sync(frag_a,
                smem_dz + k * WMMA_K * TC_HIDDEN_DIM + m_tile * WMMA_M,
                TC_HIDDEN_DIM);
            wmma::load_matrix_sync(frag_b,
                smem_h1 + k * WMMA_K * TC_HIDDEN_DIM + n_tile * WMMA_N,
                TC_HIDDEN_DIM);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        wmma::store_matrix_sync(
            tile_dL_dW2 + m_tile * WMMA_M * W2_COLS + n_tile * WMMA_N,
            frag_c, W2_COLS, wmma::mem_row_major);
    }
    __syncthreads();
}

// Layer 1: dL_dW1[32,32] = dL_dz1^T[32,256] @ input[256,32]
// Output tiles: 2×2 = 4 (warps 0-3 active)
static __device__ void wmma_gemm_layer1(
    const float* my_dL_dz1,    // [TC_HIDDEN_DIM=32] per-pixel
    const float* my_input,     // [TC_INPUT_DIM=32] per-pixel (pos 25-31 = 0)
    float* tile_dL_dW1,        // [W1_SIZE=1024] shared mem accumulator (FP32)
    __half* smem_buffer        // dynamic shared memory
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    __half* smem_dz    = smem_buffer;                            // [256, 32]
    __half* smem_input = smem_buffer + TC_BATCH * TC_HIDDEN_DIM; // [256, 32]

    // Store phase: all 256 threads write float→half
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        smem_dz[tid * TC_HIDDEN_DIM + i] = __float2half(my_dL_dz1[i]);
    }
    #pragma unroll
    for (int i = 0; i < TC_INPUT_DIM; i++) {
        smem_input[tid * TC_INPUT_DIM + i] = __float2half(my_input[i]);
    }
    __syncthreads();

    // WMMA compute: 4 output tiles distributed to warps
    for (int tile_idx = warp_id; tile_idx < BW_L1_TILES; tile_idx += WARPS_PER_BLOCK) {
        int m_tile = tile_idx / BW_L1_N_TILES;
        int n_tile = tile_idx % BW_L1_N_TILES;

        frag_acc frag_c;
        wmma::load_matrix_sync(frag_c,
            tile_dL_dW1 + m_tile * WMMA_M * W1_COLS + n_tile * WMMA_N,
            W1_COLS, wmma::mem_row_major);

        for (int k = 0; k < BW_K_CHUNKS; k++) {
            bw_frag_a frag_a;
            bw_frag_b frag_b;
            wmma::load_matrix_sync(frag_a,
                smem_dz + k * WMMA_K * TC_HIDDEN_DIM + m_tile * WMMA_M,
                TC_HIDDEN_DIM);
            wmma::load_matrix_sync(frag_b,
                smem_input + k * WMMA_K * TC_INPUT_DIM + n_tile * WMMA_N,
                TC_INPUT_DIM);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        wmma::store_matrix_sync(
            tile_dL_dW1 + m_tile * WMMA_M * W1_COLS + n_tile * WMMA_N,
            frag_c, W1_COLS, wmma::mem_row_major);
    }
    __syncthreads();
}

// Required dynamic shared memory size for WMMA backward
// Layer 3 is the largest: 256*(48+32)*sizeof(half) = 40,960 bytes
constexpr int COLLABORATIVE_SMEM_SIZE = TC_COLLABORATIVE_SMEM_SIZE;

// Required dynamic shared memory size for WMMA forward (disabled)
constexpr int FW_SMEM_HALF_BYTES = TC_FORWARD_SMEM_HALF;
constexpr int FW_SMEM_FLOAT_BYTES = TC_FORWARD_SMEM_FLOAT;
constexpr int FW_SMEM_TOTAL_BYTES = TC_FORWARD_SMEM_SIZE;

// ============================================================================
// WMMA forward layer: Out[256, N] = In[256, K] @ W^T[K, N]
// W stored as [N, K] row-major = W^T[K, N] col-major (no transpose needed)
// (Currently disabled — scalar forward is faster for small MLPs)
// ============================================================================
template <int K, int N>
__device__ void wmma_forward_layer(
    const __half* smem_input,   // [TC_BATCH, K] row-major half
    const __half* smem_W,       // [N, K] row-major half
    float* smem_output          // [TC_BATCH, N] row-major float
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;

    constexpr int M_TILES = TC_BATCH / WMMA_M;
    constexpr int N_TILES = N / WMMA_N;
    constexpr int K_CHUNKS = K / WMMA_K;
    constexpr int TOTAL_TILES = M_TILES * N_TILES;

    for (int tile_idx = warp_id; tile_idx < TOTAL_TILES; tile_idx += WARPS_PER_BLOCK) {
        int m_tile = tile_idx / N_TILES;
        int n_tile = tile_idx % N_TILES;

        frag_acc frag_c;
        wmma::fill_fragment(frag_c, 0.0f);

        for (int kk = 0; kk < K_CHUNKS; kk++) {
            fw_frag_a frag_a;
            fw_frag_b frag_b;
            wmma::load_matrix_sync(frag_a,
                smem_input + m_tile * WMMA_M * K + kk * WMMA_K, K);
            wmma::load_matrix_sync(frag_b,
                smem_W + n_tile * WMMA_N * K + kk * WMMA_K, K);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        wmma::store_matrix_sync(
            smem_output + m_tile * WMMA_M * N + n_tile * WMMA_N,
            frag_c, N, wmma::mem_row_major);
    }
}

// ============================================================================
// WMMA forward: full 3-layer MLP for all 256 pixels simultaneously
// (Currently disabled — scalar forward is faster for small MLPs)
// ============================================================================
static __device__ void wmma_forward_all(
    __half* smem_fw_half,       // [TC_BATCH * TC_INPUT_DIM] input, reused for intermediates
    float*  smem_fw_float,      // [TC_BATCH * TC_HIDDEN_DIM] accumulator output
    const __half* smem_W1,      // [W1_SIZE] = [32, 32] row-major half
    const __half* smem_W2,      // [W2_SIZE] = [32, 32] row-major half
    const __half* smem_W3       // [W3_SIZE] = [48, 32] row-major half
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Layer 1: H1[256,32] = Input[256,32] × W1^T[32,32]
    wmma_forward_layer<TC_INPUT_DIM, TC_HIDDEN_DIM>(smem_fw_half, smem_W1, smem_fw_float);
    __syncthreads();

    // ReLU + float→half: reuse smem_fw_half as [256×32]
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        float val = fmaxf(0.0f, smem_fw_float[tid * TC_HIDDEN_DIM + i]);
        smem_fw_half[tid * TC_HIDDEN_DIM + i] = __float2half(val);
    }
    __syncthreads();

    // Layer 2: H2[256,32] = H1_relu[256,32] × W2^T[32,32]
    wmma_forward_layer<TC_HIDDEN_DIM, TC_HIDDEN_DIM>(smem_fw_half, smem_W2, smem_fw_float);
    __syncthreads();

    // ReLU + float→half
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        float val = fmaxf(0.0f, smem_fw_float[tid * TC_HIDDEN_DIM + i]);
        smem_fw_half[tid * TC_HIDDEN_DIM + i] = __float2half(val);
    }
    __syncthreads();

    // Layer 3: Out[256,48] = H2_relu[256,32] × W3^T[32,48]
    wmma_forward_layer<TC_HIDDEN_DIM, TC_OUTPUT_DIM>(smem_fw_half, smem_W3, smem_fw_float);
    __syncthreads();

    // Output: smem_fw_float[tid * TC_OUTPUT_DIM + i] (raw SH coefficients)
}

namespace MODES {

// ============================================================================
// Flush tile-local MLP gradients to global memory (WMMA-padded dimensions)
// Called once per tile after processing all Gaussians
// ============================================================================
static __device__ void flush_tile_mlp_grads(
    const float* tile_dL_dW1,
    const float* tile_dL_dW2,
    const float* tile_dL_dW3,
    float* global_dL_dW1,
    float* global_dL_dW2,
    float* global_dL_dW3,
    const int thread_id
) {
    // W1: 32×32 = 1024 floats
    for (int idx = thread_id; idx < W1_SIZE; idx += 256) {
        if (tile_dL_dW1[idx] != 0.0f)
            atomicAdd(&global_dL_dW1[idx], tile_dL_dW1[idx]);
    }
    // W2: 32×32 = 1024 floats
    for (int idx = thread_id; idx < W2_SIZE; idx += 256) {
        if (tile_dL_dW2[idx] != 0.0f)
            atomicAdd(&global_dL_dW2[idx], tile_dL_dW2[idx]);
    }
    // W3: 48×32 = 1536 floats
    for (int idx = thread_id; idx < W3_SIZE; idx += 256) {
        if (tile_dL_dW3[idx] != 0.0f)
            atomicAdd(&global_dL_dW3[idx], tile_dL_dW3[idx]);
    }
}

} // namespace MODES

#endif // MODE_3D_DIRECT_FUSED_CU_INCLUDED
