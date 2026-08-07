/*
 * Mode 5: 3D_SH_res - SH base color + Hash MLP residual with WMMA Tensor Core GEMM
 *
 * Per-intersection: rgb = SH(viewdir) + MLP(hash(xyz))
 *   SH evaluated in preprocessing (per-Gaussian, once)
 *   MLP evaluated per-intersection at hash grid features
 *
 * Key optimizations:
 * - WMMA Tensor Core GEMM for backward weight gradients
 *   Each mma_sync computes 16×16×16 = 4096 FP16 FMAs in one instruction
 * - "Zeroes Matrix" strategy: non-participating pixels contribute 0
 * - FP16 weights in shared memory, FP32 accumulators for weight gradients
 * - Bias-free MLP: L1 uses input padding (input[4]=1.0, rest zeros),
 *   L2/L3 have no bias. Guarantees W@0=0 for zeroes matrix correctness.
 *
 * WMMA-aligned MLP dimensions (all 16, single tile per layer):
 *   Input:  16D [hash(4) | bias(1) | pad(11)]
 *   Hidden: 16D
 *   Output: 16D (only first 3 = RGB residual, identity activation)
 *   W1: [16, 16], W2: [16, 16], W3: [16, 16]
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
// MLP weight pointers struct (bias-free, all [16×16])
// L1: W1[16×16] (col 4 = implicit bias via input[4]=1.0, cols 5-15 = zero)
// L2: W2[16×16] (no bias)
// L3: W3[16×16] (rows 0-2 = RGB residual, rows 3-15 = zero padding)
// ============================================================================
struct MlpWeights {
    const __half* W1;   // [16 * 16] FP16
    const __half* W2;   // [16 * 16] FP16
    const __half* W3;   // [16 * 16] FP16 (only first 3 rows = RGB residual)
};

// ============================================================================
// MLP Forward Pass (bias-free, all [16×16] weights, scalar per-pixel)
// Input must have input[4] = 1.0f set by caller for implicit L1 bias
// Input positions 5-15 must be 0.0f (WMMA padding)
// Output: 3D RGB residual with identity activation (no sigmoid)
// ============================================================================
__device__ __forceinline__ void mlp_forward_inline(
    const float* input,    // [TC_INPUT_DIM=16] (input[4] = 1.0, [5..15] = 0)
    float* output,         // [ORIG_OUTPUT_DIM=3]
    float* h1,             // [TC_HIDDEN_DIM=16] hidden1 post-ReLU (for backward)
    float* h2,             // [TC_HIDDEN_DIM=16] hidden2 post-ReLU (for backward)
    bool apply_sigmoid,
    const MlpWeights& mlp  // MLP weight pointers
) {
    // Convert input to FP16 once
    __half input_h[TC_INPUT_DIM];
    #pragma unroll
    for (int i = 0; i < TC_INPUT_DIM; i++)
        input_h[i] = __float2half(input[i]);

    // Layer 1: input[16] -> h1[16] with ReLU — FP16 __half2 dot products
    #pragma unroll
    for (int h = 0; h < TC_HIDDEN_DIM; h++) {
        __half2 acc2 = __float2half2_rn(0.0f);
        const __half* w1_row = &mlp.W1[h * TC_INPUT_DIM];
        #pragma unroll
        for (int i = 0; i < TC_INPUT_DIM; i += 2) {
            __half2 in2 = *reinterpret_cast<const __half2*>(&input_h[i]);
            __half2 wt2 = *reinterpret_cast<const __half2*>(&w1_row[i]);
            acc2 = __hfma2(in2, wt2, acc2);
        }
        float acc = __half2float(acc2.x) + __half2float(acc2.y);
        h1[h] = fmaxf(0.0f, acc);  // ReLU (store FP32 for backward)
    }

    // Convert h1 to FP16 for layer 2
    __half h1_h[TC_HIDDEN_DIM];
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++)
        h1_h[i] = __float2half(h1[i]);

    // Layer 2: h1[16] -> h2[16] with ReLU — FP16 __half2 dot products
    #pragma unroll
    for (int h = 0; h < TC_HIDDEN_DIM; h++) {
        __half2 acc2 = __float2half2_rn(0.0f);
        const __half* w2_row = &mlp.W2[h * TC_HIDDEN_DIM];
        #pragma unroll
        for (int i = 0; i < TC_HIDDEN_DIM; i += 2) {
            __half2 in2 = *reinterpret_cast<const __half2*>(&h1_h[i]);
            __half2 wt2 = *reinterpret_cast<const __half2*>(&w2_row[i]);
            acc2 = __hfma2(in2, wt2, acc2);
        }
        float acc = __half2float(acc2.x) + __half2float(acc2.y);
        h2[h] = fmaxf(0.0f, acc);  // ReLU (store FP32 for backward)
    }

    // Convert h2 to FP16 for layer 3
    __half h2_h[TC_HIDDEN_DIM];
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++)
        h2_h[i] = __float2half(h2[i]);

    // Layer 3: h2[16] -> output[3] — FP16 __half2 dot products
    #pragma unroll
    for (int o = 0; o < ORIG_OUTPUT_DIM; o++) {
        __half2 acc2 = __float2half2_rn(0.0f);
        const __half* w3_row = &mlp.W3[o * TC_HIDDEN_DIM];
        #pragma unroll
        for (int h = 0; h < TC_HIDDEN_DIM; h += 2) {
            __half2 in2 = *reinterpret_cast<const __half2*>(&h2_h[h]);
            __half2 wt2 = *reinterpret_cast<const __half2*>(&w3_row[h]);
            acc2 = __hfma2(in2, wt2, acc2);
        }
        float acc = __half2float(acc2.x) + __half2float(acc2.y);
        if (apply_sigmoid) {
            output[o] = 1.0f / (1.0f + expf(-acc));
        } else {
            output[o] = acc;
        }
    }
}

// ============================================================================
// WMMA Tensor Core GEMM for MLP weight gradients
//
// Each layer: dL_dW[M,N] = dL_dz^T[M,K] @ activation[K,N]
// For 3D_SH_res all layers are 1×1 tiles (16×16), K=256 pixels / 16 = 16 chunks
//
// Memory layout:
//   dL_dz stored as [256, M] row-major in smem → matrix_a col_major, stride=M
//   activation stored as [256, N] row-major in smem → matrix_b row_major, stride=N
//   tile_dL_dW[M, N] row-major in shared memory → accumulator load/store
// ============================================================================

// Layer 3: dL_dW3[16,16] = dL_dz3^T[16,256] @ h2[256,16]
// Output tiles: 1×1 = 1 (warp 0 active)
static __device__ void wmma_gemm_layer3(
    const float* my_dL_dz3,    // [TC_OUTPUT_DIM=16] per-pixel (FP32 registers, positions 3-15 = 0)
    const float* my_h2,        // [TC_HIDDEN_DIM=16] per-pixel (FP32 registers)
    float* tile_dL_dW3,        // [W3_SIZE=256] shared mem accumulator (FP32)
    __half* smem_buffer        // dynamic shared memory (FP16)
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    __half* smem_dz = smem_buffer;                              // [256, 16]
    __half* smem_h2 = smem_buffer + TC_BATCH * TC_OUTPUT_DIM;   // [256, 16]

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

    // WMMA compute: 1 output tile (warp 0)
    for (int tile_idx = warp_id; tile_idx < BW_L3_TILES; tile_idx += WARPS_PER_BLOCK) {
        int m_tile = tile_idx / BW_L3_N_TILES;
        int n_tile = tile_idx % BW_L3_N_TILES;

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

// Layer 2: dL_dW2[16,16] = dL_dz2^T[16,256] @ h1[256,16]
// Output tiles: 1×1 = 1 (warp 0 active)
static __device__ void wmma_gemm_layer2(
    const float* my_dL_dz2,    // [TC_HIDDEN_DIM=16] per-pixel
    const float* my_h1,        // [TC_HIDDEN_DIM=16] per-pixel
    float* tile_dL_dW2,        // [W2_SIZE=256] shared mem accumulator (FP32)
    __half* smem_buffer        // dynamic shared memory
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    __half* smem_dz = smem_buffer;                              // [256, 16]
    __half* smem_h1 = smem_buffer + TC_BATCH * TC_HIDDEN_DIM;   // [256, 16]

    // Store phase: all 256 threads write float→half
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        smem_dz[tid * TC_HIDDEN_DIM + i] = __float2half(my_dL_dz2[i]);
        smem_h1[tid * TC_HIDDEN_DIM + i] = __float2half(my_h1[i]);
    }
    __syncthreads();

    // WMMA compute: 1 output tile (warp 0)
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

// Layer 1: dL_dW1[16,16] = dL_dz1^T[16,256] @ input[256,16]
// Output tiles: 1×1 = 1 (warp 0 active)
static __device__ void wmma_gemm_layer1(
    const float* my_dL_dz1,    // [TC_HIDDEN_DIM=16] per-pixel
    const float* my_input,     // [TC_INPUT_DIM=16] per-pixel (pos 5-15 = 0)
    float* tile_dL_dW1,        // [W1_SIZE=256] shared mem accumulator (FP32)
    __half* smem_buffer        // dynamic shared memory
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    __half* smem_dz    = smem_buffer;                            // [256, 16]
    __half* smem_input = smem_buffer + TC_BATCH * TC_HIDDEN_DIM; // [256, 16]

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

    // WMMA compute: 1 output tile (warp 0)
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
// All layers use same size: 256*(16+16)*sizeof(half) = 16,384 bytes
constexpr int COLLABORATIVE_SMEM_SIZE = TC_COLLABORATIVE_SMEM_SIZE;

// Required dynamic shared memory size for WMMA forward
constexpr int FW_SMEM_HALF_BYTES = TC_FORWARD_SMEM_HALF;
constexpr int FW_SMEM_FLOAT_BYTES = TC_FORWARD_SMEM_FLOAT;
constexpr int FW_SMEM_TOTAL_BYTES = TC_FORWARD_SMEM_SIZE;

// ============================================================================
// WMMA forward layer: Out[256, N] = In[256, K] @ W^T[K, N]
// W stored as [N, K] row-major = W^T[K, N] col-major (no transpose needed)
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
//
// Input:  smem_fw_half[256 * TC_INPUT_DIM] — each thread's MLP input as half
//         (non-participating threads store zeros; bias-free MLP ⇒ W@0=0)
// Output: smem_fw_float[256 * TC_OUTPUT_DIM] — raw pre-activation MLP output
//         Caller reads first ORIG_OUTPUT_DIM=3 channels (identity for residual)
//
// Shared memory reuse (all 16D):
//   L1: reads smem_fw_half[256×16], writes smem_fw_float[256×16]
//        → ReLU+half → smem_fw_half[256×16]
//   L2: reads smem_fw_half[256×16], writes smem_fw_float[256×16]
//        → ReLU+half → smem_fw_half[256×16]
//   L3: reads smem_fw_half[256×16], writes smem_fw_float[256×16]
// ============================================================================
static __device__ void wmma_forward_all(
    __half* smem_fw_half,       // [TC_BATCH * TC_INPUT_DIM] input, reused for intermediates
    float*  smem_fw_float,      // [TC_BATCH * TC_HIDDEN_DIM] accumulator output
    const __half* smem_W1,      // [W1_SIZE] = [16, 16] row-major half
    const __half* smem_W2,      // [W2_SIZE] = [16, 16] row-major half
    const __half* smem_W3       // [W3_SIZE] = [16, 16] row-major half
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Layer 1: H1[256,16] = Input[256,16] × W1^T[16,16]
    wmma_forward_layer<TC_INPUT_DIM, TC_HIDDEN_DIM>(smem_fw_half, smem_W1, smem_fw_float);
    __syncthreads();

    // ReLU + float→half: reuse smem_fw_half as [256×16]
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        float val = fmaxf(0.0f, smem_fw_float[tid * TC_HIDDEN_DIM + i]);
        smem_fw_half[tid * TC_HIDDEN_DIM + i] = __float2half(val);
    }
    __syncthreads();

    // Layer 2: H2[256,16] = H1_relu[256,16] × W2^T[16,16]
    wmma_forward_layer<TC_HIDDEN_DIM, TC_HIDDEN_DIM>(smem_fw_half, smem_W2, smem_fw_float);
    __syncthreads();

    // ReLU + float→half
    #pragma unroll
    for (int i = 0; i < TC_HIDDEN_DIM; i++) {
        float val = fmaxf(0.0f, smem_fw_float[tid * TC_HIDDEN_DIM + i]);
        smem_fw_half[tid * TC_HIDDEN_DIM + i] = __float2half(val);
    }
    __syncthreads();

    // Layer 3: Out[256,16] = H2_relu[256,16] × W3^T[16,16]
    wmma_forward_layer<TC_HIDDEN_DIM, TC_OUTPUT_DIM>(smem_fw_half, smem_W3, smem_fw_float);
    __syncthreads();

    // Output: smem_fw_float[tid * TC_OUTPUT_DIM + ch] (raw, identity activation for residual)
}

namespace MODES {

// ============================================================================
// Flush tile-local MLP gradients to global memory (all [16×16])
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
    // W1: 16×16 = 256 floats
    for (int idx = thread_id; idx < W1_SIZE; idx += 256) {
        if (tile_dL_dW1[idx] != 0.0f)
            atomicAdd(&global_dL_dW1[idx], tile_dL_dW1[idx]);
    }
    // W2: 16×16 = 256 floats
    for (int idx = thread_id; idx < W2_SIZE; idx += 256) {
        if (tile_dL_dW2[idx] != 0.0f)
            atomicAdd(&global_dL_dW2[idx], tile_dL_dW2[idx]);
    }
    // W3: 16×16 = 256 floats
    for (int idx = thread_id; idx < W3_SIZE; idx += 256) {
        if (tile_dL_dW3[idx] != 0.0f)
            atomicAdd(&global_dL_dW3[idx], tile_dL_dW3[idx]);
    }
}

} // namespace MODES

#endif // MODE_3D_DIRECT_FUSED_CU_INCLUDED
