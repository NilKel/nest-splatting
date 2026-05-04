/*
 * mma_utils.h - WMMA Tensor Core utilities for 3D_SH_res residual MLP
 *
 * Tiny residual MLP with all 16×16 tiles:
 *   Input: [hash(4) | bias(1) | pad(11)] = 16D
 *   Hidden: 16D (ReLU)
 *   Output: 16D (only first 3 = RGB residual, identity activation)
 */

#ifndef MMA_UTILS_H_INCLUDED
#define MMA_UTILS_H_INCLUDED

#include <mma.h>
using namespace nvcuda;

// ============================================================================
// WMMA tile dimensions (fixed by hardware)
// ============================================================================
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ============================================================================
// Padded MLP dimensions for WMMA alignment (all 16)
// ============================================================================
constexpr int TC_INPUT_DIM  = 16;   // [hash(hash_dim) | pad(16-hash_dim)]
constexpr int TC_HIDDEN_DIM = 16;   // single WMMA tile width
constexpr int TC_OUTPUT_DIM = 16;   // only first 3 = RGB residual
constexpr int TC_BATCH      = 256;  // pixels per tile (16x16 block)

// Original (unpadded) dimensions for masking/trimming
constexpr int ORIG_INPUT_DIM  = 5;  // [hash(4) | bias(1)]
constexpr int ORIG_OUTPUT_DIM = 3;  // RGB residual

// ============================================================================
// Weight matrix dimensions: [rows, cols] — all [16, 16]
// ============================================================================
constexpr int W1_ROWS = TC_HIDDEN_DIM;  // 16
constexpr int W1_COLS = TC_INPUT_DIM;   // 16
constexpr int W2_ROWS = TC_HIDDEN_DIM;  // 16
constexpr int W2_COLS = TC_HIDDEN_DIM;  // 16
constexpr int W3_ROWS = TC_OUTPUT_DIM;  // 16
constexpr int W3_COLS = TC_HIDDEN_DIM;  // 16

// Total weight elements per layer
constexpr int W1_SIZE = W1_ROWS * W1_COLS;  // 16*16 = 256
constexpr int W2_SIZE = W2_ROWS * W2_COLS;  // 16*16 = 256
constexpr int W3_SIZE = W3_ROWS * W3_COLS;  // 16*16 = 256

// ============================================================================
// WMMA tile counts for backward weight GEMM
// dL_dW[M,N] = dL_dz^T[M,K] @ activation[K,N], where K = TC_BATCH = 256 pixels
// All 1×1 tiles since M=N=16
// ============================================================================

// Layer 1: dL_dW1[16, 16] = dL_dz1^T[16, 256] @ input[256, 16]
constexpr int BW_L1_M_TILES = W1_ROWS / WMMA_M;   // 1
constexpr int BW_L1_N_TILES = W1_COLS / WMMA_N;    // 1
constexpr int BW_L1_TILES   = BW_L1_M_TILES * BW_L1_N_TILES;  // 1

// Layer 2: dL_dW2[16, 16] = dL_dz2^T[16, 256] @ h1[256, 16]
constexpr int BW_L2_M_TILES = W2_ROWS / WMMA_M;   // 1
constexpr int BW_L2_N_TILES = W2_COLS / WMMA_N;    // 1
constexpr int BW_L2_TILES   = BW_L2_M_TILES * BW_L2_N_TILES;  // 1

// Layer 3: dL_dW3[16, 16] = dL_dz3^T[16, 256] @ h2[256, 16]
constexpr int BW_L3_M_TILES = W3_ROWS / WMMA_M;   // 1
constexpr int BW_L3_N_TILES = W3_COLS / WMMA_N;    // 1
constexpr int BW_L3_TILES   = BW_L3_M_TILES * BW_L3_N_TILES;  // 1

// K dimension chunks (256 pixels / 16 = 16 chunks)
constexpr int BW_K_CHUNKS = TC_BATCH / WMMA_K;     // 16

// ============================================================================
// WMMA tile counts for forward batched MLP (reference, not currently used)
// ============================================================================

// Layer 1: H1[256, 16] = Input[256, 16] @ W1^T[16, 16]
constexpr int FW_L1_M_TILES = TC_BATCH / WMMA_M;       // 16
constexpr int FW_L1_N_TILES = TC_HIDDEN_DIM / WMMA_N;  // 1
constexpr int FW_L1_K_CHUNKS = TC_INPUT_DIM / WMMA_K;  // 1
constexpr int FW_L1_TILES = FW_L1_M_TILES * FW_L1_N_TILES;  // 16

// Layer 2: H2[256, 16] = H1[256, 16] @ W2^T[16, 16]
constexpr int FW_L2_M_TILES = TC_BATCH / WMMA_M;       // 16
constexpr int FW_L2_N_TILES = TC_HIDDEN_DIM / WMMA_N;  // 1
constexpr int FW_L2_K_CHUNKS = TC_HIDDEN_DIM / WMMA_K; // 1
constexpr int FW_L2_TILES = FW_L2_M_TILES * FW_L2_N_TILES;  // 16

// Layer 3: Out[256, 16] = H2[256, 16] @ W3^T[16, 16]
constexpr int FW_L3_M_TILES = TC_BATCH / WMMA_M;       // 16
constexpr int FW_L3_N_TILES = TC_OUTPUT_DIM / WMMA_N;  // 1
constexpr int FW_L3_K_CHUNKS = TC_HIDDEN_DIM / WMMA_K; // 1
constexpr int FW_L3_TILES = FW_L3_M_TILES * FW_L3_N_TILES;  // 16

// Total warps per block: 256 threads / 32 threads-per-warp = 8 warps
constexpr int WARPS_PER_BLOCK = TC_BATCH / 32;  // 8

// ============================================================================
// Fragment type aliases
// ============================================================================

// Backward GEMM: C[M,N] = A^T[M,K] @ B[K,N]
using bw_frag_a = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>;
using bw_frag_b = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
using frag_acc  = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

// Forward GEMM: H[M,N] = Input[M,K] @ W^T[K,N]
using fw_frag_a = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
using fw_frag_b = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>;

// ============================================================================
// Shared memory size constants
// ============================================================================

// Backward GEMM: largest layer is L1 = 256*(16+16)*sizeof(half) = 16,384 bytes
constexpr int TC_COLLABORATIVE_SMEM_SIZE = TC_BATCH * (TC_INPUT_DIM + TC_HIDDEN_DIM) * sizeof(half);

// Forward WMMA: half input buffer + float accumulator output
// Half region: 256*16*2 = 8,192 bytes
// Float region: 256*16*4 = 16,384 bytes
// Total: 24,576 bytes
constexpr int TC_FORWARD_SMEM_HALF = TC_BATCH * TC_INPUT_DIM * sizeof(half);
constexpr int TC_FORWARD_SMEM_FLOAT = TC_BATCH * TC_HIDDEN_DIM * sizeof(float);
constexpr int TC_FORWARD_SMEM_SIZE = TC_FORWARD_SMEM_HALF + TC_FORWARD_SMEM_FLOAT;

#endif // MMA_UTILS_H_INCLUDED
