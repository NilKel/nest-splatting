/**
 * Exact replication of rasterizer's j-loop pattern for MLP backward
 *
 * This test calls collaborative_mlp_backward_all multiple times (simulating j-loop),
 * where only the first j has participating threads.
 *
 * Build: nvcc -o mlp_jloop_exact mlp_jloop_exact.cu -arch=sm_86
 * Run: ./mlp_jloop_exact
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256
#define SUB_TILE_SIZE 128
#define IN_DIM 40
#define HIDDEN_DIM 32
#define OUT_DIM 3

// ============================================================================
// Forward pass (identical to standalone test)
// ============================================================================
__device__ void mlp_forward(
    const float* input,
    float* h1_pre, float* h1_post,
    float* h2_pre, float* h2_post,
    float* output,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* W3, const float* b3
) {
    for (int h = 0; h < 32; h++) {
        float acc = b1[h];
        for (int i = 0; i < 40; i++) acc += W1[h * 40 + i] * input[i];
        h1_pre[h] = acc;
        h1_post[h] = fmaxf(0.0f, acc);
    }
    for (int h = 0; h < 32; h++) {
        float acc = b2[h];
        for (int i = 0; i < 32; i++) acc += W2[h * 32 + i] * h1_post[i];
        h2_pre[h] = acc;
        h2_post[h] = fmaxf(0.0f, acc);
    }
    for (int o = 0; o < 3; o++) {
        float acc = b3[o];
        for (int h = 0; h < 32; h++) acc += W3[o * 32 + h] * h2_post[h];
        output[o] = 1.0f / (1.0f + expf(-acc));
    }
}

// ============================================================================
// Compute dL_dz for all layers
// ============================================================================
__device__ void compute_dL_dz_all(
    const float* h1_pre, const float* h2_pre, const float* h2_post,
    const float* output, const float* dL_dout,
    float* dL_dz3, float* dL_dz2, float* dL_dz1,
    const float* W2, const float* W3
) {
    // dL_dz3 = dL_dout * sigmoid'
    for (int o = 0; o < 3; o++) {
        float sig = output[o];
        dL_dz3[o] = dL_dout[o] * sig * (1.0f - sig);
    }

    // dL_dh2 = dL_dz3 @ W3
    float dL_dh2[32] = {0};
    for (int o = 0; o < 3; o++) {
        for (int h = 0; h < 32; h++) {
            dL_dh2[h] += dL_dz3[o] * W3[o * 32 + h];
        }
    }

    // dL_dz2 = dL_dh2 * relu'
    for (int h = 0; h < 32; h++) {
        dL_dz2[h] = (h2_pre[h] > 0) ? dL_dh2[h] : 0;
    }

    // dL_dh1 = dL_dz2 @ W2
    float dL_dh1[32] = {0};
    for (int h = 0; h < 32; h++) {
        for (int i = 0; i < 32; i++) {
            dL_dh1[i] += dL_dz2[h] * W2[h * 32 + i];
        }
    }

    // dL_dz1 = dL_dh1 * relu'
    for (int h = 0; h < 32; h++) {
        dL_dz1[h] = (h1_pre[h] > 0) ? dL_dh1[h] : 0;
    }
}

// ============================================================================
// Collaborative MLP backward - EXACT copy from mode_3d_direct_fused.cu
// ============================================================================
__device__ void collaborative_mlp_backward_all(
    const float* my_input, const float* my_h1, const float* my_h2,
    const float* my_dL_dz1, const float* my_dL_dz2, const float* my_dL_dz3,
    float* tile_dL_dW1, float* tile_dL_db1,
    float* tile_dL_dW2, float* tile_dL_db2,
    float* tile_dL_dW3, float* tile_dL_db3,
    float* smem_buffer
) {
    const int tid = threadIdx.x;

    // Layer 3
    {
        float* smem_dL_dz3 = smem_buffer;
        float* smem_h2 = smem_buffer + SUB_TILE_SIZE * 3;

        for (int batch = 0; batch < 2; batch++) {
            const int batch_start = batch * SUB_TILE_SIZE;
            const bool in_batch = (tid >= batch_start) && (tid < batch_start + SUB_TILE_SIZE);
            const int local_idx = tid - batch_start;

            if (in_batch) {
                smem_dL_dz3[local_idx * 3 + 0] = my_dL_dz3[0];
                smem_dL_dz3[local_idx * 3 + 1] = my_dL_dz3[1];
                smem_dL_dz3[local_idx * 3 + 2] = my_dL_dz3[2];
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

    // Layer 2
    {
        float* smem_dL_dz2 = smem_buffer;
        float* smem_h1 = smem_buffer + SUB_TILE_SIZE * 32;

        for (int batch = 0; batch < 2; batch++) {
            const int batch_start = batch * SUB_TILE_SIZE;
            const bool in_batch = (tid >= batch_start) && (tid < batch_start + SUB_TILE_SIZE);
            const int local_idx = tid - batch_start;

            if (in_batch) {
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

    // Layer 1
    {
        float* smem_dL_dz1 = smem_buffer;
        float* smem_input = smem_buffer + SUB_TILE_SIZE * 32;

        for (int batch = 0; batch < 2; batch++) {
            const int batch_start = batch * SUB_TILE_SIZE;
            const bool in_batch = (tid >= batch_start) && (tid < batch_start + SUB_TILE_SIZE);
            const int local_idx = tid - batch_start;

            if (in_batch) {
                for (int i = 0; i < 32; i++) {
                    smem_dL_dz1[local_idx * 32 + i] = my_dL_dz1[i];
                }
                for (int i = 0; i < 40; i++) {
                    smem_input[local_idx * 40 + i] = my_input[i];
                }
            }
            __syncthreads();

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

// ============================================================================
// Test kernel: Single call (should PASS)
// ============================================================================
__global__ void test_single_call(
    const float* inputs,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* W3, const float* b3,
    float* dL_db1, float* dL_db2, float* dL_db3
) {
    __shared__ float tile_dL_dW1[1280], tile_dL_db1[32];
    __shared__ float tile_dL_dW2[1024], tile_dL_db2[32];
    __shared__ float tile_dL_dW3[96], tile_dL_db3[3];

    int tid = threadIdx.x;

    // Zero tiles
    for (int i = tid; i < 1280; i += 256) tile_dL_dW1[i] = 0;
    for (int i = tid; i < 1024; i += 256) tile_dL_dW2[i] = 0;
    for (int i = tid; i < 96; i += 256) tile_dL_dW3[i] = 0;
    if (tid < 32) { tile_dL_db1[tid] = 0; tile_dL_db2[tid] = 0; }
    if (tid < 3) tile_dL_db3[tid] = 0;
    __syncthreads();

    // All threads participate
    float my_input[40], my_h1_pre[32], my_h1_post[32], my_h2_pre[32], my_h2_post[32];
    float my_output[3], my_dL_dout[3] = {1, 1, 1};
    float my_dL_dz1[32], my_dL_dz2[32], my_dL_dz3[3];

    for (int i = 0; i < 40; i++) my_input[i] = inputs[tid * 40 + i];
    mlp_forward(my_input, my_h1_pre, my_h1_post, my_h2_pre, my_h2_post, my_output, W1, b1, W2, b2, W3, b3);
    compute_dL_dz_all(my_h1_pre, my_h2_pre, my_h2_post, my_output, my_dL_dout, my_dL_dz3, my_dL_dz2, my_dL_dz1, W2, W3);

    __syncthreads();

    extern __shared__ float smem[];
    collaborative_mlp_backward_all(my_input, my_h1_post, my_h2_post,
                                   my_dL_dz1, my_dL_dz2, my_dL_dz3,
                                   tile_dL_dW1, tile_dL_db1, tile_dL_dW2, tile_dL_db2, tile_dL_dW3, tile_dL_db3, smem);

    __syncthreads();

    // Write biases only (for test)
    if (tid < 32) { dL_db1[tid] = tile_dL_db1[tid]; dL_db2[tid] = tile_dL_db2[tid]; }
    if (tid < 3) dL_db3[tid] = tile_dL_db3[tid];
}

// ============================================================================
// Test kernel: J-loop with multiple calls (replicates rasterizer)
// ============================================================================
__global__ void test_jloop(
    const float* inputs,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* W3, const float* b3,
    float* dL_db1, float* dL_db2, float* dL_db3,
    int num_j  // Number of j-loop iterations
) {
    __shared__ float tile_dL_dW1[1280], tile_dL_db1[32];
    __shared__ float tile_dL_dW2[1024], tile_dL_db2[32];
    __shared__ float tile_dL_dW3[96], tile_dL_db3[3];

    int tid = threadIdx.x;

    // Zero tiles ONCE at start (not per j!)
    for (int i = tid; i < 1280; i += 256) tile_dL_dW1[i] = 0;
    for (int i = tid; i < 1024; i += 256) tile_dL_dW2[i] = 0;
    for (int i = tid; i < 96; i += 256) tile_dL_dW3[i] = 0;
    if (tid < 32) { tile_dL_db1[tid] = 0; tile_dL_db2[tid] = 0; }
    if (tid < 3) tile_dL_db3[tid] = 0;
    __syncthreads();

    // ======== J-LOOP (multiple calls to collaborative_mlp_backward_all) ========
    for (int j = 0; j < num_j; j++) {
        // Only participate on j=0 (simulates sparse participation in real rasterizer)
        bool participates = (j == 0);

        // Initialize arrays INSIDE j-loop (matching rasterizer exactly)
        float my_input[40] = {0};
        float my_h1_pre[32] = {0}, my_h1_post[32] = {0};
        float my_h2_pre[32] = {0}, my_h2_post[32] = {0};
        float my_output[3] = {0};
        float my_dL_dout[3] = {0};
        float my_dL_dz1[32] = {0}, my_dL_dz2[32] = {0}, my_dL_dz3[3] = {0};

        if (participates) {
            for (int i = 0; i < 40; i++) my_input[i] = inputs[tid * 40 + i];
            mlp_forward(my_input, my_h1_pre, my_h1_post, my_h2_pre, my_h2_post, my_output, W1, b1, W2, b2, W3, b3);
            my_dL_dout[0] = my_dL_dout[1] = my_dL_dout[2] = 1.0f;
            compute_dL_dz_all(my_h1_pre, my_h2_pre, my_h2_post, my_output, my_dL_dout, my_dL_dz3, my_dL_dz2, my_dL_dz1, W2, W3);
        }

        __syncthreads();

        // Call collaborative GEMM EVERY j (matching rasterizer)
        extern __shared__ float smem[];
        collaborative_mlp_backward_all(my_input, my_h1_post, my_h2_post,
                                       my_dL_dz1, my_dL_dz2, my_dL_dz3,
                                       tile_dL_dW1, tile_dL_db1, tile_dL_dW2, tile_dL_db2, tile_dL_dW3, tile_dL_db3, smem);
    }

    __syncthreads();

    if (tid < 32) { dL_db1[tid] = tile_dL_db1[tid]; dL_db2[tid] = tile_dL_db2[tid]; }
    if (tid < 3) dL_db3[tid] = tile_dL_db3[tid];
}

// ============================================================================
// Setup unit weights
// ============================================================================
void set_unit_weights(float* W1, float* W2, float* W3) {
    memset(W1, 0, 32 * 40 * sizeof(float));
    for (int i = 0; i < 32; i++) W1[i * 40 + i] = 1.0f;

    memset(W2, 0, 32 * 32 * sizeof(float));
    for (int i = 0; i < 32; i++) W2[i * 32 + i] = 1.0f;

    memset(W3, 0, 3 * 32 * sizeof(float));
    for (int i = 0; i < 3; i++) W3[i * 32 + i] = 1.0f;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("=== J-Loop Exact Replication Test ===\n\n");

    // Host memory
    float *h_W1 = (float*)calloc(32 * 40, sizeof(float));
    float *h_W2 = (float*)calloc(32 * 32, sizeof(float));
    float *h_W3 = (float*)calloc(3 * 32, sizeof(float));
    float *h_b1 = (float*)calloc(32, sizeof(float));
    float *h_b2 = (float*)calloc(32, sizeof(float));
    float *h_b3 = (float*)calloc(3, sizeof(float));
    float *h_inputs = (float*)malloc(256 * 40 * sizeof(float));
    float *h_db1 = (float*)malloc(32 * sizeof(float));
    float *h_db2 = (float*)malloc(32 * sizeof(float));
    float *h_db3 = (float*)malloc(3 * sizeof(float));

    set_unit_weights(h_W1, h_W2, h_W3);
    for (int i = 0; i < 256 * 40; i++) h_inputs[i] = 1.0f;

    // Device memory
    float *d_W1, *d_W2, *d_W3, *d_b1, *d_b2, *d_b3, *d_inputs, *d_db1, *d_db2, *d_db3;
    cudaMalloc(&d_W1, 32 * 40 * sizeof(float));
    cudaMalloc(&d_W2, 32 * 32 * sizeof(float));
    cudaMalloc(&d_W3, 3 * 32 * sizeof(float));
    cudaMalloc(&d_b1, 32 * sizeof(float));
    cudaMalloc(&d_b2, 32 * sizeof(float));
    cudaMalloc(&d_b3, 3 * sizeof(float));
    cudaMalloc(&d_inputs, 256 * 40 * sizeof(float));
    cudaMalloc(&d_db1, 32 * sizeof(float));
    cudaMalloc(&d_db2, 32 * sizeof(float));
    cudaMalloc(&d_db3, 3 * sizeof(float));

    cudaMemcpy(d_W1, h_W1, 32 * 40 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, 32 * 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, h_W3, 3 * 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, h_inputs, 256 * 40 * sizeof(float), cudaMemcpyHostToDevice);

    // ======== Test 1: Single call ========
    printf("--- Test 1: Single call (all threads participate) ---\n");
    cudaFuncSetAttribute(test_single_call, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    test_single_call<<<1, 256, 65536>>>(d_inputs, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_db1, d_db2, d_db3);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(h_db2, d_db2, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("dL_db2 nonzeros: ");
    int nz = 0;
    for (int i = 0; i < 32; i++) {
        if (fabsf(h_db2[i]) > 1e-10f) { printf("%d ", i); nz++; }
    }
    printf("\nTotal: %d/32 (expected: 3 at 0,1,2)\n\n", nz);

    // ======== Test 2: J-loop with 4 iterations ========
    printf("--- Test 2: J-loop with 4 iterations (only j=0 participates) ---\n");
    cudaMemset(d_db1, 0, 32 * sizeof(float));
    cudaMemset(d_db2, 0, 32 * sizeof(float));
    cudaMemset(d_db3, 0, 3 * sizeof(float));

    cudaFuncSetAttribute(test_jloop, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    test_jloop<<<1, 256, 65536>>>(d_inputs, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_db1, d_db2, d_db3, 4);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(h_db2, d_db2, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("dL_db2 nonzeros: ");
    nz = 0;
    for (int i = 0; i < 32; i++) {
        if (fabsf(h_db2[i]) > 1e-10f) { printf("%d ", i); nz++; }
    }
    printf("\nTotal: %d/32 (expected: 3 at 0,1,2 if correct)\n\n", nz);

    // ======== Test 3: J-loop with 16 iterations ========
    printf("--- Test 3: J-loop with 16 iterations (only j=0 participates) ---\n");
    cudaMemset(d_db1, 0, 32 * sizeof(float));
    cudaMemset(d_db2, 0, 32 * sizeof(float));
    cudaMemset(d_db3, 0, 3 * sizeof(float));

    test_jloop<<<1, 256, 65536>>>(d_inputs, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_db1, d_db2, d_db3, 16);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(h_db2, d_db2, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("dL_db2 nonzeros: ");
    nz = 0;
    for (int i = 0; i < 32; i++) {
        if (fabsf(h_db2[i]) > 1e-10f) { printf("%d ", i); nz++; }
    }
    printf("\nTotal: %d/32 (expected: 3 at 0,1,2 if correct)\n\n", nz);

    // Print values to see if they're just accumulated (x4 or x16)
    printf("dL_db2 values [0:5]: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_db2[i]);
    printf("\n");

    // Cleanup
    free(h_W1); free(h_W2); free(h_W3);
    free(h_b1); free(h_b2); free(h_b3);
    free(h_inputs); free(h_db1); free(h_db2); free(h_db3);
    cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_W3);
    cudaFree(d_b1); cudaFree(d_b2); cudaFree(d_b3);
    cudaFree(d_inputs); cudaFree(d_db1); cudaFree(d_db2); cudaFree(d_db3);

    return 0;
}
