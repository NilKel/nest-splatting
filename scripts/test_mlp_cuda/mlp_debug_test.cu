/**
 * Debug MLP test - trace through forward/backward
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define IN_DIM 40
#define HIDDEN_DIM 32
#define OUT_DIM 3
#define BLOCK_SIZE 256

#define W1_OFFSET 0
#define B1_OFFSET (IN_DIM * HIDDEN_DIM)
#define W2_OFFSET (B1_OFFSET + HIDDEN_DIM)
#define B2_OFFSET (W2_OFFSET + HIDDEN_DIM * HIDDEN_DIM)
#define W3_OFFSET (B2_OFFSET + HIDDEN_DIM)
#define B3_OFFSET (W3_OFFSET + HIDDEN_DIM * OUT_DIM)

__constant__ float const_mlp_weights[40*32 + 32 + 32*32 + 32 + 32*3 + 3];

__device__ void mlp_forward_debug(
    const float* input,
    float* output,
    float* h1_pre,
    float* h1_post,
    float* h2_pre,
    float* h2_post,
    const float* weights,
    int tid
) {
    const float* W1 = weights + W1_OFFSET;
    const float* b1 = weights + B1_OFFSET;
    const float* W2 = weights + W2_OFFSET;
    const float* b2 = weights + B2_OFFSET;
    const float* W3 = weights + W3_OFFSET;
    const float* b3 = weights + B3_OFFSET;

    // Layer 1
    for (int h = 0; h < 32; h++) {
        float acc = b1[h];
        for (int i = 0; i < 40; i++) {
            acc += input[i] * W1[h * 40 + i];
        }
        h1_pre[h] = acc;
        h1_post[h] = fmaxf(0.0f, acc);
    }

    // Debug: print thread 0's values
    if (tid == 0) {
        printf("Thread 0 forward:\n");
        printf("  input[0:3] = %.3f, %.3f, %.3f\n", input[0], input[1], input[2]);
        printf("  h1_pre[0:3] = %.3f, %.3f, %.3f\n", h1_pre[0], h1_pre[1], h1_pre[2]);
        printf("  h1_post[0:3] = %.3f, %.3f, %.3f\n", h1_post[0], h1_post[1], h1_post[2]);
    }

    // Layer 2
    for (int h = 0; h < 32; h++) {
        float acc = b2[h];
        for (int i = 0; i < 32; i++) {
            acc += h1_post[i] * W2[h * 32 + i];
        }
        h2_pre[h] = acc;
        h2_post[h] = fmaxf(0.0f, acc);
    }

    if (tid == 0) {
        printf("  h2_pre[0:3] = %.3f, %.3f, %.3f\n", h2_pre[0], h2_pre[1], h2_pre[2]);
        printf("  h2_post[0:3] = %.3f, %.3f, %.3f\n", h2_post[0], h2_post[1], h2_post[2]);
    }

    // Layer 3
    for (int o = 0; o < 3; o++) {
        float acc = b3[o];
        for (int i = 0; i < 32; i++) {
            acc += h2_post[i] * W3[o * 32 + i];
        }
        output[o] = 1.0f / (1.0f + expf(-acc));
    }

    if (tid == 0) {
        printf("  output = %.3f, %.3f, %.3f\n", output[0], output[1], output[2]);
    }
}

__device__ void compute_dL_dz_debug(
    const float* h1_pre,
    const float* h1_post,
    const float* h2_pre,
    const float* h2_post,
    const float* output,
    const float* dL_dout,
    float* dL_dz3,
    float* dL_dz2,
    float* dL_dz1,
    const float* weights,
    int tid
) {
    const float* W2 = weights + W2_OFFSET;
    const float* W3 = weights + W3_OFFSET;

    // dL_dz3
    for (int o = 0; o < 3; o++) {
        dL_dz3[o] = dL_dout[o] * output[o] * (1.0f - output[o]);
    }

    if (tid == 0) {
        printf("Thread 0 backward:\n");
        printf("  dL_dout = %.3f, %.3f, %.3f\n", dL_dout[0], dL_dout[1], dL_dout[2]);
        printf("  dL_dz3 = %.6f, %.6f, %.6f\n", dL_dz3[0], dL_dz3[1], dL_dz3[2]);
    }

    // dL_dz2
    for (int h = 0; h < 32; h++) {
        float dL_dh2 = 0.0f;
        for (int o = 0; o < 3; o++) {
            dL_dh2 += dL_dz3[o] * W3[o * 32 + h];
        }
        dL_dz2[h] = (h2_pre[h] > 0.0f) ? dL_dh2 : 0.0f;
    }

    if (tid == 0) {
        printf("  dL_dz2[0:5] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
               dL_dz2[0], dL_dz2[1], dL_dz2[2], dL_dz2[3], dL_dz2[4]);
    }

    // dL_dz1
    for (int h = 0; h < 32; h++) {
        float dL_dh1 = 0.0f;
        for (int i = 0; i < 32; i++) {
            dL_dh1 += dL_dz2[i] * W2[i * 32 + h];
        }
        dL_dz1[h] = (h1_pre[h] > 0.0f) ? dL_dh1 : 0.0f;
    }

    if (tid == 0) {
        printf("  dL_dz1[0:5] = %.6f, %.6f, %.6f, %.6f, %.6f\n",
               dL_dz1[0], dL_dz1[1], dL_dz1[2], dL_dz1[3], dL_dz1[4]);
    }
}

__global__ void test_mlp_debug(
    const float* __restrict__ input,
    const float* __restrict__ dL_dpixel,
    float* __restrict__ dL_db2
) {
    const int tid = threadIdx.x;

    __shared__ float tile_dL_db2[32];

    // Zero tile
    if (tid < 32) tile_dL_db2[tid] = 0.0f;
    __syncthreads();

    // Per-thread arrays
    float my_input[40], my_h1_pre[32], my_h1_post[32];
    float my_h2_pre[32], my_h2_post[32], my_output[3];
    float my_dL_dout[3], my_dL_dz3[3], my_dL_dz2[32], my_dL_dz1[32];

    // Load input
    for (int i = 0; i < 40; i++) my_input[i] = input[tid * 40 + i];

    // Forward
    mlp_forward_debug(my_input, my_output, my_h1_pre, my_h1_post, my_h2_pre, my_h2_post,
                      const_mlp_weights, tid);

    // dL_dout
    for (int c = 0; c < 3; c++) my_dL_dout[c] = dL_dpixel[tid * 3 + c];

    // Backward
    compute_dL_dz_debug(my_h1_pre, my_h1_post, my_h2_pre, my_h2_post,
                        my_output, my_dL_dout, my_dL_dz3, my_dL_dz2, my_dL_dz1,
                        const_mlp_weights, tid);

    __syncthreads();

    // Accumulate dL_db2 using atomicAdd
    for (int h = 0; h < 32; h++) {
        atomicAdd(&tile_dL_db2[h], my_dL_dz2[h]);
    }

    __syncthreads();

    // Write to global
    if (tid < 32) {
        dL_db2[tid] = tile_dL_db2[tid];
        if (tid < 5) {
            printf("Final dL_db2[%d] = %.6f\n", tid, tile_dL_db2[tid]);
        }
    }
}

void setup_unit_weights(float* weights) {
    memset(weights, 0, (40*32 + 32 + 32*32 + 32 + 32*3 + 3) * sizeof(float));

    float* W1 = weights + W1_OFFSET;
    for (int i = 0; i < 32; i++) W1[i * 40 + i] = 1.0f;

    float* W2 = weights + W2_OFFSET;
    for (int i = 0; i < 32; i++) W2[i * 32 + i] = 1.0f;

    float* W3 = weights + W3_OFFSET;
    for (int i = 0; i < 3; i++) W3[i * 32 + i] = 1.0f;

    printf("Weight setup:\n");
    printf("  W1[0,0] = %.1f, W1[1,1] = %.1f, W1[2,2] = %.1f\n",
           W1[0*40+0], W1[1*40+1], W1[2*40+2]);
    printf("  W2[0,0] = %.1f, W2[1,1] = %.1f, W2[2,2] = %.1f\n",
           W2[0*32+0], W2[1*32+1], W2[2*32+2]);
    printf("  W3[0,0] = %.1f, W3[1,1] = %.1f, W3[2,2] = %.1f\n",
           W3[0*32+0], W3[1*32+1], W3[2*32+2]);
}

int main() {
    printf("=== MLP Debug Test ===\n\n");

    float h_weights[40*32 + 32 + 32*32 + 32 + 32*3 + 3];
    setup_unit_weights(h_weights);
    cudaMemcpyToSymbol(const_mlp_weights, h_weights, sizeof(h_weights));

    float h_input[256 * 40];
    for (int i = 0; i < 256 * 40; i++) h_input[i] = 1.0f;

    float h_dL_dpixel[256 * 3];
    for (int i = 0; i < 256 * 3; i++) h_dL_dpixel[i] = 1.0f;

    float *d_input, *d_dL_dpixel, *d_dL_db2;
    cudaMalloc(&d_input, 256 * 40 * sizeof(float));
    cudaMalloc(&d_dL_dpixel, 256 * 3 * sizeof(float));
    cudaMalloc(&d_dL_db2, 32 * sizeof(float));

    cudaMemcpy(d_input, h_input, 256 * 40 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dL_dpixel, h_dL_dpixel, 256 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    printf("\nLaunching kernel...\n\n");
    test_mlp_debug<<<1, 256>>>(d_input, d_dL_dpixel, d_dL_db2);
    cudaDeviceSynchronize();

    printf("\n");
    float h_db2[32];
    cudaMemcpy(h_db2, d_dL_db2, 32 * sizeof(float), cudaMemcpyDeviceToHost);

    int nz_count = 0;
    printf("dL_db2 nonzeros: ");
    for (int i = 0; i < 32; i++) {
        if (fabsf(h_db2[i]) > 1e-10f) {
            printf("%d(%.4f) ", i, h_db2[i]);
            nz_count++;
        }
    }
    printf("\nTotal: %d/32\n", nz_count);

    cudaFree(d_input);
    cudaFree(d_dL_dpixel);
    cudaFree(d_dL_db2);

    return 0;
}
