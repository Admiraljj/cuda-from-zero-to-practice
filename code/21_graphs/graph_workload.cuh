#ifndef CUDA_BOOK_GRAPH_WORKLOAD_CUH
#define CUDA_BOOK_GRAPH_WORKLOAD_CUH
#include "../common/cuda_check.h"
#include "graph_reference.h"

namespace graph_example {
__global__ void scale_input(const float* input, float* output, float alpha, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = alpha * input[i];
}
__global__ void add_bias(float* output, float beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] += beta;
}
__global__ void relu(float* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = fmaxf(output[i], 0.0F);
}
__global__ void fused(const float* input, float* output, float alpha, float beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = fmaxf(alpha * input[i] + beta, 0.0F);
}
inline void submit_three(const float* input, float* output, int n,
                         float alpha, float beta, cudaStream_t stream) {
    const int threads = 256, blocks = (n + threads - 1) / threads;
    scale_input<<<blocks, threads, 0, stream>>>(input, output, alpha, n);
    CUDA_CHECK(cudaGetLastError());
    add_bias<<<blocks, threads, 0, stream>>>(output, beta, n);
    CUDA_CHECK(cudaGetLastError());
    relu<<<blocks, threads, 0, stream>>>(output, n);
    CUDA_CHECK(cudaGetLastError());
}
}
#endif
