#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void stencil_naive(const float* input, float* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float left = i > 0 ? input[i - 1] : 0.0f;
        const float center = input[i];
        const float right = i + 1 < n ? input[i + 1] : 0.0f;
        output[i] = left + center + right;
    }
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    constexpr int n = 10, threads = 4;
    const int blocks = (n + threads - 1) / threads;
    const std::size_t bytes = n * sizeof(float);
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i + 1);
    const float expected[n] = {3, 6, 9, 12, 15, 18, 21, 24, 27, 19};
    float *d_input = nullptr, *d_output = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input")) break;
        if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
        if (!check(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;
        if (!check(cudaMemset(d_output, 0xff, bytes), "poison output")) break;
        stencil_naive<<<blocks, threads>>>(d_input, d_output, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "wait")) break;
        if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int i = 0; i < n; ++i) {
            std::cout << "i=" << i << " sum=" << output[i] << '\n';
            if (!std::isfinite(output[i]) || output[i] != expected[i]) ok = false;
        }
    } while (false);
    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS naive stencil: 10 elements; input and output are separate\n";
    return 0;
}
