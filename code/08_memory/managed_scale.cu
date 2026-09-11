#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ void scale(float* values, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] *= 2.0f;
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    const int n = 10;
    float* values = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMallocManaged(&values, n * sizeof(float)), "allocate managed")) break;
        for (int i = 0; i < n; ++i) values[i] = static_cast<float>(i + 1);
        scale<<<1, 32>>>(values, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "wait before CPU access")) break;

        ok = true;
        for (int i = 0; i < n; ++i) {
            const float expected = 2.0f * (i + 1);
            std::cout << "i=" << i << " value=" << values[i] << '\n';
            if (!std::isfinite(values[i]) || values[i] != expected) ok = false;
        }
    } while (false);
    if (values != nullptr && !check(cudaFree(values), "free managed")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS managed scale: 10 elements; no explicit memcpy\n";
    return 0;
}
