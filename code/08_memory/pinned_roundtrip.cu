#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ void add_ten(float* values, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] += 10.0f;
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    const int n = 8;
    const std::size_t bytes = n * sizeof(float);
    float* h = nullptr;
    float* d = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMallocHost(&h, bytes), "allocate pinned host")) break;
        if (!check(cudaMalloc(&d, bytes), "allocate device")) break;
        for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i);
        if (!check(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice), "H2D")) break;
        add_ten<<<1, 32>>>(d, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "wait")) break;
        if (!check(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int i = 0; i < n; ++i) {
            std::cout << "i=" << i << " value=" << h[i] << '\n';
            if (!std::isfinite(h[i]) || h[i] != static_cast<float>(i + 10)) ok = false;
        }
    } while (false);
    if (d != nullptr && !check(cudaFree(d), "free device")) ok = false;
    if (h != nullptr && !check(cudaFreeHost(h), "free pinned host")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS pinned roundtrip: 8 elements\n";
    return 0;
}
