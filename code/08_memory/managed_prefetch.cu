#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#if CUDART_VERSION < 12000
#error This teaching example targets CUDA Toolkit 12 or later
#endif

cudaError_t prefetch(float* values, std::size_t bytes, int destination) {
#if CUDART_VERSION >= 13000
    cudaMemLocation location = {};
    if (destination == cudaCpuDeviceId) {
        location.type = cudaMemLocationTypeHost;
    } else {
        location.type = cudaMemLocationTypeDevice;
        location.id = destination;
    }
    return cudaMemPrefetchAsync(values, bytes, location, 0);
#else
    return cudaMemPrefetchAsync(values, bytes, destination);
#endif
}

__global__ void triple(float* values, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] *= 3.0f;
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    int device = -1, managed = 0, concurrent = 0;
    if (!check(cudaGetDevice(&device), "current device")) return 1;
    if (!check(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, device),
               "managed support")) return 1;
    if (!check(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device),
               "prefetch support")) return 1;
    if (!managed || !concurrent) {
        std::cout << "SKIP prefetch: managed=" << managed
                  << " concurrent=" << concurrent << '\n';
        return 77;
    }
    const int n = 10;
    const std::size_t bytes = n * sizeof(float);
    float* values = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMallocManaged(&values, bytes), "allocate")) break;
        for (int i = 0; i < n; ++i) values[i] = static_cast<float>(i + 1);
        if (!check(prefetch(values, bytes, device), "prefetch to GPU")) break;
        triple<<<1, 32>>>(values, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(prefetch(values, bytes, cudaCpuDeviceId), "prefetch to CPU")) break;
        if (!check(cudaDeviceSynchronize(), "wait before CPU read")) break;
        ok = true;
        for (int i = 0; i < n; ++i) {
            std::cout << "i=" << i << " value=" << values[i] << '\n';
            if (!std::isfinite(values[i]) || values[i] != 3.0f * (i + 1)) ok = false;
        }
    } while (false);
    if (values != nullptr && !check(cudaFree(values), "free")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS managed prefetch: 10 elements\n";
    return 0;
}
