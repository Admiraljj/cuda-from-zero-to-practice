#include <cuda_runtime.h>
#include <iostream>
#include <string>

__global__ void broken(int* data, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = i;  // Deliberate bug: missing i < n.
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main(int argc, char** argv) {
    if (argc != 2 || std::string(argv[1]) != "--run-broken") {
        std::cerr << "Refusing intentional out-of-bounds demo without --run-broken\n";
        return 2;
    }
    std::cerr << "WARNING: intentional out-of-bounds demo; use memcheck\n";
    const int n = 33;
    int* data = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&data, n * sizeof(int)), "allocate")) break;
        broken<<<2, 32>>>(data, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "execute")) break;
        ok = true;
    } while (false);
    if (data != nullptr && !check(cudaFree(data), "free")) ok = false;
    if (!ok) return 1;
    std::cout << "BAD CASE completed; inspect sanitizer diagnostics\n";
    return 0;
}
