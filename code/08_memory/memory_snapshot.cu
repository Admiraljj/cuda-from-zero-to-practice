#include <cuda_runtime.h>
#include <iostream>

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool snapshot(const char* stage) {
    std::size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "memory snapshot")) return false;
    std::cout << stage << " free_bytes=" << free_bytes
              << " total_bytes=" << total_bytes << '\n';
    return true;
}

int main() {
    const std::size_t requested = 4 * 1024 * 1024;
    unsigned char* device_data = nullptr;
    if (!snapshot("before")) return 1;
    std::cout << "requested_bytes=" << requested << '\n';
    if (!check(cudaMalloc(&device_data, requested), "allocate")) return 1;
    bool ok = snapshot("allocated");
    if (!check(cudaFree(device_data), "free")) return 1;
    device_data = nullptr;
    if (!snapshot("released")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS snapshots; differences are not asserted\n";
    return 0;
}
