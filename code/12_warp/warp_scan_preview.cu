#include <cuda_runtime.h>
#include <array>
#include <iostream>

__global__ void scan_four_groups(int* output) {
    const int lane = threadIdx.x;
    const int rank = lane % 8;
    const unsigned full = 0xffffffffu;
    int value = rank + 1;
    for (int delta = 1; delta < 8; delta *= 2) {
        const int received = __shfl_up_sync(full, value, delta, 8);
        if (rank >= delta) value += received;
    }
    output[lane] = value;
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    int device = -1;
    cudaDeviceProp prop = {};
    if (!check(cudaGetDevice(&device), "current device") ||
        !check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
    if (prop.warpSize != 32 || prop.maxThreadsPerBlock < 32 ||
        prop.maxThreadsDim[0] < 32) {
        std::cerr << "this lesson requires one complete 32-thread warp\n";
        return 1;
    }
    std::array<int, 32> output{};
    int* d_output = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_output, output.size() * sizeof(int)),
                   "allocate output") ||
            !check(cudaMemset(d_output, 0xff, output.size() * sizeof(int)),
                   "poison output")) break;
        scan_four_groups<<<1, 32>>>(d_output);
        if (!check(cudaGetLastError(), "launch") ||
            !check(cudaDeviceSynchronize(), "wait") ||
            !check(cudaMemcpy(output.data(), d_output,
                              output.size() * sizeof(int),
                              cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int lane = 0; lane < 32; ++lane) {
            const int rank = lane % 8;
            const int expected = (rank + 1) * (rank + 2) / 2;
            if (output[lane] != expected) {
                std::cerr << "scan mismatch lane=" << lane << '\n';
                ok = false;
                break;
            }
            std::cout << "lane=" << lane << " group=" << lane / 8
                      << " rank=" << rank << " inclusive=" << output[lane] << '\n';
        }
    } while (false);
    if (d_output && !check(cudaFree(d_output), "free output")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS four independent 8-lane inclusive scans\n";
    return 0;
}
