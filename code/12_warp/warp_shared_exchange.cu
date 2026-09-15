#include <cuda_runtime.h>
#include <array>
#include <iostream>

struct Exchange {
    int first_neighbor;
    int second_neighbor;
};

__global__ void shared_exchange(Exchange* output) {
    __shared__ int shared[32];
    const int lane = threadIdx.x;
    const unsigned full = 0xffffffffu;

    shared[lane] = 100 + lane;
    __syncwarp(full);
    const int first = shared[lane ^ 1];

    __syncwarp(full);
    shared[lane] = first + 1000;
    __syncwarp(full);
    const int second = shared[lane ^ 1];

    output[lane] = {first, second};
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
    std::array<Exchange, 32> output{};
    Exchange* d_output = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_output, output.size() * sizeof(Exchange)),
                   "allocate output") ||
            !check(cudaMemset(d_output, 0xff, output.size() * sizeof(Exchange)),
                   "poison output")) break;
        shared_exchange<<<1, 32>>>(d_output);
        if (!check(cudaGetLastError(), "launch") ||
            !check(cudaDeviceSynchronize(), "wait") ||
            !check(cudaMemcpy(output.data(), d_output,
                              output.size() * sizeof(Exchange),
                              cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int lane = 0; lane < 32; ++lane) {
            const int expected_first = 100 + (lane ^ 1);
            const int expected_second = 1100 + lane;
            if (output[lane].first_neighbor != expected_first ||
                output[lane].second_neighbor != expected_second) {
                std::cerr << "exchange mismatch lane=" << lane << '\n';
                ok = false;
                break;
            }
            std::cout << "lane=" << lane
                      << " first_neighbor=" << output[lane].first_neighbor
                      << " second_neighbor=" << output[lane].second_neighbor << '\n';
        }
    } while (false);
    if (d_output && !check(cudaFree(d_output), "free output")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS warp shared exchange; write/read/reuse order checked\n";
    return 0;
}
