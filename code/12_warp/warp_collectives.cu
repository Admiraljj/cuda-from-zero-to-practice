#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <array>
#include <iostream>

namespace cg = cooperative_groups;

__device__ int raw_warp_sum(int value) {
    const unsigned full = 0xffffffffu;
    const int lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        const int received = __shfl_down_sync(full, value, offset);
        if (lane < offset) value += received;
    }
    return value;
}

__device__ int group_warp_sum(cg::thread_block_tile<32> tile, int value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        const int received = tile.shfl_down(value, offset);
        if (tile.thread_rank() < static_cast<unsigned>(offset)) {
            value += received;
        }
    }
    return value;
}

__global__ void compare_collectives(int* results) {
    using WarpReduce = cub::WarpReduce<int>;
    __shared__ typename WarpReduce::TempStorage cub_storage;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
    const int value = static_cast<int>(tile.thread_rank()) + 1;

    const int raw = raw_warp_sum(value);
    const int grouped = group_warp_sum(tile, value);
    const int cub_sum = WarpReduce(cub_storage).Sum(value);

    if (tile.thread_rank() == 0) {
        results[0] = raw;
        results[1] = grouped;
        results[2] = cub_sum;
    }
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
    std::array<int, 3> results{};
    int* d_results = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_results, results.size() * sizeof(int)),
                   "allocate results") ||
            !check(cudaMemset(d_results, 0xff, results.size() * sizeof(int)),
                   "poison results")) break;
        compare_collectives<<<1, 32>>>(d_results);
        if (!check(cudaGetLastError(), "launch") ||
            !check(cudaDeviceSynchronize(), "wait") ||
            !check(cudaMemcpy(results.data(), d_results,
                              results.size() * sizeof(int),
                              cudaMemcpyDeviceToHost), "D2H")) break;
        constexpr int expected = 528;
        ok = results[0] == expected && results[1] == expected &&
             results[2] == expected;
        if (!ok) {
            std::cerr << "collective comparison mismatch\n";
            break;
        }
        std::cout << "raw=" << results[0]
                  << " cooperative_groups=" << results[1]
                  << " cub=" << results[2]
                  << " expected=" << expected << '\n';
    } while (false);
    if (d_results && !check(cudaFree(d_results), "free results")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS three full-warp sum interfaces agree at lane0\n";
    return 0;
}
