#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

__global__ void shared_probe(float* output) {
    extern __shared__ float tile[];
    const int tid = threadIdx.x;
    tile[tid] = static_cast<float>(tid);
    __syncthreads();
    if (tid == 0) output[blockIdx.x] = tile[blockDim.x - 1];
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    int device = -1;
    cudaDeviceProp prop = {};
    cudaFuncAttributes attr = {};
    if (!check(cudaGetDevice(&device), "current device")) return 1;
    if (!check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
    if (!check(cudaFuncGetAttributes(&attr, shared_probe), "kernel attributes")) return 1;
    std::cout << "device=" << device << " name=" << prop.name << '\n';
    std::cout << "shared_per_block=" << prop.sharedMemPerBlock
              << " shared_per_SM=" << prop.sharedMemPerMultiprocessor << '\n';
    std::cout << "kernel_static_shared=" << attr.sharedSizeBytes
              << " kernel_max_dynamic_shared=" << attr.maxDynamicSharedSizeBytes
              << " registers_per_thread=" << attr.numRegs << '\n';
    const int widths[] = {32, 64, 128, 256};
    const int extras[] = {0, 4096, 16384};
    const int max_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    if (max_warps <= 0) {
        std::cerr << "invalid device warp capacity\n";
        return 1;
    }
    int queried = 0;
    std::cout << std::fixed << std::setprecision(2);
    for (int threads : widths) {
        for (int extra : extras) {
            const std::size_t dynamic_bytes = static_cast<std::size_t>(threads) * sizeof(float) + extra;
            if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0] ||
                threads > attr.maxThreadsPerBlock ||
                dynamic_bytes + attr.sharedSizeBytes > prop.sharedMemPerBlock ||
                dynamic_bytes > static_cast<std::size_t>(attr.maxDynamicSharedSizeBytes)) {
                std::cout << "SKIP threads=" << threads << " dynamic_bytes=" << dynamic_bytes << '\n';
                continue;
            }
            int active_blocks = 0;
            if (!check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active_blocks, shared_probe, threads, dynamic_bytes),
                       "occupancy query")) return 1;
            const int warps_per_block = (threads + prop.warpSize - 1) / prop.warpSize;
            const int active_warps = active_blocks * warps_per_block;
            const double percent = 100.0 * active_warps / max_warps;
            std::cout << "threads=" << threads << " dynamic_bytes=" << dynamic_bytes
                      << " active_blocks_per_SM=" << active_blocks
                      << " theoretical_occupancy_percent=" << percent << '\n';
            ++queried;
        }
    }
    if (queried == 0) {
        std::cerr << "no supported teaching configuration\n";
        return 1;
    }
    std::cout << "PASS resource query: " << queried << " configurations; no kernel launched\n";
    return 0;
}
