#include <cuda_runtime.h>
#include <iostream>

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorName(err) << ": "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "device count: " << count << '\n';
    if (count == 0) {
        std::cerr << "No visible CUDA device; GPU lessons are not ready.\n";
        return 2;
    }

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        err = cudaGetDeviceProperties(&p, i);
        if (err != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed for device " << i
                      << ": " << cudaGetErrorName(err)
                      << ": " << cudaGetErrorString(err) << '\n';
            return 1;
        }

        std::cout << '[' << i << "] " << p.name << '\n'
                  << "  compute capability: " << p.major << '.' << p.minor << '\n'
                  << "  global memory: "
                  << p.totalGlobalMem / (1024.0 * 1024 * 1024) << " GiB\n"
                  << "  SM count: " << p.multiProcessorCount << '\n'
                  << "  warp size: " << p.warpSize << '\n'
                  << "  max threads/block: " << p.maxThreadsPerBlock << '\n'
                  << "  max block dimensions: "
                  << p.maxThreadsDim[0] << ','
                  << p.maxThreadsDim[1] << ','
                  << p.maxThreadsDim[2] << '\n'
                  << "  max grid dimensions: "
                  << p.maxGridSize[0] << ','
                  << p.maxGridSize[1] << ','
                  << p.maxGridSize[2] << '\n'
                  << "  shared memory/block (default limit): "
                  << p.sharedMemPerBlock << " bytes\n";
    }

    std::cout << "PASS device query (no kernel executed)\n";
    return 0;
}
