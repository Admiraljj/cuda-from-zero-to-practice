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

    std::cout << "visible CUDA devices: " << count << '\n';
    if (count == 0) {
        std::cerr << "No visible CUDA device; GPU lessons are not ready.\n";
        return 2;
    }

    std::cout << "PASS enumeration (no kernel executed)\n";
    return 0;
}
