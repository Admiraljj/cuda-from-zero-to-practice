#include <cuda_runtime.h>
#include <iostream>

void print_version(const char* label, int value) {
    int major = value / 1000;
    int minor = (value % 1000) / 10;
    std::cout << label << ": " << major << '.' << minor
              << " (raw=" << value << ")\n";
}

int main() {
    int runtime_version = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtime_version);
    if (err != cudaSuccess) {
        std::cerr << "cudaRuntimeGetVersion failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }
    print_version("CUDA Runtime API version", runtime_version);

    int driver_version = 0;
    err = cudaDriverGetVersion(&driver_version);
    if (err != cudaSuccess) {
        std::cerr << "cudaDriverGetVersion failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }
    print_version("CUDA version reported by driver", driver_version);

    std::cout << "Version report only; no kernel or allocation tested.\n";
    return 0;
}
