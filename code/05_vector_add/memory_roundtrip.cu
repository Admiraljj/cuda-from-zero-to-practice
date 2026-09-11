#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    const int n = 4;
    float input[n] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[n] = {};
    const std::size_t bytes = sizeof(input);
    float* device = nullptr;
    bool ok = false;

    do {
        if (!check(cudaMalloc(&device, bytes), "allocate")) break;
        if (!check(cudaMemcpy(device, input, bytes, cudaMemcpyHostToDevice),
                   "copy to device")) break;
        if (!check(cudaMemcpy(output, device, bytes, cudaMemcpyDeviceToHost),
                   "copy to host")) break;

        ok = true;
        for (int i = 0; i < n; ++i) {
            std::cout << "i=" << i << " input=" << input[i]
                      << " output=" << output[i] << '\n';
            if (output[i] != input[i]) ok = false;
        }
    } while (false);

    if (device != nullptr) {
        if (!check(cudaFree(device), "free")) ok = false;
    }
    if (!ok) {
        std::cerr << "FAIL memory roundtrip\n";
        return 1;
    }
    std::cout << "PASS memory roundtrip: 4 elements; no kernel launched\n";
    return 0;
}
