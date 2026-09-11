#include "float_buffer.cuh"
#include <cmath>
#include <iostream>
#include <string>

__global__ void scale(float* values, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] *= 2.0f;
}

int main(int argc, char** argv) {
    const bool fail = argc == 2 && std::string(argv[1]) == "--fail-after-allocation";
    if (argc > 2 || (argc == 2 && !fail)) {
        std::cerr << "usage: raii_scale [--fail-after-allocation]\n";
        return 2;
    }
    try {
        const int n = 8;
        float host[n] = {};
        for (int i = 0; i < n; ++i) host[i] = static_cast<float>(i + 1);
        FloatBuffer device(n);
        if (fail) throw std::runtime_error("planned failure after allocation");
        require_cuda(cudaMemcpy(device.get(), host, device.bytes(), cudaMemcpyHostToDevice), "H2D");
        scale<<<1, 32>>>(device.get(), n);
        require_cuda(cudaGetLastError(), "launch");
        require_cuda(cudaDeviceSynchronize(), "wait");
        require_cuda(cudaMemcpy(host, device.get(), device.bytes(), cudaMemcpyDeviceToHost), "D2H");
        for (int i = 0; i < n; ++i) {
            std::cout << "i=" << i << " value=" << host[i] << '\n';
            if (!std::isfinite(host[i]) || host[i] != 2.0f * (i + 1)) {
                throw std::runtime_error("numeric mismatch");
            }
        }
        require_cuda(device.close(), "explicit release");
        std::cout << "PASS RAII scale: 8 elements\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
