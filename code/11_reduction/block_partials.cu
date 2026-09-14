#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

constexpr int width = 8;

__global__ void block_partials(const float* input, float* partial, int n) {
    if (blockDim.x != width) return;
    __shared__ float values[width];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * width + tid;
    values[tid] = i < n ? input[i] : 0.0f;
    __syncthreads();
    for (int stride = width / 2; stride > 0; stride /= 2) {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = values[0];
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    try {
        constexpr int n = 13;
        constexpr int blocks = n / width + (n % width != 0);
        const std::size_t input_bytes = n * sizeof(float);
        const std::size_t partial_bytes = blocks * sizeof(float);
        std::vector<float> input(n), partial(blocks);
        for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i + 1);
        float *d_input = nullptr, *d_partial = nullptr;
        bool ok = false;
        double total = 0.0;
        do {
            if (!check(cudaMalloc(&d_input, input_bytes), "allocate input")) break;
            if (!check(cudaMalloc(&d_partial, partial_bytes), "allocate partial")) break;
            if (!check(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice), "H2D")) break;
            if (!check(cudaMemset(d_partial, 0xff, partial_bytes), "poison partial")) break;
            block_partials<<<blocks, width>>>(d_input, d_partial, n);
            if (!check(cudaGetLastError(), "launch")) break;
            if (!check(cudaDeviceSynchronize(), "wait")) break;
            if (!check(cudaMemcpy(partial.data(), d_partial, partial_bytes, cudaMemcpyDeviceToHost), "D2H")) break;
            ok = true;
            for (int block = 0; block < blocks; ++block) {
                int expected = 0;
                const int base = block * width;
                for (int i = base; i < base + width && i < n; ++i) expected += i + 1;
                if (!std::isfinite(partial[block]) || partial[block] != static_cast<float>(expected)) {
                    std::cerr << "mismatch block=" << block << '\n';
                    ok = false;
                    break;
                }
                total += partial[block];
            }
            if (!std::isfinite(total) || total != 91.0) ok = false;
        } while (false);
        if (d_partial != nullptr && !check(cudaFree(d_partial), "free partial")) ok = false;
        if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
        if (!ok) return 1;
        for (int block = 0; block < blocks; ++block) {
            std::cout << "block=" << block << " partial=" << partial[block] << '\n';
        }
        std::cout << "total=" << total << '\n';
        std::cout << "PASS GPU block partials + CPU final sum; 13 inputs, 2 checked partials\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
