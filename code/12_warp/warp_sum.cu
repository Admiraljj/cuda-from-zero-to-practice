#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

__device__ float warp_sum_full(float value) {
    const unsigned full = 0xffffffffu;
    for (int offset = 16; offset > 0; offset /= 2) {
        const float received = __shfl_down_sync(full, value, offset);
        const int lane = threadIdx.x % 32;
        if (lane < offset) value += received;
    }
    return value;
}

__global__ void sum_one_warp_per_block(const float* input, float* partial, int n) {
    const int lane = threadIdx.x;
    const int i = blockIdx.x * 32 + lane;
    const float value = i < n ? input[i] : 0.0f;
    const float sum = warp_sum_full(value);
    if (lane == 0) partial[blockIdx.x] = sum;
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool parse_decimal(const char* text, int minimum, int maximum, int& result) {
    if (!text || !*text) return false;
    int value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        const int digit = *p - '0';
        if (value > (maximum - digit) / 10) return false;
        value = value * 10 + digit;
        if (value > maximum) return false;
    }
    if (value < minimum) return false;
    result = value;
    return true;
}

int main(int argc, char** argv) {
    try {
        int n = 65;
        if (argc != 1 && (argc != 2 || !parse_decimal(argv[1], 0, 1048576, n))) {
            std::cerr << "Usage: warp_sum [n]; n=0..1048576\n";
            return 2;
        }
        if (n == 0) {
            std::cout << "EMPTY sum=0 partial_count=0; no CUDA calls\n";
            return 0;
        }
        constexpr int threads = 32;
        const int partial_count = n / threads + (n % threads != 0);
        int device = -1;
        cudaDeviceProp prop = {};
        if (!check(cudaGetDevice(&device), "current device") ||
            !check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
        if (prop.warpSize != threads || prop.maxThreadsPerBlock < threads ||
            prop.maxThreadsDim[0] < threads || partial_count > prop.maxGridSize[0]) {
            std::cerr << "configuration exceeds device limits\n";
            return 1;
        }
        std::vector<float> input(n), partial(partial_count);
        long long expected_total = 0;
        for (int i = 0; i < n; ++i) {
            const int value = i % 11 - 5;
            input[i] = static_cast<float>(value);
            expected_total += value;
        }
        float *d_input = nullptr, *d_partial = nullptr;
        bool ok = false;
        do {
            if (!check(cudaMalloc(&d_input, input.size() * sizeof(float)), "allocate input") ||
                !check(cudaMalloc(&d_partial, partial.size() * sizeof(float)), "allocate partial") ||
                !check(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
                                  cudaMemcpyHostToDevice), "H2D") ||
                !check(cudaMemset(d_partial, 0xff, partial.size() * sizeof(float)),
                       "poison partial")) break;
            sum_one_warp_per_block<<<partial_count, threads>>>(d_input, d_partial, n);
            if (!check(cudaGetLastError(), "launch") ||
                !check(cudaDeviceSynchronize(), "wait") ||
                !check(cudaMemcpy(partial.data(), d_partial, partial.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost), "D2H")) break;
            ok = true;
            long long downloaded_total = 0;
            for (int block = 0; block < partial_count; ++block) {
                long long expected_partial = 0;
                const int begin = block * threads;
                const int end = begin + threads < n ? begin + threads : n;
                for (int i = begin; i < end; ++i) expected_partial += static_cast<int>(input[i]);
                if (!std::isfinite(partial[block]) ||
                    static_cast<double>(partial[block]) != static_cast<double>(expected_partial)) {
                    std::cerr << "partial mismatch block=" << block
                              << " actual=" << partial[block]
                              << " expected=" << expected_partial << '\n';
                    ok = false;
                    break;
                }
                downloaded_total += static_cast<long long>(partial[block]);
                std::cout << "block=" << block << " begin=" << begin << " end=" << end
                          << " partial=" << partial[block] << '\n';
            }
            if (ok && downloaded_total != expected_total) {
                std::cerr << "CPU final merge mismatch\n";
                ok = false;
            }
            if (ok) {
                std::cout << "n=" << n << " partial_count=" << partial_count
                          << " sum=" << downloaded_total << '\n';
            }
        } while (false);
        if (d_partial && !check(cudaFree(d_partial), "free partial")) ok = false;
        if (d_input && !check(cudaFree(d_input), "free input")) ok = false;
        if (!ok) return 1;
        std::cout << "PASS one-full-warp-per-block partial sums; CPU final merge checked\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
