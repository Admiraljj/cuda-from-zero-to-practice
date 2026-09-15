#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

__device__ float warp_sum_full(float value) {
    const unsigned full = 0xffffffffu;
    const int lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        const float received = __shfl_down_sync(full, value, offset);
        if (lane < offset) value += received;
    }
    return value;
}

__global__ void block_reduce(const float* input, float* block_sums, int n) {
    __shared__ float warp_sums[32];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int warp_count = blockDim.x / 32;
    const int i = blockIdx.x * blockDim.x + tid;

    const float value = i < n ? input[i] : 0.0f;
    const float warp_sum = warp_sum_full(value);
    if (lane == 0) warp_sums[warp] = warp_sum;

    __syncthreads();

    if (warp == 0) {
        const float warp_value = lane < warp_count ? warp_sums[lane] : 0.0f;
        const float block_sum = warp_sum_full(warp_value);
        if (lane == 0) block_sums[blockIdx.x] = block_sum;
    }
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
        int n = 1003, threads = 128;
        if (argc != 1 && (argc != 3 ||
            !parse_decimal(argv[1], 0, 1048576, n) ||
            !parse_decimal(argv[2], 32, 1024, threads))) {
            std::cerr << "Usage: block_warp_reduce [n threads]; "
                         "n=0..1048576, threads=32..1024 and multiple of 32\n";
            return 2;
        }
        if (threads % 32 != 0) {
            std::cerr << "threads must be a multiple of 32\n";
            return 2;
        }
        if (n == 0) {
            std::cout << "EMPTY sum=0 block_count=0; no CUDA calls\n";
            return 0;
        }
        const int block_count = n / threads + (n % threads != 0);
        int device = -1;
        cudaDeviceProp prop = {};
        if (!check(cudaGetDevice(&device), "current device") ||
            !check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
        if (prop.warpSize != 32 || threads > prop.maxThreadsPerBlock ||
            threads > prop.maxThreadsDim[0] || block_count > prop.maxGridSize[0]) {
            std::cerr << "configuration exceeds device limits\n";
            return 1;
        }

        std::vector<float> input(n), block_sums(block_count);
        long long expected_total = 0;
        for (int i = 0; i < n; ++i) {
            const int value = i % 31 - 15;
            input[i] = static_cast<float>(value);
            expected_total += value;
        }
        float *d_input = nullptr, *d_block_sums = nullptr;
        bool ok = false;
        do {
            if (!check(cudaMalloc(&d_input, input.size() * sizeof(float)),
                       "allocate input") ||
                !check(cudaMalloc(&d_block_sums,
                                  block_sums.size() * sizeof(float)),
                       "allocate block sums") ||
                !check(cudaMemcpy(d_input, input.data(),
                                  input.size() * sizeof(float),
                                  cudaMemcpyHostToDevice), "H2D") ||
                !check(cudaMemset(d_block_sums, 0xff,
                                  block_sums.size() * sizeof(float)),
                       "poison block sums")) break;
            block_reduce<<<block_count, threads>>>(d_input, d_block_sums, n);
            if (!check(cudaGetLastError(), "launch") ||
                !check(cudaDeviceSynchronize(), "wait") ||
                !check(cudaMemcpy(block_sums.data(), d_block_sums,
                                  block_sums.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost), "D2H")) break;
            ok = true;
            long long actual_total = 0;
            for (int block = 0; block < block_count; ++block) {
                const int begin = block * threads;
                const int end = begin + threads < n ? begin + threads : n;
                long long expected = 0;
                for (int i = begin; i < end; ++i) {
                    expected += static_cast<int>(input[i]);
                }
                if (!std::isfinite(block_sums[block]) ||
                    static_cast<double>(block_sums[block]) !=
                    static_cast<double>(expected)) {
                    std::cerr << "block sum mismatch block=" << block << '\n';
                    ok = false;
                    break;
                }
                actual_total += static_cast<long long>(block_sums[block]);
                std::cout << "block=" << block << " begin=" << begin
                          << " end=" << end << " sum=" << block_sums[block] << '\n';
            }
            if (ok && actual_total != expected_total) {
                std::cerr << "CPU final merge mismatch\n";
                ok = false;
            }
            if (ok) {
                std::cout << "n=" << n << " threads=" << threads
                          << " warps_per_block=" << threads / 32
                          << " block_count=" << block_count
                          << " total=" << actual_total << '\n';
            }
        } while (false);
        if (d_block_sums &&
            !check(cudaFree(d_block_sums), "free block sums")) ok = false;
        if (d_input && !check(cudaFree(d_input), "free input")) ok = false;
        if (!ok) return 1;
        std::cout << "PASS two-level warp/block reduction; every block sum checked\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
