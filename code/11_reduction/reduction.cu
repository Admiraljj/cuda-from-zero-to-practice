#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

__global__ void reduce_stage(const float* input, float* output, int n) {
    extern __shared__ float values[];
    const int tid = threadIdx.x;
    const int width = blockDim.x;
    const int i = blockIdx.x * width + tid;
    values[tid] = i < n ? input[i] : 0.0f;
    __syncthreads();
    for (int stride = width / 2; stride > 0; stride /= 2) {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = values[0];
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool parse_decimal(const char* text, int minimum, int maximum, int& result) {
    if (text == nullptr || *text == '\0') return false;
    int value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
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

bool power_of_two(int value) {
    if (value < 1) return false;
    while (value % 2 == 0) value /= 2;
    return value == 1;
}

int main(int argc, char** argv) {
    try {
        int n = 13, threads = 8;
        if (argc != 1 && (argc != 3 ||
            !parse_decimal(argv[1], 0, 1048576, n) ||
            !parse_decimal(argv[2], 2, 1024, threads))) {
            std::cerr << "Usage: reduction [n threads]; n=0..1048576, threads=2..1024\n";
            return 2;
        }
        if (!power_of_two(threads)) {
            std::cerr << "threads must be a power of two and at least 2\n";
            return 2;
        }
        if (n == 0) {
            std::cout << "EMPTY sum=0; no CUDA calls\n";
            return 0;
        }
        int device = -1;
        cudaDeviceProp prop = {};
        if (!check(cudaGetDevice(&device), "current device")) return 1;
        if (!check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
        const int first_blocks = n / threads + (n % threads != 0);
        const std::size_t shared_bytes = static_cast<std::size_t>(threads) * sizeof(float);
        if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0] ||
            first_blocks > prop.maxGridSize[0] || shared_bytes > prop.sharedMemPerBlock) {
            std::cerr << "configuration exceeds device limits\n";
            return 1;
        }
        std::vector<float> input(n), partial(first_blocks);
        std::vector<double> reference(n), next(first_blocks);
        long long exact_total = 0;
        for (int i = 0; i < n; ++i) {
            const int value = i % 31 - 15;
            input[i] = static_cast<float>(value);
            reference[i] = value;
            exact_total += value;
        }
        const std::size_t input_bytes = static_cast<std::size_t>(n) * sizeof(float);
        const std::size_t partial_capacity_bytes = static_cast<std::size_t>(first_blocks) * sizeof(float);
        float *d_a = nullptr, *d_b = nullptr;
        bool ok = false;
        int passes = 0;
        double result = 0.0;
        do {
            if (!check(cudaMalloc(&d_a, input_bytes), "allocate A")) break;
            if (!check(cudaMalloc(&d_b, partial_capacity_bytes), "allocate B")) break;
            if (!check(cudaMemcpy(d_a, input.data(), input_bytes, cudaMemcpyHostToDevice), "H2D")) break;
            float* read = d_a;
            float* write = d_b;
            int current_n = n;
            bool all_passes_ok = true;
            while (true) {
                const int blocks = current_n / threads + (current_n % threads != 0);
                const std::size_t output_bytes = static_cast<std::size_t>(blocks) * sizeof(float);
                next.resize(blocks);
                for (int block = 0; block < blocks; ++block) {
                    double expected = 0.0;
                    const int base = block * threads;
                    for (int i = base; i < base + threads && i < current_n; ++i) expected += reference[i];
                    next[block] = expected;
                }
                if (!check(cudaMemset(write, 0xff, output_bytes), "poison stage output")) {
                    all_passes_ok = false;
                    break;
                }
                reduce_stage<<<blocks, threads, shared_bytes>>>(read, write, current_n);
                if (!check(cudaGetLastError(), "launch") ||
                    !check(cudaDeviceSynchronize(), "wait") ||
                    !check(cudaMemcpy(partial.data(), write, output_bytes, cudaMemcpyDeviceToHost), "D2H")) {
                    all_passes_ok = false;
                    break;
                }
                for (int block = 0; block < blocks; ++block) {
                    if (!std::isfinite(partial[block]) || static_cast<double>(partial[block]) != next[block]) {
                        std::cerr << "mismatch pass=" << passes << " block=" << block << '\n';
                        all_passes_ok = false;
                        break;
                    }
                }
                if (!all_passes_ok) break;
                std::cout << "pass=" << passes << " input_n=" << current_n << " output_n=" << blocks
                          << " first_partial=" << partial[0] << '\n';
                ++passes;
                std::swap(read, write);
                reference.swap(next);
                current_n = blocks;
                if (current_n == 1) {
                    result = partial[0];
                    break;
                }
            }
            ok = all_passes_ok && std::isfinite(result) && result == static_cast<double>(exact_total);
            if (!ok) std::cerr << "GPU reduction did not match exact integer reference\n";
        } while (false);
        if (d_b != nullptr && !check(cudaFree(d_b), "free B")) ok = false;
        if (d_a != nullptr && !check(cudaFree(d_a), "free A")) ok = false;
        if (!ok) return 1;
        std::cout << "sum=" << result << " passes=" << passes << '\n';
        std::cout << "PASS GPU multipass reduction; every stage downloaded and checked\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
