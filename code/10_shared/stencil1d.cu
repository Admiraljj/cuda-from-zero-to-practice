#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

__global__ void stencil_average(const float* input, float* output, int n) {
    extern __shared__ float tile[];
    const int tid = threadIdx.x;
    const int width = blockDim.x;
    const int base = blockIdx.x * width;
    for (int j = tid; j < width + 2; j += width) {
        const int global = base + j - 1;
        tile[j] = global >= 0 && global < n ? input[global] : 0.0f;
    }
    __syncthreads();
    const int i = base + tid;
    if (i < n) {
        output[i] = (tile[tid] + tile[tid + 1] + tile[tid + 2]) / 3.0f;
    }
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

int main(int argc, char** argv) {
    try {
        int n = 10, steps = 3, threads = 4;
        if (argc != 1 && (argc != 4 ||
            !parse_decimal(argv[1], 0, 1048576, n) ||
            !parse_decimal(argv[2], 0, 32, steps) ||
            !parse_decimal(argv[3], 1, 1024, threads))) {
            std::cerr << "Usage: stencil1d [n steps threads]\n"
                      << "n: 0..1048576, steps: 0..32, threads: 1..1024\n";
            return 2;
        }
        if (n == 0) {
            std::cout << "EMPTY n=0; no CUDA calls\n";
            return 0;
        }
        std::vector<float> input(n), output(n);
        std::vector<double> reference(n), next(n);
        for (int i = 0; i < n; ++i) {
            input[i] = static_cast<float>(i % 17 - 8);
            reference[i] = input[i];
        }
        if (steps == 0) {
            output = input;
            if (output != input) return 1;
            std::cout << "PASS identity: n=" << n << " steps=0; no CUDA calls\n";
            return 0;
        }
        int device = -1;
        cudaDeviceProp prop = {};
        if (!check(cudaGetDevice(&device), "current device")) return 1;
        if (!check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
        const int blocks = n / threads + (n % threads != 0);
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
        const std::size_t shared_bytes = static_cast<std::size_t>(threads + 2) * sizeof(float);
        if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0] ||
            blocks > prop.maxGridSize[0] || shared_bytes > prop.sharedMemPerBlock) {
            std::cerr << "configuration exceeds device limits\n";
            return 1;
        }
        float *d_a = nullptr, *d_b = nullptr;
        bool ok = false;
        do {
            if (!check(cudaMalloc(&d_a, bytes), "allocate A")) break;
            if (!check(cudaMalloc(&d_b, bytes), "allocate B")) break;
            if (!check(cudaMemcpy(d_a, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;
            float* read = d_a;
            float* write = d_b;
            bool all_steps_ok = true;
            for (int step = 0; step < steps; ++step) {
                for (int i = 0; i < n; ++i) {
                    const double left = i > 0 ? reference[i - 1] : 0.0;
                    const double right = i + 1 < n ? reference[i + 1] : 0.0;
                    next[i] = (left + reference[i] + right) / 3.0;
                }
                if (!check(cudaMemset(write, 0xff, bytes), "poison destination")) {
                    all_steps_ok = false;
                    break;
                }
                stencil_average<<<blocks, threads, shared_bytes>>>(read, write, n);
                if (!check(cudaGetLastError(), "launch") ||
                    !check(cudaDeviceSynchronize(), "wait") ||
                    !check(cudaMemcpy(output.data(), write, bytes, cudaMemcpyDeviceToHost), "D2H")) {
                    all_steps_ok = false;
                    break;
                }
                double max_error = 0.0;
                for (int i = 0; i < n; ++i) {
                    const double error = std::fabs(static_cast<double>(output[i]) - next[i]);
                    const double tolerance = 1e-5 + 1e-5 * std::fabs(next[i]);
                    if (!std::isfinite(output[i]) || !std::isfinite(next[i]) || error > tolerance) {
                        std::cerr << "mismatch step=" << step + 1 << " i=" << i << '\n';
                        all_steps_ok = false;
                        break;
                    }
                    max_error = std::max(max_error, error);
                }
                if (!all_steps_ok) break;
                std::cout << "step=" << step + 1 << " max_abs_error=" << max_error
                          << " first=" << output.front() << " last=" << output.back() << '\n';
                std::swap(read, write);
                reference.swap(next);
            }
            ok = all_steps_ok;
        } while (false);
        if (d_b != nullptr && !check(cudaFree(d_b), "free B")) ok = false;
        if (d_a != nullptr && !check(cudaFree(d_a), "free A")) ok = false;
        if (!ok) return 1;
        std::cout << "PASS average: n=" << n << " steps=" << steps
                  << " threads=" << threads << "; every step checked\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
