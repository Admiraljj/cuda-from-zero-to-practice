#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

__global__ void stencil_radius(const float* input, float* output, int n, int radius) {
    extern __shared__ float tile[];
    const int tid = threadIdx.x;
    const int width = blockDim.x;
    const int base = blockIdx.x * width;
    const int slots = width + 2 * radius;
    for (int j = tid; j < slots; j += width) {
        const int global = base + j - radius;
        tile[j] = global >= 0 && global < n ? input[global] : 0.0f;
    }
    __syncthreads();
    const int i = base + tid;
    if (i < n) {
        float sum = 0.0f;
        for (int offset = 0; offset <= 2 * radius; ++offset) {
            sum += tile[tid + offset];
        }
        output[i] = sum;
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

bool run_case(int n, int radius, int threads) {
    if (n == 0) {
        std::cout << "EMPTY n=0 radius=" << radius << " threads=" << threads
                  << "; no CUDA call for this case\n";
        return true;
    }
    int device = -1;
    cudaDeviceProp prop = {};
    if (!check(cudaGetDevice(&device), "current device")) return false;
    if (!check(cudaGetDeviceProperties(&prop, device), "device properties")) return false;
    const int blocks = n / threads + (n % threads != 0);
    const std::size_t shared_bytes = static_cast<std::size_t>(threads + 2 * radius) * sizeof(float);
    if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0] ||
        blocks > prop.maxGridSize[0] || shared_bytes > prop.sharedMemPerBlock) {
        std::cerr << "configuration exceeds device limits\n";
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i % 7 - 3);
    float *d_input = nullptr, *d_output = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input")) break;
        if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
        if (!check(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;
        if (!check(cudaMemset(d_output, 0xff, bytes), "poison output")) break;
        stencil_radius<<<blocks, threads, shared_bytes>>>(d_input, d_output, n, radius);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "wait")) break;
        if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int i = 0; i < n; ++i) {
            int expected = 0;
            for (int offset = -radius; offset <= radius; ++offset) {
                const int neighbor = i + offset;
                if (neighbor >= 0 && neighbor < n) expected += neighbor % 7 - 3;
            }
            if (!std::isfinite(output[i]) || output[i] != static_cast<float>(expected)) {
                std::cerr << "mismatch i=" << i << " actual=" << output[i]
                          << " expected=" << expected << '\n';
                ok = false;
                break;
            }
        }
    } while (false);
    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (ok) {
        std::cout << "PASS n=" << n << " radius=" << radius << " threads=" << threads
                  << " blocks=" << blocks << " shared_bytes=" << shared_bytes << '\n';
    }
    return ok;
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--matrix") {
            const int sizes[] = {0, 1, 3, 4, 5, 17, 65};
            const int radii[] = {0, 1, 2, 8};
            const int widths[] = {1, 4, 7, 32};
            int cases = 0;
            for (int radius : radii) {
                for (int threads : widths) {
                    for (int n : sizes) {
                        if (!run_case(n, radius, threads)) return 1;
                        ++cases;
                    }
                }
            }
            std::cout << "PASS radius matrix: " << cases << " cases, including 16 empty cases\n";
            return 0;
        }
        int n = 10, radius = 2, threads = 4;
        if (argc != 1 && (argc != 4 ||
            !parse_decimal(argv[1], 0, 1048576, n) ||
            !parse_decimal(argv[2], 0, 64, radius) ||
            !parse_decimal(argv[3], 1, 1024, threads))) {
            std::cerr << "Usage: stencil_radius [n radius threads] | --matrix\n"
                      << "n: 0..1048576, radius: 0..64, threads: 1..1024\n";
            return 2;
        }
        return run_case(n, radius, threads) ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
