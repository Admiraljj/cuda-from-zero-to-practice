#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool parse_integer(const char* text, int minimum, int maximum, int& result) {
    if (*text == '\0') return false;
    int value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return false;
        const int digit = *p - '0';
        if (value > (maximum - digit) / 10) return false;
        value = value * 10 + digit;
    }
    if (value < minimum || value > maximum) return false;
    result = value;
    return true;
}

int main(int argc, char** argv) {
    int n = 1048576;
    int threads = 256;
    const int max_n = 1048576;
    bool inject_nan = false;

    if (argc > 4
        || (argc >= 2 && !parse_integer(argv[1], 0, max_n, n))
        || (argc >= 3 && !parse_integer(argv[2], 1, 1024, threads))
        || (argc == 4 && std::string(argv[3]) != "--inject-nan")) {
        std::cerr << "Usage: vector_add [n:0..1048576] [threads:1..1024]"
                     " [--inject-nan]\n";
        return 2;
    }
    inject_nan = (argc == 4);
    if (n == 0) {
        if (inject_nan) {
            std::cerr << "Cannot inject NaN into an empty output\n";
            return 2;
        }
        std::cout << "PASS: 0 elements (no GPU work)\n";
        return 0;
    }

    int device_id = 0;
    cudaDeviceProp properties{};
    if (!check(cudaGetDevice(&device_id), "current device")
        || !check(cudaGetDeviceProperties(&properties, device_id), "device properties")) {
        return 1;
    }
    if (threads > properties.maxThreadsPerBlock
        || threads > properties.maxThreadsDim[0]) {
        std::cerr << "threads exceeds this device's one-dimensional block limit\n";
        return 2;
    }

    const int blocks = n / threads + (n % threads != 0);
    if (blocks > properties.maxGridSize[0]) {
        std::cerr << "grid exceeds this device's x limit\n";
        return 2;
    }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = 0.5f * i;
        h_b[i] = 2.0f * i;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    bool ok = false;
    std::cout << "n=" << n << " threads=" << threads
              << " blocks=" << blocks << " slots=" << blocks * threads
              << " bytes_per_array=" << bytes << '\n';

    do {
        if (!check(cudaMalloc(&d_a, bytes), "allocate a")) break;
        if (!check(cudaMalloc(&d_b, bytes), "allocate b")) break;
        if (!check(cudaMalloc(&d_c, bytes), "allocate c")) break;
        if (!check(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice),
                   "copy a to device")) break;
        if (!check(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice),
                   "copy b to device")) break;

        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "execute")) break;
        if (!check(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost),
                   "copy c to host")) break;

        if (inject_nan) {
            h_c[n / 2] = std::numeric_limits<float>::quiet_NaN();
            std::cerr << "Injected NaN into host validation input\n";
        }
        ok = true;
        for (int i = 0; i < n; ++i) {
            const float expected = h_a[i] + h_b[i];
            const float actual = h_c[i];
            if (!std::isfinite(expected) || !std::isfinite(actual)
                || std::fabs(actual - expected) > 1e-5f) {
                std::cerr << "mismatch at " << i << ": actual=" << actual
                          << " expected=" << expected << '\n';
                ok = false;
                break;
            }
        }
        if (ok && n <= 16) {
            for (int i = 0; i < n; ++i) {
                std::cout << "i=" << i << " a=" << h_a[i]
                          << " b=" << h_b[i] << " c=" << h_c[i] << '\n';
            }
        }
    } while (false);

    if (d_c != nullptr) {
        if (!check(cudaFree(d_c), "free c")) ok = false;
    }
    if (d_b != nullptr) {
        if (!check(cudaFree(d_b), "free b")) ok = false;
    }
    if (d_a != nullptr) {
        if (!check(cudaFree(d_a), "free a")) ok = false;
    }
    if (!ok) return 1;
    std::cout << "PASS: " << n << " elements\n";
    return 0;
}
