#include <cuda_runtime.h>
#include "benchmark_stats.h"
#include <iomanip>
#include <iostream>
#include <string>

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int positive_decimal(const char* text, int limit) {
    if (*text == '\0') throw std::invalid_argument("empty integer argument");
    int value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') throw std::invalid_argument("use decimal digits only");
        const int digit = *p - '0';
        if (value > (limit - digit) / 10) throw std::invalid_argument("argument exceeds limit");
        value = value * 10 + digit;
        if (value > limit) throw std::invalid_argument("argument exceeds limit");
    }
    if (value == 0) throw std::invalid_argument("arguments must be positive");
    return value;
}

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int run(int n, int repeats, int threads, int rounds) {
    const int blocks = n / threads + (n % threads != 0);
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    int device = -1;
    cudaDeviceProp properties = {};
    if (!check(cudaGetDevice(&device), "current device")) return 1;
    if (!check(cudaGetDeviceProperties(&properties, device), "device properties")) return 1;
    if (threads > properties.maxThreadsPerBlock || threads > properties.maxThreadsDim[0] ||
        blocks > properties.maxGridSize[0]) {
        std::cerr << "configuration exceeds this device's launch limits\n";
        return 2;
    }

    std::vector<float> a(n), b(n), c(n);
    std::vector<double> samples;
    samples.reserve(rounds); // Allocate sample storage before CUDA resources.
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 251);
        b[i] = static_cast<float>(i % 113 - 56);
    }
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    bool ok = false;
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "device=" << device << " name=" << properties.name << '\n';
    std::cout << "n=" << n << " repeats=" << repeats << " threads=" << threads
              << " rounds=" << rounds << " blocks=" << blocks << '\n';
    std::cout << "bytes_per_array=" << bytes
              << " effective_bytes_per_kernel=" << 3ULL * bytes << '\n';
    do {
        if (!check(cudaMalloc(&d_a, bytes), "allocate a")) break;
        if (!check(cudaMalloc(&d_b, bytes), "allocate b")) break;
        if (!check(cudaMalloc(&d_c, bytes), "allocate c")) break;
        if (!check(cudaEventCreate(&start), "create start")) break;
        if (!check(cudaEventCreate(&stop), "create stop")) break;
        if (!check(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "H2D a")) break;
        if (!check(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "H2D b")) break;
        bool launches_ok = true;
        for (int r = 0; r < 4; ++r) {
            vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
            if (!check(cudaGetLastError(), "warmup launch")) { launches_ok = false; break; }
        }
        if (!launches_ok || !check(cudaDeviceSynchronize(), "finish warmups")) break;

        ok = true;
        for (int round = 0; round < rounds; ++round) {
            // CUDA float has IEEE binary32 representation: all-one bytes form a NaN.
            // Poison output outside the measured interval, before every batch.
            if (!check(cudaMemset(d_c, 0xff, bytes), "poison output")) { ok = false; break; }
            if (!check(cudaEventRecord(start), "record start")) { ok = false; break; }
            for (int r = 0; r < repeats; ++r) {
                vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
                if (!check(cudaGetLastError(), "timed launch")) { ok = false; break; }
            }
            if (!ok) break;
            if (!check(cudaEventRecord(stop), "record stop")) { ok = false; break; }
            if (!check(cudaEventSynchronize(stop), "wait for stop")) { ok = false; break; }
            float total_ms = 0.0f;
            if (!check(cudaEventElapsedTime(&total_ms, start, stop), "elapsed time")) { ok = false; break; }
            if (!std::isfinite(total_ms) || total_ms <= 0.0f) {
                std::cerr << "invalid event interval; adjust the experiment\n";
                ok = false;
                break;
            }
            if (!check(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "D2H c")) {
                ok = false;
                break;
            }
            for (int i = 0; i < n; ++i) {
                const float expected = static_cast<float>(i % 251 + i % 113 - 56);
                if (!std::isfinite(c[i]) || c[i] != expected) {
                    std::cerr << "mismatch round=" << round << " i=" << i << '\n';
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
            const double per_kernel_ms = static_cast<double>(total_ms) / repeats;
            samples.push_back(per_kernel_ms);
            std::cout << "round=" << round << " total_ms=" << total_ms
                      << " per_kernel_ms=" << per_kernel_ms << '\n';
        }
    } while (false);
    if (stop != nullptr && !check(cudaEventDestroy(stop), "destroy stop")) ok = false;
    if (start != nullptr && !check(cudaEventDestroy(start), "destroy start")) ok = false;
    if (d_c != nullptr && !check(cudaFree(d_c), "free c")) ok = false;
    if (d_b != nullptr && !check(cudaFree(d_b), "free b")) ok = false;
    if (d_a != nullptr && !check(cudaFree(d_a), "free a")) ok = false;
    if (!ok || samples.size() != static_cast<std::size_t>(rounds)) return 1;

    const TimingStats result = summarize(samples);
    const double seconds = result.median / 1000.0;
    const double bytes_per_kernel = 3.0 * static_cast<double>(bytes);
    const double gbps = bytes_per_kernel / seconds / 1e9;
    const double gibps = bytes_per_kernel / seconds / 1073741824.0;
    if (!std::isfinite(gbps) || !std::isfinite(gibps)) return 1;
    std::cout << "min_ms=" << result.minimum << " median_ms=" << result.median
              << " mean_ms=" << result.mean << " max_ms=" << result.maximum << '\n';
    std::cout << "effective_GBps_from_median_ms=" << gbps
              << " effective_GiBps_from_median_ms=" << gibps << '\n';
    std::cout << "PASS bandwidth result: " << n << " elements checked after each of "
              << rounds << " batches; not measured DRAM traffic\n";
    return 0;
}

int main(int argc, char** argv) {
    int n = 1048576, repeats = 100, threads = 256, rounds = 5;
    try {
        if (argc != 1 && argc != 5) throw std::invalid_argument("provide all four arguments or none");
        if (argc == 5) {
            n = positive_decimal(argv[1], 67108864);
            repeats = positive_decimal(argv[2], 1000);
            threads = positive_decimal(argv[3], 1024);
            rounds = positive_decimal(argv[4], 21);
        }
    } catch (const std::invalid_argument& error) {
        std::cerr << "argument error: " << error.what() << '\n'
                  << "usage: bandwidth [n repeats threads rounds]\n"
                  << "limits: n<=67108864 repeats<=1000 threads<=1024 rounds<=21\n";
        return 2;
    }
    try {
        return run(n, repeats, threads, rounds);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
