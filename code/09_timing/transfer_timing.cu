#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    const int n = 1048576, threads = 256, blocks = n / threads;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    float *a = nullptr, *b = nullptr, *c = nullptr;
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaEvent_t event[4] = {};
    float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f, span_ms = 0.0f;
    double request_ms = 0.0;
    bool ok = false;
    do {
        if (!check(cudaMallocHost(&a, bytes), "pinned a")) break;
        if (!check(cudaMallocHost(&b, bytes), "pinned b")) break;
        if (!check(cudaMallocHost(&c, bytes), "pinned c")) break;
        if (!check(cudaMalloc(&d_a, bytes), "device a")) break;
        if (!check(cudaMalloc(&d_b, bytes), "device b")) break;
        if (!check(cudaMalloc(&d_c, bytes), "device c")) break;
        bool created = true;
        for (int j = 0; j < 4; ++j) {
            if (!check(cudaEventCreate(&event[j]), "create event")) { created = false; break; }
        }
        if (!created) break;
        for (int i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i % 251);
            b[i] = static_cast<float>(i % 113 - 56);
        }
        if (!check(cudaMemcpyAsync(d_a, a, bytes, cudaMemcpyHostToDevice), "warm H2D a")) break;
        if (!check(cudaMemcpyAsync(d_b, b, bytes, cudaMemcpyHostToDevice), "warm H2D b")) break;
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
        if (!check(cudaGetLastError(), "warm kernel")) break;
        if (!check(cudaMemcpyAsync(c, d_c, bytes, cudaMemcpyDeviceToHost), "warm D2H")) break;
        if (!check(cudaDeviceSynchronize(), "finish warm request")) break;

        for (int i = 0; i < n; ++i) c[i] = std::numeric_limits<float>::quiet_NaN();
        if (!check(cudaMemset(d_c, 0xff, bytes), "poison device output")) break;
        if (!check(cudaDeviceSynchronize(), "finish preparation")) break;
        using Clock = std::chrono::steady_clock;
        const auto host_start = Clock::now();
        if (!check(cudaEventRecord(event[0]), "begin H2D")) break;
        if (!check(cudaMemcpyAsync(d_a, a, bytes, cudaMemcpyHostToDevice), "H2D a")) break;
        if (!check(cudaMemcpyAsync(d_b, b, bytes, cudaMemcpyHostToDevice), "H2D b")) break;
        if (!check(cudaEventRecord(event[1]), "end H2D")) break;
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
        if (!check(cudaGetLastError(), "kernel launch")) break;
        if (!check(cudaEventRecord(event[2]), "end kernel")) break;
        if (!check(cudaMemcpyAsync(c, d_c, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        if (!check(cudaEventRecord(event[3]), "end D2H")) break;
        if (!check(cudaEventSynchronize(event[3]), "finish request")) break;
        const auto host_stop = Clock::now();
        request_ms = std::chrono::duration<double, std::milli>(host_stop - host_start).count();
        if (!check(cudaEventElapsedTime(&h2d_ms, event[0], event[1]), "H2D time")) break;
        if (!check(cudaEventElapsedTime(&kernel_ms, event[1], event[2]), "kernel time")) break;
        if (!check(cudaEventElapsedTime(&d2h_ms, event[2], event[3]), "D2H time")) break;
        if (!check(cudaEventElapsedTime(&span_ms, event[0], event[3]), "device span")) break;

        ok = std::isfinite(request_ms) && request_ms > 0.0;
        for (float value : {h2d_ms, kernel_ms, d2h_ms, span_ms}) {
            if (!std::isfinite(value) || value <= 0.0f) ok = false;
        }
        for (int i = 0; i < n; ++i) {
            const float expected = static_cast<float>(i % 251 + i % 113 - 56);
            if (!std::isfinite(c[i]) || c[i] != expected) { ok = false; break; }
        }
    } while (false);
    // Best-effort wait before releasing resources, including on an early error path.
    if (d_a != nullptr && !check(cudaDeviceSynchronize(), "cleanup wait")) ok = false;
    for (int j = 3; j >= 0; --j) {
        if (event[j] != nullptr && !check(cudaEventDestroy(event[j]), "destroy event")) ok = false;
    }
    if (d_c != nullptr && !check(cudaFree(d_c), "free device c")) ok = false;
    if (d_b != nullptr && !check(cudaFree(d_b), "free device b")) ok = false;
    if (d_a != nullptr && !check(cudaFree(d_a), "free device a")) ok = false;
    if (c != nullptr && !check(cudaFreeHost(c), "free pinned c")) ok = false;
    if (b != nullptr && !check(cudaFreeHost(b), "free pinned b")) ok = false;
    if (a != nullptr && !check(cudaFreeHost(a), "free pinned a")) ok = false;
    if (!ok) { std::cerr << "request validation/timing failed\n"; return 1; }
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "n=" << n << " H2D_bytes=" << 2ULL * bytes << " D2H_bytes=" << bytes << '\n';
    std::cout << "H2D_event_ms=" << h2d_ms << " kernel_event_ms=" << kernel_ms
              << " D2H_event_ms=" << d2h_ms << '\n';
    std::cout << "device_span_ms=" << span_ms << " reused_buffer_request_ms=" << request_ms << '\n';
    std::cout << "PASS staged request: 1048576 elements; sequential default stream\n";
    return 0;
}
