#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

__global__ void transform(const float* input, float* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 2.0f * input[i] + 1.0f;
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    const int n = 1048576, repeats = 100, threads = 256, warmups = 4;
    const int blocks = (n + threads - 1) / threads;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i % 97);
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    float total_ms = 0.0f;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input")) break;
        if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
        if (!check(cudaEventCreate(&start), "create start")) break;
        if (!check(cudaEventCreate(&stop), "create stop")) break;
        if (!check(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;

        bool launch_ok = true;
        for (int r = 0; r < warmups; ++r) {
            transform<<<blocks, threads>>>(d_input, d_output, n);
            if (!check(cudaGetLastError(), "warmup launch")) { launch_ok = false; break; }
        }
        if (!launch_ok || !check(cudaDeviceSynchronize(), "finish warmups")) break;
        if (!check(cudaEventRecord(start), "record start")) break;
        for (int r = 0; r < repeats; ++r) {
            transform<<<blocks, threads>>>(d_input, d_output, n);
            if (!check(cudaGetLastError(), "timed launch")) { launch_ok = false; break; }
        }
        if (!launch_ok) break;
        if (!check(cudaEventRecord(stop), "record stop")) break;
        if (!check(cudaEventSynchronize(stop), "wait for stop")) break;
        if (!check(cudaEventElapsedTime(&total_ms, start, stop), "elapsed time")) break;
        if (!std::isfinite(total_ms) || total_ms <= 0.0f) {
            std::cerr << "No usable positive event interval; adjust the experiment\n";
            break;
        }
        if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int i = 0; i < n; ++i) {
            const float expected = static_cast<float>(2 * (i % 97) + 1);
            if (!std::isfinite(output[i]) || output[i] != expected) ok = false;
        }
    } while (false);
    if (stop != nullptr && !check(cudaEventDestroy(stop), "destroy stop")) ok = false;
    if (start != nullptr && !check(cudaEventDestroy(start), "destroy start")) ok = false;
    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (!ok) return 1;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "n=" << n << " repeats=" << repeats << " warmups=" << warmups << '\n';
    std::cout << "total_event_ms=" << total_ms << '\n';
    std::cout << "per_kernel_ms=" << static_cast<double>(total_ms) / repeats << '\n';
    std::cout << "PASS event result: 1048576 elements; one batch, not latency percentiles\n";
    return 0;
}
