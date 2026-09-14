#include <cuda_runtime.h>
#include <chrono>
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
    const int n = 65536, repeats = 100, threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i % 97);
    float* d_input = nullptr;
    float* d_output = nullptr;
    bool ok = false;
    double submit_ms = 0.0, completed_ms = 0.0;
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input")) break;
        if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
        if (!check(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;
        transform<<<blocks, threads>>>(d_input, d_output, n);
        if (!check(cudaGetLastError(), "warmup launch")) break;
        if (!check(cudaDeviceSynchronize(), "finish warmup")) break;

        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        bool launch_ok = true;
        for (int r = 0; r < repeats; ++r) {
            transform<<<blocks, threads>>>(d_input, d_output, n);
            if (!check(cudaGetLastError(), "timed launch")) {
                launch_ok = false;
                break;
            }
        }
        const auto submitted = Clock::now();
        if (!launch_ok) break;
        if (!check(cudaDeviceSynchronize(), "finish measured work")) break;
        const auto completed = Clock::now();
        submit_ms = std::chrono::duration<double, std::milli>(submitted - start).count();
        completed_ms = std::chrono::duration<double, std::milli>(completed - start).count();

        if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = std::isfinite(submit_ms) && std::isfinite(completed_ms) &&
             submit_ms >= 0.0 && completed_ms >= submit_ms;
        for (int i = 0; i < n; ++i) {
            const float expected = static_cast<float>(2 * (i % 97) + 1);
            if (!std::isfinite(output[i]) || output[i] != expected) ok = false;
        }
    } while (false);
    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (!ok) return 1;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "n=" << n << " repeats=" << repeats << '\n';
    std::cout << "submit_loop_ms=" << submit_ms << '\n';
    std::cout << "submit_and_wait_ms=" << completed_ms << '\n';
    std::cout << "remaining_wait_ms=" << completed_ms - submit_ms << '\n';
    std::cout << "PASS result: 65536 elements; timings are host intervals\n";
    return 0;
}
