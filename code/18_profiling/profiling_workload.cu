#include "../common/cuda_check.h"
#include "../17_convolution/filter_reference.h"
#include <cuda_profiler_api.h>

#include <chrono>
#include <cstdlib>
#include <vector>

#ifndef PROFILE_NVTX
#define PROFILE_NVTX 0
#endif
#if PROFILE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace {
void push(const char* name) {
#if PROFILE_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
}
void pop() {
#if PROFILE_NVTX
    nvtxRangePop();
#endif
}
__global__ void affine_kernel(const float* input, float* output, int n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<std::size_t>(n)) output[i] = 2.0F * input[i] + 1.0F;
}
}

int main(int argc, char** argv) {
    int n = 1048576, repeats = 32;
    const std::string mode = argc > 3 ? argv[3] : "queued";
    if (argc > 4 ||
        (argc > 1 && !filter_example::integer(argv[1], 0, 16777216, n)) ||
        (argc > 2 && !filter_example::integer(argv[2], 1, 10000, repeats)) ||
        (mode != "sync" && mode != "queued")) {
        std::cerr << "Usage: profiling_workload [n:0..16777216] [repeats:1..10000] [sync|queued]\n";
        return EXIT_FAILURE;
    }
    if (n == 0) { std::cout << "PASS empty input; no GPU launch.\n"; return EXIT_SUCCESS; }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> input(n), output(n), after(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i % 127 - 63) / 8.0F;
    push("allocate-and-upload");
    float *di = nullptr, *dout = nullptr;
    CUDA_CHECK(cudaMalloc(&di, bytes)); CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(di, input.data(), bytes, cudaMemcpyHostToDevice));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    const int threads = 256, blocks = (n + threads - 1) / threads;
    for (int warm = 0; warm < 3; ++warm) {
        affine_kernel<<<blocks, threads, 0, stream>>>(di, dout, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    pop();
    CUDA_CHECK(cudaMemsetAsync(dout, 0xFF, bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaProfilerStart());
    push("submit-and-complete");
    const auto host_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        affine_kernel<<<blocks, threads, 0, stream>>>(di, dout, n);
        CUDA_CHECK(cudaGetLastError());
        if (mode == "sync") CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const auto host_stop = std::chrono::steady_clock::now();
    pop();
    CUDA_CHECK(cudaProfilerStop());
    float event_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&event_ms, start, stop));
    const double host_ms = std::chrono::duration<double, std::milli>(host_stop - host_start).count();
    push("download-and-validate");
    CUDA_CHECK(cudaMemcpy(output.data(), dout, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(after.data(), di, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        const float expected = 2.0F * input[i] + 1.0F;
        if (!std::isfinite(output[i]) || output[i] != expected || after[i] != input[i]) {
            std::cerr << "FAIL index=" << i << '\n';
            return EXIT_FAILURE;
        }
    }
    pop();
    CUDA_CHECK(cudaEventDestroy(stop)); CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dout)); CUDA_CHECK(cudaFree(di));
    std::cout << "mode=" << mode << " n=" << n << " repeats=" << repeats
              << " Event_interval_ms=" << event_ms << " host_submit_and_wait_ms=" << host_ms << '\n'
              << "PASS output and immutable input; allocation/copies/validation excluded from these timers.\n";
    return EXIT_SUCCESS;
}
