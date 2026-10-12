#include "../common/cuda_check.h"
#include "../17_convolution/filter_reference.h"
#include <cstdlib>
#include <limits>

__global__ void add_one(float* values, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] += 1.0F;
}

int main(int argc, char** argv) {
    int n = 1031;
    if (argc > 2 || (argc > 1 && !filter_example::integer(argv[1], 0, 16777216, n))) {
        std::cerr << "Usage: async_roundtrip [n:0..16777216]\n"; return EXIT_FAILURE;
    }
    if (n == 0) { std::cout << "PASS empty input; no GPU launch.\n"; return EXIT_SUCCESS; }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    float *input = nullptr, *output = nullptr, *device = nullptr;
    CUDA_CHECK(cudaMallocHost(&input, bytes)); CUDA_CHECK(cudaMallocHost(&output, bytes));
    CUDA_CHECK(cudaMalloc(&device, bytes));
    for (int i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i % 127 - 63);
        output[i] = std::numeric_limits<float>::quiet_NaN();
    }
    cudaStream_t stream{}; cudaEvent_t done{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    CUDA_CHECK(cudaMemcpyAsync(device, input, bytes, cudaMemcpyHostToDevice, stream));
    add_one<<<(n + 255) / 256, 256, 0, stream>>>(device, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(output, device, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(done, stream));
    const cudaError_t state = cudaEventQuery(done);
    if (state != cudaSuccess && state != cudaErrorNotReady) CUDA_CHECK(state);
    std::cout << "query=" << (state == cudaSuccess ? "complete" : "not-ready")
              << "; neither outcome is assumed in advance.\n";
    // Do not read output or modify/free input until this completion point.
    CUDA_CHECK(cudaEventSynchronize(done));
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(output[i]) || output[i] != input[i] + 1.0F ||
            input[i] != static_cast<float>(i % 127 - 63)) {
            std::cerr << "FAIL index=" << i << '\n'; return EXIT_FAILURE;
        }
    }
    CUDA_CHECK(cudaEventDestroy(done)); CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device)); CUDA_CHECK(cudaFreeHost(output)); CUDA_CHECK(cudaFreeHost(input));
    std::cout << "PASS completed async H2D/kernel/D2H and immutable input.\n";
    return EXIT_SUCCESS;
}
