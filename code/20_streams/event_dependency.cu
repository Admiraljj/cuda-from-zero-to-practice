#include "../common/cuda_check.h"
#include "../17_convolution/filter_reference.h"
#include <cstdlib>

__global__ void produce(float* values, float value, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] = value;
}
__global__ void merge(const float* a, const float* b, float* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = a[i] + b[i];
}
int main(int argc, char** argv) {
    int n = 1031;
    if (argc > 2 || (argc > 1 && !filter_example::integer(argv[1], 0, 16777216, n))) {
        std::cerr << "Usage: event_dependency [n:0..16777216]\n"; return EXIT_FAILURE;
    }
    if (n == 0) { std::cout << "PASS empty input; no GPU launch.\n"; return EXIT_SUCCESS; }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    float *a = nullptr, *b = nullptr, *c = nullptr;
    CUDA_CHECK(cudaMalloc(&a, bytes)); CUDA_CHECK(cudaMalloc(&b, bytes)); CUDA_CHECK(cudaMalloc(&c, bytes));
    CUDA_CHECK(cudaMemset(a, 0xFF, bytes)); CUDA_CHECK(cudaMemset(b, 0xFF, bytes)); CUDA_CHECK(cudaMemset(c, 0xFF, bytes));
    // Complete default-stream initialization before using non-blocking workers.
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaStream_t first{}, second{}, consumer{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
    cudaEvent_t ready_a{}, ready_b{}, done{};
    CUDA_CHECK(cudaEventCreateWithFlags(&ready_a, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ready_b, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    const int blocks = (n + 255) / 256;
    produce<<<blocks, 256, 0, first>>>(a, 1.0F, n); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ready_a, first));
    produce<<<blocks, 256, 0, second>>>(b, 2.0F, n); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ready_b, second));
    CUDA_CHECK(cudaStreamWaitEvent(consumer, ready_a, 0));
    CUDA_CHECK(cudaStreamWaitEvent(consumer, ready_b, 0));
    merge<<<blocks, 256, 0, consumer>>>(a, b, c, n); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(done, consumer));
    CUDA_CHECK(cudaEventSynchronize(done));
    std::vector<float> output(n), after_a(n), after_b(n);
    CUDA_CHECK(cudaMemcpy(output.data(), c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(after_a.data(), a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(after_b.data(), b, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (output[i] != 3.0F || after_a[i] != 1.0F || after_b[i] != 2.0F) {
            std::cerr << "FAIL index=" << i << '\n'; return EXIT_FAILURE;
        }
    }
    CUDA_CHECK(cudaEventDestroy(done)); CUDA_CHECK(cudaEventDestroy(ready_b)); CUDA_CHECK(cudaEventDestroy(ready_a));
    CUDA_CHECK(cudaStreamDestroy(consumer)); CUDA_CHECK(cudaStreamDestroy(second)); CUDA_CHECK(cudaStreamDestroy(first));
    CUDA_CHECK(cudaFree(c)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(a));
    std::cout << "PASS consumer waits for both producers; a+b=3 everywhere.\n";
    return EXIT_SUCCESS;
}
