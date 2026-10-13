#include "../common/cuda_check.h"
#include "library_reference.h"
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>
#include <exception>

struct Saxpy {
    float alpha;
    __host__ __device__ float operator()(float x, float y) const { return alpha * x + y; }
};
__global__ void saxpy_kernel(const float* x, const float* y, float* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 2.0F * x[i] + y[i];
}
int main(int argc, char** argv) {
    try {
        int n = 1031;
        if (argc > 2 || (argc > 1 && !library_example::integer(argv[1], 0, 16777216, n))) {
            std::cerr << "Usage: thrust_saxpy [n:0..16777216]\n"; return EXIT_FAILURE;
        }
        if (n == 0) { std::cout << "PASS empty input; no GPU call.\n"; return EXIT_SUCCESS; }
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
        std::vector<float> x(n), y(n), output(n), after(n);
        for (int i = 0; i < n; ++i) { x[i] = library_example::x_value(i); y[i] = library_example::y_value(i); }
        float *dx = nullptr, *dy = nullptr, *custom = nullptr, *library = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, bytes)); CUDA_CHECK(cudaMalloc(&dy, bytes));
        CUDA_CHECK(cudaMalloc(&custom, bytes)); CUDA_CHECK(cudaMalloc(&library, bytes));
        CUDA_CHECK(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dy, y.data(), bytes, cudaMemcpyHostToDevice));
        cudaStream_t stream{}; CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMemsetAsync(custom, 0xFF, bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(library, 0xFF, bytes, stream));
        saxpy_kernel<<<(n+255)/256,256,0,stream>>>(dx, dy, custom, n); CUDA_CHECK(cudaGetLastError());
        const auto begin_x = thrust::device_pointer_cast(dx);
        const auto begin_y = thrust::device_pointer_cast(dy);
        const auto begin_out = thrust::device_pointer_cast(library);
        thrust::transform(thrust::cuda::par.on(stream), begin_x, begin_x+n, begin_y, begin_out, Saxpy{2.0F});
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(output.data(), custom, bytes, cudaMemcpyDeviceToHost));
        if (!library_example::saxpy_valid(output, "custom")) return EXIT_FAILURE;
        CUDA_CHECK(cudaMemcpy(output.data(), library, bytes, cudaMemcpyDeviceToHost));
        if (!library_example::saxpy_valid(output, "thrust")) return EXIT_FAILURE;
        CUDA_CHECK(cudaMemcpy(after.data(), dx, bytes, cudaMemcpyDeviceToHost)); if (after != x) return EXIT_FAILURE;
        CUDA_CHECK(cudaMemcpy(after.data(), dy, bytes, cudaMemcpyDeviceToHost)); if (after != y) return EXIT_FAILURE;
        CUDA_CHECK(cudaStreamDestroy(stream)); CUDA_CHECK(cudaFree(library)); CUDA_CHECK(cudaFree(custom));
        CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dx));
        std::cout << "PASS custom/Thrust SAXPY, both inputs unchanged.\n"; return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Thrust/C++ exception: " << error.what() << '\n'; return EXIT_FAILURE;
    }
}
