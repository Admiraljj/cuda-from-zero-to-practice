#include "../common/cuda_check.h"
#include "library_reference.h"
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <exception>

struct Square {
    __host__ __device__ long long operator()(long long x) const { return x * x; }
};
int main(int argc, char** argv) {
    try {
        int n = 1031;
        if (argc > 2 || (argc > 1 && !library_example::integer(argv[1], 0, 16777216, n))) {
            std::cerr << "Usage: thrust_fancy_reduce [n:0..16777216]\n"; return EXIT_FAILURE;
        }
        if (n == 0) { std::cout << "PASS empty input, application sum_squares=0.\n"; return EXIT_SUCCESS; }
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(long long);
        std::vector<long long> input(n), after(n); long long expected = 0;
        for (int i = 0; i < n; ++i) { input[i] = library_example::integer_value(i,"mixed"); expected += input[i]*input[i]; }
        long long* device = nullptr; CUDA_CHECK(cudaMalloc(&device, bytes));
        CUDA_CHECK(cudaMemcpy(device, input.data(), bytes, cudaMemcpyHostToDevice));
        cudaStream_t stream{}; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        auto begin = thrust::device_pointer_cast(device);
        const auto policy = thrust::cuda::par.on(stream);
        const long long a = thrust::transform_reduce(policy,begin,begin+n,Square{},0LL,thrust::plus<long long>{});
        const auto transformed = thrust::make_transform_iterator(begin,Square{});
        const long long b = thrust::reduce(policy,transformed,transformed+n,0LL,thrust::plus<long long>{});
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(after.data(),device,bytes,cudaMemcpyDeviceToHost));
        if (a != expected || b != expected || after != input) return EXIT_FAILURE;
        CUDA_CHECK(cudaStreamDestroy(stream)); CUDA_CHECK(cudaFree(device));
        std::cout << "sum_squares=" << a << " expected=" << expected
                  << "\nPASS transformed values without a user-allocated square array.\n"; return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Thrust/C++ exception: " << error.what() << '\n'; return EXIT_FAILURE;
    }
}
