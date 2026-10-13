#include "../common/cuda_check.h"
#include "library_reference.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <exception>

int main(int argc, char** argv) {
    try {
        int n = 1031;
        if (argc > 2 || (argc > 1 && !library_example::integer(argv[1], 0, 1048576, n))) {
            std::cerr << "Usage: thrust_vectors [n:0..1048576]\n"; return EXIT_FAILURE;
        }
        if (n == 0) { std::cout << "PASS empty range, application sum=0; no GPU call.\n"; return EXIT_SUCCESS; }
        std::vector<long long> original(n);
        long long want_sum = 0;
        for (int i = 0; i < n; ++i) { original[i] = library_example::integer_value(i, "mixed"); want_sum += original[i]; }
        auto expected_sorted = original; std::sort(expected_sorted.begin(), expected_sorted.end());
        cudaStream_t stream{}; CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        {
            thrust::device_vector<long long> data(original.begin(), original.end());
            // Container construction is outside this custom-stream workflow; establish completion explicitly.
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto policy = thrust::cuda::par.on(stream);
            const long long result = thrust::reduce(policy, data.begin(), data.end(), 0LL, thrust::plus<long long>{});
            thrust::sort(policy, data.begin(), data.end());
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
            thrust::host_vector<long long> sorted = data;
            if (result != want_sum || sorted.size() != expected_sorted.size()) return EXIT_FAILURE;
            for (std::size_t i = 0; i < sorted.size(); ++i) if (sorted[i] != expected_sorted[i]) return EXIT_FAILURE;
            if (n <= 16) { for (long long v : sorted) std::cout << v << ' '; std::cout << '\n'; }
            std::cout << "sum=" << result << " sorted_count=" << sorted.size() << '\n';
        } // device_vector owns its allocation and releases it; no cudaFree on its pointer.
        CUDA_CHECK(cudaStreamDestroy(stream));
        std::cout << "PASS host/device containers, whole-range sum and sort.\n"; return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Thrust/C++ exception: " << error.what() << '\n'; return EXIT_FAILURE;
    }
}
