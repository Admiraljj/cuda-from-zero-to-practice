#include <cuda_runtime.h>

#include "../common/cuda_check.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <system_error>
#include <charconv>
#include <vector>

enum class Pattern {
    Cycle,
    Hot,
    Skew
};

__host__ __device__ int bin_for_value(unsigned char value, int bin_count) {
    return (static_cast<unsigned int>(value) * bin_count) / 256u;
}

__global__ void histogram_dynamic_bins(const unsigned char* input,
                                       unsigned int* bins,
                                       std::size_t n,
                                       int bin_count) {
    extern __shared__ unsigned int local_bins[];

    for (int bin = threadIdx.x; bin < bin_count; bin += blockDim.x) {
        local_bins[bin] = 0;
    }
    __syncthreads();

    const std::size_t start =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = start; i < n; i += step) {
        const int bin = bin_for_value(input[i], bin_count);
        atomicAdd(&local_bins[bin], 1u);
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < bin_count; bin += blockDim.x) {
        const unsigned int count = local_bins[bin];
        if (count != 0) atomicAdd(&bins[bin], count);
    }
}

bool parse_integer(const char* text, int minimum, int maximum, int& value) {
    if (text == nullptr) return false;
    const std::string_view view(text);
    int parsed = 0;
    const auto result = std::from_chars(view.data(), view.data() + view.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != view.data() + view.size() ||
        parsed < minimum || parsed > maximum) return false;
    value = parsed;
    return true;
}

bool parse_pattern(const char* text, Pattern& pattern) {
    const std::string_view view(text ? text : "");
    if (view == "cycle") pattern = Pattern::Cycle;
    else if (view == "hot") pattern = Pattern::Hot;
    else if (view == "skew") pattern = Pattern::Skew;
    else return false;
    return true;
}

const char* pattern_name(Pattern pattern) {
    if (pattern == Pattern::Cycle) return "cycle";
    if (pattern == Pattern::Hot) return "hot";
    return "skew";
}

unsigned char make_value(std::size_t i, Pattern pattern) {
    if (pattern == Pattern::Hot) return 0;
    if (pattern == Pattern::Skew && i % 10 < 8) return 0;
    return static_cast<unsigned char>(i % 256);
}

int main(int argc, char** argv) {
    int n = 1031;
    int bin_count = 16;
    int threads = 128;
    int blocks = 4;
    Pattern pattern = Pattern::Cycle;
    if (argc != 1 &&
        (argc != 6 ||
         !parse_integer(argv[1], 0, 1 << 26, n) ||
         !parse_integer(argv[2], 16, 256, bin_count) ||
         (bin_count != 16 && bin_count != 256) ||
         !parse_integer(argv[3], 1, 1024, threads) ||
         !parse_integer(argv[4], 1, 65535, blocks) ||
         !parse_pattern(argv[5], pattern))) {
        std::cerr << "Usage: dynamic_bins_histogram "
                     "[n bins threads blocks pattern]\n"
                     "  bins: 16 | 256\n"
                     "  pattern: cycle | hot | skew\n";
        return 2;
    }
    if (n == 0) {
        std::cout << "EMPTY dynamic-bin histogram bins=" << bin_count
                  << " total=0; no CUDA calls\n";
        return 0;
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const std::size_t shared_bytes =
        static_cast<std::size_t>(bin_count) * sizeof(unsigned int);
    if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0] ||
        blocks > prop.maxGridSize[0] || shared_bytes > prop.sharedMemPerBlock) {
        std::cerr << "configuration exceeds this device's limits\n";
        return 2;
    }

    const std::size_t count = static_cast<std::size_t>(n);
    const std::size_t bins_bytes =
        static_cast<std::size_t>(bin_count) * sizeof(unsigned int);
    std::vector<unsigned char> input(count);
    std::vector<unsigned int> expected(bin_count, 0);
    std::vector<unsigned int> actual(bin_count, 0);
    for (std::size_t i = 0; i < count; ++i) {
        input[i] = make_value(i, pattern);
        ++expected[bin_for_value(input[i], bin_count)];
    }

    unsigned char* device_input = nullptr;
    unsigned int* device_bins = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&device_bins, bins_bytes));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(),
                          count * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_bins, 0, bins_bytes));

    histogram_dynamic_bins<<<blocks, threads, shared_bytes>>>(
        device_input, device_bins, count, bin_count);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(actual.data(), device_bins, bins_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_bins));
    CUDA_CHECK(cudaFree(device_input));

    std::uint64_t total = 0;
    for (int bin = 0; bin < bin_count; ++bin) {
        total += actual[bin];
        if (actual[bin] != expected[bin]) {
            std::cerr << "bin " << bin << " mismatch: actual=" << actual[bin]
                      << " expected=" << expected[bin] << '\n';
            return 1;
        }
    }
    if (total != count) {
        std::cerr << "total mismatch: actual=" << total
                  << " expected=" << count << '\n';
        return 1;
    }

    std::cout << "device=" << prop.name << " n=" << n
              << " bins=" << bin_count
              << " threads=" << threads << " blocks=" << blocks
              << " pattern=" << pattern_name(pattern)
              << " dynamic_shared_bytes_per_block=" << shared_bytes
              << " total=" << total << '\n';
    const int shown = bin_count == 16 ? 16 : 8;
    for (int bin = 0; bin < shown; ++bin) {
        std::cout << "bin[" << bin << "]=" << actual[bin]
                  << (bin + 1 == shown ? '\n' : ' ');
    }
    if (bin_count > shown) {
        std::cout << "... bin[255]=" << actual[255] << '\n';
    }
    std::cout << "PASS dynamic-bin shared histogram; every bin checked\n";
    return 0;
}
