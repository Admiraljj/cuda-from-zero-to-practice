#include <cuda_runtime.h>

#include "../common/cuda_check.h"

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <system_error>
#include <vector>

constexpr int kBins = 256;

enum class Pattern {
    Cycle,
    Hot,
    Skew
};

__global__ void histogram_shared(const unsigned char* input,
                                 unsigned int* bins,
                                 std::size_t n) {
    __shared__ unsigned int local_bins[kBins];

    for (int bin = threadIdx.x; bin < kBins; bin += blockDim.x) {
        local_bins[bin] = 0;
    }
    __syncthreads();

    const std::size_t start =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = start; i < n; i += step) {
        atomicAdd(&local_bins[input[i]], 1u);
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < kBins; bin += blockDim.x) {
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
    return static_cast<unsigned char>(i % kBins);
}

int main(int argc, char** argv) {
    int n = 1031;
    int threads = 128;
    int blocks = 4;
    Pattern pattern = Pattern::Cycle;
    if (argc != 1 &&
        (argc != 5 ||
         !parse_integer(argv[1], 0, 1 << 26, n) ||
         !parse_integer(argv[2], 1, 1024, threads) ||
         !parse_integer(argv[3], 1, 65535, blocks) ||
         !parse_pattern(argv[4], pattern))) {
        std::cerr << "Usage: shared_histogram [n threads blocks pattern]\n"
                     "  pattern: cycle | hot | skew\n";
        return 2;
    }
    if (n == 0) {
        std::cout << "EMPTY shared histogram total=0; no CUDA calls\n";
        return 0;
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    constexpr std::size_t shared_bytes = kBins * sizeof(unsigned int);
    if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0] ||
        blocks > prop.maxGridSize[0] || shared_bytes > prop.sharedMemPerBlock) {
        std::cerr << "configuration exceeds this device's limits\n";
        return 2;
    }

    const std::size_t count = static_cast<std::size_t>(n);
    std::vector<unsigned char> input(count);
    std::array<unsigned int, kBins> expected{};
    std::array<unsigned int, kBins> actual{};
    for (std::size_t i = 0; i < count; ++i) {
        input[i] = make_value(i, pattern);
        ++expected[input[i]];
    }

    unsigned char* device_input = nullptr;
    unsigned int* device_bins = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&device_bins, actual.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(),
                          count * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_bins, 0,
                          actual.size() * sizeof(unsigned int)));

    histogram_shared<<<blocks, threads>>>(device_input, device_bins, count);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(actual.data(), device_bins,
                          actual.size() * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_bins));
    CUDA_CHECK(cudaFree(device_input));

    std::uint64_t total = 0;
    for (int bin = 0; bin < kBins; ++bin) {
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
              << " blocks=" << blocks << " threads=" << threads
              << " pattern=" << pattern_name(pattern)
              << " shared_bytes_per_block=" << shared_bytes
              << " total=" << total << '\n';
    for (int bin = 0; bin < 8; ++bin) {
        std::cout << "bin[" << bin << "]=" << actual[bin]
                  << (bin == 7 ? '\n' : ' ');
    }
    std::cout << "PASS block-private shared histogram; all 256 bins checked\n";

    return 0;
}
