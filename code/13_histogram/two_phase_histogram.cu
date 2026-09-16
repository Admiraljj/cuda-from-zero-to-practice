#include <cuda_runtime.h>

#include "../common/cuda_check.h"

#include <algorithm>
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

__global__ void build_partial_histograms(const unsigned char* input,
                                         unsigned int* partials,
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
        partials[static_cast<std::size_t>(blockIdx.x) * kBins + bin] =
            local_bins[bin];
    }
}

__global__ void merge_partial_histograms(const unsigned int* partials,
                                         unsigned int* bins,
                                         int partial_count) {
    for (int bin = threadIdx.x; bin < kBins; bin += blockDim.x) {
        unsigned int sum = 0;
        for (int row = 0; row < partial_count; ++row) {
            sum += partials[static_cast<std::size_t>(row) * kBins + bin];
        }
        bins[bin] = sum;
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
         !parse_integer(argv[3], 1, 4096, blocks) ||
         !parse_pattern(argv[4], pattern))) {
        std::cerr << "Usage: two_phase_histogram [n threads blocks pattern]\n"
                     "  pattern: cycle | hot | skew\n";
        return 2;
    }
    if (n == 0) {
        std::cout << "EMPTY two-phase histogram total=0; no CUDA calls\n";
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
    const int merge_threads =
        std::min(kBins, std::min(prop.maxThreadsPerBlock, prop.maxThreadsDim[0]));

    const std::size_t count = static_cast<std::size_t>(n);
    const std::size_t partial_elements =
        static_cast<std::size_t>(blocks) * kBins;
    const std::size_t partial_bytes =
        partial_elements * sizeof(unsigned int);

    std::vector<unsigned char> input(count);
    std::vector<unsigned int> expected_partials(partial_elements, 0);
    std::vector<unsigned int> actual_partials(partial_elements, 0);
    std::array<unsigned int, kBins> expected{};
    std::array<unsigned int, kBins> actual{};

    const std::size_t grid_threads =
        static_cast<std::size_t>(blocks) * threads;
    for (std::size_t i = 0; i < count; ++i) {
        const unsigned char value = make_value(i, pattern);
        input[i] = value;
        ++expected[value];
        const std::size_t owner_block = (i % grid_threads) / threads;
        ++expected_partials[owner_block * kBins + value];
    }

    unsigned char* device_input = nullptr;
    unsigned int* device_partials = nullptr;
    unsigned int* device_bins = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&device_partials, partial_bytes));
    CUDA_CHECK(cudaMalloc(&device_bins, actual.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(),
                          count * sizeof(unsigned char), cudaMemcpyHostToDevice));

    build_partial_histograms<<<blocks, threads>>>(
        device_input, device_partials, count);
    CUDA_CHECK(cudaGetLastError());
    merge_partial_histograms<<<1, merge_threads>>>(
        device_partials, device_bins, blocks);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(actual_partials.data(), device_partials, partial_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual.data(), device_bins,
                          actual.size() * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_bins));
    CUDA_CHECK(cudaFree(device_partials));
    CUDA_CHECK(cudaFree(device_input));

    std::uint64_t partial_total = 0;
    for (std::size_t index = 0; index < partial_elements; ++index) {
        partial_total += actual_partials[index];
        if (actual_partials[index] != expected_partials[index]) {
            const std::size_t row = index / kBins;
            const std::size_t bin = index % kBins;
            std::cerr << "partial mismatch row=" << row << " bin=" << bin
                      << " actual=" << actual_partials[index]
                      << " expected=" << expected_partials[index] << '\n';
            return 1;
        }
    }

    std::uint64_t final_total = 0;
    for (int bin = 0; bin < kBins; ++bin) {
        final_total += actual[bin];
        if (actual[bin] != expected[bin]) {
            std::cerr << "final mismatch bin=" << bin
                      << " actual=" << actual[bin]
                      << " expected=" << expected[bin] << '\n';
            return 1;
        }
    }
    if (partial_total != count || final_total != count) {
        std::cerr << "total mismatch: partial=" << partial_total
                  << " final=" << final_total
                  << " expected=" << count << '\n';
        return 1;
    }

    std::cout << "device=" << prop.name << " n=" << n
              << " blocks=" << blocks << " threads=" << threads
              << " pattern=" << pattern_name(pattern)
              << " partial_shape=" << blocks << 'x' << kBins
              << " partial_bytes=" << partial_bytes << '\n';
    const int rows_to_print = std::min(blocks, 4);
    for (int row = 0; row < rows_to_print; ++row) {
        std::cout << "partial[" << row << "][0..7]=";
        for (int bin = 0; bin < 8; ++bin) {
            std::cout << actual_partials[static_cast<std::size_t>(row) * kBins + bin]
                      << (bin == 7 ? '\n' : ',');
        }
    }
    std::cout << "partial_total=" << partial_total
              << " final_total=" << final_total << '\n';
    std::cout << "PASS every partial cell, final bin, and total checked\n";
    return 0;
}
