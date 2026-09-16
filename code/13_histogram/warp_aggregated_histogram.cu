#include <cuda_runtime.h>

#include "../common/cuda_check.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <system_error>
#include <vector>

constexpr int kBins = 256;

__global__ void histogram_one_atomic_per_thread(const std::uint8_t* input,
                                                int n,
                                                unsigned int* bins,
                                                unsigned int* bin_atomic_calls) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= n) return;
    atomicAdd(&bins[input[index]], 1u);
    atomicAdd(bin_atomic_calls, 1u);
}

__global__ void histogram_warp_aggregated(const std::uint8_t* input,
                                          int n,
                                          unsigned int* bins,
                                          unsigned int* bin_atomic_calls) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const bool valid = index < n;
    const unsigned int valid_mask = __ballot_sync(0xffffffffu, valid);
    if (!valid) return;

    const unsigned int bin = input[index];
    const unsigned int same_bin = __match_any_sync(valid_mask, bin);
    const int lane = static_cast<int>(threadIdx.x % warpSize);
    const int leader = __ffs(static_cast<int>(same_bin)) - 1;
    if (lane == leader) {
        atomicAdd(&bins[bin], static_cast<unsigned int>(__popc(same_bin)));
        atomicAdd(bin_atomic_calls, 1u);
    }
}

enum class Pattern { Cycle, Hot, Skew };

const char* pattern_name(Pattern pattern) {
    switch (pattern) {
        case Pattern::Cycle: return "cycle";
        case Pattern::Hot: return "hot";
        case Pattern::Skew: return "skew";
    }
    return "unknown";
}

std::vector<std::uint8_t> make_input(int n, Pattern pattern) {
    std::vector<std::uint8_t> input(n);
    for (int i = 0; i < n; ++i) {
        if (pattern == Pattern::Hot) input[i] = 0;
        else if (pattern == Pattern::Skew && i % 10 < 8) input[i] = 0;
        else input[i] = static_cast<std::uint8_t>(i % kBins);
    }
    return input;
}

std::array<unsigned int, kBins> reference_histogram(
    const std::vector<std::uint8_t>& input) {
    std::array<unsigned int, kBins> expected{};
    for (std::uint8_t value : input) ++expected[value];
    return expected;
}

unsigned int expected_warp_groups(const std::vector<std::uint8_t>& input) {
    unsigned int groups = 0;
    for (std::size_t first = 0; first < input.size(); first += 32) {
        bool seen[kBins]{};
        const std::size_t last = std::min(first + 32, input.size());
        for (std::size_t i = first; i < last; ++i) seen[input[i]] = true;
        for (bool present : seen) groups += present ? 1u : 0u;
    }
    return groups;
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

int main(int argc, char** argv) {
    int n = 1031;
    int threads = 128;
    if (argc != 1 &&
        (argc != 3 ||
         !parse_integer(argv[1], 1, 1 << 26, n) ||
         !parse_integer(argv[2], 32, 1024, threads) || threads % 32 != 0)) {
        std::cerr << "Usage: warp_aggregated_histogram [n threads_multiple_of_32]\n";
        return 2;
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 7) {
        std::cerr << "__match_any_sync requires compute capability 7.0 or newer\n";
        return 77;
    }
    const int blocks = (n + threads - 1) / threads;
    if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0] ||
        blocks > prop.maxGridSize[0]) {
        std::cerr << "launch dimensions exceed this device's limits\n";
        return 2;
    }

    std::uint8_t* device_input = nullptr;
    unsigned int* device_bins = nullptr;
    unsigned int* device_calls = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input,
                          static_cast<std::size_t>(n) * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMalloc(&device_bins, kBins * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&device_calls, sizeof(unsigned int)));

    bool all_ok = true;
    for (Pattern pattern : {Pattern::Cycle, Pattern::Hot, Pattern::Skew}) {
        const auto input = make_input(n, pattern);
        const auto expected = reference_histogram(input);
        const unsigned int expected_group_calls = expected_warp_groups(input);
        CUDA_CHECK(cudaMemcpy(device_input, input.data(), input.size(),
                              cudaMemcpyHostToDevice));

        std::array<unsigned int, kBins> baseline{};
        unsigned int baseline_calls = 0;
        CUDA_CHECK(cudaMemset(device_bins, 0, kBins * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(device_calls, 0, sizeof(unsigned int)));
        histogram_one_atomic_per_thread<<<blocks, threads>>>(
            device_input, n, device_bins, device_calls);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(baseline.data(), device_bins,
                              kBins * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&baseline_calls, device_calls,
                              sizeof(unsigned int), cudaMemcpyDeviceToHost));

        std::array<unsigned int, kBins> aggregated{};
        unsigned int aggregated_calls = 0;
        CUDA_CHECK(cudaMemset(device_bins, 0, kBins * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(device_calls, 0, sizeof(unsigned int)));
        histogram_warp_aggregated<<<blocks, threads>>>(
            device_input, n, device_bins, device_calls);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(aggregated.data(), device_bins,
                              kBins * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&aggregated_calls, device_calls,
                              sizeof(unsigned int), cudaMemcpyDeviceToHost));

        const bool ok = baseline == expected && aggregated == expected &&
                        baseline_calls == static_cast<unsigned int>(n) &&
                        aggregated_calls == expected_group_calls;
        all_ok = all_ok && ok;
        const double reduction =
            static_cast<double>(baseline_calls) / aggregated_calls;
        std::cout << std::left << std::setw(6) << pattern_name(pattern)
                  << " baseline_bin_atomics=" << baseline_calls
                  << " aggregated_bin_atomics=" << aggregated_calls
                  << " reduction=" << std::fixed << std::setprecision(2)
                  << reduction << "x " << (ok ? "PASS" : "FAIL") << '\n';
    }

    CUDA_CHECK(cudaFree(device_calls));
    CUDA_CHECK(cudaFree(device_bins));
    CUDA_CHECK(cudaFree(device_input));
    if (!all_ok) return 1;
    std::cout << "PASS both kernels match the CPU histogram and call-count model\n";
    return 0;
}
