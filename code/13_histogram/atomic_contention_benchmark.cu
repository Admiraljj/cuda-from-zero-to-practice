#include <cuda_runtime.h>

#include "../common/cuda_check.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <system_error>
#include <string_view>
#include <vector>

constexpr int kBins = 256;

enum class Pattern : int {
    Cycle = 0,
    Hot = 1,
    Skew = 2
};

__global__ void histogram_global(const unsigned char* input,
                                 unsigned int* bins,
                                 std::size_t n) {
    const std::size_t start =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t i = start; i < n; i += step) {
        atomicAdd(&bins[input[i]], 1u);
    }
}

bool parse_positive(const char* text, int minimum, int maximum, int& value) {
    if (text == nullptr) return false;
    const std::string_view view(text);
    int parsed = 0;
    const auto result = std::from_chars(view.data(), view.data() + view.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != view.data() + view.size() ||
        parsed < minimum || parsed > maximum) return false;
    value = parsed;
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

float median(std::vector<float> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    if (samples.size() % 2 == 1) return samples[middle];
    return (samples[middle - 1] + samples[middle]) * 0.5f;
}

bool verify(Pattern pattern,
            const std::vector<unsigned char>& host_input,
            const std::array<unsigned int, kBins>& actual) {
    std::array<unsigned int, kBins> expected{};
    for (unsigned char value : host_input) ++expected[value];

    std::uint64_t total = 0;
    for (int bin = 0; bin < kBins; ++bin) {
        total += actual[bin];
        if (actual[bin] != expected[bin]) {
            std::cerr << pattern_name(pattern) << " mismatch at bin " << bin
                      << ": actual=" << actual[bin]
                      << " expected=" << expected[bin] << '\n';
            return false;
        }
    }
    if (total != host_input.size()) {
        std::cerr << pattern_name(pattern) << " total mismatch: actual=" << total
                  << " expected=" << host_input.size() << '\n';
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    int n = 1 << 24;
    int repeats = 12;
    int threads = 256;
    if (argc != 1 &&
        (argc != 4 ||
         !parse_positive(argv[1], 1, 1 << 28, n) ||
         !parse_positive(argv[2], 3, 100, repeats) ||
         !parse_positive(argv[3], 1, 1024, threads))) {
        std::cerr << "Usage: atomic_contention_benchmark [n repeats threads]\n";
        return 2;
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0]) {
        std::cerr << "threads exceeds this device's one-dimensional block limit\n";
        return 2;
    }

    const std::size_t count = static_cast<std::size_t>(n);
    const std::size_t input_bytes = count * sizeof(unsigned char);
    const std::size_t bins_bytes = kBins * sizeof(unsigned int);
    const std::size_t blocks_needed =
        (count + static_cast<std::size_t>(threads) - 1) / threads;
    const std::size_t blocks_target =
        static_cast<std::size_t>(prop.multiProcessorCount) * 8;
    const int blocks = static_cast<int>(
        std::min(blocks_needed, std::min(blocks_target,
                 static_cast<std::size_t>(prop.maxGridSize[0]))));

    constexpr std::array<Pattern, 3> patterns = {
        Pattern::Cycle, Pattern::Hot, Pattern::Skew
    };
    std::array<std::vector<unsigned char>, 3> host_inputs;
    std::array<unsigned char*, 3> device_inputs{};
    unsigned int* device_bins = nullptr;

    for (int p = 0; p < 3; ++p) {
        host_inputs[p].resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            host_inputs[p][i] = make_value(i, patterns[p]);
        }
        CUDA_CHECK(cudaMalloc(&device_inputs[p], input_bytes));
        CUDA_CHECK(cudaMemcpy(device_inputs[p], host_inputs[p].data(), input_bytes,
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&device_bins, bins_bytes));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int p = 0; p < 3; ++p) {
        CUDA_CHECK(cudaMemset(device_bins, 0, bins_bytes));
        histogram_global<<<blocks, threads>>>(device_inputs[p], device_bins, count);
        CUDA_CHECK_KERNEL();
    }

    std::array<std::vector<float>, 3> samples;
    for (auto& pattern_samples : samples) pattern_samples.reserve(repeats);

    for (int round = 0; round < repeats; ++round) {
        for (int position = 0; position < 3; ++position) {
            const int p = (round + position) % 3;
            CUDA_CHECK(cudaMemsetAsync(device_bins, 0, bins_bytes));
            CUDA_CHECK(cudaEventRecord(start));
            histogram_global<<<blocks, threads>>>(device_inputs[p], device_bins, count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            samples[p].push_back(elapsed_ms);
        }
    }

    std::array<unsigned int, kBins> actual{};
    for (int p = 0; p < 3; ++p) {
        CUDA_CHECK(cudaMemset(device_bins, 0, bins_bytes));
        histogram_global<<<blocks, threads>>>(device_inputs[p], device_bins, count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), device_bins, bins_bytes,
                              cudaMemcpyDeviceToHost));
        if (!verify(patterns[p], host_inputs[p], actual)) return 1;
    }

    std::cout << "device=" << prop.name << " n=" << n
              << " blocks=" << blocks << " threads=" << threads
              << " repeats=" << repeats << '\n';
    std::cout << std::fixed << std::setprecision(4);
    for (int p = 0; p < 3; ++p) {
        const auto [minimum, maximum] =
            std::minmax_element(samples[p].begin(), samples[p].end());
        std::cout << pattern_name(patterns[p])
                  << " median_ms=" << median(samples[p])
                  << " min_ms=" << *minimum
                  << " max_ms=" << *maximum << '\n';
    }
    std::cout << "PASS all bins checked; compare distributions only on this recorded setup\n";

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(device_bins));
    for (unsigned char* pointer : device_inputs) CUDA_CHECK(cudaFree(pointer));
    return 0;
}
