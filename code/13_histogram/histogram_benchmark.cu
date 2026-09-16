#include <cuda_runtime.h>

#include "../common/cuda_check.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>
#include <system_error>
#include <vector>

constexpr int kBins = 256;
constexpr int kMethodCount = 4;
constexpr int kPatternCount = 3;

enum class Method : int { Global = 0, Shared = 1, TwoPhase = 2, Warp = 3 };
enum class Pattern : int { Cycle = 0, Hot = 1, Skew = 2 };

__global__ void histogram_global(const std::uint8_t* input,
                                 std::size_t n,
                                 unsigned int* bins) {
    const std::size_t start =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = start; i < n; i += step) {
        atomicAdd(&bins[input[i]], 1u);
    }
}

__global__ void histogram_shared(const std::uint8_t* input,
                                 std::size_t n,
                                 unsigned int* bins) {
    extern __shared__ unsigned int local_bins[];
    for (int bin = static_cast<int>(threadIdx.x); bin < kBins;
         bin += static_cast<int>(blockDim.x)) {
        local_bins[bin] = 0;
    }
    __syncthreads();

    const std::size_t start =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = start; i < n; i += step) {
        atomicAdd(&local_bins[input[i]], 1u);
    }
    __syncthreads();

    for (int bin = static_cast<int>(threadIdx.x); bin < kBins;
         bin += static_cast<int>(blockDim.x)) {
        const unsigned int count = local_bins[bin];
        if (count != 0) atomicAdd(&bins[bin], count);
    }
}

__global__ void build_partials(const std::uint8_t* input,
                               std::size_t n,
                               unsigned int* partials) {
    extern __shared__ unsigned int local_bins[];
    for (int bin = static_cast<int>(threadIdx.x); bin < kBins;
         bin += static_cast<int>(blockDim.x)) {
        local_bins[bin] = 0;
    }
    __syncthreads();

    const std::size_t start =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = start; i < n; i += step) {
        atomicAdd(&local_bins[input[i]], 1u);
    }
    __syncthreads();

    for (int bin = static_cast<int>(threadIdx.x); bin < kBins;
         bin += static_cast<int>(blockDim.x)) {
        partials[static_cast<std::size_t>(blockIdx.x) * kBins + bin] =
            local_bins[bin];
    }
}

__global__ void merge_partials(const unsigned int* partials,
                               int partial_count,
                               unsigned int* bins) {
    const int bin = static_cast<int>(threadIdx.x);
    if (bin >= kBins) return;
    unsigned int sum = 0;
    for (int partial = 0; partial < partial_count; ++partial) {
        sum += partials[static_cast<std::size_t>(partial) * kBins + bin];
    }
    bins[bin] = sum;
}

__global__ void histogram_warp_aggregated(const std::uint8_t* input,
                                          std::size_t n,
                                          unsigned int* bins) {
    const unsigned int lane = threadIdx.x % warpSize;
    const std::size_t global_thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t warp_first = global_thread - lane;
    const std::size_t step =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t first = warp_first; first < n; first += step) {
        const std::size_t index = first + lane;
        const bool valid = index < n;
        const unsigned int valid_mask = __ballot_sync(0xffffffffu, valid);
        if (valid) {
            const unsigned int bin = input[index];
            const unsigned int same_bin = __match_any_sync(valid_mask, bin);
            const int leader = __ffs(static_cast<int>(same_bin)) - 1;
            if (static_cast<int>(lane) == leader) {
                atomicAdd(&bins[bin],
                          static_cast<unsigned int>(__popc(same_bin)));
            }
        }
    }
}

const char* method_name(Method method) {
    switch (method) {
        case Method::Global: return "global";
        case Method::Shared: return "shared";
        case Method::TwoPhase: return "two_phase";
        case Method::Warp: return "warp";
    }
    return "unknown";
}

const char* pattern_name(Pattern pattern) {
    switch (pattern) {
        case Pattern::Cycle: return "cycle";
        case Pattern::Hot: return "hot";
        case Pattern::Skew: return "skew";
    }
    return "unknown";
}

std::uint8_t make_value(std::size_t i, Pattern pattern) {
    if (pattern == Pattern::Hot) return 0;
    if (pattern == Pattern::Skew && i % 10 < 8) return 0;
    return static_cast<std::uint8_t>(i % kBins);
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

float median(std::vector<float> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    if (samples.size() % 2 == 1) return samples[middle];
    return (samples[middle - 1] + samples[middle]) * 0.5f;
}

bool verify(const std::array<unsigned int, kBins>& actual,
            const std::array<unsigned int, kBins>& expected,
            Pattern pattern,
            Method method,
            int round) {
    std::uint64_t total = 0;
    for (int bin = 0; bin < kBins; ++bin) {
        total += actual[bin];
        if (actual[bin] != expected[bin]) {
            std::cerr << "mismatch pattern=" << pattern_name(pattern)
                      << " method=" << method_name(method)
                      << " round=" << round << " bin=" << bin
                      << " actual=" << actual[bin]
                      << " expected=" << expected[bin] << '\n';
            return false;
        }
    }
    return total == std::accumulate(expected.begin(), expected.end(),
                                    std::uint64_t{0});
}

int main(int argc, char** argv) {
    int n = 1 << 24;
    int repeats = 15;
    int threads = 256;
    int requested_blocks = 0;
    if (argc != 1 &&
        (argc != 5 ||
         !parse_integer(argv[1], 1, 1 << 27, n) ||
         !parse_integer(argv[2], 5, 100, repeats) ||
         !parse_integer(argv[3], 32, 1024, threads) || threads % 32 != 0 ||
         !parse_integer(argv[4], 0, 65535, requested_blocks))) {
        std::cerr << "Usage: histogram_benchmark [n repeats threads blocks_or_0]\n";
        return 2;
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 7) {
        std::cerr << "warp method requires compute capability 7.0 or newer\n";
        return 77;
    }
    if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0]) {
        std::cerr << "threads exceeds this device's one-dimensional block limit\n";
        return 2;
    }
    const std::size_t count = static_cast<std::size_t>(n);
    const std::size_t blocks_needed =
        (count + static_cast<std::size_t>(threads) - 1) / threads;
    const std::size_t default_blocks =
        std::min(blocks_needed,
                 static_cast<std::size_t>(prop.multiProcessorCount) * 8);
    const int blocks = requested_blocks == 0
        ? static_cast<int>(default_blocks)
        : requested_blocks;
    if (blocks < 1 || blocks > prop.maxGridSize[0]) {
        std::cerr << "blocks exceeds this device's grid limit\n";
        return 2;
    }
    const std::size_t shared_bytes = kBins * sizeof(unsigned int);
    if (shared_bytes > prop.sharedMemPerBlock) {
        std::cerr << "256-bin shared histogram exceeds per-block shared memory\n";
        return 2;
    }

    const std::size_t input_bytes = count * sizeof(std::uint8_t);
    const std::size_t bins_bytes = kBins * sizeof(unsigned int);
    const std::size_t partial_count =
        static_cast<std::size_t>(blocks) * kBins;
    const std::size_t partial_bytes = partial_count * sizeof(unsigned int);

    constexpr std::array<Pattern, kPatternCount> patterns = {
        Pattern::Cycle, Pattern::Hot, Pattern::Skew
    };
    constexpr std::array<Method, kMethodCount> methods = {
        Method::Global, Method::Shared, Method::TwoPhase, Method::Warp
    };
    std::array<std::vector<std::uint8_t>, kPatternCount> host_inputs;
    std::array<std::array<unsigned int, kBins>, kPatternCount> expected{};
    std::array<std::uint8_t*, kPatternCount> device_inputs{};
    for (int p = 0; p < kPatternCount; ++p) {
        host_inputs[p].resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            host_inputs[p][i] = make_value(i, patterns[p]);
            ++expected[p][host_inputs[p][i]];
        }
        CUDA_CHECK(cudaMalloc(&device_inputs[p], input_bytes));
        CUDA_CHECK(cudaMemcpy(device_inputs[p], host_inputs[p].data(), input_bytes,
                              cudaMemcpyHostToDevice));
    }

    unsigned int* device_bins = nullptr;
    unsigned int* device_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&device_bins, bins_bytes));
    CUDA_CHECK(cudaMalloc(&device_partials, partial_bytes));

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto launch = [&](Method method, const std::uint8_t* input) {
        switch (method) {
            case Method::Global:
                histogram_global<<<blocks, threads, 0, stream>>>(
                    input, count, device_bins);
                CUDA_CHECK(cudaGetLastError());
                break;
            case Method::Shared:
                histogram_shared<<<blocks, threads, shared_bytes, stream>>>(
                    input, count, device_bins);
                CUDA_CHECK(cudaGetLastError());
                break;
            case Method::TwoPhase:
                build_partials<<<blocks, threads, shared_bytes, stream>>>(
                    input, count, device_partials);
                CUDA_CHECK(cudaGetLastError());
                merge_partials<<<1, kBins, 0, stream>>>(
                    device_partials, blocks, device_bins);
                CUDA_CHECK(cudaGetLastError());
                break;
            case Method::Warp:
                histogram_warp_aggregated<<<blocks, threads, 0, stream>>>(
                    input, count, device_bins);
                CUDA_CHECK(cudaGetLastError());
                break;
        }
    };

    constexpr int warmups = 3;
    for (int warmup = 0; warmup < warmups; ++warmup) {
        for (int p = 0; p < kPatternCount; ++p) {
            for (Method method : methods) {
                CUDA_CHECK(cudaMemsetAsync(device_bins, 0, bins_bytes, stream));
                launch(method, device_inputs[p]);
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::array<std::array<std::vector<float>, kMethodCount>, kPatternCount> samples;
    for (auto& pattern_samples : samples) {
        for (auto& method_samples : pattern_samples) method_samples.reserve(repeats);
    }
    std::array<unsigned int, kBins> actual{};
    for (int round = 0; round < repeats; ++round) {
        for (int pattern_position = 0; pattern_position < kPatternCount;
             ++pattern_position) {
            const int p = (round + pattern_position) % kPatternCount;
            for (int method_position = 0; method_position < kMethodCount;
                 ++method_position) {
                const int m = (round + p + method_position) % kMethodCount;
                const Method method = methods[m];

                CUDA_CHECK(cudaMemsetAsync(device_bins, 0, bins_bytes, stream));
                CUDA_CHECK(cudaEventRecord(start, stream));
                launch(method, device_inputs[p]);
                CUDA_CHECK(cudaEventRecord(stop, stream));
                CUDA_CHECK(cudaEventSynchronize(stop));

                float elapsed_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
                samples[p][m].push_back(elapsed_ms);
                CUDA_CHECK(cudaMemcpy(actual.data(), device_bins, bins_bytes,
                                      cudaMemcpyDeviceToHost));
                if (!verify(actual, expected[p], patterns[p], method, round)) {
                    return 1;
                }
            }
        }
    }

    for (int p = 0; p < kPatternCount; ++p) {
        std::vector<std::uint8_t> copied_back(count);
        CUDA_CHECK(cudaMemcpy(copied_back.data(), device_inputs[p], input_bytes,
                              cudaMemcpyDeviceToHost));
        if (copied_back != host_inputs[p]) {
            std::cerr << "input modified for pattern " << pattern_name(patterns[p])
                      << '\n';
            return 1;
        }
    }

    std::cout << "device=" << prop.name << " cc=" << prop.major << '.' << prop.minor
              << " n=" << n << " blocks=" << blocks << " threads=" << threads
              << " repeats=" << repeats << " warmups=" << warmups
              << " partial_MiB=" << std::fixed << std::setprecision(3)
              << partial_bytes / (1024.0 * 1024.0) << '\n';
    std::cout << "metric=input-resident kernel(s); output memset excluded; "
                 "D2H and validation excluded\n";
    for (int p = 0; p < kPatternCount; ++p) {
        for (int m = 0; m < kMethodCount; ++m) {
            const auto [minimum, maximum] =
                std::minmax_element(samples[p][m].begin(), samples[p][m].end());
            const float middle = median(samples[p][m]);
            const double gitems_per_second = count / (middle * 1.0e6);
            std::cout << std::left << std::setw(6) << pattern_name(patterns[p])
                      << ' ' << std::setw(9) << method_name(methods[m])
                      << " median_ms=" << std::right << std::fixed
                      << std::setprecision(4) << middle
                      << " min_ms=" << *minimum << " max_ms=" << *maximum
                      << " Gitems/s=" << std::setprecision(3)
                      << gitems_per_second << '\n';
        }
    }
    std::cout << "PASS every measured sample matched all 256 CPU bins; "
                 "device inputs remained unchanged\n";

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device_partials));
    CUDA_CHECK(cudaFree(device_bins));
    for (std::uint8_t* pointer : device_inputs) CUDA_CHECK(cudaFree(pointer));
    return 0;
}
