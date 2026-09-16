#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>

constexpr int kTileSize = 256;
constexpr int kMaximumItems = 4 * 1024 * 1024;

struct LevelWorkspace {
    int items = 0;
    int blocks = 0;
    std::int64_t* sums = nullptr;
    std::int64_t* offsets = nullptr;
};

enum class Method { Manual, Cub };

__global__ void benchmark_scan_tiles(const std::int64_t* input,
                                     std::int64_t* output,
                                     std::int64_t* block_sums,
                                     int n) {
    __shared__ std::int64_t data[kTileSize];
    const int tid = threadIdx.x;
    const int global_index = blockIdx.x * kTileSize + tid;
    data[tid] = global_index < n ? input[global_index] : 0;
    __syncthreads();

    for (int offset = 1; offset < kTileSize; offset <<= 1) {
        const int index = (tid + 1) * offset * 2 - 1;
        if (index < kTileSize) data[index] += data[index - offset];
        __syncthreads();
    }
    if (tid == 0) {
        block_sums[blockIdx.x] = data[kTileSize - 1];
        data[kTileSize - 1] = 0;
    }
    __syncthreads();

    for (int offset = kTileSize >> 1; offset > 0; offset >>= 1) {
        const int index = (tid + 1) * offset * 2 - 1;
        if (index < kTileSize) {
            const std::int64_t left = data[index - offset];
            data[index - offset] = data[index];
            data[index] += left;
        }
        __syncthreads();
    }
    if (global_index < n) output[global_index] = data[tid];
}

__global__ void benchmark_uniform_add(std::int64_t* output,
                                      const std::int64_t* block_offsets,
                                      int n) {
    const int index = blockIdx.x * kTileSize + threadIdx.x;
    if (index < n) output[index] += block_offsets[blockIdx.x];
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool parse_decimal(const char* text, int minimum, int maximum, int& result) {
    if (!text || !*text) return false;
    int value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        const int digit = *p - '0';
        if (value > (maximum - digit) / 10) return false;
        value = value * 10 + digit;
        if (value > maximum) return false;
    }
    if (value < minimum) return false;
    result = value;
    return true;
}

void free_levels(std::vector<LevelWorkspace>& levels) {
    for (auto& level : levels) {
        if (level.offsets) cudaFree(level.offsets);
        if (level.sums) cudaFree(level.sums);
        level.offsets = nullptr;
        level.sums = nullptr;
    }
}

bool allocate_levels(int n, const cudaDeviceProp& prop,
                     std::vector<LevelWorkspace>& levels) {
    while (true) {
        const int blocks = (n + kTileSize - 1) / kTileSize;
        if (blocks < 1 || blocks > prop.maxGridSize[0]) {
            std::cerr << "manual scan grid unsupported: " << blocks << '\n';
            return false;
        }
        LevelWorkspace level;
        level.items = n;
        level.blocks = blocks;
        const std::size_t bytes =
            static_cast<std::size_t>(blocks) * sizeof(std::int64_t);
        if (!check(cudaMalloc(&level.sums, bytes),
                   "allocate manual block sums")) return false;
        if (blocks > 1 &&
            !check(cudaMalloc(&level.offsets, bytes),
                   "allocate manual block offsets")) {
            cudaFree(level.sums);
            return false;
        }
        levels.push_back(level);
        if (blocks == 1) return true;
        n = blocks;
    }
}

bool launch_manual_scan(const std::int64_t* input,
                        std::int64_t* output,
                        const std::vector<LevelWorkspace>& levels,
                        std::size_t depth = 0) {
    const auto& level = levels.at(depth);
    benchmark_scan_tiles<<<level.blocks, kTileSize>>>(
        input, output, level.sums, level.items);
    if (!check(cudaGetLastError(), "launch benchmark tile scan")) return false;

    if (level.blocks > 1) {
        if (!launch_manual_scan(level.sums, level.offsets,
                                levels, depth + 1)) return false;
        benchmark_uniform_add<<<level.blocks, kTileSize>>>(
            output, level.offsets, level.items);
        if (!check(cudaGetLastError(),
                   "launch benchmark uniform add")) return false;
    }
    return true;
}

bool validate_output(const std::vector<std::int64_t>& actual,
                     const std::vector<std::int64_t>& expected,
                     Method method, int round, bool warmup) {
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << (method == Method::Manual ? "manual" : "cub")
                      << " mismatch phase=" << (warmup ? "warmup" : "measure")
                      << " round=" << round << " i=" << i
                      << " actual=" << actual[i]
                      << " expected=" << expected[i] << '\n';
            return false;
        }
    }
    return true;
}

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return 0.5 * (static_cast<double>(values[middle - 1]) + values[middle]);
}

void print_statistics(std::string_view name,
                      const std::vector<float>& samples, int n) {
    const auto [minimum, maximum] =
        std::minmax_element(samples.begin(), samples.end());
    const double middle = median(samples);
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(samples.size());
    const double items_per_second =
        middle > 0.0 ? static_cast<double>(n) / (middle * 1.0e-3) : 0.0;
    std::cout << name << " samples_ms=[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << samples[i];
    }
    std::cout << "] min_ms=" << *minimum
              << " median_ms=" << middle
              << " mean_ms=" << mean
              << " max_ms=" << *maximum
              << " logical_items_per_s=" << items_per_second << '\n';
}

int main(int argc, char** argv) {
    int n = 1'000'003;
    int rounds = 15;
    int warmups = 4;
    if (argc > 4 ||
        (argc >= 2 && !parse_decimal(argv[1], 1, kMaximumItems, n)) ||
        (argc >= 3 && !parse_decimal(argv[2], 3, 100, rounds)) ||
        (argc == 4 && !parse_decimal(argv[3], 1, 20, warmups))) {
        std::cerr << "Usage: scan_benchmark [n [rounds [warmups]]], "
                     "1 <= n <= " << kMaximumItems
                  << ", 3 <= rounds <= 100, 1 <= warmups <= 20\n";
        return 2;
    }

    int device = -1;
    cudaDeviceProp prop{};
    if (!check(cudaGetDevice(&device), "get current device") ||
        !check(cudaGetDeviceProperties(&prop, device), "get device properties")) {
        return 1;
    }
    if (kTileSize > prop.maxThreadsPerBlock ||
        kTileSize > prop.maxThreadsDim[0] ||
        sizeof(std::int64_t) * kTileSize > prop.sharedMemPerBlock) {
        std::cerr << "device does not support the manual 256-thread tile\n";
        return 1;
    }

    std::vector<std::int64_t> input(static_cast<std::size_t>(n));
    std::vector<std::int64_t> expected(static_cast<std::size_t>(n));
    std::vector<std::int64_t> actual(static_cast<std::size_t>(n));
    std::vector<std::int64_t> input_snapshot(static_cast<std::size_t>(n));
    std::int64_t running = 0;
    for (int i = 0; i < n; ++i) {
        input[i] = (i * 7) % 11 - 5;
        expected[i] = running;
        running += input[i];
    }

    std::int64_t* d_input = nullptr;
    std::int64_t* d_manual_output = nullptr;
    std::int64_t* d_cub_output = nullptr;
    void* d_cub_temp = nullptr;
    std::size_t cub_temp_bytes = 0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<LevelWorkspace> levels;
    std::vector<float> manual_samples;
    std::vector<float> cub_samples;
    manual_samples.reserve(static_cast<std::size_t>(rounds));
    cub_samples.reserve(static_cast<std::size_t>(rounds));
    bool ok = false;
    const std::size_t bytes = input.size() * sizeof(std::int64_t);

    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate immutable input") ||
            !check(cudaMalloc(&d_manual_output, bytes), "allocate manual output") ||
            !check(cudaMalloc(&d_cub_output, bytes), "allocate CUB output") ||
            !check(cudaMemcpy(d_input, input.data(), bytes,
                              cudaMemcpyHostToDevice), "copy input H2D") ||
            !allocate_levels(n, prop, levels) ||
            !check(cub::DeviceScan::ExclusiveSum(
                       nullptr, cub_temp_bytes, d_input, d_cub_output, n),
                   "query CUB temporary storage") ||
            !check(cudaMalloc(&d_cub_temp, cub_temp_bytes),
                   "allocate CUB temporary storage") ||
            !check(cudaEventCreate(&start), "create start event") ||
            !check(cudaEventCreate(&stop), "create stop event")) break;

        ok = true;
        for (int phase = 0; phase < 2 && ok; ++phase) {
            const bool warmup = phase == 0;
            const int phase_rounds = warmup ? warmups : rounds;
            for (int round = 0; round < phase_rounds && ok; ++round) {
                for (int position = 0; position < 2 && ok; ++position) {
                    const Method method =
                        ((round + position) % 2 == 0) ? Method::Manual
                                                       : Method::Cub;
                    std::int64_t* output = method == Method::Manual
                                               ? d_manual_output
                                               : d_cub_output;
                    if (!check(cudaMemsetAsync(output, 0xA5, bytes),
                               "fill output sentinel") ||
                        !check(cudaEventRecord(start), "record start event")) {
                        ok = false;
                        break;
                    }

                    if (method == Method::Manual) {
                        ok = launch_manual_scan(d_input, output, levels);
                    } else {
                        ok = check(cub::DeviceScan::ExclusiveSum(
                                       d_cub_temp, cub_temp_bytes,
                                       d_input, output, n),
                                   "run CUB scan");
                    }
                    if (!ok ||
                        !check(cudaEventRecord(stop), "record stop event") ||
                        !check(cudaEventSynchronize(stop), "wait for timed scan")) {
                        ok = false;
                        break;
                    }

                    float milliseconds = 0.0f;
                    if (!check(cudaEventElapsedTime(&milliseconds, start, stop),
                               "read elapsed time") ||
                        !check(cudaMemcpy(actual.data(), output, bytes,
                                          cudaMemcpyDeviceToHost),
                               "copy timed output D2H")) {
                        ok = false;
                        break;
                    }
                    ok = validate_output(actual, expected, method, round, warmup);
                    if (!warmup && ok) {
                        (method == Method::Manual ? manual_samples : cub_samples)
                            .push_back(milliseconds);
                    }
                }

                if (ok &&
                    !check(cudaMemcpy(input_snapshot.data(), d_input, bytes,
                                      cudaMemcpyDeviceToHost),
                           "copy immutable input D2H")) ok = false;
                if (ok && input_snapshot != input) {
                    std::cerr << "immutable input changed phase="
                              << (warmup ? "warmup" : "measure")
                              << " round=" << round << '\n';
                    ok = false;
                }
            }
        }
    } while (false);

    if (ok && (manual_samples.size() != static_cast<std::size_t>(rounds) ||
               cub_samples.size() != static_cast<std::size_t>(rounds))) {
        std::cerr << "sample count mismatch\n";
        ok = false;
    }
    if (ok) {
        std::cout << std::fixed << std::setprecision(6)
                  << "device=\"" << prop.name << "\" n=" << n
                  << " rounds=" << rounds << " warmups=" << warmups
                  << " type=int64 exclusive=true\n"
                  << "timed=scan-kernels-only; excludes allocation,H2D,D2H,"
                     "sentinel,validation,printing\n"
                  << "manual_levels=" << levels.size()
                  << " cub_temp_bytes=" << cub_temp_bytes << '\n';
        print_statistics("manual", manual_samples, n);
        print_statistics("cub", cub_samples, n);
        std::cout << "PASS all warmup and measured outputs; immutable input checked "
                  << (warmups + rounds) << " rounds\n";
    }

    if (stop && !check(cudaEventDestroy(stop), "destroy stop event")) ok = false;
    if (start && !check(cudaEventDestroy(start), "destroy start event")) ok = false;
    if (d_cub_temp &&
        !check(cudaFree(d_cub_temp), "free CUB temporary storage")) ok = false;
    free_levels(levels);
    if (d_cub_output &&
        !check(cudaFree(d_cub_output), "free CUB output")) ok = false;
    if (d_manual_output &&
        !check(cudaFree(d_manual_output), "free manual output")) ok = false;
    if (d_input && !check(cudaFree(d_input), "free input")) ok = false;
    return ok ? 0 : 1;
}
