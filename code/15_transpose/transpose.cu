#include "../common/cuda_check.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <system_error>
#include <vector>

namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;
constexpr std::size_t kMaximumElements = 16ULL * 1024ULL * 1024ULL;

enum class Method : int {
    copy = 0,
    naive = 1,
    tiled_no_padding = 2,
    tiled_padded = 3,
    count = 4,
};

constexpr int kMethodCount = static_cast<int>(Method::count);

struct Config {
    int width = 1003;
    int height = 777;
    int rounds = 7;
    int warmups = 2;
};

struct DeviceBuffer {
    float* pointer = nullptr;

    explicit DeviceBuffer(std::size_t bytes) {
        CUDA_CHECK(cudaMalloc(&pointer, bytes));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() {
        if (pointer != nullptr) cudaFree(pointer);
    }
};

struct Event {
    cudaEvent_t value{};

    Event() { CUDA_CHECK(cudaEventCreate(&value)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() { cudaEventDestroy(value); }
};

__global__ void copy_kernel(const float* input, float* output,
                            int width, int height) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row < height && col < width) {
        const std::size_t index = static_cast<std::size_t>(row) * width + col;
        output[index] = input[index];
    }
}
__global__ void transpose_naive_kernel(const float* input, float* output,
                                       int width, int height) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row < height && col < width) {
        const std::size_t input_index =
            static_cast<std::size_t>(row) * width + col;
        const std::size_t output_index =
            static_cast<std::size_t>(col) * height + row;
        output[output_index] = input[input_index];
    }
}

template <int Padding>
__global__ void transpose_tiled_kernel(const float* input, float* output,
                                       int width, int height) {
    __shared__ float tile[kTileDim][kTileDim + Padding];

    const int input_col =
        static_cast<int>(blockIdx.x) * kTileDim + threadIdx.x;
    const int input_row_base =
        static_cast<int>(blockIdx.y) * kTileDim + threadIdx.y;
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int input_row = input_row_base + j;
        if (input_col < width && input_row < height) {
            const std::size_t input_index =
                static_cast<std::size_t>(input_row) * width + input_col;
            tile[threadIdx.y + j][threadIdx.x] = input[input_index];
        }
    }

    __syncthreads();

    const int output_col =
        static_cast<int>(blockIdx.y) * kTileDim + threadIdx.x;
    const int output_row_base =
        static_cast<int>(blockIdx.x) * kTileDim + threadIdx.y;
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int output_row = output_row_base + j;
        if (output_col < height && output_row < width) {
            const std::size_t output_index =
                static_cast<std::size_t>(output_row) * height + output_col;
            output[output_index] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

const char* method_name(Method method) {
    switch (method) {
        case Method::copy: return "copy-baseline";
        case Method::naive: return "transpose-naive";
        case Method::tiled_no_padding: return "transpose-tiled-no-padding";
        case Method::tiled_padded: return "transpose-tiled-padded";
        default: return "unknown";
    }
}

bool parse_integer(const char* text, int minimum, int maximum, int& result) {
    if (text == nullptr || *text == '\0') return false;
    const char* end = text;
    while (*end != '\0') ++end;
    int value = 0;
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) return false;
    if (value < minimum || value > maximum) return false;
    result = value;
    return true;
}

bool parse_arguments(int argc, char** argv, Config& config) {
    if (argc > 5) return false;
    if (argc >= 2 && !parse_integer(argv[1], 1, 16384, config.width)) return false;
    if (argc >= 3 && !parse_integer(argv[2], 1, 16384, config.height)) return false;
    if (argc >= 4 && !parse_integer(argv[3], 3, 100, config.rounds)) return false;
    if (argc >= 5 && !parse_integer(argv[4], 1, 20, config.warmups)) return false;
    const std::size_t count =
        static_cast<std::size_t>(config.width) * config.height;
    return count <= kMaximumElements;
}

void make_cpu_transpose(const std::vector<float>& input,
                        std::vector<float>& expected,
                        int width, int height) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const std::size_t input_index =
                static_cast<std::size_t>(row) * width + col;
            const std::size_t output_index =
                static_cast<std::size_t>(col) * height + row;
            expected[output_index] = input[input_index];
        }
    }
}

bool validate_exact(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    Method method, int sample) {
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (actual[index] != expected[index]) {
            std::cerr << "FAIL " << method_name(method)
                      << " sample=" << sample << " index=" << index
                      << " expected=" << expected[index]
                      << " actual=" << actual[index] << '\n';
            return false;
        }
    }
    return true;
}

bool validate_input_unchanged(const std::vector<float>& original,
                              std::vector<float>& copied,
                              const float* device_input,
                              std::size_t bytes, int round) {
    CUDA_CHECK(cudaMemcpy(copied.data(), device_input, bytes,
                          cudaMemcpyDeviceToHost));
    for (std::size_t index = 0; index < original.size(); ++index) {
        if (copied[index] != original[index]) {
            std::cerr << "FAIL input changed after round " << round
                      << " at index=" << index << '\n';
            return false;
        }
    }
    return true;
}

void check_launch_limits(const Config& config,
                         const cudaDeviceProp& properties) {
    const dim3 block(kTileDim, kBlockRows);
    const dim3 direct_grid((config.width + block.x - 1) / block.x,
                           (config.height + block.y - 1) / block.y);
    const dim3 tiled_grid((config.width + kTileDim - 1) / kTileDim,
                          (config.height + kTileDim - 1) / kTileDim);
    constexpr std::size_t padded_shared_bytes =
        kTileDim * (kTileDim + 1) * sizeof(float);

    const bool block_ok = block.x <= properties.maxThreadsDim[0] &&
                          block.y <= properties.maxThreadsDim[1] &&
                          block.x * block.y <= properties.maxThreadsPerBlock;
    const bool grid_ok =
        direct_grid.x <= static_cast<unsigned>(properties.maxGridSize[0]) &&
        direct_grid.y <= static_cast<unsigned>(properties.maxGridSize[1]) &&
        tiled_grid.x <= static_cast<unsigned>(properties.maxGridSize[0]) &&
        tiled_grid.y <= static_cast<unsigned>(properties.maxGridSize[1]);
    const bool shared_ok = padded_shared_bytes <= properties.sharedMemPerBlock;
    if (!block_ok || !grid_ok || !shared_ok) {
        std::cerr << "Requested launch exceeds this GPU's block, grid, or "
                     "shared-memory limit.\n";
        std::exit(EXIT_FAILURE);
    }
}

void launch(Method method, const float* input, float* output,
            int width, int height) {
    const dim3 block(kTileDim, kBlockRows);
    const dim3 direct_grid((width + block.x - 1) / block.x,
                           (height + block.y - 1) / block.y);
    const dim3 tiled_grid((width + kTileDim - 1) / kTileDim,
                          (height + kTileDim - 1) / kTileDim);

    switch (method) {
        case Method::copy:
            copy_kernel<<<direct_grid, block>>>(input, output, width, height);
            break;
        case Method::naive:
            transpose_naive_kernel<<<direct_grid, block>>>(
                input, output, width, height);
            break;
        case Method::tiled_no_padding:
            transpose_tiled_kernel<0><<<tiled_grid, block>>>(
                input, output, width, height);
            break;
        case Method::tiled_padded:
            transpose_tiled_kernel<1><<<tiled_grid, block>>>(
                input, output, width, height);
            break;
        default:
            std::cerr << "Unknown method.\n";
            std::exit(EXIT_FAILURE);
    }
}

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return 0.5 * (values[middle - 1] + values[middle]);
}

void print_statistics(Method method,
                      const std::vector<float>& milliseconds,
                      std::size_t bytes) {
    const auto limits = std::minmax_element(milliseconds.begin(),
                                            milliseconds.end());
    const double mean = std::accumulate(milliseconds.begin(),
                                        milliseconds.end(), 0.0) /
                        milliseconds.size();
    const double middle = median(milliseconds);
    const double logical_gbps =
        2.0 * static_cast<double>(bytes) / (middle * 1.0e6);

    std::cout << std::left << std::setw(30) << method_name(method)
              << std::right << std::fixed << std::setprecision(4)
              << " min=" << std::setw(9) << *limits.first
              << " median=" << std::setw(9) << middle
              << " mean=" << std::setw(9) << mean
              << " max=" << std::setw(9) << *limits.second
              << " ms  logical_GB/s=" << std::setw(9)
              << std::setprecision(2) << logical_gbps << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    Config config;
    if (!parse_arguments(argc, argv, config)) {
        std::cerr << "Usage: " << argv[0]
                  << " [width:1..16384] [height:1..16384]"
                     " [rounds:3..100] [warmups:1..20]\n"
                  << "width*height must be <= " << kMaximumElements << ".\n";
        return EXIT_FAILURE;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    check_launch_limits(config, properties);

    const std::size_t count =
        static_cast<std::size_t>(config.width) * config.height;
    const std::size_t bytes = count * sizeof(float);

    std::vector<float> input(count);
    std::vector<float> expected_transpose(count);
    std::vector<float> host_output(count);
    std::vector<float> input_check(count);
    for (int row = 0; row < config.height; ++row) {
        for (int col = 0; col < config.width; ++col) {
            const std::size_t index =
                static_cast<std::size_t>(row) * config.width + col;
            input[index] = static_cast<float>((row * 131 + col * 17) % 4096);
        }
    }
    make_cpu_transpose(input, expected_transpose,
                       config.width, config.height);

    DeviceBuffer device_input(bytes);
    std::array<DeviceBuffer, kMethodCount> outputs{
        DeviceBuffer(bytes), DeviceBuffer(bytes),
        DeviceBuffer(bytes), DeviceBuffer(bytes)};
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), bytes,
                          cudaMemcpyHostToDevice));

    std::cout << "GPU: " << properties.name << '\n'
              << "input shape: " << config.height << " rows x "
              << config.width << " cols\n"
              << "output shape: " << config.width << " rows x "
              << config.height << " cols\n"
              << "block: (" << kTileDim << ", " << kBlockRows << ")"
              << ", rounds=" << config.rounds
              << ", warmups=" << config.warmups << '\n';

    for (int method_index = 0; method_index < kMethodCount; ++method_index) {
        const Method method = static_cast<Method>(method_index);
        for (int warmup = 0; warmup < config.warmups; ++warmup) {
            CUDA_CHECK(cudaMemset(outputs[method_index].pointer, 0xA5, bytes));
            launch(method, device_input.pointer, outputs[method_index].pointer,
                   config.width, config.height);
            CUDA_CHECK_KERNEL();
        }
        CUDA_CHECK(cudaMemcpy(host_output.data(), outputs[method_index].pointer,
                              bytes, cudaMemcpyDeviceToHost));
        const std::vector<float>& expected =
            method == Method::copy ? input : expected_transpose;
        if (!validate_exact(host_output, expected, method, -1)) {
            return EXIT_FAILURE;
        }
    }

    std::array<std::vector<float>, kMethodCount> samples;
    Event start;
    Event stop;
    for (int round = 0; round < config.rounds; ++round) {
        for (int position = 0; position < kMethodCount; ++position) {
            const int method_index = (round + position) % kMethodCount;
            const Method method = static_cast<Method>(method_index);

            CUDA_CHECK(cudaMemset(outputs[method_index].pointer, 0xA5, bytes));
            CUDA_CHECK(cudaEventRecord(start.value));
            launch(method, device_input.pointer, outputs[method_index].pointer,
                   config.width, config.height);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop.value));
            CUDA_CHECK(cudaEventSynchronize(stop.value));

            float milliseconds = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds,
                                            start.value, stop.value));
            samples[method_index].push_back(milliseconds);

            CUDA_CHECK(cudaMemcpy(host_output.data(), outputs[method_index].pointer,
                                  bytes, cudaMemcpyDeviceToHost));
            const std::vector<float>& expected =
                method == Method::copy ? input : expected_transpose;
            if (!validate_exact(host_output, expected, method, round)) {
                return EXIT_FAILURE;
            }
        }

        if (!validate_input_unchanged(input, input_check,
                                      device_input.pointer, bytes, round)) {
            return EXIT_FAILURE;
        }
    }

    std::cout << "\nKernel-only CUDA Event statistics\n";
    for (int method_index = 0; method_index < kMethodCount; ++method_index) {
        print_statistics(static_cast<Method>(method_index),
                         samples[method_index], bytes);
    }
    std::cout << "logical_GB/s counts one full input read plus one full output "
                 "write; it is not a DRAM-counter measurement.\n"
              << "PASS all outputs match their CPU oracle and the input "
                 "remained unchanged.\n";
    return EXIT_SUCCESS;
}
