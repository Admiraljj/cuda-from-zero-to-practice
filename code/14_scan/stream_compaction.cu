#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

constexpr int kThreads = 256;
constexpr int kMaximumItems = 4 * 1024 * 1024;

enum class Pattern { Mixed, None, All };

__global__ void make_positive_flags(const int* input, int* flags, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) flags[index] = input[index] > 0 ? 1 : 0;
}

__global__ void scatter_selected(const int* input,
                                 const int* flags,
                                 const int* offsets,
                                 int* output,
                                 int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n && flags[index] != 0) {
        output[offsets[index]] = input[index];
    }
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

bool parse_pattern(const char* text, Pattern& pattern) {
    const std::string value = text ? text : "";
    if (value == "mixed") pattern = Pattern::Mixed;
    else if (value == "none") pattern = Pattern::None;
    else if (value == "all") pattern = Pattern::All;
    else return false;
    return true;
}

const char* pattern_name(Pattern pattern) {
    if (pattern == Pattern::None) return "none";
    if (pattern == Pattern::All) return "all";
    return "mixed";
}

int make_value(int index, Pattern pattern) {
    if (pattern == Pattern::None) return -(index % 7 + 1);
    if (pattern == Pattern::All) return index % 7 + 1;
    return (index * 7) % 13 - 6;
}

int main(int argc, char** argv) {
    int n = 1031;
    Pattern pattern = Pattern::Mixed;
    if (argc > 3 ||
        (argc >= 2 && !parse_decimal(argv[1], 0, kMaximumItems, n)) ||
        (argc == 3 && !parse_pattern(argv[2], pattern))) {
        std::cerr << "Usage: stream_compaction [n [mixed|none|all]], 0 <= n <= "
                  << kMaximumItems << '\n';
        return 2;
    }
    if (n == 0) {
        std::cout << "input_count=0 selected_count=0 output=[]\n"
                     "PASS empty compaction; no CUDA calls\n";
        return 0;
    }

    int device = -1;
    cudaDeviceProp prop{};
    if (!check(cudaGetDevice(&device), "get current device") ||
        !check(cudaGetDeviceProperties(&prop, device), "get device properties")) {
        return 1;
    }
    const int blocks = (n + kThreads - 1) / kThreads;
    if (kThreads > prop.maxThreadsPerBlock ||
        kThreads > prop.maxThreadsDim[0] || blocks > prop.maxGridSize[0]) {
        std::cerr << "launch configuration unsupported\n";
        return 1;
    }

    std::vector<int> input(static_cast<std::size_t>(n));
    std::vector<int> expected_flags(static_cast<std::size_t>(n));
    std::vector<int> expected_offsets(static_cast<std::size_t>(n));
    std::vector<int> expected_output;
    expected_output.reserve(static_cast<std::size_t>(n));
    int running = 0;
    for (int i = 0; i < n; ++i) {
        input[i] = make_value(i, pattern);
        expected_flags[i] = input[i] > 0 ? 1 : 0;
        expected_offsets[i] = running;
        if (expected_flags[i]) {
            expected_output.push_back(input[i]);
            ++running;
        }
    }

    int* d_input = nullptr;
    int* d_flags = nullptr;
    int* d_offsets = nullptr;
    int* d_output = nullptr;
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    bool ok = false;
    const std::size_t bytes = input.size() * sizeof(int);
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input") ||
            !check(cudaMalloc(&d_flags, bytes), "allocate flags") ||
            !check(cudaMalloc(&d_offsets, bytes), "allocate offsets") ||
            !check(cudaMalloc(&d_output, bytes), "allocate output capacity") ||
            !check(cudaMemcpy(d_input, input.data(), bytes,
                              cudaMemcpyHostToDevice), "copy input H2D")) break;

        make_positive_flags<<<blocks, kThreads>>>(d_input, d_flags, n);
        if (!check(cudaGetLastError(), "launch flag kernel")) break;

        if (!check(cub::DeviceScan::ExclusiveSum(
                       nullptr, temp_storage_bytes, d_flags, d_offsets, n),
                   "query CUB scan temporary storage") ||
            !check(cudaMalloc(&d_temp_storage, temp_storage_bytes),
                   "allocate CUB temporary storage") ||
            !check(cub::DeviceScan::ExclusiveSum(
                       d_temp_storage, temp_storage_bytes,
                       d_flags, d_offsets, n),
                   "run CUB exclusive scan")) break;

        scatter_selected<<<blocks, kThreads>>>(
            d_input, d_flags, d_offsets, d_output, n);
        if (!check(cudaGetLastError(), "launch scatter kernel") ||
            !check(cudaDeviceSynchronize(), "wait for compaction")) break;

        std::vector<int> actual_flags(static_cast<std::size_t>(n));
        std::vector<int> actual_offsets(static_cast<std::size_t>(n));
        if (!check(cudaMemcpy(actual_flags.data(), d_flags, bytes,
                              cudaMemcpyDeviceToHost), "copy flags D2H") ||
            !check(cudaMemcpy(actual_offsets.data(), d_offsets, bytes,
                              cudaMemcpyDeviceToHost), "copy offsets D2H")) break;

        const int selected_count =
            actual_offsets.back() + actual_flags.back();
        if (selected_count < 0 || selected_count > n) {
            std::cerr << "invalid selected count " << selected_count << '\n';
            break;
        }
        std::vector<int> actual_output(
            static_cast<std::size_t>(selected_count));
        if (selected_count > 0 &&
            !check(cudaMemcpy(actual_output.data(), d_output,
                              actual_output.size() * sizeof(int),
                              cudaMemcpyDeviceToHost), "copy compacted output D2H")) {
            break;
        }

        ok = actual_flags == expected_flags &&
             actual_offsets == expected_offsets &&
             actual_output == expected_output;
        if (!ok) {
            std::cerr << "compaction mismatch: flags, offsets, or stable output differ\n";
            break;
        }

        const int shown = std::min(selected_count, 16);
        std::cout << "input_count=" << n
                  << " selected_count=" << selected_count
                  << " pattern=" << pattern_name(pattern)
                  << " temp_storage_bytes=" << temp_storage_bytes << '\n';
        std::cout << "output=[";
        for (int i = 0; i < shown; ++i) {
            if (i) std::cout << ", ";
            std::cout << actual_output[i];
        }
        if (shown < selected_count) std::cout << ", ...";
        std::cout << "]\n";
    } while (false);

    if (d_temp_storage &&
        !check(cudaFree(d_temp_storage), "free CUB temporary storage")) ok = false;
    if (d_output && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_offsets && !check(cudaFree(d_offsets), "free offsets")) ok = false;
    if (d_flags && !check(cudaFree(d_flags), "free flags")) ok = false;
    if (d_input && !check(cudaFree(d_input), "free input")) ok = false;

    if (!ok) return 1;
    std::cout << "PASS stable stream compaction; flags, offsets, and output checked\n";
    return 0;
}
