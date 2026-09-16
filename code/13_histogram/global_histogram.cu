#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

constexpr int kBins = 256;

enum class Pattern { Cycle, Hot, Skew };

__global__ void histogram_global(const unsigned char* input,
                                 unsigned int* bins, int n) {
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = blockDim.x * gridDim.x;
    for (int i = start; i < n; i += step) {
        atomicAdd(&bins[input[i]], 1u);
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

bool parse_pattern(const char* text, Pattern& result) {
    const std::string value = text ? text : "";
    if (value == "cycle") result = Pattern::Cycle;
    else if (value == "hot") result = Pattern::Hot;
    else if (value == "skew") result = Pattern::Skew;
    else return false;
    return true;
}

const char* pattern_name(Pattern pattern) {
    if (pattern == Pattern::Cycle) return "cycle";
    if (pattern == Pattern::Hot) return "hot";
    return "skew";
}

unsigned char make_value(int i, Pattern pattern) {
    if (pattern == Pattern::Hot) return 0;
    if (pattern == Pattern::Skew && i % 10 < 8) return 0;
    return static_cast<unsigned char>(i % kBins);
}

int main(int argc, char** argv) {
    try {
        int n = 1031, threads = 128, blocks = 4;
        Pattern pattern = Pattern::Cycle;
        if (argc != 1 && (argc != 5 ||
            !parse_decimal(argv[1], 0, 1048576, n) ||
            !parse_decimal(argv[2], 1, 1024, threads) ||
            !parse_decimal(argv[3], 1, 65535, blocks) ||
            !parse_pattern(argv[4], pattern))) {
            std::cerr << "Usage: global_histogram [n threads blocks pattern]; "
                         "pattern=cycle|hot|skew\n";
            return 2;
        }
        if (n == 0) {
            std::cout << "EMPTY histogram total=0; no CUDA calls\n";
            return 0;
        }

        int device = -1;
        cudaDeviceProp prop = {};
        if (!check(cudaGetDevice(&device), "current device") ||
            !check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
        if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0] ||
            blocks > prop.maxGridSize[0]) {
            std::cerr << "configuration exceeds device limits\n";
            return 1;
        }

        std::vector<unsigned char> input(n);
        std::array<unsigned int, kBins> expected{}, actual{};
        for (int i = 0; i < n; ++i) {
            input[i] = make_value(i, pattern);
            ++expected[input[i]];
        }

        unsigned char* d_input = nullptr;
        unsigned int* d_bins = nullptr;
        bool ok = false;
        do {
            if (!check(cudaMalloc(&d_input, input.size() * sizeof(unsigned char)),
                       "allocate input") ||
                !check(cudaMalloc(&d_bins, actual.size() * sizeof(unsigned int)),
                       "allocate bins") ||
                !check(cudaMemcpy(d_input, input.data(),
                                  input.size() * sizeof(unsigned char),
                                  cudaMemcpyHostToDevice), "H2D input") ||
                !check(cudaMemset(d_bins, 0,
                                  actual.size() * sizeof(unsigned int)),
                       "zero bins")) break;

            histogram_global<<<blocks, threads>>>(d_input, d_bins, n);
            if (!check(cudaGetLastError(), "launch") ||
                !check(cudaDeviceSynchronize(), "wait") ||
                !check(cudaMemcpy(actual.data(), d_bins,
                                  actual.size() * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "D2H bins")) break;

            ok = true;
            std::uint64_t total = 0;
            for (int bin = 0; bin < kBins; ++bin) {
                total += actual[bin];
                if (actual[bin] != expected[bin]) {
                    std::cerr << "bin mismatch bin=" << bin
                              << " actual=" << actual[bin]
                              << " expected=" << expected[bin] << '\n';
                    ok = false;
                    break;
                }
            }
            if (ok && total != static_cast<std::uint64_t>(n)) {
                std::cerr << "histogram total mismatch\n";
                ok = false;
            }
            if (ok) {
                std::cout << "n=" << n << " threads=" << threads
                          << " blocks=" << blocks
                          << " pattern=" << pattern_name(pattern)
                          << " total=" << total << '\n';
                for (int bin = 0; bin < 8; ++bin) {
                    std::cout << "bin[" << bin << "]=" << actual[bin]
                              << (bin == 7 ? '\n' : ' ');
                }
                std::cout << "bin[255]=" << actual[255] << '\n';
            }
        } while (false);

        if (d_bins && !check(cudaFree(d_bins), "free bins")) ok = false;
        if (d_input && !check(cudaFree(d_input), "free input")) ok = false;
        if (!ok) return 1;
        std::cout << "PASS global atomic histogram; all 256 bins and total checked\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
