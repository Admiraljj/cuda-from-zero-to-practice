#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

constexpr int kTileSize = 256;
constexpr int kMaximumItems = 4 * 1024 * 1024;

struct LevelRecord {
    int depth;
    int items;
    int blocks;
};

__global__ void scan_tiles(const std::int64_t* input,
                           std::int64_t* output,
                           std::int64_t* block_sums,
                           int n) {
    __shared__ std::int64_t data[kTileSize];
    const int tid = threadIdx.x;
    const int index_in_array = blockIdx.x * kTileSize + tid;
    data[tid] = index_in_array < n ? input[index_in_array] : 0;
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

    if (index_in_array < n) output[index_in_array] = data[tid];
}

__global__ void uniform_add(std::int64_t* output,
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

bool scan_recursive(const std::int64_t* input,
                    std::int64_t* output,
                    int n,
                    const cudaDeviceProp& prop,
                    int depth,
                    std::vector<LevelRecord>& levels) {
    const int blocks = (n + kTileSize - 1) / kTileSize;
    if (blocks < 1 || blocks > prop.maxGridSize[0]) {
        std::cerr << "level " << depth << " grid size unsupported: "
                  << blocks << " blocks\n";
        return false;
    }
    levels.push_back({depth, n, blocks});

    std::int64_t* block_sums = nullptr;
    std::int64_t* block_offsets = nullptr;
    const std::size_t level_bytes =
        static_cast<std::size_t>(blocks) * sizeof(std::int64_t);
    if (!check(cudaMalloc(&block_sums, level_bytes),
               "allocate block sums")) return false;

    scan_tiles<<<blocks, kTileSize>>>(input, output, block_sums, n);
    bool ok = check(cudaGetLastError(), "launch tile scans");

    if (ok && blocks > 1) {
        ok = check(cudaMalloc(&block_offsets, level_bytes),
                   "allocate block offsets");
        if (ok) {
            ok = scan_recursive(block_sums, block_offsets, blocks, prop,
                                depth + 1, levels);
        }
        if (ok) {
            uniform_add<<<blocks, kTileSize>>>(output, block_offsets, n);
            ok = check(cudaGetLastError(), "launch uniform add");
        }
    }

    if (block_offsets &&
        !check(cudaFree(block_offsets), "free block offsets")) ok = false;
    if (block_sums &&
        !check(cudaFree(block_sums), "free block sums")) ok = false;
    return ok;
}

int main(int argc, char** argv) {
    int n = 1'000'003;
    bool in_place = false;
    if (argc > 3 ||
        (argc >= 2 && !parse_decimal(argv[1], 0, kMaximumItems, n)) ||
        (argc == 3 && std::string(argv[2]) != "out" &&
         std::string(argv[2]) != "inplace")) {
        std::cerr << "Usage: hierarchical_scan [n [out|inplace]], 0 <= n <= "
                  << kMaximumItems << '\n';
        return 2;
    }
    if (argc == 3) in_place = std::string(argv[2]) == "inplace";
    if (n == 0) {
        std::cout << "levels=0 total=0\n"
                     "PASS empty hierarchical scan; no CUDA calls\n";
        return 0;
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
        std::cerr << "device does not support the fixed 256-thread tile\n";
        return 1;
    }

    std::vector<std::int64_t> input(static_cast<std::size_t>(n));
    std::vector<std::int64_t> expected(static_cast<std::size_t>(n));
    std::vector<std::int64_t> actual(static_cast<std::size_t>(n));
    std::int64_t running = 0;
    for (int i = 0; i < n; ++i) {
        input[i] = (i * 7) % 11 - 5;
        expected[i] = running;
        running += input[i];
    }

    std::int64_t* d_input = nullptr;
    std::int64_t* d_output = nullptr;
    bool ok = false;
    const std::size_t bytes = input.size() * sizeof(std::int64_t);
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input") ||
            !check(cudaMemcpy(d_input, input.data(), bytes,
                              cudaMemcpyHostToDevice), "copy input H2D")) break;
        if (in_place) {
            d_output = d_input;
        } else if (!check(cudaMalloc(&d_output, bytes), "allocate output")) {
            break;
        }

        std::vector<LevelRecord> levels;
        if (!scan_recursive(d_input, d_output, n, prop, 0, levels) ||
            !check(cudaDeviceSynchronize(), "wait for hierarchical scan") ||
            !check(cudaMemcpy(actual.data(), d_output, bytes,
                              cudaMemcpyDeviceToHost), "copy output D2H")) break;

        ok = true;
        for (int i = 0; i < n; ++i) {
            if (actual[i] != expected[i]) {
                std::cerr << "mismatch i=" << i << " actual=" << actual[i]
                          << " expected=" << expected[i] << '\n';
                ok = false;
                break;
            }
        }
        const std::int64_t actual_total = actual.back() + input.back();
        if (ok && actual_total != running) {
            std::cerr << "total mismatch actual=" << actual_total
                      << " expected=" << running << '\n';
            ok = false;
        }
        if (ok) {
            std::cout << "n=" << n << " tile=" << kTileSize
                      << " mode=" << (in_place ? "inplace" : "out")
                      << " levels=" << levels.size() << '\n';
            for (const auto& level : levels) {
                std::cout << "level=" << level.depth
                          << " items=" << level.items
                          << " blocks=" << level.blocks << '\n';
            }
            const int shown = std::min(n, 16);
            std::cout << "exclusive=[";
            for (int i = 0; i < shown; ++i) {
                if (i) std::cout << ", ";
                std::cout << actual[i];
            }
            if (shown < n) std::cout << ", ...";
            std::cout << "]\ntotal=" << actual_total << '\n';
        }
    } while (false);

    if (!in_place && d_output &&
        !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input && !check(cudaFree(d_input), "free input")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS hierarchical exclusive scan; every prefix checked\n";
    return 0;
}
