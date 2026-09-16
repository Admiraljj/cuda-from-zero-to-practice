#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

constexpr int kMaximumItems = 512;

__global__ void exclusive_scan_single_block(const std::int64_t* input,
                                            std::int64_t* output,
                                            std::int64_t* total,
                                            int n) {
    extern __shared__ std::int64_t data[];
    const int tid = threadIdx.x;
    const int capacity = blockDim.x;

    data[tid] = tid < n ? input[tid] : 0;
    __syncthreads();

    for (int offset = 1; offset < capacity; offset <<= 1) {
        const int index = (tid + 1) * offset * 2 - 1;
        if (index < capacity) {
            data[index] += data[index - offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *total = data[capacity - 1];
        data[capacity - 1] = 0;
    }
    __syncthreads();

    for (int offset = capacity >> 1; offset > 0; offset >>= 1) {
        const int index = (tid + 1) * offset * 2 - 1;
        if (index < capacity) {
            const std::int64_t left = data[index - offset];
            data[index - offset] = data[index];
            data[index] += left;
        }
        __syncthreads();
    }

    if (tid < n) output[tid] = data[tid];
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

int next_power_of_two(int n) {
    int value = 1;
    while (value < n) value <<= 1;
    return value;
}

int main(int argc, char** argv) {
    int n = 13;
    bool in_place = false;
    if (argc > 3 ||
        (argc >= 2 && !parse_decimal(argv[1], 0, kMaximumItems, n)) ||
        (argc == 3 && std::string(argv[2]) != "out" &&
         std::string(argv[2]) != "inplace")) {
        std::cerr << "Usage: single_block_scan [n [out|inplace]], 0 <= n <= "
                  << kMaximumItems << '\n';
        return 2;
    }
    if (argc == 3) in_place = std::string(argv[2]) == "inplace";
    if (n == 0) {
        std::cout << "input=[]\nexclusive=[]\ntotal=0\n"
                     "PASS empty input; no CUDA calls\n";
        return 0;
    }

    const int capacity = next_power_of_two(n);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(capacity) * sizeof(std::int64_t);

    int device = -1;
    cudaDeviceProp prop{};
    if (!check(cudaGetDevice(&device), "get current device") ||
        !check(cudaGetDeviceProperties(&prop, device), "get device properties")) {
        return 1;
    }
    if (capacity > prop.maxThreadsPerBlock ||
        capacity > prop.maxThreadsDim[0] ||
        shared_bytes > prop.sharedMemPerBlock) {
        std::cerr << "required single-block configuration is unsupported: "
                  << "threads=" << capacity
                  << " shared_bytes=" << shared_bytes << '\n';
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
    std::int64_t* d_total = nullptr;
    bool ok = false;
    do {
        const std::size_t bytes = input.size() * sizeof(std::int64_t);
        if (!check(cudaMalloc(&d_input, bytes), "allocate input") ||
            !check(cudaMemcpy(d_input, input.data(), bytes,
                              cudaMemcpyHostToDevice), "copy input H2D") ||
            !check(cudaMalloc(&d_total, sizeof(std::int64_t)),
                   "allocate total")) break;

        if (in_place) {
            d_output = d_input;
        } else if (!check(cudaMalloc(&d_output, bytes), "allocate output")) {
            break;
        }

        exclusive_scan_single_block<<<1, capacity, shared_bytes>>>(
            d_input, d_output, d_total, n);
        if (!check(cudaGetLastError(), "launch single-block scan") ||
            !check(cudaDeviceSynchronize(), "wait for single-block scan") ||
            !check(cudaMemcpy(actual.data(), d_output, bytes,
                              cudaMemcpyDeviceToHost), "copy output D2H")) break;

        std::int64_t actual_total = 0;
        if (!check(cudaMemcpy(&actual_total, d_total, sizeof(actual_total),
                              cudaMemcpyDeviceToHost), "copy total D2H")) break;

        ok = true;
        for (int i = 0; i < n; ++i) {
            if (actual[i] != expected[i]) {
                std::cerr << "mismatch i=" << i << " actual=" << actual[i]
                          << " expected=" << expected[i] << '\n';
                ok = false;
                break;
            }
        }
        if (ok && actual_total != running) {
            std::cerr << "total mismatch actual=" << actual_total
                      << " expected=" << running << '\n';
            ok = false;
        }
        if (ok) {
            const int shown = std::min(n, 16);
            std::cout << "n=" << n << " capacity=" << capacity
                      << " mode=" << (in_place ? "inplace" : "out") << '\n';
            std::cout << "input=[";
            for (int i = 0; i < shown; ++i) {
                if (i) std::cout << ", ";
                std::cout << input[i];
            }
            if (shown < n) std::cout << ", ...";
            std::cout << "]\nexclusive=[";
            for (int i = 0; i < shown; ++i) {
                if (i) std::cout << ", ";
                std::cout << actual[i];
            }
            if (shown < n) std::cout << ", ...";
            std::cout << "]\ntotal=" << actual_total << '\n';
        }
    } while (false);

    if (d_total && !check(cudaFree(d_total), "free total")) ok = false;
    if (!in_place && d_output &&
        !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input && !check(cudaFree(d_input), "free input")) ok = false;

    if (!ok) return 1;
    std::cout << "PASS single-block exclusive scan; every prefix and total checked\n";
    return 0;
}
