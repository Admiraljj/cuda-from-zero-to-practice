#include <cuda_runtime.h>
#include <array>
#include <iostream>

struct BranchRecord {
    int input;
    int path;
    int iterations;
    int result;
    unsigned positive_mask;
    unsigned reconverged_mask;
};

__global__ void record_branch_paths(const int* input, BranchRecord* records) {
    const int lane = threadIdx.x;
    const int value = input[lane];
    const bool positive = value > 0;
    const unsigned full = 0xffffffffu;
    const unsigned positive_mask = __ballot_sync(full, positive);

    int result = value;
    int path = 0;
    int iterations = 0;
    if (positive) {
        path = 1;
        for (int k = 1; k <= 3; ++k) {
            result += k;
            ++iterations;
        }
    } else {
        path = 2;
        for (int k = 1; k <= 7; ++k) {
            result -= k;
            ++iterations;
        }
    }

    const unsigned reconverged_mask = __ballot_sync(full, true);
    records[lane] = {value, path, iterations, result,
                     positive_mask, reconverged_mask};
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    constexpr int threads = 32;
    std::array<int, threads> input{};
    for (int lane = 0; lane < threads; ++lane) input[lane] = lane % 5 - 2;
    std::array<BranchRecord, threads> records{};
    int device = -1;
    cudaDeviceProp prop = {};
    if (!check(cudaGetDevice(&device), "current device") ||
        !check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
    if (prop.warpSize != threads || prop.maxThreadsPerBlock < threads ||
        prop.maxThreadsDim[0] < threads) {
        std::cerr << "this lesson requires one complete 32-thread warp\n";
        return 1;
    }
    int* d_input = nullptr;
    BranchRecord* d_records = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_input, input.size() * sizeof(int)),
                   "allocate input") ||
            !check(cudaMalloc(&d_records,
                              records.size() * sizeof(BranchRecord)),
                   "allocate records") ||
            !check(cudaMemcpy(d_input, input.data(),
                              input.size() * sizeof(int),
                              cudaMemcpyHostToDevice), "H2D") ||
            !check(cudaMemset(d_records, 0xff,
                              records.size() * sizeof(BranchRecord)),
                   "poison records")) break;
        record_branch_paths<<<1, threads>>>(d_input, d_records);
        if (!check(cudaGetLastError(), "launch") ||
            !check(cudaDeviceSynchronize(), "wait") ||
            !check(cudaMemcpy(records.data(), d_records,
                              records.size() * sizeof(BranchRecord),
                              cudaMemcpyDeviceToHost), "D2H")) break;
        unsigned expected_mask = 0;
        int expected_count = 0;
        for (int lane = 0; lane < threads; ++lane) {
            if (input[lane] > 0) {
                expected_mask |= 1u << lane;
                ++expected_count;
            }
        }
        ok = true;
        for (int lane = 0; lane < threads; ++lane) {
            const bool positive = input[lane] > 0;
            const int expected_path = positive ? 1 : 2;
            const int expected_iterations = positive ? 3 : 7;
            const int expected_result = positive ? input[lane] + 6
                                                 : input[lane] - 28;
            const BranchRecord& row = records[lane];
            if (row.input != input[lane] ||
                row.path != expected_path ||
                row.iterations != expected_iterations ||
                row.result != expected_result ||
                row.positive_mask != expected_mask ||
                row.reconverged_mask != 0xffffffffu) {
                std::cerr << "branch record mismatch lane=" << lane << '\n';
                ok = false;
                break;
            }
            std::cout << "lane=" << lane << " input=" << row.input
                      << " path=" << row.path
                      << " iterations=" << row.iterations
                      << " result=" << row.result << '\n';
        }
        if (ok) {
            std::cout << "positive_mask=0x" << std::hex << expected_mask
                      << std::dec << " positive_count="
                      << expected_count << '\n';
        }
    } while (false);
    if (d_records && !check(cudaFree(d_records), "free records")) ok = false;
    if (d_input && !check(cudaFree(d_input), "free input")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS logical branch records checked; no timing claim\n";
    return 0;
}
