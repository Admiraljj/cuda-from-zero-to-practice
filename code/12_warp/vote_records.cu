#include <cuda_runtime.h>
#include <array>
#include <iostream>

struct VoteRecord {
    unsigned valid_mask;
    unsigned positive_mask;
    int valid_count;
    int positive_count;
    int any_positive;
    int all_valid_positive;
    int positive_rank;
};

__global__ void inspect_votes(const int* input, VoteRecord* records, int n) {
    const int i = threadIdx.x;
    const int lane = i % 32;
    const bool valid = i < n;
    const int value = valid ? input[i] : 0;
    const bool positive = valid && value > 0;
    const unsigned full = 0xffffffffu;
    VoteRecord result;
    result.valid_mask = __ballot_sync(full, valid);
    result.positive_mask = __ballot_sync(full, positive);
    result.valid_count = __popc(result.valid_mask);
    result.positive_count = __popc(result.positive_mask);
    result.any_positive = __any_sync(full, positive) != 0;
    result.all_valid_positive = __all_sync(full, !valid || value > 0) != 0;
    const unsigned lower_lanes = (1u << lane) - 1u;
    result.positive_rank = positive ? __popc(result.positive_mask & lower_lanes) : -1;
    records[i] = result;
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    constexpr int n = 37, threads = 64;
    std::array<int, n> input{};
    for (int i = 0; i < n; ++i) input[i] = i % 7 - 3;
    std::array<VoteRecord, threads> records{};
    int device = -1;
    cudaDeviceProp prop = {};
    if (!check(cudaGetDevice(&device), "current device") ||
        !check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
    if (prop.warpSize != 32 || prop.maxThreadsPerBlock < threads || prop.maxThreadsDim[0] < threads) {
        std::cerr << "this lesson requires two complete 32-thread warps\n";
        return 1;
    }
    int* d_input = nullptr;
    VoteRecord* d_records = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_input, input.size() * sizeof(int)), "allocate input") ||
            !check(cudaMalloc(&d_records, records.size() * sizeof(VoteRecord)), "allocate records") ||
            !check(cudaMemcpy(d_input, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice), "H2D") ||
            !check(cudaMemset(d_records, 0xff, records.size() * sizeof(VoteRecord)), "poison records")) break;
        inspect_votes<<<1, threads>>>(d_input, d_records, n);
        if (!check(cudaGetLastError(), "launch") ||
            !check(cudaDeviceSynchronize(), "wait") ||
            !check(cudaMemcpy(records.data(), d_records, records.size() * sizeof(VoteRecord), cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int i = 0; i < threads; ++i) {
            const int warp = i / 32, lane = i % 32, base = warp * 32;
            unsigned valid_mask = 0, positive_mask = 0;
            int valid_count = 0, positive_count = 0, rank = 0;
            for (int source_lane = 0; source_lane < 32; ++source_lane) {
                const int index = base + source_lane;
                if (index >= n) continue;
                valid_mask |= 1u << source_lane;
                ++valid_count;
                if (input[index] > 0) {
                    positive_mask |= 1u << source_lane;
                    ++positive_count;
                    if (source_lane < lane) ++rank;
                }
            }
            if (i >= n || input[i] <= 0) rank = -1;
            const VoteRecord& row = records[i];
            if (row.valid_mask != valid_mask || row.positive_mask != positive_mask ||
                row.valid_count != valid_count || row.positive_count != positive_count ||
                row.any_positive != (positive_count > 0) ||
                row.all_valid_positive != (positive_count == valid_count) ||
                row.positive_rank != rank) {
                std::cerr << "vote mismatch at i=" << i << '\n';
                ok = false;
                break;
            }
            if (lane == 0) {
                std::cout << "warp=" << warp << " valid_mask=0x" << std::hex << row.valid_mask
                          << " positive_mask=0x" << row.positive_mask << std::dec
                          << " valid_count=" << row.valid_count << " positive_count=" << row.positive_count
                          << " any=" << row.any_positive << " all_valid=" << row.all_valid_positive << '\n';
            }
            if (warp == 1 && lane < 5) {
                std::cout << "i=" << i << " value=" << input[i] << " positive_rank=" << row.positive_rank << '\n';
            }
        }
    } while (false);
    if (d_records && !check(cudaFree(d_records), "free records")) ok = false;
    if (d_input && !check(cudaFree(d_input), "free input")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS 64 vote records checked; 37 valid inputs, two complete warps\n";
    return 0;
}
