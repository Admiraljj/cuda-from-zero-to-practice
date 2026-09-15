#include <cuda_runtime.h>
#include <array>
#include <iostream>

struct Record {
    int value;
    int from_zero;
    int down_one;
    int up_one;
    int partner;
    int group_zero;
};

__global__ void inspect_routes(Record* records) {
    const int lane = threadIdx.x;
    const unsigned full = 0xffffffffu;
    const int value = 100 + lane;
    Record result;
    result.value = value;
    result.from_zero = __shfl_sync(full, value, 0);
    result.down_one = __shfl_down_sync(full, value, 1);
    result.up_one = __shfl_up_sync(full, value, 1);
    result.partner = __shfl_xor_sync(full, value, 1);
    result.group_zero = __shfl_sync(full, value, 0, 8);
    records[lane] = result;
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    int device = -1;
    cudaDeviceProp prop = {};
    if (!check(cudaGetDevice(&device), "current device") ||
        !check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
    if (prop.warpSize != 32 || prop.maxThreadsPerBlock < 32 || prop.maxThreadsDim[0] < 32) {
        std::cerr << "this lesson requires a complete 32-thread warp\n";
        return 1;
    }
    std::array<Record, 32> records{};
    Record* d_records = nullptr;
    const std::size_t bytes = records.size() * sizeof(Record);
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_records, bytes), "allocate records") ||
            !check(cudaMemset(d_records, 0xff, bytes), "poison records")) break;
        inspect_routes<<<1, 32>>>(d_records);
        if (!check(cudaGetLastError(), "launch") ||
            !check(cudaDeviceSynchronize(), "wait") ||
            !check(cudaMemcpy(records.data(), d_records, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int lane = 0; lane < 32; ++lane) {
            const Record& row = records[lane];
            const int down_source = lane < 31 ? lane + 1 : lane;
            const int up_source = lane > 0 ? lane - 1 : lane;
            const int partner_source = lane % 2 == 0 ? lane + 1 : lane - 1;
            const int group_source = (lane / 8) * 8;
            if (row.value != 100 + lane || row.from_zero != 100 ||
                row.down_one != 100 + down_source || row.up_one != 100 + up_source ||
                row.partner != 100 + partner_source || row.group_zero != 100 + group_source) {
                std::cerr << "route mismatch at lane=" << lane << '\n';
                ok = false;
                break;
            }
            std::cout << "lane=" << lane << " value=" << row.value
                      << " from_zero=" << row.from_zero << " down_one=" << row.down_one
                      << " up_one=" << row.up_one << " partner=" << row.partner
                      << " group_zero=" << row.group_zero << '\n';
        }
    } while (false);
    if (d_records && !check(cudaFree(d_records), "free records")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS 32 shuffle routes; all six fields checked for every lane\n";
    return 0;
}
