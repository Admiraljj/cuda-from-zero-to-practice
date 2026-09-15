#include <cuda_runtime.h>
#include <array>
#include <iostream>

struct Result {
    unsigned selected_mask;
    int selected_count;
    int sum;
};

__device__ int warp_sum_full(int value) {
    const unsigned full = 0xffffffffu;
    const int lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        const int received = __shfl_down_sync(full, value, offset);
        if (lane < offset) value += received;
    }
    return value;
}

__global__ void sum_selected(Result* output) {
    const int lane = threadIdx.x;
    const unsigned full = 0xffffffffu;
    const bool selected = lane == 0 || lane == 3 || lane == 7 || lane == 8 ||
                          lane == 15 || lane == 21 || lane == 31;
    const unsigned selected_mask = __ballot_sync(full, selected);
    const int contribution = selected ? lane + 1 : 0;
    const int total = warp_sum_full(contribution);
    if (lane == 0) {
        output->selected_mask = selected_mask;
        output->selected_count = __popc(selected_mask);
        output->sum = total;
    }
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
    if (prop.warpSize != 32 || prop.maxThreadsPerBlock < 32 ||
        prop.maxThreadsDim[0] < 32) {
        std::cerr << "this lesson requires one complete 32-thread warp\n";
        return 1;
    }
    Result result = {};
    Result* d_result = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_result, sizeof(Result)), "allocate result") ||
            !check(cudaMemset(d_result, 0xff, sizeof(Result)), "poison result")) break;
        sum_selected<<<1, 32>>>(d_result);
        if (!check(cudaGetLastError(), "launch") ||
            !check(cudaDeviceSynchronize(), "wait") ||
            !check(cudaMemcpy(&result, d_result, sizeof(Result),
                              cudaMemcpyDeviceToHost), "D2H")) break;
        constexpr std::array<int, 7> selected = {0, 3, 7, 8, 15, 21, 31};
        unsigned expected_mask = 0;
        int expected_sum = 0;
        for (int lane : selected) {
            expected_mask |= 1u << lane;
            expected_sum += lane + 1;
        }
        ok = result.selected_mask == expected_mask &&
             result.selected_count == static_cast<int>(selected.size()) &&
             result.sum == expected_sum;
        if (!ok) {
            std::cerr << "selected reduction mismatch\n";
            break;
        }
        std::cout << "selected_mask=0x" << std::hex << result.selected_mask << std::dec
                  << " selected_count=" << result.selected_count
                  << " sum=" << result.sum << '\n';
    } while (false);
    if (d_result && !check(cudaFree(d_result), "free result")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS sparse selection via identity values and full-warp communication\n";
    return 0;
}
