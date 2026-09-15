#include <cuda_runtime.h>
#include <array>
#include <iostream>
#include <vector>

constexpr int lanes = 32;
constexpr int rounds = 6;

__global__ void trace_full_warp(int* trace) {
    const int lane = threadIdx.x;
    const unsigned full = 0xffffffffu;
    int value = lane + 1;
    trace[lane] = value;
    int round = 1;
    for (int offset = 16; offset > 0; offset /= 2) {
        const int received = __shfl_down_sync(full, value, offset);
        if (lane < offset) value += received;
        trace[round * lanes + lane] = value;
        ++round;
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
    if (prop.warpSize != lanes || prop.maxThreadsPerBlock < lanes ||
        prop.maxThreadsDim[0] < lanes) {
        std::cerr << "this lesson requires one complete 32-thread warp\n";
        return 1;
    }
    std::array<int, rounds * lanes> trace{};
    int* d_trace = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_trace, trace.size() * sizeof(int)), "allocate trace") ||
            !check(cudaMemset(d_trace, 0xff, trace.size() * sizeof(int)), "poison trace")) break;
        trace_full_warp<<<1, lanes>>>(d_trace);
        if (!check(cudaGetLastError(), "launch") ||
            !check(cudaDeviceSynchronize(), "wait") ||
            !check(cudaMemcpy(trace.data(), d_trace, trace.size() * sizeof(int),
                              cudaMemcpyDeviceToHost), "D2H")) break;
        std::array<int, lanes> current{}, next{};
        for (int lane = 0; lane < lanes; ++lane) current[lane] = lane + 1;
        ok = true;
        for (int round = 0; round < rounds; ++round) {
            for (int lane = 0; lane < lanes; ++lane) {
                if (trace[round * lanes + lane] != current[lane]) {
                    std::cerr << "trace mismatch round=" << round << " lane=" << lane
                              << " actual=" << trace[round * lanes + lane]
                              << " expected=" << current[lane] << '\n';
                    ok = false;
                    break;
                }
            }
            if (!ok || round == rounds - 1) break;
            const int offset = 16 >> round;
            for (int lane = 0; lane < lanes; ++lane) {
                next[lane] = lane < offset ? current[lane] + current[lane + offset]
                                           : current[lane];
            }
            current = next;
        }
        if (!ok) break;
        const std::array<int, rounds> offsets = {0, 16, 8, 4, 2, 1};
        for (int round = 0; round < rounds; ++round) {
            std::cout << "round=" << round << " offset=" << offsets[round]
                      << " lane0=" << trace[round * lanes]
                      << " lane1=" << trace[round * lanes + 1]
                      << " lane31=" << trace[round * lanes + 31] << '\n';
        }
        if (trace[(rounds - 1) * lanes] != 528) {
            std::cerr << "lane 0 did not receive the full sum\n";
            ok = false;
        }
    } while (false);
    if (d_trace && !check(cudaFree(d_trace), "free trace")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS full-warp reduction trace; lane0 sum=528\n";
    return 0;
}
