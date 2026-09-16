#include <cuda_runtime.h>

#include "../common/cuda_check.h"

#include <algorithm>
#include <charconv>
#include <iostream>
#include <limits>
#include <string_view>
#include <system_error>
#include <vector>

constexpr unsigned int kRejected =
    std::numeric_limits<unsigned int>::max();

__device__ unsigned int claim_below_limit(unsigned int* counter,
                                          unsigned int limit) {
    unsigned int observed = atomicCAS(counter, 0u, 0u);
    while (observed < limit) {
        const unsigned int desired = observed + 1u;
        const unsigned int actual = atomicCAS(counter, observed, desired);
        if (actual == observed) return observed;
        observed = actual;
    }
    return kRejected;
}

__global__ void claim_once(unsigned int* counter,
                           unsigned int limit,
                           unsigned int* claims) {
    const int tid = threadIdx.x;
    claims[tid] = claim_below_limit(counter, limit);
}

bool parse_integer(const char* text, int minimum, int maximum, int& value) {
    if (text == nullptr) return false;
    const std::string_view view(text);
    int parsed = 0;
    const auto result = std::from_chars(view.data(), view.data() + view.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != view.data() + view.size() ||
        parsed < minimum || parsed > maximum) return false;
    value = parsed;
    return true;
}

int main(int argc, char** argv) {
    int threads = 32;
    int limit = 10;
    if (argc != 1 &&
        (argc != 3 ||
         !parse_integer(argv[1], 1, 1024, threads) ||
         !parse_integer(argv[2], 0, threads, limit))) {
        std::cerr << "Usage: cas_bounded_counter [threads limit]\n";
        return 2;
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0]) {
        std::cerr << "threads exceeds this device's one-dimensional block limit\n";
        return 2;
    }

    unsigned int* device_counter = nullptr;
    unsigned int* device_claims = nullptr;
    CUDA_CHECK(cudaMalloc(&device_counter, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&device_claims,
                          static_cast<std::size_t>(threads) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(device_counter, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(device_claims, 0xff,
                          static_cast<std::size_t>(threads) * sizeof(unsigned int)));

    claim_once<<<1, threads>>>(device_counter,
                               static_cast<unsigned int>(limit),
                               device_claims);
    CUDA_CHECK_KERNEL();

    unsigned int counter = 0;
    std::vector<unsigned int> lane_claims(threads, kRejected);
    CUDA_CHECK(cudaMemcpy(&counter, device_counter, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lane_claims.data(), device_claims,
                          lane_claims.size() * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_claims));
    CUDA_CHECK(cudaFree(device_counter));

    std::vector<unsigned int> accepted;
    for (unsigned int claim : lane_claims) {
        if (claim != kRejected) accepted.push_back(claim);
    }
    std::sort(accepted.begin(), accepted.end());

    bool ok = counter == static_cast<unsigned int>(limit) &&
              accepted.size() == static_cast<std::size_t>(limit);
    for (int i = 0; i < limit && ok; ++i) {
        ok = accepted[i] == static_cast<unsigned int>(i);
    }
    if (!ok) {
        std::cerr << "counter or accepted claim set mismatch\n";
        return 1;
    }

    std::cout << "device=" << prop.name << " threads=" << threads
              << " limit=" << limit << " final_counter=" << counter
              << " accepted=" << accepted.size()
              << " rejected=" << threads - static_cast<int>(accepted.size())
              << '\n';
    std::cout << "lane_claims=";
    for (int lane = 0; lane < threads; ++lane) {
        if (lane_claims[lane] == kRejected) std::cout << 'X';
        else std::cout << lane_claims[lane];
        std::cout << (lane + 1 == threads ? '\n' : ',');
    }
    std::cout << "PASS accepted claims are exactly 0..limit-1; lane order not assumed\n";
    return 0;
}
