#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <iostream>

__global__ void claim_tickets(unsigned int* counter,
                              unsigned int* tickets) {
    const int lane = threadIdx.x;
    tickets[lane] = atomicAdd(counter, 1u);
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    constexpr int threads = 32;
    std::array<unsigned int, threads> tickets{};
    unsigned int counter = 0;
    unsigned int *d_counter = nullptr, *d_tickets = nullptr;
    int device = -1;
    cudaDeviceProp prop = {};
    if (!check(cudaGetDevice(&device), "current device") ||
        !check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
    if (prop.maxThreadsPerBlock < threads || prop.maxThreadsDim[0] < threads) {
        std::cerr << "this lesson requires a 32-thread block\n";
        return 1;
    }

    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_counter, sizeof(unsigned int)),
                   "allocate counter") ||
            !check(cudaMalloc(&d_tickets,
                              tickets.size() * sizeof(unsigned int)),
                   "allocate tickets") ||
            !check(cudaMemset(d_counter, 0, sizeof(unsigned int)),
                   "zero counter") ||
            !check(cudaMemset(d_tickets, 0xff,
                              tickets.size() * sizeof(unsigned int)),
                   "poison tickets")) break;

        claim_tickets<<<1, threads>>>(d_counter, d_tickets);
        if (!check(cudaGetLastError(), "launch") ||
            !check(cudaDeviceSynchronize(), "wait") ||
            !check(cudaMemcpy(&counter, d_counter, sizeof(unsigned int),
                              cudaMemcpyDeviceToHost), "D2H counter") ||
            !check(cudaMemcpy(tickets.data(), d_tickets,
                              tickets.size() * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost), "D2H tickets")) break;

        const auto lane_order = tickets;
        std::sort(tickets.begin(), tickets.end());
        ok = counter == threads;
        for (int i = 0; i < threads && ok; ++i) {
            if (tickets[i] != static_cast<unsigned int>(i)) ok = false;
        }
        if (!ok) {
            std::cerr << "ticket set or final counter mismatch\n";
            break;
        }
        std::cout << "counter=" << counter << " lane0_ticket="
                  << lane_order[0] << " lane31_ticket=" << lane_order[31]
                  << '\n';
    } while (false);

    if (d_tickets && !check(cudaFree(d_tickets), "free tickets")) ok = false;
    if (d_counter && !check(cudaFree(d_counter), "free counter")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS 32 unique old-value tickets; lane-to-ticket order not assumed\n";
    return 0;
}
