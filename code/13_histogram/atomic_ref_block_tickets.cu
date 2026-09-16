#include <cuda/atomic>
#include <cuda_runtime.h>

#include "../common/cuda_check.h"

#include <algorithm>
#include <charconv>
#include <iostream>
#include <string_view>
#include <system_error>
#include <vector>

__global__ void make_block_local_tickets(unsigned int* tickets,
                                         unsigned int* final_counters) {
    __shared__ unsigned int counter;

    if (threadIdx.x == 0) counter = 0;
    __syncthreads();

    cuda::atomic_ref<unsigned int, cuda::thread_scope_block> counter_ref(counter);
    const unsigned int ticket =
        counter_ref.fetch_add(1u, cuda::memory_order_relaxed);
    const int output = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    tickets[output] = ticket;

    __syncthreads();
    if (threadIdx.x == 0) {
        final_counters[blockIdx.x] =
            counter_ref.load(cuda::memory_order_relaxed);
    }
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
    int blocks = 4;
    int threads = 64;
    if (argc != 1 &&
        (argc != 3 ||
         !parse_integer(argv[1], 1, 65535, blocks) ||
         !parse_integer(argv[2], 1, 1024, threads))) {
        std::cerr << "Usage: atomic_ref_block_tickets [blocks threads]\n";
        return 2;
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 6) {
        std::cerr << "cuda::atomic_ref requires compute capability 6.0 or newer\n";
        return 77;
    }
    if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0] ||
        blocks > prop.maxGridSize[0]) {
        std::cerr << "launch dimensions exceed this device's limits\n";
        return 2;
    }

    const std::size_t ticket_count =
        static_cast<std::size_t>(blocks) * static_cast<std::size_t>(threads);
    unsigned int* device_tickets = nullptr;
    unsigned int* device_finals = nullptr;
    CUDA_CHECK(cudaMalloc(&device_tickets,
                          ticket_count * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&device_finals,
                          static_cast<std::size_t>(blocks) * sizeof(unsigned int)));

    make_block_local_tickets<<<blocks, threads>>>(device_tickets, device_finals);
    CUDA_CHECK_KERNEL();

    std::vector<unsigned int> tickets(ticket_count);
    std::vector<unsigned int> finals(blocks);
    CUDA_CHECK(cudaMemcpy(tickets.data(), device_tickets,
                          tickets.size() * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(finals.data(), device_finals,
                          finals.size() * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_finals));
    CUDA_CHECK(cudaFree(device_tickets));

    bool ok = true;
    std::vector<unsigned int> sorted_block(threads);
    for (int block = 0; block < blocks; ++block) {
        const auto first = tickets.begin() +
                           static_cast<std::size_t>(block) * threads;
        std::copy(first, first + threads, sorted_block.begin());
        std::sort(sorted_block.begin(), sorted_block.end());
        if (finals[block] != static_cast<unsigned int>(threads)) ok = false;
        for (int i = 0; i < threads; ++i) {
            if (sorted_block[i] != static_cast<unsigned int>(i)) ok = false;
        }
    }
    if (!ok) {
        std::cerr << "block-local ticket invariant failed\n";
        return 1;
    }

    std::cout << "device=" << prop.name << " blocks=" << blocks
              << " threads=" << threads << '\n';
    for (int block = 0; block < std::min(blocks, 4); ++block) {
        std::cout << "block " << block << " final=" << finals[block]
                  << " lane_order=";
        for (int lane = 0; lane < std::min(threads, 16); ++lane) {
            const std::size_t index =
                static_cast<std::size_t>(block) * threads + lane;
            std::cout << tickets[index]
                      << (lane + 1 == std::min(threads, 16) ? '\n' : ',');
        }
    }
    std::cout << "PASS every block independently produced tickets 0..threads-1\n";
    return 0;
}
