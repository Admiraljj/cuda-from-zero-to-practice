#include "../common/cuda_check.h"
#include "../09_timing/benchmark_stats.h"
#include "../17_convolution/filter_reference.h"
#include <array>
#include <cstdlib>

namespace {
template<int Items>
__global__ void sweep_kernel(const float* input, float* output, int n, int shared_bytes) {
    extern __shared__ float reserved[];
    if (shared_bytes != 0) {
        if (threadIdx.x == 0) reserved[0] = 0.0F;
        __syncthreads();
    }
    const float bias = shared_bytes == 0 ? 0.0F : reserved[0];
    const std::size_t first =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * Items;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x * Items;
    for (std::size_t base = first; base < static_cast<std::size_t>(n); base += stride) {
        float values[Items]{};
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            if (base + j < static_cast<std::size_t>(n)) values[j] = input[base + j];
        }
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            if (base + j < static_cast<std::size_t>(n)) output[base + j] = 2.0F * values[j] + 1.0F + bias;
        }
    }
}
}

int main(int argc, char** argv) {
    int n = 1048576, rounds = 5, repeats = 8;
    if (argc > 4 || (argc > 1 && !filter_example::integer(argv[1], 0, 16777216, n)) ||
        (argc > 2 && !filter_example::integer(argv[2], 1, 31, rounds)) ||
        (argc > 3 && !filter_example::integer(argv[3], 1, 1000, repeats))) {
        std::cerr << "Usage: resource_sweep [n:0..16777216] [rounds:1..31] [repeats:1..1000]\n";
        return EXIT_FAILURE;
    }
    if (n == 0) { std::cout << "PASS empty input; no GPU launch.\n"; return EXIT_SUCCESS; }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cudaFuncAttributes attr1{}, attr4{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr1, sweep_kernel<1>));
    CUDA_CHECK(cudaFuncGetAttributes(&attr4, sweep_kernel<4>));
    int minimum_grid = 0, suggested_block = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minimum_grid, &suggested_block, sweep_kernel<1>, 0, 0));
    std::cout << "GPU=" << prop.name << " SMs=" << prop.multiProcessorCount
              << " max_threads_SM=" << prop.maxThreadsPerMultiProcessor
              << " registers_SM=" << prop.regsPerMultiprocessor
              << " shared_SM=" << prop.sharedMemPerMultiprocessor
              << " warp_size=" << prop.warpSize << '\n'
              << "potentialBlockSize items1 shared0: minGrid=" << minimum_grid
              << " suggestedBlock=" << suggested_block << '\n';
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> input(n), output(n), after(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i % 127 - 63) / 8.0F;
    float *di = nullptr, *dout = nullptr;
    CUDA_CHECK(cudaMalloc(&di, bytes)); CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(di, input.data(), bytes, cudaMemcpyHostToDevice));
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

    struct Candidate { int items, threads, shared, blocks_sm, grid; std::vector<double> times; };
    std::vector<Candidate> candidates;
    for (int items : {1, 4}) for (int threads : {64, 128, 256, 512}) for (int shared : {0, 16384, 32768}) {
        const auto& attr = items == 1 ? attr1 : attr4;
        if (threads > prop.maxThreadsPerBlock || threads > attr.maxThreadsPerBlock ||
            static_cast<std::size_t>(shared) + attr.sharedSizeBytes > prop.sharedMemPerBlock ||
            shared > attr.maxDynamicSharedSizeBytes) {
            std::cout << "SKIP unsupported candidate items=" << items << " threads=" << threads << " shared=" << shared << '\n';
            continue;
        }
        int active = 0;
        if (items == 1) CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, sweep_kernel<1>, threads, shared));
        else CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, sweep_kernel<4>, threads, shared));
        if (active == 0) continue;
        const int covering_grid = (n + threads * items - 1) / (threads * items);
        // Two estimated residency waves; grid-stride covers the remaining work.
        const int grid = std::min(covering_grid, 2 * prop.multiProcessorCount * active);
        candidates.push_back({items, threads, shared, active, grid, {}});
    }
    if (candidates.empty()) { std::cerr << "NOT RUN: no supported candidate.\n"; return 77; }
    auto launch = [&](const Candidate& c) {
        if (c.items == 1) sweep_kernel<1><<<c.grid, c.threads, c.shared>>>(di, dout, n, c.shared);
        else sweep_kernel<4><<<c.grid, c.threads, c.shared>>>(di, dout, n, c.shared);
        CUDA_CHECK(cudaGetLastError());
    };
    auto validate = [&]() {
        CUDA_CHECK(cudaMemcpy(output.data(), dout, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after.data(), di, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(output[i]) || output[i] != 2.0F * input[i] + 1.0F || after[i] != input[i]) {
                std::cerr << "FAIL index=" << i << '\n'; std::exit(EXIT_FAILURE);
            }
        }
    };
    for (const auto& c : candidates) {
        CUDA_CHECK(cudaMemset(dout, 0xFF, bytes));
        launch(c); CUDA_CHECK(cudaDeviceSynchronize()); validate();
        for (int warm = 0; warm < 3; ++warm) launch(c);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (int round = 0; round < rounds; ++round) {
        for (std::size_t slot = 0; slot < candidates.size(); ++slot) {
            auto& c = candidates[(static_cast<std::size_t>(round) + slot) % candidates.size()];
            CUDA_CHECK(cudaMemset(dout, 0xFF, bytes));
            CUDA_CHECK(cudaEventRecord(start));
            for (int repeat = 0; repeat < repeats; ++repeat) launch(c);
            CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0F; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (!std::isfinite(ms) || ms <= 0.0F) { std::cerr << "FAIL unresolved Event time.\n"; return EXIT_FAILURE; }
            c.times.push_back(ms / repeats);
            validate();
        }
    }
    const int maximum_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    for (const auto& c : candidates) {
        const auto& attr = c.items == 1 ? attr1 : attr4;
        const int warps = c.blocks_sm * ((c.threads + prop.warpSize - 1) / prop.warpSize);
        const auto stats = summarize(c.times);
        std::cout << "items=" << c.items << " threads=" << c.threads << " dynamic_shared=" << c.shared
                  << " registers_thread=" << attr.numRegs << " local_bytes_thread=" << attr.localSizeBytes
                  << " blocks_SM=" << c.blocks_sm << " predicted_occupancy=" << warps / static_cast<double>(maximum_warps)
                  << " grid=" << c.grid << " min_ms=" << stats.minimum << " median_ms=" << stats.median
                  << " mean_ms=" << stats.mean << " max_ms=" << stats.maximum << '\n';
    }
    CUDA_CHECK(cudaEventDestroy(stop)); CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(dout)); CUDA_CHECK(cudaFree(di));
    std::cout << "PASS every supported candidate writes each expected value; input unchanged.\n";
    return EXIT_SUCCESS;
}
