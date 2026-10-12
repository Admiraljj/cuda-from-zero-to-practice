#include "../common/cuda_check.h"
#include "../09_timing/benchmark_stats.h"
#include "filter_reference.h"

#include <array>
#include <cstdlib>
#include <limits>

namespace {
constexpr int kBlock = 16, kMaxSide = 9;
__constant__ float c_filter[kMaxSide * kMaxSide];
__constant__ float c_horizontal[kMaxSide], c_vertical[kMaxSide];

__device__ float pixel(const float* input, int w, int h, int x, int y, bool clamp) {
    if (clamp) { x = min(max(x, 0), w - 1); y = min(max(y, 0), h - 1); }
    else if (x < 0 || y < 0 || x >= w || y >= h) return 0.0F;
    return input[static_cast<std::size_t>(y) * w + x];
}

template<bool Constant>
__global__ void filter_naive(const float* input, float* output, const float* weights,
                             int w, int h, int radius, bool clamp) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int side = 2 * radius + 1;
    float acc = 0.0F;
    for (int fy = 0; fy < side; ++fy) for (int fx = 0; fx < side; ++fx) {
        const int index = fy * side + fx;
        const float weight = Constant ? c_filter[index] : weights[index];
        acc += pixel(input, w, h, x + fx - radius, y + fy - radius, clamp) * weight;
    }
    output[static_cast<std::size_t>(y) * w + x] = acc;
}

__global__ void filter_shared(const float* input, float* output,
                              int w, int h, int radius, bool clamp) {
    extern __shared__ float tile[];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int shared_w = blockDim.x + 2 * radius;
    const int shared_h = blockDim.y + 2 * radius;
    const int origin_x = blockIdx.x * blockDim.x - radius;
    const int origin_y = blockIdx.y * blockDim.y - radius;
    const int threads = blockDim.x * blockDim.y;
    for (int linear = tid; linear < shared_w * shared_h; linear += threads) {
        tile[linear] = pixel(input, w, h, origin_x + linear % shared_w,
                            origin_y + linear / shared_w, clamp);
    }
    __syncthreads();
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    if (x >= w || y >= h) return; // All cooperative loads and the sole barrier finished.
    const int side = 2 * radius + 1;
    float acc = 0.0F;
    for (int fy = 0; fy < side; ++fy) for (int fx = 0; fx < side; ++fx) {
        acc += tile[(ty + fy) * shared_w + tx + fx] * c_filter[fy * side + fx];
    }
    output[static_cast<std::size_t>(y) * w + x] = acc;
}

template<bool Horizontal>
__global__ void filter_separable(const float* input, float* output,
                                 int w, int h, int radius, bool clamp) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float acc = 0.0F;
    for (int f = 0; f < 2 * radius + 1; ++f) {
        const int ix = Horizontal ? x + f - radius : x;
        const int iy = Horizontal ? y : y + f - radius;
        acc += pixel(input, w, h, ix, iy, clamp) *
               (Horizontal ? c_horizontal[f] : c_vertical[f]);
    }
    output[static_cast<std::size_t>(y) * w + x] = acc;
}
} // namespace

int main(int argc, char** argv) {
    int w = 131, h = 97, radius = 1, rounds = 5;
    const std::string name = argc > 3 ? argv[3] : "gaussian";
    const std::string boundary = argc > 4 ? argv[4] : "zero";
    const std::string operation = argc > 6 ? argv[6] : "corr";
    if (argc > 8 ||
        (argc > 1 && !filter_example::integer(argv[1], 1, 2048, w)) ||
        (argc > 2 && !filter_example::integer(argv[2], 1, 2048, h)) ||
        (argc > 5 && !filter_example::integer(argv[5], 1, 4, radius)) ||
        (argc > 7 && !filter_example::integer(argv[7], 1, 31, rounds)) ||
        (name != "gaussian" && name != "sobel" && name != "asymmetric" && name != "box") ||
        (boundary != "zero" && boundary != "clamp") ||
        (operation != "corr" && operation != "conv") ||
        (name != "box" && radius != 1) ||
        1LL * w * h * (2 * radius + 1) * (2 * radius + 1) > 268435456LL) {
        std::cerr << "Usage: convolution [width:1..2048] [height:1..2048]"
                     " [gaussian|sobel|asymmetric|box] [zero|clamp] [radius:1..4]"
                     " [corr|conv] [rounds:1..31]\n"
                     "Only box accepts radius>1; CPU terms <=268435456.\n";
        return EXIT_FAILURE;
    }
    const auto f = filter_example::make_filter(name, radius, operation == "conv");
    const bool clamp = boundary == "clamp";
    const int methods = f.separable ? 4 : 3;
    const std::array<const char*, 4> names = {"global-filter", "constant-filter", "shared", "separable"};
    const std::size_t count = static_cast<std::size_t>(w) * h;
    const std::size_t bytes = count * sizeof(float);
    const std::size_t shared_bytes = (kBlock + 2 * radius) * (kBlock + 2 * radius) * sizeof(float);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.maxThreadsPerBlock < 256 || prop.sharedMemPerBlock < shared_bytes) {
        std::cerr << "NOT RUN: insufficient block/shared resources.\n";
        return 77;
    }
    std::vector<float> input(count), output(count), after(count);
    for (std::size_t i = 0; i < count; ++i) input[i] = static_cast<float>((i * 17) % 23) / 8.0F - 1.375F;
    std::vector<double> magnitude;
    const auto expected = filter_example::reference(input, w, h, f, clamp, magnitude);
    float *d_input = nullptr, *d_weights = nullptr, *d_temp = nullptr;
    std::array<float*, 4> d_output{};
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, bytes));
    CUDA_CHECK(cudaMalloc(&d_weights, f.weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, f.weights.data(), f.weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_filter, f.weights.data(), f.weights.size() * sizeof(float)));
    if (f.separable) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_horizontal, f.horizontal.data(), f.horizontal.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_vertical, f.vertical.data(), f.vertical.size() * sizeof(float)));
    }
    for (int method = 0; method < methods; ++method) CUDA_CHECK(cudaMalloc(&d_output[method], bytes));
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const dim3 block(kBlock, kBlock), grid((w + kBlock - 1) / kBlock, (h + kBlock - 1) / kBlock);
    auto launch = [&](int method) {
        if (method == 0) filter_naive<false><<<grid, block>>>(d_input, d_output[method], d_weights, w, h, radius, clamp);
        if (method == 1) filter_naive<true><<<grid, block>>>(d_input, d_output[method], d_weights, w, h, radius, clamp);
        if (method == 2) filter_shared<<<grid, block, shared_bytes>>>(d_input, d_output[method], w, h, radius, clamp);
        if (method == 3) {
            filter_separable<true><<<grid, block>>>(d_input, d_temp, w, h, radius, clamp);
            CUDA_CHECK(cudaGetLastError());
            filter_separable<false><<<grid, block>>>(d_temp, d_output[method], w, h, radius, clamp);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    auto check = [&](int method) {
        CUDA_CHECK(cudaMemcpy(output.data(), d_output[method], bytes, cudaMemcpyDeviceToHost));
        if (!filter_example::validate(output, expected, magnitude, names[method])) std::exit(EXIT_FAILURE);
    };
    std::cout << "GPU=" << prop.name << " width=" << w << " height=" << h
              << " filter=" << name << " radius=" << radius << " boundary=" << boundary
              << " operation=" << operation << " shared_bytes=" << shared_bytes << '\n';
    for (int method = 0; method < methods; ++method) {
        CUDA_CHECK(cudaMemset(d_output[method], 0xFF, bytes));
        if (method == 3) CUDA_CHECK(cudaMemset(d_temp, 0xFF, bytes));
        launch(method);
        CUDA_CHECK(cudaDeviceSynchronize());
        check(method);
        for (int warm = 0; warm < 3; ++warm) launch(method);
        CUDA_CHECK(cudaDeviceSynchronize());
        check(method);
    }
    std::array<std::vector<double>, 4> times;
    for (int round = 0; round < rounds; ++round) {
        for (int slot = 0; slot < methods; ++slot) {
            const int method = (round + slot) % methods;
            CUDA_CHECK(cudaMemset(d_output[method], 0xFF, bytes));
            if (method == 3) CUDA_CHECK(cudaMemset(d_temp, 0xFF, bytes));
            CUDA_CHECK(cudaEventRecord(start));
            launch(method);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (!std::isfinite(ms) || ms <= 0.0F) { std::cerr << "FAIL unresolved Event time.\n"; return EXIT_FAILURE; }
            times[method].push_back(ms);
            check(method);
            std::cout << "sample round=" << round << " method=" << names[method] << " ms=" << ms << '\n';
        }
        CUDA_CHECK(cudaMemcpy(after.data(), d_input, bytes, cudaMemcpyDeviceToHost));
        if (after != input) { std::cerr << "FAIL input changed.\n"; return EXIT_FAILURE; }
    }
    for (int method = 0; method < methods; ++method) {
        const auto s = summarize(times[method]);
        std::cout << "summary " << names[method] << " min_ms=" << s.minimum << " median_ms=" << s.median
                  << " mean_ms=" << s.mean << " max_ms=" << s.maximum
                  << " pixels_per_second=" << count / (s.median * 1.0e-3) << '\n';
    }
    for (int method = 0; method < methods; ++method) CUDA_CHECK(cudaFree(d_output[method]));
    CUDA_CHECK(cudaFree(d_temp)); CUDA_CHECK(cudaFree(d_weights)); CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaEventDestroy(stop)); CUDA_CHECK(cudaEventDestroy(start));
    std::cout << "PASS all eligible methods match the CPU image; input unchanged.\n";
    return EXIT_SUCCESS;
}
