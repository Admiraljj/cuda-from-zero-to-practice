#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include "../09_timing/benchmark_stats.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

enum class Method { Single, Pair, Stride, Cub };

struct Config {
    int n = 1048576;
    int threads = 128;
    int grid_cap = 128;
    int repeats = 50;
    int rounds = 5;
};

struct Buffers {
    float* original = nullptr;
    float* a = nullptr;
    float* b = nullptr;
    float* cub_output = nullptr;
    void* cub_temp = nullptr;
    std::size_t scratch_bytes = 0;
    std::size_t cub_bytes = 0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

const char* name(Method method) {
    switch (method) {
        case Method::Single: return "single";
        case Method::Pair: return "pair";
        case Method::Stride: return "stride";
        case Method::Cub: return "cub";
    }
    return "unknown";
}

__device__ void finish_tree(float* values, float* output) {
    const int tid = threadIdx.x;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = values[0];
}

__global__ void single_stage(const float* input, float* output, int n) {
    extern __shared__ float values[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    values[threadIdx.x] = i < n ? input[i] : 0.0f;
    finish_tree(values, output);
}

__global__ void pair_stage(const float* input, float* output, int n) {
    extern __shared__ float values[];
    const int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    float local = 0.0f;
    if (i < n) local += input[i];
    if (i + static_cast<int>(blockDim.x) < n) local += input[i + blockDim.x];
    values[threadIdx.x] = local;
    finish_tree(values, output);
}

__global__ void stride_stage(const float* input, float* output, int n) {
    extern __shared__ float values[];
    float local = 0.0f;
    const int step = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step) local += input[i];
    values[threadIdx.x] = local;
    finish_tree(values, output);
}

bool enqueue(Method method, const Config& cfg, Buffers& buffers,
             const float*& result, int& stages) {
    if (method == Method::Cub) {
        std::size_t available = buffers.cub_bytes;
        stages = -1;  // Internal library kernel count is not queried.
        result = buffers.cub_output;
        return check(cub::DeviceReduce::Sum(buffers.cub_temp, available,
                     buffers.original, buffers.cub_output, cfg.n), "CUB execute");
    }
    const float* read = buffers.original;
    float* write = buffers.a;
    int current_n = cfg.n;
    stages = 0;
    while (true) {
        const int items = method == Method::Pair ? 2 * cfg.threads : cfg.threads;
        int blocks = current_n / items + (current_n % items != 0);
        if (method == Method::Stride) blocks = std::min(blocks, cfg.grid_cap);
        const std::size_t shared_bytes = static_cast<std::size_t>(cfg.threads) * sizeof(float);
        if (method == Method::Single) single_stage<<<blocks, cfg.threads, shared_bytes>>>(read, write, current_n);
        if (method == Method::Pair) pair_stage<<<blocks, cfg.threads, shared_bytes>>>(read, write, current_n);
        if (method == Method::Stride) stride_stage<<<blocks, cfg.threads, shared_bytes>>>(read, write, current_n);
        if (!check(cudaGetLastError(), "stage launch")) return false;
        ++stages;
        if (blocks == 1) {
            result = write;
            return true;
        }
        read = write;
        write = write == buffers.a ? buffers.b : buffers.a;
        current_n = blocks;
    }
}

bool verify(const float* result, long long expected, const Config& cfg,
            Buffers& buffers, const std::vector<float>& input, std::vector<float>& mirror) {
    float value = std::numeric_limits<float>::quiet_NaN();
    if (!check(cudaMemcpy(&value, result, sizeof(float), cudaMemcpyDeviceToHost), "D2H result")) return false;
    if (!std::isfinite(value) || static_cast<double>(value) != static_cast<double>(expected)) {
        std::cerr << "result mismatch: actual=" << value << " expected=" << expected << '\n';
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(cfg.n) * sizeof(float);
    if (!check(cudaMemcpy(mirror.data(), buffers.original, bytes, cudaMemcpyDeviceToHost), "check original input")) return false;
    if (mirror != input) {
        std::cerr << "original input was modified\n";
        return false;
    }
    return true;
}

bool batch(Method method, const Config& cfg, Buffers& buffers, int round,
           long long expected, const std::vector<float>& input, std::vector<float>& mirror,
           double& per_call_ms) {
    if (!check(cudaMemset(buffers.a, 0xff, buffers.scratch_bytes), "poison A") ||
        !check(cudaMemset(buffers.b, 0xff, buffers.scratch_bytes), "poison B") ||
        !check(cudaMemset(buffers.cub_output, 0xff, sizeof(float)), "poison CUB output")) return false;
    if (!check(cudaEventRecord(buffers.start), "record start")) return false;
    const float* result = nullptr;
    int stages = 0;
    for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
        if (!enqueue(method, cfg, buffers, result, stages)) return false;
    }
    if (!check(cudaEventRecord(buffers.stop), "record stop") ||
        !check(cudaEventSynchronize(buffers.stop), "wait stop")) return false;
    float elapsed = 0.0f;
    if (!check(cudaEventElapsedTime(&elapsed, buffers.start, buffers.stop), "event duration")) return false;
    if (!std::isfinite(elapsed) || elapsed <= 0.0f) {
        std::cerr << "invalid timing; increase repeats if the interval is too short\n";
        return false;
    }
    if (!verify(result, expected, cfg, buffers, input, mirror)) return false;
    per_call_ms = static_cast<double>(elapsed) / cfg.repeats;
    std::cout << "method=" << name(method) << " round=" << round
              << " stages_per_call=" << stages << " total_ms=" << elapsed
              << " per_call_ms=" << per_call_ms << " input_unchanged=1\n";
    return true;
}

bool release(Buffers& buffers) {
    bool ok = true;
    if (!check(cudaDeviceSynchronize(), "cleanup wait")) ok = false;
    if (buffers.stop && !check(cudaEventDestroy(buffers.stop), "destroy stop")) ok = false;
    if (buffers.start && !check(cudaEventDestroy(buffers.start), "destroy start")) ok = false;
    if (buffers.cub_temp && !check(cudaFree(buffers.cub_temp), "free CUB scratch")) ok = false;
    if (buffers.cub_output && !check(cudaFree(buffers.cub_output), "free CUB output")) ok = false;
    if (buffers.b && !check(cudaFree(buffers.b), "free B")) ok = false;
    if (buffers.a && !check(cudaFree(buffers.a), "free A")) ok = false;
    if (buffers.original && !check(cudaFree(buffers.original), "free original")) ok = false;
    return ok;
}

bool run(const Config& cfg) {
    std::vector<float> input(cfg.n), mirror(cfg.n);
    long long expected = 0;
    for (int i = 0; i < cfg.n; ++i) {
        const int value = i % 31 - 15;
        input[i] = static_cast<float>(value);
        expected += value;
    }
    std::array<std::vector<double>, 4> samples;
    for (auto& values : samples) values.reserve(cfg.rounds);
    const std::array<Method, 4> methods = {Method::Single, Method::Pair, Method::Stride, Method::Cub};
    int device = -1;
    cudaDeviceProp prop = {};
    if (!check(cudaGetDevice(&device), "current device") ||
        !check(cudaGetDeviceProperties(&prop, device), "device properties")) return false;
    const int capacity = cfg.n / cfg.threads + (cfg.n % cfg.threads != 0);
    const std::size_t shared_bytes = static_cast<std::size_t>(cfg.threads) * sizeof(float);
    if (cfg.threads > prop.maxThreadsPerBlock || cfg.threads > prop.maxThreadsDim[0] ||
        capacity > prop.maxGridSize[0] || shared_bytes > prop.sharedMemPerBlock) {
        std::cerr << "configuration exceeds device limits\n";
        return false;
    }
    std::cout << "device=" << device << " name=" << prop.name << '\n';
    std::cout << "n=" << cfg.n << " threads=" << cfg.threads << " grid_cap=" << cfg.grid_cap
              << " repeats=" << cfg.repeats << " rounds=" << cfg.rounds << " warmups=4\n";
    std::cout << std::fixed << std::setprecision(9);
    Buffers buffers;
    buffers.scratch_bytes = static_cast<std::size_t>(capacity) * sizeof(float);
    const std::size_t input_bytes = static_cast<std::size_t>(cfg.n) * sizeof(float);
    bool ok = false;
    try {
        do {
            if (!check(cudaMalloc(&buffers.original, input_bytes), "allocate original") ||
                !check(cudaMalloc(&buffers.a, buffers.scratch_bytes), "allocate A") ||
                !check(cudaMalloc(&buffers.b, buffers.scratch_bytes), "allocate B") ||
                !check(cudaMalloc(&buffers.cub_output, sizeof(float)), "allocate CUB output")) break;
            if (!check(cudaMemcpy(buffers.original, input.data(), input_bytes, cudaMemcpyHostToDevice), "H2D")) break;
            if (!check(cub::DeviceReduce::Sum(nullptr, buffers.cub_bytes, buffers.original,
                       buffers.cub_output, cfg.n), "CUB storage query")) break;
            buffers.cub_bytes = std::max(buffers.cub_bytes, std::size_t{1});
            if (!check(cudaMalloc(&buffers.cub_temp, buffers.cub_bytes), "allocate CUB scratch") ||
                !check(cudaEventCreate(&buffers.start), "create start") ||
                !check(cudaEventCreate(&buffers.stop), "create stop")) break;
            std::cout << "scratch_capacity_elements=" << capacity << " cub_temp_bytes=" << buffers.cub_bytes << '\n';
            bool warm_ok = true;
            for (Method method : methods) {
                const float* result = nullptr;
                int stages = 0;
                for (int warm = 0; warm < 4; ++warm) {
                    if (!enqueue(method, cfg, buffers, result, stages)) { warm_ok = false; break; }
                }
                if (!warm_ok || !check(cudaDeviceSynchronize(), "warmup wait") ||
                    !verify(result, expected, cfg, buffers, input, mirror)) { warm_ok = false; break; }
            }
            if (!warm_ok) break;
            bool batches_ok = true;
            for (int round = 0; round < cfg.rounds && batches_ok; ++round) {
                for (int position = 0; position < 4; ++position) {
                    const int index = (round + position) % 4;
                    double per_call_ms = 0.0;
                    if (!batch(methods[index], cfg, buffers, round, expected, input, mirror, per_call_ms)) {
                        batches_ok = false;
                        break;
                    }
                    samples[index].push_back(per_call_ms);
                }
            }
            ok = batches_ok;
        } while (false);
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        ok = false;
    }
    if (!release(buffers)) ok = false;
    if (!ok) return false;
    for (int index = 0; index < 4; ++index) {
        const TimingStats stats = summarize(samples[index]);
        const double rate = static_cast<double>(input_bytes) / (stats.median / 1000.0) / 1e9;
        std::cout << "summary=" << name(methods[index]) << " samples=" << stats.count
                  << " min_ms=" << stats.minimum << " median_ms=" << stats.median
                  << " mean_ms=" << stats.mean << " max_ms=" << stats.maximum
                  << " input_GBps=" << rate << '\n';
    }
    std::cout << "PASS reduction benchmark: results and original input checked after each batch; not DRAM traffic\n";
    return true;
}

bool parse(const char* text, int minimum, int maximum, int& output) {
    if (!text || !*text) return false;
    int value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        const int digit = *p - '0';
        if (value > (maximum - digit) / 10) return false;
        value = value * 10 + digit;
        if (value > maximum) return false;
    }
    if (value < minimum) return false;
    output = value;
    return true;
}

int main(int argc, char** argv) {
    try {
        Config cfg;
        if (argc != 1 && (argc != 6 ||
            !parse(argv[1], 1, 1048576, cfg.n) ||
            !parse(argv[2], 2, 1024, cfg.threads) ||
            !parse(argv[3], 1, 256, cfg.grid_cap) ||
            !parse(argv[4], 1, 1000, cfg.repeats) ||
            !parse(argv[5], 1, 21, cfg.rounds))) {
            std::cerr << "Usage: reduction_benchmark [n threads grid_cap repeats rounds]\n";
            return 2;
        }
        int remainder = cfg.threads;
        while (remainder % 2 == 0) remainder /= 2;
        if (remainder != 1) {
            std::cerr << "threads must be a power of two\n";
            return 2;
        }
        return run(cfg) ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
