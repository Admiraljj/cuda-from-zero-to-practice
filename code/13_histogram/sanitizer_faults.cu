#include <cuda_runtime.h>

#include "../common/cuda_check.h"

#include <array>
#include <iostream>
#include <string_view>

constexpr int kThreads = 32;

__global__ void safe_shared_exchange(const int* input, int* output) {
    __shared__ int values[kThreads];
    const int tid = static_cast<int>(threadIdx.x);
    values[tid] = input[tid];
    __syncthreads();
    output[tid] = values[(tid + 1) % kThreads];
}

__global__ void out_of_bounds_write(const int* input, int* output) {
    const int tid = static_cast<int>(threadIdx.x);
    output[tid + 1] = input[tid];
}

__global__ void shared_write_race(int* output) {
    __shared__ int value;
    const int tid = static_cast<int>(threadIdx.x);
    value = tid;
    __syncthreads();
    output[tid] = value;
}

__global__ void read_uninitialized_global(const int* input, int* output) {
    const int tid = static_cast<int>(threadIdx.x);
    output[tid] = input[tid];
}

__global__ void divergent_barrier(const int* input, int* output) {
    const int tid = static_cast<int>(threadIdx.x);
    if ((tid & 1) == 0) __syncthreads();
    output[tid] = input[tid];
}

bool is_mode(std::string_view mode) {
    return mode == "safe" || mode == "oob" || mode == "race" ||
           mode == "uninit" || mode == "divergent";
}

int main(int argc, char** argv) {
    const std::string_view mode = argc == 2 ? argv[1] : "safe";
    if (argc > 2 || !is_mode(mode)) {
        std::cerr << "Usage: sanitizer_faults [safe|oob|race|uninit|divergent]\n";
        return 2;
    }

    std::array<int, kThreads> host_input{};
    for (int i = 0; i < kThreads; ++i) host_input[i] = 100 + i;
    int* device_input = nullptr;
    int* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, kThreads * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_output, kThreads * sizeof(int)));
    CUDA_CHECK(cudaMemset(device_output, 0, kThreads * sizeof(int)));
    if (mode != "uninit") {
        CUDA_CHECK(cudaMemcpy(device_input, host_input.data(),
                              kThreads * sizeof(int), cudaMemcpyHostToDevice));
    }

    if (mode == "safe") {
        safe_shared_exchange<<<1, kThreads>>>(device_input, device_output);
    } else {
        std::cerr << "INTENTIONAL FAULT MODE: " << mode << '\n';
        if (mode == "oob") {
            out_of_bounds_write<<<1, kThreads>>>(device_input, device_output);
        } else if (mode == "race") {
            shared_write_race<<<1, kThreads>>>(device_output);
        } else if (mode == "uninit") {
            read_uninitialized_global<<<1, kThreads>>>(device_input, device_output);
        } else {
            divergent_barrier<<<1, kThreads>>>(device_input, device_output);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::array<int, kThreads> host_output{};
    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                          kThreads * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_input));

    if (mode == "safe") {
        for (int i = 0; i < kThreads; ++i) {
            const int expected = host_input[(i + 1) % kThreads];
            if (host_output[i] != expected) {
                std::cerr << "safe mode mismatch at " << i << '\n';
                return 1;
            }
        }
        std::cout << "PASS safe mode: synchronized shared exchange\n";
    } else {
        std::cout << "Fault mode completed; inspect Compute Sanitizer output\n";
    }
    return 0;
}
