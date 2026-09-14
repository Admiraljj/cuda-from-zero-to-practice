#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool parse_decimal(const char* text, int minimum, int maximum, int& result) {
    if (text == nullptr || *text == '\0') return false;
    int value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return false;
        const int digit = *p - '0';
        if (value > (maximum - digit) / 10) return false;
        value = value * 10 + digit;
        if (value > maximum) return false;
    }
    if (value < minimum) return false;
    result = value;
    return true;
}

int main(int argc, char** argv) {
    try {
        int n = 1003;
        if (argc != 1 && (argc != 2 || !parse_decimal(argv[1], 0, 1048576, n))) {
            std::cerr << "Usage: cub_sum [n]; n=0..1048576\n";
            return 2;
        }
        if (n == 0) {
            std::cout << "EMPTY sum=0; no CUB or CUDA call\n";
            return 0;
        }
        std::vector<float> input(n);
        long long expected = 0;
        for (int i = 0; i < n; ++i) {
            const int value = i % 31 - 15;
            input[i] = static_cast<float>(value);
            expected += value;
        }
        const std::size_t input_bytes = static_cast<std::size_t>(n) * sizeof(float);
        float *d_input = nullptr, *d_output = nullptr;
        void* d_temp = nullptr;
        std::size_t queried_temp_bytes = 0;
        std::size_t allocated_temp_bytes = 0;
        float result = std::numeric_limits<float>::quiet_NaN();
        bool ok = false;
        do {
            if (!check(cudaMalloc(&d_input, input_bytes), "allocate input")) break;
            if (!check(cudaMalloc(&d_output, sizeof(float)), "allocate output")) break;
            if (!check(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice), "H2D")) break;
            if (!check(cudaMemset(d_output, 0xff, sizeof(float)), "poison output")) break;
            if (!check(cub::DeviceReduce::Sum(
                           nullptr, queried_temp_bytes, d_input, d_output, n),
                       "CUB storage query")) break;
            allocated_temp_bytes = std::max(queried_temp_bytes, std::size_t{1});
            if (!check(cudaMalloc(&d_temp, allocated_temp_bytes), "allocate scratch")) break;
            std::size_t available_temp_bytes = allocated_temp_bytes;
            if (!check(cub::DeviceReduce::Sum(
                           d_temp, available_temp_bytes, d_input, d_output, n),
                       "CUB sum execute")) break;
            if (!check(cudaDeviceSynchronize(), "wait for CUB")) break;
            if (!check(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost), "D2H scalar")) break;
            ok = std::isfinite(result) && static_cast<double>(result) == static_cast<double>(expected);
            if (!ok) std::cerr << "CUB result mismatch\n";
        } while (false);
        if (d_temp != nullptr && !check(cudaFree(d_temp), "free scratch")) ok = false;
        if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
        if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
        if (!ok) return 1;
        std::cout << "n=" << n << " queried_temp_bytes=" << queried_temp_bytes
                  << " allocated_temp_bytes=" << allocated_temp_bytes << '\n';
        std::cout << "sum=" << result << " expected=" << expected << '\n';
        std::cout << "PASS CUB device sum; scratch query and execution checked\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
