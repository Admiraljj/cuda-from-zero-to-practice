#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

constexpr int width = 64;
constexpr int phases = 3;

__global__ void reuse_tile(float* output) {
    if (blockDim.x != width) return;
    __shared__ float tile[width];
    const int tid = threadIdx.x;
    for (int phase = 0; phase < phases; ++phase) {
        tile[tid] = static_cast<float>(100 * phase + tid);
        __syncthreads();  // All producers finished this phase.
        output[phase * width + tid] = tile[(tid + 1) % width];
        __syncthreads();  // All consumers finished before tile is overwritten.
    }
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    try {
        constexpr int count = width * phases;
        constexpr std::size_t bytes = count * sizeof(float);
        std::vector<float> output(count);
        float* d_output = nullptr;
        bool ok = false;
        do {
            if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
            if (!check(cudaMemset(d_output, 0xff, bytes), "poison output")) break;
            reuse_tile<<<1, width>>>(d_output);
            if (!check(cudaGetLastError(), "launch")) break;
            if (!check(cudaDeviceSynchronize(), "wait")) break;
            if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
            ok = true;
            for (int phase = 0; phase < phases; ++phase) {
                for (int tid = 0; tid < width; ++tid) {
                    const float expected = static_cast<float>(100 * phase + (tid + 1) % width);
                    const float actual = output[phase * width + tid];
                    if (!std::isfinite(actual) || actual != expected) {
                        std::cerr << "mismatch phase=" << phase << " tid=" << tid << '\n';
                        ok = false;
                    }
                }
            }
        } while (false);
        if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
        if (!ok) return 1;
        for (int phase = 0; phase < phases; ++phase) {
            std::cout << "phase=" << phase << " first=" << output[phase * width]
                      << " last=" << output[phase * width + width - 1] << '\n';
        }
        std::cout << "PASS shared reuse: 3 phases, 64 threads, 192 checked values\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
