#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void stencil_dynamic(const float* input, float* output, int n) {
    extern __shared__ float tile[];
    const int tid = threadIdx.x;
    const int width = blockDim.x;
    const int base = blockIdx.x * width;
    const int i = base + tid;
    tile[tid + 1] = i < n ? input[i] : 0.0f;
    if (tid == 0) tile[0] = base > 0 && base - 1 < n ? input[base - 1] : 0.0f;
    if (tid == width - 1) {
        tile[width + 1] = base + width < n ? input[base + width] : 0.0f;
    }
    __syncthreads();
    if (i < n) output[i] = tile[tid] + tile[tid + 1] + tile[tid + 2];
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool run_case(int n, int threads) {
    if (n == 0) {
        std::cout << "EMPTY n=0 threads=" << threads << "; no allocation or kernel for this case\n";
        return true;
    }
    const int blocks = n / threads + (n % threads != 0);
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t shared_bytes = static_cast<std::size_t>(threads + 2) * sizeof(float);
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i % 7 - 3);
    float *d_input = nullptr, *d_output = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input")) break;
        if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
        if (!check(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;
        if (!check(cudaMemset(d_output, 0xff, bytes), "poison output")) break;
        stencil_dynamic<<<blocks, threads, shared_bytes>>>(d_input, d_output, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "wait")) break;
        if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int i = 0; i < n; ++i) {
            const int left = i > 0 ? (i - 1) % 7 - 3 : 0;
            const int center = i % 7 - 3;
            const int right = i + 1 < n ? (i + 1) % 7 - 3 : 0;
            const float expected = static_cast<float>(left + center + right);
            if (!std::isfinite(output[i]) || output[i] != expected) {
                std::cerr << "mismatch n=" << n << " threads=" << threads << " i=" << i << '\n';
                ok = false;
                break;
            }
        }
    } while (false);
    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (ok) {
        std::cout << "PASS n=" << n << " threads=" << threads << " blocks=" << blocks
                  << " shared_bytes=" << shared_bytes << '\n';
    }
    return ok;
}

int main() {
    try {
        int device = -1;
        cudaDeviceProp properties = {};
        if (!check(cudaGetDevice(&device), "current device")) return 1;
        if (!check(cudaGetDeviceProperties(&properties, device), "device properties")) return 1;
        const int widths[] = {1, 4, 7, 32};
        const int sizes[] = {0, 1, 3, 4, 5, 6, 7, 8, 10, 17, 31, 32, 33};
        int cases = 0;
        for (int threads : widths) {
            const std::size_t shared_bytes = static_cast<std::size_t>(threads + 2) * sizeof(float);
            if (threads > properties.maxThreadsPerBlock || threads > properties.maxThreadsDim[0] ||
                shared_bytes > properties.sharedMemPerBlock) {
                std::cerr << "teaching configuration exceeds current device limits\n";
                return 1;
            }
            for (int n : sizes) {
                if (!run_case(n, threads)) return 1;
                ++cases;
            }
        }
        std::cout << "PASS dynamic matrix: " << cases << " cases, including 4 empty cases\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
