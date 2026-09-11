#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__device__ int twice(int x) {
    return 2 * x;
}

__global__ void compute_values(int base) {
    int input = base + static_cast<int>(threadIdx.x);
    int output = twice(input);
    printf("GPU: thread=%u input=%d output=%d\n",
           threadIdx.x, input, output);
}

int main() {
    const int base = 10;
    compute_values<<<1, 4>>>(base);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "compute_values launch failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "compute_values execution failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "CPU: scalar tasks finished" << std::endl;
    return 0;
}
