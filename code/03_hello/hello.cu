#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void hello_kernel() {
    printf("GPU: block=%u thread=%u block_size=%u grid_size=%u\n",
           blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
}

int main() {
    const int blocks = 2;
    const int threads = 4;
    std::cout << "CPU: before launch" << std::endl;
    hello_kernel<<<blocks, threads>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "hello_kernel launch failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "hello_kernel execution failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "CPU: GPU work finished" << std::endl;
    return 0;
}
