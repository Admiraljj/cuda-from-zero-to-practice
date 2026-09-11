#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void report_thread() {
    printf("GPU: thread=%u\n", threadIdx.x);
}

int main() {
    std::cout << "CPU: A before launch" << std::endl;
    report_thread<<<1, 4>>>();
    std::cout << "CPU: B launch call returned" << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "report_thread launch failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "report_thread execution failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "CPU: C synchronization succeeded" << std::endl;
    return 0;
}
