#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void say_hello() {
    printf("GPU: hello from one thread\n");
}

int main() {
    std::cout << "CPU: before launch" << std::endl;
    say_hello<<<1, 1>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "say_hello launch failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "say_hello execution failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "CPU: GPU work finished" << std::endl;
    return 0;
}
