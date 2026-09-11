#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void inspect_layout() {
    unsigned int tid = threadIdx.x;
    unsigned int block = blockIdx.x;
    unsigned int global_id = block * blockDim.x + tid;
    unsigned int warp_in_block = tid / warpSize;
    unsigned int lane = tid % warpSize;

    printf("block=%u tid=%u global=%u warp=%u lane=%u\n",
           block, tid, global_id, warp_in_block, lane);
}

int main() {
    const int blocks = 2;
    const int threads = 40;
    inspect_layout<<<blocks, threads>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "launch: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "execution: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "PASS launch: 2 blocks, 40 threads each; inspect 80 records.\n";
    return 0;
}
