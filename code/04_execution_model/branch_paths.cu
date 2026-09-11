#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void branch_paths(int mode) {
    unsigned int tid = threadIdx.x;
    bool choose_a;
    if (mode == 0) {
        choose_a = (tid < 32);
    } else {
        choose_a = ((tid % 2) == 0);
    }

    int value;
    char path;
    if (choose_a) {
        value = static_cast<int>(tid) + 100;
        path = 'A';
    } else {
        value = static_cast<int>(tid) + 200;
        path = 'B';
    }

    printf("mode=%d tid=%u warp=%u lane=%u path=%c value=%d\n",
           mode, tid, tid / warpSize, tid % warpSize, path, value);
}

int main() {
    for (int mode = 0; mode < 2; ++mode) {
        branch_paths<<<1, 64>>>(mode);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "launch mode=" << mode << ": "
                      << cudaGetErrorString(err) << '\n';
            return 1;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "execution mode=" << mode << ": "
                      << cudaGetErrorString(err) << '\n';
            return 1;
        }
    }

    std::cout << "PASS launch: two modes; inspect 128 records.\n";
    return 0;
}
