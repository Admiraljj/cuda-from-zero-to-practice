#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

__global__ void write_indices(int* output, int n, bool fixed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!fixed || i < n) output[i] = i;
}

__global__ void read_value(const int* input, int* output) {
    output[0] = input[0] + 1;
}

__global__ void exchange_shared(int* output, bool fixed) {
    __shared__ volatile int values[64];
    const int tid = threadIdx.x;
    values[tid] = tid;
    if (fixed) __syncthreads();
    output[tid] = values[63 - tid];
}

__global__ void conditional_barrier(int* output, bool fixed) {
    const int tid = threadIdx.x;
    if (fixed || tid < 48) __syncthreads();
    output[tid] = tid;
}

__global__ void empty_kernel() {}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main(int argc, char** argv) {
    const std::string mode = argc >= 2 ? argv[1] : "oob-good";
    const bool fixed = mode == "oob-good" || mode == "init-good"
                    || mode == "race-good" || mode == "sync-good" || mode == "launch-good";
    const bool broken = mode == "oob-bad" || mode == "init-bad"
                     || mode == "race-bad" || mode == "sync-bad" || mode == "launch-bad";
    if ((!fixed && !broken) || argc > 3
        || (broken && (argc != 3 || std::string(argv[2]) != "--run-broken"))
        || (fixed && argc == 3)) {
        std::cerr << "Use: debug_cases {oob,init,race,sync,launch}-{good,bad}"
                     " [--run-broken for bad only]\n";
        return 2;
    }
    if (broken) {
        std::cerr << "WARNING: intentional error; run under the selected sanitizer"
                     " on an authorized practice GPU\n";
    }
    if (mode == "launch-good" || mode == "launch-bad") {
        int device = 0;
        cudaDeviceProp properties{};
        if (!check(cudaGetDevice(&device), "device")
            || !check(cudaGetDeviceProperties(&properties, device), "properties")) return 1;
        const int threads = fixed ? 1 : properties.maxThreadsPerBlock + 1;
        empty_kernel<<<1, threads>>>();
        const cudaError_t peek = cudaPeekAtLastError();
        const cudaError_t first = cudaGetLastError();
        const cudaError_t second = cudaGetLastError();
        const cudaError_t sync = cudaDeviceSynchronize();
        std::cout << "peek: " << cudaGetErrorString(peek)
                  << "\nfirst get: " << cudaGetErrorString(first)
                  << "\nsecond get: " << cudaGetErrorString(second)
                  << "\nsynchronize: " << cudaGetErrorString(sync) << '\n';
        if (!fixed) {
            if (peek == cudaSuccess || first != peek || second != cudaSuccess) return 1;
            std::cout << "OBSERVED expected launch rejection; not a successful kernel\n";
            return 0;
        }
        if (peek != cudaSuccess || first != cudaSuccess
            || second != cudaSuccess || sync != cudaSuccess) return 1;
        std::cout << "PASS launch-good\n";
        return 0;
    }

    const bool init_case = mode == "init-good" || mode == "init-bad";
    const bool oob_case = mode == "oob-good" || mode == "oob-bad";
    const bool race_case = mode == "race-good" || mode == "race-bad";
    const int count = init_case ? 1 : (oob_case ? 33 : 64);
    std::vector<int> output(count, -1);
    int* d_output = nullptr;
    int* d_input = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_output, count * sizeof(int)), "allocate output")) break;
        if (init_case) {
            if (!check(cudaMalloc(&d_input, sizeof(int)), "allocate input")) break;
            if (fixed) {
                const int value = 41;
                if (!check(cudaMemcpy(d_input, &value, sizeof(int), cudaMemcpyHostToDevice),
                           "initialize input")) break;
            }
            read_value<<<1, 1>>>(d_input, d_output);
        } else if (oob_case) {
            write_indices<<<2, 32>>>(d_output, count, fixed);
        } else if (race_case) {
            exchange_shared<<<1, 64>>>(d_output, fixed);
        } else {
            conditional_barrier<<<1, 64>>>(d_output, fixed);
        }
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "execute")) break;
        if (!check(cudaMemcpy(output.data(), d_output, count * sizeof(int),
                              cudaMemcpyDeviceToHost), "copy output")) break;
        ok = true;
        if (fixed) {
            for (int i = 0; i < count; ++i) {
                const int expected = init_case ? 42 : (race_case ? 63 - i : i);
                if (output[i] != expected) {
                    std::cerr << "mismatch index=" << i << " actual=" << output[i]
                              << " expected=" << expected << '\n';
                    ok = false;
                    break;
                }
            }
        }
    } while (false);
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (!ok) return 1;
    if (fixed) {
        std::cout << "PASS " << mode << ": " << count << " checked elements\n";
    } else {
        std::cout << "BAD CASE completed; exit 0 is not correctness evidence\n";
    }
    return 0;
}
