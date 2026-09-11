#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
#include <vector>

__global__ void invert(const unsigned char* input, unsigned char* output,
                       int width, int height) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        const int i = row * width + col;
        output[i] = static_cast<unsigned char>(255 - input[i]);
    }
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool run_case(int width, int height, dim3 block) {
    if (width < 0 || height < 0 || width > 4096 || height > 4096
        || block.x == 0 || block.y == 0 || block.z != 1) {
        std::cerr << "Invalid dimensions\n";
        return false;
    }
    if (width == 0 || height == 0) {
        std::cout << "PASS empty: " << width << " x " << height << " (no GPU work)\n";
        return true;
    }
    int device = 0;
    cudaDeviceProp properties{};
    if (!check(cudaGetDevice(&device), "device")
        || !check(cudaGetDeviceProperties(&properties, device), "properties")) return false;
    const std::size_t block_threads = static_cast<std::size_t>(block.x) * block.y;
    if (block.x > static_cast<unsigned>(properties.maxThreadsDim[0])
        || block.y > static_cast<unsigned>(properties.maxThreadsDim[1])
        || block_threads > static_cast<std::size_t>(properties.maxThreadsPerBlock)) {
        std::cerr << "Block exceeds device limits\n";
        return false;
    }
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);
    if (grid.x > static_cast<unsigned>(properties.maxGridSize[0])
        || grid.y > static_cast<unsigned>(properties.maxGridSize[1])) {
        std::cerr << "Grid exceeds device limits\n";
        return false;
    }

    const std::size_t count = static_cast<std::size_t>(width) * height;
    const std::size_t bytes = count * sizeof(unsigned char);
    std::vector<unsigned char> input(count), output(count);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            input[row * width + col] = static_cast<unsigned char>((row * 17 + col * 3) % 256);
        }
    }

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input")) break;
        if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
        if (!check(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;
        invert<<<grid, block>>>(d_input, d_output, width, height);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "execute")) break;
        if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;

        ok = true;
        for (std::size_t i = 0; i < count; ++i) {
            const unsigned char expected = static_cast<unsigned char>(255 - input[i]);
            if (output[i] != expected) {
                std::cerr << "mismatch row=" << i / width << " col=" << i % width
                          << " actual=" << static_cast<int>(output[i])
                          << " expected=" << static_cast<int>(expected) << '\n';
                ok = false;
                break;
            }
        }
        if (ok && width == 5 && height == 3) {
            for (int row = 0; row < height; ++row) {
                std::cout << "row=" << row << ':';
                for (int col = 0; col < width; ++col) {
                    std::cout << ' ' << static_cast<int>(output[row * width + col]);
                }
                std::cout << '\n';
            }
        }
    } while (false);

    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (ok) {
        std::cout << "PASS: " << width << " x " << height
                  << " image; block=" << block.x << 'x' << block.y
                  << " grid=" << grid.x << 'x' << grid.y << '\n';
    }
    return ok;
}

int main() {
    bool ok = true;
    if (!run_case(0, 3, dim3(4, 2))) ok = false;
    if (!run_case(5, 3, dim3(4, 2))) ok = false;
    if (!run_case(1, 1, dim3(16, 16))) ok = false;
    if (!run_case(17, 1, dim3(16, 16))) ok = false;
    if (!run_case(1, 17, dim3(16, 16))) ok = false;
    if (!run_case(16, 16, dim3(16, 16))) ok = false;
    if (!run_case(37, 23, dim3(16, 16))) ok = false;
    if (!run_case(37, 23, dim3(32, 8))) ok = false;
    return ok ? 0 : 1;
}
