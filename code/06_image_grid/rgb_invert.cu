#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
#include <vector>

__global__ void invert_rgb(const unsigned char* input, unsigned char* output,
                           int width, int height) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        const int base = (row * width + col) * 3;
        for (int channel = 0; channel < 3; ++channel) {
            output[base + channel] = static_cast<unsigned char>(255 - input[base + channel]);
        }
    }
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    const int width = 5, height = 3, channels = 3;
    const std::size_t count = static_cast<std::size_t>(width) * height * channels;
    const std::size_t bytes = count * sizeof(unsigned char);
    std::vector<unsigned char> input(count), output(count);
    for (int i = 0; i < width * height; ++i) {
        input[3*i] = static_cast<unsigned char>(i);
        input[3*i+1] = static_cast<unsigned char>(50+i);
        input[3*i+2] = static_cast<unsigned char>(100+i);
    }
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input")) break;
        if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
        if (!check(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;
        const dim3 block(4, 2), grid(2, 2);
        invert_rgb<<<grid, block>>>(d_input, d_output, width, height);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "execute")) break;
        if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (std::size_t i = 0; i < count; ++i) {
            if (output[i] != static_cast<unsigned char>(255 - input[i])) {
                std::cerr << "mismatch component=" << i << '\n';
                ok = false;
                break;
            }
        }
        if (ok) {
            for (int i = 0; i < width * height; ++i) {
                std::cout << "pixel=" << i
                          << " R=" << static_cast<int>(output[3*i])
                          << " G=" << static_cast<int>(output[3*i+1])
                          << " B=" << static_cast<int>(output[3*i+2]) << '\n';
            }
        }
    } while (false);
    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS RGB: 5 x 3 pixels, 45 components\n";
    return 0;
}
