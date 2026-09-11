#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

__global__ void scale_rows(const float* input, std::size_t input_pitch,
                           float* output, std::size_t output_pitch,
                           int width, int height) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        const char* in_bytes = reinterpret_cast<const char*>(input) + row * input_pitch;
        char* out_bytes = reinterpret_cast<char*>(output) + row * output_pitch;
        const float* in_row = reinterpret_cast<const float*>(in_bytes);
        float* out_row = reinterpret_cast<float*>(out_bytes);
        out_row[col] = 2.0f * in_row[col];
    }
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    const int width = 5, height = 3;
    const std::size_t row_bytes = width * sizeof(float);
    std::vector<float> input(width * height), output(width * height);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) input[row * width + col] = row * 10.0f + col;
    }
    float* d_input = nullptr;
    float* d_output = nullptr;
    std::size_t input_pitch = 0, output_pitch = 0;
    bool ok = false;
    do {
        if (!check(cudaMallocPitch(&d_input, &input_pitch, row_bytes, height), "allocate input")) break;
        if (!check(cudaMallocPitch(&d_output, &output_pitch, row_bytes, height), "allocate output")) break;
        std::cout << "row_bytes=" << row_bytes << " input_pitch=" << input_pitch
                  << " output_pitch=" << output_pitch << '\n';
        if (!check(cudaMemcpy2D(d_input, input_pitch, input.data(), row_bytes,
                               row_bytes, height, cudaMemcpyHostToDevice), "H2D rows")) break;
        const dim3 block(4, 2);
        const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        scale_rows<<<grid, block>>>(d_input, input_pitch, d_output, output_pitch, width, height);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "execute")) break;
        if (!check(cudaMemcpy2D(output.data(), row_bytes, d_output, output_pitch,
                               row_bytes, height, cudaMemcpyDeviceToHost), "D2H rows")) break;

        ok = true;
        for (int i = 0; i < width * height; ++i) {
            if (!std::isfinite(output[i]) || output[i] != 2.0f * input[i]) {
                std::cerr << "mismatch index=" << i << '\n';
                ok = false;
                break;
            }
        }
        if (ok) {
            for (int row = 0; row < height; ++row) {
                std::cout << "row=" << row << ':';
                for (int col = 0; col < width; ++col) std::cout << ' ' << output[row * width + col];
                std::cout << '\n';
            }
        }
    } while (false);
    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS pitched scale: 5 x 3\n";
    return 0;
}
