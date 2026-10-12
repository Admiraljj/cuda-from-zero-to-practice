#include "../common/cuda_check.h"

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <system_error>
#include <vector>

namespace {

constexpr int kTile = 16;
constexpr std::size_t kMaximumMatrixElements = 16ULL * 1024ULL * 1024ULL;

bool parse_dimension(const char* text, int& value) {
    if (text == nullptr || *text == '\0') return false;
    const char* end = text;
    while (*end != '\0') ++end;

    int parsed_value = 0;
    const auto parsed = std::from_chars(text, end, parsed_value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) return false;
    if (parsed_value < 1 || parsed_value > 4096) return false;
    value = parsed_value;
    return true;
}

bool dimensions_are_safe(int m, int k_size, int n) {
    const std::size_t a_count = static_cast<std::size_t>(m) * k_size;
    const std::size_t b_count = static_cast<std::size_t>(k_size) * n;
    const std::size_t c_count = static_cast<std::size_t>(m) * n;
    return a_count <= kMaximumMatrixElements &&
           b_count <= kMaximumMatrixElements &&
           c_count <= kMaximumMatrixElements;
}

__global__ void matmul_tiled_kernel(const float* a,
                                    const float* b,
                                    float* c,
                                    int m,
                                    int k_size,
                                    int n) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * kTile + tx;
    const int row = static_cast<int>(blockIdx.y) * kTile + ty;
    const int phase_count = (k_size + kTile - 1) / kTile;

    float accumulator = 0.0F;
    for (int phase = 0; phase < phase_count; ++phase) {
        const int a_col = phase * kTile + tx;
        const int b_row = phase * kTile + ty;

        if (row < m && a_col < k_size) {
            const std::size_t a_index =
                static_cast<std::size_t>(row) * k_size + a_col;
            a_tile[ty][tx] = a[a_index];
        } else {
            a_tile[ty][tx] = 0.0F;
        }

        if (b_row < k_size && col < n) {
            const std::size_t b_index =
                static_cast<std::size_t>(b_row) * n + col;
            b_tile[ty][tx] = b[b_index];
        } else {
            b_tile[ty][tx] = 0.0F;
        }

        __syncthreads();

#pragma unroll
        for (int inner_tile = 0; inner_tile < kTile; ++inner_tile) {
            accumulator +=
                a_tile[ty][inner_tile] * b_tile[inner_tile][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        const std::size_t c_index =
            static_cast<std::size_t>(row) * n + col;
        c[c_index] = accumulator;
    }
}

void make_inputs(std::vector<float>& a,
                 std::vector<float>& b,
                 int m,
                 int k_size,
                 int n) {
    for (int row = 0; row < m; ++row) {
        for (int inner = 0; inner < k_size; ++inner) {
            const std::size_t index =
                static_cast<std::size_t>(row) * k_size + inner;
            const int numerator = (row * 7 + inner * 3) % 9 - 4;
            a[index] = static_cast<float>(numerator) / 16.0F;
        }
    }
    for (int inner = 0; inner < k_size; ++inner) {
        for (int col = 0; col < n; ++col) {
            const std::size_t index =
                static_cast<std::size_t>(inner) * n + col;
            const int numerator = (inner * 5 + col * 2) % 7 - 3;
            b[index] = static_cast<float>(numerator) / 16.0F;
        }
    }
}

void make_cpu_reference(const std::vector<float>& a,
                        const std::vector<float>& b,
                        std::vector<float>& expected,
                        int m,
                        int k_size,
                        int n) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float accumulator = 0.0F;
            for (int inner = 0; inner < k_size; ++inner) {
                const std::size_t a_index =
                    static_cast<std::size_t>(row) * k_size + inner;
                const std::size_t b_index =
                    static_cast<std::size_t>(inner) * n + col;
                accumulator += a[a_index] * b[b_index];
            }
            expected[static_cast<std::size_t>(row) * n + col] = accumulator;
        }
    }
}

bool validate_exact(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    const char* label) {
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (actual[index] != expected[index]) {
            std::cerr << "FAIL " << label << " index=" << index
                      << " expected=" << expected[index]
                      << " actual=" << actual[index] << '\n';
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int m = 129;
    int k_size = 83;
    int n = 117;
    if (argc > 4 ||
        (argc >= 2 && !parse_dimension(argv[1], m)) ||
        (argc >= 3 && !parse_dimension(argv[2], k_size)) ||
        (argc >= 4 && !parse_dimension(argv[3], n)) ||
        !dimensions_are_safe(m, k_size, n)) {
        std::cerr << "Usage: " << argv[0]
                  << " [M:1..4096] [K:1..4096] [N:1..4096]\n"
                  << "Each matrix must contain at most "
                  << kMaximumMatrixElements << " elements.\n";
        return EXIT_FAILURE;
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    constexpr int threads_per_block = kTile * kTile;
    constexpr std::size_t shared_bytes =
        2ULL * kTile * kTile * sizeof(float);
    if (properties.maxThreadsPerBlock < threads_per_block ||
        properties.maxThreadsDim[0] < kTile ||
        properties.maxThreadsDim[1] < kTile ||
        properties.sharedMemPerBlock < shared_bytes) {
        std::cerr << "Device cannot launch the teaching tile: need "
                  << threads_per_block << " threads/block and "
                  << shared_bytes << " shared bytes/block, but device reports "
                  << properties.maxThreadsPerBlock << " and "
                  << properties.sharedMemPerBlock << ".\n";
        return 77;
    }

    const std::size_t a_count = static_cast<std::size_t>(m) * k_size;
    const std::size_t b_count = static_cast<std::size_t>(k_size) * n;
    const std::size_t c_count = static_cast<std::size_t>(m) * n;
    const std::size_t a_bytes = a_count * sizeof(float);
    const std::size_t b_bytes = b_count * sizeof(float);
    const std::size_t c_bytes = c_count * sizeof(float);

    std::vector<float> a(a_count);
    std::vector<float> b(b_count);
    std::vector<float> expected(c_count);
    std::vector<float> actual(c_count);
    std::vector<float> a_after(a_count);
    std::vector<float> b_after(b_count);
    make_inputs(a, b, m, k_size, n);
    make_cpu_reference(a, b, expected, m, k_size, n);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0xFF, c_bytes));

    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile,
                    (m + kTile - 1) / kTile);
    matmul_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, m, k_size, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(actual.data(), d_c, c_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_after.data(), d_a, a_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_after.data(), d_b, b_bytes,
                          cudaMemcpyDeviceToHost));

    const bool output_ok = validate_exact(actual, expected, "C");
    const bool a_ok = validate_exact(a_after, a, "A changed");
    const bool b_ok = validate_exact(b_after, b, "B changed");

    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_a));

    if (!output_ok || !a_ok || !b_ok) return EXIT_FAILURE;
    std::cout << "A(" << m << 'x' << k_size << ") * B("
              << k_size << 'x' << n << ") -> C("
              << m << 'x' << n << ")\n"
              << "tile=" << kTile << " block=(" << block.x << ','
              << block.y << ") grid=(" << grid.x << ',' << grid.y
              << ") phases=" << (k_size + kTile - 1) / kTile << '\n'
              << "shared_bytes_per_block=" << shared_bytes << '\n'
              << "PASS tiled GEMM matches every CPU output; A and B unchanged.\n";
    return EXIT_SUCCESS;
}
