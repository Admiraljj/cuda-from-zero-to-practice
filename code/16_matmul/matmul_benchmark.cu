#include "../common/cuda_check.h"
#include "../09_timing/benchmark_stats.h"

#ifndef MATMUL_WITH_CUBLAS
#define MATMUL_WITH_CUBLAS 0
#endif
#if MATMUL_WITH_CUBLAS
#include <cublas_v2.h>
#endif

#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace {
constexpr int kTile = 16;
constexpr int kMethods = MATMUL_WITH_CUBLAS ? 3 : 2;
constexpr std::array<const char*, 3> kNames = {"naive", "tiled", "cublas"};
constexpr long long kMaxReferenceTerms = 256LL * 1024LL * 1024LL;

struct Configuration {
    int m = 129, k = 83, n = 117, rounds = 7, repeats = 3;
    bool exact = true;
};

bool parse_int(const char* text, int high, int& value) {
    if (!text || !*text) return false;
    const char* end = text;
    while (*end) ++end;
    int candidate = 0;
    const auto result = std::from_chars(text, end, candidate);
    if (result.ec != std::errc{} || result.ptr != end ||
        candidate < 1 || candidate > high) return false;
    value = candidate;
    return true;
}

bool configure(int argc, char** argv, Configuration& p) {
    if (argc > 7 ||
        (argc > 1 && !parse_int(argv[1], 2048, p.m)) ||
        (argc > 2 && !parse_int(argv[2], 2048, p.k)) ||
        (argc > 3 && !parse_int(argv[3], 2048, p.n)) ||
        (argc > 4 && !parse_int(argv[4], 31, p.rounds)) ||
        (argc > 5 && !parse_int(argv[5], 1000, p.repeats))) return false;
    if (argc > 6) {
        const std::string mode = argv[6];
        if (mode != "exact" && mode != "decimal") return false;
        p.exact = mode == "exact";
    }
    // All dimensions <= 2048 already bound each matrix to 4 Mi elements.
    return 1LL * p.m * p.k * p.n <= kMaxReferenceTerms;
}

__global__ void benchmark_naive_kernel(const float* a, const float* b,
                                       float* c, int m, int k, int n) {
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= m || col >= n) return;
    float acc = 0.0F;
    for (int inner = 0; inner < k; ++inner) {
        acc += a[static_cast<std::size_t>(row) * k + inner] *
               b[static_cast<std::size_t>(inner) * n + col];
    }
    c[static_cast<std::size_t>(row) * n + col] = acc;
}

__global__ void benchmark_tiled_kernel(const float* a, const float* b,
                                       float* c, int m, int k, int n) {
    __shared__ float at[kTile][kTile], bt[kTile][kTile];
    const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
    const int row = static_cast<int>(blockIdx.y) * kTile + ty;
    const int col = static_cast<int>(blockIdx.x) * kTile + tx;
    float acc = 0.0F;
    for (int base = 0; base < k; base += kTile) {
        at[ty][tx] = row < m && base + tx < k
            ? a[static_cast<std::size_t>(row) * k + base + tx] : 0.0F;
        bt[ty][tx] = base + ty < k && col < n
            ? b[static_cast<std::size_t>(base + ty) * n + col] : 0.0F;
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < kTile; ++inner) acc += at[ty][inner] * bt[inner][tx];
        __syncthreads();
    }
    if (row < m && col < n) c[static_cast<std::size_t>(row) * n + col] = acc;
}

#if MATMUL_WITH_CUBLAS
void check_cublas(cublasStatus_t status, const char* expression, int line) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    std::cerr << "cuBLAS status=" << static_cast<int>(status)
              << " expression=" << expression << " line=" << line << '\n';
    std::exit(EXIT_FAILURE);
}
#define CUBLAS_CHECK(expr) check_cublas((expr), #expr, __LINE__)
#endif

void make_inputs(std::vector<float>& a, std::vector<float>& b, const Configuration& p) {
    for (int row = 0; row < p.m; ++row) {
        for (int inner = 0; inner < p.k; ++inner) {
            const int numerator = (row * 7 + inner * 3) % 9 - 4;
            a[static_cast<std::size_t>(row) * p.k + inner] =
                p.exact ? numerator / 16.0F : numerator * 0.1F;
        }
    }
    for (int inner = 0; inner < p.k; ++inner) {
        for (int col = 0; col < p.n; ++col) {
            const int numerator = (inner * 5 + col * 2) % 7 - 3;
            b[static_cast<std::size_t>(inner) * p.n + col] =
                p.exact ? numerator / 16.0F : numerator * 0.2F;
        }
    }
}

void cpu_reference(const std::vector<float>& a, const std::vector<float>& b,
                   std::vector<double>& expected, std::vector<double>& sum_abs,
                   const Configuration& p) {
    for (int row = 0; row < p.m; ++row) {
        for (int col = 0; col < p.n; ++col) {
            double acc = 0.0, magnitude = 0.0;
            for (int inner = 0; inner < p.k; ++inner) {
                const double product =
                    static_cast<double>(a[static_cast<std::size_t>(row) * p.k + inner]) *
                    static_cast<double>(b[static_cast<std::size_t>(inner) * p.n + col]);
                acc += product;
                magnitude += std::abs(product);
            }
            const std::size_t index = static_cast<std::size_t>(row) * p.n + col;
            expected[index] = acc;
            sum_abs[index] = magnitude;
        }
    }
}

bool validate(const std::vector<float>& actual, const std::vector<double>& expected,
              const std::vector<double>& sum_abs, const Configuration& p,
              const char* name) {
    const double u = std::numeric_limits<float>::epsilon() / 2.0;
    const int padded_k = ((p.k + kTile - 1) / kTile) * kTile;
    const double qu = (2.0 * padded_k + 1.0) * u;
    const double gamma = qu / (1.0 - qu);
    double max_error = 0.0, max_ratio = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
        const double bound = p.exact ? 0.0 : 1.0e-6 + 8.0 * gamma * sum_abs[i];
        if (!std::isfinite(actual[i]) || error > bound) {
            std::cerr << std::setprecision(17) << "FAIL " << name
                      << " C[" << i / p.n << ',' << i % p.n << "] expected="
                      << expected[i] << " actual=" << actual[i]
                      << " abs_error=" << error << " bound=" << bound << '\n';
            return false;
        }
        max_error = std::max(max_error, error);
        if (bound > 0.0) max_ratio = std::max(max_ratio, error / bound);
    }
    std::cout << "check " << name << " max_abs_error=" << max_error
              << " max_bound_ratio=" << max_ratio << '\n';
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    Configuration p;
    if (!configure(argc, argv, p)) {
        std::cerr << "Usage: " << argv[0] << " [M:1..2048] [K:1..2048] [N:1..2048]"
                  << " [rounds:1..31] [repeats:1..1000] [exact|decimal]\n"
                  << "CPU reference M*K*N must be <= " << kMaxReferenceTerms << ".\n";
        return EXIT_FAILURE;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.maxThreadsPerBlock < 256 || prop.maxThreadsDim[0] < kTile ||
        prop.maxThreadsDim[1] < kTile || prop.sharedMemPerBlock < 2048) {
        std::cerr << "NOT RUN: device does not support this 16x16 block.\n";
        return 77;
    }
    int runtime = 0, driver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    cudaFuncAttributes naive_attr{}, tiled_attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&naive_attr, benchmark_naive_kernel));
    CUDA_CHECK(cudaFuncGetAttributes(&tiled_attr, benchmark_tiled_kernel));

    const std::size_t ac = static_cast<std::size_t>(p.m) * p.k;
    const std::size_t bc = static_cast<std::size_t>(p.k) * p.n;
    const std::size_t cc = static_cast<std::size_t>(p.m) * p.n;
    const std::size_t ab = ac * sizeof(float), bb = bc * sizeof(float), cb = cc * sizeof(float);
    std::vector<float> a(ac), b(bc), actual(cc), a_after(ac), b_after(bc);
    std::vector<double> expected(cc), sum_abs(cc);
    make_inputs(a, b, p);
    cpu_reference(a, b, expected, sum_abs, p);
    float *da = nullptr, *db = nullptr;
    std::array<float*, kMethods> outputs{};
    CUDA_CHECK(cudaMalloc(&da, ab));
    CUDA_CHECK(cudaMalloc(&db, bb));
    for (float*& output : outputs) CUDA_CHECK(cudaMalloc(&output, cb));
    CUDA_CHECK(cudaMemcpy(da, a.data(), ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bb, cudaMemcpyHostToDevice));

    cudaStream_t stream{};
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
#if MATMUL_WITH_CUBLAS
    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));
    int version = 0;
    CUBLAS_CHECK(cublasGetVersion(handle, &version));
    std::cout << "cublas_version=" << version << " compute=CUBLAS_COMPUTE_32F_PEDANTIC\n";
#endif
    const dim3 block(kTile, kTile);
    const dim3 grid((p.n + kTile - 1) / kTile, (p.m + kTile - 1) / kTile);
    auto launch = [&](int method) {
        if (method == 0) {
            benchmark_naive_kernel<<<grid, block, 0, stream>>>(da, db, outputs[method], p.m, p.k, p.n);
            CUDA_CHECK(cudaGetLastError());
        } else if (method == 1) {
            benchmark_tiled_kernel<<<grid, block, 0, stream>>>(da, db, outputs[method], p.m, p.k, p.n);
            CUDA_CHECK(cudaGetLastError());
        }
#if MATMUL_WITH_CUBLAS
        else {
            const float alpha = 1.0F, beta = 0.0F;
            // Row-major C = A*B is column-major C^T = B^T*A^T.
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                p.n, p.m, p.k, &alpha,
                db, CUDA_R_32F, p.n, da, CUDA_R_32F, p.k,
                &beta, outputs[method], CUDA_R_32F, p.n,
                CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
        }
#endif
    };
    auto check_result = [&](int method) {
        CUDA_CHECK(cudaMemcpy(actual.data(), outputs[method], cb, cudaMemcpyDeviceToHost));
        if (!validate(actual, expected, sum_abs, p, kNames[method])) std::exit(EXIT_FAILURE);
    };
    auto check_inputs = [&]() {
        CUDA_CHECK(cudaMemcpy(a_after.data(), da, ab, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b_after.data(), db, bb, cudaMemcpyDeviceToHost));
        if (a_after != a || b_after != b) {
            std::cerr << "FAIL A or B changed.\n";
            std::exit(EXIT_FAILURE);
        }
    };
    std::cout << "GPU=" << prop.name << " cc=" << prop.major << '.' << prop.minor
              << " runtime=" << runtime << " driver=" << driver << '\n'
              << "M=" << p.m << " K=" << p.k << " N=" << p.n
              << " rounds=" << p.rounds << " repeats=" << p.repeats
              << " mode=" << (p.exact ? "exact" : "decimal") << '\n'
              << "naive registers/thread=" << naive_attr.numRegs
              << " tiled registers/thread=" << tiled_attr.numRegs
              << " tiled static_shared_bytes/block=" << tiled_attr.sharedSizeBytes << '\n';

    // Validate one complete overwrite before repeated calls can hide stale output.
    for (int method = 0; method < kMethods; ++method) {
        CUDA_CHECK(cudaMemsetAsync(outputs[method], 0xFF, cb, stream));
        launch(method);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        check_result(method);
    }
    check_inputs();
    for (int warm = 0; warm < 3; ++warm) {
        for (int slot = 0; slot < kMethods; ++slot) launch((warm + slot) % kMethods);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int method = 0; method < kMethods; ++method) check_result(method);
    check_inputs();

    std::array<std::vector<double>, kMethods> samples;
    for (int round = 0; round < p.rounds; ++round) {
        for (int slot = 0; slot < kMethods; ++slot) {
            const int method = (round + slot) % kMethods;
            CUDA_CHECK(cudaMemsetAsync(outputs[method], 0xFF, cb, stream));
            CUDA_CHECK(cudaEventRecord(start, stream));
            for (int repeat = 0; repeat < p.repeats; ++repeat) launch(method);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float batch_ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&batch_ms, start, stop));
            if (!std::isfinite(batch_ms) || batch_ms <= 0.0F) {
                std::cerr << "FAIL unresolvable Event interval; increase repeats.\n";
                return EXIT_FAILURE;
            }
            samples[method].push_back(batch_ms / p.repeats);
            check_result(method);
            std::cout << "sample round=" << round << " method=" << kNames[method]
                      << " batch_ms=" << batch_ms << " ms_per_gemm=" << samples[method].back() << '\n';
        }
        check_inputs();
    }
    const double useful_flops = 2.0 * p.m * p.n * p.k;
    std::cout << std::setprecision(8)
              << "scope=Event interval for repeated GEMM submissions on one stream; "
                 "allocation/copies/checks excluded\n"
              << "useful_FLOPs_per_GEMM=" << useful_flops << '\n';
    for (int method = 0; method < kMethods; ++method) {
        const TimingStats s = summarize(samples[method]);
        std::cout << "summary method=" << kNames[method] << " samples=" << s.count
                  << " min_ms=" << s.minimum << " median_ms=" << s.median
                  << " mean_ms=" << s.mean << " max_ms=" << s.maximum
                  << " useful_GFLOP_s=" << useful_flops / (s.median * 1.0e6) << '\n';
    }
#if MATMUL_WITH_CUBLAS
    CUBLAS_CHECK(cublasDestroy(handle));
#endif
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (float* output : outputs) CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(da));
    std::cout << "PASS every checked GEMM output matches the CPU reference; A and B unchanged.\n";
    return EXIT_SUCCESS;
}
