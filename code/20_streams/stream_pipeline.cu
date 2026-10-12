#include "../common/cuda_check.h"
#include "../09_timing/benchmark_stats.h"
#include "../17_convolution/filter_reference.h"
#include <chrono>
#include <cstdlib>
#include <limits>

namespace {
__global__ void scale_chunk(float* values, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) values[i] *= 2.5F;
}
struct Slot {
    cudaStream_t stream{};
    cudaEvent_t done{};
    float *host_input = nullptr, *host_output = nullptr, *device = nullptr;
    std::vector<float> pageable_input, pageable_output;
    int offset = 0, count = 0;
    bool busy = false, used = false;
};
struct Pool {
    std::vector<Slot> slots;
    cudaStream_t control{};
    cudaEvent_t start{}, stop{};
    bool pinned;
    Pool(int streams, int chunk, bool use_pinned) : slots(streams), pinned(use_pinned) {
        const std::size_t bytes = static_cast<std::size_t>(chunk) * sizeof(float);
        CUDA_CHECK(cudaStreamCreateWithFlags(&control, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        for (auto& s : slots) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
            CUDA_CHECK(cudaMalloc(&s.device, bytes));
            if (pinned) {
                CUDA_CHECK(cudaMallocHost(&s.host_input, bytes));
                CUDA_CHECK(cudaMallocHost(&s.host_output, bytes));
            } else {
                s.pageable_input.resize(chunk); s.pageable_output.resize(chunk);
                s.host_input = s.pageable_input.data(); s.host_output = s.pageable_output.data();
            }
        }
    }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    ~Pool() {
        // Every run drains all slots before destruction; checked destruction is shown explicitly.
        for (auto& s : slots) {
            CUDA_CHECK(cudaEventDestroy(s.done)); CUDA_CHECK(cudaStreamDestroy(s.stream));
            CUDA_CHECK(cudaFree(s.device));
            if (pinned) { CUDA_CHECK(cudaFreeHost(s.host_output)); CUDA_CHECK(cudaFreeHost(s.host_input)); }
        }
        CUDA_CHECK(cudaEventDestroy(stop)); CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaStreamDestroy(control));
    }
};
struct Sample { double host_ms, gpu_span_ms; };

Sample execute(Pool& pool, const std::vector<float>& input, std::vector<float>& output, int chunk) {
    std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
    for (auto& s : pool.slots) { s.busy = false; s.used = false; }
    const auto before = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(pool.start, pool.control));
    for (auto& s : pool.slots) CUDA_CHECK(cudaStreamWaitEvent(s.stream, pool.start, 0));
    auto consume = [&](Slot& s) {
        if (!s.busy) return;
        CUDA_CHECK(cudaEventSynchronize(s.done));
        std::copy_n(s.host_output, s.count, output.data() + s.offset);
        s.busy = false;
    };
    const int n = static_cast<int>(input.size());
    for (int offset = 0, id = 0; offset < n; offset += chunk, ++id) {
        auto& s = pool.slots[id % pool.slots.size()];
        consume(s); // D2H complete and previous output consumed before host/device slot reuse.
        s.offset = offset; s.count = std::min(chunk, n - offset);
        std::copy_n(input.data() + offset, s.count, s.host_input);
        std::fill_n(s.host_output, s.count, std::numeric_limits<float>::quiet_NaN());
        const std::size_t bytes = static_cast<std::size_t>(s.count) * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(s.device, s.host_input, bytes, cudaMemcpyHostToDevice, s.stream));
        scale_chunk<<<(s.count + 255) / 256, 256, 0, s.stream>>>(s.device, s.count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(s.host_output, s.device, bytes, cudaMemcpyDeviceToHost, s.stream));
        CUDA_CHECK(cudaEventRecord(s.done, s.stream));
        s.busy = true; s.used = true;
    }
    // Join latest D2H completion from every used slot before recording the stop timestamp.
    for (auto& s : pool.slots) if (s.used) CUDA_CHECK(cudaStreamWaitEvent(pool.control, s.done, 0));
    CUDA_CHECK(cudaEventRecord(pool.stop, pool.control));
    for (auto& s : pool.slots) consume(s);
    CUDA_CHECK(cudaEventSynchronize(pool.stop));
    const auto after = std::chrono::steady_clock::now();
    float gpu_ms = 0.0F; CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, pool.start, pool.stop));
    const double host_ms = std::chrono::duration<double, std::milli>(after - before).count();
    if (!std::isfinite(gpu_ms) || gpu_ms <= 0.0F || !std::isfinite(host_ms) || host_ms <= 0.0) {
        std::cerr << "FAIL unresolved timing interval.\n"; std::exit(EXIT_FAILURE);
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float original = static_cast<float>(static_cast<int>(i % 127) - 63) / 8.0F;
        if (input[i] != original || !std::isfinite(output[i]) || output[i] != original * 2.5F) {
            std::cerr << "FAIL global index=" << i << '\n'; std::exit(EXIT_FAILURE);
        }
    }
    return {host_ms, gpu_ms};
}
}

int main(int argc, char** argv) {
    int n = 4194304, chunk = 262144, streams = 3, rounds = 5;
    const std::string memory = argc > 5 ? argv[5] : "pinned";
    if (argc > 6 || (argc > 1 && !filter_example::integer(argv[1], 0, 16777216, n)) ||
        (argc > 2 && !filter_example::integer(argv[2], 1, 4194304, chunk)) ||
        (argc > 3 && !filter_example::integer(argv[3], 1, 8, streams)) ||
        (argc > 4 && !filter_example::integer(argv[4], 1, 31, rounds)) ||
        (memory != "pinned" && memory != "pageable") ||
        12ULL * chunk * (streams + 1) > 256ULL * 1024ULL * 1024ULL) {
        std::cerr << "Usage: stream_pipeline [n:0..16777216] [chunk:1..4194304]"
                     " [streams:1..8] [rounds:1..31] [pinned|pageable]\n"
                     "Combined slot host/device capacity must be <=256 MiB.\n";
        return EXIT_FAILURE;
    }
    if (n == 0) { std::cout << "PASS empty input; no GPU allocation or launch.\n"; return EXIT_SUCCESS; }
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU=" << prop.name << " asyncEngineCount=" << prop.asyncEngineCount
              << " concurrentKernels=" << prop.concurrentKernels << " n=" << n
              << " chunk=" << chunk << " streams=" << streams << " memory=" << memory << '\n';
    std::vector<float> input(n), output(n);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 8.0F;
    Pool serial(1, chunk, memory == "pinned"), pipeline(streams, chunk, memory == "pinned");
    execute(serial, input, output, chunk); execute(pipeline, input, output, chunk);
    std::vector<double> serial_host, pipeline_host, serial_gpu, pipeline_gpu;
    for (int round = 0; round < rounds; ++round) {
        for (int slot = 0; slot < 2; ++slot) {
            const bool parallel = (round + slot) % 2 == 1;
            const Sample s = execute(parallel ? pipeline : serial, input, output, chunk);
            (parallel ? pipeline_host : serial_host).push_back(s.host_ms);
            (parallel ? pipeline_gpu : serial_gpu).push_back(s.gpu_span_ms);
            std::cout << "sample round=" << round << " method=" << (parallel ? "pipeline" : "serial")
                      << " host_staging_submit_gather_ms=" << s.host_ms << " GPU_fork_join_span_ms=" << s.gpu_span_ms << '\n';
        }
    }
    const auto a = summarize(serial_host), b = summarize(pipeline_host);
    const auto ga = summarize(serial_gpu), gb = summarize(pipeline_gpu);
    std::cout << "serial host min/median/mean/max=" << a.minimum << '/' << a.median << '/' << a.mean << '/' << a.maximum << '\n'
              << "pipeline host min/median/mean/max=" << b.minimum << '/' << b.median << '/' << b.mean << '/' << b.maximum << '\n'
              << "GPU span median serial=" << ga.median << " pipeline=" << gb.median << '\n'
              << "host_speedup=" << a.median / b.median
              << " pipeline_logical_transfer_GB_s=" << (2.0 * n * sizeof(float)) / (b.median * 1.0e6) << '\n'
              << "PASS serial/pipeline cover every output; input unchanged.\n";
    return EXIT_SUCCESS;
}
