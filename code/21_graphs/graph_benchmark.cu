#include "graph_workload.cuh"
#include "../09_timing/benchmark_stats.h"
#include <array>
#include <chrono>
#include <iomanip>

int main(int argc, char** argv) {
    int n = 4096, repeats = 1000, rounds = 5;
    if (argc > 4 || (argc > 1 && !graph_example::integer(argv[1], 0, 16777216, n)) ||
        (argc > 2 && !graph_example::integer(argv[2], 1, 100000, repeats)) ||
        (argc > 3 && !graph_example::integer(argv[3], 1, 31, rounds)) ||
        1LL * n * repeats > 4294967296LL) {
        std::cerr << "Usage: graph_benchmark [n:0..16777216] [repeats:1..100000]"
                     " [rounds:1..31]; n*repeats<=4294967296\n"; return EXIT_FAILURE;
    }
    if (n == 0) { std::cout << "PASS empty input; no timing.\n"; return EXIT_SUCCESS; }
    using Clock = std::chrono::steady_clock;
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    cudaDeviceProp prop{}; int driver = 0, runtime = 0;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaDriverGetVersion(&driver)); CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::cout << "GPU=" << prop.name << " driver=" << driver << " runtime=" << runtime
              << " n=" << n << " repeats=" << repeats << " rounds=" << rounds << '\n';
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> input(n), output(n), after(n);
    for (int i = 0; i < n; ++i) input[i] = graph_example::input_value(i);
    float* di = nullptr; std::array<float*, 3> dout{};
    CUDA_CHECK(cudaMalloc(&di, bytes));
    CUDA_CHECK(cudaMemcpy(di, input.data(), bytes, cudaMemcpyHostToDevice));
    for (auto& p : dout) CUDA_CHECK(cudaMalloc(&p, bytes));
    cudaStream_t stream{}; CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t start{}, stop{}; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    // Materialize kernel code before timing graph construction (not a cold-process measurement).
    graph_example::submit_three(di, dout[0], n, 2.0F, 1.0F, stream);
    graph_example::fused<<<(n + 255) / 256, 256, 0, stream>>>(di, dout[2], 2.0F, 1.0F, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    const auto build_begin = Clock::now();
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    graph_example::submit_three(di, dout[1], n, 2.0F, 1.0F, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    const auto build_end = Clock::now();
    const auto upload_begin = Clock::now();
    CUDA_CHECK(cudaGraphUpload(exec, stream)); CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto upload_end = Clock::now();
    const std::array<const char*, 3> names = {"queued-three", "graph-three", "queued-fused"};
    auto launch = [&](int method) {
        if (method == 0) graph_example::submit_three(di, dout[method], n, 2.0F, 1.0F, stream);
        else if (method == 1) CUDA_CHECK(cudaGraphLaunch(exec, stream));
        else {
            graph_example::fused<<<(n + 255) / 256, 256, 0, stream>>>(di, dout[method], 2.0F, 1.0F, n);
            CUDA_CHECK(cudaGetLastError());
        }
    };
    auto validate = [&](int method) {
        CUDA_CHECK(cudaMemcpy(output.data(), dout[method], bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after.data(), di, bytes, cudaMemcpyDeviceToHost));
        if (!graph_example::validate(output.data(), n, 2.0F, 1.0F, names[method]) || after != input)
            std::exit(EXIT_FAILURE);
    };
    for (int method = 0; method < 3; ++method) {
        CUDA_CHECK(cudaMemsetAsync(dout[method], 0xFF, bytes, stream));
        launch(method); CUDA_CHECK(cudaStreamSynchronize(stream)); validate(method);
        for (int warm = 0; warm < 3; ++warm) launch(method);
        CUDA_CHECK(cudaStreamSynchronize(stream)); validate(method);
    }
    std::array<std::vector<double>, 3> submit_samples, total_samples, event_samples;
    std::cout << std::fixed << std::setprecision(6)
              << "graph_definition_instantiate_ms=" << ms(build_begin, build_end)
              << " graph_upload_wait_ms=" << ms(upload_begin, upload_end) << '\n';
    for (int round = 0; round < rounds; ++round) for (int slot = 0; slot < 3; ++slot) {
        const int method = (round + slot) % 3;
        CUDA_CHECK(cudaMemsetAsync(dout[method], 0xFF, bytes, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream)); // sentinel and prior work outside measured batch.
        const auto total_begin = Clock::now();
        CUDA_CHECK(cudaEventRecord(start, stream));
        const auto submit_begin = Clock::now();
        for (int repeat = 0; repeat < repeats; ++repeat) launch(method);
        const auto submit_end = Clock::now();
        CUDA_CHECK(cudaEventRecord(stop, stream)); CUDA_CHECK(cudaEventSynchronize(stop));
        const auto total_end = Clock::now();
        float elapsed = 0.0F; CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        const double submit = ms(submit_begin, submit_end), total = ms(total_begin, total_end);
        if (!std::isfinite(elapsed) || elapsed <= 0 || submit <= 0 || total <= 0) return EXIT_FAILURE;
        submit_samples[method].push_back(submit); total_samples[method].push_back(total);
        event_samples[method].push_back(elapsed); validate(method);
        std::cout << "sample round=" << round << " method=" << names[method]
                  << " host_submit_batch_ms=" << submit << " host_complete_batch_ms=" << total
                  << " event_batch_ms=" << elapsed << '\n';
    }
    for (int method = 0; method < 3; ++method) {
        const auto s = summarize(submit_samples[method]), t = summarize(total_samples[method]);
        const auto e = summarize(event_samples[method]);
        std::cout << names[method] << " submit_median_batch_ms=" << s.median
                  << " complete_min/median/mean/max_batch_ms="
                  << t.minimum << '/' << t.median << '/' << t.mean << '/' << t.maximum
                  << " event_median_per_workflow_us=" << 1000.0 * e.median / repeats << '\n';
    }
    CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(stop)); CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (auto p : dout) CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFree(di));
    std::cout << "PASS equal math, immutable input, independent outputs, rotated samples.\n";
    return EXIT_SUCCESS;
}
