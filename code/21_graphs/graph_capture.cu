#include "graph_workload.cuh"
#include <limits>

int main(int argc, char** argv) {
    int n = 1031, repeats = 3;
    if (argc > 3 || (argc > 1 && !graph_example::integer(argv[1], 0, 16777216, n)) ||
        (argc > 2 && !graph_example::integer(argv[2], 1, 1000, repeats))) {
        std::cerr << "Usage: graph_capture [n:0..16777216] [repeats:1..1000]\n";
        return EXIT_FAILURE;
    }
    if (n == 0) { std::cout << "PASS empty input; no allocation or capture.\n"; return EXIT_SUCCESS; }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    float *hi = nullptr, *ho = nullptr, *di = nullptr, *dout = nullptr;
    CUDA_CHECK(cudaMallocHost(&hi, bytes)); CUDA_CHECK(cudaMallocHost(&ho, bytes));
    CUDA_CHECK(cudaMalloc(&di, bytes)); CUDA_CHECK(cudaMalloc(&dout, bytes));
    for (int i = 0; i < n; ++i) hi[i] = graph_example::input_value(i);
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    float alpha = 2.0F, beta = 1.0F;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(cudaMemcpyAsync(di, hi, bytes, cudaMemcpyHostToDevice, stream));
    graph_example::submit_three(di, dout, n, alpha, beta, stream);
    CUDA_CHECK(cudaMemcpyAsync(ho, dout, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    std::size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    std::cout << "captured_nodes=" << nodes << " n=" << n << '\n';
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    beta = 99.0F; // Does NOT update the already recorded kernel scalar argument.
    for (int round = 0; round < repeats; ++round) {
        for (int i = 0; i < n; ++i) {
            hi[i] = graph_example::input_value(i);
            ho[i] = std::numeric_limits<float>::quiet_NaN();
        }
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream)); // Protect host read AND next host-buffer rewrite.
        if (!graph_example::validate(ho, n, 2.0F, 1.0F, "captured")) return EXIT_FAILURE;
        for (int i = 0; i < n; ++i) if (hi[i] != graph_example::input_value(i)) return EXIT_FAILURE;
    }
    std::vector<float> after(n);
    CUDA_CHECK(cudaMemcpy(after.data(), di, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) if (after[i] != hi[i]) return EXIT_FAILURE;
    CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream)); CUDA_CHECK(cudaFree(dout)); CUDA_CHECK(cudaFree(di));
    CUDA_CHECK(cudaFreeHost(ho)); CUDA_CHECK(cudaFreeHost(hi));
    std::cout << "PASS repeated fresh-input workflows; captured beta=1, host beta=" << beta << ".\n";
    return EXIT_SUCCESS;
}
