#include "graph_workload.cuh"
#include <limits>

__global__ void merge_branches(const float* a, const float* b, float* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = a[i] + b[i];
}
int main(int argc, char** argv) {
    int n = 1031;
    if (argc > 2 || (argc > 1 && !graph_example::integer(argv[1], 0, 16777216, n))) {
        std::cerr << "Usage: graph_fork_join [n:0..16777216]\n"; return EXIT_FAILURE;
    }
    if (n == 0) { std::cout << "PASS empty input.\n"; return EXIT_SUCCESS; }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> input(n), output(n), after(n);
    for (int i = 0; i < n; ++i) input[i] = graph_example::input_value(i);
    float *di = nullptr, *a = nullptr, *b = nullptr, *dout = nullptr;
    CUDA_CHECK(cudaMalloc(&di, bytes)); CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes)); CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(di, input.data(), bytes, cudaMemcpyHostToDevice));
    cudaStream_t root{}, branch{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&root, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&branch, cudaStreamNonBlocking));
    cudaEvent_t fork{}, joined{};
    CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&joined, cudaEventDisableTiming));
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    const int blocks = (n + 255) / 256;
    CUDA_CHECK(cudaStreamBeginCapture(root, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(cudaMemsetAsync(dout, 0xFF, bytes, root)); // A real initial node for the fork.
    CUDA_CHECK(cudaEventRecord(fork, root));
    CUDA_CHECK(cudaStreamWaitEvent(branch, fork, 0));
    graph_example::scale_input<<<blocks, 256, 0, root>>>(di, a, 2.0F, n);
    CUDA_CHECK(cudaGetLastError());
    graph_example::scale_input<<<blocks, 256, 0, branch>>>(di, b, 3.0F, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(joined, branch));
    CUDA_CHECK(cudaStreamWaitEvent(root, joined, 0)); // Rejoin BEFORE merge AND EndCapture.
    merge_branches<<<blocks, 256, 0, root>>>(a, b, dout, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(root, &graph));
    std::size_t nodes = 0, edges = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    CUDA_CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, &edges));
    std::cout << "captured_nodes=" << nodes << " edges=" << edges << '\n';
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    for (int round = 0; round < 3; ++round) {
        CUDA_CHECK(cudaMemsetAsync(a, 0xFF, bytes, root));
        CUDA_CHECK(cudaMemsetAsync(b, 0xFF, bytes, root));
        CUDA_CHECK(cudaMemsetAsync(dout, 0xFF, bytes, root));
        CUDA_CHECK(cudaGraphLaunch(exec, root));
        CUDA_CHECK(cudaMemcpyAsync(output.data(), dout, bytes, cudaMemcpyDeviceToHost, root));
        CUDA_CHECK(cudaStreamSynchronize(root));
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(output[i]) || output[i] != 5.0F * input[i]) return EXIT_FAILURE;
        }
    }
    CUDA_CHECK(cudaMemcpy(after.data(), di, bytes, cudaMemcpyDeviceToHost));
    if (after != input) return EXIT_FAILURE;
    CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(joined)); CUDA_CHECK(cudaEventDestroy(fork));
    CUDA_CHECK(cudaStreamDestroy(branch)); CUDA_CHECK(cudaStreamDestroy(root));
    CUDA_CHECK(cudaFree(dout)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(di));
    std::cout << "PASS both producer dependencies, repeated outputs, input unchanged.\n";
    return EXIT_SUCCESS;
}
