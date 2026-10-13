#include "graph_workload.cuh"
#if CUDART_VERSION < 12000
#error "graph_update requires CUDA Toolkit 12.0 or newer."
#endif

__global__ void affine(const float* input, float* output, float alpha, float beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = alpha * input[i] + beta;
}
struct Definition { cudaGraph_t graph{}; cudaGraphNode_t affine_node{}, relu_node{}, extra{}; };
cudaKernelNodeParams affine_parameters(int n, void** arguments) {
    // arguments are supplied by the caller; CUDA copies argument VALUES during the API call.
    cudaKernelNodeParams p{};
    p.func = reinterpret_cast<void*>(affine);
    p.gridDim = dim3((n + 255) / 256); p.blockDim = dim3(256);
    p.kernelParams = arguments; return p;
}
Definition define(const float* input, float* output, int n, float alpha, float beta, bool extra) {
    Definition d;
    CUDA_CHECK(cudaGraphCreate(&d.graph, 0));
    void* arguments[] = {&input, &output, &alpha, &beta, &n};
    auto p = affine_parameters(n, arguments);
    CUDA_CHECK(cudaGraphAddKernelNode(&d.affine_node, d.graph, nullptr, 0, &p));
    void* relu_arguments[] = {&output, &n};
    cudaKernelNodeParams r{};
    r.func = reinterpret_cast<void*>(graph_example::relu);
    r.gridDim = p.gridDim; r.blockDim = p.blockDim; r.kernelParams = relu_arguments;
    CUDA_CHECK(cudaGraphAddKernelNode(&d.relu_node, d.graph, &d.affine_node, 1, &r));
    if (extra) CUDA_CHECK(cudaGraphAddEmptyNode(&d.extra, d.graph, &d.relu_node, 1));
    return d;
}
void update_or_rebuild(cudaGraphExec_t& exec, cudaGraph_t candidate, bool expect_topology_failure) {
    cudaGraphExecUpdateResultInfo info{};
    const cudaError_t error = cudaGraphExecUpdate(exec, candidate, &info);
    if (error == cudaSuccess && info.result == cudaGraphExecUpdateSuccess) {
        if (expect_topology_failure) { std::cerr << "FAIL topology change unexpectedly accepted.\n"; std::exit(EXIT_FAILURE); }
        std::cout << "whole_update=success\n"; return;
    }
    if (error != cudaErrorGraphExecUpdateFailure) CUDA_CHECK(error);
    if (error == cudaSuccess || (expect_topology_failure &&
        info.result != cudaGraphExecUpdateErrorTopologyChanged)) {
        std::cerr << "FAIL unexpected update result.\n"; std::exit(EXIT_FAILURE);
    }
    std::cout << "whole_update=rejected result=" << static_cast<int>(info.result) << " rebuild=yes\n";
    const cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess && last != cudaErrorGraphExecUpdateFailure) CUDA_CHECK(last);
    cudaGraphExec_t replacement{};
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&replacement, candidate, 0));
    CUDA_CHECK(cudaGraphExecDestroy(exec)); exec = replacement;
}
int main(int argc, char** argv) {
    int n = 1031;
    if (argc > 2 || (argc > 1 && !graph_example::integer(argv[1], 0, 16777216, n))) {
        std::cerr << "Usage: graph_update [n:0..16777216]\n"; return EXIT_FAILURE;
    }
    if (n == 0) { std::cout << "PASS empty input.\n"; return EXIT_SUCCESS; }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> input(n), output(n), after(n);
    for (int i = 0; i < n; ++i) input[i] = graph_example::input_value(i);
    float *di = nullptr, *dout = nullptr;
    CUDA_CHECK(cudaMalloc(&di, bytes)); CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(di, input.data(), bytes, cudaMemcpyHostToDevice));
    cudaStream_t stream{}; CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto original = define(di, dout, n, 2.0F, 1.0F, false);
    cudaGraphExec_t exec{}; CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, original.graph, 0));
    auto run = [&](float alpha, float beta, const char* label) {
        CUDA_CHECK(cudaMemsetAsync(dout, 0xFF, bytes, stream));
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaMemcpyAsync(output.data(), dout, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!graph_example::validate(output.data(), n, alpha, beta, label)) std::exit(EXIT_FAILURE);
        std::cout << "verified=" << label << " alpha=" << alpha << " beta=" << beta << '\n';
    };
    run(2.0F, 1.0F, "initial");
    const float* in = di; float* out = dout; float alpha = 4.0F, beta = -1.0F;
    void* arguments[] = {&in, &out, &alpha, &beta, &n};
    auto p = affine_parameters(n, arguments);
    CUDA_CHECK(cudaGraphKernelNodeSetParams(original.affine_node, &p));
    run(2.0F, 1.0F, "template-only-exec-unchanged");
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(exec, original.affine_node, &p));
    run(4.0F, -1.0F, "exec-node-update");
    auto compatible = define(di, dout, n, 1.0F, 2.0F, false);
    update_or_rebuild(exec, compatible.graph, false); // run() above already completed.
    run(1.0F, 2.0F, "whole-update-or-rebuild");
    auto changed = define(di, dout, n, 2.0F, -1.0F, true);
    update_or_rebuild(exec, changed.graph, true); // Added empty node changes topology.
    run(2.0F, -1.0F, "topology-rebuild");
    CUDA_CHECK(cudaMemcpy(after.data(), di, bytes, cudaMemcpyDeviceToHost));
    if (after != input) return EXIT_FAILURE;
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(changed.graph)); CUDA_CHECK(cudaGraphDestroy(compatible.graph));
    CUDA_CHECK(cudaGraphDestroy(original.graph)); CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dout)); CUDA_CHECK(cudaFree(di));
    std::cout << "PASS template/exec distinction, update/fallback, input unchanged.\n";
    return EXIT_SUCCESS;
}
