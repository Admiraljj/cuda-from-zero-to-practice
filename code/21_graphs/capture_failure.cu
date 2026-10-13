#include "../common/cuda_check.h"
#include <string>

__global__ void write_one(int* output) { *output = 1; }
int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "safe";
    if (argc > 2 || (mode != "safe" && mode != "invalid-sync")) {
        std::cerr << "Usage: capture_failure [safe|invalid-sync]\n"; return EXIT_FAILURE;
    }
    int* device = nullptr; CUDA_CHECK(cudaMalloc(&device, sizeof(int)));
    cudaStream_t stream{}; cudaGraph_t graph{}; cudaGraphExec_t exec{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (mode == "invalid-sync") {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        write_one<<<1, 1, 0, stream>>>(device); CUDA_CHECK(cudaGetLastError());
        // Intentional API-protocol error only: no OOB access, no data race.
        const cudaError_t bad = cudaStreamSynchronize(stream);
        cudaStreamCaptureStatus status{};
        CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
        const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
        std::cout << "forbidden_sync=" << cudaGetErrorString(bad)
                  << " invalidated=" << (status == cudaStreamCaptureStatusInvalidated)
                  << " end_capture=" << cudaGetErrorString(ended)
                  << " null_graph=" << (graph == nullptr) << '\n';
        if (bad != cudaErrorStreamCaptureUnsupported ||
            status != cudaStreamCaptureStatusInvalidated ||
            ended != cudaErrorStreamCaptureInvalidated || graph != nullptr) return EXIT_FAILURE;
        const cudaError_t last = cudaGetLastError();
        if (last != cudaSuccess && last != bad && last != ended) CUDA_CHECK(last);
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    write_one<<<1, 1, 0, stream>>>(device); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int output = 0;
    CUDA_CHECK(cudaMemcpy(&output, device, sizeof(int), cudaMemcpyDeviceToHost));
    if (output != 1) return EXIT_FAILURE;
    CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream)); CUDA_CHECK(cudaFree(device));
    std::cout << "PASS valid capture and result; mode=" << mode << '\n'; return EXIT_SUCCESS;
}
