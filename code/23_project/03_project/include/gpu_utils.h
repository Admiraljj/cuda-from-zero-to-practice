#ifndef BOOK23_GPU_UTILS_H
#define BOOK23_GPU_UTILS_H
#include <cuda_runtime.h>
#include <string>

namespace book23 {
inline bool runtime_ok(cudaError_t status,const char* operation,std::string& error) {
    if (status==cudaSuccess) return true;
    if (error.empty()) error=std::string(operation)+": "+cudaGetErrorString(status);
    return false;
}
inline int probe_gpu(std::string& detail) {
    int count=0;
    const cudaError_t status=cudaGetDeviceCount(&count);
    if (status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver ||
        (status==cudaSuccess && count==0)) {
        detail="no usable CUDA device/driver"; return 77;
    }
    if (!runtime_ok(status,"device count",detail)) return 1;
    cudaDeviceProp prop{};
    if (!runtime_ok(cudaGetDeviceProperties(&prop,0),"device properties",detail)) return 1;
    detail=prop.name; return 0;
}
// Caller has already attempted stream completion. Every release is attempted,
// even if a preceding release fails; the FIRST diagnostic is retained.
inline bool free_resources(float* input,float* output,cudaStream_t stream,std::string& error) {
    bool ok=true;
    if (output) ok=runtime_ok(cudaFree(output),"free output",error)&&ok;
    if (input) ok=runtime_ok(cudaFree(input),"free input",error)&&ok;
    if (stream) ok=runtime_ok(cudaStreamDestroy(stream),"destroy stream",error)&&ok;
    return ok;
}
}
#endif
