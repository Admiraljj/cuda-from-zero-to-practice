#include "backend.h"
#include "gpu_utils.h"
#include <cublas_v2.h>
#include <vector>

namespace {
bool blas_ok(cublasStatus_t status,const char* operation,std::string& error) {
    if (status==CUBLAS_STATUS_SUCCESS) return true;
    if (error.empty()) error=std::string(operation)+": cuBLAS status="+std::to_string(static_cast<int>(status));
    return false;
}
}
namespace book23 {
int transform_cublas(const std::vector<float>& input,std::vector<float>& output,
                     float alpha,float beta,std::string& detail) {
    if (input.empty()) { output.clear(); detail="empty; no cuBLAS work"; return 0; }
    if (input.size()>16777216) { detail="backend input exceeds contract"; return 1; }
    std::string gpu;
    const int probe=probe_gpu(gpu);
    if (probe!=0) { detail=gpu; return probe; }
    detail.clear();
    const int n=static_cast<int>(input.size());
    const std::size_t bytes=input.size()*sizeof(float);
    std::vector<float> old_output(input.size(),beta),after(input.size());
    output.resize(input.size());
    float *di=nullptr,*dout=nullptr; cudaStream_t stream{}; cublasHandle_t handle{};
    bool ok=false;
    do {
        if (!runtime_ok(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream",detail)) break;
        if (!runtime_ok(cudaMalloc(&di,bytes),"allocate input",detail)) break;
        if (!runtime_ok(cudaMalloc(&dout,bytes),"allocate output",detail)) break;
        if (!runtime_ok(cudaMemcpyAsync(di,input.data(),bytes,cudaMemcpyHostToDevice,stream),"H2D input",detail)) break;
        if (!runtime_ok(cudaMemcpyAsync(dout,old_output.data(),bytes,cudaMemcpyHostToDevice,stream),"H2D beta values",detail)) break;
        if (!blas_ok(cublasCreate(&handle),"create handle",detail)) break;
        if (!blas_ok(cublasSetStream(handle,stream),"bind stream",detail)) break;
        if (!blas_ok(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST),"scalar mode",detail)) break;
        if (!blas_ok(cublasSaxpy(handle,n,&alpha,di,1,dout,1),"SAXPY",detail)) break;
        if (!runtime_ok(cudaGetLastError(),"library launch state",detail)) break;
        if (!runtime_ok(cudaMemcpyAsync(output.data(),dout,bytes,cudaMemcpyDeviceToHost,stream),"D2H output",detail)) break;
        if (!runtime_ok(cudaMemcpyAsync(after.data(),di,bytes,cudaMemcpyDeviceToHost,stream),"D2H input snapshot",detail)) break;
        if (!runtime_ok(cudaStreamSynchronize(stream),"wait for result",detail)) break;
        if (after!=input) { detail="device input changed"; break; }
        ok=true;
    } while (false);
    if (stream) ok=runtime_ok(cudaStreamSynchronize(stream),"cleanup wait",detail)&&ok;
    if (handle) ok=blas_ok(cublasDestroy(handle),"destroy handle",detail)&&ok;
    ok=free_resources(di,dout,stream,detail)&&ok;
    if (ok) detail="GPU="+gpu+"; synchronous cuBLAS roundtrip";
    return ok?0:1;
}
}
