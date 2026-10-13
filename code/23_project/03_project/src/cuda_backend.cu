#include "backend.h"
#include "gpu_utils.h"
#include <limits>
#include <vector>

namespace {
__global__ void affine_kernel(const float* input,float* output,int n,float alpha,float beta) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) output[i]=alpha*input[i]+beta;
}
}
namespace book23 {
int transform_cuda(const std::vector<float>& input,std::vector<float>& output,
                   float alpha,float beta,std::string& detail) {
    if (input.empty()) { output.clear(); detail="empty; no GPU work"; return 0; }
    if (input.size()>16777216) { detail="backend input exceeds contract"; return 1; }
    std::string gpu;
    const int probe=probe_gpu(gpu);
    if (probe!=0) { detail=gpu; return probe; }
    detail.clear();
    const int n=static_cast<int>(input.size());
    const std::size_t bytes=input.size()*sizeof(float);
    output.assign(input.size(),std::numeric_limits<float>::quiet_NaN());
    std::vector<float> after(input.size());
    float *di=nullptr,*dout=nullptr; cudaStream_t stream{};
    bool ok=false;
    do {
        if (!runtime_ok(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream",detail)) break;
        if (!runtime_ok(cudaMalloc(&di,bytes),"allocate input",detail)) break;
        if (!runtime_ok(cudaMalloc(&dout,bytes),"allocate output",detail)) break;
        if (!runtime_ok(cudaMemcpyAsync(di,input.data(),bytes,cudaMemcpyHostToDevice,stream),"H2D input",detail)) break;
        if (!runtime_ok(cudaMemsetAsync(dout,0xFF,bytes,stream),"output sentinel",detail)) break;
        affine_kernel<<<(n+255)/256,256,0,stream>>>(di,dout,n,alpha,beta);
        if (!runtime_ok(cudaGetLastError(),"kernel launch",detail)) break;
        if (!runtime_ok(cudaMemcpyAsync(output.data(),dout,bytes,cudaMemcpyDeviceToHost,stream),"D2H output",detail)) break;
        if (!runtime_ok(cudaMemcpyAsync(after.data(),di,bytes,cudaMemcpyDeviceToHost,stream),"D2H input snapshot",detail)) break;
        if (!runtime_ok(cudaStreamSynchronize(stream),"wait for result",detail)) break;
        if (after!=input) { detail="device input changed"; break; }
        ok=true;
    } while (false);
    if (stream) ok=runtime_ok(cudaStreamSynchronize(stream),"cleanup wait",detail)&&ok;
    ok=free_resources(di,dout,stream,detail)&&ok;
    if (ok) detail="GPU="+gpu+"; synchronous CUDA roundtrip";
    return ok?0:1;
}
}
