#include "cuda_check.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <vector>

__global__ void row_max(const float* input,float* maximum,int cols) {
    extern __shared__ float shared[];
    const int row=static_cast<int>(blockIdx.x),tid=static_cast<int>(threadIdx.x);
    const int base=row*cols;
    float local=-CUDART_INF_F;
    for(int col=tid;col<cols;col+=static_cast<int>(blockDim.x)) local=fmaxf(local,input[base+col]);
    shared[tid]=local;__syncthreads();
    for(unsigned stride=blockDim.x/2;stride>0;stride/=2) {
        if(threadIdx.x<stride) shared[tid]=fmaxf(shared[tid],shared[tid+stride]);
        __syncthreads();
    }
    if(tid==0) maximum[row]=shared[0];
}

int main() {
    const std::vector<float> input{1,2,3,-5,-5,-5};
    std::vector<float> output(2,std::numeric_limits<float>::quiet_NaN());
    float *d_input=nullptr,*d_output=nullptr;int result=1;
    try {
        int devices=0;const auto probe=cudaGetDeviceCount(&devices);
        if(probe==cudaErrorNoDevice || probe==cudaErrorInsufficientDriver || (probe==cudaSuccess && devices==0)) {
            std::cout<<"NOT RUN: no usable CUDA device/driver.\n";return 77;
        }
        check_cuda(probe,"device count");
        check_cuda(cudaMalloc(&d_input,input.size()*sizeof(float)),"allocate input");
        check_cuda(cudaMalloc(&d_output,output.size()*sizeof(float)),"allocate output");
        check_cuda(cudaMemcpy(d_input,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice),"upload");
        check_cuda(cudaMemcpy(d_output,output.data(),output.size()*sizeof(float),cudaMemcpyHostToDevice),"output sentinel");
        row_max<<<2,4,4*sizeof(float)>>>(d_input,d_output,3);
        check_cuda(cudaGetLastError(),"max launch");check_cuda(cudaDeviceSynchronize(),"max complete");
        check_cuda(cudaMemcpy(output.data(),d_output,2*sizeof(float),cudaMemcpyDeviceToHost),"max download");
        if(output[0]!=3.0F || output[1]!=-5.0F) throw std::runtime_error("max oracle mismatch");
        std::cout<<"PASS row maxima: 3, -5; four threads, three columns.\n";
        result=0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; }
    auto cleanup=[&](cudaError_t status) {
        if(status!=cudaSuccess) { std::cerr<<"cleanup: "<<cudaGetErrorString(status)<<'\n';result=1; }
    };
    if(d_input || d_output) cleanup(cudaDeviceSynchronize());
    if(d_output) cleanup(cudaFree(d_output));
    if(d_input) cleanup(cudaFree(d_input));
    return result;
}
