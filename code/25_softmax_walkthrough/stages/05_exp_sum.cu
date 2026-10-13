#include "cuda_check.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <vector>

__global__ void exp_and_sum(const float* input,float* exponentials,float* stats,int cols) {
    extern __shared__ float shared[];
    const int row=static_cast<int>(blockIdx.x),tid=static_cast<int>(threadIdx.x),base=row*cols;
    float local_max=-CUDART_INF_F;
    for(int col=tid;col<cols;col+=static_cast<int>(blockDim.x)) local_max=fmaxf(local_max,input[base+col]);
    shared[tid]=local_max;__syncthreads();
    for(unsigned stride=blockDim.x/2;stride>0;stride/=2) {
        if(threadIdx.x<stride) shared[tid]=fmaxf(shared[tid],shared[tid+stride]);
        __syncthreads();
    }
    const float maximum=shared[0];__syncthreads();
    float local_sum=0.0F;
    for(int col=tid;col<cols;col+=static_cast<int>(blockDim.x)) {
        const float e=expf(input[base+col]-maximum);exponentials[base+col]=e;local_sum+=e;
    }
    shared[tid]=local_sum;__syncthreads();
    for(unsigned stride=blockDim.x/2;stride>0;stride/=2) {
        if(threadIdx.x<stride) shared[tid]+=shared[tid+stride];
        __syncthreads();
    }
    if(tid==0) { stats[row*2]=maximum;stats[row*2+1]=shared[0]; }
}

int main() {
    const std::vector<float> input{1,2,3,-5,-5,-5};
    std::vector<float> output(6,std::numeric_limits<float>::quiet_NaN()),stats(4,std::numeric_limits<float>::quiet_NaN());
    float *d_input=nullptr,*d_output=nullptr,*d_stats=nullptr;int result=1;
    try {
        int devices=0;const auto probe=cudaGetDeviceCount(&devices);
        if(probe==cudaErrorNoDevice || probe==cudaErrorInsufficientDriver || (probe==cudaSuccess && devices==0)) {
            std::cout<<"NOT RUN: no usable CUDA device/driver.\n";return 77;
        }
        check_cuda(probe,"device count");
        check_cuda(cudaMalloc(&d_input,input.size()*sizeof(float)),"allocate input");
        check_cuda(cudaMalloc(&d_output,output.size()*sizeof(float)),"allocate output");
        check_cuda(cudaMalloc(&d_stats,stats.size()*sizeof(float)),"allocate stats");
        check_cuda(cudaMemcpy(d_input,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice),"upload");
        check_cuda(cudaMemcpy(d_output,output.data(),output.size()*sizeof(float),cudaMemcpyHostToDevice),"output sentinel");
        check_cuda(cudaMemcpy(d_stats,stats.data(),stats.size()*sizeof(float),cudaMemcpyHostToDevice),"stats sentinel");
        exp_and_sum<<<2,4,4*sizeof(float)>>>(d_input,d_output,d_stats,3);
        check_cuda(cudaGetLastError(),"exp/sum launch");check_cuda(cudaDeviceSynchronize(),"exp/sum complete");
        check_cuda(cudaMemcpy(output.data(),d_output,output.size()*sizeof(float),cudaMemcpyDeviceToHost),"exp download");
        check_cuda(cudaMemcpy(stats.data(),d_stats,stats.size()*sizeof(float),cudaMemcpyDeviceToHost),"stats download");
        for(int row=0;row<2;++row) {
            const double maximum=row==0?3.0:-5.0;double sum=0;
            for(int col=0;col<3;++col) {
                const double expected=std::exp(static_cast<double>(input[row*3+col])-maximum);sum+=expected;
                if(!std::isfinite(output[row*3+col]) || std::fabs(output[row*3+col]-expected)>1e-6)
                    throw std::runtime_error("exponential oracle mismatch");
            }
            if(stats[row*2]!=maximum || !std::isfinite(stats[row*2+1]) || std::fabs(stats[row*2+1]-sum)>1e-6)
                throw std::runtime_error("max/sum oracle mismatch");
        }
        std::cout<<"PASS intermediate exponentials, maxima and sums; not normalized yet.\n";
        result=0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; }
    auto cleanup=[&](cudaError_t status) {
        if(status!=cudaSuccess) { std::cerr<<"cleanup: "<<cudaGetErrorString(status)<<'\n';result=1; }
    };
    if(d_input || d_output || d_stats) cleanup(cudaDeviceSynchronize());
    if(d_stats) cleanup(cudaFree(d_stats));
    if(d_output) cleanup(cudaFree(d_output));
    if(d_input) cleanup(cudaFree(d_input));
    return result;
}
