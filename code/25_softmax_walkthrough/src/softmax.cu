#include "softmax.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>

namespace {
__global__ void serial_rows(const float* input,float* output,int rows,int cols) {
    const int row=static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);
    if(row>=rows) return;
    const std::size_t base=static_cast<std::size_t>(row)*cols;
    float maximum=-CUDART_INF_F;
    for(int col=0;col<cols;++col) maximum=fmaxf(maximum,input[base+col]);
    float sum=0.0F;
    for(int col=0;col<cols;++col) { const float e=expf(input[base+col]-maximum);output[base+col]=e;sum+=e; }
    for(int col=0;col<cols;++col) output[base+col]/=sum;
}
__global__ void block_rows(const float* input,float* output,int cols) {
    extern __shared__ float shared[];
    const int row=static_cast<int>(blockIdx.x),tid=static_cast<int>(threadIdx.x);
    const std::size_t base=static_cast<std::size_t>(row)*cols;
    float local_max=-CUDART_INF_F;
    for(int col=tid;col<cols;col+=static_cast<int>(blockDim.x)) local_max=fmaxf(local_max,input[base+col]);
    shared[tid]=local_max;
    __syncthreads();
    for(unsigned stride=blockDim.x/2;stride>0;stride/=2) {
        if(threadIdx.x<stride) shared[tid]=fmaxf(shared[tid],shared[tid+stride]);
        __syncthreads();
    }
    const float row_max=shared[0];
    __syncthreads(); // Every thread has read the old shared[0] before reuse.
    float local_sum=0.0F;
    for(int col=tid;col<cols;col+=static_cast<int>(blockDim.x)) {
        const float e=expf(input[base+col]-row_max);
        output[base+col]=e;local_sum+=e;
    }
    shared[tid]=local_sum;
    __syncthreads();
    for(unsigned stride=blockDim.x/2;stride>0;stride/=2) {
        if(threadIdx.x<stride) shared[tid]+=shared[tid+stride];
        __syncthreads();
    }
    const float row_sum=shared[0];
    for(int col=tid;col<cols;col+=static_cast<int>(blockDim.x)) output[base+col]/=row_sum;
}
__global__ void warp_rows(const float* input,float* output,int rows,int cols) {
    const unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int row=static_cast<int>(blockIdx.x*4+warp);
    if(row>=rows) return; // Uniform decision for the entire warp.
    const std::size_t base=static_cast<std::size_t>(row)*cols;
    float maximum=lane<static_cast<unsigned>(cols)?input[base+lane]:-CUDART_INF_F;
    for(int offset=16;offset>0;offset/=2) maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffU,maximum,offset));
    maximum=__shfl_sync(0xffffffffU,maximum,0);
    const float e=lane<static_cast<unsigned>(cols)?expf(input[base+lane]-maximum):0.0F;
    float sum=e;
    for(int offset=16;offset>0;offset/=2) sum+=__shfl_down_sync(0xffffffffU,sum,offset);
    sum=__shfl_sync(0xffffffffU,sum,0);
    if(lane<static_cast<unsigned>(cols)) output[base+lane]=e/sum;
}
cudaError_t properties(const float* input,float* output,int rows,int cols,cudaDeviceProp& prop) {
    if(!input || !output || input==output || rows<=0 || rows>65535 || cols<=0 || cols>16384 ||
       static_cast<std::size_t>(rows)*static_cast<std::size_t>(cols)>4194304U) return cudaErrorInvalidValue;
    int device=0;cudaError_t status=cudaGetDevice(&device);
    if(status!=cudaSuccess) return status;
    return cudaGetDeviceProperties(&prop,device);
}
}
namespace gpu {
cudaError_t launch_softmax(const float* input,float* output,int rows,int cols,cudaStream_t stream,int threads) {
    cudaDeviceProp prop{};auto status=properties(input,output,rows,cols,prop);
    if(status!=cudaSuccess) return status;
    if(threads<32 || threads>512 || (threads&(threads-1)) || threads>prop.maxThreadsPerBlock ||
       rows>prop.maxGridSize[0] || static_cast<std::size_t>(threads)*sizeof(float)>prop.sharedMemPerBlock)
        return cudaErrorInvalidConfiguration;
    block_rows<<<rows,threads,static_cast<std::size_t>(threads)*sizeof(float),stream>>>(input,output,cols);
    return cudaGetLastError();
}
cudaError_t launch_serial_softmax(const float* input,float* output,int rows,int cols,cudaStream_t stream) {
    cudaDeviceProp prop{};auto status=properties(input,output,rows,cols,prop);
    if(status!=cudaSuccess) return status;
    const int blocks=(rows+127)/128;
    if(prop.maxThreadsPerBlock<128 || blocks>prop.maxGridSize[0]) return cudaErrorInvalidConfiguration;
    serial_rows<<<blocks,128,0,stream>>>(input,output,rows,cols);return cudaGetLastError();
}
cudaError_t launch_warp_softmax(const float* input,float* output,int rows,int cols,cudaStream_t stream) {
    cudaDeviceProp prop{};auto status=properties(input,output,rows,cols,prop);
    if(status!=cudaSuccess) return status;
    const int blocks=(rows+3)/4;
    if(cols>32) return cudaErrorInvalidValue;
    if(prop.warpSize!=32 || prop.maxThreadsPerBlock<128 || blocks>prop.maxGridSize[0]) return cudaErrorInvalidConfiguration;
    warp_rows<<<blocks,128,0,stream>>>(input,output,rows,cols);return cudaGetLastError();
}
}
