#include "launch.h"
#include "device_math.cuh"
namespace {
__global__ void apply(float* values,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) values[i]=device_affine(values[i]);
}
}
cudaError_t launch_device_math(float* values,int n,cudaStream_t stream) {
    apply<<<(n+127)/128,128,0,stream>>>(values,n);
    return cudaGetLastError();
}
