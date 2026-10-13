#include "launch.h"
#include <cmath>
#include <iostream>
#include <vector>

bool check(cudaError_t status,const char* operation) {
    if(status==cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n'; return false;
}
int main() {
    int count=0;const cudaError_t probe=cudaGetDeviceCount(&count);
    if(probe==cudaErrorNoDevice || probe==cudaErrorInsufficientDriver || (probe==cudaSuccess && count==0)) {
        std::cout << "NOT RUN: no usable CUDA device/driver.\n";return 77;
    }
    if(!check(probe,"device count")) return 1;
    constexpr int n=7;std::vector<float> input(n),output(n);
    for(int i=0;i<n;++i) input[i]=static_cast<float>(i-3)/4.0F;
    float* device=nullptr;cudaStream_t stream{};bool ok=false;
    do {
        if(!check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream")) break;
        if(!check(cudaMalloc(&device,n*sizeof(float)),"allocate")) break;
        if(!check(cudaMemcpyAsync(device,input.data(),n*sizeof(float),cudaMemcpyHostToDevice,stream),"upload")) break;
        if(!check(launch_device_math(device,n,stream),"launch")) break;
        if(!check(cudaMemcpyAsync(output.data(),device,n*sizeof(float),cudaMemcpyDeviceToHost,stream),"download")) break;
        if(!check(cudaStreamSynchronize(stream),"wait")) break;
        ok=true;
        for(int i=0;i<n;++i)
            if(!std::isfinite(output[i]) || output[i]!=2.0F*input[i]+1.0F) ok=false;
    } while(false);
    if(stream) ok=check(cudaStreamSynchronize(stream),"cleanup wait")&&ok;
    if(device) ok=check(cudaFree(device),"free")&&ok;
    if(stream) ok=check(cudaStreamDestroy(stream),"destroy stream")&&ok;
    if(!ok) return 1;
    std::cout << "PASS real cross-.cu device call, all seven results checked.\n";return 0;
}
