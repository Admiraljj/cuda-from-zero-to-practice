#ifndef BOOK24_CUDA_HELPERS_H
#define BOOK24_CUDA_HELPERS_H
#include <cuda_runtime.h>
#include <iostream>

namespace book24 {
inline bool check(cudaError_t status,int device,const char* operation) {
    if(status==cudaSuccess) return true;
    std::cerr << "device=" << device << " operation=" << operation << " error="
              << cudaGetErrorString(status) << '\n';return false;
}
inline int require_devices(int needed) {
    int count=0;const auto status=cudaGetDeviceCount(&count);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver ||
       (status==cudaSuccess && count<needed)) {
        std::cout << "NOT RUN: need " << needed << " visible CUDA device(s); found " << count << '\n';return 77;
    }
    return check(status,-1,"device count")?0:1;
}
}
#endif
