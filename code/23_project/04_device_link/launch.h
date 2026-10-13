#ifndef BOOK23_DEVICE_LAUNCH_H
#define BOOK23_DEVICE_LAUNCH_H
#include <cuda_runtime.h>
cudaError_t launch_device_math(float* values,int n,cudaStream_t stream);
#endif
