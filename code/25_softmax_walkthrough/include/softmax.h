#ifndef BOOK25_SOFTMAX_H
#define BOOK25_SOFTMAX_H
#include <cuda_runtime.h>
namespace gpu {
// Device ranges must be separate/non-overlapping and contain rows*cols floats.
// Caller guarantees finite input, positive dimensions, and valid lifetimes.
// These wrappers report SUBMISSION status, not execution completion.
cudaError_t launch_softmax(const float* input,float* output,int rows,int cols,
                           cudaStream_t stream=nullptr,int threads=256);
cudaError_t launch_serial_softmax(const float* input,float* output,int rows,int cols,
                                  cudaStream_t stream=nullptr);
cudaError_t launch_warp_softmax(const float* input,float* output,int rows,int cols,
                                cudaStream_t stream=nullptr);
}
#endif
