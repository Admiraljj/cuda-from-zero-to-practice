#ifndef BOOK23_BACKEND_H
#define BOOK23_BACKEND_H
#include <string>
#include <vector>

namespace book23 {
// Synchronous host interface: input/output are host vectors, not device pointers.
// 0: completed; 1: real failure; 77: backend unavailable.
// Output is valid only on success. Neither function changes input.
int transform_cuda(const std::vector<float>& input,std::vector<float>& output,
                   float alpha,float beta,std::string& detail);
int transform_cublas(const std::vector<float>& input,std::vector<float>& output,
                     float alpha,float beta,std::string& detail);
}
#endif
