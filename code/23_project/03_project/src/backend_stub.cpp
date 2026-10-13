#include "backend.h"
namespace book23 {
int transform_cuda(const std::vector<float>&,std::vector<float>&,float,float,std::string& detail) {
    detail="CUDA backend was not built"; return 77;
}
int transform_cublas(const std::vector<float>&,std::vector<float>&,float,float,std::string& detail) {
    detail="cuBLAS backend was not built"; return 77;
}
}
