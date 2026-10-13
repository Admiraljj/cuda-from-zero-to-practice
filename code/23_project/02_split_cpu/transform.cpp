#include "transform.h"
void transform_cpu(const float* input,float* output,std::size_t n) {
    for (std::size_t i=0;i<n;++i) output[i]=2.0F*input[i]+1.0F;
}
