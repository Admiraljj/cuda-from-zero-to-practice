#include "transform.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    const std::vector<float> input={-2.0F,-0.5F,0.0F,1.0F,2.0F};
    const std::vector<float> expected={-3.0F,0.0F,1.0F,3.0F,5.0F};
    std::vector<float> output(input.size(),std::numeric_limits<float>::quiet_NaN());
    transform_cpu(input.data(),output.data(),input.size());
    for (std::size_t i=0;i<output.size();++i)
        if (!std::isfinite(output[i]) || output[i]!=expected[i]) return EXIT_FAILURE;
    transform_cpu(nullptr,nullptr,0); // Defined empty range: no pointer dereference.
    std::cout << "PASS split CPU: declaration, definition, fixed oracle and empty range.\n";
    return EXIT_SUCCESS;
}
