#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    const std::vector<float> input = {-2.0F,-0.5F,0.0F,1.0F,2.0F};
    const std::vector<float> expected = {-3.0F,0.0F,1.0F,3.0F,5.0F};
    std::vector<float> output(input.size());
    for (std::size_t i=0;i<input.size();++i) output[i]=2.0F*input[i]+1.0F;
    for (std::size_t i=0;i<input.size();++i) {
        if (!std::isfinite(output[i]) || output[i]!=expected[i]) return EXIT_FAILURE;
        std::cout << "i=" << i << " input=" << input[i] << " output=" << output[i] << '\n';
    }
    std::cout << "PASS single-file CPU transform and fixed oracle.\n";
    return EXIT_SUCCESS;
}
