#include "softmax_reference.h"
#include "input_patterns.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    for(int rows:{1,2,7}) for(int cols:{1,3,31,32,33,255,256,257,1003,16384})
        for(const char* pattern:{"mixed","equal","negative","extreme"}) {
            const auto input=gpu::make_input(rows,cols,pattern);
            const auto expected=gpu::softmax_reference(input.data(),rows,cols);
            const std::vector<float> rounded(expected.begin(),expected.end());
            if(!gpu::validate_softmax(rounded.data(),expected,rows,cols).ok) return 1;
            if(std::string(pattern)=="equal") for(double value:expected)
                if(std::fabs(value-1.0/cols)>1e-12) return 1;
        }
    const std::vector<float> input{1,2,3};
    const auto reference=gpu::softmax_reference(input.data(),1,3);
    std::vector<float> wrong(reference.begin(),reference.end());
    std::swap(wrong[0],wrong[2]); // Row sum still one: element checks must fail.
    if(gpu::validate_softmax(wrong.data(),reference,1,3).ok) return 1;
    wrong[0]=std::numeric_limits<float>::quiet_NaN();
    if(gpu::validate_softmax(wrong.data(),reference,1,3).ok) return 1;
    int rejected=0;
    try { gpu::checked_elements(0,3); } catch(const std::invalid_argument&) { ++rejected; }
    try { gpu::checked_elements(65535,16384); } catch(const std::invalid_argument&) { ++rejected; }
    try { gpu::softmax_reference(nullptr,1,3); } catch(const std::invalid_argument&) { ++rejected; }
    const float infinity=std::numeric_limits<float>::infinity();
    try { gpu::softmax_reference(&infinity,1,1); } catch(const std::invalid_argument&) { ++rejected; }
    try { gpu::validate_softmax(input.data(),reference,1,3,-1); } catch(const std::invalid_argument&) { ++rejected; }
    if(rejected!=5) return 1;
    int parsed=0;for(const char* bad:{"0","-1","+1","2x","999999999999999999"})
        if(gpu::parse_positive(bad,65535,parsed)) return 1;
    std::cout<<"PASS 120 CPU-reference cases and validator/rejection tests; no GPU executed.\n";return 0;
}
