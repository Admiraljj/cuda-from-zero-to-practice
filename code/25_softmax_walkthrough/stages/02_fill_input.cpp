#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    std::vector<float> input(6);
    for(float value:input) { value=7.0F;(void)value; } // Each value is a copy.
    std::cout<<"copy loop first="<<input[0]<<" (expected 0)\n";
    for(float& value:input) value=7.0F;
    std::cout<<"reference loop first="<<input[0]<<" (expected 7)\n";
    for(std::size_t i=0;i<input.size();++i) input[i]=static_cast<float>(i)-2.0F;
    for(float value:input) std::cout<<value<<' ';
    std::cout<<"\nDeterministic CPU input only; no GPU executed.\n";
    return input==std::vector<float>({-2,-1,0,1,2,3})?0:1;
}
