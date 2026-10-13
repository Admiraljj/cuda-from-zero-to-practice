#include "core.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
int main() {
    const auto fixed=book26::reference({-2,0,3,-1,4});
    if(fixed.selected!=std::vector<int>({3,4}) || fixed.square_sum!=25) return 1;
    for(int n:{0,1,7,255,256,257,1031,65537}) for(const char* mode:{"mixed","none","all"}) {
        const auto input=book26::make_input(n,mode);const auto result=book26::reference(input);
        book26::Sum independent=0;std::vector<int> selected;
        for(int value:input) if(value>0) { selected.push_back(value);independent+=static_cast<book26::Sum>(value)*value; }
        if(result.selected!=selected || result.square_sum!=independent) return 1;
    }
    auto wrong=fixed;wrong.selected={4,3};std::string detail;
    if(book26::verify(wrong,fixed,detail)) return 1;
    wrong=fixed;wrong.square_sum=24;if(book26::verify(wrong,fixed,detail)) return 1;
    int value=0;for(const char* bad:{"-1","+1","1x","1048577","999999999999999999"})
        if(book26::parse(bad,1048576,value)) return 1;
    int rejected=0;
    try { book26::reference({1001}); } catch(const std::invalid_argument&) { ++rejected; }
    try { book26::make_input(1,"unknown"); } catch(const std::invalid_argument&) { ++rejected; }
    if(rejected!=2) return 1;
    std::cout<<"PASS fixed 25, 24 CPU cases, order/sum rejection; no GPU executed.\n";return 0;
}
