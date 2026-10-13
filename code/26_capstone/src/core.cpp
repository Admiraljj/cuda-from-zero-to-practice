#include "core.h"
#include <charconv>
#include <stdexcept>
#include <system_error>
namespace book26 {
bool parse(const char* text,int maximum,int& value) {
    if(!text || !*text) return false;
    const char* end=text;while(*end) { if(*end<'0' || *end>'9') return false;++end; }
    int candidate=0;const auto result=std::from_chars(text,end,candidate);
    if(result.ec!=std::errc{} || result.ptr!=end || candidate>maximum) return false;
    value=candidate;return true;
}
std::vector<int> make_input(int n,const std::string& mode) {
    if(n<0 || n>1048576 || (mode!="mixed" && mode!="none" && mode!="all"))
        throw std::invalid_argument("invalid count/mode");
    std::vector<int> input(static_cast<std::size_t>(n));
    for(int i=0;i<n;++i) input[i]=mode=="none"?-(i%1000+1):(mode=="all"?i%1000+1:(i*17)%2001-1000);
    return input;
}
Result reference(const std::vector<int>& input) {
    if(input.size()>1048576U) throw std::invalid_argument("count budget");
    Result result;
    for(int value:input) {
        if(value< -1000 || value>1000) throw std::invalid_argument("value budget");
        if(value>0) { result.selected.push_back(value);result.square_sum+=static_cast<Sum>(value)*value; }
    }
    return result;
}
bool verify(const Result& actual,const Result& expected,std::string& detail) {
    detail.clear();
    if(actual.selected!=expected.selected) { detail="selected length/values/stable order mismatch";return false; }
    if(actual.square_sum!=expected.square_sum) { detail="square sum mismatch";return false; }
    return true;
}
}
