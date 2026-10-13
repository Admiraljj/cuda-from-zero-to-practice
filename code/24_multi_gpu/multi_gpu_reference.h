#ifndef BOOK24_MULTI_GPU_REFERENCE_H
#define BOOK24_MULTI_GPU_REFERENCE_H
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <string>
#include <system_error>
#include <vector>

namespace book24 {
struct Range { std::size_t offset, count; };
inline Range partition(std::size_t n,int parts,int index) {
    const std::size_t q=n/static_cast<std::size_t>(parts), r=n%static_cast<std::size_t>(parts);
    const auto d=static_cast<std::size_t>(index);
    return {d*q+std::min(d,r),q+(d<r?1U:0U)};
}
inline bool parse(const char* text,int maximum,int& value) {
    if(!text || !*text) return false;
    const char* end=text;
    while(*end) { if(*end<'0' || *end>'9') return false; ++end; }
    int candidate=0;const auto result=std::from_chars(text,end,candidate);
    if(result.ec!=std::errc{} || result.ptr!=end || candidate>maximum) return false;
    value=candidate;return true;
}
inline std::vector<float> input(std::size_t n) {
    std::vector<float> values(n);
    for(std::size_t i=0;i<n;++i) values[i]=static_cast<float>(static_cast<int>(i%17)-8);
    return values;
}
inline bool verify(const float* actual,const std::vector<float>& source,int factor,int bias) {
    for(std::size_t i=0;i<source.size();++i)
        if(!std::isfinite(actual[i]) || actual[i]!=factor*source[i]+bias) return false;
    return true;
}
}
#endif
