#ifndef BOOK25_INPUT_PATTERNS_H
#define BOOK25_INPUT_PATTERNS_H
#include "softmax_reference.h"
#include <charconv>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
namespace gpu {
inline bool parse_positive(const char* text,int maximum,int& value) {
    if(!text || !*text) return false;
    const char* end=text;while(*end) { if(*end<'0' || *end>'9') return false;++end; }
    int candidate=0;const auto result=std::from_chars(text,end,candidate);
    if(result.ec!=std::errc{} || result.ptr!=end || candidate<1 || candidate>maximum) return false;
    value=candidate;return true;
}
inline std::vector<float> make_input(int rows,int cols,const std::string& pattern) {
    if(pattern!="mixed" && pattern!="equal" && pattern!="negative" && pattern!="extreme")
        throw std::invalid_argument("unknown input pattern");
    std::vector<float> input(checked_elements(rows,cols));
    for(int row=0;row<rows;++row) for(int col=0;col<cols;++col) {
        const auto index=static_cast<std::size_t>(row)*cols+col;
        if(pattern=="equal") input[index]=-5.0F;
        else if(pattern=="negative") input[index]=-1.0F-static_cast<float>((row+col)%31);
        else if(pattern=="extreme") input[index]=col%3==0?1000.0F:(col%3==1?-1000.0F:999.0F);
        else input[index]=static_cast<float>((row*13+col*17)%101-50)/8.0F;
    }
    return input;
}
}
#endif
