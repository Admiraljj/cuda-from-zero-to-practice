#include "core.h"
#include <charconv>
#include <cmath>
#include <system_error>

namespace book23 {
bool parse_unsigned(const char* text,int maximum,int& result) {
    if (!text || !*text) return false;
    const char* end=text;
    while (*end) { if (*end<'0' || *end>'9') return false; ++end; }
    int candidate=0;
    const auto parsed=std::from_chars(text,end,candidate);
    if (parsed.ec!=std::errc{} || parsed.ptr!=end || candidate<0 || candidate>maximum) return false;
    result=candidate; return true;
}
std::vector<float> make_input(std::size_t n,int seed) {
    std::vector<float> input(n);
    for (std::size_t i=0;i<n;++i)
        input[i]=static_cast<float>(static_cast<int>((i*17+static_cast<std::size_t>(seed))%31)-15)/8.0F;
    return input;
}
std::vector<double> reference(const std::vector<float>& input,float alpha,float beta) {
    std::vector<double> expected(input.size());
    for (std::size_t i=0;i<input.size();++i)
        expected[i]=static_cast<double>(alpha)*static_cast<double>(input[i])+static_cast<double>(beta);
    return expected;
}
std::vector<float> transform_cpu(const std::vector<float>& input,float alpha,float beta) {
    std::vector<float> output(input.size());
    for (std::size_t i=0;i<input.size();++i) output[i]=alpha*input[i]+beta;
    return output;
}
bool verify(const std::vector<float>& actual,const std::vector<double>& expected,std::string& message) {
    message.clear();
    if (actual.size()!=expected.size()) { message="output length mismatch"; return false; }
    for (std::size_t i=0;i<actual.size();++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
            static_cast<double>(actual[i])!=expected[i]) {
            message="output mismatch at index "+std::to_string(i); return false;
        }
    }
    return true;
}
}
