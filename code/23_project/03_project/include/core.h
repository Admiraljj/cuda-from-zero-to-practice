#ifndef BOOK23_CORE_H
#define BOOK23_CORE_H
#include <cstddef>
#include <string>
#include <vector>

namespace book23 {
bool parse_unsigned(const char* text,int maximum,int& result);
std::vector<float> make_input(std::size_t n,int seed);
std::vector<double> reference(const std::vector<float>& input,float alpha,float beta);
std::vector<float> transform_cpu(const std::vector<float>& input,float alpha,float beta);
bool verify(const std::vector<float>& actual,const std::vector<double>& expected,std::string& message);
}
#endif
