#ifndef BOOK26_CORE_H
#define BOOK26_CORE_H
#include <cstddef>
#include <string>
#include <vector>
namespace book26 {
using Sum=long long;
static_assert(sizeof(Sum)==8,"This example needs a 64-bit signed sum type");
struct Result { std::vector<int> selected;Sum square_sum=0; };
bool parse(const char* text,int maximum,int& value);
std::vector<int> make_input(int n,const std::string& mode);
Result reference(const std::vector<int>& input);
bool verify(const Result& actual,const Result& expected,std::string& detail);
}
#endif
