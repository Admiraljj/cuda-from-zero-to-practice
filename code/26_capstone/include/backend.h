#ifndef BOOK26_BACKEND_H
#define BOOK26_BACKEND_H
#include "core.h"
#include <string>
#include <vector>
namespace book26 {
// Synchronous complete host roundtrip using logical GPU0, not a launch-only API.
// input immutable, size<=1048576, each value in [-1000,1000].
// 0=completed, 1=real failure, 77=unavailable; caller still verifies mathematics.
int run_cub(const std::vector<int>& input,Result& output,std::string& detail);
}
#endif
