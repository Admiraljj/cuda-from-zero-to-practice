#ifndef CUDA_BOOK_GRAPH_REFERENCE_H
#define CUDA_BOOK_GRAPH_REFERENCE_H
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace graph_example {
inline bool integer(const char* text, int low, int high, int& output) {
    if (text == nullptr || *text == '\0') return false;
    for (const char* p = text; *p; ++p) if (*p < '0' || *p > '9') return false;
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno || *end || value < low || value > high) return false;
    output = static_cast<int>(value); return true;
}
inline float input_value(std::size_t i) {
    return static_cast<float>(static_cast<int>(i % 17) - 8) / 4.0F;
}
inline float expected(float x, float alpha, float beta) {
    return std::max(alpha * x + beta, 0.0F);
}
inline bool validate(const float* output, int n, float alpha, float beta, const char* label) {
    for (int i = 0; i < n; ++i) {
        const float want = expected(input_value(i), alpha, beta);
        if (!std::isfinite(output[i]) || output[i] != want) {
            std::cerr << "FAIL " << label << " index=" << i
                      << " actual=" << output[i] << " expected=" << want << '\n';
            return false;
        }
    }
    return true;
}
}
#endif
