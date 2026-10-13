#ifndef CUDA_BOOK_LIBRARY_REFERENCE_H
#define CUDA_BOOK_LIBRARY_REFERENCE_H
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace library_example {
inline bool integer(const char* text, int low, int high, int& result) {
    if (!text || !*text) return false;
    const char* end = text;
    while (*end) { if (*end < '0' || *end > '9') return false; ++end; }
    int candidate = 0;
    const auto parsed = std::from_chars(text, end, candidate);
    if (parsed.ec != std::errc{} || parsed.ptr != end || candidate < low || candidate > high) return false;
    result = candidate; return true;
}
inline float x_value(int i) { return static_cast<float>(i % 17 - 8) / 4.0F; }
inline float y_value(int i) { return static_cast<float>(i % 11 - 5) / 8.0F; }
inline long long integer_value(int i, const std::string& mode) {
    if (mode == "none") return -(i % 5 + 1LL);
    if (mode == "all") return i % 5 + 1LL;
    return i % 11 - 5LL;
}
inline bool saxpy_valid(const std::vector<float>& actual, const char* label) {
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float want = 2.0F * x_value(static_cast<int>(i)) + y_value(static_cast<int>(i));
        if (!std::isfinite(actual[i]) || actual[i] != want) {
            std::cerr << "FAIL " << label << " index=" << i << " actual=" << actual[i]
                      << " expected=" << want << '\n'; return false;
        }
    }
    return true;
}
}
#endif
