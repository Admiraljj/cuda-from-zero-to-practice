#ifndef CUDA_BOOK_FILTER_REFERENCE_H
#define CUDA_BOOK_FILTER_REFERENCE_H

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace filter_example {
struct Filter {
    int radius;
    bool separable;
    std::vector<float> weights, horizontal, vertical;
};

inline bool integer(const char* text, int low, int high, int& value) {
    if (!text || !*text) return false;
    const char* end = text;
    while (*end) ++end;
    int candidate = 0;
    const auto result = std::from_chars(text, end, candidate);
    if (result.ec != std::errc{} || result.ptr != end ||
        candidate < low || candidate > high) return false;
    value = candidate;
    return true;
}

inline Filter make_filter(const std::string& name, int radius, bool convolution) {
    Filter f{radius, true, {}, {}, {}};
    const int side = 2 * radius + 1;
    if (name == "gaussian") {
        f.horizontal = f.vertical = {0.25F, 0.5F, 0.25F};
    } else if (name == "sobel") {
        f.horizontal = {-1.0F, 0.0F, 1.0F};
        f.vertical = {1.0F, 2.0F, 1.0F};
    } else if (name == "box") {
        f.horizontal.assign(side, 1.0F / side);
        f.vertical.assign(side, 1.0F / side);
    } else {
        f.separable = false;
        f.weights = {1.0F, 2.0F, 0.0F, 0.0F, -1.0F, 3.0F, 2.0F, 0.0F, 1.0F};
    }
    if (f.separable) {
        f.weights.resize(static_cast<std::size_t>(side) * side);
        for (int y = 0; y < side; ++y) {
            for (int x = 0; x < side; ++x) {
                f.weights[y * side + x] = f.vertical[y] * f.horizontal[x];
            }
        }
    }
    if (convolution) {
        std::reverse(f.weights.begin(), f.weights.end());
        std::reverse(f.horizontal.begin(), f.horizontal.end());
        std::reverse(f.vertical.begin(), f.vertical.end());
    }
    return f;
}

inline float sample(const std::vector<float>& input, int width, int height,
                    int x, int y, bool clamp) {
    if (clamp) {
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
    } else if (x < 0 || y < 0 || x >= width || y >= height) {
        return 0.0F;
    }
    return input[static_cast<std::size_t>(y) * width + x];
}

inline std::vector<double> reference(const std::vector<float>& input,
                                     int width, int height, const Filter& f,
                                     bool clamp, std::vector<double>& magnitude) {
    const int side = 2 * f.radius + 1;
    std::vector<double> out(input.size(), 0.0);
    magnitude.assign(input.size(), 0.0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * width + x;
            for (int fy = 0; fy < side; ++fy) {
                for (int fx = 0; fx < side; ++fx) {
                    const double term = static_cast<double>(sample(
                        input, width, height, x + fx - f.radius,
                        y + fy - f.radius, clamp)) * f.weights[fy * side + fx];
                    out[index] += term;
                    magnitude[index] += std::abs(term);
                }
            }
        }
    }
    return out;
}

inline bool validate(const std::vector<float>& out, const std::vector<double>& expected,
                     const std::vector<double>& magnitude, const char* label) {
    double maximum = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double error = std::abs(static_cast<double>(out[i]) - expected[i]);
        const double bound = 1.0e-5 + 2.0e-5 * magnitude[i];
        if (!std::isfinite(out[i]) || error > bound) {
            std::cerr << "FAIL " << label << " index=" << i << " expected=" << expected[i]
                      << " actual=" << out[i] << " bound=" << bound << '\n';
            return false;
        }
        maximum = std::max(maximum, error);
    }
    std::cout << "check " << label << " max_abs_error=" << maximum << '\n';
    return true;
}
}  // namespace filter_example
#endif
