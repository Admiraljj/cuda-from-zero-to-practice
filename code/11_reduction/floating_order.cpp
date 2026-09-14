#include <cfenv>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

float rounded_add(float a, float b) {
    volatile float result = a + b;
    return result;
}

int main() {
    if (!std::numeric_limits<float>::is_iec559 ||
        std::numeric_limits<float>::radix != 2 ||
        std::numeric_limits<float>::digits != 24 ||
        std::fegetround() != FE_TONEAREST) {
        std::cout << "SKIP: this lesson requires binary32 float and round-to-nearest\n";
        return 77;
    }
    const float a = 16777216.0f;
    const float b = 1.0f;
    const float c = -16777216.0f;
    const float left = rounded_add(rounded_add(a, b), c);
    const float right = rounded_add(a, rounded_add(b, c));
    const double reference = static_cast<double>(a) + static_cast<double>(b) + static_cast<double>(c);
    if (!std::isfinite(left) || !std::isfinite(right) || !std::isfinite(reference) ||
        left != 0.0f || right != 1.0f || reference != 1.0) {
        std::cerr << "unexpected arithmetic; check compiler and floating-point options\n";
        return 1;
    }
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "left_associated=" << left << " right_associated=" << right
              << " double_reference=" << reference << '\n';
    std::cout << "left_abs_error=" << std::fabs(static_cast<double>(left) - reference)
              << " right_abs_error=" << std::fabs(static_cast<double>(right) - reference) << '\n';
    std::cout << "PASS CPU floating-point order lesson; no GPU result measured\n";
    return 0;
}
