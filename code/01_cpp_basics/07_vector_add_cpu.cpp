#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    const std::size_t n = 8;
    std::vector<float> a(n);
    std::vector<float> b(n);
    std::vector<float> c(n);

    // Step 1: prepare inputs.
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f * static_cast<float>(i);
    }

    // Step 2: compute, one element at a time.
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }

    // Step 3: check the result against this input's known formula.
    for (std::size_t i = 0; i < n; ++i) {
        const float expected = 3.0f * static_cast<float>(i);
        const float error = std::fabs(c[i] - expected);
        if (!std::isfinite(c[i]) || error > 1e-6f) {
            std::cerr << "FAIL i=" << i
                      << " got=" << c[i]
                      << " expected=" << expected << '\n';
            return 1;
        }
        std::cout << "c[" << i << "]=" << c[i] << '\n';
    }

    std::cout << "PASS n=" << n << '\n';
    return 0;
}
