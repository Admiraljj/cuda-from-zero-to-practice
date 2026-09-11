#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    const std::size_t n = 5;
    const float alpha = 3.0f;
    std::vector<float> input(n);
    std::vector<float> output(n);

    for (std::size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i) + 1.0f;
    }
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = alpha * input[i];
    }

    // Small exactly representable values: exact comparison is appropriate here.
    const float expected[5] = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f};
    for (std::size_t i = 0; i < n; ++i) {
        if (output[i] != expected[i]) {
            std::cerr << "FAIL i=" << i << '\n';
            return 1;
        }
        std::cout << output[i] << '\n';
    }
    std::cout << "PASS scale\n";
    return 0;
}
