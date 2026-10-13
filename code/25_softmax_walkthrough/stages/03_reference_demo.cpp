#include "softmax_reference.h"
#include <iostream>
#include <vector>

int main() {
    const std::vector<float> input{1, 2, 3, -5, -5, -5};
    const auto reference = gpu::softmax_reference(input.data(), 2, 3);
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 3; ++col) std::cout << reference[row * 3 + col] << ' ';
        std::cout << '\n';
    }
    // CPU-rounded stand-in to exercise validation; not a GPU result.
    const std::vector<float> rounded(reference.begin(), reference.end());
    const auto report = gpu::validate_softmax(rounded.data(), reference, 2, 3);
    std::cout << report.message << " (CPU-only demonstration)\n";
    return report.ok ? 0 : 1;
}
