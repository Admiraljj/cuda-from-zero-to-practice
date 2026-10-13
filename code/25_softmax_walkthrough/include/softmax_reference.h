#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace gpu {
// Host-side helpers; the namespace name does not make these GPU functions.
std::size_t checked_elements(int rows, int cols);
std::vector<double> softmax_reference(const float* input, int rows, int cols);

struct ValidationResult {
    bool ok = false;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;  // Diagnostic only, denominator clamped to 1e-12.
    double max_row_sum_error = 0.0;
    std::string message;
};

ValidationResult validate_softmax(
    const float* gpu_output, const std::vector<double>& reference,
    int rows, int cols, double atol = 1e-5, double rtol = 1e-4,
    double row_sum_tol = 1e-4);
}  // namespace gpu
