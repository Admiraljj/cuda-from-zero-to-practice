#include "softmax_reference.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gpu {
std::size_t checked_elements(int rows, int cols) {
    if (rows <= 0 || cols <= 0) throw std::invalid_argument("rows and cols must be positive");
    const auto r = static_cast<std::size_t>(rows);
    const auto c = static_cast<std::size_t>(cols);
    if (r > std::numeric_limits<std::size_t>::max() / c)
        throw std::overflow_error("element count overflow");
    const std::size_t n = r * c;
    if (rows > 65535 || cols > 16384 || n > 4194304)
        throw std::invalid_argument("teaching budget exceeded: rows<=65535, cols<=16384, elements<=4194304");
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::overflow_error("float byte count overflow");
    return n;
}

std::vector<double> softmax_reference(const float* input, int rows, int cols) {
    const std::size_t n = checked_elements(rows, cols);
    if (!input) throw std::invalid_argument("null input");
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(input[i])) throw std::invalid_argument("input must be finite");
    std::vector<double> result(n);
    for (int row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * cols;
        double maximum = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < cols; ++col)
            maximum = std::max(maximum, static_cast<double>(input[base + col]));
        double sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            result[base + col] = std::exp(static_cast<double>(input[base + col]) - maximum);
            sum += result[base + col];
        }
        for (int col = 0; col < cols; ++col) result[base + col] /= sum;
    }
    return result;
}

ValidationResult validate_softmax(const float* gpu_output,
    const std::vector<double>& reference, int rows, int cols,
    double atol, double rtol, double row_sum_tol) {
    const std::size_t n = checked_elements(rows, cols);
    if (!gpu_output || reference.size() != n)
        throw std::invalid_argument("output/reference size mismatch or null pointer");
    if (!std::isfinite(atol) || !std::isfinite(rtol) || !std::isfinite(row_sum_tol) ||
        atol < 0 || rtol < 0 || row_sum_tol < 0)
        throw std::invalid_argument("tolerances must be finite and nonnegative");
    ValidationResult report;
    bool ok = true;
    for (int row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * cols;
        double row_sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double got = gpu_output[base + col];
            const double expected = reference[base + col];
            if (!std::isfinite(got) || !std::isfinite(expected)) {
                report.message = "non-finite value at row=" + std::to_string(row) +
                                 " col=" + std::to_string(col);
                return report;
            }
            const double error = std::fabs(got - expected);
            report.max_abs_error = std::max(report.max_abs_error, error);
            report.max_rel_error = std::max(report.max_rel_error,
                error / std::max(std::fabs(expected), 1e-12));
            if (got < 0 || expected < 0 || error > atol + rtol * std::fabs(expected)) ok = false;
            row_sum += got;
        }
        const double row_error = std::fabs(row_sum - 1.0);
        report.max_row_sum_error = std::max(report.max_row_sum_error, row_error);
        if (!std::isfinite(row_sum) || row_error > row_sum_tol) ok = false;
    }
    report.ok = ok;
    report.message = ok ? "PASS" : "element tolerance, nonnegativity, or row-sum check failed";
    return report;
}
}  // namespace gpu
