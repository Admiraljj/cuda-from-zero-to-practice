#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    const int n = 4096;
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i % 100);

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    for (int i = 0; i < n; ++i) output[i] = 2.0f * input[i] + 1.0f;
    const auto stop = Clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();

    double checksum = 0.0;
    for (int i = 0; i < n; ++i) {
        const float expected = static_cast<float>(2 * (i % 100) + 1);
        if (!std::isfinite(output[i]) || output[i] != expected) return 1;
        checksum += output[i];
    }
    if (!std::isfinite(ms) || ms < 0.0 || checksum != 409216.0) return 1;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "cpu_loop_ms=" << ms << '\n';
    std::cout << "first=" << output.front() << " last=" << output.back()
              << " checksum=" << checksum << '\n';
    std::cout << "PASS CPU clock lesson: 4096 elements; not a CPU/GPU benchmark\n";
    return 0;
}
