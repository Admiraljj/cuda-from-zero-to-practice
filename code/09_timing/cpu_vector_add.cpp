#include "benchmark_stats.h"
#include <chrono>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const int n = 1048576, rounds = 5;
        std::vector<float> a(n), b(n), c(n);
        std::vector<double> samples;
        samples.reserve(rounds);
        using Clock = std::chrono::steady_clock;
        // Round -1 warms the same source loop; recorded rounds are 0 through 4.
        for (int round = -1; round < rounds; ++round) {
            for (int i = 0; i < n; ++i) {
                a[i] = static_cast<float>(i % 251 + round);
                b[i] = static_cast<float>(i % 113 - 56);
                c[i] = 10000.0f;
            }
            const auto start = Clock::now();
            for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
            const auto stop = Clock::now();
            const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
            for (int i = 0; i < n; ++i) {
                const float expected = static_cast<float>(i % 251 + round + i % 113 - 56);
                if (!std::isfinite(c[i]) || c[i] != expected) return 1;
            }
            if (round >= 0) samples.push_back(ms);
        }
        const TimingStats result = summarize(samples);
        std::cout << std::fixed << std::setprecision(9);
        std::cout << "n=" << n << " rounds=" << rounds << " CPU_threads_requested=1\n";
        for (std::size_t i = 0; i < samples.size(); ++i) {
            std::cout << "round=" << i << " cpu_loop_ms=" << samples[i] << '\n';
        }
        std::cout << "min_ms=" << result.minimum << " median_ms=" << result.median
                  << " mean_ms=" << result.mean << " max_ms=" << result.maximum << '\n';
        std::cout << "PASS CPU vector add: every round checked; compiler may vectorize\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
