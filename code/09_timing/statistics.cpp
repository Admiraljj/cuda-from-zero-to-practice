#include "benchmark_stats.h"
#include <iomanip>
#include <iostream>
#include <limits>

int main() {
    try {
        const std::vector<double> samples = {3.0, 1.0, 12.0, 2.0, 2.0};
        const TimingStats result = summarize(samples);
        if (result.count != 5 || result.minimum != 1.0 || result.median != 2.0 ||
            std::abs(result.mean - 4.0) > 1e-12 || result.maximum != 12.0) return 1;
        if (samples[0] != 3.0) return 1; // summarize sorts its own copy.
        const TimingStats even = summarize({4.0, 1.0, 3.0, 2.0});
        if (even.median != 2.5) return 1;

        int rejected = 0;
        const std::vector<std::vector<double>> bad = {
            {}, {0.0}, {-1.0}, {std::numeric_limits<double>::quiet_NaN()},
            {std::numeric_limits<double>::infinity()}
        };
        for (const auto& sample : bad) {
            try {
                (void)summarize(sample);
            } catch (const std::invalid_argument&) {
                ++rejected;
            }
        }
        if (rejected != 5) return 1;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "synthetic_samples_ms=3,1,12,2,2\n";
        std::cout << "count=" << result.count << " min_ms=" << result.minimum
                  << " median_ms=" << result.median << " mean_ms=" << result.mean
                  << " max_ms=" << result.maximum << '\n';
        std::cout << "even_median_ms=" << even.median << " rejected=" << rejected << '\n';
        std::cout << "PASS synthetic statistics; no GPU timing performed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
