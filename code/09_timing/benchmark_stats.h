#ifndef CUDA_BOOK_BENCHMARK_STATS_H
#define CUDA_BOOK_BENCHMARK_STATS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

struct TimingStats {
    std::size_t count;
    double minimum;
    double median;
    double mean;
    double maximum;
};

inline TimingStats summarize(std::vector<double> samples) {
    if (samples.empty()) throw std::invalid_argument("no timing samples");
    double mean = 0.0;
    for (double value : samples) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument("timing samples must be finite and positive");
        }
        mean += value / static_cast<double>(samples.size());
    }
    if (!std::isfinite(mean)) throw std::overflow_error("timing mean overflow");
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    const double median = samples.size() % 2 != 0
        ? samples[middle]
        : samples[middle - 1] / 2.0 + samples[middle] / 2.0;
    return {samples.size(), samples.front(), median, mean, samples.back()};
}

#endif
