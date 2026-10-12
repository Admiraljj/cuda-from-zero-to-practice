#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <system_error>
#include <utility>
#include <vector>

bool parse(const char* text, int low, int high, int& out) {
    if (!text || !*text) return false;
    const char* end = text;
    while (*end) ++end;
    int value = 0;
    const auto result = std::from_chars(text, end, value);
    if (result.ec != std::errc{} || result.ptr != end || value < low || value > high) return false;
    out = value;
    return true;
}

int main(int argc, char** argv) {
    int fraction_percent = 60, kernel_speedup = 3;
    if (argc > 3 || (argc > 1 && !parse(argv[1], 0, 100, fraction_percent)) ||
        (argc > 2 && !parse(argv[2], 1, 1000, kernel_speedup))) {
        std::cerr << "Usage: performance_model [fraction_percent:0..100] [speedup:1..1000]\n";
        return EXIT_FAILURE;
    }
    const double fraction = fraction_percent / 100.0;
    const double relative_new_time = 1.0 - fraction + fraction / kernel_speedup;
    std::cout << "SYNTHETIC MODEL, not measured GPU data\n"
              << "baseline_ms=100 new_ms=" << 100.0 * relative_new_time
              << " application_speedup=" << 1.0 / relative_new_time << '\n';
    std::vector<std::pair<int, int>> spans = {{0, 4}, {2, 6}};
    std::sort(spans.begin(), spans.end());
    int union_length = 0, start = spans.front().first, stop = spans.front().second;
    for (std::size_t i = 1; i < spans.size(); ++i) {
        if (spans[i].first > stop) {
            union_length += stop - start;
            start = spans[i].first;
            stop = spans[i].second;
        } else stop = std::max(stop, spans[i].second);
    }
    union_length += stop - start;
    if (union_length != 6 || !std::isfinite(relative_new_time) || relative_new_time <= 0.0) return EXIT_FAILURE;
    std::cout << "overlapping intervals [0,4],[2,6]: summed_duration=8 union_duration="
              << union_length << '\n'
              << "PASS Amdahl arithmetic and interval union; not a GPU execution test.\n";
    return EXIT_SUCCESS;
}
