#include <charconv>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

bool parse_int(std::string_view text, int& value) {
    const char* first = text.data();
    const char* last = text.data() + text.size();
    const auto result = std::from_chars(first, last, value);
    return result.ec == std::errc{} && result.ptr == last;
}

std::vector<std::int64_t> exclusive_scan_reference(
    const std::vector<int>& input) {
    std::vector<std::int64_t> output(input.size());
    std::int64_t running = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = running;
        running += input[i];
    }
    return output;
}

std::vector<std::int64_t> inclusive_scan_reference(
    const std::vector<int>& input) {
    std::vector<std::int64_t> output(input.size());
    std::int64_t running = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        running += input[i];
        output[i] = running;
    }
    return output;
}

template <typename T>
void print_vector(std::string_view name, const std::vector<T>& values) {
    std::cout << name << "=[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << values[i];
    }
    std::cout << "]\n";
}

bool validate(const std::vector<int>& input,
              const std::vector<std::int64_t>& exclusive,
              const std::vector<std::int64_t>& inclusive) {
    if (exclusive.size() != input.size() ||
        inclusive.size() != input.size()) return false;

    for (std::size_t i = 0; i < input.size(); ++i) {
        if (inclusive[i] != exclusive[i] + input[i]) return false;
        if (i == 0) {
            if (exclusive[i] != 0) return false;
        } else if (exclusive[i] != inclusive[i - 1]) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> input;
    if (argc == 1) {
        input = {3, 1, 7, 0, 4, 1, 6, 3};
    } else {
        constexpr int kMagnitudeLimit = 1'000'000;
        constexpr int kItemLimit = 64;
        if (argc - 1 > kItemLimit) {
            std::cerr << "error: at most " << kItemLimit << " integers\n";
            return 2;
        }
        input.reserve(static_cast<std::size_t>(argc - 1));
        for (int i = 1; i < argc; ++i) {
            int value = 0;
            if (!parse_int(argv[i], value) || value < -kMagnitudeLimit ||
                value > kMagnitudeLimit) {
                std::cerr << "error: argument " << i
                          << " must be an integer in [-1000000, 1000000]\n";
                return 2;
            }
            input.push_back(value);
        }
    }

    const auto exclusive = exclusive_scan_reference(input);
    const auto inclusive = inclusive_scan_reference(input);
    if (!validate(input, exclusive, inclusive)) {
        std::cerr << "FAIL: scan invariants do not hold\n";
        return 1;
    }

    print_vector("input", input);
    print_vector("exclusive", exclusive);
    print_vector("inclusive", inclusive);
    const std::int64_t total = inclusive.empty() ? 0 : inclusive.back();
    std::cout << "total=" << total << '\n';
    std::cout << "PASS: every prefix and cross-relation validated\n";
    return 0;
}
