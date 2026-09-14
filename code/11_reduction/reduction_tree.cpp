#include <array>
#include <iostream>
#include <vector>

constexpr int width = 8;

void print_state(int block, int stride, int active, const std::array<long long, width>& values) {
    std::cout << "block=" << block << " stride=" << stride << " active=" << active << " values=";
    for (int i = 0; i < width; ++i) {
        if (i != 0) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << '\n';
}

long long serial_sum(const std::vector<int>& input) {
    long long total = 0;
    for (int value : input) total += value;
    return total;
}

int main() {
    const std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    const int n = static_cast<int>(input.size());
    const int blocks = n / width + (n % width != 0);
    std::vector<long long> partial(blocks);
    for (int block = 0; block < blocks; ++block) {
        std::array<long long, width> values = {};
        const int base = block * width;
        for (int tid = 0; tid < width; ++tid) {
            if (base + tid < n) values[tid] = input[base + tid];
        }
        print_state(block, 0, width, values);
        for (int stride = width / 2; stride > 0; stride /= 2) {
            for (int tid = 0; tid < stride; ++tid) {
                values[tid] += values[tid + stride];
            }
            print_state(block, stride, stride, values);
        }
        partial[block] = values[0];
        long long expected_block = 0;
        for (int i = base; i < base + width && i < n; ++i) expected_block += input[i];
        if (partial[block] != expected_block) return 1;
    }
    long long total = 0;
    for (long long value : partial) total += value;
    if (partial != std::vector<long long>{36, 55} || total != 91 || total != serial_sum(input)) return 1;
    if (serial_sum({}) != 0 || serial_sum({-5}) != -5 || serial_sum({-3, 0, 2}) != -1) return 1;
    std::cout << "partial=36,55 total=91\n";
    std::cout << "PASS CPU reduction model; no concurrent threads or CUDA barriers executed\n";
    return 0;
}
