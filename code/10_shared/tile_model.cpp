#include <cmath>
#include <iostream>
#include <vector>

float sample(const std::vector<float>& input, int i) {
    return i >= 0 && i < static_cast<int>(input.size()) ? input[i] : 0.0f;
}

int main() {
    constexpr int n = 10, width = 4;
    const int blocks = (n + width - 1) / width;
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i + 1);
    const float expected[n] = {3, 6, 9, 12, 15, 18, 21, 24, 27, 19};

    for (int block = 0; block < blocks; ++block) {
        float tile[width + 2] = {};
        const int base = block * width;
        // Model phase 1: all positions are prepared before phase 2.
        for (int tid = 0; tid < width; ++tid) {
            tile[tid + 1] = sample(input, base + tid);
        }
        tile[0] = sample(input, base - 1);
        tile[width + 1] = sample(input, base + width);
        std::cout << "block=" << block << " tile=";
        for (int j = 0; j < width + 2; ++j) {
            if (j != 0) std::cout << ',';
            std::cout << tile[j];
        }
        std::cout << '\n';
        // Model phase 2: consume the completed local tile.
        for (int tid = 0; tid < width; ++tid) {
            const int i = base + tid;
            if (i < n) output[i] = tile[tid] + tile[tid + 1] + tile[tid + 2];
        }
    }
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(output[i]) || output[i] != expected[i]) return 1;
        std::cout << "i=" << i << " sum=" << output[i] << '\n';
    }
    std::cout << "PASS CPU tile model; no CUDA threads or barriers executed\n";
    return 0;
}
