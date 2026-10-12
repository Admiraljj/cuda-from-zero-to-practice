#include "filter_reference.h"
#include <cstdlib>
#include <utility>

int main(int argc, char** argv) {
    const std::string name = argc > 1 ? argv[1] : "gaussian";
    const std::string boundary = argc > 2 ? argv[2] : "zero";
    const std::string operation = argc > 3 ? argv[3] : "corr";
    if (argc > 4 || (name != "gaussian" && name != "sobel" && name != "asymmetric") ||
        (boundary != "zero" && boundary != "clamp") ||
        (operation != "corr" && operation != "conv")) {
        std::cerr << "Usage: filter_model [gaussian|sobel|asymmetric] [zero|clamp] [corr|conv]\n";
        return EXIT_FAILURE;
    }
    constexpr int width = 5, height = 5;
    std::vector<float> input(width * height);
    for (int i = 0; i < width * height; ++i) input[i] = static_cast<float>(i + 1);
    const auto f = filter_example::make_filter(name, 1, operation == "conv");
    std::vector<double> magnitude;
    const auto expected = filter_example::reference(input, width, height, f,
                                                    boundary == "clamp", magnitude);
    for (const auto point : {std::pair<int, int>{2, 2}, {0, 0}}) {
        const int x = point.first, y = point.second;
        std::cout << "output(x=" << x << ",y=" << y << ") terms:\n";
        double acc = 0.0;
        for (int fy = 0; fy < 3; ++fy) {
            for (int fx = 0; fx < 3; ++fx) {
                const int ix = x + fx - 1, iy = y + fy - 1;
                const float value = filter_example::sample(input, width, height, ix, iy,
                                                           boundary == "clamp");
                const float weight = f.weights[fy * 3 + fx];
                acc += static_cast<double>(value) * weight;
                std::cout << "  input(" << ix << ',' << iy << ")=" << value
                          << " filter[" << fy * 3 + fx << "]=" << weight
                          << " acc=" << acc << '\n';
            }
        }
        if (acc != expected[y * width + x]) return EXIT_FAILURE;
    }
    if (name == "gaussian" && expected[12] != 13.0) return EXIT_FAILURE;
    if (name == "gaussian" && boundary == "zero" && expected[0] != 1.6875) return EXIT_FAILURE;
    if (name == "gaussian" && boundary == "clamp" && expected[0] != 2.5) return EXIT_FAILURE;
    std::cout << "PASS center/corner trace agrees with the full CPU image.\n";
    return EXIT_SUCCESS;
}
