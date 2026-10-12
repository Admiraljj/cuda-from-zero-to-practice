#include <charconv>
#include <cstdlib>
#include <iostream>
#include <system_error>
#include <vector>

namespace {

bool parse_tile(const char* text, int& tile) {
    if (text == nullptr || *text == '\0') return false;
    const char* end = text;
    while (*end != '\0') ++end;
    int value = 0;
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) return false;
    if (value < 1 || value > 16) return false;
    tile = value;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int tile = 4;
    if (argc > 2 || (argc == 2 && !parse_tile(argv[1], tile))) {
        std::cerr << "Usage: " << argv[0] << " [tile:1..16]\n";
        return EXIT_FAILURE;
    }

    const int elements = tile * tile;
    std::vector<int> a_uses(static_cast<std::size_t>(elements), 0);
    std::vector<int> b_uses(static_cast<std::size_t>(elements), 0);

    for (int output_row = 0; output_row < tile; ++output_row) {
        for (int output_col = 0; output_col < tile; ++output_col) {
            for (int inner = 0; inner < tile; ++inner) {
                ++a_uses[output_row * tile + inner];
                ++b_uses[inner * tile + output_col];
            }
        }
    }

    for (int index = 0; index < elements; ++index) {
        if (a_uses[index] != tile || b_uses[index] != tile) {
            std::cerr << "FAIL reuse count at tile index " << index << '\n';
            return EXIT_FAILURE;
        }
    }

    const long long output_elements = 1LL * tile * tile;
    const long long multiply_add_terms = output_elements * tile;
    const long long naive_logical_loads = 2 * multiply_add_terms;
    const long long tiled_global_loads = 2 * output_elements;

    std::cout << "one C tile: " << tile << 'x' << tile << '\n'
              << "one K phase: " << tile << " terms per output\n"
              << "output elements: " << output_elements << '\n'
              << "multiply-add terms: " << multiply_add_terms << '\n'
              << "naive logical A+B reads: " << naive_logical_loads << '\n'
              << "tiled cooperative global A+B loads: "
              << tiled_global_loads << '\n'
              << "reuse per loaded A or B element: " << tile << '\n'
              << "logical request reduction factor: " << tile << "x\n"
              << "PASS each A tile element feeds one output row and each B "
                 "tile element feeds one output column.\n";
    return EXIT_SUCCESS;
}
