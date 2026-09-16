#include <charconv>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

bool parse_dimension(std::string_view text, int& value) {
    const char* first = text.data();
    const char* last = text.data() + text.size();
    const auto result = std::from_chars(first, last, value);
    return result.ec == std::errc{} && result.ptr == last &&
           value >= 1 && value <= 16;
}
void print_matrix(std::string_view name, const std::vector<int>& values,
                  int rows, int cols) {
    std::cout << name << " rows=" << rows << " cols=" << cols << '\n';
    for (int row = 0; row < rows; ++row) {
        std::cout << "  [";
        for (int col = 0; col < cols; ++col) {
            if (col) std::cout << ", ";
            std::cout << values[row * cols + col];
        }
        std::cout << "]\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    int width = 3;
    int height = 2;
    if (argc != 1 &&
        (argc != 3 || !parse_dimension(argv[1], width) ||
         !parse_dimension(argv[2], height))) {
        std::cerr << "Usage: transpose_mapping [width height], each in [1,16]\n";
        return 2;
    }

    const int count = width * height;
    std::vector<int> input(static_cast<std::size_t>(count));
    std::vector<int> output(static_cast<std::size_t>(count), -1);
    std::vector<int> owners(static_cast<std::size_t>(count), 0);
    for (int index = 0; index < count; ++index) input[index] = index + 1;

    std::cout << "mapping: input(row,col) -> output(col,row)\n";
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const int input_index = row * width + col;
            const int output_index = col * height + row;
            output[output_index] = input[input_index];
            ++owners[output_index];
            std::cout << "  (" << row << ',' << col << ") in["
                      << input_index << "]=" << input[input_index]
                      << " -> (" << col << ',' << row << ") out["
                      << output_index << "]\n";
        }
    }

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const int input_index = row * width + col;
            const int output_index = col * height + row;
            if (owners[output_index] != 1 ||
                output[output_index] != input[input_index]) {
                std::cerr << "FAIL mapping row=" << row
                          << " col=" << col << '\n';
                return 1;
            }
        }
    }

    std::vector<int> restored(static_cast<std::size_t>(count), -1);
    for (int out_row = 0; out_row < width; ++out_row) {
        for (int out_col = 0; out_col < height; ++out_col) {
            restored[out_col * width + out_row] =
                output[out_row * height + out_col];
        }
    }
    if (restored != input) {
        std::cerr << "FAIL transpose twice did not restore input\n";
        return 1;
    }

    print_matrix("input", input, height, width);
    print_matrix("output", output, width, height);
    std::cout << "PASS every output has one owner; transpose twice restored input\n";
    return 0;
}
