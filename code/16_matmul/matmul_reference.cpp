#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace {

bool parse_dimension(const char* text, int& value) {
    if (text == nullptr || *text == '\0') return false;
    const char* end = text;
    while (*end != '\0') ++end;

    int parsed_value = 0;
    const auto parsed = std::from_chars(text, end, parsed_value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) return false;
    if (parsed_value < 1 || parsed_value > 8) return false;
    value = parsed_value;
    return true;
}

void print_matrix(const std::string& name,
                  const std::vector<std::int64_t>& values,
                  int rows,
                  int cols) {
    std::cout << name << " shape=" << rows << 'x' << cols << '\n';
    for (int row = 0; row < rows; ++row) {
        std::cout << "  ";
        for (int col = 0; col < cols; ++col) {
            const std::size_t index =
                static_cast<std::size_t>(row) * cols + col;
            std::cout << std::setw(7) << values[index];
        }
        std::cout << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    int m = 2;
    int k_size = 3;
    int n = 2;
    if (argc > 4 ||
        (argc >= 2 && !parse_dimension(argv[1], m)) ||
        (argc >= 3 && !parse_dimension(argv[2], k_size)) ||
        (argc >= 4 && !parse_dimension(argv[3], n))) {
        std::cerr << "Usage: " << argv[0]
                  << " [M:1..8] [K:1..8] [N:1..8]\n";
        return EXIT_FAILURE;
    }

    const std::size_t a_count = static_cast<std::size_t>(m) * k_size;
    const std::size_t b_count = static_cast<std::size_t>(k_size) * n;
    const std::size_t c_count = static_cast<std::size_t>(m) * n;
    std::vector<std::int64_t> a(a_count);
    std::vector<std::int64_t> b(b_count);
    std::vector<std::int64_t> c(c_count, 0);
    std::vector<int> owners(c_count, 0);

    for (std::size_t index = 0; index < a_count; ++index) {
        a[index] = static_cast<std::int64_t>(index + 1);
    }
    for (std::size_t index = 0; index < b_count; ++index) {
        b[index] = static_cast<std::int64_t>(a_count + index + 1);
    }

    std::cout << "Contract: A(" << m << 'x' << k_size << ") * B("
              << k_size << 'x' << n << ") -> C("
              << m << 'x' << n << ")\n\n";
    print_matrix("A", a, m, k_size);
    print_matrix("B", b, k_size, n);
    std::cout << '\n';

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            const std::size_t c_index =
                static_cast<std::size_t>(row) * n + col;
            std::int64_t accumulator = 0;
            std::cout << "C[" << row << ',' << col << "] = ";
            for (int inner = 0; inner < k_size; ++inner) {
                const std::size_t a_index =
                    static_cast<std::size_t>(row) * k_size + inner;
                const std::size_t b_index =
                    static_cast<std::size_t>(inner) * n + col;
                const std::int64_t product = a[a_index] * b[b_index];
                accumulator += product;
                if (inner != 0) std::cout << " + ";
                std::cout << "A[" << a_index << "](" << a[a_index]
                          << ")*B[" << b_index << "](" << b[b_index]
                          << ')';
            }
            c[c_index] = accumulator;
            ++owners[c_index];
            std::cout << " = " << accumulator
                      << " -> C linear index " << c_index << '\n';
        }
    }

    for (std::size_t index = 0; index < owners.size(); ++index) {
        if (owners[index] != 1) {
            std::cerr << "FAIL C index " << index
                      << " owner count=" << owners[index] << '\n';
            return EXIT_FAILURE;
        }
    }

    std::cout << '\n';
    print_matrix("C", c, m, n);
    std::cout << "PASS every C element has one owner and a K-term dot product.\n";
    return EXIT_SUCCESS;
}
