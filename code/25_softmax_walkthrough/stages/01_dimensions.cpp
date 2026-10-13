#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    const int rows = 257;
    const int cols = 1003;
    const std::size_t elements = static_cast<std::size_t>(rows) * cols;
    const std::size_t bytes = elements * sizeof(float);
    std::vector<float> h_input(elements);
    std::vector<float> h_output(elements);
    std::cout << "elements=" << elements << ", bytes=" << bytes
              << ", first=" << h_input[0] << '\n';
    return 0;
}
