#include <cstddef>
#include <iostream>

int main() {
    const int n = 8;
    float factor = 2.0f;
    double reference = 1.0 / 3.0;
    bool ready = true;
    std::size_t elements = 8;
    std::size_t bytes = elements * sizeof(float);

    std::cout << "n=" << n << '\n';
    std::cout << "factor=" << factor << '\n';
    std::cout << "reference=" << reference << '\n';
    std::cout << "ready=" << ready << '\n';
    std::cout << "integer division=" << 5 / 2 << '\n';
    std::cout << "floating division=" << 5.0f / 2.0f << '\n';
    std::cout << "elements=" << elements << '\n';
    std::cout << "float bytes=" << sizeof(float) << '\n';
    std::cout << "array bytes=" << bytes << '\n';
    return 0;
}
