#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

float rounded_add(float left, float right) {
    volatile float result = left + right;
    return result;
}

int main() {
    const std::uint32_t maximum = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t wrapped = maximum;
    wrapped += 1u;

    if (wrapped != 0u) {
        std::cerr << "unexpected uint32_t arithmetic model\n";
        return 1;
    }
    std::cout << "uint32_max=" << maximum
              << " after_plus_one=" << wrapped << '\n';

    if (!std::numeric_limits<float>::is_iec559) {
        std::cout << "SKIP floating-order model: IEC 60559 float unavailable\n";
        return 77;
    }

    volatile float positive_large = 1.0e20f;
    volatile float negative_large = -1.0e20f;
    volatile float small = 3.25f;

    const float order_a = rounded_add(
        rounded_add(positive_large, negative_large), small);
    const float order_b = rounded_add(
        rounded_add(positive_large, small), negative_large);

    std::cout << std::setprecision(9)
              << "order_A=(large+negative)+small=" << order_a << '\n'
              << "order_B=(large+small)+negative=" << order_b << '\n';
    if (order_a != 3.25f || order_b != 0.0f || order_a == order_b) {
        std::cerr << "floating-order model did not show the expected distinction\n";
        return 1;
    }

    std::cout << "PASS unsigned wrap and order-dependent float addition modeled\n";
    return 0;
}
