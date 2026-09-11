#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

int main() {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "This representation experiment requires IEEE-754 binary32 float");
    float values[2] = {};
    std::memset(values, 1, sizeof(values));
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(values);
    bool all_bytes_one = true;
    for (std::size_t i = 0; i < sizeof(values); ++i) {
        if (bytes[i] != 1) all_bytes_one = false;
    }
    std::cout << "after memset(1): bytes_all_one=" << all_bytes_one
              << " value_is_one=" << (values[0] == 1.0f) << '\n';
    if (!all_bytes_one || values[0] == 1.0f || !std::isfinite(values[0])) return 1;
    std::cout << "numeric_value=" << std::scientific << values[0] << '\n';

    std::fill(values, values + 2, 1.0f);
    if (values[0] != 1.0f || values[1] != 1.0f) return 1;
    std::cout << "after fill(1.0f): value_is_one=" << (values[0] == 1.0f) << '\n';
    std::cout << "PASS byte fill versus element fill\n";
    return 0;
}
