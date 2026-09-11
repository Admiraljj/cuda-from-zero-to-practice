#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    const std::size_t n = 4;
    std::vector<float> values(n, 2.0f);
    std::vector<float> copied = values;
    float* p = values.data();

    p[0] = 9.0f;
    std::cout << "values[0]=" << values[0] << '\n';
    std::cout << "copied[0]=" << copied[0] << '\n';

    for (float value : values) {
        value = 7.0f;
        (void)value;
    }
    std::cout << "after value loop=" << values[0] << '\n';

    for (float& value : values) {
        value = 7.0f;
    }
    std::cout << "after reference loop=" << values[0] << '\n';
    std::cout << "size=" << values.size() << '\n';
    return 0;
}
