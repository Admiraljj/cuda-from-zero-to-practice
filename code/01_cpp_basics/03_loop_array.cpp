#include <iostream>

int main() {
    const int n = 4;
    float values[n] = {10.0f, 20.0f, 30.0f, 40.0f};

    for (int i = 0; i < n; ++i) {
        values[i] = values[i] + 1.0f;
        std::cout << "i=" << i << " value=" << values[i] << '\n';
    }
    return 0;
}
