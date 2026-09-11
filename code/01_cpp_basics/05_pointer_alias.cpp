#include <iostream>

void add_one(float* data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] += 1.0f;
    }
}

float first_value(const float* data) {
    return data[0];
}

int main() {
    float values[3] = {10.0f, 20.0f, 30.0f};
    float* p = values;
    float* q = p;

    q[1] = 99.0f;
    std::cout << "p[1]=" << p[1] << '\n';

    q = p + 1;
    *q = 7.0f;
    std::cout << "values[1]=" << values[1] << '\n';

    add_one(p, 3);
    std::cout << "values=" << values[0] << ',' << values[1]
              << ',' << values[2] << '\n';
    std::cout << "first=" << first_value(p) << '\n';
    return 0;
}
