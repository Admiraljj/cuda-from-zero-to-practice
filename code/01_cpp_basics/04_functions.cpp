#include <iostream>

int square(int x) {
    return x * x;
}

void change_copy(int x) {
    x = 99;
    std::cout << "inside copy=" << x << '\n';
}

void change_reference(int& x) {
    x = 99;
}

int main() {
    int value = 3;
    std::cout << "square=" << square(value) << '\n';
    change_copy(value);
    std::cout << "after copy=" << value << '\n';
    change_reference(value);
    std::cout << "after reference=" << value << '\n';
    return 0;
}
