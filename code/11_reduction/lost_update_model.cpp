#include <iostream>

int main() {
    int shared_result = 0;
    const int input0 = 3, input1 = 5;
    const int t0_read = shared_result;
    const int t1_read = shared_result;
    shared_result = t0_read + input0;
    std::cout << "after_T0_write=" << shared_result << '\n';
    shared_result = t1_read + input1;
    std::cout << "after_T1_write=" << shared_result << '\n';
    const int serial_result = input0 + input1;
    if (shared_result != 5 || serial_result != 8) return 1;
    std::cout << "serial_expected=" << serial_result << " lost_update_model=" << shared_result << '\n';
    std::cout << "PASS deterministic interleaving model; no actual data race executed\n";
    return 0;
}
