#include <array>
#include <iostream>

int main() {
    int ordinary_counter = 0;
    const int old_a = ordinary_counter;
    const int old_b = ordinary_counter;
    const int new_a = old_a + 1;
    const int new_b = old_b + 1;
    ordinary_counter = new_a;
    ordinary_counter = new_b;

    if (ordinary_counter != 1) return 1;
    std::cout << "ordinary_reads=" << old_a << ',' << old_b
              << " ordinary_final=" << ordinary_counter
              << " logical_updates=2 lost=1\n";

    int atomic_counter_model = 0;
    std::array<int, 2> returned_old{};
    for (int operation = 0; operation < 2; ++operation) {
        returned_old[operation] = atomic_counter_model;
        atomic_counter_model += 1;
    }

    if (returned_old != std::array<int, 2>{0, 1} ||
        atomic_counter_model != 2) return 1;
    std::cout << "atomic_old_values=" << returned_old[0] << ','
              << returned_old[1] << " atomic_final="
              << atomic_counter_model << '\n';
    std::cout << "PASS deterministic interleaving model; no CPU/GPU race executed\n";
    return 0;
}
