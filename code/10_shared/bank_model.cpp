#include <algorithm>
#include <array>
#include <iostream>
#include <set>

int main() {
    constexpr int lanes = 32;
    const int strides[] = {0, 1, 2, 32, 33};
    const int expected[] = {1, 1, 2, 32, 1};
    for (int case_id = 0; case_id < 5; ++case_id) {
        const int stride = strides[case_id];
        std::array<std::set<int>, 32> words_by_bank;
        for (int lane = 0; lane < lanes; ++lane) {
            const int word = stride * lane;
            const int bank = word % 32;
            words_by_bank[bank].insert(word);
        }
        int busiest = 0;
        int used = 0;
        for (const auto& words : words_by_bank) {
            busiest = std::max(busiest, static_cast<int>(words.size()));
            if (!words.empty()) ++used;
        }
        if (busiest != expected[case_id]) return 1;
        std::cout << "stride=" << stride << " used_banks=" << used
                  << " max_distinct_words_per_bank=" << busiest << '\n';
    }
    std::cout << "PASS scalar 32-bit read model; not a GPU transaction measurement\n";
    return 0;
}
