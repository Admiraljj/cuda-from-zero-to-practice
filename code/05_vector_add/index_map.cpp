#include <iostream>

int main() {
    const int n = 10;
    const int threads = 4;
    const int blocks = n / threads + (n % threads != 0);

    for (int block = 0; block < blocks; ++block) {
        for (int tid = 0; tid < threads; ++tid) {
            const int i = block * threads + tid;
            std::cout << "block=" << block << " tid=" << tid
                      << " i=" << i;
            if (i < n) {
                std::cout << " work\n";
            } else {
                std::cout << " skip\n";
            }
        }
    }
    std::cout << "CPU model: n=" << n << " blocks=" << blocks
              << " slots=" << blocks * threads << '\n';
}
