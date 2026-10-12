#include "../17_convolution/filter_reference.h"
#include <algorithm>
#include <cstdlib>

int main(int argc, char** argv) {
    int n = 23, chunk = 8, streams = 2;
    if (argc > 4 || (argc > 1 && !filter_example::integer(argv[1], 0, 128, n)) ||
        (argc > 2 && !filter_example::integer(argv[2], 1, 64, chunk)) ||
        (argc > 3 && !filter_example::integer(argv[3], 1, 8, streams))) {
        std::cerr << "Usage: chunk_model [n:0..128] [chunk:1..64] [slots:1..8]\n";
        return EXIT_FAILURE;
    }
    struct Slot { int id = -1, offset = 0, count = 0; bool busy = false; };
    std::vector<Slot> slots(streams);
    std::vector<int> owners(n, 0);
    auto retire = [&](int s) {
        auto& slot = slots[s];
        if (!slot.busy) return;
        std::cout << "wait/consume slot=" << s << " chunk=" << slot.id
                  << " range=[" << slot.offset << ',' << slot.offset + slot.count << ")\n";
        for (int i = 0; i < slot.count; ++i) ++owners[slot.offset + i];
        slot.busy = false;
    };
    for (int offset = 0, id = 0; offset < n; offset += chunk, ++id) {
        const int s = id % streams;
        retire(s);
        slots[s] = {id, offset, std::min(chunk, n - offset), true};
        std::cout << "submit slot=" << s << " chunk=" << id << " count=" << slots[s].count << '\n';
    }
    for (int s = 0; s < streams; ++s) retire(s);
    if (std::any_of(owners.begin(), owners.end(), [](int value) { return value != 1; })) return EXIT_FAILURE;
    std::cout << "PASS disjoint chunks cover each element once; this model assumes completion at retire.\n";
    return EXIT_SUCCESS;
}
