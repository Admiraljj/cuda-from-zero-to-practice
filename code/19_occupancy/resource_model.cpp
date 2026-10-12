#include "../17_convolution/filter_reference.h"
#include <algorithm>
#include <cstdlib>

int main(int argc, char** argv) {
    int threads = 256, registers = 64, shared = 32768;
    if (argc > 4 || (argc > 1 && !filter_example::integer(argv[1], 32, 1024, threads)) ||
        (argc > 2 && !filter_example::integer(argv[2], 1, 255, registers)) ||
        (argc > 3 && !filter_example::integer(argv[3], 0, 98304, shared)) || threads % 32 != 0) {
        std::cerr << "Usage: resource_model [threads:32..1024,multiple-of-32] [registers:1..255] [shared:0..98304]\n";
        return EXIT_FAILURE;
    }
    // A deliberately synthetic SM. Ignores hardware allocation granularity.
    constexpr int sm_threads = 2048, sm_registers = 65536, sm_shared = 98304, sm_blocks = 32;
    const int thread_bound = sm_threads / threads;
    const int register_bound = sm_registers / (threads * registers);
    const int shared_bound = shared == 0 ? sm_blocks : sm_shared / shared;
    const int blocks = std::min({thread_bound, register_bound, shared_bound, sm_blocks});
    const int active_warps = blocks * (threads / 32);
    const double occupancy = active_warps / 64.0;
    std::cout << "SYNTHETIC SM: 2048 threads,65536 registers,98304 shared bytes,32 blocks\n"
              << "bounds threads=" << thread_bound << " registers=" << register_bound
              << " shared=" << shared_bound << " architecture=" << sm_blocks << '\n'
              << "blocks_per_SM=" << blocks << " active_warps=" << active_warps
              << " occupancy=" << occupancy << '\n';
    if (threads == 256 && registers == 64 && shared == 32768 && blocks != 3) return EXIT_FAILURE;
    std::cout << "PASS integer resource model; hardware granularity and GPU timing are not modeled.\n";
    return EXIT_SUCCESS;
}
