#include <bitset>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
#include <iostream>

int count_bits(std::uint32_t value) {
    int count = 0;
    for (int bit = 0; bit < 32; ++bit) {
        if ((value & (std::uint32_t{1} << bit)) != 0) ++count;
    }
    return count;
}

int main() {
    constexpr int dim_x = 16, dim_y = 3, block_count = 2;
    constexpr int threads_per_block = dim_x * dim_y;
    constexpr int warps_per_block = (threads_per_block + 31) / 32;
    int records = 0;
    for (int block = 0; block < block_count; ++block) {
        for (int y = 0; y < dim_y; ++y) {
            for (int x = 0; x < dim_x; ++x) {
                const int linear = x + dim_x * y;
                const int lane = linear % 32;
                const int warp_in_block = linear / 32;
                const int global_thread = block * threads_per_block + linear;
                const int warp_record = block * warps_per_block + warp_in_block;
                if (linear < 0 || linear >= threads_per_block || lane < 0 || lane >= 32 ||
                    warp_record < 0 || warp_record >= block_count * warps_per_block) return 1;
                if (linear == 0 || linear == 15 || linear == 16 ||
                    linear == 31 || linear == 32 || linear == 47) {
                    std::cout << "block=" << block << " x=" << x << " y=" << y
                              << " linear=" << linear << " lane=" << lane
                              << " warp_in_block=" << warp_in_block
                              << " global_thread=" << global_thread
                              << " warp_record=" << warp_record << '\n';
                }
                ++records;
            }
        }
    }
    std::uint32_t selected = 0;
    for (int lane : {0, 2, 4, 6}) selected |= std::uint32_t{1} << lane;
    if (selected != 0x55u || count_bits(selected) != 4) return 1;
    std::cout << "mask=0x" << std::hex << selected << std::dec
              << " binary=" << std::bitset<32>(selected)
              << " count=" << count_bits(selected) << '\n';
    for (int lane : {0, 2, 4, 6}) {
        const std::uint32_t lower = (std::uint32_t{1} << lane) - 1u;
        const int rank = count_bits(selected & lower);
        if (rank != lane / 2) return 1;
        std::cout << "selected_lane=" << lane << " rank=" << rank << '\n';
    }
    if (records != 96) return 1;
    std::cout << "PASS CPU lane/mask model; no CUDA scheduling observed\n";
    return 0;
}
