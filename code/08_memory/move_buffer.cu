#include "device_buffer.cuh"
#include <iostream>
#include <utility>

static_assert(!std::is_copy_constructible_v<DeviceBuffer<int>>);
static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer<int>>);
static_assert(!std::is_move_assignable_v<DeviceBuffer<int>>);

int main() {
    try {
        const int input[4] = {9, 8, 7, 6};
        int output[4] = {};
        DeviceBuffer<float> empty(0);
        if (empty.get() != nullptr || empty.size() != 0 || empty.bytes() != 0) return 1;
        DeviceBuffer<int> first(4);
        require_cuda(cudaMemcpy(first.get(), input, first.bytes(), cudaMemcpyHostToDevice), "H2D");
        int* original = first.get();
        DeviceBuffer<int> second(std::move(first));
        if (first.get() != nullptr || first.size() != 0 || second.get() != original ||
            second.size() != 4 || second.bytes() != sizeof(input)) {
            throw std::runtime_error("ownership transfer failed");
        }
        std::cout << "source_size=" << first.size()
                  << " destination_size=" << second.size() << " same_address=1\n";
        require_cuda(cudaMemcpy(output, second.get(), second.bytes(), cudaMemcpyDeviceToHost), "D2H");
        for (int i = 0; i < 4; ++i) {
            std::cout << "i=" << i << " value=" << output[i] << '\n';
            if (output[i] != input[i]) throw std::runtime_error("copy-back mismatch");
        }
        require_cuda(second.close(), "release moved buffer");
        require_cuda(second.close(), "close empty owner");
        std::cout << "PASS move ownership; no kernel launched\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
