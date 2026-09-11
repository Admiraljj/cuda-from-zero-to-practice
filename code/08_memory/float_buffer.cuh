#ifndef CUDA_BOOK_FLOAT_BUFFER_CUH
#define CUDA_BOOK_FLOAT_BUFFER_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

inline void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Single-device teaching owner: do not change/reset the current device while it owns memory.
class FloatBuffer {
public:
    explicit FloatBuffer(std::size_t count) : count_(count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::length_error("FloatBuffer byte count overflow");
        }
        if (count != 0) require_cuda(cudaMalloc(&ptr_, count * sizeof(float)), "allocate buffer");
    }

    ~FloatBuffer() noexcept {
        const cudaError_t status = close();
        if (status != cudaSuccess) {
            std::fprintf(stderr, "buffer cleanup failed: %s\n", cudaGetErrorString(status));
        }
    }

    FloatBuffer(const FloatBuffer&) = delete;
    FloatBuffer& operator=(const FloatBuffer&) = delete;
    FloatBuffer(FloatBuffer&&) = delete;
    FloatBuffer& operator=(FloatBuffer&&) = delete;

    float* get() noexcept { return ptr_; }
    const float* get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(float); }

    // One release attempt. On error, report it and end the operation; do not retry blindly.
    cudaError_t close() noexcept {
        float* old = ptr_;
        ptr_ = nullptr;
        count_ = 0;
        return old == nullptr ? cudaSuccess : cudaFree(old);
    }

private:
    float* ptr_ = nullptr;
    std::size_t count_ = 0;
};

#endif
