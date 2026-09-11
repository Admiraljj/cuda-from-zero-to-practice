#ifndef CUDA_BOOK_DEVICE_BUFFER_CUH
#define CUDA_BOOK_DEVICE_BUFFER_CUH

#include "float_buffer.cuh"
#include <type_traits>

// Deliberately restricted to the two scalar types used by this lesson.
template<class T>
class DeviceBuffer {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, int>,
                  "This teaching buffer supports only float and int");
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::length_error("DeviceBuffer byte count overflow");
        }
        if (count != 0) require_cuda(cudaMalloc(&ptr_, count * sizeof(T)), "allocate typed buffer");
    }
    ~DeviceBuffer() noexcept {
        const cudaError_t status = close();
        if (status != cudaSuccess) {
            std::fprintf(stderr, "typed buffer cleanup failed: %s\n", cudaGetErrorString(status));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }

    cudaError_t close() noexcept {
        T* old = ptr_;
        ptr_ = nullptr;
        count_ = 0;
        return old == nullptr ? cudaSuccess : cudaFree(old);
    }
private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

#endif
