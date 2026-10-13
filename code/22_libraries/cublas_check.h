#ifndef CUDA_BOOK_CUBLAS_CHECK_H
#define CUDA_BOOK_CUBLAS_CHECK_H
#include <cublas_v2.h>
#include <cstdlib>
#include <iostream>
inline const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
        default: return "OTHER";
    }
}
inline void cublas_check(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    std::cerr << "cuBLAS error=" << cublas_status_name(status) << " numeric=" << static_cast<int>(status)
              << "\n expression=" << expression << "\n location=" << file << ':' << line << '\n';
    std::exit(EXIT_FAILURE);
}
#define CUBLAS_CHECK(expression) cublas_check((expression), #expression, __FILE__, __LINE__)
#endif
