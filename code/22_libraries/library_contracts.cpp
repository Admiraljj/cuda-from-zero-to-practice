#include "library_reference.h"

int main() {
    const std::vector<float> x = {-2.0F, -0.5F, 0.0F, 1.0F, 2.0F};
    const std::vector<float> y = {1.0F, 2.0F, -1.0F, 0.5F, -3.0F};
    const std::vector<float> expected = {-3.0F, 1.0F, -1.0F, 2.5F, 1.0F};
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float output = 2.0F * x[i] + y[i];
        if (output != expected[i]) return EXIT_FAILURE;
        std::cout << "saxpy i=" << i << " x=" << x[i] << " y_old=" << y[i] << " output=" << output << '\n';
    }
    const std::vector<long long> input = {-2, 0, 3, -1, 4};
    const std::vector<long long> expected_scan = {0, -2, -2, 1, 0}, expected_selected = {3, 4};
    std::vector<long long> scan, selected;
    long long sum = 0, squares = 0;
    for (long long value : input) {
        scan.push_back(sum); sum += value; squares += value * value;
        if (value > 0) selected.push_back(value);
    }
    if (sum != 4 || squares != 30 || scan != expected_scan || selected != expected_selected) return EXIT_FAILURE;
    std::cout << "integer sum=" << sum << " squares=" << squares << " selected_count=" << selected.size() << '\n';
    constexpr int m = 2, k = 3, n = 2, lda = 4, ldb = 3, ldc = 3;
    const float guard = -777.0F;
    const std::vector<float> a = {1,2,3,guard, 4,5,6,guard};
    const std::vector<float> b = {1,2,guard, 3,4,guard, 5,6,guard};
    const std::vector<float> c0 = {1,-1,guard, 2,-2,guard};
    const std::vector<float> want = {44.5F,55.5F,guard, 99,127,guard};
    std::vector<float> c = c0;
    for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col) {
        float dot = 0;
        for (int q = 0; q < k; ++q) dot += a[row*lda+q] * b[q*ldb+col];
        c[row*ldc+col] = 2.0F * dot + 0.5F * c0[row*ldc+col];
    }
    if (c != want) return EXIT_FAILURE;
    std::cout << "padded row strides A/B/C=" << lda << '/' << ldb << '/' << ldc
              << " cublas m/n/k=" << n << '/' << m << '/' << k << '\n';
    std::cout << "PASS CPU contracts: SAXPY, sum/scan/select/squares, padded alpha-beta GEMM.\n";
    return EXIT_SUCCESS;
}
