#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    constexpr int m = 2, k = 3, n = 2;
    const std::vector<int> a = {1, 2, 3, 4, 5, 6};
    const std::vector<int> b = {7, 8, 9, 10, 11, 12};
    const std::vector<int> expected = {58, 64, 139, 154};
    std::vector<int> row_c(m * n, 0), column_ct(n * m, 0);

    // Row-major C(row,col) = dot(A row, B column).
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            for (int inner = 0; inner < k; ++inner) {
                row_c[row * n + col] += a[row * k + inner] * b[inner * n + col];
            }
        }
    }

    // The same B bytes represent column-major B^T of shape n x k.
    // The same A bytes represent column-major A^T of shape k x m.
    for (int column = 0; column < m; ++column) {
        for (int row = 0; row < n; ++row) {
            for (int inner = 0; inner < k; ++inner) {
                const int bt_index = inner * n + row;     // ld(B^T) = n
                const int at_index = column * k + inner; // ld(A^T) = k
                const int ct_index = column * n + row;   // ld(C^T) = n
                column_ct[ct_index] += b[bt_index] * a[at_index];
            }
        }
    }

    if (row_c != expected || column_ct != expected) {
        std::cerr << "FAIL layout reinterpretation changed the product.\n";
        return EXIT_FAILURE;
    }
    std::cout << "row-major C shape=(2,2), bytes=[";
    for (int value : row_c) std::cout << value << ' ';
    std::cout << "]\ncolumn-major C^T shape=(2,2), bytes=[";
    for (int value : column_ct) std::cout << value << ' ';
    std::cout << "]\ncuBLAS call: m=N=2, n=M=2, k=K=3; "
                 "first=B ld=N=2; second=A ld=K=3; output=C ld=N=2\n";
    std::cout << "PASS swapped operands and dimensions preserve every row-major output.\n";
    return EXIT_SUCCESS;
}
