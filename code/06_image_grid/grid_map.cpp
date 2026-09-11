#include <iostream>

int main() {
    const int width = 5, height = 3;
    const int block_x = 4, block_y = 2;
    const int grid_x = (width + block_x - 1) / block_x;
    const int grid_y = (height + block_y - 1) / block_y;
    int valid = 0;
    int skipped = 0;
    for (int by = 0; by < grid_y; ++by) {
        for (int bx = 0; bx < grid_x; ++bx) {
            for (int ty = 0; ty < block_y; ++ty) {
                for (int tx = 0; tx < block_x; ++tx) {
                    const int col = bx * block_x + tx;
                    const int row = by * block_y + ty;
                    std::cout << "block=(" << bx << ',' << by << ") thread=("
                              << tx << ',' << ty << ") row=" << row << " col=" << col;
                    if (row < height && col < width) {
                        std::cout << " index=" << row * width + col << '\n';
                        ++valid;
                    } else {
                        std::cout << " skip\n";
                        ++skipped;
                    }
                }
            }
        }
    }
    std::cout << "CPU grid model: valid=" << valid << " skipped=" << skipped << '\n';
    return (valid == width * height && skipped == 17) ? 0 : 1;
}
