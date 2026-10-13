#include "graph_reference.h"
#include <array>
#include <iomanip>

int main(int argc, char** argv) {
    int repeats = 10;
    if (argc > 2 || (argc > 1 && !graph_example::integer(argv[1], 1, 100000, repeats))) {
        std::cerr << "Usage: dag_model [repeats:1..100000]\n"; return EXIT_FAILURE;
    }
    // Synthetic durations, in arbitrary time units; NOT measured hardware times.
    const std::array<const char*, 5> name = {"upload", "branch-A", "branch-B", "merge", "download"};
    const std::array<int, 5> duration = {2, 3, 5, 1, 2};
    const std::array<std::vector<int>, 5> parents = {
        std::vector<int>{}, std::vector<int>{0}, std::vector<int>{0},
        std::vector<int>{1, 2}, std::vector<int>{3}
    };
    std::array<int, 5> finish{};
    int sum = 0;
    for (int node = 0; node < 5; ++node) {
        int start = 0;
        for (int parent : parents[node]) {
            if (parent >= node) return EXIT_FAILURE;
            start = std::max(start, finish[parent]);
        }
        finish[node] = start + duration[node]; sum += duration[node];
        std::cout << name[node] << " earliest_start=" << start
                  << " earliest_finish=" << finish[node] << '\n';
    }
    if (sum != 13 || finish[4] != 10) return EXIT_FAILURE;
    std::cout << "synthetic_serial_sum=" << sum << " critical_path_lower_bound=" << finish[4]
              << "\nthree_kernel_host_calls=" << 3LL * repeats
              << " graph_host_launch_calls=" << repeats << '\n';
    // Another synthetic model: extra setup S=120; steady-state t_stream=9, t_graph=6.
    constexpr int setup = 120, stream_time = 9, graph_time = 6;
    const int first_win = setup / (stream_time - graph_time) + 1;
    std::cout << "synthetic_setup=" << setup << " first_strict_win_repeats=" << first_win << '\n';
    if (first_win != 41) return EXIT_FAILURE;
    for (int i = 0; i < 17; ++i)
        std::cout << "x=" << graph_example::input_value(i) << " y="
                  << graph_example::expected(graph_example::input_value(i), 2.0F, 1.0F) << '\n';
    std::cout << "PASS synthetic DAG, call counts and break-even arithmetic.\n";
    return EXIT_SUCCESS;
}
