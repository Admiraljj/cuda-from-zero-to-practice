#include "core.h"
#include <cstdlib>
#include <iostream>
#include <limits>

int main() {
    for (std::size_t n : {0U,1U,7U,255U,256U,257U,1031U}) for (int seed : {0,7,65535}) {
        const auto input=book23::make_input(n,seed),snapshot=input;
        const auto expected=book23::reference(input,2.0F,1.0F);
        const auto output=book23::transform_cpu(input,2.0F,1.0F);
        std::string detail;
        if (!book23::verify(output,expected,detail) || input!=snapshot) return EXIT_FAILURE;
    }
    int value=0;
    for (const char* bad : {"","-1","-0","+1","1x","16777217","999999999999999999999"})
        if (book23::parse_unsigned(bad,16777216,value)) return EXIT_FAILURE;
    if (!book23::parse_unsigned("001",16777216,value) || value!=1) return EXIT_FAILURE;
    const std::vector<double> expected={1};
    std::string problem;
    if (book23::verify({},expected,problem) || problem.empty()) return EXIT_FAILURE;
    if (book23::verify({std::numeric_limits<float>::quiet_NaN()},expected,problem)) return EXIT_FAILURE;
    if (book23::verify({2.0F},expected,problem)) return EXIT_FAILURE;
    std::cout << "PASS CPU core, parser, sizes, finite check and negative checker cases.\n";
    return EXIT_SUCCESS;
}
