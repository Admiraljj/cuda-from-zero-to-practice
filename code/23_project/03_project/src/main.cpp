#include "core.h"
#include "backend.h"
#include <exception>
#include <iostream>
#include <limits>

#ifndef BOOK23_BUILD_LABEL
#define BOOK23_BUILD_LABEL "manual"
#endif

int main(int argc,char** argv) {
    try {
        int n=1031,seed=7;
        const std::string backend=argc>2?argv[2]:"cpu";
        if (argc>4 || (argc>1 && !book23::parse_unsigned(argv[1],16777216,n)) ||
            (argc>3 && !book23::parse_unsigned(argv[3],65535,seed)) ||
            (backend!="cpu" && backend!="cuda" && backend!="cublas")) {
            std::cerr << "Usage: book23_app [n:0..16777216] [cpu|cuda|cublas] [seed:0..65535]\n";
            return 2;
        }
        const float alpha=2.0F,beta=1.0F;
        const auto input=book23::make_input(static_cast<std::size_t>(n),seed);
        const auto before=input;
        const auto expected=book23::reference(input,alpha,beta);
        std::vector<float> output(input.size(),std::numeric_limits<float>::quiet_NaN());
        std::string detail;
        std::cout << "build=" << BOOK23_BUILD_LABEL << " backend=" << backend
                  << " n=" << n << " seed=" << seed << " alpha=2 beta=1\n";
        if (n==0) { std::cout << "PASS empty contract; no backend execution.\n"; return 0; }
        int status=0;
        if (backend=="cpu") { output=book23::transform_cpu(input,alpha,beta); detail="host computation"; }
        else if (backend=="cuda") status=book23::transform_cuda(input,output,alpha,beta,detail);
        else status=book23::transform_cublas(input,output,alpha,beta,detail);
        if (status==77) { std::cout << "NOT RUN: " << detail << '\n'; return 77; }
        if (status!=0) { std::cerr << "FAIL backend: " << detail << '\n'; return 1; }
        std::string problem;
        if (!book23::verify(output,expected,problem) || input!=before) {
            std::cerr << "FAIL validation: " << problem << '\n'; return 1;
        }
        std::cout << "PASS complete output and immutable input; " << detail << '\n'; return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL C++ exception: " << error.what() << '\n'; return 1;
    }
}
