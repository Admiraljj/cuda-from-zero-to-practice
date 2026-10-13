#include "core.h"
#include "backend.h"
#include <chrono>
#include <exception>
#include <iostream>
#include <string>
int main(int argc,char** argv) {
    int n=1031;const std::string mode=argc>2?argv[2]:"mixed",backend=argc>3?argv[3]:"cpu";
    if(argc>4 || (argc>1 && !book26::parse(argv[1],1048576,n)) ||
       (mode!="mixed" && mode!="none" && mode!="all") || (backend!="cpu" && backend!="cub")) {
        std::cerr<<"Usage: capstone [n:0..1048576] [mixed|none|all] [cpu|cub]\n";return 2;
    }
    try {
        const auto input=book26::make_input(n,mode),before=input;
        const auto expected=book26::reference(input);
        book26::Result output;std::string detail;int status=0;
        if(n==0) { std::cout<<"PASS empty contract: count=0 sum=0; no backend executed.\n";return 0; }
        const auto start=std::chrono::steady_clock::now();
        if(backend=="cpu") { output=book26::reference(input);detail="CPU-only route"; }
        else status=book26::run_cub(input,output,detail);
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(status==77) { std::cout<<"NOT RUN: "<<detail<<'\n';return 77; }
        if(status!=0) { std::cerr<<"FAIL backend: "<<detail<<'\n';return 1; }
        std::string problem;
        if(input!=before || !book26::verify(output,expected,problem)) { std::cerr<<"FAIL: "<<problem<<'\n';return 1; }
        std::cout<<"PASS backend="<<backend<<" n="<<n<<" mode="<<mode<<" selected="<<output.selected.size()
                 <<" square_sum="<<output.square_sum<<"; "<<detail<<'\n';
        std::cout<<"backend_call_sample_ms="<<ms<<" (one correctness run, not a fair performance comparison)\n";return 0;
    } catch(const std::exception& error) { std::cerr<<"FAIL: "<<error.what()<<'\n';return 1; }
}
