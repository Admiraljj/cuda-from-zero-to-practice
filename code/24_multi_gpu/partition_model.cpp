#include "multi_gpu_reference.h"
#include <iostream>

int main(int argc,char** argv) {
    int n=7,parts=2;
    if(argc>3 || (argc>1 && !book24::parse(argv[1],16777216,n)) ||
       (argc>2 && (!book24::parse(argv[2],8,parts) || parts<1))) {
        std::cerr << "Usage: partition_model [n:0..16777216] [parts:1..8]\n";return 2;
    }
    const auto source=book24::input(static_cast<std::size_t>(n));
    std::vector<float> result(source.size());
    std::vector<int> visits(source.size(),0);
    for(int d=0;d<parts;++d) {
        const auto range=book24::partition(source.size(),parts,d);
        std::cout << "part=" << d << " offset=" << range.offset << " count=" << range.count << '\n';
        for(std::size_t local=0;local<range.count;++local) {
            const auto global=range.offset+local;
            ++visits[global];result[global]=3.0F*source[global];
        }
    }
    for(int count:visits) if(count!=1) return 1;
    if(!book24::verify(result.data(),source,3,0)) return 1;
    std::cout << "PASS disjoint coverage and CPU oracle; no GPU executed.\n";return 0;
}
