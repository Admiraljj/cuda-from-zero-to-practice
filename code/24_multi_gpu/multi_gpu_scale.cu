#include "multi_gpu_reference.h"
#include "cuda_helpers.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <vector>

__global__ void scale(const float* input,float* output,std::size_t n) {
    const std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) output[i]=3.0F*input[i];
}
struct Shard {
    int device=0;book24::Range range{};
    float* input=nullptr;float* output=nullptr;
    cudaStream_t stream{};cudaEvent_t begin{},end{};
};
bool release(Shard& shard) {
    if(!book24::check(cudaSetDevice(shard.device),shard.device,"select for cleanup")) return false;
    bool ok=true;
    if(shard.stream) ok=book24::check(cudaStreamSynchronize(shard.stream),shard.device,"cleanup wait")&&ok;
    if(shard.end) ok=book24::check(cudaEventDestroy(shard.end),shard.device,"destroy end")&&ok;
    if(shard.begin) ok=book24::check(cudaEventDestroy(shard.begin),shard.device,"destroy begin")&&ok;
    if(shard.output) ok=book24::check(cudaFree(shard.output),shard.device,"free output")&&ok;
    if(shard.input) ok=book24::check(cudaFree(shard.input),shard.device,"free input")&&ok;
    if(shard.stream) ok=book24::check(cudaStreamDestroy(shard.stream),shard.device,"destroy stream")&&ok;
    return ok;
}
int main(int argc,char** argv) {
    int n=1031,workers=2;
    if(argc>3 || (argc>1 && !book24::parse(argv[1],16777216,n)) ||
       (argc>2 && (!book24::parse(argv[2],2,workers) || workers<1))) {
        std::cerr<<"Usage: multi_gpu_scale [n:0..16777216] [workers:1|2]\n";return 2;
    }
    if(n==0) { std::cout<<"PASS empty contract; no GPU executed.\n";return 0; }
    const int probe=book24::require_devices(workers);if(probe) return probe;
    const auto source=book24::input(static_cast<std::size_t>(n));
    const std::size_t bytes=source.size()*sizeof(float);
    float *host_input=nullptr,*host_output=nullptr,*host_after=nullptr;
    std::vector<Shard> shards(static_cast<std::size_t>(workers));
    for(int d=0;d<workers;++d) { shards[d].device=d;shards[d].range=book24::partition(source.size(),workers,d); }
    bool ok=false;
    do {
        if(!book24::check(cudaHostAlloc(&host_input,bytes,cudaHostAllocPortable),-1,"portable input") ||
           !book24::check(cudaHostAlloc(&host_output,bytes,cudaHostAllocPortable),-1,"portable output") ||
           !book24::check(cudaHostAlloc(&host_after,bytes,cudaHostAllocPortable),-1,"portable input snapshot")) break;
        std::copy(source.begin(),source.end(),host_input);
        std::fill_n(host_output,source.size(),std::numeric_limits<float>::quiet_NaN());
        bool setup=true;
        for(auto& shard:shards) {
            if(!shard.range.count) continue;
            const int d=shard.device;const auto amount=shard.range.count*sizeof(float);
            if(!book24::check(cudaSetDevice(d),d,"select for setup") ||
               !book24::check(cudaStreamCreateWithFlags(&shard.stream,cudaStreamNonBlocking),d,"stream") ||
               !book24::check(cudaEventCreate(&shard.begin),d,"begin event") ||
               !book24::check(cudaEventCreate(&shard.end),d,"end event") ||
               !book24::check(cudaMalloc(&shard.input,amount),d,"input") ||
               !book24::check(cudaMalloc(&shard.output,amount),d,"output")) { setup=false;break; }
        }
        if(!setup) break;
        const auto start=std::chrono::steady_clock::now();
        bool submitted=true;
        for(auto& shard:shards) {
            if(!shard.range.count) continue;
            const int d=shard.device;const auto count=shard.range.count;
            const auto amount=count*sizeof(float),offset=shard.range.offset;
            if(!book24::check(cudaSetDevice(d),d,"select for submit") ||
               !book24::check(cudaMemcpyAsync(shard.input,host_input+offset,amount,cudaMemcpyHostToDevice,shard.stream),d,"H2D") ||
               !book24::check(cudaMemcpyAsync(shard.output,host_output+offset,amount,cudaMemcpyHostToDevice,shard.stream),d,"output sentinel") ||
               !book24::check(cudaEventRecord(shard.begin,shard.stream),d,"record begin")) { submitted=false;break; }
            scale<<<static_cast<unsigned>((count+255)/256),256,0,shard.stream>>>(shard.input,shard.output,count);
            if(!book24::check(cudaGetLastError(),d,"scale launch") ||
               !book24::check(cudaEventRecord(shard.end,shard.stream),d,"record end") ||
               !book24::check(cudaMemcpyAsync(host_output+offset,shard.output,amount,cudaMemcpyDeviceToHost,shard.stream),d,"D2H") ||
               !book24::check(cudaMemcpyAsync(host_after+offset,shard.input,amount,cudaMemcpyDeviceToHost,shard.stream),d,"input snapshot")) { submitted=false;break; }
        }
        if(!submitted) break;
        bool completed=true;
        for(auto& shard:shards) {
            if(!shard.range.count) continue;
            if(!book24::check(cudaSetDevice(shard.device),shard.device,"select for wait") ||
               !book24::check(cudaStreamSynchronize(shard.stream),shard.device,"complete shard")) { completed=false;break; }
        }
        if(!completed) break;
        const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        bool reports=true;
        for(auto& shard:shards) {
            if(!shard.range.count) continue;
            float ms=0;
            if(!book24::check(cudaSetDevice(shard.device),shard.device,"select for timing") ||
               !book24::check(cudaEventElapsedTime(&ms,shard.begin,shard.end),shard.device,"same-device event span")) { reports=false;break; }
            std::cout<<"device="<<shard.device<<" offset="<<shard.range.offset<<" count="<<shard.range.count<<" kernel_span_ms="<<ms<<'\n';
        }
        if(!reports) break;
        ok=book24::verify(host_output,source,3,0) && std::equal(source.begin(),source.end(),host_after) &&
           std::equal(source.begin(),source.end(),host_input);
        if(!ok) std::cerr<<"FAIL full output or input immutability\n";
        std::cout<<"host_roundtrip_sample_ms="<<wall_ms<<" (includes sentinel/snapshot; not a benchmark ranking)\n";
    } while(false);
    for(auto& shard:shards) ok=release(shard)&&ok;
    if(host_after) ok=book24::check(cudaFreeHost(host_after),-1,"free snapshot")&&ok;
    if(host_output) ok=book24::check(cudaFreeHost(host_output),-1,"free host output")&&ok;
    if(host_input) ok=book24::check(cudaFreeHost(host_input),-1,"free host input")&&ok;
    if(!ok) return 1;
    std::cout<<"PASS all shards, full CPU oracle, immutable input.\n";return 0;
}
