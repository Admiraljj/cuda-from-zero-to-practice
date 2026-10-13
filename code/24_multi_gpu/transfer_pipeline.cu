#include "multi_gpu_reference.h"
#include "cuda_helpers.h"
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

__global__ void affine(const float* input,float* output,std::size_t n,float factor,float bias) {
    const std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) output[i]=factor*input[i]+bias;
}
int main(int argc,char** argv) {
    int n=1031;const std::string requested=argc>2?argv[2]:"staged";
    if(argc>3 || (argc>1 && !book24::parse(argv[1],1048576,n)) ||
       (requested!="staged" && requested!="peer" && requested!="auto")) {
        std::cerr<<"Usage: transfer_pipeline [n:0..1048576] [staged|peer|auto]\n";return 2;
    }
    if(n==0) { std::cout<<"PASS empty contract; no transfer executed.\n";return 0; }
    const int probe=book24::require_devices(2);if(probe) return probe;
    int can=0;
    if(!book24::check(cudaDeviceCanAccessPeer(&can,1,0),1,"can GPU1 access GPU0")) return 1;
    if(requested=="peer" && !can) { std::cout<<"NOT RUN: required peer direction 1 -> 0 unavailable.\n";return 77; }
    const bool peer=requested!="staged" && can;
    std::cout<<"requested="<<requested<<" selected="<<(peer?"peer":"staged")<<" can_1_to_0="<<can<<'\n';
    const auto input=book24::input(static_cast<std::size_t>(n));
    const auto bytes=input.size()*sizeof(float);
    std::vector<float> output(input.size(),std::numeric_limits<float>::quiet_NaN());
    std::vector<float> received(input.size()),after(input.size());
    float *d_input=nullptr,*d_source=nullptr,*d_received=nullptr,*d_output=nullptr,*relay=nullptr;
    cudaStream_t source_stream{},destination_stream{};cudaEvent_t ready{};
    bool enabled_here=false,ok=false;
    do {
        if(!book24::check(cudaSetDevice(0),0,"source select") ||
           !book24::check(cudaStreamCreateWithFlags(&source_stream,cudaStreamNonBlocking),0,"source stream") ||
           !book24::check(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming),0,"ready event") ||
           !book24::check(cudaMalloc(&d_input,bytes),0,"source input") ||
           !book24::check(cudaMalloc(&d_source,bytes),0,"source result") ||
           !book24::check(cudaMemcpyAsync(d_input,input.data(),bytes,cudaMemcpyHostToDevice,source_stream),0,"source upload")) break;
        affine<<<static_cast<unsigned>((input.size()+255)/256),256,0,source_stream>>>(d_input,d_source,input.size(),3.0F,0.0F);
        if(!book24::check(cudaGetLastError(),0,"source launch") ||
           !book24::check(cudaEventRecord(ready,source_stream),0,"record source ready")) break;
        if(!book24::check(cudaSetDevice(1),1,"destination select") ||
           !book24::check(cudaStreamCreateWithFlags(&destination_stream,cudaStreamNonBlocking),1,"destination stream") ||
           !book24::check(cudaMalloc(&d_received,bytes),1,"received array") ||
           !book24::check(cudaMalloc(&d_output,bytes),1,"final array")) break;
        if(peer) {
            // Standalone program: enable once, own this enable, disable only if it succeeded.
            if(!book24::check(cudaDeviceEnablePeerAccess(0,0),1,"enable 1 -> 0")) break;
            enabled_here=true;
            if(!book24::check(cudaStreamWaitEvent(destination_stream,ready,0),1,"cross-device producer dependency") ||
               !book24::check(cudaMemcpyPeerAsync(d_received,1,d_source,0,bytes,destination_stream),1,"peer copy 0 to 1")) break;
        } else {
            if(!book24::check(cudaHostAlloc(&relay,bytes,cudaHostAllocPortable),-1,"portable relay") ||
               !book24::check(cudaSetDevice(0),0,"select for relay download") ||
               !book24::check(cudaMemcpyAsync(relay,d_source,bytes,cudaMemcpyDeviceToHost,source_stream),0,"relay D2H") ||
               !book24::check(cudaStreamSynchronize(source_stream),0,"relay ready for host handoff") ||
               !book24::check(cudaSetDevice(1),1,"select for relay upload") ||
               !book24::check(cudaMemcpyAsync(d_received,relay,bytes,cudaMemcpyHostToDevice,destination_stream),1,"relay H2D")) break;
        }
        if(!book24::check(cudaMemcpyAsync(d_output,output.data(),bytes,cudaMemcpyHostToDevice,destination_stream),1,"output sentinel")) break;
        affine<<<static_cast<unsigned>((input.size()+255)/256),256,0,destination_stream>>>(d_received,d_output,input.size(),2.0F,1.0F);
        if(!book24::check(cudaGetLastError(),1,"destination launch") ||
           !book24::check(cudaMemcpyAsync(output.data(),d_output,bytes,cudaMemcpyDeviceToHost,destination_stream),1,"final D2H") ||
           !book24::check(cudaMemcpyAsync(received.data(),d_received,bytes,cudaMemcpyDeviceToHost,destination_stream),1,"received snapshot") ||
           !book24::check(cudaStreamSynchronize(destination_stream),1,"destination complete") ||
           !book24::check(cudaSetDevice(0),0,"select for input snapshot") ||
           !book24::check(cudaMemcpyAsync(after.data(),d_input,bytes,cudaMemcpyDeviceToHost,source_stream),0,"source input snapshot") ||
           !book24::check(cudaStreamSynchronize(source_stream),0,"source complete")) break;
        ok=book24::verify(output.data(),input,6,1) && book24::verify(received.data(),input,3,0) && after==input;
        if(!ok) std::cerr<<"FAIL intermediate/final/input oracle\n";
    } while(false);
    // Complete BOTH devices before freeing shared lifetimes, including partial failure paths.
    const bool selected0=book24::check(cudaSetDevice(0),0,"cleanup source select");
    ok=selected0&&ok;
    if(selected0 && source_stream) ok=book24::check(cudaStreamSynchronize(source_stream),0,"cleanup source wait")&&ok;
    const bool selected1=book24::check(cudaSetDevice(1),1,"cleanup destination select");
    ok=selected1&&ok;
    if(selected1 && destination_stream) ok=book24::check(cudaStreamSynchronize(destination_stream),1,"cleanup destination wait")&&ok;
    if(selected1) {
        if(d_output) ok=book24::check(cudaFree(d_output),1,"free destination output")&&ok;
        if(d_received) ok=book24::check(cudaFree(d_received),1,"free received")&&ok;
        if(destination_stream) ok=book24::check(cudaStreamDestroy(destination_stream),1,"destroy destination stream")&&ok;
        if(enabled_here) ok=book24::check(cudaDeviceDisablePeerAccess(0),1,"disable owned peer enable")&&ok;
    }
    if(book24::check(cudaSetDevice(0),0,"cleanup source reselect")) {
        if(source_stream) ok=book24::check(cudaStreamSynchronize(source_stream),0,"cleanup source rewait")&&ok;
        if(ready) ok=book24::check(cudaEventDestroy(ready),0,"destroy ready")&&ok;
        if(d_source) ok=book24::check(cudaFree(d_source),0,"free source result")&&ok;
        if(d_input) ok=book24::check(cudaFree(d_input),0,"free source input")&&ok;
        if(source_stream) ok=book24::check(cudaStreamDestroy(source_stream),0,"destroy source stream")&&ok;
    } else ok=false;
    if(relay && selected0 && selected1) ok=book24::check(cudaFreeHost(relay),-1,"free relay")&&ok;
    if(!ok) return 1;
    std::cout<<"PASS 3*x on GPU0 -> transfer -> 2*y+1 on GPU1; all elements checked.\n";return 0;
}
