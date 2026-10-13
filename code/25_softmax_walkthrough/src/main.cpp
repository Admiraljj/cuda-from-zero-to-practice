#include "cuda_check.h"
#include "softmax.h"
#include "softmax_reference.h"
#include "input_patterns.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main(int argc,char** argv) {
    int rows=257,cols=1003,threads=256,samples=3;
    const std::string pattern=argc>3?argv[3]:"mixed",requested=argc>4?argv[4]:"block";
    if(argc>7 || (argc>1 && !gpu::parse_positive(argv[1],65535,rows)) ||
       (argc>2 && !gpu::parse_positive(argv[2],16384,cols)) ||
       (argc>5 && !gpu::parse_positive(argv[5],512,threads)) ||
       (argc>6 && !gpu::parse_positive(argv[6],100,samples)) || threads<32 || (threads&(threads-1)) ||
       static_cast<std::size_t>(rows)*cols>4194304U ||
       (pattern!="mixed" && pattern!="equal" && pattern!="negative" && pattern!="extreme") ||
       (requested!="serial" && requested!="block" && requested!="warp" && requested!="compare")) {
        std::cerr<<"Usage: softmax [rows:1..65535] [cols:1..16384] [mixed|equal|negative|extreme] "
                 <<"[serial|block|warp|compare] [block_threads:32|64|128|256|512] [samples:1..100]; elements<=4194304\n";return 2;
    }
    if(requested=="warp" && cols>32) { std::cout<<"NOT RUN: warp implementation requires cols<=32.\n";return 77; }
    float *input_storage=nullptr,*output_storage=nullptr;
    cudaStream_t stream{};cudaEvent_t begin{},end{};int exit_code=1;
    std::vector<float> host_input,host_output,after; // Outlive catch and cleanup wait.
    try {
        const std::size_t elements=gpu::checked_elements(rows,cols);
        const auto input=gpu::make_input(rows,cols,pattern);
        const auto reference=gpu::softmax_reference(input.data(),rows,cols);
        int devices=0;const auto probe=cudaGetDeviceCount(&devices);
        if(probe==cudaErrorNoDevice || probe==cudaErrorInsufficientDriver || (probe==cudaSuccess && devices==0)) {
            std::cout<<"NOT RUN: no usable CUDA device/driver.\n";return 77;
        }
        check_cuda(probe,"device count");check_cuda(cudaSetDevice(0),"select device");
        cudaDeviceProp prop{};check_cuda(cudaGetDeviceProperties(&prop,0),"properties");
        int runtime=0,driver=0;check_cuda(cudaRuntimeGetVersion(&runtime),"runtime version");
        check_cuda(cudaDriverGetVersion(&driver),"driver version");
        std::cout<<"GPU="<<prop.name<<" runtime="<<runtime<<" driver_api="<<driver
                 <<" rows="<<rows<<" cols="<<cols<<" pattern="<<pattern<<" block_threads="<<threads<<'\n';
        std::vector<std::string> methods=requested=="compare"?std::vector<std::string>{"serial","block"}:std::vector<std::string>{requested};
        if(requested=="compare" && cols<=32) methods.push_back("warp");
        if(requested=="compare" && cols>32) std::cout<<"warp not included: width outside its contract.\n";
        constexpr float guard=-1234.5F;
        const auto storage_bytes=(elements+2)*sizeof(float);
        host_input.assign(elements+2,guard);host_output.resize(elements+2);after.resize(elements+2);
        std::copy(input.begin(),input.end(),host_input.begin()+1);
        check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream");
        check_cuda(cudaEventCreate(&begin),"create begin");check_cuda(cudaEventCreate(&end),"create end");
        check_cuda(cudaMalloc(&input_storage,storage_bytes),"allocate guarded input");
        check_cuda(cudaMalloc(&output_storage,storage_bytes),"allocate guarded output");
        check_cuda(cudaMemcpyAsync(input_storage,host_input.data(),storage_bytes,cudaMemcpyHostToDevice,stream),"upload input");
        float* d_output=output_storage+1;const float* d_input=input_storage+1;
        auto launch=[&](const std::string& method) {
            if(method=="serial") return gpu::launch_serial_softmax(d_input,d_output,rows,cols,stream);
            if(method=="warp") return gpu::launch_warp_softmax(d_input,d_output,rows,cols,stream);
            return gpu::launch_softmax(d_input,d_output,rows,cols,stream,threads);
        };
        auto reset_output=[&] {
            std::fill(host_output.begin(),host_output.end(),std::numeric_limits<float>::quiet_NaN());
            host_output.front()=guard;host_output.back()=guard;
            check_cuda(cudaMemcpyAsync(output_storage,host_output.data(),storage_bytes,cudaMemcpyHostToDevice,stream),"upload sentinel");
        };
        auto validate=[&](const std::string& method) {
            check_cuda(cudaMemcpyAsync(host_output.data(),output_storage,storage_bytes,cudaMemcpyDeviceToHost,stream),"download output guards");
            check_cuda(cudaMemcpyAsync(after.data(),input_storage,storage_bytes,cudaMemcpyDeviceToHost,stream),"download input snapshot");
            check_cuda(cudaStreamSynchronize(stream),"complete validation transfers");
            const auto report=gpu::validate_softmax(host_output.data()+1,reference,rows,cols);
            if(!report.ok || host_output.front()!=guard || host_output.back()!=guard || after!=host_input)
                throw std::runtime_error(method+": output/guard/input validation failed: "+report.message);
            std::cout<<"PASS "<<method<<" max_abs="<<report.max_abs_error<<" max_rel_diag="<<report.max_rel_error
                     <<" row_sum_error="<<report.max_row_sum_error<<'\n';
        };
        for(const auto& method:methods) {
            reset_output();check_cuda(launch(method),"precheck launch");validate(method);
            for(int warm=0;warm<3;++warm) check_cuda(launch(method),"warm-up launch");
            check_cuda(cudaStreamSynchronize(stream),"warm-up complete");
        }
        constexpr int iterations=10;
        std::vector<std::vector<double>> event_samples(methods.size()),host_samples(methods.size());
        for(int sample=0;sample<samples;++sample) for(std::size_t step=0;step<methods.size();++step) {
            const std::size_t index=(static_cast<std::size_t>(sample)+step)%methods.size();
            const auto& method=methods[index];
            reset_output();check_cuda(cudaStreamSynchronize(stream),"sentinel ready outside timing");
            const auto start=std::chrono::steady_clock::now();
            check_cuda(cudaEventRecord(begin,stream),"record begin");
            for(int repeat=0;repeat<iterations;++repeat) check_cuda(launch(method),"timed launch");
            check_cuda(cudaEventRecord(end,stream),"record end");check_cuda(cudaEventSynchronize(end),"timed completion");
            const double host_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            float event_ms=0;check_cuda(cudaEventElapsedTime(&event_ms,begin,end),"elapsed");
            event_samples[index].push_back(event_ms/iterations);host_samples[index].push_back(host_ms/iterations);
            validate(method); // Transfers and CPU oracle are outside both timed ranges.
        }
        for(std::size_t i=0;i<methods.size();++i) {
            std::sort(event_samples[i].begin(),event_samples[i].end());std::sort(host_samples[i].begin(),host_samples[i].end());
            std::cout<<"method="<<methods[i]<<" samples="<<samples<<" iterations="<<iterations
                     <<" event_median_ms="<<event_samples[i][samples/2]<<" event_min_ms="<<event_samples[i].front()
                     <<" event_max_ms="<<event_samples[i].back()<<" host_complete_median_ms="<<host_samples[i][samples/2]<<'\n';
        }
        std::cout<<"All methods read identical immutable input; no H2D/D2H/allocation in timed span.\n";
        exit_code=0;
    } catch(const std::exception& error) { std::cerr<<"FAIL: "<<error.what()<<'\n'; }
    auto cleanup=[&](cudaError_t status,const char* operation) {
        if(status!=cudaSuccess) { std::cerr<<operation<<": "<<cudaGetErrorString(status)<<'\n';exit_code=1; }
    };
    if(stream) cleanup(cudaStreamSynchronize(stream),"cleanup wait");
    if(end) cleanup(cudaEventDestroy(end),"destroy end");
    if(begin) cleanup(cudaEventDestroy(begin),"destroy begin");
    if(output_storage) cleanup(cudaFree(output_storage),"free output");
    if(input_storage) cleanup(cudaFree(input_storage),"free input");
    if(stream) cleanup(cudaStreamDestroy(stream),"destroy stream");
    return exit_code;
}
