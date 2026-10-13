#include "backend.h"
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/version.cuh>
#include <algorithm>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Positive { __host__ __device__ bool operator()(int value) const { return value>0; } };
struct Square { __host__ __device__ book26::Sum operator()(int value) const { return static_cast<book26::Sum>(value)*value; } };
void check(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
}
namespace book26 {
int run_cub(const std::vector<int>& input,Result& output,std::string& detail) {
    detail.clear();output=Result{};
    if(input.empty()) { detail="empty contract; no CUB execution";return 0; }
    if(input.size()>1048576U) { detail="count budget";return 1; }
    for(int value:input) if(value< -1000 || value>1000) { detail="value budget";return 1; }
    int devices=0;const auto probe=cudaGetDeviceCount(&devices);
    if(probe==cudaErrorNoDevice || probe==cudaErrorInsufficientDriver || (probe==cudaSuccess && devices==0)) {
        detail="no usable CUDA device/driver";return 77;
    }
    if(probe!=cudaSuccess) { detail=cudaGetErrorString(probe);return 1; }
    int *d_input=nullptr,*selected_storage=nullptr,*d_count=nullptr;
    Sum* d_sum=nullptr;void* scratch=nullptr;cudaStream_t stream{};
    std::size_t capacity=0;int count=-1,status=1;
    constexpr int guard=-2000000000;
    std::vector<int> selected_host(input.size()+2,guard),after(input.size());
    Sum sum_sentinel=-1; // Host buffers/sentinels outlive catch and cleanup wait.
    try {
        check(cudaSetDevice(0),"select GPU0");
        const int n=static_cast<int>(input.size());const auto bytes=input.size()*sizeof(int);
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
        check(cudaMalloc(&d_input,bytes),"input");
        check(cudaMalloc(&selected_storage,selected_host.size()*sizeof(int)),"guarded selected");
        check(cudaMalloc(&d_count,sizeof(int)),"selected count");
        check(cudaMemcpyAsync(d_input,input.data(),bytes,cudaMemcpyHostToDevice,stream),"upload input");
        check(cudaMemcpyAsync(selected_storage,selected_host.data(),selected_host.size()*sizeof(int),cudaMemcpyHostToDevice,stream),"selected sentinel");
        check(cudaMemcpyAsync(d_count,&count,sizeof(int),cudaMemcpyHostToDevice,stream),"count sentinel");
        int* d_selected=selected_storage+1;
        auto ensure=[&](std::size_t required) {
            required=std::max(required,std::size_t{1});
            if(required<=capacity) return;
            if(scratch) { void* old=scratch;scratch=nullptr;capacity=0;check(cudaFree(old),"grow: free completed workspace"); }
            check(cudaMalloc(&scratch,required),"allocate workspace");capacity=required;
        };
        std::size_t required=0;
        check(cub::DeviceSelect::If(nullptr,required,d_input,d_selected,d_count,n,Positive{},stream),"select query");
        ensure(required);std::size_t available=capacity;
        check(cub::DeviceSelect::If(scratch,available,d_input,d_selected,d_count,n,Positive{},stream),"select execute");
        check(cudaGetLastError(),"select launch state");
        check(cudaMemcpyAsync(&count,d_count,sizeof(int),cudaMemcpyDeviceToHost,stream),"count D2H");
        check(cudaStreamSynchronize(stream),"count ready and select workspace no longer in use");
        if(count<0 || count>n) throw std::runtime_error("selected count outside [0,n]");
        if(count>0) {
            check(cudaMalloc(&d_sum,sizeof(Sum)),"sum");
            check(cudaMemcpyAsync(d_sum,&sum_sentinel,sizeof(Sum),cudaMemcpyHostToDevice,stream),"sum sentinel");
            using Iterator=cub::TransformInputIterator<Sum,Square,const int*>;
            Iterator squares(d_selected,Square{});
            required=0;check(cub::DeviceReduce::Sum(nullptr,required,squares,d_sum,count,stream),"reduce actual count query");
            ensure(required);available=capacity;
            check(cub::DeviceReduce::Sum(scratch,available,squares,d_sum,count,stream),"square reduce execute");
            check(cudaGetLastError(),"reduce launch state");
            check(cudaMemcpyAsync(&output.square_sum,d_sum,sizeof(Sum),cudaMemcpyDeviceToHost,stream),"sum D2H");
            check(cudaStreamSynchronize(stream),"sum complete; sentinel lifetime ends");
        }
        check(cudaMemcpyAsync(selected_host.data(),selected_storage,selected_host.size()*sizeof(int),cudaMemcpyDeviceToHost,stream),"selected and guards D2H");
        check(cudaMemcpyAsync(after.data(),d_input,bytes,cudaMemcpyDeviceToHost,stream),"input snapshot D2H");
        check(cudaStreamSynchronize(stream),"complete result transfers");
        if(selected_host.front()!=guard || selected_host.back()!=guard || after!=input)
            throw std::runtime_error("guard or immutable input violated");
        output.selected.assign(selected_host.begin()+1,selected_host.begin()+1+count);
        status=0;
    } catch(const std::exception& error) { detail=error.what(); }
    auto cleanup=[&](cudaError_t code,const char* operation) {
        if(code!=cudaSuccess) { if(detail.empty()) detail=std::string(operation)+": "+cudaGetErrorString(code);status=1; }
    };
    if(stream) cleanup(cudaStreamSynchronize(stream),"cleanup wait");
    if(scratch) cleanup(cudaFree(scratch),"free workspace");
    if(d_sum) cleanup(cudaFree(d_sum),"free sum");
    if(d_count) cleanup(cudaFree(d_count),"free count");
    if(selected_storage) cleanup(cudaFree(selected_storage),"free selected");
    if(d_input) cleanup(cudaFree(d_input),"free input");
    if(stream) cleanup(cudaStreamDestroy(stream),"destroy stream");
    if(status==0) detail="CUB_VERSION="+std::to_string(CUB_VERSION)+" count="+std::to_string(count)+" workspace_capacity="+std::to_string(capacity);
    return status;
}
}
