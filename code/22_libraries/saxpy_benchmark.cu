#include "../common/cuda_check.h"
#include "../09_timing/benchmark_stats.h"
#include "library_reference.h"
#include "cublas_check.h"
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/version.h>
#include <array>
#include <chrono>
#include <exception>

struct SaxpyOperation {
    __host__ __device__ float operator()(float x,float y) const { return 2.0F*x+y; }
};
__global__ void saxpy_inplace(const float* x,float* y,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) y[i]=2.0F*x[i]+y[i];
}
int main(int argc,char** argv) {
    try {
        int n=1048576,repeats=16,rounds=5;
        if(argc>4 || (argc>1 && !library_example::integer(argv[1],0,16777216,n)) ||
            (argc>2 && !library_example::integer(argv[2],1,1000,repeats)) ||
            (argc>3 && !library_example::integer(argv[3],1,31,rounds)) ||
            1LL*n*repeats>1073741824LL) {
            std::cerr << "Usage: saxpy_benchmark [n:0..16777216] [repeats:1..1000] [rounds:1..31]; n*repeats<=1073741824\n";
            return EXIT_FAILURE;
        }
        if(n==0) { std::cout << "PASS empty input, no timing.\n"; return EXIT_SUCCESS; }
        const std::size_t bytes=static_cast<std::size_t>(n)*sizeof(float);
        std::vector<float> x(n),y0(n),result(n),after(n);
        for(int i=0;i<n;++i) { x[i]=library_example::x_value(i); y0[i]=library_example::y_value(i); }
        float *dx=nullptr,*dy0=nullptr; std::array<float*,4> output{};
        CUDA_CHECK(cudaMalloc(&dx,bytes)); CUDA_CHECK(cudaMalloc(&dy0,bytes));
        CUDA_CHECK(cudaMemcpy(dx,x.data(),bytes,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dy0,y0.data(),bytes,cudaMemcpyHostToDevice));
        for(auto& p:output) CUDA_CHECK(cudaMalloc(&p,bytes));
        cudaStream_t stream{}; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cudaEvent_t start{},stop{}; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        cublasHandle_t handle{}; CUBLAS_CHECK(cublasCreate(&handle)); CUBLAS_CHECK(cublasSetStream(handle,stream));
        CUBLAS_CHECK(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
        const float alpha=2.0F;
        const std::array<const char*,4> name={"custom","thrust-par","thrust-nosync","cublas"};
        auto first_x=thrust::device_pointer_cast(dx);
        auto submit=[&](int method) {
            // ALL methods restore the same y_old inside EVERY measured workflow.
            CUDA_CHECK(cudaMemcpyAsync(output[method],dy0,bytes,cudaMemcpyDeviceToDevice,stream));
            auto first_y=thrust::device_pointer_cast(output[method]);
            if(method==0) {
                saxpy_inplace<<<(n+255)/256,256,0,stream>>>(dx,output[method],n); CUDA_CHECK(cudaGetLastError());
            } else if(method==1) {
                thrust::transform(thrust::cuda::par.on(stream),first_x,first_x+n,first_y,first_y,SaxpyOperation{});
                CUDA_CHECK(cudaGetLastError());
            } else if(method==2) {
                thrust::transform(thrust::cuda::par_nosync.on(stream),first_x,first_x+n,first_y,first_y,SaxpyOperation{});
                CUDA_CHECK(cudaGetLastError());
            } else CUBLAS_CHECK(cublasSaxpy(handle,n,&alpha,dx,1,output[method],1));
        };
        auto validate=[&](int method) {
            CUDA_CHECK(cudaMemcpy(result.data(),output[method],bytes,cudaMemcpyDeviceToHost));
            if(!library_example::saxpy_valid(result,name[method])) std::exit(EXIT_FAILURE);
            CUDA_CHECK(cudaMemcpy(after.data(),dx,bytes,cudaMemcpyDeviceToHost)); if(after!=x) std::exit(EXIT_FAILURE);
            CUDA_CHECK(cudaMemcpy(after.data(),dy0,bytes,cudaMemcpyDeviceToHost)); if(after!=y0) std::exit(EXIT_FAILURE);
        };
        cudaDeviceProp prop{}; int driver=0,runtime=0,library_version=0;
        CUDA_CHECK(cudaGetDeviceProperties(&prop,0)); CUDA_CHECK(cudaDriverGetVersion(&driver)); CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        CUBLAS_CHECK(cublasGetVersion(handle,&library_version));
        std::cout << "GPU=" << prop.name << " driver=" << driver << " runtime=" << runtime
                  << " THRUST_VERSION=" << THRUST_VERSION << " cublas_version=" << library_version
                  << " n=" << n << " repeats=" << repeats << " rounds=" << rounds << " scope=restore-plus-SAXPY\n";
        for(int method=0;method<4;++method) {
            CUDA_CHECK(cudaMemsetAsync(output[method],0xFF,bytes,stream));
            submit(method); CUDA_CHECK(cudaStreamSynchronize(stream)); validate(method);
            for(int warm=0;warm<3;++warm) submit(method);
            CUDA_CHECK(cudaStreamSynchronize(stream)); validate(method);
        }
        std::array<std::vector<double>,4> event_times,host_times;
        for(int round=0;round<rounds;++round) for(int slot=0;slot<4;++slot) {
            const int method=(round+slot)%4;
            CUDA_CHECK(cudaMemsetAsync(output[method],0xFF,bytes,stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto before=std::chrono::steady_clock::now();
            CUDA_CHECK(cudaEventRecord(start,stream));
            for(int repeat=0;repeat<repeats;++repeat) submit(method);
            CUDA_CHECK(cudaEventRecord(stop,stream)); CUDA_CHECK(cudaEventSynchronize(stop));
            const auto after_time=std::chrono::steady_clock::now();
            float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
            const double host_ms=std::chrono::duration<double,std::milli>(after_time-before).count();
            if(!std::isfinite(ms)||ms<=0 || !std::isfinite(host_ms)||host_ms<=0) return EXIT_FAILURE;
            event_times[method].push_back(ms/repeats); host_times[method].push_back(host_ms/repeats);
            validate(method);
            std::cout << "sample round=" << round << " method=" << name[method] << " event_ms_workflow=" << ms/repeats
                      << " host_complete_ms_workflow=" << host_ms/repeats << '\n';
        }
        for(int method=0;method<4;++method) {
            const auto e=summarize(event_times[method]),h=summarize(host_times[method]);
            std::cout << name[method] << " event_min/median/mean/max=" << e.minimum << '/' << e.median << '/' << e.mean << '/' << e.maximum
                      << " host_complete_median=" << h.median << '\n';
        }
        CUBLAS_CHECK(cublasDestroy(handle)); CUDA_CHECK(cudaEventDestroy(stop)); CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaStreamDestroy(stream)); for(auto p:output) CUDA_CHECK(cudaFree(p));
        CUDA_CHECK(cudaFree(dy0)); CUDA_CHECK(cudaFree(dx));
        std::cout << "PASS equal task and reset, four methods, immutable x/y0.\n"; return EXIT_SUCCESS;
    } catch(const std::exception& error) {
        std::cerr << "Thrust/C++ exception: " << error.what() << '\n'; return EXIT_FAILURE;
    }
}
