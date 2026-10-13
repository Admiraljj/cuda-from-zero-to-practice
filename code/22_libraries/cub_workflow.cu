#include "../common/cuda_check.h"
#include "library_reference.h"
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/version.cuh>

struct Positive {
    __host__ __device__ bool operator()(long long value) const { return value > 0; }
};
int main(int argc, char** argv) {
    int n = 1031, rounds = 3;
    const std::string mode = argc > 2 ? argv[2] : "mixed";
    if (argc > 4 || (argc > 1 && !library_example::integer(argv[1],0,16777216,n)) ||
        (argc > 3 && !library_example::integer(argv[3],1,31,rounds)) ||
        (mode != "mixed" && mode != "none" && mode != "all")) {
        std::cerr << "Usage: cub_workflow [n:0..16777216] [mixed|none|all] [rounds:1..31]\n"; return EXIT_FAILURE;
    }
    if (n == 0) { std::cout << "PASS empty input, application sum=0 count=0; no CUB execution.\n"; return EXIT_SUCCESS; }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(long long);
    std::vector<long long> input(n), scan_expected(n), selected_expected, scan(n), after(n);
    long long expected = 0;
    for (int i = 0; i < n; ++i) {
        input[i] = library_example::integer_value(i,mode); scan_expected[i] = expected; expected += input[i];
        if (input[i] > 0) selected_expected.push_back(input[i]);
    }
    long long *di=nullptr, *d_sum=nullptr, *d_scan=nullptr, *d_selected=nullptr;
    int* d_count=nullptr; void* scratch=nullptr;
    CUDA_CHECK(cudaMalloc(&di,bytes)); CUDA_CHECK(cudaMalloc(&d_sum,sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_scan,bytes)); CUDA_CHECK(cudaMalloc(&d_selected,bytes));
    CUDA_CHECK(cudaMalloc(&d_count,sizeof(int)));
    CUDA_CHECK(cudaMemcpy(di,input.data(),bytes,cudaMemcpyHostToDevice));
    cudaStream_t stream{}; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::size_t reduce_bytes=0, scan_bytes=0, select_bytes=0;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr,reduce_bytes,di,d_sum,n,stream));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,scan_bytes,di,d_scan,n,stream));
    CUDA_CHECK(cub::DeviceSelect::If(nullptr,select_bytes,di,d_selected,d_count,n,Positive{},stream));
    const std::size_t capacity=std::max({reduce_bytes,scan_bytes,select_bytes,std::size_t{1}});
    CUDA_CHECK(cudaMalloc(&scratch,capacity));
    std::cout << "CUB_VERSION=" << CUB_VERSION << " query bytes reduce/scan/select=" << reduce_bytes << '/' << scan_bytes << '/' << select_bytes
              << " shared_sequential_capacity=" << capacity << '\n';
    for (int round=0; round<rounds; ++round) {
        CUDA_CHECK(cudaMemsetAsync(d_sum,0xA5,sizeof(long long),stream));
        CUDA_CHECK(cudaMemsetAsync(d_scan,0xA5,bytes,stream));
        CUDA_CHECK(cudaMemsetAsync(d_selected,0xA5,bytes,stream));
        CUDA_CHECK(cudaMemsetAsync(d_count,0xFF,sizeof(int),stream));
        std::size_t available=capacity;
        CUDA_CHECK(cub::DeviceReduce::Sum(scratch,available,di,d_sum,n,stream));
        available=capacity;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scratch,available,di,d_scan,n,stream));
        available=capacity;
        CUDA_CHECK(cub::DeviceSelect::If(scratch,available,di,d_selected,d_count,n,Positive{},stream));
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
        long long sum=0; int count=-1;
        CUDA_CHECK(cudaMemcpy(&sum,d_sum,sizeof(sum),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&count,d_count,sizeof(count),cudaMemcpyDeviceToHost));
        if (sum != expected || count < 0 || count > n ||
            static_cast<std::size_t>(count) != selected_expected.size()) return EXIT_FAILURE;
        CUDA_CHECK(cudaMemcpy(scan.data(),d_scan,bytes,cudaMemcpyDeviceToHost));
        std::vector<long long> selected(count);
        if (count != 0) CUDA_CHECK(cudaMemcpy(selected.data(),d_selected,count*sizeof(long long),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after.data(),di,bytes,cudaMemcpyDeviceToHost));
        if (scan != scan_expected || selected != selected_expected || after != input) return EXIT_FAILURE;
        std::cout << "round=" << round << " sum=" << sum << " selected_count=" << count << '\n';
    }
    CUDA_CHECK(cudaStreamDestroy(stream)); CUDA_CHECK(cudaFree(scratch)); CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_selected)); CUDA_CHECK(cudaFree(d_scan)); CUDA_CHECK(cudaFree(d_sum)); CUDA_CHECK(cudaFree(di));
    std::cout << "PASS reduce/exclusive-scan/stable-select; scratch reused only in stream order.\n";
    return EXIT_SUCCESS;
}
