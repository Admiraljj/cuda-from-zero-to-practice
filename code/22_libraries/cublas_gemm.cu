#include "../common/cuda_check.h"
#include "cublas_check.h"
#include "library_reference.h"

int main(int argc,char** argv) {
    int m=3,k=5,n=2,pad=3;
    if(argc>5 || (argc>1 && !library_example::integer(argv[1],1,256,m)) ||
       (argc>2 && !library_example::integer(argv[2],1,256,k)) ||
       (argc>3 && !library_example::integer(argv[3],1,256,n)) ||
       (argc>4 && !library_example::integer(argv[4],0,8,pad))) {
        std::cerr << "Usage: cublas_gemm [M:1..256] [K:1..256] [N:1..256] [padding:0..8]\n"; return EXIT_FAILURE;
    }
    const int lda=k+pad,ldb=n+pad,ldc=n+pad;
    const float guard=-777.0F,alpha=2.0F,beta=0.5F;
    std::vector<float> a(static_cast<std::size_t>(m)*lda,guard),b(static_cast<std::size_t>(k)*ldb,guard);
    std::vector<float> c0(static_cast<std::size_t>(m)*ldc,guard),out(c0.size()),after_a(a.size()),after_b(b.size());
    for(int row=0;row<m;++row) for(int q=0;q<k;++q) a[row*lda+q]=static_cast<float>((row*7+q*3)%9-4)/16.0F;
    for(int q=0;q<k;++q) for(int col=0;col<n;++col) b[q*ldb+col]=static_cast<float>((q*5+col*2)%7-3)/16.0F;
    for(int row=0;row<m;++row) for(int col=0;col<n;++col) c0[row*ldc+col]=static_cast<float>((row+col)%5-2)/8.0F;
    std::vector<double> expected(c0.begin(),c0.end());
    for(int row=0;row<m;++row) for(int col=0;col<n;++col) {
        double dot=0;
        for(int q=0;q<k;++q) dot+=static_cast<double>(a[row*lda+q])*b[q*ldb+col];
        expected[row*ldc+col]=alpha*dot+beta*c0[row*ldc+col];
    }
    const std::size_t ab=a.size()*sizeof(float),bb=b.size()*sizeof(float),cb=c0.size()*sizeof(float);
    float *da=nullptr,*db=nullptr,*dc=nullptr,*scalars=nullptr;
    CUDA_CHECK(cudaMalloc(&da,ab)); CUDA_CHECK(cudaMalloc(&db,bb)); CUDA_CHECK(cudaMalloc(&dc,cb));
    CUDA_CHECK(cudaMalloc(&scalars,2*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da,a.data(),ab,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,b.data(),bb,cudaMemcpyHostToDevice));
    const float host_scalars[]={alpha,beta};
    CUDA_CHECK(cudaMemcpy(scalars,host_scalars,sizeof(host_scalars),cudaMemcpyHostToDevice));
    cudaStream_t stream{}; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cublasHandle_t handle{}; CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle,stream)); CUBLAS_CHECK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
    auto release=[&]() {
        CUBLAS_CHECK(cublasDestroy(handle)); CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(scalars)); CUDA_CHECK(cudaFree(dc)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(da));
    };
    std::cout << "row_major M/K/N=" << m << '/' << k << '/' << n
              << " strides_A/B/C=" << lda << '/' << ldb << '/' << ldc << " alpha=2 beta=0.5\n";
    for(int mode=0;mode<2;++mode) {
        CUBLAS_CHECK(cublasSetPointerMode(handle,mode==0?CUBLAS_POINTER_MODE_HOST:CUBLAS_POINTER_MODE_DEVICE));
        CUDA_CHECK(cudaMemcpyAsync(dc,c0.data(),cb,cudaMemcpyHostToDevice,stream)); // reset C_old, including padding.
        const cublasStatus_t status=cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,
            n,m,k,mode==0?&alpha:scalars,db,CUDA_R_32F,ldb,da,CUDA_R_32F,lda,
            mode==0?&beta:scalars+1,dc,CUDA_R_32F,ldc,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT);
        if(status==CUBLAS_STATUS_NOT_SUPPORTED) {
            CUDA_CHECK(cudaStreamSynchronize(stream)); release();
            std::cout << "NOT RUN: strict FP32 GemmEx unsupported.\n"; return 77;
        }
        CUBLAS_CHECK(status); CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(out.data(),dc,cb,cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(std::size_t i=0;i<out.size();++i)
            if(!std::isfinite(out[i]) || static_cast<double>(out[i])!=expected[i]) {
                std::cerr << "FAIL padded C index=" << i << '\n'; return EXIT_FAILURE;
            }
        std::cout << "verified_pointer_mode=" << (mode==0?"host":"device") << '\n';
    }
    CUDA_CHECK(cudaMemcpy(after_a.data(),da,ab,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(after_b.data(),db,bb,cudaMemcpyDeviceToHost));
    if(after_a!=a || after_b!=b) return EXIT_FAILURE;
    release();
    std::cout << "PASS padded non-square GEMM, nonzero beta, guards and A/B unchanged.\n"; return EXIT_SUCCESS;
}
