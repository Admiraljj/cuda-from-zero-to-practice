#include "../common/cuda_check.h"
#include "cublas_check.h"
#include "library_reference.h"

int main(int argc,char** argv) {
    int n=1031,incx=2,incy=3;
    if (argc>4 || (argc>1 && !library_example::integer(argv[1],0,1048576,n)) ||
        (argc>2 && !library_example::integer(argv[2],1,8,incx)) ||
        (argc>3 && !library_example::integer(argv[3],1,8,incy))) {
        std::cerr << "Usage: cublas_saxpy [n:0..1048576] [incx:1..8] [incy:1..8]\n"; return EXIT_FAILURE;
    }
    if (n==0) { std::cout << "PASS empty vector; no cuBLAS call.\n"; return EXIT_SUCCESS; }
    const std::size_t nx=static_cast<std::size_t>(n-1)*incx+1, ny=static_cast<std::size_t>(n-1)*incy+1;
    const float guard=-777.0F,alpha=2.0F;
    std::vector<float> x(nx,guard),y0(ny,guard),expected(ny,guard),output(ny),after(nx);
    for(int i=0;i<n;++i) {
        x[static_cast<std::size_t>(i)*incx]=library_example::x_value(i);
        y0[static_cast<std::size_t>(i)*incy]=library_example::y_value(i);
        expected[static_cast<std::size_t>(i)*incy]=2.0F*library_example::x_value(i)+library_example::y_value(i);
    }
    float *dx=nullptr,*dy=nullptr,*d_alpha=nullptr;
    CUDA_CHECK(cudaMalloc(&dx,nx*sizeof(float))); CUDA_CHECK(cudaMalloc(&dy,ny*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_alpha,sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx,x.data(),nx*sizeof(float),cudaMemcpyHostToDevice));
    cudaStream_t stream{}; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cublasHandle_t handle{}; CUBLAS_CHECK(cublasCreate(&handle)); CUBLAS_CHECK(cublasSetStream(handle,stream));
    CUDA_CHECK(cudaMemcpyAsync(d_alpha,&alpha,sizeof(float),cudaMemcpyHostToDevice,stream));
    for(int mode=0;mode<2;++mode) {
        CUBLAS_CHECK(cublasSetPointerMode(handle,mode==0?CUBLAS_POINTER_MODE_HOST:CUBLAS_POINTER_MODE_DEVICE));
        CUDA_CHECK(cudaMemcpyAsync(dy,y0.data(),ny*sizeof(float),cudaMemcpyHostToDevice,stream));
        CUBLAS_CHECK(cublasSaxpy(handle,n,mode==0?&alpha:d_alpha,dx,incx,dy,incy));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(output.data(),dy,ny*sizeof(float),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if(output!=expected) return EXIT_FAILURE; // includes all finite guard positions.
        std::cout << "verified_pointer_mode=" << (mode==0?"host":"device") << '\n';
    }
    CUDA_CHECK(cudaMemcpy(after.data(),dx,nx*sizeof(float),cudaMemcpyDeviceToHost));
    if(after!=x) return EXIT_FAILURE;
    CUBLAS_CHECK(cublasDestroy(handle)); CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_alpha)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dx));
    std::cout << "PASS strided SAXPY, holes unchanged, x unchanged, both scalar modes.\n"; return EXIT_SUCCESS;
}
