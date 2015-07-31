#ifndef CXXLAPACK_AUXILIARY_CUDA_H
#define CXXLAPACK_AUXILIARY_CUDA_H 1

#if defined(HAVE_CUSOLVER)

#include <vector>

#include <cxxblas/auxiliary/cuda.h>

namespace cxxlapack {

using cxxblas::CudaEnv;

class CusolverEnv {
  public:
    static void
    init();

    static void
    release();

    static cusolverDnHandle_t &
    handle();

    static int*
    devInfo();

    static void
    setStream(int _streamID);

  //private:
    static cusolverDnHandle_t handle_;
#pragma omp threadprivate(handle_)
    static int streamID_;
#pragma omp threadprivate(streamID_)
    static std::vector<int*>  devinfo_;
#pragma omp threadprivate(devinfo_)
};

// XXX?
cusolverDnHandle_t          CusolverEnv::handle_   =  0;
int                         CusolverEnv::streamID_ = -1;
std::vector<int*>           CusolverEnv::devinfo_  = {};


using cxxblas::checkStatus;

void
checkStatus(cusolverStatus_t status);


/// TODO: cxxlapack interface should use Transpose, etc enums instead of chars

cublasOperation_t
F77Trans2Cusolver(char trans);

cublasFillMode_t
F77UpLo2Cusolver(char upLo);

} // end cxxlapack

#endif // HAVE_CUSOLVER

#endif // CXXLAPACK_AUXILIARY_CUDA_H
