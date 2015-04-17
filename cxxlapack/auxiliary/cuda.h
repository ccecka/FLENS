#ifndef CXXLAPACK_AUXILIARY_CUDA_H
#define CXXLAPACK_AUXILIARY_CUDA_H 1

#if defined(HAVE_CUSOLVER)

#include <vector>

#include <cxxblas/auxiliary/cuda.h>

namespace cxxlapack {

using cxxblas::CudaEnv;
using cxxblas::checkStatus;


class CusolverEnv {
 public:

  static void
  init();

  static cusolverDnHandle_t &
  handle();

  static int*
  devInfo();

  static void
  release();

 private:
  static cusolverDnHandle_t handle_;
  static std::vector<int*>  devinfo_;
};

// XXX
cusolverDnHandle_t          CusolverEnv::handle_   = 0;
std::vector<int*>           CusolverEnv::devinfo_  = std::vector<int*>();

void
checkStatus(cusolverStatus_t status);


/// TODO: cxxlapack interface should use Transpose, etc enums instead of chars

cublasOperation_t
F77Trans2Cusolver(char trans);


} // end cxxlapack

#endif // HAVE_CUSOLVER

#endif // CXXLAPACK_AUXILIARY_CUDA_H
