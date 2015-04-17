#ifndef CXXLAPACK_AUXILIARY_CUDA_H
#define CXXLAPACK_AUXILIARY_CUDA_H 1

#if defined(HAVE_CUSOLVER)

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

  static void
  release();

 private:
  static cusolverDnHandle_t handle_;
};

// XXX
cusolverDnHandle_t          CusolverEnv::handle_   = 0;


void
checkStatus(cusolverStatus_t status);

} // end cxxlapack

#endif // HAVE_CUSOLVER

#endif // CXXLAPACK_AUXILIARY_CUDA_H
