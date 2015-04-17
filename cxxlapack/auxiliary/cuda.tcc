#ifndef CXXLAPACK_AUXILIARY_CUDA_TCC
#define CXXLAPACK_AUXILIARY_CUDA_TCC 1

#if defined(HAVE_CUSOLVER)

#include <cxxblas/auxiliary/cuda.tcc>

namespace cxxlapack {

void
CusolverEnv::init()
{
  CudaEnv::init();

  // create SOLVER handle
  checkStatus(cusolverDnCreate(&handle_));
}

void
CusolverEnv::release()
{
  // destroy SOLVER handle
  checkStatus(cusolverDnDestroy(handle_));

  CudaEnv::release();
}

cusolverDnHandle_t &
CusolverEnv::handle()
{
    // TODO: Safety checks? Error msgs?

    return handle_;
}


void
checkStatus(cusolverStatus_t status)
{
    if (status==CUSOLVER_STATUS_SUCCESS) {
        return;
    }

    if (status==CUSOLVER_STATUS_NOT_INITIALIZED) {
        std::cerr << "CUSOLVER: Library was not initialized!" << std::endl;
    } else if  (status==CUSOLVER_STATUS_INVALID_VALUE) {
        std::cerr << "CUSOLVER: Parameter had illegal value!" << std::endl;
    } else if  (status==CUSOLVER_STATUS_ALLOC_FAILED) {
        std::cerr << "CUSOLVER: allocation failed!" << std::endl;
    } else if  (status==CUSOLVER_STATUS_ARCH_MISMATCH) {
        std::cerr << "CUSOLVER: Device does not support double precision!" << std::endl;
    } else if  (status==CUSOLVER_STATUS_EXECUTION_FAILED) {
        std::cerr << "CUSOLVER: Failed to launch function of the GPU" << std::endl;
    } else if  (status==CUSOLVER_STATUS_INTERNAL_ERROR) {
        std::cerr << "CUSOLVER: An internal operation failed" << std::endl;
    } else if  (status==CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED) {
        std::cerr << "CUSOLVER: Invalid matrix descriptor" << std::endl;
    } else {
        std::cerr << "CUSOLVER: Unkown error" << std::endl;
    }

    ASSERT(status==CUSOLVER_STATUS_SUCCESS); // false
}


} // end cxxlapack

#endif // HAVE_CUSOLVER

#endif // CXXLAPACK_AUXILIARY_CUDA_TCC
