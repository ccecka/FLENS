/*
 *   Copyright (c) 2013, Klaus Pototzky
 *
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1) Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2) Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *   3) Neither the name of the FLENS development group nor the names of
 *      its contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PLAYGROUND_CXXBLAS_LEVEL1_IAMAX_TCC
#define PLAYGROUND_CXXBLAS_LEVEL1_IAMAX_TCC 1

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/cxxblas.h>

namespace cxxblas {


#ifdef HAVE_CLBLAS

// iamax template for complex vectors, result on host
template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
iamax(IndexType n,
    const flens::device_ptr<const T, flens::StorageType::OpenCL> x, IndexType incX,
    IndexType &result)
{
    flens::CustomAllocator<T, flens::StorageType::OpenCL>                 allocator_real;
    flens::CustomAllocator<IndexType, flens::StorageType::OpenCL>         allocator_int;
    
    // Allocate memory for work and result
    flens::device_ptr<T, flens::StorageType::OpenCL>          dev_work   = allocator_real.allocate(2*n);
    flens::device_ptr<IndexType, flens::StorageType::OpenCL>  dev_result = allocator_int.allocate(1);

    cxxblas::iamax(n, x, incX, dev_result, dev_work);
    
    // Copy result to CPU
    cxxblas::copy(1, dev_result, 1, &result, 1);
    allocator_real.deallocate(dev_work, n);
    allocator_int.deallocate(dev_result, 1);
         
    // Correct Indexing
    result--; // cuBLAS is one-based, cblas is zero-based

}

// siamax, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
iamax(IndexType n,
     const flens::device_ptr<const float, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<IndexType, flens::StorageType::OpenCL> result,
     flens::device_ptr<float, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasiSamax");

    cl_int status = CLBLAS_IMPL(iSamax)(n, 
                                        result.get(), result.getOffset(),
                                        x.get(), x.getOffset(), incX,
                                        work.get(),
                                        1, flens::OpenCLEnv::getQueuePtr(),
                                        0, NULL, NULL);

    flens::checkStatus(status);
}

// diamax, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
iamax(IndexType n,
     const flens::device_ptr<const double, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<IndexType, flens::StorageType::OpenCL> result,
     flens::device_ptr<double, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasiDamax");

    cl_int status = CLBLAS_IMPL(iDamax)(n, 
                                        result.get(), result.getOffset(),
                                        x.get(), x.getOffset(), incX,
                                        work.get(),
                                        1, flens::OpenCLEnv::getQueuePtr(),
                                        0, NULL, NULL);

    flens::checkStatus(status);
}

// ciamax, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
iamax(IndexType n,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<IndexType, flens::StorageType::OpenCL> result,
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasiCamax");

    cl_int status = CLBLAS_IMPL(iCamax)(n, 
                                        result.get(), result.getOffset(),
                                        x.get(), x.getOffset(), incX,
                                        work.get(),
                                        1, flens::OpenCLEnv::getQueuePtr(),
                                        0, NULL, NULL);

    flens::checkStatus(status);
}

// ziamax, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
iamax(IndexType n,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<IndexType, flens::StorageType::OpenCL> result,
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasiZamax");

    cl_int status = CLBLAS_IMPL(iZamax)(n, 
                                        result.get(), result.getOffset(),
                                        x.get(), x.getOffset(), incX,
                                        work.get(),
                                        1, flens::OpenCLEnv::getQueuePtr(),
                                        0, NULL, NULL);

    flens::checkStatus(status);
}

#endif  // HAVE_CLBLAS


#ifdef HAVE_CUBLAS

// siamax
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
iamax(IndexType n,
      const flens::device_ptr<const float, flens::StorageType::CUDA> x, IndexType incX,
      IndexType &result)
{
    CXXBLAS_DEBUG_OUT("cublasIsamax");

    cublasStatus_t status = cublasIsamax(flens::CudaEnv::getHandle(), n, 
                                         x.get(), incX, &result);
    
    flens::checkStatus(status);
    flens::syncStream();
    
    // We have to correct the result -> only syncModed allowed
    ASSERT(flens::CudaEnv::isSyncCopyEnabled());
    
    // Correct Indexing
    result--; // cuBLAS is one-based, cblas is zero-based
  
}


// diamax
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
iamax(IndexType n,
      const flens::device_ptr<const double, flens::StorageType::CUDA> x, IndexType incX,
      IndexType &result)
{
    CXXBLAS_DEBUG_OUT("cublasIdamax");

    cublasStatus_t status = cublasIdamax(flens::CudaEnv::getHandle(), n,  
                                         x.get(), incX, &result);
    
    flens::checkStatus(status);
    flens::syncStream();
    
    // We have to correct the result -> only syncModed allowed
    ASSERT(flens::CudaEnv::isSyncCopyEnabled());

    // Correct Indexing
    result--; // cuBLAS is one-based, cblas is zero-based
}


// ciamax
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
iamax(IndexType n,
      const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> x, IndexType incX,
      IndexType &result)
{
    CXXBLAS_DEBUG_OUT("cublasIcamax");
    
    cublasStatus_t status = cublasIcamax(flens::CudaEnv::getHandle(), n, 
                                         reinterpret_cast<const cuFloatComplex*>(x.get()), incX, 
                                         &result);

    
    flens::checkStatus(status);
    flens::syncStream();
    
    // We have to correct the result -> only syncModed allowed
    ASSERT(flens::CudaEnv::isSyncCopyEnabled());
    // Correct Indexing
    result--; // cuBLAS is one-based, cblas is zero-based
}

// ziamax
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
iamax(IndexType n,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> x, IndexType incX,
      IndexType &result)
{
    CXXBLAS_DEBUG_OUT("cublasIzamax");

    cublasStatus_t status = cublasIzamax(flens::CudaEnv::getHandle(), n, 
                                         reinterpret_cast<const cuDoubleComplex*>(x.get()), incX, 
                                         &result);
    
    flens::checkStatus(status);
    flens::checkStatus(status);
    flens::syncStream();
    
    // We have to correct the result -> only syncModed allowed
    ASSERT(flens::CudaEnv::isSyncCopyEnabled());
    result--; // cuBLAS is one-based, cblas is zero-based
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL1_IAMAX_TCC
