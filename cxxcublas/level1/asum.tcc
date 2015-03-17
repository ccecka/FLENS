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

#ifndef PLAYGROUND_CXXBLAS_LEVEL1_ASUM_TCC
#define PLAYGROUND_CXXBLAS_LEVEL1_ASUM_TCC 1

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/cxxblas.h>

namespace cxxblas {

#ifdef HAVE_CLBLAS

// asum template for complex vectors, result on host
template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
asum(IndexType n,
    const flens::device_ptr<const T, flens::StorageType::OpenCL> x, IndexType incX,
    T &result)
{
    flens::CustomAllocator<T, flens::StorageType::OpenCL>                 allocator_real;

    // Allocate memory for work and result
    flens::device_ptr<T, flens::StorageType::OpenCL>  dev_work   = allocator_real.allocate(2*n);
    flens::device_ptr<T, flens::StorageType::OpenCL>  dev_result = allocator_real.allocate(1);

    cxxblas::asum(n, x, incX, dev_result, dev_work);
    
    // Copy result to CPU
    cxxblas::copy(1, dev_result, 1, &result, 1);
    allocator_real.deallocate(dev_work, n);
    allocator_real.deallocate(dev_result, 1);

}


// asum template for complex vectors, result on host
template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
asum(IndexType n,
    const flens::device_ptr<const std::complex<T>, flens::StorageType::OpenCL> x, IndexType incX,
    T &result)
{

    flens::CustomAllocator<T, flens::StorageType::OpenCL>                 allocator_real;
    flens::CustomAllocator<std::complex<T>, flens::StorageType::OpenCL>   allocator_complex;

    // Allocate memory for work and result
    flens::device_ptr<std::complex<T>, flens::StorageType::OpenCL> dev_work   = allocator_complex.allocate(2*n);
    flens::device_ptr<T, flens::StorageType::OpenCL>               dev_result = allocator_real.allocate(1);

    cxxblas::asum(n, x, incX, dev_result, dev_work);
    
    // Copy result to CPU
    cxxblas::copy(1, dev_result, 1, &result, 1);
    allocator_complex.deallocate(dev_work, n);
    allocator_real.deallocate(dev_result, 1);

}

// sasum, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
asum(IndexType n,
     const flens::device_ptr<const float, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<float, flens::StorageType::OpenCL> result,
     flens::device_ptr<float, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasSasum");

    cl_int status = CLBLAS_IMPL(Sasum)(n, 
                                       result.get(), result.getOffset(),
                                       x.get(), x.getOffset(), incX,
                                       work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// dasum, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
asum(IndexType n,
     const flens::device_ptr<const double, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<double, flens::StorageType::OpenCL> result,
     flens::device_ptr<double, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasDasum");

    cl_int status = CLBLAS_IMPL(Dasum)(n, 
                                       result.get(), result.getOffset(),
                                       x.get(), x.getOffset(), incX,
                                       work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// casum, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
asum(IndexType n,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<float, flens::StorageType::OpenCL> result,
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasScasum");

    cl_int status = CLBLAS_IMPL(Scasum)(n, 
                                        result.get(), result.getOffset(),
                                        x.get(), x.getOffset(), incX,
                                        work.get(),
                                        1, flens::OpenCLEnv::getQueuePtr(),
                                        0, NULL, NULL);

    flens::checkStatus(status);
}

// zasum, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
asum(IndexType n,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<double, flens::StorageType::OpenCL> result,
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasDzasum");

    cl_int status = CLBLAS_IMPL(Dzasum)(n, 
                                        result.get(), result.getOffset(),
                                        x.get(), x.getOffset(), incX,
                                        work.get(),
                                        1, flens::OpenCLEnv::getQueuePtr(),
                                        0, NULL, NULL);

    flens::checkStatus(status);
}

#endif

#ifdef HAVE_CUBLAS

// scopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
asum(IndexType n,
    const flens::device_ptr<const float, flens::StorageType::CUDA> x, IndexType incX,
    float &result)
{
    CXXBLAS_DEBUG_OUT(" cublasSasum");

    cublasStatus_t status = cublasSasum(flens::CudaEnv::getHandle(), n, x, incX, &result);
    
    flens::checkStatus(status);
    if (flens::CudaEnv::isSyncCopyEnabled()) {
        flens::syncStream();
    }
}


// dasum
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
asum(IndexType n,
    const flens::device_ptr<const double, flens::StorageType::CUDA> x, IndexType incX,
    double &result)
{
    CXXBLAS_DEBUG_OUT("cublasDasum");

    cublasStatus_t status = cublasDasum(flens::CudaEnv::getHandle(), n, 
                                        x.get(), incX, 
                                        &result);
    
    flens::checkStatus(status);
    if (flens::CudaEnv::isSyncCopyEnabled()) {
        flens::syncStream();
    }
}


// casum
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
asum(IndexType n,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> x, IndexType incX,
     float &result)
{
    CXXBLAS_DEBUG_OUT("cublasCasum");
    
    cublasStatus_t status = cublasScasum(flens::CudaEnv::getHandle(), n, 
                                         reinterpret_cast<const cuFloatComplex*>(x.get()), incX, 
                                         &result);
    
    flens::checkStatus(status);
    if (flens::CudaEnv::isSyncCopyEnabled()) {
        flens::syncStream();
    }
}

// zasum
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
asum(IndexType n,
    const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> x, IndexType incX,
    double &result)
{
    CXXBLAS_DEBUG_OUT("cublasDzasum");
 
    cublasStatus_t status = cublasDzasum(flens::CudaEnv::getHandle(), n, 
                                      reinterpret_cast<const cuDoubleComplex*>(x.get()), incX, 
                                      &result);
    
    flens::checkStatus(status);
    if (flens::CudaEnv::isSyncCopyEnabled()) {
        flens::syncStream();
    }
  
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL1_ASUM_TCC
