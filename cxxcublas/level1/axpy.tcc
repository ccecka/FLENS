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

#ifndef PLAYGROUND_CXXBLAS_LEVEL1_AXPY_TCC
#define PLAYGROUND_CXXBLAS_LEVEL1_AXPY_TCC 1

#include <cxxblas/cxxblas.h>

namespace cxxblas {

#ifdef HAVE_CLBLAS

// saxpy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
axpy(IndexType n, float alpha,
     const flens::device_ptr<const float, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<float, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clblasSaxpy");

    cl_int status = CLBLAS_IMPL(Saxpy)(n, alpha,
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// daxpy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
axpy(IndexType n, double alpha,
     const flens::device_ptr<const double, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<double, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clblasDaxpy");

    cl_int status = CLBLAS_IMPL(Daxpy)(n, alpha,
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// caxpy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
axpy(IndexType n, ComplexFloat alpha,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clblasCaxpy");

    cl_int status = CLBLAS_IMPL(Caxpy)(n, *(reinterpret_cast<cl_float2*>(&alpha)),
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// zaxpy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
axpy(IndexType n, ComplexDouble alpha,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clblasZaxpy");

    cl_int status = CLBLAS_IMPL(Zaxpy)(n, *(reinterpret_cast<cl_double2*>(&alpha)),
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

#endif // HAVE_CLBLAS

#ifdef HAVE_CUBLAS

// saxpy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
axpy(IndexType n, float alpha,
     const flens::device_ptr<const float, flens::StorageType::CUDA> x, IndexType incX, 
     flens::device_ptr<float, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT(" cublasSaxpy");
    
    ASSERT(x.getDeviceID()==y.getDeviceID());

    cublasStatus_t status = cublasSaxpy(flens::CudaEnv::getHandle(), 
                                        n, &alpha, 
                                        x.get(), incX, 
                                        y.get(), incY);
    
    flens::checkStatus(status);
  
}

// daxpy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
axpy(IndexType n, double alpha, 
     const flens::device_ptr<const double, flens::StorageType::CUDA> x, IndexType incX,
     flens::device_ptr<double, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasDaxpy");
    
    ASSERT(x.getDeviceID()==y.getDeviceID());

    cublasStatus_t status = cublasDaxpy(flens::CudaEnv::getHandle(), 
                                        n, &alpha, 
                                        x.get(), incX, 
                                        y.get(), incY);
    
    flens::checkStatus(status);
  
}

// caxpy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
axpy(IndexType n, ComplexFloat alpha,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> x, IndexType incX,
     flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasCaxpy");

    ASSERT(x.getDeviceID()==y.getDeviceID());

    cublasStatus_t status = cublasCaxpy(flens::CudaEnv::getHandle(), n, reinterpret_cast<cuFloatComplex*>(&alpha),
                                        reinterpret_cast<const cuFloatComplex*>(x.get()), incX, 
                                        reinterpret_cast<cuFloatComplex*>(y.get()), incY);
    
    flens::checkStatus(status);
    
}

// zaxpy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
axpy(IndexType n, ComplexDouble alpha,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> x, IndexType incX,
     flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasZaxpy");

    ASSERT(x.getDeviceID()==y.getDeviceID());
    
    cublasStatus_t status = cublasZaxpy(flens::CudaEnv::getHandle(), n, reinterpret_cast<cuDoubleComplex*>(&alpha),
                                        reinterpret_cast<const cuDoubleComplex*>(x.get()), incX, 
                                        reinterpret_cast<cuDoubleComplex*>(y.get()), incY);
    
    flens::checkStatus(status);
 
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL1_AXPY_TCC
