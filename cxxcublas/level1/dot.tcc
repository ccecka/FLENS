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

#ifndef PLAYGROUND_CXXBLAS_LEVEL1_DOT_CC
#define PLAYGROUND_CXXBLAS_LEVEL1_DOT_CC 1

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/cxxblas.h>

namespace cxxblas {

#ifdef HAVE_CLBLAS

// dot template, result on host
template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const flens::device_ptr<const T, flens::StorageType::OpenCL> x, IndexType incX,
    const flens::device_ptr<const T, flens::StorageType::OpenCL> y, IndexType incY,
    T &result)
{
     flens::CustomAllocator<T, flens::StorageType::OpenCL>   allocator;
    // Allocate memory for work and result
    flens::device_ptr<T, flens::StorageType::OpenCL> dev_work   = allocator.allocate(n);
    flens::device_ptr<T, flens::StorageType::OpenCL> dev_result = allocator.allocate(1);

    cxxblas::dot(n, x, incX, y, incY, dev_result, dev_work);
    
    // Copy result to CPU
    cxxblas::copy(1, dev_result, 1, &result, 1);
    allocator.deallocate(dev_work, n);
    allocator.deallocate(dev_result, 1);

}

// dotu template, result on host
template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
dotu(IndexType n,
     const flens::device_ptr<const T, flens::StorageType::OpenCL> x, IndexType incX,
     const flens::device_ptr<const T, flens::StorageType::OpenCL> y, IndexType incY,
     T &result)
{
     flens::CustomAllocator<T, flens::StorageType::OpenCL>   allocator;
    // Allocate memory for work and result
    flens::device_ptr<T, flens::StorageType::OpenCL> dev_work   = allocator.allocate(n);
    flens::device_ptr<T, flens::StorageType::OpenCL> dev_result = allocator.allocate(1);

    cxxblas::dotu(n, x, incX, y, incY, dev_result, dev_work);
    
    // Copy result to CPU
    cxxblas::copy(1, dev_result, 1, &result, 1);
    allocator.deallocate(dev_work, n);
    allocator.deallocate(dev_result, 1);

}

// sdot, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const flens::device_ptr<const float, flens::StorageType::OpenCL> x, IndexType incX,
    const flens::device_ptr<const float, flens::StorageType::OpenCL> y, IndexType incY,
    flens::device_ptr<float, flens::StorageType::OpenCL> result,
    flens::device_ptr<float, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasSdot");

    cl_int status = CLBLAS_IMPL(Sdot)(n, 
                                      result.get(), result.getOffset(),
                                      x.get(), x.getOffset(), incX,
                                      y.get(), y.getOffset(), incY,
                                      work.get(),
                                      1, flens::OpenCLEnv::getQueuePtr(),
                                      0, NULL, NULL);

    flens::checkStatus(status);
}

// sdot, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dotu(IndexType n,
     const flens::device_ptr<const float, flens::StorageType::OpenCL> x, IndexType incX,
     const flens::device_ptr<const float, flens::StorageType::OpenCL> y, IndexType incY,
     flens::device_ptr<float, flens::StorageType::OpenCL> result,
     flens::device_ptr<float, flens::StorageType::OpenCL> work)
{
    cxxblas::dot(n, x, incX, y, incY, result, work);
}


// ddot, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const flens::device_ptr<const double, flens::StorageType::OpenCL> x, IndexType incX,
    const flens::device_ptr<const double, flens::StorageType::OpenCL> y, IndexType incY,
    flens::device_ptr<double, flens::StorageType::OpenCL> result,
    flens::device_ptr<double, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasDdot");

    cl_int status = CLBLAS_IMPL(Ddot)(n, 
                                      result.get(), result.getOffset(),
                                      x.get(), x.getOffset(), incX,
                                      y.get(), y.getOffset(), incY,
                                      work.get(),
                                      1, flens::OpenCLEnv::getQueuePtr(),
                                      0, NULL, NULL);

    flens::checkStatus(status);
}

// ddotu, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dotu(IndexType n,
     const flens::device_ptr<const double, flens::StorageType::OpenCL> x, IndexType incX,
     const flens::device_ptr<const double, flens::StorageType::OpenCL> y, IndexType incY,
     flens::device_ptr<double, flens::StorageType::OpenCL> result,
     flens::device_ptr<double, flens::StorageType::OpenCL> work)
{
    cxxblas::dot(n, x, incX, y, incY, result, work);
}

// sdot, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX,
    const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> y, IndexType incY,
    flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> result,
    flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasCdotc");

    cl_int status = CLBLAS_IMPL(Cdotc)(n, 
                                       result.get(), result.getOffset(),
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// cdotu, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dotu(IndexType n,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> y, IndexType incY,
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> result,
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasCdotu");

    cl_int status = CLBLAS_IMPL(Cdotu)(n, 
                                       result.get(), result.getOffset(),
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// sdot, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX,
    const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> y, IndexType incY,
    flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> result,
    flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasZdotc");

    cl_int status = CLBLAS_IMPL(Zdotc)(n, 
                                       result.get(), result.getOffset(),
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// cdotu, result on device (with temporary workspace)
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dotu(IndexType n,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> y, IndexType incY,
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> result,
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clblasZdotu");

    cl_int status = CLBLAS_IMPL(Zdotu)(n, 
                                       result.get(), result.getOffset(),
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

#endif // HAVE_CLBLAS

#ifdef HAVE_CUBLAS

// scopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const flens::device_ptr<const float, flens::StorageType::CUDA> x, IndexType incX,
    const flens::device_ptr<const float, flens::StorageType::CUDA> y, IndexType incY,
    float &result)
{
    CXXBLAS_DEBUG_OUT(" cublasSdot");
   
    ASSERT(x.getDeviceID()==y.getDeviceID());

    cublasStatus_t status = cublasSdot(flens::CudaEnv::getHandle(), n, 
                                       x.get(), incX, 
                                       y.get(), incY, 
                                       &result);
    
    flens::checkStatus(status);
    if (flens::CudaEnv::isSyncCopyEnabled()) {
        flens::syncStream();
    }
}

// sdotu
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dotu(IndexType n,
     const flens::device_ptr<const float, flens::StorageType::CUDA> x, IndexType incX,
     const flens::device_ptr<const float, flens::StorageType::CUDA> y, IndexType incY,
     float &result)
{
    dot(n, x, incX, y, incY, result);
}

// ddot
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const flens::device_ptr<const double, flens::StorageType::CUDA> x, IndexType incX,
    const flens::device_ptr<const double, flens::StorageType::CUDA> y, IndexType incY,
    double &result)
{
    CXXBLAS_DEBUG_OUT("cublasDdot");
    
    ASSERT(x.getDeviceID()==y.getDeviceID());    

    cublasStatus_t status = cublasDdot(flens::CudaEnv::getHandle(), n, 
                                       x.get(), incX, 
                                       y.get(), incY, &result);
    
    flens::checkStatus(status);
    if (flens::CudaEnv::isSyncCopyEnabled()) {
        flens::syncStream();
    }
}

// ddotu
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dotu(IndexType n,
    const flens::device_ptr<const double, flens::StorageType::CUDA> x, IndexType incX,
    const flens::device_ptr<const double, flens::StorageType::CUDA> y, IndexType incY,
    double &result)
{
    dot(n, x, incX, y, incY, result);
}

// cdotc
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> x, IndexType incX,
    const flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> y, IndexType incY, 
    ComplexFloat &result)
{
    CXXBLAS_DEBUG_OUT("cublasCdotc");
    
    ASSERT(x.getDeviceID()==y.getDeviceID());
    
    cublasStatus_t status = cublasCdotc(flens::CudaEnv::getHandle(), n, 
                                        reinterpret_cast<const cuFloatComplex*>(x.get()), incX, 
                                        reinterpret_cast<const cuFloatComplex*>(y.get()), incY,
                                        reinterpret_cast<cuFloatComplex*>(&result));

    
    flens::checkStatus(status);
    if (flens::CudaEnv::isSyncCopyEnabled()) {
        flens::syncStream();
    }
}

// cdotu
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dotu(IndexType n,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> x, IndexType incX,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> y, IndexType incY, 
     ComplexFloat &result)
{
    CXXBLAS_DEBUG_OUT("cublasCdotu");
    
    ASSERT(x.getDeviceID()==y.getDeviceID());
    
    cublasStatus_t status = cublasCdotu(flens::CudaEnv::getHandle(), n, 
                                        reinterpret_cast<const cuFloatComplex*>(x.get()), incX, 
                                        reinterpret_cast<const cuFloatComplex*>(y.get()), incY,
                                        reinterpret_cast<cuFloatComplex*>(&result));

    
    flens::checkStatus(status);
    if (flens::CudaEnv::isSyncCopyEnabled()) {
        flens::syncStream();
    }
}

// zdotc
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dot(IndexType n,
    const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> x, IndexType incX,
    const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> y, IndexType incY,
    ComplexDouble &result)
{
    CXXBLAS_DEBUG_OUT("cublasZdot");
    
    ASSERT(x.getDeviceID()==y.getDeviceID());
    
    cublasStatus_t status = cublasZdotc(flens::CudaEnv::getHandle(), n, 
                                        reinterpret_cast<const cuDoubleComplex*>(x.get()), incX, 
                                        reinterpret_cast<const cuDoubleComplex*>(y.get()), incY,
                                        reinterpret_cast<cuDoubleComplex*>(&result));
    
    flens::checkStatus(status);
    if (flens::CudaEnv::isSyncCopyEnabled()) {
        flens::syncStream();
    }
}

//zdotu
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
dotu(IndexType n,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> x, IndexType incX,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> y, IndexType incY,
     ComplexDouble &result)
{
    CXXBLAS_DEBUG_OUT("cublasZdotu");
    
    ASSERT(x.getDeviceID()==y.getDeviceID());

    cublasStatus_t status = cublasZdotu(flens::CudaEnv::getHandle(), n, 
                                        reinterpret_cast<const cuDoubleComplex*>(x.get()), incX, 
                                        reinterpret_cast<const cuDoubleComplex*>(y.get()), incY,
                                        reinterpret_cast<cuDoubleComplex*>(&result));
    
    flens::checkStatus(status);
    if (flens::CudaEnv::isSyncCopyEnabled()) {
        flens::syncStream();
    }
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL1_DOT_CC
