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

#ifndef PLAYGROUND_CXXBLAS_LEVEL1_COPY_TCC
#define PLAYGROUND_CXXBLAS_LEVEL1_COPY_TCC 1

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/cxxblas.h>

namespace cxxblas {

#ifdef HAVE_CLBLAS

template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, 
     const T *x, IndexType incX, 
     flens::device_ptr<T, flens::StorageType::OpenCL> y, IndexType incY)
{
    if(flens::OpenCLEnv::isSyncCopyEnabled()) {
        CXXBLAS_DEBUG_OUT("clblasSetVector [extensionm, sync]");
    } else {
        CXXBLAS_DEBUG_OUT("clblasSetVector [extension, async]");
    }
    cl_bool syncCopy = (flens::OpenCLEnv::isSyncCopyEnabled()==true) ? CL_TRUE : CL_FALSE;
    
    if (incX==1 && incY==1) {

        cl_int status = clEnqueueWriteBuffer(flens::OpenCLEnv::getQueue(), 
                                             y.get(), 
                                             syncCopy, 
                                             y.getOffset()*sizeof(T),
                                             n * sizeof(T), 
                                             x, 
                                             0, NULL, flens::OpenCLEnv::getEventPtr());
    
        flens::checkStatus(status);   
    } else {
        cl_int status;
        for (IndexType i=0; i<n; ++i) {
            status = clEnqueueWriteBuffer(flens::OpenCLEnv::getQueue(), 
                                          y.get(), 
                                          syncCopy, 
                                          (y.getOffset()+i*incY)*sizeof(T),
                                          sizeof(T), 
                                          x+i*incX, 
                                          0, NULL, flens::OpenCLEnv::getEventPtr());
            flens::checkStatus(status);  
        }
    }
}

template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, 
     const flens::device_ptr<const T, flens::StorageType::OpenCL> x, IndexType incX, 
     T *y, IndexType incY)
{
    if(flens::OpenCLEnv::isSyncCopyEnabled()) {
        CXXBLAS_DEBUG_OUT("clblasGetVector [extensionm, sync]");
    } else {
        CXXBLAS_DEBUG_OUT("clblasGetVector [extension, async]");
    }
    
    cl_bool syncCopy = (flens::OpenCLEnv::isSyncCopyEnabled()==true) ? CL_TRUE : CL_FALSE;
    
    if (incX==1 && incY==1) {
        cl_int status = clEnqueueReadBuffer(flens::OpenCLEnv::getQueue(), 
                                            x.get(), 
                                            syncCopy,
                                            x.getOffset()*sizeof(T),
                                            n * sizeof(T),
                                            y, 0, NULL, flens::OpenCLEnv::getEventPtr());
        flens::checkStatus(status); 
    } else {
        cl_int status;
        for (IndexType i=0; i<n; ++i) {
            status = clEnqueueReadBuffer(flens::OpenCLEnv::getQueue(), 
                                         x.get(), 
                                         syncCopy, 
                                         (x.getOffset()+i*incX)*sizeof(T),
                                         sizeof(T),
                                         y+i*incY, 0, NULL, flens::OpenCLEnv::getEventPtr());
            flens::checkStatus(status);  
        }

    }  
}

// scopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, 
     const flens::device_ptr<const float, flens::StorageType::OpenCL> x, IndexType incX, 
     flens::device_ptr<float, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasScopy");

    cl_int status = CLBLAS_IMPL(Scopy)(n, 
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, flens::OpenCLEnv::getEventPtr());

    flens::checkStatus(status); 
}

// dcopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, 
     const flens::device_ptr<const double, flens::StorageType::OpenCL> x, IndexType incX, 
     flens::device_ptr<double, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clBlasDcopy");

    cl_int status = CLBLAS_IMPL(Dcopy)(n, 
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, flens::OpenCLEnv::getEventPtr());

    flens::checkStatus(status); 
}

// ccopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, 
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX, 
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clblasCcopy");

    cl_int status = CLBLAS_IMPL(Ccopy)(n, 
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, flens::OpenCLEnv::getEventPtr());

    flens::checkStatus(status); 
}

// zcopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, 
     const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX, 
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clblasZcopy");

    cl_int status = CLBLAS_IMPL(Zcopy)(n, 
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, flens::OpenCLEnv::getEventPtr());

    flens::checkStatus(status); 
}

#endif // WITH_CLBLAS

#ifdef HAVE_CUBLAS

template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, 
     const T *x, IndexType incX, 
     flens::device_ptr<T, flens::StorageType::CUDA> y, IndexType incY)
{
    if (flens::CudaEnv::isSyncCopyEnabled()) {
    
        CXXBLAS_DEBUG_OUT("cublasSetVector [sync]");      
        cublasStatus_t status = cublasSetVector(n, sizeof(T), x, incX, y.get(), incY);
        flens::checkStatus(status); 

    } else {
      
        CXXBLAS_DEBUG_OUT("cublasSetVector [async]");   
        cublasStatus_t status = cublasSetVectorAsync(n, sizeof(T), x, incX, y.get(), incY, flens::CudaEnv::getStream());
        flens::checkStatus(status);  

    }  
}

template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, 
     const flens::device_ptr<const T, flens::StorageType::CUDA> x, IndexType incX, 
     T *y, IndexType incY)
{

    if (flens::CudaEnv::isSyncCopyEnabled()) {
      
        CXXBLAS_DEBUG_OUT("cublasGetVector [sync]");
        cublasStatus_t status = cublasGetVector(n, sizeof(T), x.get(), incX, y, incY);
        flens::checkStatus(status);   
	
    } else {
      
        CXXBLAS_DEBUG_OUT("cublasGetVector [async]");
        cublasStatus_t status = cublasGetVectorAsync(n, sizeof(T), x.get(), incX, y, incY, flens::CudaEnv::getStream());
        flens::checkStatus(status); 
	
    }
}

// scopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, 
     const flens::device_ptr<const float, flens::StorageType::CUDA> x, IndexType incX, 
     flens::device_ptr<float, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasScopy");
   
    ASSERT(x.getDeviceID()==y.getDeviceID());

    cublasStatus_t status = cublasScopy(flens::CudaEnv::getHandle(), n, 
                                        x.get(), incX, 
                                        y.get(), incY);
    
    flens::checkStatus(status);
}

// dcopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, 
     const flens::device_ptr<const double, flens::StorageType::CUDA> x, IndexType incX,
     flens::device_ptr<double, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasDcopy");
    
    ASSERT(x.getDeviceID()==y.getDeviceID());  

    cublasStatus_t status = cublasDcopy(flens::CudaEnv::getHandle(), n, 
                                        x.get(), incX, 
                                        y.get(), incY);
    
    flens::checkStatus(status);
  
}

// ccopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, 
     const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> x, IndexType incX,
     flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasCcopy");
    
    ASSERT(x.getDeviceID()==y.getDeviceID());
    
    cublasStatus_t status = cublasCcopy(flens::CudaEnv::getHandle(), n, 
                                        reinterpret_cast<const cuFloatComplex*>(x.get()), incX, 
                                        reinterpret_cast<cuFloatComplex*>(y.get()), incY);

    
    flens::checkStatus(status);
}

// zcopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, 
     const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> x, IndexType incX,
     flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasZcopy");
    
    ASSERT(x.getDeviceID()==y.getDeviceID());
    
    cublasStatus_t status = cublasZcopy(flens::CudaEnv::getHandle(), n, 
                                        reinterpret_cast<const cuDoubleComplex*>(x.get()), incX, 
                                        reinterpret_cast<cuDoubleComplex*>(y.get()), incY);
    
    flens::checkStatus(status);
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL1_COPY_TCC
