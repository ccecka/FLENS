/*
 *   Copyright (c) 2009, Michael Lehn
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

#ifndef PLAYGROUND_CXXBLAS_LEVEL2_GER_TCC
#define PLAYGROUND_CXXBLAS_LEVEL2_GER_TCC 1

#include <complex>
#include <cxxblas/cxxblas.h>

namespace cxxblas {

#ifdef HAVE_CLBLAS

// sger
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
ger(StorageOrder order,
    IndexType m, IndexType n,
    const float &alpha,
    const flens::device_ptr<const float, flens::StorageType::OpenCL> x, IndexType incX,
    const flens::device_ptr<const float, flens::StorageType::OpenCL> y, IndexType incY,
    flens::device_ptr<float, flens::StorageType::OpenCL> A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasSger");

    cl_int status = CLBLAS_IMPL_EX(Sger)(CLBLAS::getClblasType(order), 
                                         m,  n,
                                         alpha,
                                         x.get(), x.getOffset(), incX,
                                         y.get(), y.getOffset(), incY,
                                         A.get(), A.getOffset(), ldA,
                                         1, flens::OpenCLEnv::getQueuePtr(),
                                         0, NULL, NULL);

    flens::checkStatus(status);
    
}

// dger
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
ger(StorageOrder order,
    IndexType m, IndexType n,
    const double &alpha,
    const flens::device_ptr<const double, flens::StorageType::OpenCL> x, IndexType incX,
    const flens::device_ptr<const double, flens::StorageType::OpenCL> y, IndexType incY,
    flens::device_ptr<double, flens::StorageType::OpenCL> A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasDger");

    cl_int status = CLBLAS_IMPL_EX(Dger)(CLBLAS::getClblasType(order), 
                                         m,  n,
                                         alpha,
                                         x.get(), x.getOffset(), incX,
                                         y.get(), y.getOffset(), incY,
                                         A.get(), A.getOffset(), ldA,
                                         1, flens::OpenCLEnv::getQueuePtr(),
                                         0, NULL, NULL);

    flens::checkStatus(status);
    
}


// cgeru
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
ger(StorageOrder order,
    IndexType m, IndexType n,
    ComplexFloat alpha,
    const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX,
    const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> y, IndexType incY,
    flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasCgeru");

    cl_int status = CLBLAS_IMPL_EX(Cgeru)(CLBLAS::getClblasType(order), 
                                          m,  n,
                                          *(reinterpret_cast<cl_float2*>(&alpha)),
                                          x.get(), x.getOffset(), incX,
                                          y.get(), y.getOffset(), incY,
                                          A.get(), A.getOffset(), ldA,
                                          1, flens::OpenCLEnv::getQueuePtr(),
                                          0, NULL, NULL);

    flens::checkStatus(status);
    
}


// zgeru
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
ger(StorageOrder order,
    IndexType m, IndexType n,
    ComplexDouble alpha,
    const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX,
    const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> y, IndexType incY,
    flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasZgeru");

    cl_int status = CLBLAS_IMPL_EX(Zgeru)(CLBLAS::getClblasType(order), 
                                          m,  n,
                                          *(reinterpret_cast<cl_double2*>(&alpha)),
                                          x.get(), x.getOffset(), incX,
                                          y.get(), y.getOffset(), incY,
                                          A.get(), A.getOffset(), ldA,
                                          1, flens::OpenCLEnv::getQueuePtr(),
                                          0, NULL, NULL);

    flens::checkStatus(status);
    
}

// cgerc
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gerc(StorageOrder order,
      IndexType m, IndexType n,
      ComplexFloat alpha,
      const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX,
      const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> y, IndexType incY,
      flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasCgerc");

    cl_int status = CLBLAS_IMPL_EX(Cgerc)(CLBLAS::getClblasType(order), 
                                          m,  n,
                                          *(reinterpret_cast<cl_float2*>(&alpha)),
                                          x.get(), x.getOffset(), incX,
                                          y.get(), y.getOffset(), incY,
                                          A.get(), A.getOffset(), ldA,
                                          1, flens::OpenCLEnv::getQueuePtr(),
                                          0, NULL, NULL);

    flens::checkStatus(status);
    
}

// zgerc
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gerc(StorageOrder order,
      IndexType m, IndexType n,
      ComplexDouble alpha,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> y, IndexType incY,
      flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasZgerc");

    cl_int status = CLBLAS_IMPL_EX(Zgerc)(CLBLAS::getClblasType(order), 
                                          m,  n,
                                          *(reinterpret_cast<cl_double2*>(&alpha)),
                                          x.get(), x.getOffset(), incX,
                                          y.get(), y.getOffset(), incY,
                                          A.get(), A.getOffset(), ldA,
                                          1, flens::OpenCLEnv::getQueuePtr(),
                                          0, NULL, NULL);

    flens::checkStatus(status);
    
}


#endif // HAVE_CLBLAS
    
#ifdef HAVE_CUBLAS

// sger
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
    ger(StorageOrder order,
        IndexType m, IndexType n,
        const float &alpha,
        const flens::device_ptr<const float, flens::StorageType::CUDA> x, IndexType incX,
        const flens::device_ptr<const float, flens::StorageType::CUDA> y, IndexType incY,
        flens::device_ptr<float, flens::StorageType::CUDA> A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("cublasSger");
      
    if (order==RowMajor) {
        ger(ColMajor, n, m,
            alpha, y, incY, x, incX,
            A, ldA);
        return;
    } 
    
    cublasStatus_t status = cublasSger(flens::CudaEnv::getHandle(), 
                                        m, n,
                                        &alpha,
                                        x.get(), incX,
                                        y.get(), incY,
                                        A.get(), ldA);
    
    flens::checkStatus(status);      
}

// dger
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
    ger(StorageOrder order,
        IndexType m, IndexType n,
        const double &alpha,
        const flens::device_ptr<const double, flens::StorageType::CUDA> x, IndexType incX,
        const flens::device_ptr<const double, flens::StorageType::CUDA> y, IndexType incY,
        flens::device_ptr<double, flens::StorageType::CUDA> A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("cublasDger");
      
    if (order==RowMajor) {
        ger(ColMajor, n, m,
            alpha, y, incY, x, incX,
            A, ldA);
        return;
    } 
    
    cublasStatus_t status = cublasDger(flens::CudaEnv::getHandle(), 
                                        m, n,
                                        &alpha,
                                        x.get(), incX,
                                        y.get(), incY,
                                        A.get(), ldA);
    
    flens::checkStatus(status);      
}

// cgeru
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
ger(StorageOrder order,
    IndexType m, IndexType n,
    const ComplexFloat &alpha,
    const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> x, IndexType incX,
    const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> y, IndexType incY,
    flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("cublasCgeru");
    
    if (order==RowMajor) {
        ger(ColMajor, n, m,
            alpha, y, incY, x, incX,
            A, ldA);
        return;
    } 
    
    cublasStatus_t status = cublasCgeru(flens::CudaEnv::getHandle(), 
                                        m, n,
                                        reinterpret_cast<const cuFloatComplex*>(&alpha),
                                        reinterpret_cast<const cuFloatComplex*>(x.get()), incX,
                                        reinterpret_cast<const cuFloatComplex*>(y.get()), incY,
                                        reinterpret_cast<cuFloatComplex*>(A.get()), ldA);
    
    flens::checkStatus(status);      
}
// zgeru
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
ger(StorageOrder order,
    IndexType m, IndexType n,
    const ComplexDouble &alpha,
    const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> x, IndexType incX,
    const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> y, IndexType incY,
    flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("cublasZgeru");
    
    if (order==RowMajor) {
        ger(ColMajor, n, m,
            alpha, y, incY, x, incX,
            A, ldA);
        return;
    } 
    
    cublasStatus_t status = cublasZgeru(flens::CudaEnv::getHandle(), 
                                        m, n,
                                        reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                        reinterpret_cast<const cuDoubleComplex*>(x.get()), incX,
                                        reinterpret_cast<const cuDoubleComplex*>(y.get()), incY,
                                        reinterpret_cast<cuDoubleComplex*>(A.get()), ldA);
    
    flens::checkStatus(status);
    
}
      

// cgerc
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gerc(StorageOrder order,
     IndexType m, IndexType n,
     const ComplexFloat &alpha,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> x, IndexType incX,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> y, IndexType incY,
     flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("cublasCgerc");
    
    ASSERT (order==ColMajor);

    cublasStatus_t status = cublasCgerc(flens::CudaEnv::getHandle(), 
                                        m, n,
                                        reinterpret_cast<const cuFloatComplex*>(&alpha),
                                        reinterpret_cast<const cuFloatComplex*>(x.get()), incX,
                                        reinterpret_cast<const cuFloatComplex*>(y.get()), incY,
                                        reinterpret_cast<cuFloatComplex*>(A.get()), ldA);
    
    flens::checkStatus(status);
}

// zgerc
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gerc(StorageOrder order,
     IndexType m, IndexType n,
     const ComplexDouble &alpha,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> x, IndexType incX,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> y, IndexType incY,
     flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> A, IndexType ldA)
{
    CXXBLAS_DEBUG_OUT("cublasZgerc");
    
    ASSERT (order==ColMajor);

    cublasStatus_t status = cublasZgerc(flens::CudaEnv::getHandle(), 
                                        m, n,
                                        reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                        reinterpret_cast<const cuDoubleComplex*>(x.get()), incX,
                                        reinterpret_cast<const cuDoubleComplex*>(y.get()), incY,
                                        reinterpret_cast<cuDoubleComplex*>(A.get()), ldA);
    
    flens::checkStatus(status);
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL2_GER_TCC
