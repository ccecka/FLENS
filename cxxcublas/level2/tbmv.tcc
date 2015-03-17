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

#ifndef PLAYGROUND_CXXBLAS_LEVEL2_TBMV_TCC
#define PLAYGROUND_CXXBLAS_LEVEL2_TBMV_TCC 1

#include <cxxblas/cxxblas.h>

namespace cxxblas {

#ifdef HAVE_CLBLAS

// tbmv [generic]
template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const T, flens::StorageType::OpenCL> A, IndexType ldA,
     flens::device_ptr<T, flens::StorageType::OpenCL> x, IndexType incX)
{
    using std::abs;
    
    flens::CustomAllocator<T, flens::StorageType::OpenCL>   allocator;

    // Allocate memory for work
    flens::device_ptr<T, flens::StorageType::OpenCL> dev_work   = allocator.allocate(1 + (n-1)*abs(incX));
      
    tbmv(order, upLo, transA, diag, n, k, A, ldA, x, incX, dev_work);
      
    // Deallocate memory again
    allocator.deallocate(dev_work, 1 + (n-1)*abs(incX));
}

// stbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const float, flens::StorageType::OpenCL> A, IndexType ldA,
     flens::device_ptr<float, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<float, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasStbmv");

    cl_int status = CLBLAS_IMPL(Stbmv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n, k,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX, work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
    
}

// dtbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const double, flens::StorageType::OpenCL> A, IndexType ldA,
     flens::device_ptr<double, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<double, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasDtbmv");

    cl_int status = CLBLAS_IMPL(Dtbmv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n, k,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX, work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                        0, NULL, NULL);

    flens::checkStatus(status);
}

// ctbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> A, IndexType ldA,
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasCtbmv");

    cl_int status = CLBLAS_IMPL(Ctbmv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n, k,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX, work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// ztbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> A, IndexType ldA,
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasZtbmv");
    
    cl_int status = CLBLAS_IMPL(Ztbmv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n, k,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX,
                                       work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}
    
#endif // HAVE_CLBLAS

#ifdef HAVE_CUBLAS

// stbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const float, flens::StorageType::CUDA> A, IndexType ldA,
     flens::device_ptr<float, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasStbmv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tbmv(ColMajor, upLo, transA, diag, n, k, A, ldA, x, incX);
        return;
    }
    cublasStatus_t status = cublasStbmv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n, k,
                                        A.get(), ldA,
                                        x.get(), incX);
    
    flens::checkStatus(status);
}

// dtbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const double, flens::StorageType::CUDA> A, IndexType ldA,
      flens::device_ptr<double, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasDtbmv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tbmv(ColMajor, upLo, transA, diag, n, k, A, ldA, x, incX);
        return;
    }
    cublasStatus_t status = cublasDtbmv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n, k,
                                        A.get(), ldA,
                                        x.get(), incX);
    
    flens::checkStatus(status);
}
// ctbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> A, IndexType ldA,
     flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasCtbmv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tbmv(ColMajor, upLo, transA, diag, n, k, A, ldA, x, incX);
        return;
    }
    cublasStatus_t status = cublasCtbmv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n, k,
                                        reinterpret_cast<const cuFloatComplex*>(A.get()), ldA,
                                        reinterpret_cast<cuFloatComplex*>(x.get()), incX);
    
    flens::checkStatus(status);
}

// ztbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> A, IndexType ldA,
     flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasZtbmv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tbmv(ColMajor, upLo, transA, diag, n, k, A, ldA, x, incX);
        return;
    }
    cublasStatus_t status = cublasZtbmv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n, k,
                                        reinterpret_cast<const cuDoubleComplex*>(A.get()), ldA,
                                        reinterpret_cast<cuDoubleComplex*>(x.get()), incX);
    
    flens::checkStatus(status);
}

#endif // HAVE_CUBLAS


} // namespace flens

#endif // PLAYGROUND_CXXBLAS_LEVEL2_TBMV_TCC
