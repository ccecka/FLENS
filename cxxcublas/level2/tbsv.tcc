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

#ifndef PLAYGROUND_CXXBLAS_LEVEL2_TBSV_TCC
#define PLAYGROUND_CXXBLAS_LEVEL2_TBSV_TCC 1

#include <cxxblas/cxxblas.h>

namespace cxxblas {

#ifdef HAVE_CLBLAS

// stbsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbsv(StorageOrder order, StorageUpLo upLo,
      Transpose transA, Diag diag,
      IndexType n, IndexType k,
      const flens::device_ptr<const float, flens::StorageType::OpenCL> A, IndexType ldA,
      flens::device_ptr<float, flens::StorageType::OpenCL> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasStbsv");

    cl_int status = CLBLAS_IMPL(Stbsv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n, k,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
    
}

// dtbsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbsv(StorageOrder order, StorageUpLo upLo,
      Transpose transA, Diag diag,
      IndexType n, IndexType k,
      const flens::device_ptr<const double, flens::StorageType::OpenCL> A, IndexType ldA,
      flens::device_ptr<double, flens::StorageType::OpenCL> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasDtbsv");

    cl_int status = CLBLAS_IMPL(Dtbsv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n, k,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// ctbsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbsv(StorageOrder order, StorageUpLo upLo,
      Transpose transA, Diag diag,
      IndexType n, IndexType k,
      const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> A, IndexType ldA,
      flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasCtbsv");

    cl_int status = CLBLAS_IMPL(Ctbsv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n,  k,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// ztbsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbsv(StorageOrder order, StorageUpLo upLo,
      Transpose transA, Diag diag,
      IndexType n, IndexType k,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> A, IndexType ldA,
      flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasZtbsv");
    
    cl_int status = CLBLAS_IMPL(Ztbsv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n, k,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}
    
#endif // HAVE_CLBLAS

#ifdef HAVE_CUBLAS

// stbsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const float, flens::StorageType::CUDA> A, IndexType ldA,
     flens::device_ptr<float, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasStbsv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tbsv(ColMajor, upLo, transA, diag, n, k, A, ldA, x, incX);
        return;
    }
    cublasStatus_t status = cublasStbsv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n, k,
                                        A.get(), ldA,
                                        x.get(), incX);
    
    flens::checkStatus(status);
}

// dtbsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const double, flens::StorageType::CUDA> A, IndexType ldA,
      flens::device_ptr<double, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasDtbsv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tbsv(ColMajor, upLo, transA, diag, n, k, A, ldA, x, incX);
        return;
    }
    cublasStatus_t status = cublasDtbsv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n, k,
                                        A.get(), ldA,
                                        x.get(), incX);
    
    flens::checkStatus(status);
}
// ctbsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> A, IndexType ldA,
     flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasCtbsv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tbsv(ColMajor, upLo, transA, diag, n, k, A, ldA, x, incX);
        return;
    }
    cublasStatus_t status = cublasCtbsv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n, k,
                                        reinterpret_cast<const cuFloatComplex*>(A.get()), ldA,
                                        reinterpret_cast<cuFloatComplex*>(x.get()), incX);
    
    flens::checkStatus(status);
}

// ztbsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tbsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n, IndexType k,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> A, IndexType ldA,
     flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasZtbsv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tbsv(ColMajor, upLo, transA, diag, n, k, A, ldA, x, incX);
        return;
    }
    cublasStatus_t status = cublasZtbsv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n, k,
                                        reinterpret_cast<const cuDoubleComplex*>(A.get()), ldA,
                                        reinterpret_cast<cuDoubleComplex*>(x.get()), incX);
    
    flens::checkStatus(status);
}

#endif // HAVE_CUBLAS


} // namespace flens

#endif // PLAYGROUND_CXXBLAS_LEVEL2_TBSV_TCC
