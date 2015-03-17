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

#ifndef PLAYGROUND_CXXBLAS_LEVEL2_TPMV_TCC
#define PLAYGROUND_CXXBLAS_LEVEL2_TPMV_TCC 1

#include <cxxblas/cxxblas.h>

namespace cxxblas {

#ifdef HAVE_CLBLAS

// tpmv [generic]
template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const T, flens::StorageType::OpenCL> A,
     flens::device_ptr<T, flens::StorageType::OpenCL> x, IndexType incX)
{
    using std::abs;
    
    flens::CustomAllocator<T, flens::StorageType::OpenCL>   allocator;

    // Allocate memory for work
    flens::device_ptr<T, flens::StorageType::OpenCL> dev_work   = allocator.allocate(1 + (n-1)*abs(incX));
      
    tpmv(order, upLo, transA, diag, n, A, x, incX, dev_work);
      
    // Deallocate memory again
    allocator.deallocate(dev_work, 1 + (n-1)*abs(incX));
}

// stpmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const float, flens::StorageType::OpenCL> A,
     flens::device_ptr<float, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<float, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasStpmv");

    cl_int status = CLBLAS_IMPL(Stpmv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n,
                                       A.get(), A.getOffset(),
                                       x.get(), x.getOffset(), incX, work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
    
}

// dtpmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const double, flens::StorageType::OpenCL> A,
     flens::device_ptr<double, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<double, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasDtpmv");

    cl_int status = CLBLAS_IMPL(Dtpmv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n,
                                       A.get(), A.getOffset(),
                                       x.get(), x.getOffset(), incX, work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// ctpmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> A,
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasCtpmv");

    cl_int status = CLBLAS_IMPL(Ctpmv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n,
                                       A.get(), A.getOffset(),
                                       x.get(), x.getOffset(), incX, work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// ztpmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> A,
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX,
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> work)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasZtpmv");
    
    cl_int status = CLBLAS_IMPL(Ztpmv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n,
                                       A.get(), A.getOffset(),
                                       x.get(), x.getOffset(), incX,
                                       work.get(),
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}
    
#endif // HAVE_CLBLAS

#ifdef HAVE_CUBLAS

// stpmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const float, flens::StorageType::CUDA> A,
     flens::device_ptr<float, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasStpmv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tpmv(ColMajor, upLo, transA, diag, n, A, x, incX);
        return;
    }
    cublasStatus_t status = cublasStpmv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n,
                                        A.get(),
                                        x.get(), incX);
    
    flens::checkStatus(status);
}

// dtpmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const double, flens::StorageType::CUDA> A,
      flens::device_ptr<double, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasDtpmv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tpmv(ColMajor, upLo, transA, diag, n, A, x, incX);
        return;
    }
    cublasStatus_t status = cublasDtpmv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n,
                                        A.get(),
                                        x.get(), incX);
    
    flens::checkStatus(status);
}
// ctpmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> A,
     flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasCtpmv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tpmv(ColMajor, upLo, transA, diag, n, A, x, incX);
        return;
    }
    cublasStatus_t status = cublasCtpmv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n,
                                        reinterpret_cast<const cuFloatComplex*>(A.get()),
                                        reinterpret_cast<cuFloatComplex*>(x.get()), incX);
    
    flens::checkStatus(status);
}

// ztpmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
tpmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> A,
     flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasZtpmv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        tpmv(ColMajor, upLo, transA, diag, n, A, x, incX);
        return;
    }
    cublasStatus_t status = cublasZtpmv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n,
                                        reinterpret_cast<const cuDoubleComplex*>(A.get()),
                                        reinterpret_cast<cuDoubleComplex*>(x.get()), incX);
    
    flens::checkStatus(status);
}

#endif // HAVE_CUBLAS


} // namespace flens

#endif // PLAYGROUND_CXXBLAS_LEVEL2_TPMV_TCC
