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

#ifndef CXXBLAS_LEVEL3_SYMM_TCC
#define CXXBLAS_LEVEL3_SYMM_TCC 1

#include <cxxblas/cxxblas.h>

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename MB,
          typename BETA, typename MC>
void
symm_generic(StorageOrder order, Side sideA, StorageUpLo upLoA,
             IndexType m, IndexType n,
             const ALPHA &alpha,
             const MA *A, IndexType ldA,
             const MB *B, IndexType ldB,
             const BETA &beta,
             MC *C, IndexType ldC)
{
    if (order==ColMajor) {
        upLoA = (upLoA==Upper) ? Lower : Upper;
        sideA = (sideA==Left) ? Right : Left;
        symm_generic(RowMajor, sideA, upLoA, n, m,
                     alpha, A, ldA, B, ldB,
                     beta,
                     C, ldC);
        return;
    }
    gescal(order, m, n, beta, C, ldC);
    if (sideA==Right) {
        for (IndexType i=0; i<m; ++i) {
            symv(order, upLoA, n, alpha, A, ldA, B+i*ldB, IndexType(1),
                 BETA(1), C+i*ldC, IndexType(1));
        }
    }
    if (sideA==Left) {
        for (IndexType j=0; j<n; ++j) {
            symv(order, upLoA, m, alpha, A, ldA, B+j, ldB,
                 BETA(1), C+j, ldC);
        }
    }
}

template <typename IndexType, typename ALPHA, typename MA, typename MB,
          typename BETA, typename MC>
void
symm(StorageOrder order, Side side, StorageUpLo upLo,
     IndexType m, IndexType n,
     const ALPHA &alpha,
     const MA *A, IndexType ldA,
     const MB *B, IndexType ldB,
     const BETA &beta,
     MC *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("symm_generic");

    symm_generic(order, side, upLo, m, n, alpha, A, ldA, B, ldB, beta, C, ldC);
}

#ifdef HAVE_CBLAS

// ssymm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
symm(StorageOrder order, Side side, StorageUpLo upLo,
     IndexType m, IndexType n,
     float alpha,
     const float *A, IndexType ldA,
     const float *B, IndexType ldB,
     float beta,
     float *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ssymm");

    cblas_ssymm(CBLAS::getCblasType(order),
                CBLAS::getCblasType(side), CBLAS::getCblasType(upLo),
                m, n,
                alpha,
                A, ldA,
                B, ldB,
                beta,
                C, ldC);
}

// dsymm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
symm(StorageOrder order, Side side, StorageUpLo upLo,
     IndexType m, IndexType n,
     double alpha,
     const double *A, IndexType ldA,
     const double *B, IndexType ldB,
     double beta,
     double *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dsymm");

    cblas_dsymm(CBLAS::getCblasType(order),
                CBLAS::getCblasType(side), CBLAS::getCblasType(upLo),
                m, n,
                alpha,
                A, ldA,
                B, ldB,
                beta,
                C, ldC);
}

// csymm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
symm(StorageOrder order, Side side, StorageUpLo upLo,
     IndexType m, IndexType n,
     const ComplexFloat &alpha,
     const ComplexFloat *A, IndexType ldA,
     const ComplexFloat *B, IndexType ldB,
     const ComplexFloat &beta,
     ComplexFloat *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_csymm");

    cblas_csymm(CBLAS::getCblasType(order),
                CBLAS::getCblasType(side), CBLAS::getCblasType(upLo),
                m, n,
                reinterpret_cast<const float *>(&alpha),
                reinterpret_cast<const float *>(A), ldA,
                reinterpret_cast<const float *>(B), ldB,
                reinterpret_cast<const float *>(&beta),
                reinterpret_cast<float *>(C), ldC);
}

// zsymm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
symm(StorageOrder order, Side side, StorageUpLo upLo,
     IndexType m, IndexType n,
     const ComplexDouble &alpha,
     const ComplexDouble *A, IndexType ldA,
     const ComplexDouble *B, IndexType ldB,
     const ComplexDouble &beta,
     ComplexDouble *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zsymm");

    cblas_zsymm(CBLAS::getCblasType(order),
                CBLAS::getCblasType(side), CBLAS::getCblasType(upLo),
                m, n,
                reinterpret_cast<const double *>(&alpha),
                reinterpret_cast<const double *>(A), ldA,
                reinterpret_cast<const double *>(B), ldB,
                reinterpret_cast<const double *>(&beta),
                reinterpret_cast<double *>(C), ldC);
}

#endif // HAVE_CBLAS

#ifdef HAVE_CUBLAS

// ssymm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
symm(StorageOrder order, Side side, StorageUpLo upLo,
      IndexType m, IndexType n,
      const float &alpha,
      const thrust::device_ptr<const float> A, IndexType ldA,
      const thrust::device_ptr<const float> B, IndexType ldB,
      const float &beta,
      thrust::device_ptr<float> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasSsymm");
    
    if (order==RowMajor) {
        side = (side==Left) ? Right : Left;
        upLo = (upLo==Upper) ? Lower : Upper;
        symm(ColMajor, side, upLo, n, m,
             alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
    cublasStatus_t status = cublasSsymm(flens::CudaEnv::getHandle(), CUBLAS::getCublasType(side),
                                        CUBLAS::getCublasType(upLo), 
                                        m, n, &alpha,
                                        A.get(), ldA,
                                        B.get(), ldB,
                                        &beta, C.get(), ldC);
    flens::checkStatus(status);
}

// dsymm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
symm(StorageOrder order, Side side, StorageUpLo upLo,
      IndexType m, IndexType n,
      const double &alpha,
      const thrust::device_ptr<const double> A, IndexType ldA,
      const thrust::device_ptr<const double> B, IndexType ldB,
      const double &beta,
      thrust::device_ptr<double> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasDsymm");
    
    if (order==RowMajor) {
        side = (side==Left) ? Right : Left;
        upLo = (upLo==Upper) ? Lower : Upper;
        symm(ColMajor, side, upLo, n, m,
             alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
    cublasStatus_t status = cublasDsymm(flens::CudaEnv::getHandle(), CUBLAS::getCublasType(side),
                                        CUBLAS::getCublasType(upLo), 
                                        m, n, &alpha,
                                        A.get(), ldA,
                                        B.get(), ldB,
                                        &beta, C.get(), ldC);  
    flens::checkStatus(status);
}
// csymm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
symm(StorageOrder order, Side side, StorageUpLo upLo,
      IndexType m, IndexType n,
      const ComplexFloat &alpha,
      const thrust::device_ptr<const ComplexFloat> A, IndexType ldA,
      const thrust::device_ptr<const ComplexFloat> B, IndexType ldB,
      const ComplexFloat &beta,
      thrust::device_ptr<ComplexFloat> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasCsymm");
    
    if (order==RowMajor) {
        side = (side==Left) ? Right : Left;
        upLo = (upLo==Upper) ? Lower : Upper;
        symm(ColMajor, side, upLo, n, m,
             alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
    cublasStatus_t status = cublasCsymm(flens::CudaEnv::getHandle(), CUBLAS::getCublasType(side),
                                        CUBLAS::getCublasType(upLo),
                                        m, n, reinterpret_cast<const cuFloatComplex*>(&alpha),
                                        reinterpret_cast<const cuFloatComplex*>(A.get()), ldA,
                                        reinterpret_cast<const cuFloatComplex*>(B.get()), ldB,
                                        reinterpret_cast<const cuFloatComplex*>(&beta),
                                        reinterpret_cast<cuFloatComplex*>(C.get()), ldC);
                                        
    flens::checkStatus(status);
}

// zsymm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
symm(StorageOrder order, Side side, StorageUpLo upLo,
      IndexType m, IndexType n,
      const ComplexDouble &alpha,
      const thrust::device_ptr<const ComplexDouble> A, IndexType ldA,
      const thrust::device_ptr<const ComplexDouble> B, IndexType ldB,
      const ComplexDouble &beta,
      thrust::device_ptr<ComplexDouble> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasZsymm");
    
    if (order==RowMajor) {
        side = (side==Left) ? Right : Left;
        upLo = (upLo==Upper) ? Lower : Upper;
        symm(ColMajor, side, upLo, n, m,
             alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
    cublasStatus_t status = cublasZsymm(flens::CudaEnv::getHandle(),  CUBLAS::getCublasType(side),
                                        CUBLAS::getCublasType(upLo), 
                                        m, n, reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                        reinterpret_cast<const cuDoubleComplex*>(A.get()), ldA,
                                        reinterpret_cast<const cuDoubleComplex*>(B.get()), ldB,
                                        reinterpret_cast<const cuDoubleComplex*>(&beta),
                                        reinterpret_cast<cuDoubleComplex*>(C.get()), ldC);
    
    flens::checkStatus(status);
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL3_SYMM_TCC
