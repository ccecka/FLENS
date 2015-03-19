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

#ifndef CXXBLAS_LEVEL3_SYRK_TCC
#define CXXBLAS_LEVEL3_SYRK_TCC 1

#include <cxxblas/cxxblas.h>

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA,
          typename BETA, typename MC>
void
syrk_generic(StorageOrder order, StorageUpLo upLoC,
             Transpose transA,
             IndexType n, IndexType k,
             const ALPHA &alpha,
             const MA *A, IndexType ldA,
             const BETA &beta,
             MC *C, IndexType ldC)
{
    if (order==ColMajor) {
        upLoC = (upLoC==Upper) ? Lower : Upper;
        transA = Transpose(transA^Trans);
        syrk_generic(RowMajor, upLoC, transA, n, k,
                     alpha, A, ldA, beta, C, ldC);
        return;
    }
    syscal(order, upLoC, n, beta, C, ldC);
    if (k==0) {
        return;
    }
    if (transA==NoTrans) {
        for (IndexType l=0; l<k; ++l) {
            syr(order,  upLoC, n, alpha, A+l, ldA, C, ldC);
        }
    }
    if (transA==Conj) {
        for (IndexType l=0; l<k; ++l) {
            syr(order,  upLoC, n, alpha, A+l, ldA, C, ldC);
        }
    }
    if (transA==Trans) {
        for (IndexType l=0; l<k; ++l) {
            syr(order,  upLoC, n, alpha, A+l*ldA, IndexType(1), C, ldC);
        }
    }
    if (transA==ConjTrans) {
        for (IndexType l=0; l<k; ++l) {
            syr(order,  upLoC, n, alpha, A+l*ldA, IndexType(1), C, ldC);
        }
    }
}

template <typename IndexType, typename ALPHA, typename MA,
          typename BETA, typename MC>
void
syrk(StorageOrder order, StorageUpLo upLo,
     Transpose trans,
     IndexType n, IndexType k,
     const ALPHA &alpha,
     const MA *A, IndexType ldA,
     const BETA &beta,
     MC *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("syrk_generic");

    syrk_generic(order, upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
}

#ifdef HAVE_CBLAS

// ssyrk
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
syrk(StorageOrder order, StorageUpLo upLo,
     Transpose trans,
     IndexType n, IndexType k,
     float alpha,
     const float *A, IndexType ldA,
     float beta,
     float *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ssyrk");

    cblas_ssyrk(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(trans),
                n, k,
                alpha,
                A, ldA,
                beta,
                C, ldC);
}

// dsyrk
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
syrk(StorageOrder order, StorageUpLo upLo,
     Transpose trans,
     IndexType n, IndexType k,
     double alpha,
     const double *A, IndexType ldA,
     double beta,
     double *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dsyrk");

    cblas_dsyrk(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(trans),
                n, k,
                alpha,
                A, ldA,
                beta,
                C, ldC);
}

// csyrk
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
syrk(StorageOrder order, StorageUpLo upLo,
     Transpose trans,
     IndexType n, IndexType k,
     const ComplexFloat &alpha,
     const ComplexFloat *A, IndexType ldA,
     const ComplexFloat &beta,
     ComplexFloat *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_csyrk");

    cblas_csyrk(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(trans),
                n, k,
                reinterpret_cast<const float *>(&alpha),
                reinterpret_cast<const float *>(A), ldA,
                reinterpret_cast<const float *>(&beta),
                reinterpret_cast<float *>(C), ldC);
}

// zsyrk
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
syrk(StorageOrder order, StorageUpLo upLo,
     Transpose trans,
     IndexType n, IndexType k,
     const ComplexDouble &alpha,
     const ComplexDouble *A, IndexType ldA,
     const ComplexDouble &beta,
     ComplexDouble *C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zsyrk");

    cblas_zsyrk(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
                CBLAS::getCblasType(trans),
                n, k,
                reinterpret_cast<const double *>(&alpha),
                reinterpret_cast<const double *>(A), ldA,
                reinterpret_cast<const double *>(&beta),
                reinterpret_cast<double *>(C), ldC);
}

#endif // HAVE_CBLAS

#ifdef HAVE_CUBLAS

// ssyrk
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
syrk(StorageOrder order, StorageUpLo upLo,
      Transpose trans,
      IndexType n, IndexType k,
      const float &alpha,
      const flens::device_ptr<const float, flens::StorageType::CUDA> A, IndexType ldA,
      const float &beta,
      flens::device_ptr<float, flens::StorageType::CUDA> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasSsyrk");
      
    if (order==RowMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
        trans = Transpose(trans^ConjTrans);
        syrk(ColMajor, upLo, trans, n, k,
              conjugate(alpha), A, ldA, 
              beta, C, ldC);
        return;
    }
   
      
    cublasStatus_t status = cublasSsyrk(flens::CudaEnv::getHandle(), CUBLAS::getCublasType(upLo),
                                        CUBLAS::getCublasType(trans), n, k,
                                        &alpha,
                                        A.get(), ldA,
                                        &beta,
                                        C.get(), ldC);
    
    flens::checkStatus(status);
}

// dsyrk
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
syrk(StorageOrder order, StorageUpLo upLo,
      Transpose trans,
      IndexType n, IndexType k,
      const double &alpha,
      const flens::device_ptr<const double, flens::StorageType::CUDA> A, IndexType ldA,
      const double &beta,
      flens::device_ptr<double, flens::StorageType::CUDA> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasDsyrk");
      
    if (order==RowMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
        trans = Transpose(trans^ConjTrans);
        syrk(ColMajor, upLo, trans, n, k,
              conjugate(alpha), A, ldA,
              beta, C, ldC);
        return;
    }
   
      
    cublasStatus_t status = cublasDsyrk(flens::CudaEnv::getHandle(), CUBLAS::getCublasType(upLo),
                                        CUBLAS::getCublasType(trans), n, k,
                                        &alpha,
                                        A.get(), ldA,
                                        &beta,
                                        C.get(), ldC);
    
    flens::checkStatus(status);
}

// csyrk
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
syrk(StorageOrder order, StorageUpLo upLo,
      Transpose trans,
      IndexType n, IndexType k,
      const ComplexFloat &alpha,
      const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> A, IndexType ldA,
      const ComplexFloat &beta,
      flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasCsyrk");
      
    if (order==RowMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
        trans = Transpose(trans^ConjTrans);
        syrk(ColMajor, upLo, trans, n, k,
              conjugate(alpha), A, ldA, 
              beta, C, ldC);
        return;
    }
   
      
    cublasStatus_t status = cublasCsyrk(flens::CudaEnv::getHandle(), CUBLAS::getCublasType(upLo),
                                        CUBLAS::getCublasType(trans), n, k,
                                        reinterpret_cast<const cuFloatComplex*>(&alpha),
                                        reinterpret_cast<const cuFloatComplex*>(A.get()), ldA,
                                        reinterpret_cast<const cuFloatComplex*>(&beta),
                                        reinterpret_cast<cuFloatComplex*>(C.get()), ldC);
    
    flens::checkStatus(status);
}

// zsyrk
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
syrk(StorageOrder order, StorageUpLo upLo,
      Transpose trans,
      IndexType n, IndexType k,
      const ComplexDouble &alpha,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> A, IndexType ldA,
      const ComplexDouble &beta,
      flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasZsyrk");
      
    if (order==RowMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
        trans = Transpose(trans^ConjTrans);
        syrk(ColMajor, upLo, trans, n, k,
              conjugate(alpha), A, ldA,
              beta, C, ldC);
        return;
    }
   
      
    cublasStatus_t status = cublasZsyrk(flens::CudaEnv::getHandle(), CUBLAS::getCublasType(upLo),
                                        CUBLAS::getCublasType(trans), n, k,
                                        reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                        reinterpret_cast<const cuDoubleComplex*>(A.get()), ldA,
                                        reinterpret_cast<const cuDoubleComplex*>(&beta),
                                        reinterpret_cast<cuDoubleComplex*>(C.get()), ldC);
    
    flens::checkStatus(status);
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL3_SYRK_TCC
