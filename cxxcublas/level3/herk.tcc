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

#ifndef PLAYGROUND_CXXBLAS_LEVEL3_HERK_TCC
#define PLAYGROUND_CXXBLAS_LEVEL3_HERK_TCC 1

#include <cxxblas/cxxblas.h>

namespace cxxblas {

#ifdef HAVE_CLBLAS

// cherk
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
herk(StorageOrder order, StorageUpLo upLo,
      Transpose trans,
      IndexType n, IndexType k,
      float alpha,
      const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> A, IndexType ldA,
      float beta,
      flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasCherk");

    cl_int status = CLBLAS_IMPL(Cherk)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo), 
                                       CLBLAS::getClblasType(trans), n, k,
                                       *(reinterpret_cast<const cl_float2*>(&alpha)),
                                        A.get(), A.getOffset(), ldA,
                                        beta,
                                        C.get(), C.getOffset(), ldC,
                                        1, flens::OpenCLEnv::getQueuePtr(),
                                         0, NULL, NULL);

    flens::checkStatus(status); 
}

// zherk
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
herk(StorageOrder order, StorageUpLo upLo,
      Transpose trans,
      IndexType n, IndexType k,
      double alpha,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> A, IndexType ldA,
      double beta,
      flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasZherk");

    cl_int status = CLBLAS_IMPL(Zherk)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo), 
                                       CLBLAS::getClblasType(trans), n, k,
                                       alpha,
                                       A.get(), A.getOffset(), ldA,
                                       beta,
                                       C.get(), C.getOffset(), ldC,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status); 
}

#endif // HAVE_CLBLAS

#ifdef HAVE_CUBLAS

// cherk
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
herk(StorageOrder order, StorageUpLo upLo,
      Transpose trans,
      IndexType n, IndexType k,
      float alpha,
      const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> A, IndexType ldA,
      const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> B, IndexType ldB,
      float beta,
      flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasZherk");
      
    if (order==RowMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
        trans = Transpose(trans^ConjTrans);
        herk(ColMajor, upLo, trans, n, k,
              conjugate(alpha), A, ldA, 
              beta, C, ldC);
        return;
    }
   
      
    cublasStatus_t status = cublasZherk(flens::CudaEnv::getHandle(), CUBLAS::getCublasType(upLo),
                                        CUBLAS::getCublasType(trans), n, k,
                                        &alpha,
                                        reinterpret_cast<const cuFloatComplex*>(A.get()), ldA,
                                        &beta,
                                        reinterpret_cast<cuFloatComplex*>(C.get()), ldC);
    
    flens::checkStatus(status);
}

// zherk
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
herk(StorageOrder order, StorageUpLo upLo,
      Transpose trans,
      IndexType n, IndexType k,
      double alpha,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> A, IndexType ldA,
      double beta,
      flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasZherk");
      
    if (order==RowMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
        trans = Transpose(trans^ConjTrans);
        herk(ColMajor, upLo, trans, n, k,
              conjugate(alpha), A, ldA,
              beta, C, ldC);
        return;
    }
   
      
    cublasStatus_t status = cublasZherk(flens::CudaEnv::getHandle(), CUBLAS::getCublasType(upLo),
                                        CUBLAS::getCublasType(trans), n, k,
                                        &alpha,
                                        reinterpret_cast<const cuDoubleComplex*>(A.get()), ldA,
                                        &beta,
                                        reinterpret_cast<cuDoubleComplex*>(C.get()), ldC);
    
    flens::checkStatus(status);
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL3_HERK_TCC

