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

#ifndef PLAYGROUND_CXXBLAS_LEVEL3_GEMM_TCC
#define PLAYGROUND_CXXBLAS_LEVEL3_GEMM_TCC 1

#include <cxxblas/cxxblas.h>

namespace cxxblas {
  
#ifdef HAVE_CLBLAS

// sgemm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemm(StorageOrder order, Transpose transA, Transpose transB,
     IndexType m, IndexType n, IndexType k,
     float alpha,
     const flens::device_ptr<const float, flens::StorageType::OpenCL> A, IndexType ldA,
     const flens::device_ptr<const float, flens::StorageType::OpenCL> B, IndexType ldB,
     float beta,
     flens::device_ptr<float, flens::StorageType::OpenCL> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasSgemm");

    cl_int status = CLBLAS_IMPL_EX(Sgemm)(CLBLAS::getClblasType(order), 
                                          CLBLAS::getClblasType(transA),  CLBLAS::getClblasType(transB),
                                          m,  n, k,
                                          alpha,
                                          A.get(), A.getOffset(), ldA,
                                          B.get(), B.getOffset(), ldB,
                                          beta,
                                          C.get(), C.getOffset(), ldC,
                                          1, flens::OpenCLEnv::getQueuePtr(),
                                          0, NULL, NULL);

    flens::checkStatus(status);
}

// dgemm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemm(StorageOrder order, Transpose transA, Transpose transB,
     IndexType m, IndexType n, IndexType k,
     double alpha,
     const flens::device_ptr<const double, flens::StorageType::OpenCL> A, IndexType ldA,
     const flens::device_ptr<const double, flens::StorageType::OpenCL> B, IndexType ldB,
     double beta,
     flens::device_ptr<double, flens::StorageType::OpenCL> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasDgemm");

    cl_int status = CLBLAS_IMPL_EX(Dgemm)(CLBLAS::getClblasType(order), 
                                          CLBLAS::getClblasType(transA),  CLBLAS::getClblasType(transB),
                                          m,  n, k,
                                          alpha,
                                          A.get(), A.getOffset(), ldA,
                                          B.get(), B.getOffset(), ldB,
                                          beta,
                                          C.get(), C.getOffset(), ldC,
                                          1, flens::OpenCLEnv::getQueuePtr(),
                                          0, NULL, NULL);

    flens::checkStatus(status);
}

// cgemm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemm(StorageOrder order, Transpose transA, Transpose transB,
     IndexType m, IndexType n, IndexType k,
     ComplexFloat alpha,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> A, IndexType ldA,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> B, IndexType ldB,
     ComplexFloat beta,
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasCgemm");
    
    cl_int status = CLBLAS_IMPL_EX(Cgemm)(CLBLAS::getClblasType(order), 
                                          CLBLAS::getClblasType(transA),  CLBLAS::getClblasType(transB),
                                          m,  n, k,
                                          *(reinterpret_cast<cl_float2*>(&alpha)),
                                          A.get(), A.getOffset(), ldA,
                                          B.get(), B.getOffset(), ldB,
                                          *(reinterpret_cast<cl_float2*>(&beta)),
                                          C.get(), C.getOffset(), ldC,
                                          1, flens::OpenCLEnv::getQueuePtr(),
                                          0, NULL, NULL);

    flens::checkStatus(status);
}

// zgemm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemm(StorageOrder order, Transpose transA, Transpose transB,
      IndexType m, IndexType n, IndexType k,
      ComplexDouble alpha,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> A, IndexType ldA,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> B, IndexType ldB,
      ComplexDouble beta,
      flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasZgemm");
    
    cl_int status = CLBLAS_IMPL_EX(Zgemm)(CLBLAS::getClblasType(order), 
                                          CLBLAS::getClblasType(transA),  CLBLAS::getClblasType(transB),
                                          m,  n, k,
                                          *(reinterpret_cast<cl_double2*>(&alpha)),
                                          A.get(), A.getOffset(), ldA,
                                          B.get(), B.getOffset(), ldB,
                                          *(reinterpret_cast<cl_double2*>(&beta)),
                                          C.get(), C.getOffset(), ldC,
                                          1, flens::OpenCLEnv::getQueuePtr(),
                                          0, NULL, NULL);


    flens::checkStatus(status);
}

#endif // HAVE_CLBLAS

#ifdef HAVE_CUBLAS

// sgemm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemm(StorageOrder order, Transpose transA, Transpose transB,
      IndexType m, IndexType n, IndexType k,
      float alpha,
      const flens::device_ptr<const float, flens::StorageType::CUDA> A, IndexType ldA,
      const flens::device_ptr<const float, flens::StorageType::CUDA> B, IndexType ldB,
      float beta,
      flens::device_ptr<float, flens::StorageType::CUDA> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasSgemm");
    
    if (order==RowMajor) {
        gemm(ColMajor, transB, transA,
             n, m, k, alpha,
             B, ldB, A, ldA,
             beta,
             C, ldC);
        return;
    }

    cublasStatus_t status = cublasSgemm(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(transA), 
                                        CUBLAS::getCublasType(transB),
                                        m,  n, k,
                                        &alpha,
                                        A.get(), ldA,
                                        B.get(), ldB,
                                        &beta,
                                        C.get(), ldC);
    
    flens::checkStatus(status);
}

// dgemm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemm(StorageOrder order, Transpose transA, Transpose transB,
      IndexType m, IndexType n, IndexType k,
      double alpha,
      const flens::device_ptr<const double, flens::StorageType::CUDA> A, IndexType ldA,
      const flens::device_ptr<const double, flens::StorageType::CUDA> B, IndexType ldB,
      double beta,
      flens::device_ptr<double, flens::StorageType::CUDA> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasDgemm");
    
    if (order==RowMajor) {
        gemm(ColMajor, transB, transA,
             n, m, k, alpha,
             B, ldB, A, ldA,
             beta,
             C, ldC);
        return;
    }

    cublasStatus_t status = cublasDgemm(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(transA), 
                                        CUBLAS::getCublasType(transB),
                                        m,  n, k,
                                        &alpha,
                                        A.get(), ldA,
                                        B.get(), ldB,
                                        &beta,
                                        C.get(), ldC);
    
    flens::checkStatus(status);
}

// cgemm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemm(StorageOrder order, Transpose transA, Transpose transB,
      IndexType m, IndexType n, IndexType k,
      ComplexFloat alpha,
      const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> A, IndexType ldA,
      const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> B, IndexType ldB,
      ComplexFloat beta,
      flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasCgemm");
    
    if (order==RowMajor) {
        gemm(ColMajor, transB, transA,
             n, m, k, alpha,
             B, ldB, A, ldA,
             beta,
             C, ldC);
        return;
    }
    
    cublasStatus_t status = cublasCgemm(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(transA), 
                                        CUBLAS::getCublasType(transB),
                                        m, n, k,
                                        reinterpret_cast<const cuFloatComplex*>(&alpha),
                                        reinterpret_cast<const cuFloatComplex*>(A.get()), ldA,
                                        reinterpret_cast<const cuFloatComplex*>(B.get()), ldB,
                                        reinterpret_cast<const cuFloatComplex*>(&beta),
                                        reinterpret_cast<cuFloatComplex*>(C.get()), ldC);
    
    flens::checkStatus(status);
}

// zgemm
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemm(StorageOrder order, Transpose transA, Transpose transB,
      IndexType m, IndexType n, IndexType k,
      ComplexDouble alpha,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> A, IndexType ldA,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> B, IndexType ldB,
      ComplexDouble beta,
      flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> C, IndexType ldC)
{
    CXXBLAS_DEBUG_OUT("cublasZgemm");
    
    if (order==RowMajor) {
        gemm(ColMajor, transB, transA,
             n, m, k, alpha,
             B, ldB, A, ldA,
             beta,
             C, ldC);
        return;
    }

    cublasStatus_t status = cublasZgemm(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(transA), 
                                        CUBLAS::getCublasType(transB),
                                        m, n, k,
                                        reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                        reinterpret_cast<const cuDoubleComplex*>(A.get()), ldA,
                                        reinterpret_cast<const cuDoubleComplex*>(B.get()), ldB,
                                        reinterpret_cast<const cuDoubleComplex*>(&beta),
                                        reinterpret_cast<cuDoubleComplex*>(C.get()), ldC);
    
    flens::checkStatus(status);
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL3_GEMM_TCC