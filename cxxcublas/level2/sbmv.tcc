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

#ifndef PLAYGROUND_CXXBLAS_LEVEL2_SBMV_TCC
#define PLAYGROUND_CXXBLAS_LEVEL2_SBMV_TCC 1

#include <complex>
#include <cxxblas/cxxblas.h>

namespace cxxblas {
  
#ifdef HAVE_CLBLAS

// sgbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
sbmv(StorageOrder order, StorageUpLo upLo,
      IndexType n, IndexType k,
      float alpha,
      const flens::device_ptr<const float, flens::StorageType::OpenCL> A, IndexType ldA,
      const flens::device_ptr<const float, flens::StorageType::OpenCL> x, IndexType incX,
      float beta,
      flens::device_ptr<float, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasSsbmv");
      
    cl_int status = CLBLAS_IMPL(Ssbmv)(CLBLAS::getClblasType(order),
                                       CLBLAS::getClblasType(upLo),
                                       n, k,
                                       alpha,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX,
                                       beta,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

// sgbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
sbmv(StorageOrder order, StorageUpLo upLo,
      IndexType n, IndexType k,
      double alpha,
      const flens::device_ptr<const double, flens::StorageType::OpenCL> A, IndexType ldA,
      const flens::device_ptr<const double, flens::StorageType::OpenCL> x, IndexType incX,
      double beta,
      flens::device_ptr<double, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasDsbmv");
      
    cl_int status = CLBLAS_IMPL(Dsbmv)(CLBLAS::getClblasType(order), 
                                       CLBLAS::getClblasType(upLo),
                                       n, k,
                                       alpha,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX,
                                       beta,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
}

#endif 

#ifdef HAVE_CUBLAS

// csbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
sbmv(StorageOrder order, StorageUpLo upLo,
      IndexType n, IndexType k,
      float alpha,
      const flens::device_ptr<const float, flens::StorageType::CUDA> A, IndexType ldA,
      const flens::device_ptr<const float, flens::StorageType::CUDA> x, IndexType incX,
      float beta,
      flens::device_ptr<float, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasSsbmv");
    
    if (order==RowMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
        sbmv(ColMajor, upLo, n, k, alpha, A, ldA,
             x, incX, beta, y, incY);
        return;
    }
    
    cublasStatus_t status = cublasSsbmv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo),
                                        n, k,
                                        &alpha,
                                        A.get(), ldA,
                                        x.get(), incX,
                                        &beta,
                                        y.get(), incY);
    
    flens::checkStatus(status);
}

// zsbmv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
sbmv(StorageOrder order, StorageUpLo upLo,
      IndexType n, IndexType k,
      double alpha,
      const flens::device_ptr<const double, flens::StorageType::CUDA> A, IndexType ldA,
      const flens::device_ptr<const double, flens::StorageType::CUDA> x, IndexType incX,
      double beta,
      flens::device_ptr<double, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasDsbmv");
    
    if (order==RowMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
        sbmv(ColMajor, upLo, n, k, alpha, A, ldA,
             x, incX, beta, y, incY);
        return;
    }
    
    cublasStatus_t status = cublasDsbmv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo),
                                        n, k,
                                        &alpha,
                                        A.get(), ldA,
                                        x.get(), incX,
                                        &beta,
                                        y.get(), incY);
    
    flens::checkStatus(status);
  
}

#endif // HAVE_CUBLAS


} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL2_SBMV_TCC
