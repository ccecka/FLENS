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

#ifndef PLAYGROUND_CXXBLAS_LEVEL1_SWAP_TCC
#define PLAYGROUND_CXXBLAS_LEVEL1_SWAP_TCC 1

#include <cxxblas/cxxblas.h>

namespace cxxblas {

#ifdef HAVE_CLBLAS
// scopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
swap(IndexType n, 
     flens::device_ptr<float, flens::StorageType::OpenCL> x, IndexType incX, 
     flens::device_ptr<float, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clblasScopy");

    cl_int status =CLBLAS_IMPL(Sswap)(n, 
                                      x.get(), x.getOffset(), incX,
                                      y.get(), y.getOffset(), incY,
                                      1, flens::OpenCLEnv::getQueuePtr(),
                                      0, NULL, NULL);

    flens::checkStatus(status); 
}

// dcopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
swap(IndexType n, 
     flens::device_ptr<double, flens::StorageType::OpenCL> x, IndexType incX, 
     flens::device_ptr<double, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clblasDcopy");

    cl_int status = CLBLAS_IMPL(Dswap)(n, 
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status); 
}

// ccopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
swap(IndexType n, 
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX, 
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clblasCswap");

    cl_int status = CLBLAS_IMPL(Cswap)(n, 
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status); 
}

// zcopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
swap(IndexType n, 
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX, 
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clZswap");

    cl_int status = CLBLAS_IMPL(Zswap)(n, 
                                       x.get(), x.getOffset(), incX,
                                       y.get(), y.getOffset(), incY,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status); 
}

#endif // WITH_CLBLAS

#ifdef HAVE_CUBLAS

// sswap
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
swap(IndexType n, 
     flens::device_ptr<float, flens::StorageType::CUDA> x, IndexType incX, 
     flens::device_ptr<float, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasSswap");
    

    cublasStatus_t status = cublasSswap(flens::CudaEnv::getHandle(), n,
                                        x.get(), incX, 
                                        y.get(), incY);
    
    flens::checkStatus(status);
}

// dcopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
swap(IndexType n, 
     flens::device_ptr<double, flens::StorageType::CUDA> x, IndexType incX,
     flens::device_ptr<double, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasDswap");
    

    cublasStatus_t status = cublasDswap(flens::CudaEnv::getHandle(), n,
                                        x.get(), incX, 
                                        y.get(), incY);
    
    flens::checkStatus(status);
}

// ccopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
swap(IndexType n, 
     flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> x, IndexType incX,
     flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasCswap");
    
    
    cublasStatus_t status = cublasCswap(flens::CudaEnv::getHandle(), n, 
                                        reinterpret_cast<cuFloatComplex*>(x.get()), incX, 
                                        reinterpret_cast<cuFloatComplex*>(y.get()), incY);

    
    flens::checkStatus(status);
}

// zcopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
swap(IndexType n, 
     flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> x, IndexType incX,
     flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasZswap");
    
    
    cublasStatus_t status = cublasZswap(flens::CudaEnv::getHandle(), n, 
                                        reinterpret_cast<cuDoubleComplex*>(x.get()), incX, 
                                        reinterpret_cast<cuDoubleComplex*>(y.get()), incY);
    
    flens::checkStatus(status);
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL1_SWAP_TCC
