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

#ifndef CXXBLAS_LEVEL1_COPY_TCC
#define CXXBLAS_LEVEL1_COPY_TCC 1

#include <cxxblas/cxxblas.h>

namespace cxxblas {

template <typename IndexType, typename X, typename Y>
void
copy_generic(IndexType n, const X *x, IndexType incX, Y *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("copy_generic");

    for (IndexType i=0, iX=0, iY=0; i<n; ++i, iX+=incX, iY+=incY) {
        y[iY] = x[iX];
    }
}

template <typename IndexType, typename X, typename Y>
void
copy(IndexType n, const X *x, IndexType incX, Y *y, IndexType incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    copy_generic(n, x, incX, y, incY);
}

#ifdef HAVE_CBLAS

// scopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, const float *x, IndexType incX, float *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_scopy");

    cblas_scopy(n, x, incX, y, incY);
}

// dcopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, const double *x, IndexType incX, double *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_dcopy");

    cblas_dcopy(n, x, incX, y, incY);
}

// ccopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, const ComplexFloat *x, IndexType incX,
     ComplexFloat *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_ccopy");

    cblas_ccopy(n, reinterpret_cast<const float *>(x), incX,
                   reinterpret_cast<float *>(y), incY);

}

// zcopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n, const ComplexDouble *x, IndexType incX,
     ComplexDouble *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zcopy");

    cblas_zcopy(n, reinterpret_cast<const double *>(x), incX,
                   reinterpret_cast<double *>(y), incY);
}

#endif // HAVE_CBLAS

#ifdef HAVE_CUBLAS

template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n,
     const T *x, IndexType incX,
     thrust::device_ptr<T> y, IndexType incY)
{
    if (flens::CudaEnv::isSyncCopyEnabled()) {

        CXXBLAS_DEBUG_OUT("cublasSetVector [sync]");
        cublasStatus_t status = cublasSetVector(n, sizeof(T), x, incX, y.get(), incY);
        flens::checkStatus(status);

    } else {

        CXXBLAS_DEBUG_OUT("cublasSetVector [async]");
        cublasStatus_t status = cublasSetVectorAsync(n, sizeof(T), x, incX, y.get(), incY, flens::CudaEnv::getStream());
        flens::checkStatus(status);

    }
}

template <typename IndexType, typename T>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n,
     const thrust::device_ptr<const T> x, IndexType incX,
     T *y, IndexType incY)
{

    if (flens::CudaEnv::isSyncCopyEnabled()) {

        CXXBLAS_DEBUG_OUT("cublasGetVector [sync]");
        cublasStatus_t status = cublasGetVector(n, sizeof(T), x.get(), incX, y, incY);
        flens::checkStatus(status);

    } else {

        CXXBLAS_DEBUG_OUT("cublasGetVector [async]");
        cublasStatus_t status = cublasGetVectorAsync(n, sizeof(T), x.get(), incX, y, incY, flens::CudaEnv::getStream());
        flens::checkStatus(status);

    }
}

// scopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n,
     const thrust::device_ptr<const float> x, IndexType incX,
     thrust::device_ptr<float> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasScopy");


    cublasStatus_t status = cublasScopy(flens::CudaEnv::getHandle(), n,
                                        x.get(), incX,
                                        y.get(), incY);

    flens::checkStatus(status);
}

// dcopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n,
     const thrust::device_ptr<const double> x, IndexType incX,
     thrust::device_ptr<double> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasDcopy");


    cublasStatus_t status = cublasDcopy(flens::CudaEnv::getHandle(), n,
                                        x.get(), incX,
                                        y.get(), incY);

    flens::checkStatus(status);
}

// ccopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n,
     const thrust::device_ptr<const ComplexFloat> x, IndexType incX,
     thrust::device_ptr<ComplexFloat> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasCcopy");


    cublasStatus_t status = cublasCcopy(flens::CudaEnv::getHandle(), n,
                                        reinterpret_cast<const cuFloatComplex*>(x.get()), incX,
                                        reinterpret_cast<cuFloatComplex*>(y.get()), incY);


    flens::checkStatus(status);
}

// zcopy
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
copy(IndexType n,
     const thrust::device_ptr<const ComplexDouble> x, IndexType incX,
     thrust::device_ptr<ComplexDouble> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasZcopy");


    cublasStatus_t status = cublasZcopy(flens::CudaEnv::getHandle(), n,
                                        reinterpret_cast<const cuDoubleComplex*>(x.get()), incX,
                                        reinterpret_cast<cuDoubleComplex*>(y.get()), incY);

    flens::checkStatus(status);
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1_COPY_TCC
