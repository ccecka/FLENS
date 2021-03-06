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

#ifndef CXXBLAS_LEVEL1_AXPBY_TCC
#define CXXBLAS_LEVEL1_AXPBY_TCC 1

#include <cxxstd/cstdio.h>
#include <cxxblas/cxxblas.h>

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename X,
          typename BETA, typename Y>
void
axpby(IndexType n, const ALPHA &alpha, const X *x, IndexType incX,
                   const BETA &beta, Y *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("axpby_generic");

    scal(n, beta, y, incY);
    axpy(n, alpha, x, incX, y, incY);
}

#ifdef HAVE_CBLAS_AXPBY

// saxpby
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
axpby(IndexType n, const float &alpha, const float *x, IndexType incX,
      const float &beta, float *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_saxpby");

    BLAS_EXT(saxpby)(n, alpha, x, incX, beta, y, incY);
}

// daxpby
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
axpby(IndexType n, const double &alpha, const double *x, IndexType incX,
      const double &beta, double *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_daxpby");

    BLAS_EXT(daxpby)(n, alpha, x, incX, beta, y, incY);
}

// caxpby
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
axpby(IndexType n, const ComplexFloat &alpha,
      const ComplexFloat *x, IndexType incX,
      const ComplexFloat &beta, ComplexFloat *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_caxpby");

    BLAS_EXT(caxpby)(n, reinterpret_cast<const float *>(&alpha),
                        reinterpret_cast<const float *>(x), incX,
                        reinterpret_cast<const float *>(&beta),
                        reinterpret_cast<float *>(y), incY);
}

// zaxpby
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
axpby(IndexType n, const ComplexDouble &alpha,
      const ComplexDouble *x, IndexType incX,
      const ComplexDouble &beta, ComplexDouble *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zaxpby");

    BLAS_EXT(zaxpby)(n, reinterpret_cast<const double *>(&alpha),
                        reinterpret_cast<const double *>(x), incX,
                        reinterpret_cast<const double *>(&beta),
                        reinterpret_cast<double *>(y), incY);
}

#endif // HAVE_CBLAS_AXPBY

#ifdef HAVE_CUBLAS

template <typename TALPHA, typename TBETA, typename TX, typename TY>
struct axpby_func {
  const TALPHA a;
  const TBETA b;

  axpby_func(const TALPHA& _a, const TBETA& _b)
      : a(_a), b(_b) {}

  __device__
  TY operator()(const TX& tx, const TY& ty) const {
    return a * tx + b * ty;
  }
};


template <typename IndexType, typename ALPHA, typename X,
          typename BETA, typename Y>
void
axpby(IndexType n,
      const ALPHA &alpha, const thrust::device_ptr<X> x, IndexType incX,
      const BETA &beta, thrust::device_ptr<Y> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("axpby_generic [cuda]");

    typedef typename ThrustType<X>::Type      TX;
    typedef typename ThrustType<Y>::Type      TY;
    typedef typename ThrustType<ALPHA>::Type  TALPHA;
    typedef typename ThrustType<BETA>::Type   TBETA;

    StridedRange<const TX*> xr(reinterpret_cast<const TX*>(x.get()),
                               reinterpret_cast<const TX*>(x.get()) + n*incX,
                               incX);
    StridedRange<      TY*> yr(reinterpret_cast<      TY*>(y.get()),
                               reinterpret_cast<      TY*>(y.get()) + n*incY,
                               incY);

    thrust::transform(thrust::cuda::par.on(CublasEnv::stream()),
                      xr.begin(), xr.end(),
                      yr.begin(),
                      yr.begin(),
                      axpby_func<TALPHA,TBETA,TX,TY>(alpha,beta));
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1_AXPBY_TCC
