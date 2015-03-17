/*
 *   Copyright (c) 2012, Klaus Pototzky
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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_CLASSES_FUNCTIONS_HSUM_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_CLASSES_FUNCTIONS_HSUM_TCC 1

#include <playground/cxxblas/intrinsics/includes.h>

#ifdef HAVE_SSE

//--- hsum
float
inline _intrinsic_hsum(const Intrinsics<float, IntrinsicsLevel::SSE> &x)
{
    __m128 tmp = _mm_hadd_ps(x.get(),x.get());
    tmp = _mm_hadd_ps(tmp,tmp);
    return _mm_cvtss_f32(tmp);
}

double
inline _intrinsic_hsum(const Intrinsics<double, IntrinsicsLevel::SSE> &x)
{
    Intrinsics<double, IntrinsicsLevel::SSE> tmp = _mm_hadd_pd(x.get(),x.get());
    return _mm_cvtsd_f64(tmp.get());
}

std::complex<float>
inline _intrinsic_hsum(const Intrinsics<std::complex<float>, IntrinsicsLevel::SSE> &x)
{
   Intrinsics<std::complex<float>, IntrinsicsLevel::SSE> tmp =  _mm_castpd_ps(_mm_permute_pd(_mm_castps_pd(x.get()),1));
   tmp = _intrinsic_add(tmp, x);
   
   std::complex<float> result;
   double *presult = reinterpret_cast<double*>(&result);
   *presult = _mm_cvtsd_f64(_mm_castps_pd(tmp.get()));
   return result;
}

std::complex<double>
inline _intrinsic_hsum(const Intrinsics<std::complex<double>, IntrinsicsLevel::SSE> &x)
{
    std::complex<double> tmp;
    _mm_storeu_pd(reinterpret_cast<double*>(&tmp), x.get());
    return tmp;
}

#endif // HAVE_SSE



#ifdef HAVE_AVX

//--- hsum
float
inline _intrinsic_hsum(const Intrinsics<float, IntrinsicsLevel::AVX> &x)
{
    Intrinsics<float, IntrinsicsLevel::AVX> tmp = _mm256_hadd_ps(x.get(),x.get());
    tmp = _mm256_hadd_ps(tmp.get(),tmp.get());
    Intrinsics<float, IntrinsicsLevel::SSE> tmpSSE = _mm256_extractf128_ps(tmp.get(),1);
    tmpSSE = _mm_add_ss(_mm256_castps256_ps128(tmp.get()),tmpSSE.get());
    return _mm_cvtss_f32(tmpSSE.get()); 

}

double
inline _intrinsic_hsum(const Intrinsics<double, IntrinsicsLevel::AVX> &x)
{
    Intrinsics<double, IntrinsicsLevel::AVX> tmp = _mm256_hadd_pd(x.get(),x.get());
    Intrinsics<double, IntrinsicsLevel::SSE> tmpSSE = _mm256_extractf128_pd(tmp.get(),1);
    tmpSSE = _mm_add_sd(_mm256_castpd256_pd128(tmp.get()),tmpSSE.get());
    return _mm_cvtsd_f64(tmpSSE.get()); 
}

std::complex<float>
inline _intrinsic_hsum(const Intrinsics<std::complex<float>, IntrinsicsLevel::AVX> &x)
{ 
    __m128 tmp0 = _mm256_extractf128_ps(x.get(),0);
    __m128 tmp1 = _mm256_extractf128_ps(x.get(),1);

    Intrinsics<std::complex<float>, IntrinsicsLevel::SSE> tmp = _mm_add_ps(tmp0, tmp1);

    return _intrinsic_hsum(tmp); 
}

std::complex<double>
inline _intrinsic_hsum(const Intrinsics<std::complex<double>, IntrinsicsLevel::AVX> &x)
{
    __m128d tmp0 = _mm256_extractf128_pd(x.get(),0);
    __m128d tmp1 = _mm256_extractf128_pd(x.get(),1);

    Intrinsics<std::complex<double>, IntrinsicsLevel::SSE> tmp = _mm_add_pd(tmp0, tmp1);

    return _intrinsic_hsum(tmp);

}

#endif // HAVE_AVX

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_CLASSES_FUNCTIONS_HSUM_
