/*
 *   Copyright (c) 2014, Klaus Pototzky
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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_SCAL_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_SCAL_TCC 1

#include <algorithm>    

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/intrinsics/auxiliary/auxiliary.h>
#include <playground/cxxblas/intrinsics/includes.h>

namespace cxxblas {

#ifdef USE_INTRINSIC


template <typename T, int N>
inline
typename flens::RestrictTo<flens::IsReal<T>::value,
                           void>::Type
scal_kernel(const T & alpha, T *y) 
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const int numElements = IntrinsicType::numElements;
    IntrinsicType _y;
    IntrinsicType _alpha(alpha);  

    for (int i=0; i<N; ++i){

        _y.load(y);
        _y = _intrinsic_mul(_alpha, _y);
        _y.store(y);
        y+=numElements;
   }

}

template <typename T, int N>
inline
typename flens::RestrictTo<flens::IsComplex<T>::value,
                           void>::Type
scal_kernel(const T & alpha, T *y) 
{
    using std::real;
    using std::imag;

    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType PT;
    typedef Intrinsics<PT, IntrinsicsLevel::SSE> IntrinsicPrimitiveType;
    const int numElements = IntrinsicType::numElements;

    IntrinsicType _tmp, _y;
    IntrinsicPrimitiveType _real_alpha(real(alpha));
    IntrinsicPrimitiveType _imag_alpha(imag(alpha)); 

    for (int i=0; i<N; ++i){

        _y.loadu(y);
        _tmp = _intrinsic_mul(_real_alpha, _y);
        _y = _intrinsic_swap_real_imag(_y);
        _y = _intrinsic_mul(_imag_alpha, _y);
        _y = _intrinsic_addsub(_tmp, _y);
        _y.store(y);
        y+=numElements;
   }
}

template <typename IndexType, typename T, 
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
scal_unroller(IndexType length, const T & alpha, T *y) 
{

}

template <typename IndexType, typename T, 
          int N = 16, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
scal_unroller(IndexType length, const T & alpha, T *y) 
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=length-N*numElements; i+=N*numElements) {

            scal_kernel<T,N>(alpha, y); 

            y+=N*numElements;

        }
        scal_unroller<IndexType, T, N/2, false>(length%(N*numElements), alpha, y);

    } else {
        if (length>=N*numElements) {

            scal_kernel<T,N>(alpha, y); 

            y+=N*numElements;

            length-=N*numElements;
        }
        scal_unroller<IndexType, T, N/2, false>(length, alpha, y);
    }
}


template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
scal(IndexType n, const T &alpha, T *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("scal_intrinsics [" INTRINSIC_NAME "]");

    using std::real;
    using std::imag;
    using std::fill;

    typedef Intrinsics<T, IntrinsicsLevel::SSE>     IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType  PT;
    typedef Intrinsics<PT, IntrinsicsLevel::SSE>    IntrinsicPrimitiveType;

    if (alpha==T(1))
        return;

    if (incY==1) {

        if (imag(alpha) == PT(0) && IsComplex<T>::value) {
            scal(2*n, real(alpha), reinterpret_cast<PT*>(y), 1);
            return;
        }

        const int numElements = IntrinsicType::numElements;

        IndexType i=0;

        if (alpha==T(0)) {
            const T zero(0);
            std::fill_n(y, n, zero);

        } else {
	    IndexType i=0;

	    int n_rest = n%numElements;

	    if (n_rest>=2) {
	        (*y++) *= alpha; 
	        (*y++) *= alpha;
	        n_rest-=2;
	    }
	    if (n_rest==1) { 
	        (*y++) *= alpha;
	    }

	    scal_unroller<IndexType, T>(n-n%numElements, alpha, y);
        }
    } else {
        cxxblas::scal<IndexType, T, T>(n, alpha, y, incY);
    }
}

#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_SCAL_TCC
