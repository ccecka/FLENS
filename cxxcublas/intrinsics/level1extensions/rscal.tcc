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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_RSCAL_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_RSCAL_TCC 1

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/intrinsics/auxiliary/auxiliary.h>
#include <playground/cxxblas/intrinsics/includes.h>

namespace cxxblas {

#ifdef USE_INTRINSIC

template <typename T, int N>
inline
typename flens::RestrictTo<flens::IsReal<T>::value,
                           void>::Type
rscal_kernel(const T & alpha, T *y) 
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const int numElements = IntrinsicType::numElements;
    IntrinsicType _y;
    IntrinsicType _alpha(alpha);  

    for (int i=0; i<N; ++i){

        _y.load(y);
        _y = _intrinsic_div(_y, _alpha);
        _y.store(y);
        y+=numElements;
   }

}

template <typename T, int N>
inline
typename flens::RestrictTo<flens::IsComplex<T>::value,
                           void>::Type
rscal_kernel(const T & alpha, T *y) 
{

    using std::real;
    using std::imag;
    using std::abs;

    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType PT;
    typedef Intrinsics<PT, IntrinsicsLevel::SSE> IntrinsicPrimitiveType;
    const int numElements = IntrinsicType::numElements;

    IntrinsicType _y, _tmp;
    IntrinsicPrimitiveType _mr, _den;

    if (abs(real(alpha)) < abs(imag(alpha))) {

        const PT r   = real(alpha)/imag(alpha);
        const PT den = imag(alpha) + r*real(alpha);

        _mr.fill(-r);
        _den.fill(-den);

        for (int i=0; i<N; ++i){
            _y.load(y);

            _tmp = _intrinsic_mul(_mr, _y);
            _y   = _intrinsic_swap_real_imag(_y);
            _tmp = _intrinsic_addsub(_tmp, _y);
            _y   = _intrinsic_div(_tmp, _den);
            _y.store(y);
            y+=numElements;

        }
    } else {
        const PT r   = imag(alpha)/real(alpha);
        const PT den = real(alpha) + r*imag(alpha);

        _mr.fill(-r);
        _den.fill(den);

        for (int i=0; i<N; ++i){
            _y.load(y);

            _tmp = _intrinsic_mul(_mr,_intrinsic_swap_real_imag(_y));
            _y   = _intrinsic_addsub(_y, _tmp);
            _y   = _intrinsic_div(_y, _den);

            _y.store(y);
            y+=numElements;
        }

    }

}

template <typename IndexType, typename T, 
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
rscal_unroller(IndexType length, const T & alpha, T *y) 
{

}

template <typename IndexType, typename T, 
          int N = 16, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
rscal_unroller(IndexType length, const T & alpha, T *y) 
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=length-N*numElements; i+=N*numElements) {

            rscal_kernel<T,N>(alpha, y); 

            y+=N*numElements;

        }
        rscal_unroller<IndexType, T, N/2, false>(length%(N*numElements), alpha, y);

    } else {
        if (length>=N*numElements) {

            rscal_kernel<T,N>(alpha, y); 

            y+=N*numElements;

            length-=N*numElements;
        }
        rscal_unroller<IndexType, T, N/2, false>(length, alpha, y);
    }
}

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
rscal(IndexType n, const T &alpha, T *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("rscal_intrinsics [" INTRINSIC_NAME "]");

    using std::real;
    using std::imag;

    typedef Intrinsics<T, IntrinsicsLevel::SSE>     IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType  PT;

    if (alpha==T(1))
        return;

    if (incY==1) {

        if ((IsComplex<T>::value) && (imag(alpha) == PT(0))) {
            rscal(2*n, real(alpha), reinterpret_cast<PT*>(y), 1);
            return;
        }

        const int numElements = IntrinsicType::numElements;

        IndexType i=0;

	int n_rest = n%numElements;

	if (n_rest>=4) {
	    (*y++) /= alpha; 
	    (*y++) /= alpha;
	    (*y++) /= alpha; 
	    (*y++) /= alpha;
	    n_rest-=4;
	}
	if (n_rest>=2) {
	    (*y++) /= alpha; 
	    (*y++) /= alpha;
	    n_rest-=2;
	}
	if (n_rest==1) { 
	    (*y++) /= alpha;
	}
	rscal_unroller<IndexType, T>(n-n%numElements, alpha, y);


    } else {
        cxxblas::rscal<IndexType, T, T>(n, alpha, y, incY);
    }
}

#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_RSCAL_TCC
