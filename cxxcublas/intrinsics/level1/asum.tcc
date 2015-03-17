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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_ASUM_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_ASUM_TCC 1

#include <algorithm>    

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/intrinsics/auxiliary/auxiliary.h>
#include <playground/cxxblas/intrinsics/includes.h>

namespace cxxblas {

#ifdef USE_INTRINSIC

template <typename T, typename S, int N>
inline
typename flens::RestrictTo<flens::IsReal<T>::value,
                           void>::Type
asum_kernel(const T *x, S &_result) 
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const int numElements = IntrinsicType::numElements;
    IntrinsicType _x;

    for (int i=0; i<N; ++i) {
        _x.load(x);
        _result = _intrinsic_add(_result, _abs(_x));
        x+=numElements;
   }
}


template <typename IndexType, typename T, typename S,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
asum_unroller(IndexType length, const T *x, S &_result)
{

}

template <typename IndexType, typename T, typename S,
          int N = 16, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
asum_unroller(IndexType length, const T *x, S &_result) 
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {
        _result.setZero();
        for (IndexType i=0; i<=length-N*numElements; i+=N*numElements) {

            asum_kernel<T,IntrinsicType,N>(x, _result); 

            x+=N*numElements; 

        }
        asum_unroller<IndexType, T, IntrinsicType, N/2, false>(length%(N*numElements), x, _result);


    } else {
        if (length>=N*numElements) {

            asum_kernel<T,IntrinsicType,N>(x, _result); 

            x+=N*numElements; 

            length-=N*numElements;
        }
        asum_unroller<IndexType, T, IntrinsicType, N/2, false>(length, x, _result);
    }
}


template <typename IndexType, typename T>
inline
typename flens::RestrictTo<flens::IsReal<T>::value && 
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
asum(IndexType n, const T *y, IndexType incY, T &absSum)
{
    CXXBLAS_DEBUG_OUT("asum_intrinsics [real, " INTRINSIC_NAME "]");

    using std::abs;

    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL>     IntrinsicType;

    if (incY==1) {

        const int numElements = IntrinsicType::numElements;

        IndexType i=0;
        absSum = T(0);

	int n_rest = n%numElements;

        if (n_rest>=4) {
	    absSum += abs(*y++);
	    absSum += abs(*y++);
	    absSum += abs(*y++);
	    absSum += abs(*y++);
	    n_rest-=4;
        }

        if (n_rest>=2) {
	    absSum += abs(*y++);
	    absSum += abs(*y++);
	    n_rest-=2;
        }
        if (n_rest==1) { 
	    absSum += abs(*y++);
	}

        IntrinsicType _result;
	asum_unroller<IndexType, T>(n-n%numElements, y, _result);
        absSum += _intrinsic_hsum(_result);

    } else {
        cxxblas::asum<IndexType, T, T>(n, y, incY, absSum);
    }
}

template <typename IndexType, typename T>
inline
typename flens::RestrictTo<flens::IsReal<T>::value && 
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
asum(IndexType n, const std::complex<T> *y, IndexType incY, T &absSum)
{
    CXXBLAS_DEBUG_OUT("asum_intrinsics [complex, " INTRINSIC_NAME "]");

    if (incY==1) {
        asum(2*n, reinterpret_cast<const T*>(y), 1, absSum);
    } else {
        cxxblas::asum<IndexType, std::complex<T>, T>(n, y, incY, absSum);
    }
}

#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_ASUM_TCC
