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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_ACXPY_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_ACXPY_TCC 1

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/intrinsics/auxiliary/auxiliary.h>
#include <playground/cxxblas/intrinsics/includes.h>
#include <playground/cxxblas/intrinsics/level1extensions/acxpy.h>

namespace cxxblas {

#ifdef USE_INTRINSIC

template <typename T, int N>
inline
typename flens::RestrictTo<flens::IsComplex<T>::value,
                           void>::Type
ccopy_kernel(const T *x, T *y) 
{
    using std::real;
    using std::imag;

    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const int numElements = IntrinsicType::numElements;

    IntrinsicType _x, _y;

    for (int i=0; i<N; ++i){
        _x.load(x);
        _y = _conj(_x);
        _y.store(y);
        x+=numElements;
        y+=numElements;
   }
}

template <typename IndexType, typename T, 
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value &&
                           flens::IsComplex<T>::value,
                           void>::Type
ccopy_unroller(IndexType length, const T *x, T *y) 
{

}

template <typename IndexType, typename T, 
          int N = 16, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value &&
                           flens::IsComplex<T>::value,
                           void>::Type
ccopy_unroller(IndexType length, const T *x, T *y) 
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=length-N*numElements; i+=N*numElements) {

            ccopy_kernel<T,N>(x, y); 

            x+=N*numElements; 
            y+=N*numElements;

        }
        ccopy_unroller<IndexType, T, N/2, false>(length%(N*numElements), x, y);

    } else {
        if (length>=N*numElements) {

            ccopy_kernel<T,N>(x, y); 

            x+=N*numElements; 
            y+=N*numElements;

            length-=N*numElements;
        }
        ccopy_unroller<IndexType, T, N/2, false>(length, x, y);
    }
}

template <typename IndexType, typename T>
inline
typename flens::RestrictTo<flens::IsIntrinsicsCompatible<T>::value &&
                           flens::IsReal<T>::value,
                           void>::Type
ccopy(IndexType n, const T *x, IndexType incX, T *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("ccopy_intrinsics [ real, " INTRINSIC_NAME "]");

    copy(n, x, incX, y, incY);

}

template <typename IndexType, typename T>
inline
typename flens::RestrictTo<flens::IsIntrinsicsCompatible<T>::value &&
                           flens::IsComplex<T>::value,
                           void>::Type
ccopy(IndexType n, const T *x, IndexType incX, T *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("ccopy_intrinsics [ complex, " INTRINSIC_NAME "]");

    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const int numElements = IntrinsicType::numElements;

    if (incX==1 && incY==1) {
        
        IndexType i=0;

        int n_rest = n%numElements;

        if (n_rest>=2) {
            (*y++) = conj(*x++); 
            (*y++) = conj(*x++);
            n_rest-=2;
        }
        if (n_rest==1) { 
	    (*y++) = conj(*x++);
        }

        ccopy_unroller<IndexType, T>(n-n%numElements, x, y);
        

    } else {

        cxxblas::ccopy<IndexType, T, T>(n, x, incX, y, incY);

    }
}

#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_CCOPY_TCC
