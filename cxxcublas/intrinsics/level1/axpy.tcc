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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_AXPY_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_AXPY_TCC 1

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/intrinsics/auxiliary/auxiliary.h>
#include <playground/cxxblas/intrinsics/includes.h>
#include <playground/cxxblas/intrinsics/level1/axpy.h>

namespace cxxblas {

#ifdef USE_INTRINSIC

template <typename T, int N>
inline
typename flens::RestrictTo<flens::IsReal<T>::value,
                           void>::Type
axpy_kernel(const T & alpha, const T *x, T *y) 
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const int numElements = IntrinsicType::numElements;
    IntrinsicType _x, _y;
    IntrinsicType _alpha(alpha);  

    for (int i=0; i<N; ++i){
        _x.load(x);
        _y.load(y);
        _y = _intrinsic_add(_y, _intrinsic_mul(_alpha, _x));
        _y.store(y);
        x+=numElements;
        y+=numElements;
   }

}

template <typename T, int N>
inline
typename flens::RestrictTo<flens::IsComplex<T>::value,
                           void>::Type
axpy_kernel(const T & alpha, const T *x, T *y) 
{
    using std::real;
    using std::imag;

    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType PT;
    typedef Intrinsics<PT, IntrinsicsLevel::SSE> IntrinsicPrimitiveType;
    const int numElements = IntrinsicType::numElements;

    IntrinsicType _x, _y;
    IntrinsicPrimitiveType _real_alpha(real(alpha));
    IntrinsicPrimitiveType _imag_alpha(imag(alpha)); 

    for (int i=0; i<N; ++i){
        _x.load(x);
        _y.load(y);
        _y = _intrinsic_add(_y, _intrinsic_mul(_real_alpha, _x));
        _x = _intrinsic_swap_real_imag(_x);
        _y = _intrinsic_addsub(_y, _intrinsic_mul(_imag_alpha, _x));
        _y.store(y);
        x+=numElements;
        y+=numElements;
   }
}

template <typename IndexType, typename T, 
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
axpy_unroller(IndexType length, const T & alpha, const T *x, T *y) 
{

}

template <typename IndexType, typename T, 
          int N = 16, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
axpy_unroller(IndexType length, const T & alpha, const T *x, T *y) 
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=length-N*numElements; i+=N*numElements) {

            axpy_kernel<T,N>(alpha, x, y); 

            x+=N*numElements; 
            y+=N*numElements;

        }
        axpy_unroller<IndexType, T, N/2, false>(length%(N*numElements), alpha, x, y);

    } else {
        if (length>=N*numElements) {

            axpy_kernel<T,N>(alpha, x, y); 

            x+=N*numElements; 
            y+=N*numElements;

            length-=N*numElements;
        }
        axpy_unroller<IndexType, T, N/2, false>(length, alpha, x, y);
    }
}

template <typename IndexType, typename T>
inline
typename flens::RestrictTo<flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
axpy(IndexType n, const T &alpha, const T *x,
     IndexType incX, T *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("axpy_intrinsics [" INTRINSIC_NAME "]");

    using std::real;
    using std::imag;
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType  PT;
    const int numElements = IntrinsicType::numElements;


    if (alpha==T(0))
        return;

    if (incX==1 && incY==1) {
        
        if (IsComplex<T>::value && imag(alpha)==PT(0)) {
            axpy(2*n, real(alpha),
                 reinterpret_cast<const PT*>(x), 1,
                 reinterpret_cast<PT*>(y), 1);
            return;
        }

        IndexType i=0;

        int n_rest = n%numElements;

        if (n_rest>=2) {
            (*y++) += alpha*(*x++); 
            (*y++) += alpha*(*x++);
            n_rest-=2;
        }
        if (n_rest==1) { 
	    (*y++) += alpha*(*x++);
        }

        axpy_unroller<IndexType, T>(n-n%numElements, alpha, x, y);
        

    } else {

        cxxblas::axpy<IndexType, T, T ,T>(n, alpha, x, incX, y, incY);

    }
}

#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_AXPY_TCC
