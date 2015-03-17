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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1EXTENSIONS_AXPBY_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1EXTENSIONS_AXPBY_TCC 1

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/intrinsics/auxiliary/auxiliary.h>
#include <playground/cxxblas/intrinsics/includes.h>
#include <playground/cxxblas/intrinsics/level1extensions/axpby.h>

namespace cxxblas {

#ifdef USE_INTRINSIC

template <typename T, int N>
inline
typename flens::RestrictTo<flens::IsReal<T>::value,
                           void>::Type
axpby_kernel(const T & alpha, const T *x, const T & beta, T *y) 
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const int numElements = IntrinsicType::numElements;
    IntrinsicType _x, _y;
    IntrinsicType _alpha(alpha), _beta(beta);  

    for (int i=0; i<N; ++i){
        _x.load(x);
        _y.load(y);
        _y = _intrinsic_add(_intrinsic_mul(_beta, _y), _intrinsic_mul(_alpha, _x));
        _y.store(y);
        x+=numElements;
        y+=numElements;
   }

}

template <typename T, int N>
inline
typename flens::RestrictTo<flens::IsComplex<T>::value,
                           void>::Type
axpby_kernel(const T & alpha, const T *x, const T & beta, T *y) 
{
    using std::real;
    using std::imag;

    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType PT;
    typedef Intrinsics<PT, IntrinsicsLevel::SSE> IntrinsicPrimitiveType;
    const int numElements = IntrinsicType::numElements;

    IntrinsicType _x, _y, _result;
    IntrinsicPrimitiveType _real_alpha(real(alpha));
    IntrinsicPrimitiveType _imag_alpha(imag(alpha)); 
    IntrinsicPrimitiveType _real_beta(real(beta));
    IntrinsicPrimitiveType _imag_beta(imag(beta)); 

    for (int i=0; i<N; ++i){
        _x.load(x);
        _y.load(y);
        _result = _intrinsic_add(_intrinsic_mul(_real_beta, _y), _intrinsic_mul(_real_alpha, _x));
        _x = _intrinsic_swap_real_imag(_x);
        _y = _intrinsic_swap_real_imag(_y);
        _result = _intrinsic_addsub(_result, _intrinsic_mul(_imag_beta, _y));
        _result = _intrinsic_addsub(_result, _intrinsic_mul(_imag_alpha, _x));
        _result.store(y);
        x+=numElements;
        y+=numElements;
   }
}

template <typename IndexType, typename T, 
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
axpby_unroller(IndexType length, const T & alpha, const T *x, const T & beta, T *y) 
{

}

template <typename IndexType, typename T, 
          int N = 16, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
axpby_unroller(IndexType length, const T & alpha, const T *x, const T &beta,  T *y) 
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=length-N*numElements; i+=N*numElements) {

            axpby_kernel<T,N>(alpha, x, beta, y); 

            x+=N*numElements; 
            y+=N*numElements;

        }
        axpby_unroller<IndexType, T, N/2, false>(length%(N*numElements), alpha, x, beta, y);

    } else {
        if (length>=N*numElements) {

            axpby_kernel<T,N>(alpha, x, beta, y); 

            x+=N*numElements; 
            y+=N*numElements;

            length-=N*numElements;
        }
        axpby_unroller<IndexType, T, N/2, false>(length, alpha, x, beta, y);
    }
}

template <typename T, int N>
inline
typename flens::RestrictTo<flens::IsReal<T>::value,
                           void>::Type
axeqy_kernel(const T & alpha, const T *x, T *y)
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const int numElements = IntrinsicType::numElements;
    IntrinsicType _x, _y;
    IntrinsicType _alpha(alpha);

    for (int i=0; i<N; ++i){
        _x.load(x);
        _y = _intrinsic_mul(_alpha, _x);
        _y.store(y);
        x+=numElements;
        y+=numElements;
   }

}

template <typename T, int N>
inline
typename flens::RestrictTo<flens::IsComplex<T>::value,
                           void>::Type
axeqy_kernel(const T & alpha, const T *x, T *y)
{
    using std::real;
    using std::imag;

    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType PT;
    typedef Intrinsics<PT, IntrinsicsLevel::SSE> IntrinsicPrimitiveType;
    const int numElements = IntrinsicType::numElements;

    IntrinsicType _x, _y, _result;
    IntrinsicPrimitiveType _real_alpha(real(alpha));
    IntrinsicPrimitiveType _imag_alpha(imag(alpha));

    for (int i=0; i<N; ++i){
        _x.load(x);
        _result = _intrinsic_mul(_real_alpha, _x);
        _x = _intrinsic_swap_real_imag(_x);
        _result = _intrinsic_addsub(_result, _intrinsic_mul(_imag_alpha, _x));
        _result.store(y);
        x+=numElements;
        y+=numElements;
   }
}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
axeqy_unroller(IndexType length, const T & alpha, const T *x, T *y)
{

}

template <typename IndexType, typename T,
          int N = 16, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
axeqy_unroller(IndexType length, const T & alpha, const T *x, T *y)
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=length-N*numElements; i+=N*numElements) {

            axeqy_kernel<T,N>(alpha, x, y);

            x+=N*numElements;
            y+=N*numElements;

        }
        axeqy_unroller<IndexType, T, N/2, false>(length%(N*numElements), alpha, x, y);

    } else {
        if (length>=N*numElements) {

            axeqy_kernel<T,N>(alpha, x, y);

            x+=N*numElements;
            y+=N*numElements;

            length-=N*numElements;
        }
        axeqy_unroller<IndexType, T, N/2, false>(length, alpha, x, y);
    }
}

template <typename IndexType, typename T>
inline
typename flens::RestrictTo<flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
axpby(IndexType n, const T &alpha, const T *x,
      IndexType incX, const T &beta, T *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("axpby_intrinsics [" INTRINSIC_NAME "]");

    using std::real;
    using std::imag;
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType  PT;
    const int numElements = IntrinsicType::numElements;


    if (alpha==T(0)) {
        scal(n, beta, y, incY);
        return;
    }
    if (beta==T(1)) {
        axpy(n, alpha, x, incX, y, incY);
        return;
    }
    if (incX==1 && incY==1) {
        
        if (IsComplex<T>::value && imag(alpha)==PT(0) && imag(beta)==PT(0)) {
            axpby(2*n, real(alpha),
                  reinterpret_cast<const PT*>(x), 1,
                  real(beta), reinterpret_cast<PT*>(y), 1);
            return;
        }

        IndexType i=0;

        int n_rest = n%numElements;

        if (n_rest>=2) {
            (*y) = beta*(*y) + alpha*(*x);
            x++;
            y++;
            (*y) = beta*(*y) + alpha*(*x);
            x++;
            y++;
            n_rest-=2;
        }
        if (n_rest==1) { 
	    (*y) = beta*(*y) + alpha*(*x);
            x++;
            y++;
        }
        if (beta==T(0)) {
            axeqy_unroller<IndexType, T>(n-n%numElements, alpha, x, y);
        } else {
            axpby_unroller<IndexType, T>(n-n%numElements, alpha, x, beta, y);
        }

    } else {

        cxxblas::axpby<IndexType, T, T, T, T>(n, alpha, x, incX, beta, y, incY);

    }
}

#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1EXTENSIONS_AXPBY_TCC
