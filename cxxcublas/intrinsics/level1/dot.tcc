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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_DOT_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_DOT_TCC 1

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/intrinsics/auxiliary/auxiliary.h>
#include <playground/cxxblas/intrinsics/includes.h>

namespace cxxblas {

#ifdef USE_INTRINSIC

template <typename T, typename S, int N>
inline
typename flens::RestrictTo<flens::IsReal<T>::value,
                           void>::Type
dot_kernel(const T *x, const T *y, S &_result) 
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const int numElements = IntrinsicType::numElements;
    IntrinsicType _x, _y;

    for (int i=0; i<N; ++i) {

        _x.load(x);
        _y.load(y);
        _result = _intrinsic_add(_result, _intrinsic_mul(_x, _y));
        x+=numElements;
        y+=numElements;
   }
}

template <typename T, typename S, int N>
inline
typename flens::RestrictTo<flens::IsComplex<T>::value,
                           void>::Type
dot_kernel(const T *x, const T *y, S &_result_real, S &_result_imag) 
{
    using std::real;
    using std::imag;

    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType PT;
    typedef Intrinsics<PT, IntrinsicsLevel::SSE> IntrinsicPrimitiveType;
    const int numElements = IntrinsicType::numElements;

    IntrinsicType          _x;
    IntrinsicPrimitiveType _y;

    for (int i=0; i<N; ++i) {

        _x.load(x);
        _y.load(reinterpret_cast<const PT*>(y));

        _result_real = _intrinsic_add(_result_real, _intrinsic_mul(_x, _y));

        _x = _intrinsic_swap_real_imag(_x);

        _result_imag = _intrinsic_add(_result_imag, _intrinsic_mul(_x, _y));

        x+=numElements;
        y+=numElements;
   }
}

template <typename IndexType, typename T, typename S,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
dot_unroller(IndexType length, const T *x, const T *y, S &_result)
{

}

template <typename IndexType, typename T, typename S,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
dot_unroller(IndexType length, const T *x, const T *y, S &_result_real, S &_result_imag) 
{

}

template <typename IndexType, typename T, typename S,
          int N = 16, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
dot_unroller(IndexType length, const T *x, const T *y, S &_result) 
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {
        _result.setZero();
        for (IndexType i=0; i<=length-N*numElements; i+=N*numElements) {

            dot_kernel<T,IntrinsicType,N>(x, y, _result); 

            x+=N*numElements; 
            y+=N*numElements;

        }
        dot_unroller<IndexType, T, IntrinsicType, N/2, false>(length%(N*numElements), x, y, _result);

    } else {
        if (length>=N*numElements) {

            dot_kernel<T,IntrinsicType,N>(x, y, _result); 

            x+=N*numElements; 
            y+=N*numElements;

            length-=N*numElements;
        }
        dot_unroller<IndexType, T, IntrinsicType, N/2, false>(length, x, y, _result);
    }
}

template <typename IndexType, typename T, typename S,
          int N = 16, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
dot_unroller(IndexType length, const T *x, const T *y, S &_result_real, S &_result_imag)
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {
        _result_real.setZero();
        _result_imag.setZero();
        for (IndexType i=0; i<=length-N*numElements; i+=N*numElements) {

            dot_kernel<T,IntrinsicType,N>(x, y, _result_real, _result_imag);

            x+=N*numElements;
            y+=N*numElements;

        }
        dot_unroller<IndexType, T, IntrinsicType, N/2, false>(length%(N*numElements), x, y, _result_real, _result_imag);

    } else {
        if (length>=N*numElements) {

            dot_kernel<T,IntrinsicType,N>(x, y, _result_real, _result_imag);

            x+=N*numElements;
            y+=N*numElements;

            length-=N*numElements;
        }
        dot_unroller<IndexType, T, IntrinsicType, N/2, false>(length, x, y, _result_real, _result_imag);
    }
}

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsReal<T>::value &&
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
dotu(IndexType n,
     const T *x, IndexType incX,
     const T *y, IndexType incY,
     T &result)
{
    CXXBLAS_DEBUG_OUT("dotu_intrinsic [ real, " INTRINSIC_NAME "]");

    if (incX==1 && incY==1) {

        result = T(0);

        typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
        const int numElements = IntrinsicType::numElements;

        int n_rest = n%numElements;

        if (n_rest>=2) {
            result += (*y++)*(*x++);
            result += (*y++)*(*x++);
            n_rest-=2;
        }
        if (n_rest==1) {
            result += (*y++)*(*x++);
        }

        IntrinsicType _result;
        dot_unroller<IndexType, T, IntrinsicType>(n-n%numElements, x, y, _result);
        result += _intrinsic_hsum(_result);

    } else {

        cxxblas::dotu<IndexType, T, T ,T>(n, x, incX, y, incY, result);

    }
}

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsComplex<T>::value &&
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
dotu(IndexType n,
     const T *x, IndexType incX,
     const T *y, IndexType incY,
     T &result)
{
    CXXBLAS_DEBUG_OUT("dotu_intrinsic [ complex, " INTRINSIC_NAME "]");

    if (incX==1 && incY==1) {

        result = T(0);

        typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
        typedef typename IntrinsicType::PrimitiveDataType PT;
        typedef Intrinsics<PT, IntrinsicsLevel::SSE> IntrinsicPrimitiveType;
        const int numElements = IntrinsicType::numElements;

        int n_rest = n%numElements;

        if (n_rest>=2) {
            result += (*y++)*(*x++); 
            result += (*y++)*(*x++); 
            n_rest-=2;
        }
        if (n_rest==1) { 
            result += (*y++)*(*x++); 
        }

        IntrinsicType _result_real, _result_imag;
        dot_unroller<IndexType, T, IntrinsicType>(n-n%numElements, x, y, _result_real, _result_imag);

        PT tmp_result_real[2*numElements], tmp_result_imag[2*numElements];
        _result_real.store(reinterpret_cast<T*>(tmp_result_real));
        _result_imag.store(reinterpret_cast<T*>(tmp_result_imag));

        for (IndexType k=0; k<numElements; ++k) {
            result += T(tmp_result_real[2*k]-tmp_result_real[2*k+1], tmp_result_imag[2*k]+tmp_result_imag[2*k+1]);
        }


    } else {

        cxxblas::dotu<IndexType, T, T ,T>(n, x, incX, y, incY, result);

    }
}

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsReal<T>::value && 
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
dot(IndexType n,
    const T *x, IndexType incX,
    const T *y, IndexType incY,
    T &result)
{
    CXXBLAS_DEBUG_OUT("dot_intrinsic [real, " INTRINSIC_NAME "]");
    dotu(n, x, incX, y, incY, result);
}

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsComplex<T>::value && 
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
dot(IndexType n,
    const T *x, IndexType incX,
    const T *y, IndexType incY,
    T &result)
{
    CXXBLAS_DEBUG_OUT("dot_intrinsic [complex, " INTRINSIC_NAME "]");

    using std::conj;


    if (incX==1 && incY==1) {

        typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
        typedef typename IntrinsicType::PrimitiveDataType PT;
        typedef Intrinsics<PT, IntrinsicsLevel::SSE> IntrinsicPrimitiveType;
        const int numElements = IntrinsicType::numElements;

        int n_rest = n%numElements;

        if (n_rest>=2) {
            result += (*y++)*conj(*x++);
            result += (*y++)*conj(*x++);
            n_rest-=2;
        }
        if (n_rest==1) {
            result += (*y++)*conj(*x++);
        }

        IntrinsicType _result_real, _result_imag;
        dot_unroller<IndexType, T, IntrinsicType>(n-n%numElements, x, y, _result_real, _result_imag);

        PT tmp_result_real[2*numElements], tmp_result_imag[2*numElements];
        _result_real.store(reinterpret_cast<T*>(tmp_result_real));
        _result_imag.store(reinterpret_cast<T*>(tmp_result_imag));

        for (IndexType k=0; k<numElements; ++k) {
            result += T(tmp_result_real[2*k]+tmp_result_real[2*k+1], -tmp_result_imag[2*k]+tmp_result_imag[2*k+1]);
        }

    } else {

        cxxblas::dot<IndexType, T, T ,T>(n, x, incX, y, incY, result);

    }
}

#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_DOT_TCC
