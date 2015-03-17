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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_NRM2_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_NRM2_TCC 1

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
nrm2_kernel(const T *x, S &_ssq, S &_scale) 
{

    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const int numElements = IntrinsicType::numElements;

    IntrinsicType _Xi, _absXi, _absXiMax, _tmp;
    IntrinsicType _zero(T(0));
    for (int i=0; i<N; ++i) {
        _Xi.load(x);
        _absXi = _abs(_Xi);
        x+=numElements;
        if(!(_absXi<=_zero)) {
            if (!(_scale>=_absXi)) {
                _tmp   = _absXi.max();
                _scale = _intrinsic_div(_scale, _tmp) ;
                _scale = _intrinsic_mul(_scale, _scale) ;
                _ssq   = _intrinsic_mul(_ssq, _scale) ;
                _scale = _tmp;
            } 

            _absXi = _intrinsic_div(_absXi, _scale);
            _absXi = _intrinsic_mul(_absXi, _absXi);
            _ssq   = _intrinsic_add(_ssq, _absXi) ; 
        }
    }
}


template <typename IndexType, typename T, typename S,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
nrm2_unroller(IndexType length, const T *x, S &_ssq, S &_scale)
{

}

template <typename IndexType, typename T, typename S,
          int N = 16, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
nrm2_unroller(IndexType length, const T *x, S &_ssq, S &_scale) 
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {
        for (IndexType i=0; i<=length-N*numElements; i+=N*numElements) {

            nrm2_kernel<T,IntrinsicType,N>(x, _ssq, _scale); 

            x+=N*numElements; 

        }
        nrm2_unroller<IndexType, T, IntrinsicType, N/2, false>(length%(N*numElements), x, _ssq, _scale);


    } else {
        if (length>=N*numElements) {

            nrm2_kernel<T,IntrinsicType,N>(x, _ssq, _scale);

            x+=N*numElements; 

            length-=N*numElements;
        }
        nrm2_unroller<IndexType, T, IntrinsicType, N/2, false>(length, x, _ssq, _scale);
    }
}


template <typename IndexType, typename T>
inline
typename flens::RestrictTo<flens::IsReal<T>::value && 
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
nrm2(IndexType n, const T *x, IndexType incX, T &norm)
{
    CXXBLAS_DEBUG_OUT("nrm2_intrinsics [" INTRINSIC_NAME "]");

    using std::abs;

    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL>     IntrinsicType;

    if (incX==1) {

        const int numElements = IntrinsicType::numElements;
        T One(1), Zero(0);
        T scale = 0;
        T ssq   = 0;
        IndexType i=0;

        for (; i<n%numElements; ++i) {
            T absXi = abs(*x++);
            if (absXi>Zero) {
                if (scale<absXi) {
                    ssq = ssq * pow(scale/absXi, 2);
                    scale = absXi;
                } 

                ssq += pow(absXi/scale, 2);
            }
        }


        IntrinsicType _ssq, _scale;
        IntrinsicType _Xi, _absXi, _tmp;

        _ssq.setZero();
        _ssq.load_partial(&ssq,1);
        _scale.fill(scale);

        nrm2_unroller(n-(n%numElements), x, _ssq, _scale);

        T tmp[numElements];
        _ssq.store(tmp);
        ssq = T(0);
        for (int j=0; j<numElements; ++j) {
            ssq += tmp[j];
        }

        _scale.store_partial(&scale, 1);
        norm = scale*sqrt(ssq);


    } else {
        cxxblas::nrm2<IndexType, T, T>(n, x, incX, norm);
    }
}

template <typename IndexType, typename T>
inline
typename flens::RestrictTo<flens::IsReal<T>::value && 
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
nrm2(IndexType n, const std::complex<T> *y, IndexType incY, T &norm)
{
    CXXBLAS_DEBUG_OUT("asum_intrinsics [complex, " INTRINSIC_NAME "]");

    if (incY==1) {
        nrm2(2*n, reinterpret_cast<const T*>(y), 1, norm);
    } else {
        cxxblas::nrm2<IndexType, std::complex<T>, T>(n, y, incY, norm);
    }
}

#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_NRM2_TCC
