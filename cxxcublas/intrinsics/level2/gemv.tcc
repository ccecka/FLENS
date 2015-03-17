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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL2_GEMV_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL2_GEMV_TCC 1

#include <array>
#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/intrinsics/auxiliary/auxiliary.h>
#include <playground/cxxblas/intrinsics/includes.h>

namespace cxxblas {

#ifdef USE_INTRINSIC

template <typename IndexType, typename T, int N>
inline
typename flens::RestrictTo<flens::IsReal<T>::value,
                           void>::Type
gemv_n_kernel(IndexType n, const T & alpha, const T *x, const T *A, IndexType ldA, T *y, IndexType incY)
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    std::array<IntrinsicType, N> _y;
    IntrinsicType _A, _x;

    for (IndexType k=0; k<N; ++k) {
        _y[k].setZero();
    }
    
    for (IndexType j=0; j<=n-numElements; j+=numElements) {
        _x.load(x);
        for(IndexType k=0; k<N; ++k) {
            _A.load(A+k*ldA);
            _y[k] = _intrinsic_add(_y[k], _intrinsic_mul(_A, _x));
        }
        x+=numElements;
        A+=numElements;
    }

    const IndexType n_rest = n%numElements;
    _A.setZero();
    _x.setZero();
    _x.load_partial(x, n_rest);

    for(IndexType k=0; k<N; ++k) {
        _A.load_partial(A+k*ldA, n_rest);
        _y[k] = _intrinsic_add(_y[k], _intrinsic_mul(_A, _x));
    }

    for (IndexType k=0; k<N; ++k) {
        (*y) += alpha*_intrinsic_hsum(_y[k]);
        y+=incY;
    }
}

template <typename IndexType, typename T, int N>
inline
typename flens::RestrictTo<flens::IsComplex<T>::value,
                           void>::Type
gemv_n_kernel(IndexType n, const T & alpha, const T *x, const T *A, IndexType ldA, T *y, IndexType incY)
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType  PT;
    typedef Intrinsics<PT, DEFAULT_INTRINSIC_LEVEL>    IntrinsicPrimitiveType;
    const IndexType numElements = IntrinsicType::numElements;

    std::array<IntrinsicType, N> _y;
    IntrinsicType _A, _x;
    IntrinsicPrimitiveType _real_x, _imag_x;

    for (IndexType k=0; k<N; ++k) {
        _y[k].setZero();
    }

    for (IndexType j=0; j<=n-numElements; j+=numElements) {
        _x.load(x);
        _real_x = _intrinsic_real(_x);
        _imag_x = _intrinsic_imag(_x);

        for(IndexType k=0; k<N; ++k) {
            _A.load(A+k*ldA);
            _y[k] = _intrinsic_add(_y[k], _intrinsic_mul(_A, _real_x));
            _A    = _intrinsic_swap_real_imag(_A);
            _y[k] = _intrinsic_addsub(_y[k], _intrinsic_mul(_A, _imag_x));
        }
        x+=numElements;
        A+=numElements;
    }

    const IndexType n_rest = n%numElements;
    _A.setZero();
    _x.setZero();
    _x.load_partial(x, n_rest);
    _real_x = _intrinsic_real(_x);
    _imag_x = _intrinsic_imag(_x);
    for(IndexType k=0; k<N; ++k) {
        _A.load_partial(A+k*ldA, n_rest);

        _y[k] = _intrinsic_add(_y[k], _intrinsic_mul(_A, _real_x));
        _A    = _intrinsic_swap_real_imag(_A);
        _y[k] = _intrinsic_addsub(_y[k], _intrinsic_mul(_A, _imag_x));
    }

    for (IndexType k=0; k<N; ++k) {
        (*y) += alpha*_intrinsic_hsum(_y[k]);
        y+=incY;
    }
}


template <typename IndexType, typename T, int N>
inline
typename flens::RestrictTo<flens::IsReal<T>::value,
                           void>::Type
gemv_c_kernel(IndexType n, const T & alpha, const T *x, const T *A, IndexType ldA, T *y, IndexType incY)
{
    gemv_n_kernel<IndexType, T,N>(n, alpha, x, A, ldA, y, incY);
}

template <typename IndexType, typename T, int N>
inline
typename flens::RestrictTo<flens::IsComplex<T>::value,
                           void>::Type
gemv_c_kernel(IndexType n, const T & alpha, const T *x, const T *A, IndexType ldA, T *y, IndexType incY)
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType  PT;
    typedef Intrinsics<PT, DEFAULT_INTRINSIC_LEVEL>    IntrinsicPrimitiveType;
    const IndexType numElements = IntrinsicType::numElements;

    std::array<IntrinsicType, N> _y;
    IntrinsicType _A, _x;
    IntrinsicPrimitiveType _real_x, _imag_x;

    for (IndexType k=0; k<N; ++k) {
        _y[k].setZero();
    }

    for (IndexType j=0; j<=n-numElements; j+=numElements) {
        _x.load(x);
        _x = _x;
        _real_x = _swap_sign(_intrinsic_real(_x));
        _imag_x = _intrinsic_imag(_x);

        for(IndexType k=0; k<N; ++k) {
            _A.load(A+k*ldA);
            _y[k] = _intrinsic_addsub(_y[k], _intrinsic_mul(_A, _real_x));
            _A    = _intrinsic_swap_real_imag(_A);
            _y[k] = _intrinsic_add(_y[k], _intrinsic_mul(_A, _imag_x));
        }
        x+=numElements;
        A+=numElements;
    }

    const IndexType n_rest = n%numElements;
    _A.setZero();
    _x.setZero();
    _x.load_partial(x, n_rest);
    _real_x = _swap_sign(_intrinsic_real(_x));
    _imag_x = _intrinsic_imag(_x);
    for(IndexType k=0; k<N; ++k) {
        _A.load_partial(A+k*ldA, n_rest);
        _y[k] = _intrinsic_addsub(_y[k], _intrinsic_mul(_A, _real_x));
        _A    = _intrinsic_swap_real_imag(_A);
        _y[k] = _intrinsic_add(_y[k], _intrinsic_mul(_A, _imag_x));
    }

    for (IndexType k=0; k<N; ++k) {
        (*y) += alpha*_intrinsic_hsum(_y[k]);
        y+=incY;
    }
}

template <typename IndexType, typename T, int N>
inline
typename flens::RestrictTo<flens::IsReal<T>::value,
                           void>::Type
gemv_t_kernel(IndexType n, const T & alpha, const T *x, IndexType incX, const T *A, IndexType ldA, T *y) 
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    std::array<IntrinsicType, N> _x;
    IntrinsicType _A, _y;

    for (IndexType i=0; i<N; ++i) {
        _x[i].fill(alpha*(*x)); 
        x+=incX;
    }

    for (IndexType i=0; i<=n-numElements; i+=numElements) {
        _y.load(y);

        for (IndexType j=0; j<N; ++j) {
            _A.load(A+j*ldA);

            _y = _intrinsic_add(_y, _intrinsic_mul(_A, _x[j]));
        }
        _y.store(y);
        A+=numElements;
        y+=numElements;
    }

    const IndexType n_rest = n%numElements;
    if ( n_rest==0 ) {
        return;
    }
    _y.load_partial(y, n_rest);

    for (IndexType j=0; j<N; ++j) {
        _A.load_partial(A+j*ldA, n_rest);

        _y = _intrinsic_add(_y, _intrinsic_mul(_A, _x[j]));
    }
    _y.store_partial(y, n_rest);
    
}

template <typename IndexType, typename T, int N>
inline
typename flens::RestrictTo<flens::IsComplex<T>::value,
                           void>::Type
gemv_t_kernel(IndexType n, const T & alpha, const T *x, IndexType incX, const T *A, IndexType ldA, T *y)
{
    using std::imag;
    using std::real;

    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType  PT;
    typedef Intrinsics<PT, DEFAULT_INTRINSIC_LEVEL>    IntrinsicPrimitiveType;
    const IndexType numElements = IntrinsicType::numElements;

    std::array<IntrinsicPrimitiveType, N> _real_x, _imag_x;
    IntrinsicType _A, _y;

    for (IndexType i=0; i<N; ++i) {
        _real_x[i].fill(real(alpha)*real(*x) - imag(alpha)*imag(*x));
        _imag_x[i].fill(real(alpha)*imag(*x) + imag(alpha)*real(*x));
        x+=incX;
    }

    for (IndexType i=0; i<=n-numElements; i+=numElements) {
        _y.load(y);

        for (IndexType j=0; j<N; ++j) {
            _A.load(A+j*ldA);

            _y = _intrinsic_add(_y, _intrinsic_mul(_A, _real_x[j]));
            _A = _intrinsic_swap_real_imag(_A);
            _y = _intrinsic_addsub(_y, _intrinsic_mul(_A, _imag_x[j]));
        }
        _y.store(y);
        A+=numElements;
        y+=numElements;
    }

    const IndexType n_rest = n%numElements;
    if ( n_rest==0 ) {
        return;
    }
    _y.load_partial(y, n_rest);

    for (IndexType j=0; j<N; ++j) {
        _A.load_partial(A+j*ldA, n_rest);

        _y = _intrinsic_add(_y, _intrinsic_mul(_A, _real_x[j]));
        _A = _intrinsic_swap_real_imag(_A);
        _y = _intrinsic_addsub(_y, _intrinsic_mul(_A, _imag_x[j]));
    }
    _y.store_partial(y, n_rest);

}


template <typename IndexType, typename T, int N>
inline
typename flens::RestrictTo<flens::IsReal<T>::value,
                           void>::Type
gemv_ct_kernel(IndexType n, const T & alpha, const T *x, IndexType incX, const T *A, IndexType ldA, T *y)
{
    gemv_ct_kernel<IndexType, T, N>(n, alpha, x, incX, A, ldA, y);
}

template <typename IndexType, typename T, int N>
inline
typename flens::RestrictTo<flens::IsComplex<T>::value,
                           void>::Type
gemv_ct_kernel(IndexType n, const T & alpha, const T *x, IndexType incX, const T *A, IndexType ldA, T *y)
{
    using std::imag;
    using std::real;

    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType  PT;
    typedef Intrinsics<PT, DEFAULT_INTRINSIC_LEVEL>    IntrinsicPrimitiveType;
    const IndexType numElements = IntrinsicType::numElements;

    std::array<IntrinsicPrimitiveType, N> _real_x, _imag_x;
    IntrinsicType _A, _y;

    for (IndexType i=0; i<N; ++i) {
        _real_x[i].fill(-real(alpha)*real(*x) + imag(alpha)*imag(*x));
        _imag_x[i].fill( real(alpha)*imag(*x) + imag(alpha)*real(*x));
        x+=incX;
    }

    for (IndexType i=0; i<=n-numElements; i+=numElements) {
        _y.load(y);

        for (IndexType j=0; j<N; ++j) {
            _A.loadu(A+j*ldA);

            _y = _intrinsic_addsub(_y, _intrinsic_mul(_A, _real_x[j]));
            _A = _intrinsic_swap_real_imag(_A);
            _y = _intrinsic_add(_y, _intrinsic_mul(_A, _imag_x[j]));
        }
        _y.store(y);
        A+=numElements;
        y+=numElements;
    }
    const IndexType n_rest = n%numElements;
    if ( n_rest==0 ) {
        return;
    }
    _y.load_partial(y, n_rest);

    for (IndexType j=0; j<N; ++j) {
        _A.load_partial(A+j*ldA, n_rest);

        _y = _intrinsic_addsub(_y, _intrinsic_mul(_A, _real_x[j]));
        _A = _intrinsic_swap_real_imag(_A);
        _y = _intrinsic_add(_y, _intrinsic_mul(_A, _imag_x[j]));
    }
    _y.store_partial(y, n_rest);
}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
gemv_n_unroller(IndexType m, IndexType n, const T & alpha,
                const T *x, const T *A, IndexType ldA, 
                T *y, IndexType incY)
{

}

template <typename IndexType, typename T,
          int N, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
gemv_n_unroller(IndexType m, IndexType n, const T & alpha,
                const T *x, const T *A, IndexType ldA, 
                T *y, IndexType incY)
{

    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=m-N; i+=N) {
            gemv_n_kernel<IndexType,T,N>(n, alpha, x, A, ldA, y, incY);

            y+=N*incY;
            A+=N*ldA;

        }
        gemv_n_unroller<IndexType, T, N/2, false>(m%N,n, alpha, x, A, ldA, y, incY);

    } else {
        if (m>=N) {

            gemv_n_kernel<IndexType,T,N>(n, alpha, x, A, ldA, y, incY);

            y+=N*incY;
            A+=N*ldA;

            m-=N;
        }
        gemv_n_unroller<IndexType, T, N/2, false>(m, n, alpha, x, A, ldA, y, incY);
    }
}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
gemv_c_unroller(IndexType m, IndexType n, const T & alpha,
                const T *x, const T *A, IndexType ldA,
                T *y, IndexType incY)
{

}

template <typename IndexType, typename T,
          int N, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
gemv_c_unroller(IndexType m, IndexType n, const T & alpha,
                const T *x, const T *A, IndexType ldA,
                T *y, IndexType incY)
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=m-N; i+=N) {
            gemv_c_kernel<IndexType,T,N>(n, alpha, x, A, ldA, y, incY);

            y+=N*incY;
            A+=N*ldA;

        }
        gemv_c_unroller<IndexType, T, N/2, false>(m%N,n, alpha, x, A, ldA, y, incY);

    } else {
        if (m>=N) {

            gemv_c_kernel<IndexType,T,N>(n, alpha, x, A, ldA, y, incY);

            y+=N*incY;
            A+=N*ldA;

            m-=N;
        }
        gemv_c_unroller<IndexType, T, N/2, false>(m, n, alpha, x, A, ldA, y, incY);
    }
}




template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
gemv_t_unroller(IndexType m, IndexType n, const T & alpha,
                const T *x, IndexType incX,
                const T *A, IndexType ldA, T *y)
{

}

template <typename IndexType, typename T, 
          int N, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
gemv_t_unroller(IndexType m, IndexType n, const T & alpha, 
                const T *x, IndexType incX, 
                const T *A, IndexType ldA, T *y) 
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=m-N; i+=N) {
            gemv_t_kernel<IndexType,T,N>(n, alpha, x, incX, A, ldA, y); 

            x+=N*incX; 
            A+=N*ldA;

        }
        gemv_t_unroller<IndexType, T, N/2, false>(m%N,n, alpha, x, incX, A, ldA, y);

    } else {
        if (m>=N) {

            gemv_t_kernel<IndexType,T,N>(n, alpha, x, incX, A, ldA, y); 

            x+=N*incX; 
            A+=N*ldA;

            m-=N;
        }
        gemv_t_unroller<IndexType, T, N/2, false>(m, n, alpha, x, incX, A, ldA, y);
    }
}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
gemv_ct_unroller(IndexType m, IndexType n, const T & alpha,
                 const T *x, IndexType incX,
                 const T *A, IndexType ldA, T *y)
{

}

template <typename IndexType, typename T,
          int N, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
gemv_ct_unroller(IndexType m, IndexType n, const T & alpha,
                 const T *x, IndexType incX,
                 const T *A, IndexType ldA, T *y)
{
    typedef Intrinsics<T, IntrinsicsLevel::SSE> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=m-N; i+=N) {
            gemv_ct_kernel<IndexType,T,N>(n, alpha, x, incX, A, ldA, y);

            x+=N*incX;
            A+=N*ldA;

        }
        gemv_ct_unroller<IndexType, T, N/2, false>(m%N,n, alpha, x, incX, A, ldA, y);

    } else {
        if (m>=N) {

            gemv_ct_kernel<IndexType,T,N>(n, alpha, x, incX, A, ldA, y);

            x+=N*incX;
            A+=N*ldA;

            m-=N;
        }
        gemv_ct_unroller<IndexType, T, N/2, false>(m, n, alpha, x, incX, A, ldA, y);
    }
}

template <typename IndexType, typename T>
inline
typename flens::RestrictTo<flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
gemv(StorageOrder order, Transpose transA,
     IndexType m, IndexType n,
     const T &alpha,
     const T *A, IndexType ldA,
     const T *x, IndexType incX,
     const T &beta,
     T *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("gemv_intrinsics [complex, " INTRINSIC_NAME "]");

    if (order==ColMajor) {
        transA = Transpose(transA^Trans);
        gemv(RowMajor, transA, n, m, alpha, A, ldA,
             x, incX, beta, y, incY);
        return;
    }


    scal(n, beta, y, incY);

    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;
    
    T *tmp_mem = NULL;
   
    if ( transA==NoTrans ) {

        if (incX==1) {
	    if (incY<0) {
		y -= incY*(m-1);
	    }
            
            if (flens::IsReal<T>::value) {
	        gemv_n_unroller<IndexType, T, 4>(m, n, alpha, x, A, ldA, y, incY);
            } else {
                gemv_n_unroller<IndexType, T, 4>(m, n, alpha, x, A, ldA, y, incY);
            }
        } else {

            tmp_mem = new T[n];
            copy(n, x, incX, tmp_mem, 1);
            gemv(RowMajor, NoTrans, m, n, alpha, A, ldA, tmp_mem, 1, T(1), y, incY);
            delete[] tmp_mem; 

        }

    } else if (transA==Conj) {
  
        if (incX==1) {
	    if (incY<0) {
		y -= incY*(m-1);
	    }

	    gemv_c_unroller<IndexType, T, 4>(m,n, alpha, x, A, ldA, y, incY);


        } else {

            tmp_mem = new T[n];
            copy(n, x, incX, tmp_mem, 1);
            gemv(RowMajor, Conj, m, n, alpha, A, ldA, tmp_mem, 1, T(1), y, incY);
            delete[] tmp_mem; 
        
        }


    } else if (transA==Trans) {

        if (incY==1) {

	    if (incX<0) {
		x -= incX*(m-1);
	    }
            if ( flens::IsReal<T>::value ) {
	        gemv_t_unroller<IndexType, T, 8>(m, n, alpha, x, incX, A, ldA, y);
            } else {
                gemv_t_unroller<IndexType, T, 4>(m, n, alpha, x, incX, A, ldA, y);
            }

        } else {

            tmp_mem = new T[n];
            std::fill_n(tmp_mem, n, T(0));
            gemv(RowMajor, Trans, m, n, alpha, A, ldA, x, incX, T(1), tmp_mem, 1);
            axpy(n, T(1), tmp_mem, 1, y, incY);
            delete[] tmp_mem;

        }

    } else if (transA==ConjTrans) {

        if (incY==1) {

	    if (incX<0) {
		x -= incX*(m-1);
	    }

	    gemv_ct_unroller<IndexType, T, 8>(m, n, alpha, x, incX, A, ldA, y);


        } else {

            tmp_mem = new T[n];
            std::fill_n(tmp_mem, n, T(0));
            gemv(RowMajor, ConjTrans, m, n, alpha, A, ldA, x, incX, T(1), tmp_mem, 1);
            axpy(n, T(1), tmp_mem, 1, y, incY);
            delete[] tmp_mem;
        
        }
    } 
    
}

#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL2_GEMV_TCC
