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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL2_TRMV_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL2_TRMV_TCC 1

#include <iostream>

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/intrinsics/auxiliary/auxiliary.h>
#include <playground/cxxblas/intrinsics/includes.h>

namespace cxxblas {

#ifdef USE_INTRINSIC

template <typename IndexType, typename T, int N>
inline
void
trmv_up_n_kernel(Diag diag, IndexType n,
                 const T *A, IndexType ldA, T *x)
{
    for (IndexType i=0; i<N; ++i) {

        if (diag==NonUnit)
            x[i] = A[i*ldA+i]*x[i];

        for (IndexType j=i+1; j<N; ++j) {
           x[i] += A[i*ldA+j]*x[j];
        }
    }
    gemv_n_kernel<IndexType, T, N>(n-N, T(1), x+N, A+N, ldA, x, 1);

}

template <typename IndexType, typename T, int N>
inline
void
trmv_up_c_kernel(Diag diag, IndexType n,
                 const T *A, IndexType ldA, T *x)
{

    for (IndexType i=0; i<N; ++i) {

        if (diag==NonUnit)
            x[i] = conjugate(A[i*ldA+i])*x[i];

        for (IndexType j=i+1; j<N; ++j) {
           x[i] += conjugate(A[i*ldA+j])*x[j];
        }
    }
    gemv_c_kernel<IndexType, T, N>(n-N, T(1), x+N, A+N, ldA, x, 1);

}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trmv_up_n_unroller(Diag diag, IndexType n,
                   const T *A, IndexType ldA, T *x)
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trmv_up_n_unroller(Diag diag, IndexType n,
                   const T *A, IndexType ldA, T *x)
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for(IndexType i=0; i<=n-N; i+=N) {
            trmv_up_n_kernel<IndexType, T,N>(diag, n-i, A, ldA, x);
            A+=N*(ldA+1);
            x+=N;
        }
        trmv_up_n_unroller<IndexType, T, N/2, false>(diag, n%N, A, ldA, x);

    } else {
        if (n>=N) {

            trmv_up_n_kernel<IndexType, T,N>(diag, n, A, ldA, x);

            A+=N*(ldA+1);
            x+=N;

            n-=N;
        }
        trmv_up_n_unroller<IndexType, T, N/2, false>(diag, n%N, A, ldA, x);
    }
}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trmv_up_c_unroller(Diag diag, IndexType n,
                   const T *A, IndexType ldA, T *x)
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trmv_up_c_unroller(Diag diag, IndexType n,
                   const T *A, IndexType ldA, T *x)
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for(IndexType i=0; i<=n-N; i+=N) {
            trmv_up_c_kernel<IndexType, T,N>(diag, n-i, A, ldA, x);
            A+=N*(ldA+1);
            x+=N;
        }
        trmv_up_c_unroller<IndexType, T, N/2, false>(diag, n%N, A, ldA, x);

    } else {
        if (n>=N) {

            trmv_up_c_kernel<IndexType, T,N>(diag, n, A, ldA, x);

            A+=N*(ldA+1);
            x+=N;

            n-=N;
        }
        trmv_up_c_unroller<IndexType, T, N/2, false>(diag, n%N, A, ldA, x);
    }
}


template <typename IndexType, typename T, int N>
inline
void
trmv_low_n_kernel(Diag diag, IndexType n,
                  const T *A, IndexType ldA, T *x)
{
    for (IndexType i=n-1; i>=n-N; --i) {

	if (diag==NonUnit) {
	    x[i] = A[i*ldA+i]*x[i];
	}
	
	for (IndexType j=n-N; j<i; ++j) {
	   x[i] += A[i*ldA+j]*x[j];
	}
    }
    gemv_n_kernel<IndexType, T,N>(n-N, T(1), x, A+(n-N)*ldA, ldA, x+(n-N), 1);

}


template <typename IndexType, typename T, int N>
inline
void
trmv_low_c_kernel(Diag diag, IndexType n,
                  const T *A, IndexType ldA, T *x)
{
    for (IndexType i=n-1; i>=n-N; --i) {

        if (diag==NonUnit) {
            x[i] = conjugate(A[i*ldA+i])*x[i];
        }

        for (IndexType j=n-N; j<i; ++j) {
           x[i] += conjugate(A[i*ldA+j])*x[j];
        }
    }
    gemv_c_kernel<IndexType, T,N>(n-N, T(1), x, A+(n-N)*ldA, ldA, x+(n-N), 1);

}


template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trmv_low_n_unroller(Diag diag, IndexType n,
                   const T *A, IndexType ldA, T *x)
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trmv_low_n_unroller(Diag diag, IndexType n,
                    const T *A, IndexType ldA, T *x)
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0;i<=n-N; i+=N) {
	    trmv_low_n_kernel<IndexType, T, N>(diag, n-i, A, ldA, x);
        }
        trmv_low_n_unroller<IndexType, T, N/2, false>(diag, n%N, A, ldA, x);

    } else {
        if (n>=N) {
            trmv_low_n_kernel<IndexType, T, N>(diag, n, A, ldA, x);

            n-=N;
        }
        trmv_low_n_unroller<IndexType, T, N/2, false>(diag, n, A, ldA, x);
    }
}


template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trmv_low_c_unroller(Diag diag, IndexType n,
                   const T *A, IndexType ldA, T *x)
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trmv_low_c_unroller(Diag diag, IndexType n,
                    const T *A, IndexType ldA, T *x)
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0;i<=n-N; i+=N) {
            trmv_low_c_kernel<IndexType, T, N>(diag, n-i, A, ldA, x);
        }
        trmv_low_c_unroller<IndexType, T, N/2, false>(diag, n%N, A, ldA, x);

    } else {
        if (n>=N) {
            trmv_low_c_kernel<IndexType, T, N>(diag, n, A, ldA, x);

            n-=N;
        }
        trmv_low_c_unroller<IndexType, T, N/2, false>(diag, n, A, ldA, x);
    }
}


template <typename IndexType, typename T, int N>
inline
void
trmv_low_t_kernel(Diag diag, IndexType n, const T *x, IndexType incX, const T *A, IndexType ldA, T *y)
{

    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;
    const IndexType rest_n = (n-N)%numElements;
    gemv_t_kernel<IndexType, T,N>(n-N-rest_n, T(1), x, incX, A, ldA, y);

    y+=(n-N);
    A+=(n-N);
    for (IndexType k=0; k<N; ++k) {
        const T tmp_x = *x;

        axpy(k+rest_n, tmp_x, A-rest_n, 1, y-rest_n, 1);

        if (diag==NonUnit) {
            y[k] += A[k]*tmp_x;
        } else {
            y[k] += tmp_x;
        }
        x+=incX;
        A+=ldA;
    }
    return;
}

template <typename IndexType, typename T, int N>
inline
void
trmv_low_ct_kernel(Diag diag, IndexType n, const T *x, IndexType incX, const T *A, IndexType ldA, T *y)
{

    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;
    const IndexType rest_n = (n-N)%numElements;
    gemv_ct_kernel<IndexType,T,N>(n-N-rest_n, T(1), x, incX, A, ldA, y);

    y+=(n-N);
    A+=(n-N);
    for (IndexType k=0; k<N; ++k) {
        const T tmp_x = *x;

        acxpy(k+rest_n, tmp_x, A-rest_n, 1, y-rest_n, 1);

        if (diag==NonUnit) {
            y[k] += conjugate(A[k])*tmp_x;
        } else {
            y[k] += tmp_x;
        }
        x+=incX;
        A+=ldA;
    }
    return;
}


template <typename IndexType, typename T, int N>
inline
void
trmv_up_t_kernel(Diag diag, IndexType n, const T *x, IndexType incX, const T *A, IndexType ldA, T *y)
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;
    const IndexType rest_n = (n-N)%numElements;

    for (IndexType k=0; k<N; ++k) {
	const T tmp_xx = x[k*incX];

	if (diag==NonUnit) {
	    y[k] += A[k*(ldA+1)]*tmp_xx;
	} else {
	    y[k] += tmp_xx;
        }
	axpy(N-1-k+rest_n, tmp_xx, A+k*(ldA+1)+1, 1, y+k+1, 1);

    }

    gemv_t_kernel<IndexType,T,N>(n-N-rest_n, T(1), x, incX, A+N+rest_n, ldA, y+N+rest_n);

}

template <typename IndexType, typename T, int N>
inline
void
trmv_up_ct_kernel(Diag diag, IndexType n, const T *x, IndexType incX, const T *A, IndexType ldA, T *y)
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;
    const IndexType rest_n = (n-N)%numElements;

    for (IndexType k=0; k<N; ++k) {
        const T tmp_xx = x[k*incX];

        if (diag==NonUnit) {
            y[k] += conjugate(A[k*(ldA+1)])*tmp_xx;
        } else {
            y[k] += tmp_xx;
        }
        acxpy(N-1-k+rest_n, tmp_xx, A+k*(ldA+1)+1, 1, y+k+1, 1);

    }

    gemv_ct_kernel<IndexType,T,N>(n-N-rest_n, T(1), x, incX, A+N+rest_n, ldA, y+N+rest_n);

}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trmv_up_t_unroller(Diag diag, IndexType m, IndexType n, const T *x, IndexType incX, const T *A, IndexType ldA, T *y) 
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trmv_up_t_unroller(Diag diag, IndexType m, IndexType n, const T *x, IndexType incX, const T *A, IndexType ldA, T *y) 
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=n-N; i+=N) {
            trmv_up_t_kernel<IndexType, T,N>(diag, n-i, x, incX, A, ldA, y);

            x += N*incX;
            y += N;
            A += N*(ldA+1);
        }

        trmv_up_t_unroller<IndexType, T, N/2, false>(diag, m%N, n, x, incX, A, ldA, y);

    } else {
        if (m>=N) {

            trmv_up_t_kernel<IndexType, T,N>(diag, m, x, incX, A, ldA, y);

            x += N*incX;
            y += N;
            A += N*(ldA+1);

            m-=N;
        }
        trmv_up_t_unroller<IndexType, T, N/2, false>(diag, m, n, x, incX, A, ldA, y);
    }
}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trmv_up_ct_unroller(Diag diag, IndexType m, IndexType n, const T *x, IndexType incX, const T *A, IndexType ldA, T *y)
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trmv_up_ct_unroller(Diag diag, IndexType m, IndexType n, const T *x, IndexType incX, const T *A, IndexType ldA, T *y)
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=n-N; i+=N) {
            trmv_up_ct_kernel<IndexType, T,N>(diag, n-i, x, incX, A, ldA, y);

            x += N*incX;
            y += N;
            A += N*(ldA+1);
        }

        trmv_up_ct_unroller<IndexType, T, N/2, false>(diag, m%N, n, x, incX, A, ldA, y);

    } else {
        if (m>=N) {

            trmv_up_ct_kernel<IndexType, T,N>(diag, m, x, incX, A, ldA, y);

            x += N*incX;
            y += N;
            A += N*(ldA+1);

            m-=N;
        }
        trmv_up_ct_unroller<IndexType, T, N/2, false>(diag, m, n, x, incX, A, ldA, y);
    }
}


template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trmv_low_t_unroller(Diag diag, IndexType m, IndexType n, const T *x, IndexType incX, const T *A, IndexType ldA, T *y) 
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trmv_low_t_unroller(Diag diag, IndexType m, IndexType n, const T *x, IndexType incX, const T *A, IndexType ldA, T *y) 
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=n-N; i+=N) {

           trmv_low_t_kernel<IndexType, T,N>(diag, i+N, x, incX, A, ldA, y); 
           x += N*incX;
           A += N*ldA;
        }

        trmv_low_t_unroller<IndexType, T, N/2, false>(diag, m%N, n, x, incX, A, ldA, y);

    } else {
        if (m>=N) {

           trmv_low_t_kernel<IndexType, T,N>(diag, n-m+N, x, incX, A, ldA, y); 
           x += N*incX;
           A += N*ldA;

           m-=N;
        }
        trmv_low_t_unroller<IndexType, T, N/2, false>(diag, m, n, x, incX, A, ldA, y);
    }
}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trmv_low_ct_unroller(Diag diag, IndexType m, IndexType n, const T *x, IndexType incX, const T *A, IndexType ldA, T *y)
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trmv_low_ct_unroller(Diag diag, IndexType m, IndexType n, const T *x, IndexType incX, const T *A, IndexType ldA, T *y)
{
    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
    const IndexType numElements = IntrinsicType::numElements;

    if (firstCall==true) {

        for (IndexType i=0; i<=n-N; i+=N) {

           trmv_low_ct_kernel<IndexType, T,N>(diag, i+N, x, incX, A, ldA, y);
           x += N*incX;
           A += N*ldA;
        }

        trmv_low_ct_unroller<IndexType, T, N/2, false>(diag, m%N, n, x, incX, A, ldA, y);

    } else {
        if (m>=N) {

           trmv_low_ct_kernel<IndexType, T,N>(diag, n-m+N, x, incX, A, ldA, y);
           x += N*incX;
           A += N*ldA;

           m-=N;
        }
        trmv_low_ct_unroller<IndexType, T, N/2, false>(diag, m, n, x, incX, A, ldA, y);
    }
}

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
trmv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const T *A, IndexType ldA,
     T *x, IndexType incX)
{

    if (order==ColMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        trmv(RowMajor, upLo, transA, diag, n, A, ldA, x, incX);
        return;
    }

    if ( upLo==Upper) {

        if  ( transA==NoTrans) {
 
            if ( incX==1 ) {

                trmv_up_n_unroller<IndexType, T>(diag, n, A, ldA, x);

            } else {

                T *tmp_x = new T[n];
                copy(n, x, incX, tmp_x, 1);
                trmv_up_n_unroller<IndexType, T>(diag, n, A, ldA, tmp_x);
                copy(n, tmp_x, 1, x, incX);
                delete[] tmp_x;

            }

        } else if ( transA==Conj ) {

            if ( incX==1 ) {

                trmv_up_c_unroller<IndexType, T>(diag, n, A, ldA, x);

            } else {

                T *tmp_x = new T[n];
                copy(n, x, incX, tmp_x, 1);
                trmv_up_c_unroller<IndexType, T>(diag, n, A, ldA, tmp_x);
                copy(n, tmp_x, 1, x, incX);
                delete[] tmp_x;

            }

        } else if ( transA==Trans ) {

            T *tmp_x = new T[n];
            std::fill_n(tmp_x, n, T(0));
            trmv_up_t_unroller<IndexType, T>(diag, n, n, x, incX, A, ldA, tmp_x); 
            copy(n, tmp_x, 1, x, incX);
            delete[] tmp_x;

        } else if ( transA==ConjTrans ) {

            T *tmp_x = new T[n];
            std::fill_n(tmp_x, n, T(0));
            trmv_up_ct_unroller<IndexType, T>(diag, n, n, x, incX, A, ldA, tmp_x);
            copy(n, tmp_x, 1, x, incX);
            delete[] tmp_x;

        } else {
            ASSERT(0);
        }

    } else if (upLo==Lower) {

        if  ( transA==NoTrans ) {

            if (incX==1) {
                trmv_low_n_unroller<IndexType, T>(diag, n, A, ldA, x);
            } else {
                T *tmp_x = new T[n];
                copy(n, x, incX, tmp_x, 1);
                trmv_low_n_unroller<IndexType, T>(diag, n, A, ldA, tmp_x);
                copy(n, tmp_x, 1, x, incX);
                delete[] tmp_x;
            }
 
       } else if (transA==Conj) {

            if (incX==1) {
                trmv_low_c_unroller<IndexType, T>(diag, n, A, ldA, x);
            } else {
                T *tmp_x = new T[n];
                copy(n, x, incX, tmp_x, 1);
                trmv_low_c_unroller<IndexType, T>(diag, n, A, ldA, tmp_x);
                copy(n, tmp_x, 1, x, incX);
                delete[] tmp_x;
            }
 
        } else if ( transA==Trans ) {

            T *tmp_x = new T[n];
            std::fill_n(tmp_x, n, T(0));
            trmv_low_t_unroller<IndexType, T>(diag, n, n, x, incX, A, ldA, tmp_x); 
            copy(n, tmp_x, 1, x, incX);
            delete[] tmp_x;

        } else if (transA==ConjTrans) {

            T *tmp_x = new T[n];
            std::fill_n(tmp_x, n, T(0));
            trmv_low_ct_unroller<IndexType, T>(diag, n, n, x, incX, A, ldA, tmp_x);
            copy(n, tmp_x, 1, x, incX);
            delete[] tmp_x;

        } else {

            ASSERT(0);

        }
    }
}



#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL2_TRMV_TCC
