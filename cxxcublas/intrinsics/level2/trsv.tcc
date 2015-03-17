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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL2_TRSV_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL2_TRSV_TCC 1

#include <iostream>

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/intrinsics/auxiliary/auxiliary.h>
#include <playground/cxxblas/intrinsics/includes.h>

namespace cxxblas {

#ifdef USE_INTRINSIC

template <typename IndexType, typename T, int N>
inline
void
trsv_up_n_kernel(Diag diag, IndexType n, IndexType i, const T *A, IndexType ldA, T *x)
{

    gemv_n_kernel<IndexType, T, N>(n-i-1, T(-1), x+i+1, A+(i-N+1)*ldA+i+1, ldA, x+i-N+1, 1);

    for (IndexType k=0; k<N; ++k) {
        for (IndexType j=0; j<k; ++j) {
           x[i-k] -= A[(i-k)*(ldA+1)+1+j]*x[i-k+j+1];
        }
        if(diag==NonUnit) {
            x[i-k] /= A[(i-k)*(ldA+1)];
        }
    }
}

template <typename IndexType, typename T, int N>
inline
void
trsv_up_c_kernel(Diag diag, IndexType n, IndexType i, const T *A, IndexType ldA, T *x)
{
    gemv_c_kernel<IndexType, T, N>(n-i-1, T(-1), x+i+1, A+(i-N+1)*ldA+i+1, ldA, x+i-N+1, 1);

    for (IndexType k=0; k<N; ++k) {
        for (IndexType j=0; j<k; ++j) {
           x[i-k] -= conjugate(A[(i-k)*(ldA+1)+1+j])*x[i-k+j+1];
        }
        if(diag==NonUnit) {
            x[i-k] /= conjugate(A[(i-k)*(ldA+1)]);
        }
    }
}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trsv_up_n_unroller(Diag diag, IndexType n, IndexType m,
                   const T *A, IndexType ldA, T *x)
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trsv_up_n_unroller(Diag diag, IndexType n, IndexType m,
                   const T *A, IndexType ldA, T *x)
{
    if (firstCall==true) {
        
        for (IndexType i=n-1; i>=N-1; i-=N) {
            trsv_up_n_kernel<IndexType, T, N>(diag, n, i, A, ldA, x);
        }
        trsv_up_n_unroller<IndexType, T, N/2, false>(diag, n, n%N-1, A, ldA, x);
    } else {

	if (m>=N-1) {
            trsv_up_n_kernel<IndexType, T, N>(diag, n, m, A, ldA, x);
            m -= N;
	}
        trsv_up_n_unroller<IndexType, T, N/2, false>(diag, n, m, A, ldA, x);

    }
}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trsv_up_c_unroller(Diag diag, IndexType n, IndexType m,
                   const T *A, IndexType ldA, T *x)
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trsv_up_c_unroller(Diag diag, IndexType n, IndexType m,
                   const T *A, IndexType ldA, T *x)
{
    if (firstCall==true) {
        
        for (IndexType i=n-1; i>=N-1; i-=N) {
            trsv_up_c_kernel<IndexType, T, N>(diag, n, i, A, ldA, x);
        }
        trsv_up_c_unroller<IndexType, T, N/2, false>(diag, n, n%N-1, A, ldA, x);
    } else {

	if (m>=N-1) {
            trsv_up_c_kernel<IndexType, T, N>(diag, n, m, A, ldA, x);
            m -= N;
	}
        trsv_up_c_unroller<IndexType, T, N/2, false>(diag, n, m, A, ldA, x);

    }
}


template <typename IndexType, typename T, int N>
inline
void
trsv_low_n_kernel(Diag diag, IndexType n,
                  const T *A, IndexType ldA, T *x)
{

    gemv_n_kernel<IndexType, T, N>(n, T(-1), x, A, ldA, x+n, 1);

    for (IndexType j=0; j<N; ++j) {
        T _x(0);
        for (IndexType k=n; k<n+j; ++k) {
            _x += A[j*ldA+k]*x[k];
        }
        x[n+j] -= _x;
	if (diag==NonUnit) {
	    x[n+j] /= A[n+j*(ldA+1)];
	}
    }

}


template <typename IndexType, typename T, int N>
inline
void
trsv_low_c_kernel(Diag diag, IndexType n,
                  const T *A, IndexType ldA, T *x)
{
    gemv_c_kernel<IndexType, T, N>(n, T(-1), x, A, ldA, x+n, 1);

    for (IndexType j=0; j<N; ++j) {
        T _x(0);
        for (IndexType k=n; k<n+j; ++k) {
            _x += conjugate(A[j*ldA+k])*x[k];
        }
        x[n+j] -= _x;
	if (diag==NonUnit) {
	    x[n+j] /= conjugate(A[n+j*(ldA+1)]);
	}
    }
}


template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trsv_low_n_unroller(Diag diag, IndexType n, IndexType m,
                   const T *A, IndexType ldA, T *x)
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trsv_low_n_unroller(Diag diag, IndexType n, IndexType m, 
                    const T *A, IndexType ldA, T *x)
{
    if (firstCall==true) {

        for (IndexType i=0; i<=n-N; i+=N) {
            trsv_low_n_kernel<IndexType, T, N>(diag, i, A, ldA, x); 
            A += N*ldA;
        }
        trsv_low_n_unroller<IndexType, T, N/2, false>(diag, n, m%N, A, ldA, x);

    } else {
	if (m>=N) {
            trsv_low_n_kernel<IndexType, T, N>(diag, n-m, A, ldA, x); 
            A += N*ldA;
            m -= N;
        }
        trsv_low_n_unroller<IndexType, T, N/2, false>(diag, n, m, A, ldA, x);
    }
}


template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trsv_low_c_unroller(Diag diag, IndexType n, IndexType m,
                   const T *A, IndexType ldA, T *x)
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trsv_low_c_unroller(Diag diag, IndexType n, IndexType m,
                   const T *A, IndexType ldA, T *x)
{
    if (firstCall==true) {

        for (IndexType i=0; i<=n-N; i+=N) {
            trsv_low_c_kernel<IndexType, T, N>(diag, i, A, ldA, x); 
            A += N*ldA;
        }
        trsv_low_c_unroller<IndexType, T, N/2, false>(diag, n, m%N, A, ldA, x);

    } else {
	if (m>=N) {
            trsv_low_c_kernel<IndexType, T, N>(diag, n-m, A, ldA, x); 
            A += N*ldA;
            m -= N;
        }
        trsv_low_c_unroller<IndexType, T, N/2, false>(diag, n, m, A, ldA, x);
    }
}





template <typename IndexType, typename T, int N>
inline
void
trsv_up_t_kernel(Diag diag, IndexType n, IndexType i, T *x, IndexType incX, const T *A, IndexType ldA)
{ 
    for (IndexType j=0; j<N; ++j) {
        if(diag==NonUnit) {
            x[i+j] /= A[(i+j)*(ldA+1)];
        }
        for (IndexType k=i+j+1; k<i+N; ++k) {
            x[k] -= A[k+(i+j)*ldA]*x[i+j];
        }
    }

    gemv_t_kernel<IndexType, T, N>(n-i-N, T(-1), x+i, 1, A+N+i*(ldA+1), ldA, x+i+N);
}

template <typename IndexType, typename T, int N>
inline
void
trsv_up_ct_kernel(Diag diag, IndexType n, IndexType i, T *x, IndexType incX, const T *A, IndexType ldA)
{
    for (IndexType j=0; j<N; ++j) {
        if(diag==NonUnit) {
            x[i+j] /= conjugate(A[(i+j)*(ldA+1)]);
        }
        for (IndexType k=i+j+1; k<i+N; ++k) {
            x[k] -= conjugate(A[k+(i+j)*ldA])*x[i+j];
        }
    }

    gemv_ct_kernel<IndexType, T, N>(n-i-N, T(-1), x+i, 1, A+N+i*(ldA+1), ldA, x+i+N);
}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trsv_up_t_unroller(Diag diag, IndexType n, IndexType m, T *x, IndexType incX, const T *A, IndexType ldA) 
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trsv_up_t_unroller(Diag diag, IndexType n, IndexType m, T *x, IndexType incX, const T *A, IndexType ldA) 
{

    if (firstCall==true) {
        for (IndexType i=0; i<=n-N; i+=N) {
            trsv_up_t_kernel<IndexType, T, N>(diag, n, i, x, 1, A, ldA);
        }
        trsv_up_t_unroller<IndexType, T, N/2, false>(diag, n, n-(n%N), x, incX, A, ldA);
    } else {
        if (m<=n-N) {
            trsv_up_t_kernel<IndexType, T, N>(diag, n, m, x, 1, A, ldA);
            m+=N;
        }
        trsv_up_t_unroller<IndexType, T, N/2, false>(diag, n, m, x, incX, A, ldA);
    }
}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trsv_up_ct_unroller(Diag diag, IndexType n, IndexType m, T *x, IndexType incX, const T *A, IndexType ldA) 
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trsv_up_ct_unroller(Diag diag, IndexType n, IndexType m, T *x, IndexType incX, const T *A, IndexType ldA) 
{
    if (firstCall==true) {
        for (IndexType i=0; i<=n-N; i+=N) {
            trsv_up_ct_kernel<IndexType, T, N>(diag, n, i, x, 1, A, ldA);
        }
        trsv_up_ct_unroller<IndexType, T, N/2, false>(diag, n, n-(n%N), x, incX, A, ldA);
    } else {
        if (m<=n-N) {
            trsv_up_ct_kernel<IndexType, T, N>(diag, n, m, x, 1, A, ldA);
            m+=N;
        }
        trsv_up_ct_unroller<IndexType, T, N/2, false>(diag, n, m, x, incX, A, ldA);
    }
}

template <typename IndexType, typename T, int N>
inline
void
trsv_low_ct_kernel(Diag diag, IndexType n, IndexType i, T *x, IndexType incX, const T *A, IndexType ldA)
{

    for (IndexType k=0; k<N; ++k) {
        if (diag==NonUnit) {
            x[i-k] /= conjugate(A[(i-k)*(ldA+1)]);
        }
        for (IndexType j=i-N+1; j<i-k; ++j) {
            x[j] -= conjugate(A[(i-k)*ldA+j])*x[i-k];
        }
    }

    gemv_ct_kernel<IndexType, T, N>(i-N+1, T(-1), x+i-N+1, 1, A+(i-N+1)*ldA, ldA, x);
}

template <typename IndexType, typename T, int N>
inline
void
trsv_low_t_kernel(Diag diag, IndexType n, IndexType i, T *x, IndexType incX, const T *A, IndexType ldA)
{
    for (IndexType k=0; k<N; ++k) {
        if (diag==NonUnit) {
            x[i-k] /= A[(i-k)*(ldA+1)];
        }
        for (IndexType j=i-N+1; j<i-k; ++j) {
            x[j] -= A[(i-k)*ldA+j]*x[i-k];
        }
    }

    gemv_t_kernel<IndexType, T, N>(i-N+1, T(-1), x+i-N+1, 1, A+(i-N+1)*ldA, ldA, x);

}
template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trsv_low_t_unroller(Diag diag, IndexType n, IndexType m, T *x, IndexType incX, const T *A, IndexType ldA) 
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trsv_low_t_unroller(Diag diag, IndexType n, IndexType m, T *x, IndexType incX, const T *A, IndexType ldA) 
{

    if (firstCall==true) {
        for (IndexType i=n-1; i>=N-1; i-=N) {
            trsv_low_t_kernel<IndexType, T, N>(diag, n, i, x, 1, A, ldA);
        }
        trsv_low_t_unroller<IndexType, T, N/2, false>(diag, n, n%N-1, x, incX, A, ldA);
    } else {
        if (m>=N-1) {
            trsv_low_t_kernel<IndexType, T, N>(diag, n, m, x, 1, A, ldA);
            m-=N;
        }
        trsv_low_t_unroller<IndexType, T, N/2, false>(diag, n, m, x, incX, A, ldA);
    }
}

template <typename IndexType, typename T,
          int N, bool firstCall>
inline
typename flens::RestrictTo<IsSameInt<N,0>::value,
                           void>::Type
trsv_low_ct_unroller(Diag diag, IndexType n, IndexType m, T *x, IndexType incX, const T *A, IndexType ldA) 
{

}

template <typename IndexType, typename T,
          int N = 4, bool firstCall = true>
inline
typename flens::RestrictTo<!IsSameInt<N,0>::value,
                           void>::Type
trsv_low_ct_unroller(Diag diag, IndexType n, IndexType m, T *x, IndexType incX, const T *A, IndexType ldA) 
{

    if (firstCall==true) {
        for (IndexType i=n-1; i>=N-1; i-=N) {
            trsv_low_ct_kernel<IndexType, T, N>(diag, n, i, x, 1, A, ldA);
        }
        trsv_low_ct_unroller<IndexType, T, N/2, false>(diag, n, n%N-1, x, incX, A, ldA);
    } else {
        if (m>=N-1) {
            trsv_low_ct_kernel<IndexType, T, N>(diag, n, m, x, 1, A, ldA);
            m-=N;
        }
        trsv_low_ct_unroller<IndexType, T, N/2, false>(diag, n, m, x, incX, A, ldA);
    }
}

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const T *A, IndexType ldA,
     T *x, IndexType incX)
{

    if (order==ColMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        trsv(RowMajor, upLo, transA, diag, n, A, ldA, x, incX);
        return;
    }

    if (incX!=1) {
        T *tmp_x = new T[n];
        copy(n, x, incX, tmp_x, 1);
        trsv(order, upLo, transA, diag, n, A, ldA, tmp_x, 1);
        copy(n, tmp_x, 1, x, incX);
        delete[] tmp_x;
        return;
    }

    ASSERT(incX==1);

    if ( upLo==Upper) {

        if  ( transA==NoTrans) {

            trsv_up_n_unroller<IndexType, T>(diag, n, n, A, ldA, x);


        } else if ( transA==Conj ) {

            trsv_up_c_unroller<IndexType, T>(diag, n, n, A, ldA, x);

        } else if ( transA==Trans ) {

            trsv_up_t_unroller<IndexType, T>(diag, n, 0, x, 1, A, ldA); 

        } else if ( transA==ConjTrans ) {

            trsv_up_ct_unroller<IndexType, T>(diag, n, 0, x, 1, A, ldA); 

        } else {
            ASSERT(0);
        }

    } else if (upLo==Lower) {

        if  ( transA==NoTrans ) {

            trsv_low_n_unroller<IndexType, T>(diag, n, n, A, ldA, x);

 
       } else if (transA==Conj) {

            trsv_low_c_unroller<IndexType, T>(diag, n, n, A, ldA, x);

        } else if ( transA==Trans ) {

            trsv_low_t_unroller<IndexType, T>(diag, n, n, x, 1, A, ldA);

        } else if (transA==ConjTrans) {

            trsv_low_ct_unroller<IndexType, T>(diag, n, n, x, 1, A, ldA);

        } else {

            ASSERT(0);

        }
    }
}



#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL2_TRSV_TCC
