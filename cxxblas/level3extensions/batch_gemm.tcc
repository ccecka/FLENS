/*
 *   Copyright (c) 2015, Cris Cecka
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

#ifndef CXXBLAS_LEVEL3EXTENSION_BATCHEDGEMM_TCC
#define CXXBLAS_LEVEL3EXTENSION_BATCHEDGEMM_TCC 1

#include <cxxstd/complex.h>
#include <cxxblas/cxxblas.h>

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename MA, typename MB,
          typename BETA, typename MC>
void
batch_gemm(StorageOrder order, Transpose transA, Transpose transB,
           IndexType m, IndexType n, IndexType k,
           const ALPHA &alpha,
           const MA *A, IndexType ldA, IndexType loA,
           const MB *B, IndexType ldB, IndexType loB,
           const BETA &beta,
           MC *C, IndexType ldC, IndexType loC,
           IndexType p)
{
  // TODO: Parallel
  for (int i = 0; i < p; ++i) {
    gemm(order,
         transA, transB,
         m, n, k,
         alpha,
         A + i*loA, ldA,
         B + i*loB, ldB,
         beta,
         C + i*loC, ldC);
  }
}

#ifdef HAVE_CUBLAS

// sgemm
template <typename IndexType>
void
batch_gemm(StorageOrder order, Transpose transA, Transpose transB,
           IndexType m, IndexType n, IndexType k,
           float alpha,
           const thrust::device_ptr<const float> A, IndexType ldA, IndexType loA,
           const thrust::device_ptr<const float> B, IndexType ldB, IndexType loB,
           float beta,
           thrust::device_ptr<float> C, IndexType ldC, IndexType loC,
           IndexType p)
{
  // XXX: TODO
  assert(false);
}

// dgemm
template <typename IndexType>
void
batch_gemm(StorageOrder order, Transpose transA, Transpose transB,
           IndexType m, IndexType n, IndexType k,
           double alpha,
           const thrust::device_ptr<const double> A, IndexType ldA, IndexType loA,
           const thrust::device_ptr<const double> B, IndexType ldB, IndexType loB,
           double beta,
           thrust::device_ptr<double> C, IndexType ldC, IndexType loC,
           IndexType p)
{
  // XXX: TODO
  assert(false);
}

// cgemm
template <typename IndexType>
void
batch_gemm(StorageOrder order, Transpose transA, Transpose transB,
           IndexType m, IndexType n, IndexType k,
           const ComplexFloat &alpha,
           const thrust::device_ptr<const ComplexFloat> A, IndexType ldA, IndexType loA,
           const thrust::device_ptr<const ComplexFloat> B, IndexType ldB, IndexType loB,
           const ComplexFloat &beta,
           thrust::device_ptr<ComplexFloat> C, IndexType ldC, IndexType loC,
           IndexType p)
{
  // XXX: TODO
  assert(false);
}

// zgemm
template <typename IndexType>
void
batch_gemm(StorageOrder order, Transpose transA, Transpose transB,
           IndexType m, IndexType n, IndexType k,
           const ComplexDouble &alpha,
           const thrust::device_ptr<const ComplexDouble> A, IndexType ldA, IndexType loA,
           const thrust::device_ptr<const ComplexDouble> B, IndexType ldB, IndexType loB,
           const ComplexDouble &beta,
           thrust::device_ptr<ComplexDouble> C, IndexType ldC, IndexType loC,
           IndexType p)
{
  // XXX: TODO
  assert(false);
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL3EXTENSION_BATCHEDGEMM_TCC
