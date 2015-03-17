/*
 *   Copyright (c) 2010, Michael Lehn
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

#ifndef PLAYGROUND_CXXBLAS_LEVEL1EXTENSIONS_GECOPY_H
#define PLAYGROUND_CXXBLAS_LEVEL1EXTENSIONS_GECOPY_H 1

#include <playground/cxxblas/drivers/drivers.h>
#include <cxxblas/typedefs.h>

#include <playground/flens/auxiliary/devicePtr.h>

namespace cxxblas {

//
//  B = A  or B = A^T
//

#if defined(HAVE_CLBLAS) || defined(HAVE_CUBLAS)

//
// Copy host -> device
//
template <typename IndexType, typename T, flens::StorageType STORAGETYPE>
    void
    gecopy(StorageOrder order,
           Transpose trans, IndexType m, IndexType n,
           const T *A, IndexType ldA,
           flens::device_ptr<T, STORAGETYPE> B, IndexType ldB);
 
//
// Copy device -> host
//
template <typename IndexType, typename T, flens::StorageType STORAGETYPE>
    void
    gecopy(StorageOrder order,
           Transpose trans, IndexType m, IndexType n,
           const flens::device_ptr<const T, STORAGETYPE> A, IndexType ldA,
           T *B, IndexType ldB);
//
// Copy within the device
//
template <typename IndexType, typename T, flens::StorageType STORAGETYPE>
    void
    gecopy(StorageOrder order,
           Transpose trans, IndexType m, IndexType n,
           const flens::device_ptr<const T, STORAGETYPE> A, IndexType ldA,
           flens::device_ptr<T, STORAGETYPE> B, IndexType ldB);
    
#endif // HAVE_CLBLAS or HAVE_CUBLAS
  

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL1EXTENSIONS_GECOPY_H
