/*
 *   Copyright (c) 2013, Klaus Pototzky
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

#ifndef PLAYGROUND_CXXBLAS_DRIVERS_DRIVERS_TCC
#define PLAYGROUND_CXXBLAS_DRIVERS_DRIVERS_TCC 1

#include <cxxblas/auxiliary/auxiliary.h>
#include <cxxblas/drivers/drivers.h>
#include <playground/cxxblas/drivers/drivers.h>

namespace cxxblas {
  
#ifdef HAVE_CLBLAS

namespace CLBLAS {

//TODO: rename these to getCblasEnum

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,StorageOrder>::value, CLBLAS_IMPL(Order)>::Type
getClblasType(ENUM order)
{
    if (order==RowMajor) {
        return CLBLAS_IMPL(RowMajor);
    }
    return CLBLAS_IMPL(ColumnMajor);
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Transpose>::value, CLBLAS_IMPL(Transpose)>::Type
getClblasType(ENUM trans)
{
    if (trans==NoTrans) {
        return CLBLAS_IMPL(NoTrans);
    }
    if (trans==Conj) {
        ASSERT(0);
        //return CLBLAS_IMPL(ConjNoTrans);
    }
    if (trans==Trans) {
        return CLBLAS_IMPL(Trans);
    }
    return CLBLAS_IMPL(ConjTrans);
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,StorageUpLo>::value, CLBLAS_IMPL(Uplo)>::Type
getClblasType(ENUM upLo)
{
    if (upLo==Upper) {
        return CLBLAS_IMPL(Upper);
    }
    return CLBLAS_IMPL(Lower);
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Side>::value, CLBLAS_IMPL(Side)>::Type
getClblasType(ENUM side)
{
    if (side==Left) {
        return CLBLAS_IMPL(Left);
    }
    return CLBLAS_IMPL(Right);
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Diag>::value, CLBLAS_IMPL(Diag)>::Type
getClblasType(ENUM diag)
{
    if (diag==Unit) {
        return  CLBLAS_IMPL(Unit);
    }
    return  CLBLAS_IMPL(NonUnit);
}

} // namespace CLBLAS

#endif // HAVE_CLBLAS


#ifdef HAVE_CUBLAS

namespace CUBLAS {

//TODO: rename these to getCblasEnum

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Transpose>::value, cublasOperation_t>::Type
getCublasType(ENUM trans)
{
    if (trans==NoTrans) {
        return CUBLAS_OP_N;
    }
    if (trans==Conj) {
        ASSERT(0);
//         return CUBLAS_OP_R;
    }
    if (trans==Trans) {
        return CUBLAS_OP_T;
    }
    return CUBLAS_OP_C;
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,StorageUpLo>::value, cublasFillMode_t>::Type
getCublasType(ENUM upLo)
{
    if (upLo==Upper) {
        return CUBLAS_FILL_MODE_UPPER;
    }
    return CUBLAS_FILL_MODE_LOWER;
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Side>::value, cublasSideMode_t>::Type
getCublasType(ENUM side)
{
    if (side==Left) {
        return CUBLAS_SIDE_LEFT;
    }
    return CUBLAS_SIDE_RIGHT;
}

template <typename ENUM>
typename RestrictTo<IsSame<ENUM,Diag>::value, cublasDiagType_t>::Type
getCublasType(ENUM diag)
{
    if (diag==Unit) {
        return CUBLAS_DIAG_UNIT;
    }
    return CUBLAS_DIAG_NON_UNIT;
}

} // namespace CUBLAS

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_DRIVERS_DRIVERS_TCC
