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

#ifndef PLAYGROUND_CXXBLAS_DRIVERS_DRIVERS_H
#define PLAYGROUND_CXXBLAS_DRIVERS_DRIVERS_H 1


#include <cxxblas/drivers/drivers.h>

// define implementation specific constants, macros, etc.
#ifdef WITH_CUBLAS
#   ifndef HAVE_CUBLAS
#       define HAVE_CUBLAS
#   endif
#   ifndef HAVE_CUDA
#       define HAVE_CUDA
#   endif
#   ifndef HAVE_DEVICE_STORAGE
#       define HAVE_DEVICE_STORAGE
#   endif
#   ifndef DEFAULT_DEVICE_STORAGE_TYPE
#       define DEFAULT_DEVICE_STORAGE_TYPE CUDA
#   endif
#   include <cublas_v2.h>
#endif

#ifdef WITH_CLAMDBLAS
#   ifndef HAVE_CLBLAS
#       define HAVE_CLBLAS
#   endif
#   ifndef HAVE_OPENCL
#       define HAVE_OPENCL
#   endif
#   ifndef HAVE_DEVICE_STORAGE
#       define HAVE_DEVICE_STORAGE
#   endif
#   ifndef DEFAULT_DEVICE_STORAGE_TYPE
#       define DEFAULT_DEVICE_STORAGE_TYPE OpenCL
#   endif
#   ifndef CLBLAS_IMPL
#       define CLBLAS_IMPL(x)         clAmdBlas##x
#   endif
#   ifndef CLBLAS_IMPL_EX
#       define CLBLAS_IMPL_EX(str)      clAmdBlas##str##Ex
#   endif
#   include <clAmdBlas.h>
#endif

#ifdef WITH_CLBLAS
#   ifndef HAVE_CLBLAS
#       define HAVE_CLBLAS
#   endif
#   ifndef HAVE_OPENCL
#       define HAVE_OPENCL
#   endif
#   ifndef HAVE_DEVICE_STORAGE
#       define HAVE_DEVICE_STORAGE
#   endif
#   ifndef DEFAULT_DEVICE_STORAGE_TYPE
#       define DEFAULT_DEVICE_STORAGE_TYPE OpenCL
#   endif
#   ifndef CLBLAS_IMPL
#       define CLBLAS_IMPL(x)         clblas##x
#   endif
#   ifndef CLBLAS_IMPL_EX
#       define CLBLAS_IMPL_EX(x)      clAmdBlas##x
#   endif
#   include <clBlas.h>
#endif

namespace cxxblas {
  
#ifdef HAVE_CLBLAS

namespace CLBLAS {

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,StorageOrder>::value, CLBLAS_IMPL(Order)>::Type
    getClblasType(ENUM order);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Transpose>::value, CLBLAS_IMPL(Transpose)>::Type
    getClblasType(ENUM trans);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,StorageUpLo>::value, CLBLAS_IMPL(Uplo)>::Type
    getClblasType(ENUM upLo);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Side>::value, CLBLAS_IMPL(Side)>::Type
    getClblasType(ENUM side);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Diag>::value, CLBLAS_IMPL(Diag)>::Type
    getClblasType(ENUM diag);

} // namespace CLBLAS

#endif // HAVE_CLBLAS


#ifdef HAVE_CUBLAS

namespace CUBLAS {

//TODO: rename these to getCblasEnum

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Transpose>::value, cublasOperation_t>::Type
    getCublasType(ENUM trans);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,StorageUpLo>::value, cublasFillMode_t>::Type
    getCublasType(ENUM upLo);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Side>::value, cublasSideMode_t>::Type
    getCublasType(ENUM side);

template <typename ENUM>
    typename RestrictTo<IsSame<ENUM,Diag>::value, cublasDiagType_t>::Type
    getCublasType(ENUM diag);

} // namespace CUBLAS

#endif // HAVE_CUBLAS

} // namespace cxxblas


#endif // PLAYGROUND_CXXBLAS_DRIVERS_DRIVERS_H
