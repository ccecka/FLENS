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

#ifndef PLAYGROUND_CXXBLAS_AUXILIARY_CAST_TCC
#define PLAYGROUND_CXXBLAS_AUXILIARY_CAST_TCC 1

#include <playground/cxxblas/auxiliary/cast.h>

namespace cxxblas {
  
#ifdef HAVE_OPENCL

template <typename T>
flens::device_ptr<T, flens::StorageType::OpenCL>
cast_ptr_to_real(flens::device_ptr<T, flens::StorageType::OpenCL> x)
{
    return x;
}

template <typename T>
flens::device_ptr<T, flens::StorageType::OpenCL>
cast_ptr_to_real(flens::device_ptr<std::complex<T>, flens::StorageType::OpenCL> x)
{
    flens::device_ptr<T, flens::StorageType::OpenCL> ptr(x.get(), 2*x.getOffset());
    return ptr; 
}

template <typename T>
flens::device_ptr<T, flens::StorageType::OpenCL>
cast_ptr_to_imag(flens::device_ptr<T, flens::StorageType::OpenCL> x)
{
    ASSERT(0);
    return x;
}

template <typename T>
flens::device_ptr<T, flens::StorageType::OpenCL>
cast_ptr_to_imag(flens::device_ptr<std::complex<T>, flens::StorageType::OpenCL> x)
{
    flens::device_ptr<T, flens::StorageType::OpenCL> ptr(x.get(), 2*x.getOffset()+1);
    return ptr; 
}

#endif // HAVE_OPENCL

#ifdef HAVE_CUBLAS

template <typename T>
flens::device_ptr<T, flens::StorageType::CUDA>
cast_ptr_to_real(flens::device_ptr<T, flens::StorageType::CUDA> x)
{
    return x; 
}

template <typename T>
flens::device_ptr<T, flens::StorageType::CUDA>
cast_ptr_to_real(flens::device_ptr<std::complex<T>, flens::StorageType::CUDA> x)
{
    return ptr; 
}

template <typename T>
flens::device_ptr<T, flens::StorageType::CUDA>
cast_ptr_to_imag(flens::device_ptr<T, flens::StorageType::CUDA> x)
{
    ASSERT(0);
    return x;
}

template <typename T>
flens::device_ptr<T, flens::StorageType::CUDA>
cast_ptr_to_imag(flens::device_ptr<std::complex<T>, flens::StorageType::CUDA> x)
{
    return ptr; 
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_AUXILIARY_CAST_TCC
