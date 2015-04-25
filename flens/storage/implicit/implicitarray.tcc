/*
 *   Copyright (c) 2007, Michael Lehn
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

#ifndef FLENS_STORAGE_IMPLICIT_IMPLICITARRAY_TCC
#define FLENS_STORAGE_IMPLICIT_IMPLICITARRAY_TCC 1

#include <cxxstd/cassert.h>
#include <cxxstd/algorithm.h>
#include <cxxblas/level1/copy.h>
#include <flens/auxiliary/auxiliary.h>
#include <flens/storage/array/array.h>
#include <flens/storage/array/arrayview.h>
#include <flens/storage/array/constarrayview.h>


namespace flens {

template <typename F, typename I>
ImplicitArray<F, I>::ImplicitArray()
    : functor_(),
      length_(0),
      firstIndex_(1)
{
}

template <typename F, typename I>
ImplicitArray<F, I>::ImplicitArray(IndexType length,
                                   const FunctorType& f,
                                   IndexType firstIndex)
    : functor_(),
      length_(length),
      firstIndex_(firstIndex)
{
    ASSERT(length_>=0);
}

template <typename F, typename I>
ImplicitArray<F, I>::ImplicitArray(const ImplicitArray &rhs)
    : functor_(rhs.functor()),
      length_(rhs.length()),
      firstIndex_(rhs.firstIndex())
{
    ASSERT(length_>=0);
}

//-- operators -----------------------------------------------------------------

template <typename F, typename I>
typename ImplicitArray<F, I>::const_reference
ImplicitArray<F, I>::operator()(IndexType index) const
{
    ASSERT(index>=firstIndex());
    ASSERT(index<=lastIndex());
    return functor_(index-firstIndex_);
}

template <typename F, typename I>
typename ImplicitArray<F, I>::reference
ImplicitArray<F, I>::operator()(IndexType index)
{
    ASSERT(index>=firstIndex());
    ASSERT(index<=lastIndex());
    return functor_(index-firstIndex_);
}

template <typename F, typename I>
typename ImplicitArray<F, I>::IndexType
ImplicitArray<F, I>::firstIndex() const
{
    return firstIndex_;
}

template <typename F, typename I>
typename ImplicitArray<F, I>::IndexType
ImplicitArray<F, I>::lastIndex() const
{
    return firstIndex_+length_-IndexType(1);
}

template <typename F, typename I>
typename ImplicitArray<F, I>::IndexType
ImplicitArray<F, I>::length() const
{
    return length_;
}

template <typename F, typename I>
typename ImplicitArray<F, I>::IndexType
ImplicitArray<F, I>::stride() const
{
    return IndexType(1);
}

template <typename F, typename I>
const typename ImplicitArray<F, I>::FunctorType &
ImplicitArray<F, I>::functor() const
{
    return functor_;
}

template <typename F, typename I>
typename ImplicitArray<F, I>::FunctorType &
ImplicitArray<F, I>::functor()
{
    return functor_;
}

template <typename F, typename I>
std::allocator<typename ImplicitArray<F, I>::ElementType>
ImplicitArray<F, I>::allocator() const
{
  return {};
}

template <typename F, typename I>
bool
ImplicitArray<F, I>::resize(IndexType length, IndexType firstIndex)
{
    if (length!=length_) {
        length_ = length;
        firstIndex_ = firstIndex;
        return true;
    }
    changeIndexBase(firstIndex);
    return false;
}

template <typename F, typename I>
template <typename ARRAY>
bool
ImplicitArray<F, I>::resize(const ARRAY &rhs)
{
    return resize(rhs.length(), rhs.firstIndex());
}

template <typename F, typename I>
void
ImplicitArray<F, I>::changeIndexBase(IndexType firstIndex)
{
    firstIndex_ = firstIndex;
}

template <typename F, typename I>
const typename ImplicitArray<F, I>::ConstView
ImplicitArray<F, I>::view(IndexType from, IndexType to,
                          IndexType stride, IndexType firstViewIndex) const
{
    const IndexType length = (to-from)/stride+1;

    return ConstView(length,
                     functor(),
                     from-firstIndex_,
                     stride,
                     firstViewIndex);
}

template <typename F, typename I>
typename ImplicitArray<F, I>::View
ImplicitArray<F, I>::view(IndexType from, IndexType to,
                          IndexType stride, IndexType firstViewIndex)
{
    const IndexType     length = (to-from)/stride+1;

    return View(length,
                functor(),
                from-firstIndex_,
                stride,
                firstViewIndex);
}

} // namespace flens

#endif // FLENS_STORAGE_IMPLICIT_IMPLICITARRAY_TCC
