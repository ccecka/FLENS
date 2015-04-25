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

#ifndef FLENS_STORAGE_IMPLICIT_IMPLICITARRAYVIEW_TCC
#define FLENS_STORAGE_IMPLICIT_IMPLICITARRAYVIEW_TCC 1

#include <cxxstd/cassert.h>
#include <cxxblas/level1/copy.h>
#include <flens/auxiliary/auxiliary.h>
#include <flens/storage/implicit/implicitarray.h>
#include <flens/storage/implicit/implicitarrayview.h>
#include <flens/storage/implicit/constimplicitarrayview.h>

namespace flens {

template <typename F, typename I>
ImplicitArrayView<F, I>::ImplicitArrayView(IndexType length,
                                           FunctorType &functor,
                                           IndexType from,
                                           IndexType stride,
                                           IndexType firstIndex)
    : functor_(functor),
      length_(length),
      from_(from),
      stride_(stride),
      firstIndex_(firstIndex)
{
    ASSERT(length_>=0);
    ASSERT(stride_>0);
}

template <typename F, typename I>
ImplicitArrayView<F, I>::ImplicitArrayView(const ImplicitArrayView &rhs)
    : functor_(rhs.functor_),
      length_(rhs.length_),
      from_(rhs.from_),
      stride_(rhs.stride()),
      firstIndex_(rhs.firstIndex())
{
    ASSERT(stride_>0);
}

//-- operators -----------------------------------------------------------------

template <typename F, typename I>
typename ImplicitArrayView<F, I>::const_reference
ImplicitArrayView<F, I>::operator()(IndexType index) const
{
#   ifndef NDEBUG
    if (lastIndex()>=firstIndex()) {
        ASSERT(index>=firstIndex());
        ASSERT(index<=lastIndex());
    }
#   endif

    return functor_(from_+stride_*(index-firstIndex_));
}

template <typename F, typename I>
typename ImplicitArrayView<F, I>::reference
ImplicitArrayView<F, I>::operator()(IndexType index)
{
#   ifndef NDEBUG
    if (lastIndex()>=firstIndex()) {
        ASSERT(index>=firstIndex());
        ASSERT(index<=lastIndex());
    }
#   endif

    return functor_(from_+stride_*(index-firstIndex_));
}

template <typename F, typename I>
typename ImplicitArrayView<F, I>::IndexType
ImplicitArrayView<F, I>::firstIndex() const
{
    return firstIndex_;
}

template <typename F, typename I>
typename ImplicitArrayView<F, I>::IndexType
ImplicitArrayView<F, I>::lastIndex() const
{
    return firstIndex_+length_-1;
}

template <typename F, typename I>
typename ImplicitArrayView<F, I>::IndexType
ImplicitArrayView<F, I>::length() const
{
    return length_;
}

template <typename F, typename I>
typename ImplicitArrayView<F, I>::IndexType
ImplicitArrayView<F, I>::stride() const
{
    return stride_;
}

template <typename F, typename I>
const typename ImplicitArrayView<F, I>::FunctorType &
ImplicitArrayView<F, I>::functor() const
{
    return functor_;
}

template <typename F, typename I>
typename ImplicitArrayView<F, I>::FunctorType &
ImplicitArrayView<F, I>::functor()
{
    return functor_;
}

template <typename F, typename I>
std::allocator<typename ImplicitArrayView<F, I>::ElementType>
ImplicitArrayView<F, I>::allocator() const
{
    return {};
}

template <typename F, typename I>
bool
ImplicitArrayView<F, I>::resize(IndexType DEBUG_VAR(length),
                                IndexType firstIndex)
{
    ASSERT(length==length_);

    changeIndexBase(firstIndex);
    return false;
}

template <typename F, typename I>
template <typename ARRAY>
bool
ImplicitArrayView<F, I>::resize(const ARRAY &rhs)
{
    return resize(rhs.length(), rhs.firstIndex());
}

template <typename F, typename I>
void
ImplicitArrayView<F, I>::changeIndexBase(IndexType firstIndex)
{
    firstIndex_ = firstIndex;
}

template <typename F, typename I>
const typename ImplicitArrayView<F, I>::ConstView
ImplicitArrayView<F, I>::view(IndexType from, IndexType to,
                              IndexType stride, IndexType firstViewIndex) const
{
    const IndexType length = (to-from)/stride+1;

    ASSERT(firstIndex()<=from);
    ASSERT(lastIndex()>=to);
    ASSERT(from<=to);
    ASSERT(stride>0);
    return ConstView(length,                // length
                     functor(),             // functor
                     from_+stride_*(from-firstIndex_),
                     stride*stride_,        // stride
                     firstViewIndex);       // firstIndex in view
}

template <typename F, typename I>
ImplicitArrayView<F, I>
ImplicitArrayView<F, I>::view(IndexType from, IndexType to,
                              IndexType stride, IndexType firstViewIndex)
{
    const IndexType length = (to-from)/stride+1;

    ASSERT(stride>0);
    return ImplicitArrayView(length,                // length
                             functor(),             // functor
                             from_+stride_*(from-firstIndex_),
                             stride*stride_,        // stride
                             firstViewIndex);       // firstIndex in view
}

} // namespace flens

#endif // FLENS_STORAGE_IMPLICIT_IMPLICITARRAYVIEW_TCC
