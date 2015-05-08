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

#ifndef FLENS_STORAGE_ARRAY_ARRAY_TCC
#define FLENS_STORAGE_ARRAY_ARRAY_TCC 1

#include <cxxstd/cassert.h>
#include <cxxstd/algorithm.h>
#include <cxxblas/level1/copy.h>
#include <flens/auxiliary/auxiliary.h>
#include <flens/storage/array/array.h>
#include <flens/storage/array/arrayview.h>
#include <flens/storage/array/constarrayview.h>


namespace flens {

template <typename T, typename I, typename A>
Array<T, I, A>::Array()
    : data_(),
      length_(0),
      firstIndex_(1)
{
}

template <typename T, typename I, typename A>
Array<T, I, A>::Array(IndexType length, IndexType firstIndex,
                      const ElementType &value, const Allocator &allocator)
    : data_(),
      length_(length),
      firstIndex_(firstIndex),
      allocator_(allocator)
{
    ASSERT(length_>=0);

    allocate_(value);
}

template <typename T, typename I, typename A>
Array<T, I, A>::Array(const Array &rhs)
    : data_(),
      length_(rhs.length()),
      firstIndex_(rhs.firstIndex()),
      allocator_(rhs.allocator())
{
    ASSERT(length_>=0);

    if (length()>0) {
        raw_allocate_();
        cxxblas::copy(length(), rhs.data(), rhs.stride(), data(), stride());
    }
}

template <typename T, typename I, typename A>
template <typename RHS>
Array<T, I, A>::Array(const RHS &rhs)
    : data_(),
      length_(rhs.length()),
      firstIndex_(rhs.firstIndex())
      // XXX: HACK WAR?
      //, allocator_(rhs.allocator())
{
    if (length()>0) {
        raw_allocate_();
        cxxblas::copy(length(), rhs.data(), rhs.stride(), data(), stride());
    }
}

template <typename T, typename I, typename A>
Array<T, I, A>::~Array()
{
    release_();
}

//-- operators -----------------------------------------------------------------

template <typename T, typename I, typename A>
typename Array<T, I, A>::const_reference
Array<T, I, A>::operator()(IndexType index) const
{
    ASSERT(index>=firstIndex());
    ASSERT(index<=lastIndex());
    return data_[index-firstIndex_];
}

template <typename T, typename I, typename A>
typename Array<T, I, A>::reference
Array<T, I, A>::operator()(IndexType index)
{
    ASSERT(index>=firstIndex());
    ASSERT(index<=lastIndex());
    return data_[index-firstIndex_];
}

template <typename T, typename I, typename A>
typename Array<T, I, A>::IndexType
Array<T, I, A>::firstIndex() const
{
    return firstIndex_;
}

template <typename T, typename I, typename A>
typename Array<T, I, A>::IndexType
Array<T, I, A>::lastIndex() const
{
    return firstIndex_+length_-IndexType(1);
}

template <typename T, typename I, typename A>
typename Array<T, I, A>::IndexType
Array<T, I, A>::length() const
{
    return length_;
}

template <typename T, typename I, typename A>
typename Array<T, I, A>::IndexType
Array<T, I, A>::stride() const
{
    return IndexType(1);
}

template <typename T, typename I, typename A>
typename Array<T, I, A>::const_pointer
Array<T, I, A>::data() const
{
    return data_;
}

template <typename T, typename I, typename A>
typename Array<T, I, A>::pointer
Array<T, I, A>::data()
{
    return data_;
}

template <typename T, typename I, typename A>
const typename Array<T, I, A>::Allocator &
Array<T, I, A>::allocator() const
{
    return allocator_;
}

template <typename T, typename I, typename A>
bool
Array<T, I, A>::resize(IndexType length, IndexType firstIndex,
                       const ElementType &value)
{
    if (length!=length_) {
        release_();
        length_ = length;
        firstIndex_ = firstIndex;
        allocate_(value);
        return true;
    }
    changeIndexBase(firstIndex);
    return false;
}

template <typename T, typename I, typename A>
template <typename ARRAY>
bool
Array<T, I, A>::resize(const ARRAY &rhs, const ElementType &value)
{
    return resize(rhs.length(), rhs.firstIndex(), value);
}

template <typename T, typename I, typename A>
bool
Array<T, I, A>::reserve(IndexType length, IndexType firstIndex)
{
    if (length!=length_) {
        release_();
        length_ = length;
        firstIndex_ = firstIndex;
        raw_allocate_();
        return true;
    }
    changeIndexBase(firstIndex);
    return false;
}

template <typename T, typename I, typename A>
template <typename ARRAY>
bool
Array<T, I, A>::reserve(const ARRAY &rhs)
{
    return reserve(rhs.length(), rhs.firstIndex());
}

template <typename T, typename I, typename A>
bool
Array<T, I, A>::fill(const ElementType &value)
{
    flens::alg::fill_n(data(), length(), value);
    return true;
}

template <typename T, typename I, typename A>
void
Array<T, I, A>::changeIndexBase(IndexType firstIndex)
{
    firstIndex_ = firstIndex;
}

template <typename T, typename I, typename A>
const typename Array<T, I, A>::ConstView
Array<T, I, A>::view(IndexType from, IndexType to,
                     IndexType stride, IndexType firstViewIndex) const
{
    const IndexType length = (to-from)/stride+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    const_pointer data = (length!=0) ? &operator()(from) : pointer();

    if (length!=0) {
        ASSERT(firstIndex()<=from);
        ASSERT(lastIndex()>=to);
        ASSERT(from<=to);
    }
    ASSERT(stride>=1);
#   else
    const_pointer data = &operator()(from);
#   endif

    return ConstView(length, data, stride, firstViewIndex, allocator());
}

template <typename T, typename I, typename A>
typename Array<T, I, A>::View
Array<T, I, A>::view(IndexType from, IndexType to,
                     IndexType stride, IndexType firstViewIndex)
{
    const IndexType     length = (to-from)/stride+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    pointer data = (length!=0) ? &operator()(from) : pointer();

    if (length!=0) {
        ASSERT(firstIndex()<=from);
        ASSERT(lastIndex()>=to);
        ASSERT(from<=to);
    }
    ASSERT(stride>=1);
#   else
    pointer data = &operator()(from);
#   endif

    return View(length, data, stride, firstViewIndex, allocator());
}

//-- private methods -----------------------------------------------------------

template <typename T, typename I, typename A>
void
Array<T, I, A>::raw_allocate_()
{
    ASSERT(data_==pointer());
    ASSERT(length()>=0);

    if (length()>0) {
        data_ = allocator_.allocate(length_);
        ASSERT(data_!=pointer());
    }
}

template <typename T, typename I, typename A>
void
Array<T, I, A>::allocate_(const ElementType &value)
{
    raw_allocate_();
    flens::alg::uninitialized_fill_n(data(), length(), value);
}

template <typename T, typename I, typename A>
void
Array<T, I, A>::release_()
{
    if (data_ != pointer()) {
        ASSERT(length()>0);
        // XXX: Assume T is trivially destructible
        // TODO: std::destroy(first, last, alloc).  See gcc's std::_Destroy
        allocator_.deallocate(data(), length());
        data_ = pointer();
    }
    ASSERT(data_==pointer());
}

//-- Array specific functions --------------------------------------------------

//
//  fillRandom
//

template <typename T, typename I, typename A>
bool
fillRandom(Array<T, I, A> &x)
{
    typedef typename Array<T,I,A>::IndexType    IndexType;

    for (IndexType i=x.firstIndex(); i<=x.lastIndex(); ++i) {
        x[i] = randomValue<T>();
    }
    return true;
}

} // namespace flens

#endif // FLENS_STORAGE_ARRAY_ARRAY_TCC
