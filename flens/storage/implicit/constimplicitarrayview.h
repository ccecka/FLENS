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

#ifndef FLENS_STORAGE_IMPLICIT_CONSTIMPLICITARRAYVIEW_H
#define FLENS_STORAGE_IMPLICIT_CONSTIMPLICITARRAYVIEW_H 1

#include <cxxstd/utility.h>
#include <flens/storage/indexoptions.h>

namespace flens {

template <typename F, typename I>
    class ImplicitArrayView;

template <typename F, typename I>
    class ImplicitArray;

template <typename F,
          typename I = IndexOptions<> >
class ConstImplicitArrayView
{
    public:
        typedef F                       FunctorType;
        typedef typename I::IndexType   IndexType;
        typedef decltype(std::declval<FunctorType>()(std::declval<IndexType>()))
                                        ElementType;


        // std:: typedefs -- TODO: more meta
        typedef IndexType                                 size_type;
        typedef ElementType                               value_type;
        typedef ElementType*                              pointer;
        typedef const ElementType*                        const_pointer;
        typedef ElementType                               reference;
        typedef ElementType                               const_reference;

        typedef ConstImplicitArrayView<F, I> ConstView;
        typedef ImplicitArrayView<F, I>      View;
        typedef ImplicitArray<F, I>          NoView;

        static const IndexType          defaultIndexBase = I::defaultIndexBase;

        ConstImplicitArrayView(IndexType length,
                               FunctorType& functor,
                               IndexType from = defaultIndexBase,
                               IndexType stride = IndexType(1),
                               IndexType firstIndex = defaultIndexBase);

        ConstImplicitArrayView(const ConstImplicitArrayView &rhs);

        //-- operators ---------------------------------------------------------

        const_reference
        operator()(IndexType index) const;

        //-- methods -----------------------------------------------------------

        IndexType
        firstIndex() const;

        IndexType
        lastIndex() const;

        IndexType
        length() const;

        IndexType
        stride() const;

        const F&
        functor() const;

        std::allocator<ElementType>
        allocator() const;

        void
        changeIndexBase(IndexType firstIndex);

        const ConstView
        view(IndexType from, IndexType to,
             IndexType stride = IndexType(1),
             IndexType firstViewIndex =  defaultIndexBase) const;

    private:
        FunctorType& functor_;
        IndexType    length_, from_, stride_, firstIndex_;
};

} // namespace flens

#endif // FLENS_STORAGE_IMPLICIT_IMPLICITARRAYVIEW_H
