/*
 *   Copyright (c) 2012, Michael Lehn, Klaus Pototzky
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

#ifndef FLENS_MATRIXTYPES_GENERAL_IMPL_GBMATRIX_H
#define FLENS_MATRIXTYPES_GENERAL_IMPL_GBMATRIX_H 1

#include <cxxblas/typedefs.h>
#include <flens/auxiliary/auxiliary.h>
#include <flens/typedefs.h>
#include <flens/matrixtypes/general/generalmatrix.h>
#include <flens/typedefs.h>

namespace flens {

// forward declarations
template <typename A>
    class DenseVector;

template <typename FS>
    class TbMatrix;

template <typename FS>
    class GeMatrix;

template <typename FS>
class GbMatrix
    : public GeneralMatrix<GbMatrix<FS> >
{
    public:
        typedef FS                                  Engine;
        typedef typename Engine::ElementType        ElementType;
        typedef typename Engine::IndexType          IndexType;

        // std:: typedefs
        typedef typename Engine::size_type        size_type;
        typedef ElementType                       value_type;
        typedef typename Engine::pointer          pointer;
        typedef typename Engine::const_pointer    const_pointer;
        typedef typename Engine::reference        reference;
        typedef typename Engine::const_reference  const_reference;

        // view types from Engine
        typedef typename Engine::ConstView          EngineConstView;
        typedef typename Engine::View               EngineView;
        typedef typename Engine::NoView             EngineNoView;

        typedef typename Engine::ConstArrayView     ConstArrayView;
        typedef typename Engine::ArrayView          ArrayView;
        typedef typename Engine::Array              Array;

        typedef typename Engine::ConstFullStorageView   ConstFullStorageView;
        typedef typename Engine::FullStorageView        FullStorageView;
        typedef typename Engine::FullStorage            FullStorage;

        // view types
        typedef DenseVector<ConstArrayView>         ConstVectorView;
        typedef DenseVector<ArrayView>              VectorView;
        typedef DenseVector<Array>                  Vector;

        typedef GbMatrix<EngineConstView>           ConstView;
        typedef GbMatrix<EngineView>                View;
        typedef GbMatrix<EngineNoView>              NoView;

        typedef TbMatrix<EngineConstView>           ConstTriangularView;
        typedef TbMatrix<EngineView>                TriangularView;
        typedef TbMatrix<EngineNoView>              TriangularNoView;

        typedef GeMatrix<ConstFullStorageView>      ConstGeView;
        typedef GeMatrix<FullStorageView>           GeView;
        typedef GeMatrix<FullStorage>               GeNoView;

        GbMatrix();

        explicit
        GbMatrix(IndexType m, IndexType n,
                 IndexType numSubDiags, IndexType numSuperDiags,
                 IndexType firstIndex = Engine::defaultIndexBase);

        GbMatrix(const Engine &engine);

        GbMatrix(const GbMatrix &rhs);

        template <typename RHS>
            GbMatrix(const GbMatrix<RHS> &rhs);

        template <typename RHS>
            GbMatrix(GbMatrix<RHS> &rhs);

        template <typename RHS>
            GbMatrix(const Matrix<RHS> &rhs);

        // -- operators --------------------------------------------------------

        GbMatrix &
        operator=(const GbMatrix &rhs);

        template <typename RHS>
            GbMatrix &
            operator=(const Matrix<RHS> &rhs);

        template <typename RHS>
            GbMatrix &
            operator+=(const Matrix<RHS> &rhs);

        template <typename RHS>
            GbMatrix &
            operator-=(const Matrix<RHS> &rhs);

        GbMatrix<FS> &
        operator=(const ElementType &alpha);

        GbMatrix<FS> &
        operator+=(const ElementType &alpha);

        GbMatrix<FS> &
        operator-=(const ElementType &alpha);

        GbMatrix &
        operator*=(const ElementType &alpha);

        GbMatrix &
        operator/=(const ElementType &alpha);

        const_reference
        operator()(IndexType row, IndexType col) const;

        reference
        operator()(IndexType row, IndexType col);

        // -- views ------------------------------------------------------------

        // triangular views

        // upper
        const ConstTriangularView
        upper(Diag diag = NonUnit) const;

        TriangularView
        upper(Diag diag = NonUnit);

        const ConstView
        strictUpper() const;

        View
        strictUpper();

        // triangular view
        const ConstTriangularView
        lower(Diag diag = NonUnit) const;

        TriangularView
        lower(Diag diag = NonUnit);

        const ConstView
        strictLower() const;

        View
        strictLower();

        // Diagonals

        const ConstVectorView
        diag(IndexType diag) const;

        VectorView
        diag(IndexType diag);

        const ConstView
        diags(IndexType fromDiag, IndexType toDiag) const;

        View
        diags(IndexType fromDiag, IndexType toDiag);

        // row view (vector view)
        const ConstVectorView
        operator()(IndexType row, const Underscore<IndexType> &) const;

        VectorView
        operator()(IndexType row, const Underscore<IndexType> &);

        const ConstVectorView
        operator()(IndexType row, const Range<IndexType> &cols) const;

        VectorView
        operator()(IndexType row, const Range<IndexType> &cols);

        // column view (vector view)
        const ConstVectorView
        operator()(const Underscore<IndexType> &, IndexType col) const;

        VectorView
        operator()(const Underscore<IndexType> &, IndexType col);

        const ConstVectorView
        operator()(const Range<IndexType> &rows, IndexType col) const;

        VectorView
        operator()(const Range<IndexType> &rows, IndexType col);

        // view to the underlying Storage Matrix
        const ConstGeView
        viewStorageGeMatrix() const;

        GeView
        viewStorageGeMatrix();

        // -- methods ----------------------------------------------------------

        IndexType
        numCols() const;

        IndexType
        numRows() const;

        IndexType
        firstCol() const;

        IndexType
        lastCol() const;

        IndexType
        firstRow() const;

        IndexType
        lastRow() const;

        IndexType
        firstIndex() const;

        IndexType
        lastIndex() const;

        IndexType
        numSubDiags() const;

        IndexType
        numSuperDiags() const;

        IndexType
        leadingDimension() const;

        StorageOrder
        order() const;

        const_pointer
        data() const;

        pointer
        data();

        template <typename RHS>
            bool
            resize(const GbMatrix<RHS> &rhs,
                   const ElementType &value = ElementType());

        bool
        resize(IndexType numRow, IndexType numCols,
               IndexType numSubDiags, IndexType numSuperDiags,
               IndexType firstIndex = Engine::defaultIndexBase,
               const ElementType &value = ElementType());

        template <typename RHS>
            bool
            reserve(const GbMatrix<RHS> &rhs);

        bool
        reserve(IndexType numRow, IndexType numCols,
                IndexType numSubDiags, IndexType numSuperDiags,
                IndexType firstIndex = Engine::defaultIndexBase);

        bool
        fill(const ElementType &value = ElementType());

        bool
        fillRandom();

        // -- implementation ---------------------------------------------------

        const Engine &
        engine() const;

        Engine &
        engine();

    private:
        Engine       engine_;
};

//-- Traits --------------------------------------------------------------------
//
//  IsGbMatrix
//
struct GbMatrixChecker_
{

    struct Two {
        char x;
        char y;
    };

    static Two
    check(AnyConversion_);

    template <typename Any>
        static char
        check(GbMatrix<Any>);
};

template <typename T>
struct IsGbMatrix
{
    static T var;
    static const bool value = sizeof(GbMatrixChecker_::check(var))==1;
};

//
//  IsRealGbMatrix
//
template <typename T>
struct IsRealGbMatrix
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsGbMatrix<TT>::value
                           && IsNotComplex<typename TT::ElementType>::value;
};

//
//  IsComplexGbMatrix
//
template <typename T>
struct IsComplexGbMatrix
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsGbMatrix<TT>::value
                           && IsComplex<typename TT::ElementType>::value;
};

} // namespace flens

#endif // FLENS_MATRIXTYPES_GENERAL_IMPL_GBMATRIX_H
