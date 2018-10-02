#ifndef __CUMAT_DECOMPOSITION_BASE_H__
#define __CUMAT_DECOMPOSITION_BASE_H__

#include "Macros.h"
#include "SolverBase.h"

CUMAT_NAMESPACE_BEGIN

template<typename _DecompositionImpl>
class DecompositionBase : public SolverBase<_DecompositionImpl>
{
public:
    using Base = SolverBase<_DecompositionImpl>;
    using typename Base::Scalar;
    using Base::Rows;
    using Base::Columns;
    using Base::Batches;
    using Base::impl;

    typedef SolveOp<_DecompositionImpl, NullaryOp<Scalar, Rows, Columns, Batches, ColumnMajor, functor::IdentityFunctor<Scalar> > > InverseResultType;
    /**
     * \brief Computes the inverse of the input matrix.
     * \return The inverse matrix
     */
    InverseResultType inverse() const
    {
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(Rows > 0 && Columns > 0, Rows == Columns),
            "Static count of rows and columns must be equal (square matrix)");
        CUMAT_ASSERT(impl().rows() == impl().cols());

        return impl().solve(NullaryOp<Scalar, Rows, Columns, Batches, ColumnMajor, functor::IdentityFunctor<Scalar> >(
            impl().rows(), impl().cols(), impl().batches(), functor::IdentityFunctor<Scalar>()));
    }

};

CUMAT_NAMESPACE_END

#endif
