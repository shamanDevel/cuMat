#ifndef __CUMAT_SOLVER_BASE_H__
#define __CUMAT_SOLVER_BASE_H__

#include "Macros.h"
#include "ForwardDeclarations.h"

CUMAT_NAMESPACE_BEGIN

template<typename _DecompositionImpl>
class SolverBase
{
protected:
	CUMAT_STRONG_INLINE _DecompositionImpl& impl() { return *static_cast<_DecompositionImpl*>(this); }
	CUMAT_STRONG_INLINE const _DecompositionImpl& impl() const { return *static_cast<const _DecompositionImpl*>(this); }

public:
    using Scalar = typename internal::traits<_DecompositionImpl>::Scalar;
    using MatrixType = typename internal::traits<_DecompositionImpl>::MatrixType;
    enum
    {
        Flags = internal::traits<MatrixType>::Flags,
        Rows = internal::traits<MatrixType>::RowsAtCompileTime,
        Columns = internal::traits<MatrixType>::ColsAtCompileTime,
        Batches = internal::traits<MatrixType>::BatchesAtCompileTime,
    };

    /**
     * \brief Solves the system of linear equations
     * \tparam _RHS the type of the right hand side
     * \param rhs the right hand side matrix
     * \return The operation that computes the solution of the linear system
     */
    template<typename _RHS>
    SolveOp<_DecompositionImpl, _RHS> solve(const MatrixBase<_RHS>& rhs) const
    {
        return SolveOp<_DecompositionImpl, _RHS>(impl(), rhs.derived());
    }
};

namespace internal
{
    struct SolveSrcTag {};
    template<typename _Solver, typename _RHS>
    struct traits<SolveOp<_Solver, _RHS> >
    {
        //using Scalar = typename internal::traits<_Child>::Scalar;
        using Scalar = typename internal::traits<_RHS>::Scalar;
        enum
        {
            Flags = ColumnMajor,
            RowsAtCompileTime = internal::traits<_RHS>::RowsAtCompileTime,
            ColsAtCompileTime = internal::traits<_RHS>::ColsAtCompileTime,
            BatchesAtCompileTime = internal::traits<_RHS>::BatchesAtCompileTime,
            AccessFlags = 0
        };
        typedef SolveSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
    };
}

/**
 * \brief General solver operation.
 * Delegates to _solve_impl of the solver implementation.
 * \tparam _Solver 
 * \tparam _RHS 
 */
template<typename _Solver, typename _RHS>
class SolveOp : public MatrixBase<SolveOp<_Solver, _RHS>>
{
public:
    typedef MatrixBase<SolveOp<_Solver, _RHS>> Base;
    typedef SolveOp<_Solver, _RHS> Type;
    using Scalar = typename internal::traits<_RHS>::Scalar;
    enum
    {
        Flags = ColumnMajor,
        Rows = internal::traits<_RHS>::RowsAtCompileTime,
        Columns = internal::traits<_RHS>::ColsAtCompileTime,
        Batches = internal::traits<_RHS>::BatchesAtCompileTime
    };

private:
    const _Solver& decomposition_;
    const _RHS rhs_;

public:
    SolveOp(const _Solver& decomposition, const MatrixBase<_RHS>& rhs)
        : decomposition_(decomposition)
        , rhs_(rhs.derived())
    {
        CUMAT_STATIC_ASSERT((std::is_same<
				typename internal::NumTraits<typename _Solver::Scalar>::ElementalType, 
				typename internal::NumTraits<typename internal::traits<_RHS>::Scalar>::ElementalType>::value),
            "Datatype of left- and right hand side must match");
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Solver::Batches > 1 && _RHS::Batches > 0, _Solver::Batches == _RHS::Batches),
            "Static count of batches must match"); //note: _Solver::Batches>1 to allow broadcasting
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Solver::Rows > 0 && _Solver::Columns > 0, _Solver::Rows == _Solver::Columns),
            "Static count of rows and columns must be equal (square matrix)");
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Solver::Rows > 0 && _RHS::Rows > 0, _Solver::Rows == _RHS::Rows),
            "Left and right hand side are not compatible");

        CUMAT_ASSERT(CUMAT_IMPLIES(_Solver::Batches!=1, decomposition.batches() == rhs.batches()) && "batch size of the matrix and the right hand side does not match");
        CUMAT_ASSERT(decomposition.rows() == decomposition.cols() && "matrix must be square"); //TODO: relax for Least-Squares problems
        CUMAT_ASSERT(decomposition.cols() == rhs.rows() && "matrix size does not match right-hand-side");
    }

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return rhs_.rows(); }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return rhs_.cols(); }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return rhs_.batches(); }

    const _Solver& getDecomposition() const { return decomposition_; }
    const _RHS& getRhs() const { return rhs_; }
};

namespace internal
{
    //Assignment for decompositions that call SolveOp::evalTo
    template<typename _Dst, typename _Src, AssignmentMode _Mode>
    struct Assignment<_Dst, _Src, _Mode, DenseDstTag, SolveSrcTag>
    {
        static void assign(_Dst& dst, const _Src& src)
        {
            static_assert(_Mode == AssignmentMode::ASSIGN, "Decompositions only support AssignmentMode::ASSIGN (operator=)");
            src.derived().getDecomposition()._solve_impl(src.derived().getRhs(), dst.derived());
        }
    };
}

CUMAT_NAMESPACE_END

#endif