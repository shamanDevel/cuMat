#ifndef __CUMAT_ITERATIVE_SOLVER_BASE_H__
#define __CUMAT_ITERATIVE_SOLVER_BASE_H__

#include "Macros.h"
#include "SolverBase.h"
#include "NumTraits.h"

CUMAT_NAMESPACE_BEGIN

template<typename _SolverImpl>
class IterativeSolverBase : public SolverBase<_SolverImpl>
{
public:
    using Base = SolverBase<_SolverImpl>;
    using typename Base::Scalar;
    using RealScalar = typename internal::NumTraits<Scalar>::RealType;
    using Base::Rows;
    using Base::Columns;
    using Base::Batches;
    using Base::impl;
    using typename Base::MatrixType;
    using Preconditioner = typename internal::traits<_SolverImpl>::Preconditioner;

protected:
    MatrixType matrix_;
    Preconditioner preconditioner_;
    RealScalar tolerance_;
    Index maxIterations_;
    bool initialized_;
    mutable Index iterations_;
    mutable RealScalar error_;

public:

    IterativeSolverBase() {init();}
    IterativeSolverBase(const MatrixBase<MatrixType>& matrix)
        : matrix_(matrix.derived()), preconditioner_(matrix.derived())
    {
        init();
        initialized_= true;
    }
    IterativeSolverBase(const MatrixBase<MatrixType>& matrix, const Preconditioner& preconditioner)
        : matrix_(matrix.derived()), preconditioner_(preconditioner)
    {
        init();
        initialized_= true;
    }

    CUMAT_STRONG_INLINE Index rows() const { return matrix_.rows(); }
    CUMAT_STRONG_INLINE Index cols() const { return matrix_.cols(); }
    CUMAT_STRONG_INLINE Index batches() const { return matrix_.batches(); }
    const MatrixType& getMatrix() const { return matrix_; }

    /** \returns the tolerance threshold used by the stopping criteria.
    * \sa setTolerance()
    */
    RealScalar tolerance() const { return tolerance_; }

    /** Sets the tolerance threshold used by the stopping criteria.
      *
      * This value is used as an upper bound to the relative residual error: |Ax-b|/|b|.
      * The default value is the machine precision given by NumTraits<Scalar>::epsilon()
      */
    _SolverImpl& setTolerance(const RealScalar& tolerance)
    {
        tolerance_ = tolerance;
        return impl();
    }

    /** \returns the matrix. */
    const MatrixType& matrix() const {return matrix_;}

    /** \returns a read-write reference to the preconditioner for custom configuration. */
    Preconditioner& preconditioner() { return preconditioner_; }

    /** \returns a read-only reference to the preconditioner. */
    const Preconditioner& preconditioner() const { return preconditioner_; }

    /** \returns the max number of iterations.
      * It is either the value setted by setMaxIterations or, by default,
      * twice the number of columns of the matrix.
      */
    Index maxIterations() const
    {
        return (maxIterations_ < 0) ? 2 * matrix_.cols() : maxIterations_;
    }

    /** Sets the max number of iterations.
      * Default is twice the number of columns of the matrix.
      */
    _SolverImpl& setMaxIterations(Index maxIters)
    {
        maxIterations_ = maxIters;
        return impl();
    }

    /** \returns the number of iterations performed during the last solve */
    Index iterations() const
    {
        CUMAT_ASSERT(initialized_ && "Iterative Solver is not initialized.");
        return iterations_;
    }

    /** \returns the tolerance error reached during the last solve.
      * It is a close approximation of the true relative residual error |Ax-b|/|b|.
      */
    RealScalar error() const
    {
        CUMAT_ASSERT(initialized_ && "Iterative Solver is not initialized.");
        return error_;
    }

    /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A
      * and \a x0 as an initial solution.
      *
      * \sa solve(), compute()
      */
    template <typename Rhs, typename Guess>
    CUMAT_STRONG_INLINE SolveWithGuessOp<_SolverImpl, Rhs, Guess>
    solveWithGuess(const MatrixBase<Rhs>& b, const Guess& x0) const
    {
        CUMAT_ASSERT(initialized_ && "Solver is not initialized.");
        return SolveWithGuessOp<_SolverImpl, Rhs, Guess>(impl(), b.derived(), x0);
    }

protected:
    void init()
    {
        initialized_ = false;
        maxIterations_ = -1;
        tolerance_ = internal::NumTraits<Scalar>::epsilon();
        iterations_ = 0;
        error_ = 0;
    }
};

namespace internal
{
    struct SolveWithGuessSrcTag {};
    template<typename _Solver, typename _RHS, typename _Guess>
    struct traits<SolveWithGuessOp<_Solver, _RHS, _Guess> >
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
        typedef SolveWithGuessSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
    };
}

/**
 * \brief General solver operation.
 * Delegates to _solve_impl of the solver implementation.
 * \tparam _Solver 
 * \tparam _RHS 
 */
template<typename _Solver, typename _RHS, typename _Guess>
class SolveWithGuessOp : public MatrixBase<SolveWithGuessOp<_Solver, _RHS, _Guess>>
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
    const _Guess guess_;

public:
    SolveWithGuessOp(const _Solver& decomposition, const MatrixBase<_RHS>& rhs, const MatrixBase<_Guess>& guess)
        : decomposition_(decomposition)
        , rhs_(rhs.derived())
        , guess_(guess.derived())
    {
		CUMAT_STATIC_ASSERT((std::is_same<
			typename internal::NumTraits<typename _Solver::Scalar>::ElementalType,
			typename internal::NumTraits<typename internal::traits<_RHS>::Scalar>::ElementalType>::value),
			"Datatype of left- and right hand side must match");
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Solver::Batches > 1 && _RHS::Batches > 0, _Solver::Batches == _RHS::Batches),
            "Static count of batches must match"); //note: _Solver::Batches>1 to allow broadcasting
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Solver::Rows > 0 && _Solver::Columns > 0, _Solver::Rows == _RHS::Columns),
            "Static count of rows and columns must be equal (square matrix)");
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Solver::Rows > 0 && _RHS::Rows > 0, _Solver::Rows == _RHS::Rows),
            "Left and right hand side are not compatible");

        CUMAT_ASSERT(CUMAT_IMPLIES(_Solver::Batches!=1, decomposition.batches() == rhs.batches()) && "Number of batches must be compatible");
        CUMAT_ASSERT(decomposition.rows() == decomposition.cols() && "Matrix must be square"); //TODO: relax that for Least Squares problems?
        CUMAT_ASSERT(decomposition.rows() == rhs.rows() && "Matrix must be compatible with the right hand side");
    }

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return rhs_.rows(); }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return rhs_.cols(); }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return rhs_.batches(); }

    const _Solver& getDecomposition() const { return decomposition_; }
    const _RHS& getRhs() const { return rhs_; }
    const _Guess& getGuess() const {return guess_; }
};

namespace internal
{
    //Assignment for decompositions that call SolveOp::evalTo
    template<typename _Dst, typename _Src, AssignmentMode _Mode>
    struct Assignment<_Dst, _Src, _Mode, DenseDstTag, SolveWithGuessSrcTag>
    {
        static void assign(_Dst& dst, const _Src& src)
        {
            static_assert(_Mode == AssignmentMode::ASSIGN, "Decompositions only support AssignmentMode::ASSIGN (operator=)");
            src.derived().getDecomposition()._solve_with_guess_impl(
                src.derived().getRhs().derived(), dst.derived().derived(), src.derived().getGuess().derived());
        }
    };
}

CUMAT_NAMESPACE_END

#endif