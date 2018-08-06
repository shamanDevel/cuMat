#ifndef __CUMAT_CONJUGATED_GRADIENT__
#define __CUMAT_CONJUGATED_GRADIENT__

#include "Macros.h"

#include <cmath>

#include "Matrix.h"
#include "UnaryOps.h"
#include "BinaryOps.h"
#include "ReductionOps.h"
#include "IterativeSolverBase.h"

CUMAT_NAMESPACE_BEGIN

namespace internal
{
    template<typename _MatrixType, typename _Preconditioner>
    struct traits<ConjugateGradient<_MatrixType, _Preconditioner> >
    {
        using Scalar = typename internal::traits<_MatrixType>::Scalar;
        using MatrixType = _MatrixType;
        using Preconditioner = _Preconditioner;
    };
}

/**
 * \brief Conjugate gradient solver for arbitrary (dense, sparse, matrix-free) matrices.
 * 
 * This class allows to solve for A*x=b linear problems using \code x = ConjugateGradient(A).solve(b) \endcode.
 * Only non-batched matrices and only non-batched, column vectors for the right hand side are supported.
 * 
 * The right hand side b can be any matrix expression that is a column vector.
 * The result type \c x has to be a dense column vector (an instance of the Matrix class).
 * The matrix type can be any object, as long as it supports an \c operator* that takes a dense vector as right hand side
 * and returns an expression of a column vector.
 * 
 * The preconditioner must support a method \code .solve(r) \endcode that takes a dense column vector as input and
 * returns an expression of a column vector that approximates the solution of A.x=r.
 * Examples are the DiagonalPreconditioner and IdentityPreconditioner.
 * 
 * \tparam _MatrixType any matrix expression with an operator* that takes a dense column vector as right hand side
 * \tparam _Preconditioner the preconditioner object, default is DiagonalPreconditioner
 */
template<typename _MatrixType, typename _Preconditioner = DiagonalPreconditioner<_MatrixType>>
class ConjugateGradient : public IterativeSolverBase<ConjugateGradient<_MatrixType, _Preconditioner>>
{
    CUMAT_STATIC_ASSERT(_MatrixType::Batches == 1, "Conjugate Gradient can only work on non-batched matrices");

public:
    using Type = ConjugateGradient<_MatrixType, _Preconditioner>;
    using Base = IterativeSolverBase<Type>;
    using Base::MatrixType;
    using Base::Preconditioner;
    using Base::Scalar;
    using Base::RealScalar;
    using Base::maxIterations;

private:
    using Base::matrix_;
    using Base::preconditioner_;
    using Base::tolerance_;
    using Base::iterations_;
    using Base::error_;

public:

    ConjugateGradient() = default;

    /**
     * \brief Initializes the Conjugate Gradient with the specified matrix.
     * The preconditioner is created with \code Preconditioner(matrix) \endcode.
     * \param matrix the matrix that is used in the CG.
     */
    ConjugateGradient(const MatrixBase<MatrixType>& matrix)
        : Base(matrix)
    {
        CUMAT_ASSERT(matrix.rows() == matrix.cols() && "Matrix must be square");
    }

    /**
     * \brief Initializes the Conjugate Gradient with the specified matrix
     * and specified preconditioner.
     * \param matrix the matrix that is used in the CG.
     * \param preconditioner the preconditioner that is used
     */
    ConjugateGradient(const MatrixBase<MatrixType>& matrix, const Preconditioner& preconditioner)
        : Base(matrix, preconditioner)
    {
        CUMAT_ASSERT(matrix.rows() == matrix.cols() && "Matrix must be square");
    }

    template<typename _RHS, typename _Target>
    void _solve_impl(const MatrixBase<_RHS>& rhs, MatrixBase<_Target>& target) const
    {
        typedef Matrix<typename _Target::Scalar, Dynamic, 1, 1, _Target::Flags> GuessType;
        _solve_with_guess_impl(rhs.derived(), target.derived(), GuessType::Zero(target.rows()));
    }

    template<typename _RHS, typename _Target, typename _Guess>
    void _solve_with_guess_impl(_RHS& rhs, _Target& target, _Guess& guess) const
    {
        CUMAT_STATIC_ASSERT(_Target::Batches == 1, "The target matrix must be non-batched");
        CUMAT_STATIC_ASSERT(_Target::Columns == 1, "The target must be a compile-time column vector");
        CUMAT_STATIC_ASSERT(_Guess::Batches == 1, "The initial guess must be non-batched");
        CUMAT_STATIC_ASSERT(_Guess::Columns == 1, "The initial guess must be a compile-time column vector");
        CUMAT_ASSERT(matrix_.cols() == rhs.rows());

        //initialize result and counter
        iterations_ = 0;
        error_ = 0;
        target.inplace() = guess;

        //----------------
        // CG ALGORITHM, ported from Eigen
        //----------------
        typedef Matrix<RealScalar, 1, 1, 1, 0> RealScalarDevice;
        using std::sqrt;
        using std::abs;
        typedef Matrix<Scalar, Dynamic, 1, 1, _Target::Flags> VectorType;
        Index n = matrix_.cols();

        VectorType residual = rhs - matrix_ * target; //initial residual
        RealScalar rhsNorm2 = static_cast<RealScalar>(rhs.squaredNorm()); //SLOW: device->host memcopy (explicit conversion operator)
        if (rhsNorm2 == 0)
        {
            //early out, right-hand side is zero
            target.setZero();
            return;
        }
        RealScalar threshold = tolerance_ * tolerance_ * rhsNorm2;
        RealScalar residualNorm2 = static_cast<RealScalar>(residual.squaredNorm()); //SLOW: device->host memcopy
        if (residualNorm2 < threshold)
        {
            //early out, already close enough to the solution
            error_ = sqrt(residualNorm2 / rhsNorm2);
            return;
        }

        VectorType p(n);
        p = preconditioner_.solve(residual); // initial search direction

        VectorType z(n), tmp(n);
        RealScalarDevice absNew = residual.dot(p).real(); // the square of the absolute value of r scaled by invM
        Index i = 0;
        const Index maxIter = maxIterations();
        while (i < maxIter)
        {
            tmp.inplace() = matrix_ * p; // the bottleneck of the algorithm

            auto alpha = absNew.cwiseDiv(p.dot(tmp)); // the amount we travel on dir; expression, cwiseDiv not evaluated (dot is)
            target += alpha.cwiseMul(p); // update solution
            residual -= alpha.cwiseMul(tmp); // update residual

            residualNorm2 = static_cast<RealScalar>(residual.squaredNorm()); //SLOW: device->host memcopy
            //TODO: move the squared norm above "alpha.cwiseMul(tmp)", start the memcopy asynchronously and retrieve the result here
            if (residualNorm2 < threshold)
                break;

            z = preconditioner_.solve(residual); // approximately solve for "A z = residual"

            RealScalarDevice absOld = absNew;
            absNew = residual.dot(z).real(); // update the absolute value of r
            auto beta = absNew.cwiseDiv(absOld); //expression, not evaluated
            // calculate the Gram-Schmidt value used to create the new search direction
            p = z + beta.cwiseMul(p); // update search direction
            i++;
        }
        error_ = sqrt(residualNorm2 / rhsNorm2);
        iterations_ = i;
    }
};

/**
 * \brief Preconditioner for iterative solvers using the main diagonal of the matrix.
 * This class allows to approximately solve for A.x = b problems assuming A is a diagonal matrix.
 * In other words, this preconditioner neglects all off diagonal entries and solves for:
    \code
    A.diagonal().asDiagonal() . x = b
    \endcode
 * \tparam _MatrixType 
 */
template<typename _MatrixType>
class DiagonalPreconditioner
{
private:
    typedef typename internal::ExtractDiagonalFunctor<typename _MatrixType::Scalar>::VectorType Scalar;
    typedef Matrix<Scalar, Dynamic, 1, 1, ColumnMajor> Vector;
    Vector entries_;

public:
    DiagonalPreconditioner(const MatrixBase<_MatrixType>& matrix)
        : entries_(matrix.diagonal().cwiseInverseCheck())
    {}

    template<typename _Rhs>
    using SolveReturnType = BinaryOp<Vector, _Rhs, functor::BinaryMathFunctor_cwiseMul<Scalar>, false>;
    /**
     * \brief Solves for an approximation of A.x=b
     * \tparam _Rhs the type of right hand side
     * \param b the right hand side of the equation
     * \return the approximate solution of x
     */
    template<typename _Rhs>
    SolveReturnType<_Rhs>
    solve(const MatrixBase<_Rhs>& b) const
    {
        return entries_.cwiseMul(b.derived());
    }
};

/**
 * \brief A trivial preconditioner which approximates any matrix as the identity matrix.
 * \tparam _MatrixType 
 */
template<typename _MatrixType>
class IdentityPreconditioner
{

public:
    IdentityPreconditioner(const MatrixBase<_MatrixType>& matrix)
    {}

    template<typename _Rhs>
    _Rhs solve(const MatrixBase<_Rhs>& b) const
    {
        return b;
    }
};

CUMAT_NAMESPACE_END

#endif