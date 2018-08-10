#ifndef __CUMAT_CHOLESKY_DECOMPOSITION_H__
#define __CUMAT_CHOLESKY_DECOMPOSITION_H__

#include "Macros.h"
#include "Matrix.h"
#include "CusolverApi.h"
#include "DecompositionBase.h"
#include <algorithm>
#include <vector>
#include <type_traits>

CUMAT_NAMESPACE_BEGIN

namespace internal
{
    template<typename _MatrixType>
    struct traits<CholeskyDecomposition<_MatrixType>>
    {
        using Scalar = typename internal::traits<_MatrixType>::Scalar;
        using MatrixType = _MatrixType;
    };
}

template<typename _MatrixType>
class CholeskyDecomposition : public DecompositionBase<CholeskyDecomposition<_MatrixType>>
{
public:
    using Scalar = typename internal::traits<_MatrixType>::Scalar;
    typedef CholeskyDecomposition<_MatrixType> Type;
    enum
    {
        Flags = internal::traits<_MatrixType>::Flags,
        Rows = internal::traits<_MatrixType>::RowsAtCompileTime,
        Columns = internal::traits<_MatrixType>::ColsAtCompileTime,
        Batches = internal::traits<_MatrixType>::BatchesAtCompileTime,
        InputIsMatrix = std::is_same< _MatrixType, Matrix<Scalar, Rows, Columns, Batches, Flags> >::value
    };
    typedef Matrix<Scalar, Dynamic, Dynamic, Batches, Flags> EvaluatedMatrix;
    typedef Matrix<Scalar, 1, 1, Batches, ColumnMajor> DeterminantMatrix;
private:
    EvaluatedMatrix decompositedMatrix_;
    std::vector<int> singular_;

public:
    /**
     * \brief Performs the Cholesky decomposition of the specified matrix and stores the result for future use.
     * \param matrix the Hermetian, positive definite input matrix
     * \param inplace true to enforce inplace operation. The matrix contents will be destroyed
     */
    explicit CholeskyDecomposition(const MatrixBase<_MatrixType>& matrix, bool inplace=false)
        : decompositedMatrix_(matrix.derived()) //this evaluates every matrix expression into a matrix
        , singular_(matrix.batches())
    {
        //Check if the input is symmetric
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(Rows>0 && Columns>0, Rows==Columns), "Input Matrix must be symmetric");
        CUMAT_ASSERT_ARGUMENT(matrix.cols() == matrix.rows());
    
        //optionally, copy input
        //(copy is never needed if the input is not a matrix and is evaluated into the matrix during the initializer list)
        if (!inplace && InputIsMatrix)
            decompositedMatrix_ = decompositedMatrix_.deepClone();
        
        //perform  factorization
        const int n = decompositedMatrix_.rows();
        const int batches = decompositedMatrix_.batches();
        const int lda = n;
        Matrix<int, 1, 1, Batches, RowMajor> devInfo(1, 1, batches);
        for (Index batch = 0; batch < batches; ++batch) {
            internal::CusolverApi::current().cusolverPotrf(
                CUBLAS_FILL_MODE_UPPER, n,
                internal::CusolverApi::cast(decompositedMatrix_.data() + batch*n*n), lda,
                devInfo.data() + batch);
        }

        //check if the operation was successfull
        devInfo.copyToHost(&singular_[0]);
        for (Index i=0; i<batches; ++i)
        {
            if (singular_[i]<0)
            {
                throw cuda_error(internal::ErrorHelpers::format("Potrf failed at batch %d, parameter %d was invalid", i, -singular_[i]));
            } else if (singular_[i]>0)
            {
                throw cuda_error(internal::ErrorHelpers::format("Potrf failed at batch %d, the leaading minor %d was was not positive definite", i, singular_[i]));
            }
        }
    }

    CUMAT_STRONG_INLINE Index rows() const { return decompositedMatrix_.rows(); }
    CUMAT_STRONG_INLINE Index cols() const { return decompositedMatrix_.cols(); }
    CUMAT_STRONG_INLINE Index batches() const { return decompositedMatrix_.batches(); }

    /**
     * \brief Returns the Cholesky factorization. Only the upper triangular part contains meaningful data, the Cholesky factor U.
     * The lower triangular part contains skrap data.
     */
    const EvaluatedMatrix& getMatrixCholesky() const
    {
        return decompositedMatrix_;
    }

    /**
     * \brief Computes the determinant of this matrix
     * \return The determinant
     */
    DeterminantMatrix determinant() const
    {
        if (rows()==0)
        {
            return DeterminantMatrix::Constant(1, 1, batches(), Scalar(1));
        }
		return decompositedMatrix_.diagonal().template prod<ReductionAxis::Row | ReductionAxis::Column>().cwiseAbs(); //multiply diagonal elements
            //the product of the diagonal elements is the squareroot of the determinant
    }

    /**
    * \brief Computes the log-determinant of this matrix.
    * \return The log-determinant
    */
    DeterminantMatrix logDeterminant() const
    {
        if (rows() == 0 || cols() != rows())
        {
            return DeterminantMatrix::Constant(1, 1, batches(), Scalar(0));
        }
        return (decompositedMatrix_.diagonal().cwiseLog().template sum<ReductionAxis::Row | ReductionAxis::Column>()) * 2; //multiply diagonal elements;
    }

    template<typename _RHS, typename _Target>
    void _solve_impl(const MatrixBase<_RHS>& rhs, MatrixBase<_Target>& target) const
    {
        //for now, enforce column major storage of m
        CUMAT_STATIC_ASSERT(CUMAT_IS_COLUMN_MAJOR(internal::traits<_Target>::Flags),
            "Cholesky-Decomposition-Solve can only be evaluated into a Column-Major matrix");

        //broadcasting over the batches is allowed
        int batches = rhs.batches();
        Index strideA = Batches == 1 ? 1 : rows()*rows();

        //1. copy the rhs into m (with optional transposition)
        internal::Assignment<_Target, const _RHS, AssignmentMode::ASSIGN, typename _Target::DstTag, typename _RHS::SrcTag>::assign(target.derived(), rhs.derived());

        //2. assemble arguments to POTRS
        int n = rhs.rows();
        int nrhs = rhs.cols();
        const Scalar* A = decompositedMatrix_.data();
        int lda = n;
        Scalar* B = target.derived().data();
        int ldb = n;
        Index strideB = n * nrhs;

        //3. perform solving
        Matrix<int, 1, 1, Batches, RowMajor> devInfo(1, 1, batches);
        for (Index batch = 0; batch < batches; ++batch) {
            internal::CusolverApi::current().cusolverPotrs(
                CUBLAS_FILL_MODE_UPPER,
                n, nrhs,
                internal::CusolverApi::cast(A + batch*strideA), lda,
                internal::CusolverApi::cast(B + batch*strideB), ldb,
                devInfo.data() + batch);
        }

        //4. check if it was successfull
        std::vector<int> hostInfo(batches);
        devInfo.copyToHost(&hostInfo[0]);
        for (Index i = 0; i<batches; ++i)
        {
            if (hostInfo[i]<0)
            {
                throw cuda_error(internal::ErrorHelpers::format("Potrs failed at batch %d, parameter %d was invalid", i, -hostInfo[i]));
            }
        }
    }
};

CUMAT_NAMESPACE_END

#endif
