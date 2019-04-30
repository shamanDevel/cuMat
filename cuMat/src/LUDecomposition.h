#ifndef __CUMAT_LU_DECOMPOSITION_H__
#define __CUMAT_LU_DECOMPOSITION_H__

#include "Macros.h"
#include "Matrix.h"
#include "CusolverApi.h"
#include "DecompositionBase.h"
#include <algorithm>
#include <vector>
#include <type_traits>

CUMAT_NAMESPACE_BEGIN

namespace
{
    class PermutationSignFunctor
	{
	public:
        typedef int ReturnType;
		__device__ CUMAT_STRONG_INLINE int operator()(const int& x, Index row, Index col, Index batch) const
		{
			return (row+1 != x) ? -1 : 1;
		}
    };
}

namespace internal
{
    template<typename _MatrixType>
    struct traits<LUDecomposition<_MatrixType>>
    {
        using Scalar = typename internal::traits<_MatrixType>::Scalar;
        using MatrixType = _MatrixType;
    };
}

template<typename _MatrixType>
class LUDecomposition : public DecompositionBase<LUDecomposition<_MatrixType>>
{
public:
    typedef LUDecomposition<_MatrixType> Type;
    using Scalar = typename internal::traits<Type>::Scalar;
    using Base = DecompositionBase<LUDecomposition<_MatrixType>>;

    enum
    {
        Flags = internal::traits<_MatrixType>::Flags,
        Rows = internal::traits<_MatrixType>::RowsAtCompileTime,
        Columns = internal::traits<_MatrixType>::ColsAtCompileTime,
        Batches = internal::traits<_MatrixType>::BatchesAtCompileTime,
        Transposed = CUMAT_IS_ROW_MAJOR(Flags),
        InputIsMatrix = std::is_same< _MatrixType, Matrix<Scalar, Rows, Columns, Batches, Flags> >::value
    };
    typedef Matrix<Scalar, Dynamic, Dynamic, Batches, Flags> EvaluatedMatrix;
    typedef Matrix<int, Dynamic, 1, Batches, Flags> PivotArray;
    using typename Base::DeterminantMatrix;
private:
    EvaluatedMatrix decompositedMatrix_;
    PivotArray pivots_;
    std::vector<int> singular_;

public:
	/**
	 * \brief Uninitialized LU decomposition,
	 * but with the specified matrix dimensions.
	 * Since LU decompositions also allow non-symmetric matrices, 
	 * you have to specify both row and column count.
	 * Call \ref compute() before use.
	 */
	LUDecomposition(Index rows, Index cols, Index batches = Batches)
		: decompositedMatrix_(rows, cols, batches)
		, pivots_(std::min(rows, cols), 1, batches)
		, singular_(batches)
	{}
    /**
     * \brief Performs an LU-decomposition of the specified matrix and stores the result for future use.
     * \param matrix the matrix
     * \param inplace true to enforce inplace operation. The matrix contents will be destroyed
     */
	explicit LUDecomposition(const MatrixBase<_MatrixType>& matrix, bool inplace = false)
		: LUDecomposition(matrix.rows(), matrix.cols(), matrix.batches())
	{
		compute(matrix, inplace);
	}
	/**
	 * \brief Performs an LU-decomposition of the specified matrix and stores the result for future use.
	 * \param matrix the matrix
	 * \param inplace true to enforce inplace operation. The matrix contents will be destroyed
	 */
	void compute(const MatrixBase<_MatrixType>& matrix, bool inplace = false)
	{
        //optionally, copy input
        //(copy is never needed if the input is not a matrix and is evaluated into the matrix during the initializer list)
		if (!inplace && InputIsMatrix) {
			decompositedMatrix_ = matrix; //shallow copy
			decompositedMatrix_ = decompositedMatrix_.deepClone();
		} 
		else
			decompositedMatrix_ = matrix;
        
        //perform LU factorization
        const int m = internal::narrow_cast<int>(Transposed ? decompositedMatrix_.cols() : decompositedMatrix_.rows());
        const int n = internal::narrow_cast<int>(Transposed ? decompositedMatrix_.rows() : decompositedMatrix_.cols());
        const int batches = internal::narrow_cast<int>(decompositedMatrix_.batches());
        const int lda = m;
        Matrix<int, 1, 1, Batches, RowMajor> devInfo(1, 1, batches);
        for (Index batch = 0; batch < batches; ++batch) {
            internal::CusolverApi::current().cusolverGetrf(
                m, n,
                internal::CusolverApi::cast(decompositedMatrix_.data() + batch*m*n), lda,
                pivots_.data() + batch*std::min(m,n),
                devInfo.data() + batch);
        }

        //check if the operation was successfull
        devInfo.copyToHost(&singular_[0]);
        for (Index i=0; i<batches; ++i)
        {
            if (singular_[i]<0)
            {
                throw cuda_error(internal::ErrorHelpers::format("Getrf failed at batch %d, parameter %d was invalid", i, -singular_[i]));
            }
        }
    }

    CUMAT_STRONG_INLINE Index rows() const { return decompositedMatrix_.rows(); }
    CUMAT_STRONG_INLINE Index cols() const { return decompositedMatrix_.cols(); }
    CUMAT_STRONG_INLINE Index batches() const { return decompositedMatrix_.batches(); }

    const EvaluatedMatrix& getMatrixLU() const
    {
        return decompositedMatrix_;
    }

    const PivotArray& getPivots() const
    {
        return pivots_;
    }

    /**
     * \brief Tests if the matrix at the specified batch was singular during the initial LU decomposition
     * \param batch the batch
     * \return true iff that matrix was singular
     */
    bool isSingular(Index batch=0) const
    {
        return singular_[batch] > 0;
    }

    /**
     * \brief Computes the determinant of this matrix
     * \return The determinant
     */
    DeterminantMatrix determinant() const
    {
        if (rows()==0 || cols()!=rows())
        {
            return Matrix<Scalar, 1, 1, Batches, Flags>::Constant(1, 1, batches(), Scalar(1));
        }
        return decompositedMatrix_.diagonal().template prod<Axis::Row | Axis::Column>() //multiply diagonal elements
            .cwiseMul(
                UnaryOp<PivotArray, PermutationSignFunctor>(pivots_, PermutationSignFunctor())
                .template prod<Axis::Row | Axis::Column>().template cast<Scalar>() //compute sign of the permutation
            );
    }

    /**
    * \brief Computes the log-determinant of this matrix.
    * This is only supported for hermitian positive definite matrices, because no sign is computed.
    * A negative determinant would return in an complex logarithm which requires to return
    * a complex result for real matrices. This is not desired.
    * \return The log-determinant
    */
    DeterminantMatrix logDeterminant() const
    {
        if (rows() == 0 || cols() != rows())
        {
            return Matrix<Scalar, 1, 1, Batches, Flags>::Constant(1, 1, batches(), Scalar(0));
        }
        return decompositedMatrix_.diagonal().cwiseLog().template sum<Axis::Row | Axis::Column>(); //multiply diagonal elements;
    }

    //Internal solve implementation
    template<typename _RHS, typename _Target>
    void _solve_impl(const MatrixBase<_RHS>& rhs, MatrixBase<_Target>& target) const
    {
        //for now, enforce column major storage of m
        CUMAT_STATIC_ASSERT(CUMAT_IS_COLUMN_MAJOR(internal::traits<_Target>::Flags),
            "LUDecomposition-Solve can only be evaluated into a Column-Major matrix");

        //check if any batch was singular
        for (Index i=0; i<batches(); ++i)
        {
            if (isSingular(i)) throw cuda_error(internal::ErrorHelpers::format("Batch %d is singular, can't solve the system", i));
        }

        //broadcasting over the batches is allowed
        int batches = rhs.batches();
        Index strideA = Batches == 1 ? 1 : rows()*rows();

        //1. copy the rhs into m (with optional transposition)
        internal::Assignment<_Target, const _RHS, AssignmentMode::ASSIGN, typename _Target::DstTag, typename _RHS::SrcTag>::assign(target.derived(), rhs.derived());

        //2. assemble arguments to GETRS
        cublasOperation_t trans = Transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
        int n = internal::narrow_cast<int>(rhs.rows());
        int nrhs = internal::narrow_cast<int>(rhs.cols());
        const Scalar* A = decompositedMatrix_.data();
        int lda = n;
        const int *devIpiv = pivots_.data();
        Scalar* B = target.derived().data();
        int ldb = n;
        Index strideB = n * nrhs;

        //3. perform solving
        Matrix<int, 1, 1, Batches, RowMajor> devInfo(1, 1, batches);
        for (Index batch = 0; batch < batches; ++batch) {
            internal::CusolverApi::current().cusolverGetrs(
                trans,
                n, nrhs,
                internal::CusolverApi::cast(A + batch*strideA), lda,
                devIpiv + batch*n,
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
                throw cuda_error(internal::ErrorHelpers::format("Getrs failed at batch %d, parameter %d was invalid", i, -hostInfo[i]));
            }
        }
    }
};


CUMAT_NAMESPACE_END

#endif
