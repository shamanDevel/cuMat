#ifndef __CUMAT_MULT_OP_H__
#define __CUMAT_MULT_OP_H__

#include <type_traits>
#include <cuda.h>

#include "Macros.h"
#include "Constants.h"
#include "ForwardDeclarations.h"
#include "MatrixBase.h"
#include "Logging.h"
#include "CublasApi.h"


CUMAT_NAMESPACE_BEGIN

namespace internal {
    template<typename _Left, typename _Right, bool _TransposedLeft, bool _TransposedRight, bool _TransposedOutput>
    struct traits<MultOp<_Left, _Right, _TransposedLeft, _TransposedRight, _TransposedOutput> >
    {
        using Scalar = typename internal::traits<_Left>::Scalar;
        enum
        {
            FlagsLeft = internal::traits<_Left>::Flags,
            RowsLeft = internal::traits<_Left>::RowsAtCompileTime,
            ColumnsLeft = internal::traits<_Left>::ColsAtCompileTime,
            BatchesLeft = internal::traits<_Left>::BatchesAtCompileTime,

            FlagsRight = internal::traits<_Right>::Flags,
            RowsRight = internal::traits<_Right>::RowsAtCompileTime,
            ColumnsRight = internal::traits<_Right>::ColsAtCompileTime,
            BatchesRight = internal::traits<_Right>::BatchesAtCompileTime,

            RowsNonT = _TransposedLeft ? ColumnsLeft : RowsLeft,
            ColumnsNonT = _TransposedRight ? RowsRight : ColumnsRight,

            Flags = ColumnMajor, //TODO: pick best flag
            RowsAtCompileTime = _TransposedOutput ? ColumnsNonT : RowsNonT,
            ColsAtCompileTime = _TransposedOutput ? RowsNonT : ColumnsNonT,
            BatchesAtCompileTime = BatchesLeft, //TODO: add broadcasting of batches

            AccessFlags = 0 //must be fully evaluated
        };
    };

} //end namespace internal

/**
 * \brief Operation for matrix-matrix multiplication.
 * It calls cuBLAS internally, therefore, it is only available for floating-point types.
 * \tparam _Left the left matrix
 * \tparam _Right the right matrix
 * \tparam _TransposedLeft true iff the left matrix is transposed
 * \tparam _TransposedRight true iff the right matrix is transposed
 * \tparam _TransposedOutput true iff the output shall be transposed
 */
template<typename _Left, typename _Right, bool _TransposedLeft, bool _TransposedRight, bool _TransposedOutput>
class MultOp : public MatrixBase<MultOp<_Left, _Right, _TransposedLeft, _TransposedRight, _TransposedOutput>>
{
public:
    typedef MatrixBase<MultOp<_Left, _Right, _TransposedLeft, _TransposedRight, _TransposedOutput>> Base;
    using Scalar = typename internal::traits<_Left>::Scalar;
    using Type = MultOp<_Left, _Right, _TransposedLeft, _TransposedRight, _TransposedOutput>;

    enum
    {
        FlagsLeft = internal::traits<_Left>::Flags,
        RowsLeft = internal::traits<_Left>::RowsAtCompileTime,
        ColumnsLeft = internal::traits<_Left>::ColsAtCompileTime,
        BatchesLeft = internal::traits<_Left>::BatchesAtCompileTime,

        FlagsRight = internal::traits<_Right>::Flags,
        RowsRight = internal::traits<_Right>::RowsAtCompileTime,
        ColumnsRight = internal::traits<_Right>::ColsAtCompileTime,
        BatchesRight = internal::traits<_Right>::BatchesAtCompileTime,

        RowsNonT = _TransposedLeft ? ColumnsLeft : RowsLeft,
        ColumnsNonT = _TransposedRight ? RowsRight : ColumnsRight,

        Flags = ColumnMajor, //TODO: pick best flag
        Rows = _TransposedOutput ? ColumnsNonT : RowsNonT,
        Columns = _TransposedOutput ? RowsNonT : ColumnsNonT,
        Batches = BatchesLeft, //TODO: add broadcasting of batches
    };

    using Base::size;
    using Base::derived;
    using Base::eval_t;

private:
    //wrapper that evaluate any cwise-expression to a matrix.
    //For BLAS we need the actual evaluated matrices.
    typedef typename MatrixReadWrapper<_Left, AccessFlags::ReadDirect>::type left_wrapped_t;
    typedef typename MatrixReadWrapper<_Right, AccessFlags::ReadDirect>::type right_wrapped_t;
    const left_wrapped_t left_;
    const right_wrapped_t right_;

public:
    MultOp(const MatrixBase<_Left>& left, const MatrixBase<_Right>& right)
        : left_(left.derived()), right_(right.derived())
    {
        CUMAT_STATIC_ASSERT((std::is_same<typename internal::traits<_Left>::Scalar, typename internal::traits<_Right>::Scalar>::value),
            "No implicit casting is allowed in binary operations.");

        if (ColumnsLeft == Dynamic || RowsRight == Dynamic)
        {
            CUMAT_ASSERT_ARGUMENT((_TransposedLeft ? left_.rows() : left_.cols()) == (_TransposedRight ? right_.cols() : right_.rows()));
        }
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES((ColumnsLeft >= 1 && RowsRight >= 1), 
            (_TransposedLeft ? RowsLeft : ColumnsLeft) == (_TransposedRight ? ColumnsRight : RowsRight)),
            "matrix sizes not compatible");

        if (BatchesLeft == Dynamic || BatchesRight == Dynamic) {
            CUMAT_ASSERT_ARGUMENT(left.batches() == right.batches());
        }
        CUMAT_STATIC_ASSERT(!(BatchesLeft > 1 && BatchesRight > 1) || (BatchesLeft == BatchesRight), "batch count doesn't match");
    }

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const
    {
        if (_TransposedOutput)
            return _TransposedRight ? right_.rows() : right_.cols();
        else
            return _TransposedLeft ? left_.cols() : left_.rows();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const
    {
        if (_TransposedOutput)
            return _TransposedLeft ? left_.cols() : left_.rows();
        else
            return _TransposedRight ? right_.rows() : right_.cols();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const
    {
        return left_.batches();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index innerSize() const
    {
        return _TransposedLeft ? left_.rows() : left_.cols();
        //equivalent:
        //return _TransposedRight ? right_.cols() : right_.rows();
    }

private:
    template<int _Rows, int _Columns, int _Batches, int _Flags>
    void evalImpl(Matrix<Scalar, _Rows, _Columns, _Batches, _Flags>& mat) const
    {
        CUMAT_ASSERT_ARGUMENT(mat.rows() == rows());
        CUMAT_ASSERT_ARGUMENT(mat.cols() == cols());
        CUMAT_ASSERT_ARGUMENT(mat.batches() == batches());

        //call cuBLAS
        
        int m = rows();
        int n = cols();
        int k = innerSize();

        //to be filled
        const Scalar *A, *B;
        cublasOperation_t transA, transB;

        Scalar* C = mat.data();
        if ((!_TransposedOutput && CUMAT_IS_COLUMN_MAJOR(_Flags)) || (_TransposedOutput && CUMAT_IS_ROW_MAJOR(_Flags)))
        {
            //C = A*B
            A = left_.data();
            B = right_.data();
            transA = (_TransposedLeft == CUMAT_IS_COLUMN_MAJOR(FlagsLeft)) ? CUBLAS_OP_T : CUBLAS_OP_N;
            transB = (_TransposedRight == CUMAT_IS_COLUMN_MAJOR(FlagsRight)) ? CUBLAS_OP_T : CUBLAS_OP_N;
        } else
        {
            //C' = B'*A'
            A = right_.data();
            B = left_.data();
            transA = (_TransposedRight == CUMAT_IS_COLUMN_MAJOR(FlagsRight)) ? CUBLAS_OP_N : CUBLAS_OP_T;
            transB = (_TransposedLeft == CUMAT_IS_COLUMN_MAJOR(FlagsLeft)) ? CUBLAS_OP_N : CUBLAS_OP_T;
        }

        if (CUMAT_IS_ROW_MAJOR(_Flags))
        {
            //flip rows and cols
            n = rows();
            m = cols();
        }

        //compute strides
        int lda = transA == CUBLAS_OP_N ? m : k;
        int ldb = transB == CUBLAS_OP_N ? k : n;
        int ldc = m;

        //alpha+beta will later be filled if more information of the parent nodes are available
        Scalar alpha = Scalar(1);
        Scalar beta = Scalar(0);

        if (_Batches > 1 || batches() > 1)
        {
            //batched evaluation
            long long int strideA = m * k;
            long long int strideB = k * n;
            long long int strideC = m * n;
            internal::CublasApi::current().cublasGemmBatched(transA, transB, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batches());

        } else
        {
            //single non-batched evaluation
            internal::CublasApi::current().cublasGemm(transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        }

        CUMAT_PROFILING_INC(EvalAny);
        CUMAT_PROFILING_INC(EvalMatmul);
    }

public:
    template<typename Derived>
    void evalTo(MatrixBase<Derived>& m) const
    {
        evalImpl(m.derived());
    }

    //Overwrites transpose()
    typedef MultOp<left_wrapped_t, right_wrapped_t, _TransposedLeft, _TransposedRight, !_TransposedOutput> transposed_mult_t;
    transposed_mult_t transpose() const
    {
        //transposition just changes the _TransposedOutput-flag
        return transposed_mult_t(left_, right_);
    }
};


//operator overloading, must handle all four cases of transposed inputs

/**
 * \brief Performs the matrix-matrix multiplication <tt>C = A' * B'<tt>.
 * This is a specialization if both input matrices are transposed. This avoids the explicit evaluation of \c .transpose() .
 * \tparam _Left the left type
 * \tparam _Right the right type
 * \param left the left matrix
 * \param right the right matrix
 * \return the result of the matrix-matrix multiplication
 */
template<typename _Left, typename _Right>
MultOp<_Left, _Right, true, true, false>
operator*(const TransposeOp<_Left>& left, const TransposeOp<_Right>& right)
{
    return MultOp<_Left, _Right, true, true, false>(left.getUnderlyingMatrix(), right.getUnderlyingMatrix());
}

/**
* \brief Performs the matrix-matrix multiplication <tt>C = A * B'<tt>.
* This is a specialization if the right input matri is transposed. This avoids the explicit evaluation of \c .transpose() .
* \tparam _Left the left type
* \tparam _Right the right type
* \param left the left matrix
* \param right the right matrix
* \return the result of the matrix-matrix multiplication
*/
template<typename _Left, typename _Right>
MultOp<_Left, _Right, false, true, false>
operator*(const MatrixBase<_Left>& left, const TransposeOp<_Right>& right)
{
    return MultOp<_Left, _Right, false, true, false>(left, right.getUnderlyingMatrix());
}

/**
* \brief Performs the matrix-matrix multiplication <tt>C = A' * B<tt>.
* This is a specialization if the left input matri is transposed. This avoids the explicit evaluation of \c .transpose() .
* \tparam _Left the left type
* \tparam _Right the right type
* \param left the left matrix
* \param right the right matrix
* \return the result of the matrix-matrix multiplication
*/
template<typename _Left, typename _Right>
MultOp<_Left, _Right, true, false, false>
operator*(const TransposeOp<_Left>& left, const MatrixBase<_Right>& right)
{
    return MultOp<_Left, _Right, true, false, false>(left.getUnderlyingMatrix(), right);
}

/**
* \brief Performs the matrix-matrix multiplication <tt>C = A * B<tt>.
* \tparam _Left the left type
* \tparam _Right the right type
* \param left the left matrix
* \param right the right matrix
* \return the result of the matrix-matrix multiplication
*/
template<typename _Left, typename _Right>
MultOp<_Left, _Right, false, false, false>
operator*(const MatrixBase<_Left>& left, const MatrixBase<_Right>& right)
{
    return MultOp<_Left, _Right, false, false, false>(left, right);
}


CUMAT_NAMESPACE_END

#endif
