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

    /**
     * \brief Specifies the modifications on the two input arguments and the output argument.
     */
    enum class ProductArgOp
    {
        NONE = 0b00,
        TRANSPOSED = 0b01,
        CONJUGATED = 0b10,
        ADJOINT = 0b11
    };

    template<typename _Left, typename _Right, ProductArgOp _OpLeft, ProductArgOp _OpRight, ProductArgOp _OpOutput>
    struct traits<ProductOp<_Left, _Right, _OpLeft, _OpRight, _OpOutput> >
    {
        using Scalar = typename internal::traits<_Left>::Scalar;
        enum
        {
            TransposedLeft = int(_OpLeft)&int(ProductArgOp::TRANSPOSED) ? true : false,
            TransposedRight = int(_OpRight)&int(ProductArgOp::TRANSPOSED) ? true : false,
            TransposedOutput = int(_OpOutput)&int(ProductArgOp::TRANSPOSED) ? true : false,

            FlagsLeft = internal::traits<_Left>::Flags,
            RowsLeft = internal::traits<_Left>::RowsAtCompileTime,
            ColumnsLeft = internal::traits<_Left>::ColsAtCompileTime,
            BatchesLeft = internal::traits<_Left>::BatchesAtCompileTime,

            FlagsRight = internal::traits<_Right>::Flags,
            RowsRight = internal::traits<_Right>::RowsAtCompileTime,
            ColumnsRight = internal::traits<_Right>::ColsAtCompileTime,
            BatchesRight = internal::traits<_Right>::BatchesAtCompileTime,

            RowsNonT = TransposedLeft ? ColumnsLeft : RowsLeft,
            ColumnsNonT = TransposedRight ? RowsRight : ColumnsRight,

            Flags = ColumnMajor, //TODO: pick best flag
            RowsAtCompileTime = TransposedOutput ? ColumnsNonT : RowsNonT,
            ColsAtCompileTime = TransposedOutput ? RowsNonT : ColumnsNonT,
            BatchesAtCompileTime = BatchesLeft, //TODO: add broadcasting of batches

            AccessFlags = 0 //must be fully evaluated
        };
        typedef ProductSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
    };
} //end namespace internal

/**
 * \brief Operation for matrix-matrix multiplication.
 * It calls cuBLAS internally, therefore, it is only available for floating-point types.
 * 
 * TODO: also catch if the child expressions are conjugated/adjoint, not just transposed
 * 
 * \tparam _Left the left matrix
 * \tparam _Right the right matrix
 * \tparam _OpLeft operation (transposed, adjoint) applied to the left operand
 * \tparam _OpRight operation (transposed, adjoint) applied to the right operand
 * \tparam _OpOutput operation (transposed, adjoint) applied to the output
 */
template<typename _Left, typename _Right, internal::ProductArgOp _OpLeft, internal::ProductArgOp _OpRight, internal::ProductArgOp _OpOutput>
class ProductOp : public MatrixBase<ProductOp<_Left, _Right, _OpLeft, _OpRight, _OpOutput> >
{
public:
    using Type = ProductOp<_Left, _Right, _OpLeft, _OpRight, _OpOutput>;
    using Base = MatrixBase<Type>;
    CUMAT_PUBLIC_API

    using LeftType = typename _Left::Type;
    using RightType = typename _Right::Type;
    enum
    {
        LeftOp = _OpLeft,
        RightOp = _OpRight,
        OutputOp = _OpOutput,

        TransposedLeft = int(_OpLeft)&int(internal::ProductArgOp::TRANSPOSED) ? true : false,
        TransposedRight = int(_OpRight)&int(internal::ProductArgOp::TRANSPOSED) ? true : false,
        TransposedOutput = int(_OpOutput)&int(internal::ProductArgOp::TRANSPOSED) ? true : false,

        FlagsLeft = internal::traits<_Left>::Flags,
        RowsLeft = internal::traits<_Left>::RowsAtCompileTime,
        ColumnsLeft = internal::traits<_Left>::ColsAtCompileTime,
        BatchesLeft = internal::traits<_Left>::BatchesAtCompileTime,

        FlagsRight = internal::traits<_Right>::Flags,
        RowsRight = internal::traits<_Right>::RowsAtCompileTime,
        ColumnsRight = internal::traits<_Right>::ColsAtCompileTime,
        BatchesRight = internal::traits<_Right>::BatchesAtCompileTime,

        RowsNonT = TransposedLeft ? ColumnsLeft : RowsLeft,
        ColumnsNonT = TransposedRight ? RowsRight : ColumnsRight,
    };
    using Base::size;


private:
    LeftType left_;
    RightType right_;

public:
    ProductOp(const MatrixBase<_Left>& left, const MatrixBase<_Right>& right)
        : left_(left.derived()), right_(right.derived())
    {
        CUMAT_STATIC_ASSERT((std::is_same<typename internal::traits<_Left>::Scalar, typename internal::traits<_Right>::Scalar>::value),
            "No implicit casting is allowed in binary operations.");

        if (ColumnsLeft == Dynamic || RowsRight == Dynamic)
        {
            CUMAT_ASSERT_ARGUMENT((TransposedLeft ? left_.rows() : left_.cols()) == (TransposedRight ? right_.cols() : right_.rows()));
        }
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES((ColumnsLeft >= 1 && RowsRight >= 1), 
            (TransposedLeft ? RowsLeft : ColumnsLeft) == (TransposedRight ? ColumnsRight : RowsRight)),
            "matrix sizes not compatible");

        if (BatchesLeft == Dynamic || BatchesRight == Dynamic) {
            CUMAT_ASSERT_ARGUMENT(left.batches() == right.batches());
        }
        CUMAT_STATIC_ASSERT(!(BatchesLeft > 1 && BatchesRight > 1) || (BatchesLeft == BatchesRight), "batch count doesn't match");
    }

    const LeftType& left() const { return left_; }
    const RightType& right() const { return right_; }

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const
    {
        if (TransposedOutput)
            return TransposedRight ? right_.rows() : right_.cols();
        else
            return TransposedLeft ? left_.cols() : left_.rows();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const
    {
        if (TransposedOutput)
            return TransposedLeft ? left_.cols() : left_.rows();
        else
            return TransposedRight ? right_.rows() : right_.cols();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const
    {
        return left_.batches();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index innerSize() const
    {
        return TransposedLeft ? left_.rows() : left_.cols();
        //equivalent:
        //return _TransposedRight ? right_.cols() : right_.rows();
    }

public:
    /*
    template<typename Derived, AssignmentMode Mode>
    void evalTo(MatrixBase<Derived>& m) const
    {
        //TODO: Handle different assignment modes
        static_assert((Mode == AssignmentMode::ASSIGN || Mode == AssignmentMode::ADD),
            "Matrix multiplication only supports the following assignment modes: =, +=");
        float beta = Mode == AssignmentMode::ASSIGN ? 0 : 1;
        evalImpl(m.derived(), beta);
    }
    */

    //Overwrites transpose()
    typedef ProductOp<_Left, _Right, _OpLeft, _OpRight, internal::ProductArgOp(int(_OpOutput)^int(internal::ProductArgOp::TRANSPOSED))> transposed_mult_t;
    transposed_mult_t transpose() const
    {
        //transposition just changes the _TransposedOutput-flag
        return transposed_mult_t(left_, right_);
    }
};

namespace internal
{
    template<
        typename _Dst, typename _DstTag, ProductArgOp _DstOp,
        typename _SrcLeft, typename _SrcLeftTag, ProductArgOp _SrcLeftOp,
        typename _SrcRight, typename _SrcRightTag, ProductArgOp _SrcRightOp,
        AssignmentMode _AssignmentMode
    >
    struct ProductAssignment;
    //{
    //    using Op = ProductOp<_SrcLeft, _SrcRight, _SrcLeftOp, _SrcRightOp, _DstOp>;
    //    static void assign(_Dst& dst, const Op& op) { CUMAT_STATIC_ASSERT(false, "Product not implemented for these arguments"); }
    //};

    template<typename _Dst, typename _Src, AssignmentMode _AssignmentMode, typename _DstTag>
    struct Assignment<_Dst, _Src, _AssignmentMode, _DstTag, ProductSrcTag>
    {
        static void assign(_Dst& dst, const _Src& src)
        {
            typedef typename _Src::Type SrcActual; //must be an instance of ProductOp, TODO: check that
            //launch ProductAssigment
            ProductAssignment<
                typename _Dst::Type, _DstTag, internal::ProductArgOp(SrcActual::OutputOp),
                typename SrcActual::LeftType, typename traits<typename SrcActual::LeftType>::SrcTag, internal::ProductArgOp(SrcActual::LeftOp),
                typename SrcActual::RightType, typename traits<typename SrcActual::RightType>::SrcTag, internal::ProductArgOp(SrcActual::RightOp),
                _AssignmentMode>
                ::assign(dst.derived(), src.derived());
        }
    };

    // NOW, HERE COME THE ACTUAL IMPLEMENTATIONS

    // CwiseSrcTag * CwiseSrcTag -> DenseDstTag
    //This handles all dense cwise+matrix inputs and dense matrix output
    //The sparse methods (SparseSrcTag, SparseDstTag) are handled seperately
    template<
        typename _Dst, ProductArgOp _DstOp,
        typename _SrcLeft, ProductArgOp _SrcLeftOp,
        typename _SrcRight, ProductArgOp _SrcRightOp,
        AssignmentMode _AssignmentMode
    >
    struct ProductAssignment<_Dst, DenseDstTag, _DstOp, _SrcLeft, CwiseSrcTag, _SrcLeftOp, _SrcRight, CwiseSrcTag, _SrcRightOp, _AssignmentMode>
    {
        using Op = ProductOp<_SrcLeft, _SrcRight, _SrcLeftOp, _SrcRightOp, _DstOp>;
        using Scalar = typename Op::Scalar;
        typedef typename MatrixReadWrapper<_SrcLeft, AccessFlags::ReadDirect>::type left_wrapped_t;
        typedef typename MatrixReadWrapper<_SrcRight, AccessFlags::ReadDirect>::type right_wrapped_t;

        //implementation for direct matrix output
        template<typename Dst = typename std::enable_if<(int(traits<_Dst>::AccessFlags) & int(AccessFlags::WriteDirect)) != 0, _Dst>::type>
        static void evalImpl(Dst& mat, float betaIn,
            const left_wrapped_t& left, const right_wrapped_t& right, const Op& op,
            std::integral_constant<bool, true> /*direct-write*/)
        {
            CUMAT_ASSERT_ARGUMENT(mat.rows() == op.rows());
            CUMAT_ASSERT_ARGUMENT(mat.cols() == op.cols());
            CUMAT_ASSERT_ARGUMENT(mat.batches() == op.batches());

            //call cuBLAS

            int m = op.rows();
            int n = op.cols();
            int k = op.innerSize();

            //to be filled
            const Scalar *A, *B;
            cublasOperation_t transA, transB;

            Scalar* C = mat.data(); //This is required by AccessFlags::WriteDirect
            if ((!Op::TransposedOutput && CUMAT_IS_COLUMN_MAJOR(traits<_Dst>::Flags)) 
                || (Op::TransposedOutput && CUMAT_IS_ROW_MAJOR(traits<_Dst>::Flags)))
            {
                //C = A*B
                A = left.data();
                B = right.data();
                transA = (Op::TransposedLeft == CUMAT_IS_COLUMN_MAJOR(Op::FlagsLeft)) ? CUBLAS_OP_T : CUBLAS_OP_N;
                transB = (Op::TransposedRight == CUMAT_IS_COLUMN_MAJOR(Op::FlagsRight)) ? CUBLAS_OP_T : CUBLAS_OP_N;
            }
            else
            {
                //C' = B'*A'
                A = right.data();
                B = left.data();
                transA = (Op::TransposedRight == CUMAT_IS_COLUMN_MAJOR(Op::FlagsRight)) ? CUBLAS_OP_N : CUBLAS_OP_T;
                transB = (Op::TransposedLeft == CUMAT_IS_COLUMN_MAJOR(Op::FlagsLeft)) ? CUBLAS_OP_N : CUBLAS_OP_T;
            }

            if (CUMAT_IS_ROW_MAJOR(traits<_Dst>::Flags))
            {
                //flip rows and cols
                n = op.rows();
                m = op.cols();
            }

            //compute strides
            int lda = transA == CUBLAS_OP_N ? m : k;
            int ldb = transB == CUBLAS_OP_N ? k : n;
            int ldc = m;

            //thrust::complex<double> has no alignment requirements,
            //while cublas cuComplexDouble requires 16B-alignment.
            //If this is not fullfilled, a segfault is thrown.
            //This hack enforces that.
#ifdef _MSC_VER
            __declspec(align(16)) Scalar alpha(1);
            __declspec(align(16)) Scalar beta(betaIn);
#else
            Scalar alpha __attribute__((aligned(16))) = 1;
            Scalar beta __attribute__((aligned(16))) = 0;
#endif

            if (Op::Batches > 1 || op.batches() > 1)
            {
                //batched evaluation
                long long int strideA = m * k;
                long long int strideB = k * n;
                long long int strideC = m * n;
                internal::CublasApi::current().cublasGemmBatched(
                    transA, transB, m, n, k,
                    internal::CublasApi::cast(&alpha), internal::CublasApi::cast(A), lda, strideA,
                    internal::CublasApi::cast(B), ldb, strideB,
                    internal::CublasApi::cast(&beta), internal::CublasApi::cast(C), ldc, strideC, op.batches());

            }
            else
            {
                //single non-batched evaluation
                internal::CublasApi::current().cublasGemm(
                    transA, transB, m, n, k,
                    internal::CublasApi::cast(&alpha), internal::CublasApi::cast(A), lda,
                    internal::CublasApi::cast(B), ldb,
                    internal::CublasApi::cast(&beta), internal::CublasApi::cast(C), ldc);
            }

            CUMAT_PROFILING_INC(EvalAny);
            CUMAT_PROFILING_INC(EvalMatmul);
        }

        template<typename Dst = typename std::enable_if<(int(traits<_Dst>::AccessFlags) & int(AccessFlags::WriteDirect)) == 0, _Dst>::type>
        static void evalImpl(Dst& mat, float betaIn,
            const left_wrapped_t& left, const right_wrapped_t& right, const Op& op,
            std::integral_constant<bool, false> /*non-direct-write*/)
        {
            //Dst is not ready for direct write,
            //we have to evaluate it first into a temporary matrix.
            //Without the inplace compound-assignment
            typedef Matrix<Scalar, Op::Rows, Op::Columns, Op::Batches, Op::Flags> DstTmp;
            DstTmp tmp(op.rows(), op.cols(), op.batches());
            ProductAssignment<
                DstTmp, DenseDstTag, _DstOp, 
                _SrcLeft, CwiseSrcTag, _SrcLeftOp,
                _SrcRight, CwiseSrcTag, _SrcRightOp,
                AssignmentMode::ASSIGN>
            ::evalImpl(tmp, betaIn, left, right, op, std::integral_constant<bool, true>());
            //and now copy tmp to the output dst (cwise)
            Assignment<_Dst, DstTmp, _AssignmentMode, DenseDstTag, CwiseSrcTag>::assign(mat, tmp);
        }

        static void assign(_Dst& dst, const Op& op)
        {
            //evaluate cwise-expressions into the actual matrix
            //(at least until we can directly read them).
            //This is needed for cuBLAS
            left_wrapped_t left(op.left());
            right_wrapped_t right(op.right());

            //TODO: Handle different assignment modes
            static_assert((_AssignmentMode == AssignmentMode::ASSIGN || _AssignmentMode == AssignmentMode::ADD),
                "Matrix multiplication only supports the following assignment modes: =, +=");
            float beta = _AssignmentMode == AssignmentMode::ASSIGN ? 0 : 1;

            using DirectWriteTag = std::integral_constant<bool, (int(traits<_Dst>::AccessFlags) & int(AccessFlags::WriteDirect)) != 0>;
            evalImpl(dst, beta, left, right, op, DirectWriteTag());
        }
    };
}

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
ProductOp<_Left, _Right, internal::ProductArgOp::TRANSPOSED, internal::ProductArgOp::TRANSPOSED, internal::ProductArgOp::NONE>
operator*(const TransposeOp<_Left, false>& left, const TransposeOp<_Right, false>& right)
{
    return ProductOp<_Left, _Right, internal::ProductArgOp::TRANSPOSED, internal::ProductArgOp::TRANSPOSED, internal::ProductArgOp::NONE>(left.getUnderlyingMatrix(), right.getUnderlyingMatrix());
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
ProductOp<_Left, _Right, internal::ProductArgOp::NONE, internal::ProductArgOp::TRANSPOSED, internal::ProductArgOp::NONE>
operator*(const MatrixBase<_Left>& left, const TransposeOp<_Right, false>& right)
{
    return ProductOp<_Left, _Right, internal::ProductArgOp::NONE, internal::ProductArgOp::TRANSPOSED, internal::ProductArgOp::NONE>(left, right.getUnderlyingMatrix());
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
ProductOp<_Left, _Right, internal::ProductArgOp::TRANSPOSED, internal::ProductArgOp::NONE, internal::ProductArgOp::NONE>
operator*(const TransposeOp<_Left, false>& left, const MatrixBase<_Right>& right)
{
    return ProductOp<_Left, _Right, internal::ProductArgOp::TRANSPOSED, internal::ProductArgOp::NONE, internal::ProductArgOp::NONE>(left.getUnderlyingMatrix(), right);
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
ProductOp<_Left, _Right, internal::ProductArgOp::NONE, internal::ProductArgOp::NONE, internal::ProductArgOp::NONE>
operator*(const MatrixBase<_Left>& left, const MatrixBase<_Right>& right)
{
    return ProductOp<_Left, _Right, internal::ProductArgOp::NONE, internal::ProductArgOp::NONE, internal::ProductArgOp::NONE>(left, right);
}


CUMAT_NAMESPACE_END

#endif
