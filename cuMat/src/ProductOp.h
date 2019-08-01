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
#include "Errors.h"

CUMAT_NAMESPACE_BEGIN

namespace internal {
    struct ProductSrcTag {};
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

    /**
     * \brief Functor for multiplying single elements in a cwise matrix multiplication.
     * This is used for the outer product and sparse matrix multiplication on custom types.
     * 
     * The default version is provided for matrices with the same scalar type as left and right argument.
     * Provide your own specialization for custom types.
     * 
     * Implementations of this interface must provide:
     * - a typedef \c Scalar with the type of the return value
     * - a function \code static __device__ Scalar mult(const _LeftScalar& left, const _RightScalar& right) \endcode
     * 
     * \tparam _LeftScalar the scalar type of the left matrix
     * \tparam _RightScalar the scalar type of the right matrix
     * \tparam _OpLeft the left operation
     * \tparam _OpRight the right opertion
     * \tparam _OpOutput the output operation
     */
    template<typename _LeftScalar, typename _RightScalar, ProductArgOp _OpLeft, ProductArgOp _OpRight, ProductArgOp _OpOutput>
    struct ProductElementFunctor
    {
        CUMAT_STATIC_ASSERT((std::is_same<_LeftScalar, _RightScalar>::value), 
            "The default version is only available if the left and right scalar match");

        using Scalar = _LeftScalar; //same as _RightScalar

        static CUMAT_STRONG_INLINE __device__ Scalar mult(const _LeftScalar& left, const _RightScalar& right)
        {
            return left * right;
        }
    };

    template<typename _Left, typename _Right, ProductArgOp _OpLeft, ProductArgOp _OpRight, ProductArgOp _OpOutput>
    struct traits<ProductOp<_Left, _Right, _OpLeft, _OpRight, _OpOutput> >
    {
        using LeftScalar = typename internal::traits<_Left>::Scalar;
        using RightScalar = typename internal::traits<_Right>::Scalar;
        using Scalar = typename internal::ProductElementFunctor<LeftScalar, RightScalar, _OpLeft, _OpRight, _OpOutput>::Scalar;
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
            InnerSizeLeft = TransposedLeft ? RowsLeft : ColumnsLeft,
            ColumnsNonT = TransposedRight ? RowsRight : ColumnsRight,
            InnerSizeRight = TransposedRight ? ColumnsRight : RowsRight,

            IsOuterProduct = (InnerSizeLeft==1) && (InnerSizeRight==1),

            Flags = ColumnMajor, //TODO: pick best flag
            RowsAtCompileTime = TransposedOutput ? ColumnsNonT : RowsNonT,
            ColsAtCompileTime = TransposedOutput ? RowsNonT : ColumnsNonT,

            //broadcasting only supported for outer product
            BroadcastBatchesLeft = (internal::traits<_Left>::BatchesAtCompileTime == 1),
            BroadcastBatchesRight = (internal::traits<_Right>::BatchesAtCompileTime == 1),
            BatchesAtCompileTime = (BatchesLeft == Dynamic || BatchesRight == Dynamic) ?
                Dynamic : (BroadcastBatchesRight ? BatchesLeft : BatchesRight),

            AccessFlags = (IsOuterProduct ? ReadCwise : 0) //must be fully evaluated if not an outer product
        };
        typedef ProductSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
    };
} //end namespace internal

/**
 * \brief Operation for matrix-matrix multiplication.
 * 
 * - For dense matrix-matrix multiplications, it calls cuBLAS internally, 
 *   therefore, it is only available for floating-point types.
 * - For sparse matrix - dense vector multiplications, it either calls cuSparse
 *   for primitive types or a custom implementation for custom types
 * - For dense vector - dense vector outer product, the product op implements CwiseRead,
 *   and can therefore be used in a chain of cwise operations.
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

    enum
    {
        LeftOp = int(_OpLeft),
        RightOp = int(_OpRight),
        OutputOp = int(_OpOutput),

        TransposedLeft = int(_OpLeft)&int(internal::ProductArgOp::TRANSPOSED) ? true : false,
        TransposedRight = int(_OpRight)&int(internal::ProductArgOp::TRANSPOSED) ? true : false,
        TransposedOutput = int(_OpOutput)&int(internal::ProductArgOp::TRANSPOSED) ? true : false,
        ConjugateLeft = int(_OpLeft)&int(internal::ProductArgOp::CONJUGATED) ? true : false,
        ConjugateRight = int(_OpRight)&int(internal::ProductArgOp::CONJUGATED) ? true : false,
        ConjugateOutput = int(_OpOutput)&int(internal::ProductArgOp::CONJUGATED) ? true : false,

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

        IsOuterProduct = internal::traits<Type>::IsOuterProduct
    };
    using Base::size;

    //if the ProductOp represents an outer product, the left and right arguments
    //must support ReadCwise, so that coeff() is valid.
    typedef typename MatrixReadWrapper<_Left, AccessFlags::ReadCwise>::type left_wrapped_t;
    typedef typename MatrixReadWrapper<_Right, AccessFlags::ReadCwise>::type right_wrapped_t;
    using LeftType = typename std::conditional<IsOuterProduct, left_wrapped_t, typename _Left::Type>::type;
    using RightType = typename std::conditional<IsOuterProduct, right_wrapped_t, typename _Right::Type>::type;

private:
    LeftType left_;
    RightType right_;

public:
    ProductOp(const MatrixBase<_Left>& left, const MatrixBase<_Right>& right)
        : left_(left.derived()), right_(right.derived())
    {
        CUMAT_STATIC_ASSERT((std::is_same<
				typename internal::NumTraits<typename internal::traits<_Left>::Scalar>::ElementalType, 
				typename internal::NumTraits<typename internal::traits<_Right>::Scalar>::ElementalType>::value),
            "No implicit casting is allowed in binary operations.");

        if (ColumnsLeft == Dynamic || RowsRight == Dynamic)
        {
            CUMAT_ASSERT_ARGUMENT((TransposedLeft ? left_.rows() : left_.cols()) == (TransposedRight ? right_.cols() : right_.rows()));
        }
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES((ColumnsLeft >= 1 && RowsRight >= 1), 
            (TransposedLeft ? RowsLeft : ColumnsLeft) == (TransposedRight ? ColumnsRight : RowsRight)),
            "matrix sizes not compatible");

        if (BatchesLeft == Dynamic && BatchesRight == Dynamic) {
            CUMAT_ASSERT_ARGUMENT(left.batches() == right.batches());
        }
        CUMAT_STATIC_ASSERT(!(BatchesLeft > 1 && BatchesRight > 1) || (BatchesLeft == BatchesRight), "batch sizes don't match");
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
        if (BatchesLeft == 1) //broadcast left
            return right_.batches();
        else //maybe broadcast right
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
    typedef ProductOp<_Left, _Right, _OpLeft, _OpRight, internal::ProductArgOp((int)(_OpOutput)^(int)(internal::ProductArgOp::TRANSPOSED))> transposed_mult_t;
    transposed_mult_t transpose() const
    {
        //transposition just changes the _TransposedOutput-flag
        return transposed_mult_t(left_, right_);
    }

    //Cwise-evaluation if this product op is an outer product
    template<typename Dummy = Type, typename = std::enable_if<internal::traits<Dummy>::IsOuterProduct> >
    CUMAT_STRONG_INLINE __device__ Scalar coeff(Index row, Index col, Index batch, Index index) const
    {
        //all the following index transformations should be optimized out (compile-time if)
        Index r = TransposedOutput ? col : row;
        Index c = TransposedOutput ? row : col;
        //access left
        typedef typename internal::traits<Type>::LeftScalar LeftScalar;
        LeftScalar left = left_.coeff(
            TransposedLeft ? 0 : r, TransposedLeft ? r : 0, 
            internal::traits<Type>::BroadcastBatchesLeft ? 0 : batch,
            -1);
        left = ConjugateLeft ? internal::NumOps<LeftScalar>::conj(left) : left;
        //access right
        typedef typename internal::traits<Type>::RightScalar RightScalar;
        RightScalar right = right_.coeff(
            TransposedRight ? c : 0, TransposedRight ? 0 : c, 
            internal::traits<Type>::BroadcastBatchesRight ? 0 : batch,
            -1);
        right = ConjugateRight ? internal::NumOps<RightScalar>::conj(right) : right;
        //output
        Scalar output = internal::ProductElementFunctor<LeftScalar, RightScalar, _OpLeft, _OpRight, _OpOutput>::mult(left, right);
        output = ConjugateOutput ? internal::NumOps<Scalar>::conj(output) : output;
        return output;
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
        static void evalImpl(_Dst& mat, float betaIn,
            const left_wrapped_t& left, const right_wrapped_t& right, const Op& op,
            std::integral_constant<bool, true> /*direct-write*/)
        {
            CUMAT_ASSERT_ARGUMENT(mat.rows() == op.rows());
            CUMAT_ASSERT_ARGUMENT(mat.cols() == op.cols());
            CUMAT_ASSERT_ARGUMENT(mat.batches() == op.batches());

            //call cuBLAS

            int m = internal::narrow_cast<int>(op.rows());
            int n = internal::narrow_cast<int>(op.cols());
            int k = internal::narrow_cast<int>(op.innerSize());
			if (m==0 || n==0 || k==0)
			{
				CUMAT_THROW_INVALID_ARGUMENT("Attempt to multiply empty matrices");
			}

            //to be filled
            const Scalar *A, *B;
            cublasOperation_t transA, transB;
            bool broadcastA, broadcastB;

            Scalar* C = mat.data(); //This is required by AccessFlags::WriteDirect
            if ((!Op::TransposedOutput && CUMAT_IS_COLUMN_MAJOR(traits<_Dst>::Flags)) 
                || (Op::TransposedOutput && CUMAT_IS_ROW_MAJOR(traits<_Dst>::Flags)))
            {
                //C = A*B
                A = left.data();
                B = right.data();
                transA = (Op::TransposedLeft == CUMAT_IS_COLUMN_MAJOR(Op::FlagsLeft)) ? CUBLAS_OP_T : CUBLAS_OP_N;
                transB = (Op::TransposedRight == CUMAT_IS_COLUMN_MAJOR(Op::FlagsRight)) ? CUBLAS_OP_T : CUBLAS_OP_N;
                broadcastA = Op::BatchesLeft == 1;
                broadcastB = Op::BatchesRight == 1;
            }
            else
            {
                //C' = B'*A'
                A = right.data();
                B = left.data();
                transA = (Op::TransposedRight == CUMAT_IS_COLUMN_MAJOR(Op::FlagsRight)) ? CUBLAS_OP_N : CUBLAS_OP_T;
                transB = (Op::TransposedLeft == CUMAT_IS_COLUMN_MAJOR(Op::FlagsLeft)) ? CUBLAS_OP_N : CUBLAS_OP_T;
                broadcastA = Op::BatchesRight == 1;
                broadcastB = Op::BatchesLeft == 1;
            }

            if (CUMAT_IS_ROW_MAJOR(traits<_Dst>::Flags))
            {
                //flip rows and cols
                n = internal::narrow_cast<int>(op.rows());
                m = internal::narrow_cast<int>(op.cols());
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
                long long int strideA = broadcastA ? 0 : m * k;
                long long int strideB = broadcastB ? 0 : k * n;
                long long int strideC = m * n;
                internal::CublasApi::current().cublasGemmBatched(
                    transA, transB, m, n, k,
                    internal::CublasApi::cast(&alpha), internal::CublasApi::cast(A), lda, strideA,
                    internal::CublasApi::cast(B), ldb, strideB,
                    internal::CublasApi::cast(&beta), internal::CublasApi::cast(C), ldc, strideC,
                    internal::narrow_cast<int>(op.batches()));
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

        static void evalImpl(_Dst& mat, float betaIn,
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
