#ifndef __CUMAT_TRANSPOSE_OPS_H__
#define __CUMAT_TRANSPOSE_OPS_H__

#include <type_traits>

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "CwiseOp.h"
#include "Logging.h"
#include "Profiling.h"
#include "NumTraits.h"
#include "CublasApi.h"
#include "cuMat/Dense"

CUMAT_NAMESPACE_BEGIN

namespace internal {
    struct TransposeSrcTag {};
	template<typename _Derived, bool _Conjugated>
	struct traits<TransposeOp<_Derived, _Conjugated> >
	{
		using Scalar = typename internal::traits<_Derived>::Scalar;
		enum
		{
			Flags = (internal::traits<_Derived>::Flags == RowMajor) ? ColumnMajor : RowMajor,
			RowsAtCompileTime = internal::traits<_Derived>::ColsAtCompileTime,
			ColsAtCompileTime = internal::traits<_Derived>::RowsAtCompileTime,
			BatchesAtCompileTime = internal::traits<_Derived>::BatchesAtCompileTime,
            AccessFlags = ReadCwise | WriteCwise
		};
        typedef TransposeSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
	};

    /**
     * \brief Functor that transposes a single entry.
     * Specialize for custom scalar types
     */
    template<typename _Scalar> 
    struct TransposeFunctor
    {
        /**
         * \Brief Transposes the scalar type.
         * The default implementation does nothing.
         */
        __device__ _Scalar operator()(const _Scalar& val) const { return val; }
    };

	//helper functions to conjugate the argument if supported
	template<typename Scalar, bool _IsConjugated> __device__ CUMAT_STRONG_INLINE Scalar conjugateCoeff(const Scalar& val) { return val; };
	template<> __device__ CUMAT_STRONG_INLINE cfloat conjugateCoeff<cfloat, true>(const cfloat& val) { return conj(val); }
	template<> __device__ CUMAT_STRONG_INLINE cdouble conjugateCoeff<cdouble, true>(const cdouble& val) { return conj(val); }
} //end namespace internal

namespace internal
{
    /**
     * \brief Performs a direct copy-transpose operation from the src matrix (column major) into the dst matrix (column major)
     * \param dst destination matrix, column major
     * \param src source matrix, column major
     * \param rows number of rows in the source matrix
     * \param cols number of columns in the source matrix
     * \param batches number of batches in the source matrix
     */
    template<typename Scalar>
    void directTranspose(Scalar* dst, const Scalar* src, Index rows, Index cols, Index batches, cublasOperation_t transOp = CUBLAS_OP_T)
    {
        //thrust::complex<double> has no alignment requirements,
        //while cublas cuComplexDouble requires 16B-alignment.
        //If this is not fullfilled, a segfault is thrown.
        //This hack enforces that.
#ifdef _MSC_VER
        __declspec(align(16)) Scalar alpha(1);
        __declspec(align(16)) Scalar beta(0);
#else
        Scalar alpha __attribute__((aligned(16))) = 1;
        Scalar beta __attribute__((aligned(16))) = 0;
#endif

        int m = static_cast<int>(rows);
        int n = static_cast<int>(cols);
        
        cublasOperation_t transB = CUBLAS_OP_N;

        const Scalar* A = src;
        int lda = n;
        const Scalar* B = nullptr;
        int ldb = m;
        Scalar* C = dst;
        int ldc = m;
        size_t batch_offset = size_t(m) * n;
        //TODO: parallelize over multiple streams
        for (Index batch = 0; batch < batches; ++batch) {
            internal::CublasApi::current().cublasGeam(
                transOp, transB, m, n,
                internal::CublasApi::cast(&alpha), internal::CublasApi::cast(A + batch*batch_offset), lda, 
                internal::CublasApi::cast(&beta), internal::CublasApi::cast(B), ldb,
                internal::CublasApi::cast(C + batch*batch_offset), ldc);
        }

        CUMAT_PROFILING_INC(EvalTranspose);
        CUMAT_PROFILING_INC(EvalAny);
    }
}

/**
 * \brief Transposes the matrix.
 * This expression can be used on the right hand side and the left hand side.
 * \tparam _Derived the matrix type
 */
template<typename _Derived, bool _Conjugated>
class TransposeOp : public CwiseOp<TransposeOp<_Derived, _Conjugated>>
{
public:
    typedef CwiseOp<TransposeOp<_Derived, _Conjugated>> Base;
    using Type = TransposeOp<_Derived, _Conjugated>;
    CUMAT_PUBLIC_API

    enum
    {
        OriginalFlags = internal::traits<_Derived>::Flags,
        IsMatrix = std::is_same< _Derived, Matrix<Scalar, Columns, Rows, Batches, OriginalFlags> >::value,
        IsConjugated = _Conjugated && internal::NumTraits<typename internal::traits<_Derived>::Scalar>::IsComplex
    };

    using Base::size;

protected:
    const _Derived matrix_;

public:
    explicit TransposeOp(const MatrixBase<_Derived>& child)
        : matrix_(child.derived())
    {}

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return matrix_.cols(); }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return matrix_.rows(); }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return matrix_.batches(); }

    __device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch, Index index) const
    { //read acces (cwise)
        Scalar val = matrix_.coeff(col, row, batch, -1);
        val = internal::TransposeFunctor<Scalar>()(val);
        val = internal::conjugateCoeff<Scalar, IsConjugated>(val);
        return val;
    }
    __device__ CUMAT_STRONG_INLINE Scalar& coeff(Index row, Index col, Index batch, Index index)
    { //write acces (cwise)
        //adjoint not allowed here
        return matrix_.coeff(col, row, batch, -1);
    }

    const _Derived& getUnderlyingMatrix() const
    {
        return matrix_;
    }

	//ASSIGNMENT
	template<typename Derived>
	CUMAT_STRONG_INLINE Type& operator=(const MatrixBase<Derived>& expr)
	{
		CUMAT_ASSERT_ARGUMENT(rows() == expr.rows());
		CUMAT_ASSERT_ARGUMENT(cols() == expr.cols());
		CUMAT_ASSERT_ARGUMENT(batches() == expr.batches());
        internal::Assignment<Type, Derived, AssignmentMode::ASSIGN, internal::DenseDstTag, typename internal::traits<Derived>::SrcTag>
            ::assign(*this, expr);
		//expr.template evalTo<Type, AssignmentMode::ASSIGN>(*this);
		return *this;
	}

    //Overwrites transpose() to catch double transpositions
    const _Derived& transpose() const
    {
        //return the original matrix
        return matrix_;
    }

    TransposeOp<_Derived, !_Conjugated> conjugate() const
    {
        return TransposeOp<_Derived, !_Conjugated>(matrix_);
    }
};

namespace internal
{
    template<typename _Dst, typename _Src, AssignmentMode _AssignmentMode, typename _DstTag>
    struct Assignment<_Dst, _Src, _AssignmentMode, _DstTag, TransposeSrcTag>
    {
        using Scalar = typename _Src::Scalar;
        using Op = typename _Src::Type;

        //Everything else: Cwise-evaluation
        template<typename Derived, bool _Conj>
        static CUMAT_STRONG_INLINE void evalToImpl(const Op& src, MatrixBase<Derived>& m, std::false_type, std::integral_constant<bool, _Conj>)
        {
			CUMAT_ERROR_IF_NO_NVCC(cwise_transpose)
            //don't pass _Conj further, it is equal to IsConjugated
            Assignment<Derived, Op, AssignmentMode::ASSIGN, typename Derived::DstTag, CwiseSrcTag>::assign(m.derived(), src);
        }

        // No-Op version, just reinterprets the result
        template<int _Rows, int _Columns, int _Batches>
        static CUMAT_STRONG_INLINE void evalToImpl(const Op& src, Matrix<Scalar, _Rows, _Columns, _Batches, Op::Flags>& m, std::true_type, std::false_type)
        {
            m = Matrix<Scalar, _Rows, _Columns, _Batches, Op::Flags>(src.getUnderlyingMatrix().dataPointer(), src.rows(), src.cols(), src.batches());
        }

        // No-Op version, reinterprets the result + conjugates it
        template<int _Rows, int _Columns, int _Batches>
        static CUMAT_STRONG_INLINE void evalToImpl(const Op& src, Matrix<Scalar, _Rows, _Columns, _Batches, Op::Flags>& m, std::true_type, std::true_type)
        {
            m = Matrix<Scalar, _Rows, _Columns, _Batches, Op::Flags>(src.getUnderlyingMatrix().dataPointer(), src.rows(), src.cols(), src.batches()).conjugate();
        }

        // Explicit transposition
        template<int _Rows, int _Columns, int _Batches>
        static void evalToImplDirect(const Op& src, Matrix<Scalar, _Rows, _Columns, _Batches, Op::OriginalFlags>& mat, std::true_type)
        {
            //Call cuBlas for transposing

            CUMAT_ASSERT_ARGUMENT(mat.rows() == src.rows());
            CUMAT_ASSERT_ARGUMENT(mat.cols() == src.cols());
            CUMAT_ASSERT_ARGUMENT(mat.batches() == src.batches());

            CUMAT_LOG_DEBUG("Transpose: Direct transpose using cuBLAS");

            cublasOperation_t transA = Op::IsConjugated ? CUBLAS_OP_C : CUBLAS_OP_T;
            int m = static_cast<int>(Op::OriginalFlags == ColumnMajor ? mat.rows() : mat.cols());
            int n = static_cast<int>(Op::OriginalFlags == ColumnMajor ? mat.cols() : mat.rows());

            //perform transposition
            internal::directTranspose(mat.data(), src.getUnderlyingMatrix().data(), m, n, src.batches(), transA);
        }
        template<int _Rows, int _Columns, int _Batches>
        static CUMAT_STRONG_INLINE void evalToImplDirect(const Op& src, Matrix<Scalar, _Rows, _Columns, _Batches, Op::OriginalFlags>& mat, std::false_type)
        {
            //fallback for integer types
			CUMAT_ERROR_IF_NO_NVCC(transpose_integer)
            using Derived = Matrix<Scalar, _Rows, _Columns, _Batches, Op::OriginalFlags>;
            Assignment<Derived, Op, AssignmentMode::ASSIGN, typename Derived::DstTag, CwiseSrcTag>::assign(mat, src);
        }
        template<int _Rows, int _Columns, int _Batches, bool _Conj>
        static CUMAT_STRONG_INLINE void evalToImpl(const Op& src, Matrix<Scalar, _Rows, _Columns, _Batches, Op::OriginalFlags>& mat,
            std::true_type, std::integral_constant<bool, _Conj>)
        {
            //I don't need to pass _Conj further, because it is equal to IsConjugated
            evalToImplDirect(src, mat, std::integral_constant<bool, internal::NumTraits<Scalar>::IsCudaNumeric>());
        }

        static void assign(_Dst& dst, const _Src& src)
        {
            typedef typename _Src::Type SrcActual; //instance of TransposeOp
            //TODO: Handle different assignment modes
            static_assert(_AssignmentMode == AssignmentMode::ASSIGN, "Currently, only AssignmentMode::ASSIGN is supported");
            evalToImpl(src.derived(), dst.derived(), std::integral_constant<bool, SrcActual::IsMatrix>(), std::integral_constant<bool, SrcActual::IsConjugated>());
        }
    };
}

CUMAT_NAMESPACE_END

#endif
