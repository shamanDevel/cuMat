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

CUMAT_NAMESPACE_BEGIN

namespace internal {
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
	};

} //end namespace internal

//helper functions to conjugate the argument if supported
namespace
{
    template<typename Scalar, bool _IsConjugated> __device__ CUMAT_STRONG_INLINE Scalar conjugateCoeff(const Scalar& val) { return val; };
    template<> __device__ CUMAT_STRONG_INLINE cfloat conjugateCoeff<cfloat, true>(const cfloat& val) { return conj(val); }
    template<> __device__ CUMAT_STRONG_INLINE cdouble conjugateCoeff<cdouble, true>(const cdouble& val) { return conj(val); }
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
    typedef typename internal::traits<_Derived>::Scalar Scalar;
    using Type = TransposeOp<_Derived, _Conjugated>;

    enum
    {
        OriginalFlags = internal::traits<_Derived>::Flags,
        Flags = (internal::traits<_Derived>::Flags == RowMajor) ? ColumnMajor : RowMajor,
        Rows = internal::traits<_Derived>::ColsAtCompileTime,
        Columns = internal::traits<_Derived>::RowsAtCompileTime,
        Batches = internal::traits<_Derived>::BatchesAtCompileTime,
        IsMatrix = std::is_same< _Derived, Matrix<Scalar, Columns, Rows, Batches, OriginalFlags> >::value,
        IsConjugated = _Conjugated && internal::NumTraits<typename internal::traits<_Derived>::Scalar>::IsComplex
    };

    using Base::size;
    using Base::derived;
    using Base::eval_t;

protected:
    const _Derived matrix_;

public:
    TransposeOp(const MatrixBase<_Derived>& child)
        : matrix_(child.derived())
    {}

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return matrix_.cols(); }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return matrix_.rows(); }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return matrix_.batches(); }

    __device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch) const
    { //read acces (cwise)
        Scalar val = matrix_.coeff(col, row, batch);
        val = conjugateCoeff<Scalar, IsConjugated>(val);
        return val;
    }
    __device__ CUMAT_STRONG_INLINE Scalar& coeff(Index row, Index col, Index batch)
    { //write acces (cwise)
        //adjoint not allowed here
        return matrix_.coeff(col, row, batch);
    }

    const _Derived& getUnderlyingMatrix() const
    {
        return matrix_;
    }

private:

    //Everything else: Cwise-evaluation
    template<typename Derived, bool _Conj>
    CUMAT_STRONG_INLINE void evalToImpl(MatrixBase<Derived>& m, std::false_type, std::integral_constant<bool, _Conj>) const
    {
        //don't pass _Conj further, it is equal to IsConjugated
        CwiseOp<TransposeOp<_Derived, _Conjugated>>::evalTo(m);
    }

    // No-Op version, just reinterprets the result
    template<int _Rows, int _Columns, int _Batches>
    CUMAT_STRONG_INLINE void evalToImpl(Matrix<Scalar, _Rows, _Columns, _Batches, Flags>& m, std::true_type, std::false_type) const
    {
        m = Matrix<Scalar, _Rows, _Columns, _Batches, Flags>(matrix_.dataPointer(), rows(), cols(), batches());
    }

    // No-Op version, reinterprets the result + conjugates it
    template<int _Rows, int _Columns, int _Batches>
    CUMAT_STRONG_INLINE void evalToImpl(Matrix<Scalar, _Rows, _Columns, _Batches, Flags>& m, std::true_type, std::true_type) const
    {
        m = Matrix<Scalar, _Rows, _Columns, _Batches, Flags>(matrix_.dataPointer(), rows(), cols(), batches()).conjugate();
    }

    // Explicit transposition
    template<int _Rows, int _Columns, int _Batches>
    void evalToImplDirect(Matrix<Scalar, _Rows, _Columns, _Batches, OriginalFlags>& mat, std::true_type) const
    {
        //Call cuBlas for transposing

        CUMAT_ASSERT_ARGUMENT(mat.rows() == rows());
        CUMAT_ASSERT_ARGUMENT(mat.cols() == cols());
        CUMAT_ASSERT_ARGUMENT(mat.batches() == batches());
        
        CUMAT_LOG(CUMAT_LOG_WARNING) << "Transpose: Direct transpose using cuBLAS";

        cublasOperation_t transA = IsConjugated ? CUBLAS_OP_C : CUBLAS_OP_T;
        cublasOperation_t transB = CUBLAS_OP_N;
        int m = static_cast<int>(OriginalFlags == ColumnMajor ? mat.rows() : mat.cols());
        int n = static_cast<int>(OriginalFlags == ColumnMajor ? mat.cols() : mat.rows());

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

        const Scalar* A = matrix_.data();
        int lda = n;
        const Scalar* B = nullptr;
        int ldb = m;
        Scalar* C = mat.data();
        int ldc = m;
        size_t batch_offset = size_t(m) * n;
        //TODO: parallelize over multiple streams
        for (Index batch = 0; batch < batches(); ++batch) {
            internal::CublasApi::current().cublasGeam(
                transA, transB, m, n,
                internal::CublasApi::cast(&alpha), internal::CublasApi::cast(A + batch*batch_offset), lda, 
                internal::CublasApi::cast(&beta), internal::CublasApi::cast(B), ldb,
                internal::CublasApi::cast(C + batch*batch_offset), ldc);
        }

        CUMAT_PROFILING_INC(EvalTranspose);
        CUMAT_PROFILING_INC(EvalAny);
    }
    template<int _Rows, int _Columns, int _Batches>
    CUMAT_STRONG_INLINE void evalToImplDirect(Matrix<Scalar, _Rows, _Columns, _Batches, OriginalFlags>& mat, std::false_type) const
    {
        //fallback for integer types
        CwiseOp<TransposeOp<_Derived, _Conjugated>>::evalTo(mat);
    }
    template<int _Rows, int _Columns, int _Batches, bool _Conj>
    CUMAT_STRONG_INLINE void evalToImpl(Matrix<Scalar, _Rows, _Columns, _Batches, OriginalFlags>& mat,
        std::true_type, std::integral_constant<bool, _Conj>) const
    {
        //I don't need to pass _Conj further, because it is equal to IsConjugated
        evalToImplDirect(mat, std::integral_constant<bool, internal::NumTraits<Scalar>::IsCudaNumeric>());
    }

public:
    template<typename Derived>
    void evalTo(MatrixBase<Derived>& m) const
	{
        evalToImpl(m.derived(), std::integral_constant<bool, IsMatrix>(), std::integral_constant<bool, IsConjugated>());
	}

	//ASSIGNMENT
	template<typename Derived>
	CUMAT_STRONG_INLINE Type& operator=(const MatrixBase<Derived>& expr)
	{
		CUMAT_ASSERT_ARGUMENT(rows() == expr.rows());
		CUMAT_ASSERT_ARGUMENT(cols() == expr.cols());
		CUMAT_ASSERT_ARGUMENT(batches() == expr.batches());
		expr.evalTo(*this);
		return *this;
	}

    //Overwrites transpose() to catch double transpositions
    const _Derived& transpose() const
    {
        //return the original matrix
        return matrix_;
    }

    const TransposeOp<_Derived, !_Conjugated> conjugate() const
    {
        return TransposeOp<_Derived, !_Conjugated>(matrix_);
    }
};

CUMAT_NAMESPACE_END

#endif
