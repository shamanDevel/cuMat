#ifndef __CUMAT_TRANSPOSE_OPS_H__
#define __CUMAT_TRANSPOSE_OPS_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "CwiseOp.h"
#include "Logging.h"
#include "Profiling.h"

CUMAT_NAMESPACE_BEGIN

namespace internal {
	template<typename _Derived>
	struct traits<TransposeOp<_Derived> >
	{
		using Scalar = typename internal::traits<_Derived>::Scalar;
		enum
		{
			Flags = (internal::traits<_Derived>::Flags == Flags::RowMajor) ? Flags::ColumnMajor : Flags::RowMajor,
			RowsAtCompileTime = internal::traits<_Derived>::ColsAtCompileTime,
			ColsAtCompileTime = internal::traits<_Derived>::RowsAtCompileTime,
			BatchesAtCompileTime = internal::traits<_Derived>::BatchesAtCompileTime
		};
	};

} //end namespace internal

/**
 * \brief Transposes the matrix.
 * This expression can be used on the right hand side and the left hand side.
 * \tparam _Derived the matrix type
 */
template<typename _Derived>
class TransposeOp : public CwiseOp<TransposeOp<_Derived>>
{
public:
	typedef MatrixBase<TransposeOp<_Derived>> Base;
	using Base::Scalar;
	using Type = TransposeOp<_Derived>;

    enum
    {
        OriginalFlags = internal::traits<_Derived>::Flags,
        Flags = (internal::traits<_Derived>::Flags == Flags::RowMajor) ? Flags::ColumnMajor : Flags::RowMajor,
        Rows = internal::traits<_Derived>::ColsAtCompileTime,
        Columns = internal::traits<_Derived>::RowsAtCompileTime,
        Batches = internal::traits<_Derived>::BatchesAtCompileTime,
        IsMatrix = std::is_same< _Derived, Matrix<Scalar, Columns, Rows, Batches, OriginalFlags> >::value
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

	__device__ CUMAT_STRONG_INLINE const Scalar& coeff(Index row, Index col, Index batch) const
	{ //read acces (cwise)
		return matrix_.coeff(col, row, batch);
	}
	__device__ CUMAT_STRONG_INLINE Scalar& coeff(Index row, Index col, Index batch)
	{ //write acces (cwise)
		return matrix_.coeff(col, row, batch);
	}

	//TODO: overwrite eval() and evalTo() to optimize direct matrix-matrix transposition
	
private:

    //Everything else: Cwise-evaluation
    template<typename Derived>
    void evalToImpl(MatrixBase<Derived>& m, std::false_type) const
    {
        CwiseOp<TransposeOp<_Derived>>::evalTo(m);
    }

    // No-Op version, just reinterprets the result
    template<int _Rows, int _Columns, int _Batches>
    void evalToImpl(Matrix<Scalar, _Rows, _Columns, _Batches, Flags>& m, std::true_type) const
    {
        m = Matrix<Scalar, _Rows, _Columns, _Batches, Flags>(matrix_.dataPointer(), rows(), cols(), batches());
    }

    // Explicit transposition
    template<int _Rows, int _Columns, int _Batches>
    void evalToImpl(Matrix<Scalar, _Rows, _Columns, _Batches, OriginalFlags>& m, std::true_type) const
    {
        //TODO: call cuBlas.transpose
        CUMAT_LOG(CUMAT_LOG_WARNING) << "Transpose: Direct transpose not supported yet";
    }

public:
    template<typename Derived>
    void evalTo(MatrixBase<Derived>& m) const
	{
        evalToImpl(m.derived(), std::bool_constant<IsMatrix == 1>());
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
};

CUMAT_NAMESPACE_END

#endif