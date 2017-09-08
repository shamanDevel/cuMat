#ifndef __CUMAT_NULLARY_OPS_H__
#define __CUMAT_NULLARY_OPS_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "CwiseOp.h"

CUMAT_NAMESPACE_BEGIN

namespace internal {
	template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags, typename _NullaryFunctor>
	struct traits<NullaryOp<_Scalar, _Rows, _Columns, _Batches, _Flags, _NullaryFunctor> >
	{
		typedef typename _Scalar Scalar;
		enum
		{
			Flags = _Flags,
			RowsAtCompileTime = _Rows,
			ColsAtCompileTime = _Columns,
			BatchesAtCompileTime = _Batches
		};
	};
}

/**
 * \brief A generic nullary operator.
 * This is used as a leaf in the expression template tree.
 * The nullary functor can be any class or structure that supports
 * the following method:
 * \code
 * __device__ const _Scalar& operator()(Index row, Index col, Index batch);
 * \endcode
 * \tparam _Scalar 
 * \tparam _NullaryFunctor 
 */
template<typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags, typename _NullaryFunctor>
class NullaryOp : public CwiseOp<NullaryOp<_Scalar, _Rows, _Columns, _Batches, _Flags, _NullaryFunctor> >
{
public:
	typedef CwiseOp<NullaryOp<_Scalar, _Rows, _Columns, _Batches, _Flags, _NullaryFunctor> > Base;
	using Scalar = _Scalar;
	enum
	{
		Flags = _Flags,
		Rows = _Rows,
		Columns = _Columns,
		Batches = _Batches
	};

protected:
	const Index rows_;
	const Index cols_;
	const Index batches_;
	const _NullaryFunctor functor_;

public:
	NullaryOp(Index rows, Index cols, Index batches, const _NullaryFunctor& functor)
		: rows_(rows), cols_(cols), batches_(batches), functor_(functor)
	{}

	__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return rows_; }
	__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return cols_; }
	__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return batches_; }

	__device__ CUMAT_STRONG_INLINE const Scalar& coeff(Index row, Index col, Index batch) const
	{
		return functor_(row, col, batch);
	}
};

/**
 * \brief A tag for nullary functors
 */
struct NullaryFunctorTag {};

namespace functor
{
	template<typename _Scalar>
	class ConstantFunctor : public NullaryFunctorTag
	{
	private:
		const _Scalar value_;
	public:
		ConstantFunctor(_Scalar value)
			: value_(value)
		{}

		__device__ CUMAT_STRONG_INLINE const _Scalar& operator()(Index row, Index col, Index batch) const
		{
			return value_;
		}
	};
}

CUMAT_NAMESPACE_END

#endif